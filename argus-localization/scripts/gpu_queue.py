"""Run a command on an idle GPU, waiting (queueing) until one is free.

Built for the shared GPU workstation's rules (README "Running on the shared GPU
workstation"): if a GPU is running someone else's process, wait; never
interrupt, kill, or reset anything. This script only ever signals the one child
process it started itself.

    python scripts/gpu_queue.py -- python scripts/evaluate.py --retriever remoteclip --skip-matching
    python scripts/gpu_queue.py --status        # show what it sees, run nothing

A GPU counts as idle when it has no compute processes, uses at most
--max-used-mb of memory, and is at most --max-util percent utilized. The child
gets CUDA_VISIBLE_DEVICES set to that GPU's UUID and runs at lower CPU priority
(nice). While it runs, a GPU snapshot is appended to a CSV log every
--monitor-seconds, and a warning is printed if another process shows up on
"our" GPU. Nothing is done about it except logging.

A lock file (fcntl) serializes GPU selection across concurrently queued jobs of
this user, and is held for --claim-seconds after launch so the next queued job
sees this one's memory before choosing.
"""

import argparse
import csv
import fcntl
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime

GPU_QUERY = "index,uuid,name,memory.used,memory.total,utilization.gpu"
APP_QUERY = "gpu_uuid,pid,process_name,used_memory"


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_pct: int


@dataclass(frozen=True)
class ComputeApp:
    gpu_uuid: str
    pid: int
    process_name: str
    used_memory_mb: int


def _int(value: str) -> int:
    value = value.strip()
    return int(value) if value.isdigit() else 0  # "[N/A]" and friends count as 0


def parse_gpus(csv_text: str) -> list[Gpu]:
    gpus = []
    for row in csv.reader(line for line in csv_text.splitlines() if line.strip()):
        index, uuid, name, used, total, util = (field.strip() for field in row)
        gpus.append(Gpu(int(index), uuid, name, _int(used), _int(total), _int(util)))
    return gpus


def parse_apps(csv_text: str) -> list[ComputeApp]:
    apps = []
    for row in csv.reader(line for line in csv_text.splitlines() if line.strip()):
        gpu_uuid, pid, process_name, used = (field.strip() for field in row)
        apps.append(ComputeApp(gpu_uuid, _int(pid), process_name, _int(used)))
    return apps


def busy_reasons(
    gpu: Gpu, apps: list[ComputeApp], max_used_mb: int, max_util: int
) -> list[str]:
    """Why this GPU isn't idle; empty means it's free to use."""
    reasons = [
        f"pid {app.pid} ({app.process_name}, {app.used_memory_mb} MiB)"
        for app in apps
        if app.gpu_uuid == gpu.uuid
    ]
    if gpu.memory_used_mb > max_used_mb:
        reasons.append(f"{gpu.memory_used_mb} MiB used")
    if gpu.utilization_pct > max_util:
        reasons.append(f"{gpu.utilization_pct}% utilized")
    return reasons


def pick_idle_gpu(
    gpus: list[Gpu],
    apps: list[ComputeApp],
    allowed: set[int] | None,
    max_used_mb: int,
    max_util: int,
    require_all_idle: bool = False,
) -> Gpu | None:
    if require_all_idle and any(busy_reasons(gpu, apps, max_used_mb, max_util) for gpu in gpus):
        return None
    candidates = [gpu for gpu in gpus if allowed is None or gpu.index in allowed]
    idle = [gpu for gpu in candidates if not busy_reasons(gpu, apps, max_used_mb, max_util)]
    return idle[0] if idle else None


def query_nvidia_smi() -> tuple[list[Gpu], list[ComputeApp]]:
    def run(query_flag: str, fields: str) -> str:
        return subprocess.run(
            ["nvidia-smi", f"--query-{query_flag}={fields}", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=60,
        ).stdout

    return parse_gpus(run("gpu", GPU_QUERY)), parse_apps(run("compute-apps", APP_QUERY))


def describe(gpus: list[Gpu], apps: list[ComputeApp], max_used_mb: int, max_util: int) -> str:
    lines = []
    for gpu in gpus:
        reasons = busy_reasons(gpu, apps, max_used_mb, max_util)
        state = "busy: " + "; ".join(reasons) if reasons else "idle"
        lines.append(
            f"  GPU {gpu.index} {gpu.name}: {gpu.memory_used_mb}/{gpu.memory_total_mb} MiB, "
            f"{gpu.utilization_pct}% util -> {state}"
        )
    return "\n".join(lines)


def log(message: str) -> None:
    print(f"[gpu_queue {datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for an idle GPU, then run a command on it. Never touches other processes."
    )
    parser.add_argument("--gpus", default=None, help="comma-separated GPU indices allowed (default: any)")
    parser.add_argument("--max-used-mb", type=int, default=1024, help="idle = at most this much memory in use")
    parser.add_argument("--max-util", type=int, default=10, help="idle = at most this %% utilization")
    parser.add_argument(
        "--require-all-idle", action="store_true",
        help="stricter: only start when every GPU on the machine is idle",
    )
    parser.add_argument("--poll-seconds", type=int, default=120, help="re-check interval while queued")
    parser.add_argument("--monitor-seconds", type=int, default=60, help="GPU log interval while running")
    parser.add_argument("--claim-seconds", type=int, default=120, help="hold the selection lock after launch")
    parser.add_argument("--nice", type=int, default=10, help="CPU niceness for the child (0 = unchanged)")
    parser.add_argument("--log-dir", default="output/gpu_logs")
    parser.add_argument("--lock-file", default=os.path.expanduser("~/.cache/argus_gpu_queue.lock"))
    parser.add_argument("--status", action="store_true", help="print GPU state and exit")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command to run")
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    allowed = {int(i) for i in args.gpus.split(",")} if args.gpus else None

    gpus, apps = query_nvidia_smi()
    if args.status:
        print(describe(gpus, apps, args.max_used_mb, args.max_util))
        return 0
    if not command:
        parser.error("no command given (put it after --)")

    os.makedirs(os.path.dirname(args.lock_file), exist_ok=True)
    lock = open(args.lock_file, "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    gpus, apps = query_nvidia_smi()  # fresh, now that no other queued job of ours is choosing
    while True:
        gpu = pick_idle_gpu(gpus, apps, allowed, args.max_used_mb, args.max_util, args.require_all_idle)
        if gpu is not None:
            break
        log(f"no idle GPU, queued; checking again in {args.poll_seconds}s\n"
            + describe(gpus, apps, args.max_used_mb, args.max_util))
        fcntl.flock(lock, fcntl.LOCK_UN)  # let other queued jobs look while we sleep
        time.sleep(args.poll_seconds)
        fcntl.flock(lock, fcntl.LOCK_EX)
        gpus, apps = query_nvidia_smi()

    # By UUID: CUDA's default device order can differ from nvidia-smi's index order.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu.uuid)
    niceness = args.nice

    def lower_priority() -> None:
        if niceness:
            os.nice(niceness)

    log(f"starting on GPU {gpu.index} ({gpu.name}): {' '.join(command)}")
    # Own session, so a terminal Ctrl-C reaches the child once, through forward().
    child = subprocess.Popen(command, env=env, preexec_fn=lower_priority, start_new_session=True)

    def forward(signum, _frame) -> None:
        # Only ever our own child; the processes of other users are never touched.
        child.send_signal(signum)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"gpu{gpu.index}_{datetime.now():%Y%m%d-%H%M%S}_pid{child.pid}.csv")
    claim_deadline = time.time() + args.claim_seconds
    lock_held = True
    warned_pids: set[int] = set()
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["time", "gpu", "memory_used_mb", "memory_total_mb", "utilization_pct", "processes"])
        while child.poll() is None:
            if lock_held and time.time() >= claim_deadline:
                fcntl.flock(lock, fcntl.LOCK_UN)
                lock_held = False
            try:
                gpus, apps = query_nvidia_smi()
            except (subprocess.SubprocessError, OSError) as err:
                log(f"nvidia-smi failed ({err}); still waiting on the child")
            else:
                now = datetime.now().isoformat(timespec="seconds")
                for g in gpus:
                    procs = ";".join(
                        f"{a.pid}:{a.process_name}:{a.used_memory_mb}" for a in apps if a.gpu_uuid == g.uuid
                    )
                    writer.writerow([now, g.index, g.memory_used_mb, g.memory_total_mb, g.utilization_pct, procs])
                log_file.flush()
                for app in apps:
                    if app.gpu_uuid != gpu.uuid or app.pid in warned_pids or _is_ours(app.pid, child.pid):
                        continue
                    log(f"note: another process is now on GPU {gpu.index}: pid {app.pid} "
                        f"({app.process_name}); leaving it alone")
                    warned_pids.add(app.pid)
            try:
                child.wait(timeout=args.monitor_seconds)
            except subprocess.TimeoutExpired:
                pass
    if lock_held:
        fcntl.flock(lock, fcntl.LOCK_UN)
    log(f"child exited with code {child.returncode}; GPU log at {log_path}")
    return child.returncode


def _is_ours(pid: int, root_pid: int) -> bool:
    """True if pid is root_pid or one of its descendants (reads /proc, Linux only)."""
    while pid > 1:
        if pid == root_pid:
            return True
        try:
            with open(f"/proc/{pid}/stat") as f:
                # Field 4 is the parent pid; the process name (field 2) may contain spaces.
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


if __name__ == "__main__":
    sys.exit(main())
