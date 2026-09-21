# Shared GPU workstation rules

The team's shared lab GPU workstation (2 GPUs; hosts the EarthLoc data at `/mnt/sdc1/astroloc/data`) runs other people's long-running jobs. The admin's rules are hard guardrails for any command run there:

1. **Use only what is already installed.** Installing or downloading software or executable code there is off-limits: no pip/conda/apt, no cloning third-party repos, no `trust_remote_code`, no fetching model weights. When something is missing, stop and ask the user, who asks the admin.
2. **Keep personal data off it.** Only project code, configs, logs and results.
3. **Write under your own working directory.** Writing to a shared drive (e.g. `/mnt/...`) needs the admin's OK first, asked through the user.
4. **Queue behind busy GPUs.** Launch GPU work through `argus-localization/scripts/gpu_queue.py`, which waits for an idle GPU. Every process you did not start stays untouched: no kill, pkill, or `nvidia-smi` reset.
5. **Keep the machine up.** No reboot, shutdown, service restart, or driver reload.
6. **Run lean and watch.** Batch, bf16, `torch.no_grad`; check `nvidia-smi` before and during runs (`gpu_queue.py` logs it to `output/gpu_logs/`).

Preflight with `argus-localization/scripts/check_env.py` before a run; it only reads.
