"""Create PNG, GIF, and CSV visualizations for the Agentic GNC baseline."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from agentic_gnc.environment import AgenticGNCEnvironment, SensingAction


OUTPUT_DIR = Path("argus-agentic-gnc/outputs")


def run_episode(name: str, should_capture) -> tuple[AgenticGNCEnvironment, list[dict]]:
    """Run one deterministic sensing policy and record its visible outcomes."""
    environment = AgenticGNCEnvironment()
    environment.reset(seed=0)
    history: list[dict] = []

    for frame in range(environment.config.steps):
        action = SensingAction.CAPTURE if should_capture(frame) else SensingAction.SKIP
        _, reward, done, info = environment.step(action)
        history.append(
            {
                "policy": name,
                "frame": frame,
                "time_s": frame * environment.config.dt_s,
                "action": action.name,
                "images_taken": info["images_taken"],
                "solver_ran": info["solver_ran"],
                "position_error_m": info["position_error_m"],
                "reward": reward,
                "done": done,
            }
        )
    return environment, history


def last_error(history: list[dict]) -> float:
    values = [row["position_error_m"] for row in history if row["position_error_m"] is not None]
    return float(values[-1])


def save_csv(histories: list[list[dict]]) -> None:
    rows = [row for history in histories for row in history]
    with (OUTPUT_DIR / "episode_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(
    fixed_environment: AgenticGNCEnvironment,
    fixed: list[dict],
    sparse_environment: AgenticGNCEnvironment,
    sparse: list[dict],
) -> None:
    fixed_positions = fixed_environment._truth.positions_eci_m / 1_000.0
    sparse_positions = sparse_environment._truth.positions_eci_m / 1_000.0
    fixed_capture = [row["action"] == "CAPTURE" for row in fixed]
    sparse_capture = [row["action"] == "CAPTURE" for row in sparse]

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    figure.suptitle("Argus Agentic GNC: End-to-End Baseline", fontsize=16)

    orbit_axis = axes[0, 0]
    orbit_axis.plot(fixed_positions[:, 0], fixed_positions[:, 1], color="#4C78A8", label="truth orbit")
    orbit_axis.scatter(
        fixed_positions[fixed_capture, 0], fixed_positions[fixed_capture, 1],
        color="#54A24B", marker="o", label="capture-all image",
    )
    orbit_axis.scatter(
        sparse_positions[sparse_capture, 0], sparse_positions[sparse_capture, 1],
        color="#F58518", marker="x", s=70, label="sparse image",
    )
    orbit_axis.set_title("Synthetic orbit and image captures")
    orbit_axis.set_xlabel("ECI x (km)")
    orbit_axis.set_ylabel("ECI y (km)")
    orbit_axis.axis("equal")
    orbit_axis.legend(fontsize=8)

    schedule_axis = axes[0, 1]
    frames = np.arange(len(fixed))
    schedule_axis.scatter(frames[fixed_capture], np.ones(sum(fixed_capture)), color="#54A24B", s=70, label="capture-all")
    schedule_axis.scatter(frames[sparse_capture], np.zeros(sum(sparse_capture)), color="#F58518", s=70, label="sparse")
    schedule_axis.set_yticks([0, 1], ["sparse", "capture-all"])
    schedule_axis.set_xticks(frames)
    schedule_axis.set_xlabel("frame")
    schedule_axis.set_title("Agent action schedule")
    schedule_axis.grid(axis="x", alpha=0.3)

    error_axis = axes[1, 0]
    for label, history, color in [("capture-all", fixed, "#54A24B"), ("sparse", sparse, "#F58518")]:
        error_axis.plot(
            [row["frame"] for row in history if row["position_error_m"] is not None],
            [row["position_error_m"] for row in history if row["position_error_m"] is not None],
            marker="o", color=color, label=label,
        )
    error_axis.set_title("OD error when the solver runs")
    error_axis.set_xlabel("frame")
    error_axis.set_ylabel("position error (m)")
    error_axis.legend()
    error_axis.grid(alpha=0.3)

    comparison_axis = axes[1, 1]
    names = ["capture-all", "sparse"]
    image_counts = [fixed[-1]["images_taken"], sparse[-1]["images_taken"]]
    errors = [last_error(fixed), last_error(sparse)]
    x = np.arange(2)
    width = 0.34
    left = comparison_axis.bar(x - width / 2, image_counts, width, label="images", color="#4C78A8")
    right = comparison_axis.bar(x + width / 2, errors, width, label="final error (m)", color="#E45756")
    comparison_axis.bar_label(left, fmt="%.0f", padding=3)
    comparison_axis.bar_label(right, fmt="%.1f", padding=3)
    comparison_axis.set_xticks(x, names)
    comparison_axis.set_title("Sensing-cost / localization trade-off")
    comparison_axis.legend()

    figure.savefig(OUTPUT_DIR / "agentic_gnc_summary.png", dpi=180)
    plt.close(figure)


def save_animation(
    fixed_environment: AgenticGNCEnvironment,
    fixed: list[dict],
    sparse_environment: AgenticGNCEnvironment,
    sparse: list[dict],
) -> None:
    fixed_positions = fixed_environment._truth.positions_eci_m / 1_000.0
    sparse_positions = sparse_environment._truth.positions_eci_m / 1_000.0
    limit = float(max(np.abs(fixed_positions[:, :2]).max(), np.abs(sparse_positions[:, :2]).max()) * 1.1)

    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    def draw(frame: int) -> None:
        for axis, title, positions, history, color in [
            (axes[0], "Capture every frame", fixed_positions, fixed, "#54A24B"),
            (axes[1], "Sparse sensing policy", sparse_positions, sparse, "#F58518"),
        ]:
            axis.clear()
            axis.plot(positions[: frame + 1, 0], positions[: frame + 1, 1], color="#4C78A8", linewidth=2, label="orbit path")
            captured = [row["action"] == "CAPTURE" for row in history[: frame + 1]]
            axis.scatter(positions[: frame + 1][captured, 0], positions[: frame + 1][captured, 1], color=color, s=55, label="image captured")
            axis.scatter(positions[frame, 0], positions[frame, 1], color="#111111", marker="*", s=150, label="satellite")
            axis.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="ECI x (km)", ylabel="ECI y (km)", title=title)
            axis.set_aspect("equal")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8, loc="upper right")
            row = history[frame]
            error = row["position_error_m"]
            detail = "no OD update" if error is None else f"OD error: {error:.1f} m"
            axis.text(0.02, 0.02, f"frame {frame} | {row['action']} | images {row['images_taken']}\n{detail}", transform=axis.transAxes, fontsize=9, va="bottom")
        figure.suptitle("Agentic sensing decision sequence")

    animation = FuncAnimation(figure, draw, frames=len(fixed), interval=900, repeat=True)
    animation.save(OUTPUT_DIR / "agentic_gnc_capture_policy.gif", writer=PillowWriter(fps=1.1), dpi=120)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fixed_environment, fixed = run_episode("capture_all", lambda frame: True)
    sparse_environment, sparse = run_episode("sparse", lambda frame: frame < 2 or frame % 2 == 1)
    save_csv([fixed, sparse])
    plot_summary(fixed_environment, fixed, sparse_environment, sparse)
    save_animation(fixed_environment, fixed, sparse_environment, sparse)
    print("Created:")
    print(" -", OUTPUT_DIR / "agentic_gnc_summary.png")
    print(" -", OUTPUT_DIR / "agentic_gnc_capture_policy.gif")
    print(" -", OUTPUT_DIR / "episode_metrics.csv")


if __name__ == "__main__":
    main()
