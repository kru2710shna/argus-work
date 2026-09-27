"""Prove the two current Agentic GNC baselines.

1. EarthLoc retrieves known-overlap Earth map tiles from real ISS imagery.
2. FSW-Payload estimates an orbit from synthetic landmark and gyro readings.

This is deliberately a baseline report, not a real-flight validation:
the EarthLoc images are real, while the FSW sensor batch is synthetic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from agentic_gnc.fsw_od_adapter import solve_with_fsw_python
from agentic_gnc.real_earthloc_env import (
    EarthLocAction,
    RealEarthLocCaptureEnv,
    RealEarthLocConfig,
)
from agentic_gnc.simulator import circular_orbit_truth, generate_synthetic_batch


def _tile_label(path: str) -> str:
    """Return the short map-tile label embedded in an EarthLoc filename."""
    parts = Path(path).name.split("@")
    return parts[9] if len(parts) > 9 else Path(path).name


def prove_real_earthloc_retrieval(data_root: Path, device: str) -> dict:
    """Retrieve one map-tile ranking for every real ISS query in the mini episode."""
    environment = RealEarthLocCaptureEnv(
        RealEarthLocConfig(
            data_root=data_root,
            checkpoint_path=data_root / "best_trained_model.pt",
            device=device,
        )
    )
    environment.reset()

    rows: list[dict] = []
    while True:
        _, _, done, info = environment.step(EarthLocAction.CAPTURE)
        row = {
            "query": _tile_label(info["query_path"]),
            "top_tile": _tile_label(info["top_paths"][0]),
            "top_score": round(float(info["top_scores"][0]), 4),
            "known_positive_rank": info["positive_rank"],
        }
        rows.append(row)
        print(
            f"{row['query']}: top tile={row['top_tile']} | "
            f"positive rank={row['known_positive_rank']} | score={row['top_score']:.4f}"
        )
        if done:
            break

    top5_successes = sum(row["known_positive_rank"] is not None for row in rows)
    print("\n=== Real EarthLoc Retrieval Proof ===")
    print("Real ISS query images evaluated:", len(rows))
    print("Top-5 known-overlap retrievals:", f"{top5_successes}/{len(rows)}")

    if top5_successes == 0:
        raise RuntimeError("EarthLoc did not retrieve a known-overlap tile in the top 5.")

    return {
        "real_iss_queries": len(rows),
        "top5_known_overlap_retrievals": top5_successes,
        "per_query": rows,
    }


def prove_synthetic_fsw_od() -> dict:
    """Run FSW-Payload on a deterministic synthetic visual-inertial batch."""
    truth = circular_orbit_truth(steps=8, dt_s=10.0)
    batch = generate_synthetic_batch(
        truth,
        landmarks_per_frame=4,
        bearing_noise_rad=0.0,
        seed=0,
    )
    result = solve_with_fsw_python(batch)

    estimated_position_m = np.asarray(result["positions"], dtype=float) * 1_000.0
    truth_position_m = truth.positions_eci_m[: len(estimated_position_m)]
    error_m = np.linalg.norm(estimated_position_m - truth_position_m, axis=1)
    gyro_bias = np.asarray(result["gyro_bias"], dtype=float)

    print("\n=== Synthetic FSW-Payload OD Proof ===")
    print("State estimates:", len(estimated_position_m))
    print("Mean position error (m):", round(float(error_m.mean()), 3))
    print("Final position error (m):", round(float(error_m[-1]), 3))
    print("Estimated gyro bias (rad/s):", gyro_bias)

    if not np.isfinite(error_m).all():
        raise RuntimeError("FSW-Payload returned a non-finite state estimate.")

    return {
        "state_estimates": int(len(estimated_position_m)),
        "mean_position_error_m": float(error_m.mean()),
        "final_position_error_m": float(error_m[-1]),
        "estimated_gyro_bias_rad_s": gyro_bias.tolist(),
        "landmark_measurements": int(len(batch.landmarks)),
        "gyro_measurements": int(len(batch.gyros)),
    }


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    data_root = Path(os.environ["EARTHLOC_DATA_ROOT"])
    device = os.environ.get("EARTHLOC_DEVICE", "cpu")

    print("=== Agentic GNC Baseline Proof ===")
    print("EarthLoc data root:", data_root)
    print("EarthLoc device:", device)

    report = {
        "earthloc_real_iss_retrieval": prove_real_earthloc_retrieval(data_root, device),
        "fsw_synthetic_visual_inertial_od": prove_synthetic_fsw_od(),
        "scope": {
            "earthloc": "Real ISS query images and real reference map tiles.",
            "fsw": "Synthetic landmark and gyro measurements passed to the real FSW Python optimizer.",
            "not_yet": "Real-image feature matches have not yet been converted into FSW landmark bearings.",
        },
    }

    output_path = workspace_root / "argus-agentic-gnc" / "outputs" / "baseline_proof_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== Conclusion ===")
    print("PASS: EarthLoc retrieves likely Earth regions from real ISS imagery.")
    print("PASS: FSW-Payload estimates a trajectory from synthetic visual and gyro measurements.")
    print("NEXT: Convert verified real visual matches into FSW-ready landmark measurements.")
    print("Report:", output_path)


if __name__ == "__main__":
    main()
