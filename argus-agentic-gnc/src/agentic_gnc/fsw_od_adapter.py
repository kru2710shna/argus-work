"""Adapter from the shared ODBatch contract to FSW-Payload's Python OD solver."""

from pathlib import Path
import sys

from agentic_gnc.contracts import ODBatch


def solve_with_fsw_python(batch: ODBatch) -> dict:
    """Run FSW-Payload's Python/CasADi batch orbit-determination solver."""

    workspace_root = Path(__file__).resolve().parents[3]
    fsw_python_od = workspace_root / "FSW-Payload" / "python_od"

    if not fsw_python_od.is_dir():
        raise FileNotFoundError(f"FSW Python OD directory not found: {fsw_python_od}")

    if str(fsw_python_od) not in sys.path:
        sys.path.insert(0, str(fsw_python_od))

    from optimizer import build_and_solve

    # The Python optimizer documents its position states in km.
    landmark_rows, group_starts, gyro_rows, sigmas = batch.to_fsw_arrays(
        landmark_position_unit="km"
    )

    return build_and_solve(
        landmark_measurements=landmark_rows,
        landmark_group_starts=group_starts,
        gyro_measurements=gyro_rows,
        landmark_uncertainties=sigmas,
        use_j2=False,
        use_drag=False,
        compute_covariance=False,
    )