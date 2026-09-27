"""Render a mission-style replay of the current Agentic GNC sensing baseline.

This is a visual simulator: it shows the synthetic orbit, a nadir camera
footprint on an optional Earth texture, CAPTURE/SKIP actions, and the FSW OD
estimate. It does not claim to run EarthLoc retrieval yet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from PIL import Image

from agentic_gnc.environment import AgenticGNCEnvironment, SensingAction


OUTPUT_DIR = Path("argus-agentic-gnc/outputs")


def eci_to_latlon(positions_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate sub-satellite latitude/longitude for this spherical-Earth demo."""
    radii = np.linalg.norm(positions_m, axis=1)
    latitude = np.degrees(np.arcsin(positions_m[:, 2] / radii))
    longitude = np.degrees(np.arctan2(positions_m[:, 1], positions_m[:, 0]))
    return latitude, longitude


def find_texture(explicit_path: str | None) -> Path | None:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Earth texture does not exist: {path}")
        return path
    texture_root = Path("Vision_Payload/blue_marble")
    if texture_root.exists():
        candidates = sorted(
            path for path in texture_root.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if candidates:
            return candidates[0]
    return None


def load_texture(texture_path: Path | None) -> np.ndarray:
    if texture_path is not None:
        image = Image.open(texture_path).convert("RGB")
        image.thumbnail((2048, 1024))
        return np.asarray(image)

    # Honest fallback when no real Earth image has been supplied.
    height, width = 360, 720
    latitude = np.linspace(-90, 90, height)[:, None]
    ocean = np.clip(0.45 + 0.28 * np.cos(np.radians(latitude)), 0, 1)
    texture = np.zeros((height, width, 3), dtype=float)
    texture[..., 0] = 0.05
    texture[..., 1] = 0.15 + 0.35 * ocean
    texture[..., 2] = 0.25 + 0.55 * ocean
    return texture


def camera_crop(texture: np.ndarray, longitude: float, latitude: float, fov_deg: float = 18.0) -> np.ndarray:
    """Take a wrapped equirectangular crop representing a nadir camera image."""
    height, width = texture.shape[:2]
    crop_width = max(32, int(width * fov_deg / 360.0))
    crop_height = max(32, int(height * fov_deg / 180.0))
    center_x = int((longitude + 180.0) / 360.0 * width) + width
    center_y = int((90.0 - latitude) / 180.0 * height)
    extended = np.concatenate([texture, texture, texture], axis=1)
    x0, x1 = center_x - crop_width // 2, center_x + crop_width // 2
    y0 = max(0, center_y - crop_height // 2)
    y1 = min(height, center_y + crop_height // 2)
    return extended[y0:y1, x0:x1]


def policy_action(policy: str, frame: int) -> SensingAction:
    if policy == "capture-all":
        return SensingAction.CAPTURE
    return SensingAction.CAPTURE if frame < 2 or frame % 2 == 1 else SensingAction.SKIP


def run_mission(policy: str) -> tuple[AgenticGNCEnvironment, list[dict]]:
    environment = AgenticGNCEnvironment()
    environment.reset(seed=0)
    records: list[dict] = []
    for frame in range(environment.config.steps):
        action = policy_action(policy, frame)
        observation, reward, done, info = environment.step(action)
        records.append(
            {
                "frame": frame,
                "action": action.name,
                "reward": reward,
                "images": info["images_taken"],
                "solver_ran": info["solver_ran"],
                "error_m": info["position_error_m"],
                "estimate_km": observation["estimated_position_km"],
                "done": done,
            }
        )
    return environment, records


def render(policy: str, texture_path: Path | None, fps: float) -> tuple[Path, Path]:
    environment, records = run_mission(policy)
    truth = environment._truth.positions_eci_m
    latitude, longitude = eci_to_latlon(truth)
    texture = load_texture(texture_path)
    used_real_texture = texture_path is not None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(15, 7), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1.45, 1, 1), height_ratios=(1, 1))
    earth_axis = figure.add_subplot(grid[:, 0])
    camera_axis = figure.add_subplot(grid[0, 1])
    estimate_axis = figure.add_subplot(grid[1, 1])
    status_axis = figure.add_subplot(grid[:, 2])
    last_capture_crop: np.ndarray | None = None

    def draw(frame: int) -> None:
        nonlocal last_capture_crop
        record = records[frame]
        action = record["action"]
        current_lon, current_lat = longitude[frame], latitude[frame]
        if action == "CAPTURE" or last_capture_crop is None:
            last_capture_crop = camera_crop(texture, current_lon, current_lat)

        earth_axis.clear()
        earth_axis.imshow(texture, extent=(-180, 180, -90, 90), origin="upper")
        earth_axis.plot(longitude[: frame + 1], latitude[: frame + 1], color="white", linewidth=2, label="ground track")
        captured = [entry["action"] == "CAPTURE" for entry in records[: frame + 1]]
        earth_axis.scatter(longitude[: frame + 1][captured], latitude[: frame + 1][captured], color="#FFD23F", edgecolor="black", s=45, label="captured tile")
        earth_axis.scatter([current_lon], [current_lat], color="#FF6B6B", edgecolor="black", marker="*", s=180, label="satellite")
        earth_axis.add_patch(Rectangle((current_lon - 9, current_lat - 9), 18, 18, fill=False, edgecolor="#FF6B6B", linewidth=2, label="camera footprint"))
        earth_axis.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="longitude (deg)", ylabel="latitude (deg)", title="Earth view: orbit ground track and camera footprint")
        earth_axis.legend(loc="lower left", fontsize=8)

        camera_axis.clear()
        camera_axis.imshow(last_capture_crop)
        camera_axis.axhline(last_capture_crop.shape[0] / 2, color="white", alpha=0.8)
        camera_axis.axvline(last_capture_crop.shape[1] / 2, color="white", alpha=0.8)
        camera_axis.set_xticks([])
        camera_axis.set_yticks([])
        camera_axis.set_title("Nadir camera tile" if action == "CAPTURE" else "Previous captured tile (SKIP)")
        camera_axis.text(0.02, 0.04, f"{action}\nlat {current_lat:.2f}°, lon {current_lon:.2f}°", color="white", fontsize=10, transform=camera_axis.transAxes, bbox={"facecolor": "black", "alpha": 0.6, "pad": 4})

        estimate_axis.clear()
        update_frames = [item["frame"] for item in records[: frame + 1] if item["error_m"] is not None]
        update_errors = [item["error_m"] for item in records[: frame + 1] if item["error_m"] is not None]
        if update_frames:
            estimate_axis.plot(update_frames, update_errors, marker="o", color="#4C78A8", label="FSW OD error")
            estimate_axis.legend(fontsize=8)
        else:
            estimate_axis.text(0.5, 0.5, "Waiting for 3 captured frames\nbefore first OD solution", ha="center", va="center", transform=estimate_axis.transAxes)
        estimate_axis.set(xlim=(-0.2, len(records) - 0.8), ylim=(0, max(100, max(update_errors, default=0) * 1.2)), xlabel="frame", ylabel="position error (m)", title="Orbit determination update")
        estimate_axis.grid(alpha=0.3)

        status_axis.clear()
        status_axis.axis("off")
        error_text = "not available" if record["error_m"] is None else f"{record['error_m']:.1f} m"
        estimate_text = "waiting for OD" if not np.any(record["estimate_km"]) else "FSW OD estimate available"
        texture_text = "real Blue Marble texture" if used_real_texture else "schematic fallback (no texture found)"
        status_axis.text(0.05, 0.94, "Agentic GNC mission replay", fontsize=15, fontweight="bold", va="top", transform=status_axis.transAxes)
        status_axis.text(0.05, 0.76, f"Frame: {frame + 1} / {len(records)}\nAction: {action}\nImages used: {record['images']}\n\nFSW solver ran: {record['solver_ran']}\nPosition error: {error_text}\nEstimate: {estimate_text}\n\nTexture: {texture_text}", fontsize=12, va="top", transform=status_axis.transAxes)
        status_axis.text(0.05, 0.12, "Current scope\n• synthetic orbit + gyro\n• visual camera footprint\n• FSW Python OD\n\nNext scope\n• EarthLoc retrieval\n• actual matched reference tiles\n• learned policy", fontsize=10, va="top", transform=status_axis.transAxes)
        figure.suptitle(f"{policy} policy — 10 s per simulation frame", fontsize=16)

    animation = FuncAnimation(figure, draw, frames=len(records), interval=1000 / fps, repeat=True)
    gif_path = OUTPUT_DIR / f"mission_replay_{policy}.gif"
    png_path = OUTPUT_DIR / f"mission_replay_{policy}_final.png"
    animation.save(gif_path, writer=PillowWriter(fps=fps), dpi=120)
    draw(len(records) - 1)
    figure.savefig(png_path, dpi=160)
    plt.close(figure)
    return gif_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("sparse", "capture-all"), default="sparse")
    parser.add_argument("--texture", help="Optional equirectangular Blue Marble PNG/JPG path")
    parser.add_argument("--fps", type=float, default=1.2)
    args = parser.parse_args()
    texture_path = find_texture(args.texture)
    gif_path, png_path = render(args.policy, texture_path, args.fps)
    print("Created:")
    print(" -", gif_path)
    print(" -", png_path)


if __name__ == "__main__":
    main()
