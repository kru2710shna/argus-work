"""Offline, recommendation-only flood target hunter.

No Argus telemetry, uplink, or flight command is consumed or produced here.
Post-event pixels become available only after a simulated capture and downlink.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw

EARTH_RADIUS_KM = 6371.0088
EDGES = ("north", "south", "east", "west")


def utc_seconds(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamps must include a UTC offset")
    return parsed.timestamp()


def load_case(path: Path) -> dict:
    case = json.loads(path.read_text())
    case["_root"] = str(path.resolve().parent)

    ids = [tile["id"] for tile in case["tiles"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Tile IDs must be unique and nonempty")
    if not case["track"] or not case["initial_targets"]:
        raise ValueError("Need a predicted track and an initial target")
    if not set(case["initial_targets"]).issubset(ids):
        raise ValueError("Initial target is missing from the tile catalog")

    for tile in case["tiles"]:
        west, south, east, north = tile["bbox"]
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError(f"Invalid lon/lat bbox: {tile['id']}")
        if float(tile["priority"]) <= 0:
            raise ValueError("Priorities must be positive")

    for sample in case["track"]:
        utc_seconds(sample["ts"])
        if not (-90 <= sample["lat"] <= 90 and -180 <= sample["lon"] <= 180):
            raise ValueError("Track positions must be latitude/longitude degrees")
        if float(sample["alt_km"]) <= 0:
            raise ValueError("Track altitude must be positive")
        if "quality" in sample and not 0 <= sample["quality"] <= 1:
            raise ValueError("Quality must be between 0 and 1")

    if not 0 < case["camera_fov_degrees"] < 170:
        raise ValueError("Camera full field of view must be in (0, 170) degrees")
    if case["resources"]["max_captures"] < 1:
        raise ValueError("max_captures must be at least 1")
    return case


def image_path(case: dict, tile: dict, key: str) -> Path:
    if key not in tile:
        raise ValueError(f"{tile['id']} has no {key} image path")
    return Path(case["_root"], tile[key]).resolve()


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    value = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(value)))


def nadir_radius_km(alt_km: float, full_fov_degrees: float) -> float:
    """Approximate circular nadir footprint on a spherical Earth."""
    satellite_radius = EARTH_RADIUS_KM + alt_km
    half_angle = math.radians(full_fov_degrees / 2)
    horizon_angle = math.asin(EARTH_RADIUS_KM / satellite_radius)
    half_angle = min(half_angle, horizon_angle - 1e-9)

    travel = (
        satellite_radius * math.cos(half_angle)
        - math.sqrt(
            max(
                0.0,
                EARTH_RADIUS_KM**2
                - satellite_radius**2 * math.sin(half_angle) ** 2,
            )
        )
    )
    ground_angle = math.atan2(
        travel * math.sin(half_angle),
        satellite_radius - travel * math.cos(half_angle),
    )
    return EARTH_RADIUS_KM * ground_angle


def tile_visible(tile: dict, sample: dict, fov: float) -> bool:
    """Require all four tile corners inside the approximate camera footprint."""
    west, south, east, north = tile["bbox"]
    radius = nadir_radius_km(float(sample["alt_km"]), fov)
    return all(
        great_circle_km(sample["lat"], sample["lon"], lat, lon) <= radius
        for lat in (south, north)
        for lon in (west, east)
    )


def touching_on_edge(
    first: dict,
    second: dict,
    edge: str,
    tolerance_km: float = 0.2,
) -> bool:
    aw, a_s, ae, an = first["bbox"]
    bw, bs, be, bn = second["bbox"]
    mid_lat = (a_s + an + bs + bn) / 4
    lon_km = 111.32 * max(0.01, math.cos(math.radians(mid_lat)))

    if edge in ("east", "west"):
        gap = abs((ae - bw) if edge == "east" else (aw - be)) * lon_km
        overlap = min(an, bn) - max(a_s, bs)
        return gap <= tolerance_km and overlap >= 0.5 * min(
            an - a_s, bn - bs
        )

    gap = abs((an - bs) if edge == "north" else (a_s - bn)) * 111.32
    overlap = min(ae, be) - max(aw, bw)
    return gap <= tolerance_km and overlap >= 0.5 * min(ae - aw, be - bw)


def mask_observation(path: Path, edge_fraction: float = 0.08) -> dict:
    """Read a binary mask: white=water, black=other."""
    with Image.open(path) as original:
        mask = original.convert("L")
        width, height = mask.size
        if min(width, height) < 8:
            raise ValueError("Water mask must be at least 8 by 8")

        strip = max(1, round(min(width, height) * 0.05))
        bands = {
            "north": (0, 0, width, strip),
            "south": (0, height - strip, width, height),
            "west": (0, 0, strip, height),
            "east": (width - strip, 0, width, height),
        }

        flooded = sum(pixel >= 128 for pixel in mask.tobytes())
        touched = []
        for edge, box in bands.items():
            region = mask.crop(box)
            fraction = sum(
                pixel >= 128 for pixel in region.tobytes()
            ) / (region.width * region.height)
            if fraction >= edge_fraction and flooded >= 5:
                touched.append(edge)

        return {
            "flood_observed": flooded >= 5,
            "touches_edges": touched,
            "confidence": 1.0,
            "flood_pixels": flooded,
        }


def ollama_chat(
    model: str,
    messages: list[dict],
    schema: dict,
    url: str,
) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "format": schema,
        "options": {"temperature": 0},
        "messages": messages,
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        answer = json.load(response)

    result = json.loads(answer["message"]["content"])
    if not isinstance(result, dict):
        raise ValueError("Model did not return a JSON object")
    return result


def jpeg_b64(path: Path) -> str:
    with Image.open(path) as original:
        image = original.convert("RGB")
        image.thumbnail((768, 768))
        stream = io.BytesIO()
        image.save(stream, "JPEG", quality=88)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def observe(
    case: dict,
    tile: dict,
    mode: str,
    vision_model: str,
    url: str,
) -> dict:
    if mode in ("oracle", "mask"):
        key = (
            "truth_water_mask"
            if mode == "oracle"
            else "predicted_water_mask"
        )
        result = mask_observation(image_path(case, tile, key))
        result["source"] = (
            "ground_truth_upper_bound"
            if mode == "oracle"
            else "external_predicted_mask"
        )
        return result

    if mode != "ollama":
        raise ValueError(f"Unknown vision mode: {mode}")

    post = jpeg_b64(image_path(case, tile, "post_image"))
    images = [post]
    prompt = (
        "Inspect this post-event satellite image for NEW floodwater. "
        "Return flood_observed, confidence 0..1, and only frame edges "
        "where new floodwater plausibly continues past the image. "
        "Do not confuse permanent water or clouds with flooding. "
        "If uncertain, return no edges."
    )
    if tile.get("pre_image"):
        images.insert(0, jpeg_b64(image_path(case, tile, "pre_image")))
        prompt += " The first image is pre-event and the second is post-event."

    schema = {
        "type": "object",
        "properties": {
            "flood_observed": {"type": "boolean"},
            "confidence": {"type": "number"},
            "touches_edges": {
                "type": "array",
                "items": {"type": "string", "enum": list(EDGES)},
            },
        },
        "required": ["flood_observed", "confidence", "touches_edges"],
        "additionalProperties": False,
    }
    result = ollama_chat(
        vision_model,
        [{"role": "user", "content": prompt, "images": images}],
        schema,
        url,
    )
    if (
        type(result.get("flood_observed")) is not bool
        or type(result.get("confidence")) not in (int, float)
        or not 0 <= result["confidence"] <= 1
        or not isinstance(result.get("touches_edges"), list)
        or any(edge not in EDGES for edge in result["touches_edges"])
    ):
        raise ValueError("Invalid image model observation")

    result["source"] = f"vision_model:{vision_model}"
    return result


def options(
    case: dict,
    active: set[str],
    captured: set[str],
    now: float,
    battery: float,
    storage: float,
    frontier: set[str],
) -> list[dict]:
    resources = case["resources"]
    if (
        battery - resources["battery_cost_pct"]
        < resources["min_battery_pct"]
    ):
        return []
    if (
        storage - resources["image_size_mb"]
        < resources["min_storage_mb"]
    ):
        return []

    catalog = {tile["id"]: tile for tile in case["tiles"]}
    found = []

    for tile_id in sorted(active - captured):
        tile = catalog[tile_id]
        for sample in case["track"]:
            timestamp = utc_seconds(sample["ts"])
            if timestamp <= now + resources["command_lead_s"]:
                continue
            if not tile_visible(
                tile, sample, case["camera_fov_degrees"]
            ):
                continue

            quality = float(sample.get("quality", 0.5))
            score = (
                float(tile["priority"])
                * quality
                * (3.0 if tile_id in frontier else 1.0)
                / (1.0 + (timestamp - now) / 7200.0)
            )
            found.append({
                "id": f"{tile_id}@{sample['ts']}",
                "tile_id": tile_id,
                "ts": sample["ts"],
                "score": round(score, 6),
                "quality": quality,
                "quality_source": sample.get(
                    "quality_source",
                    "provided"
                    if "quality" in sample
                    else "assumed_neutral",
                ),
            })

    return found


def select(
    candidates: list[dict],
    case: dict,
    planner: str,
    model: str,
    url: str,
) -> tuple[dict, dict]:
    if planner == "static":
        priorities = {
            tile["id"]: float(tile["priority"])
            for tile in case["tiles"]
        }
        choice = max(
            candidates,
            key=lambda item: (
                priorities[item["tile_id"]],
                item["score"],
                -utc_seconds(item["ts"]),
                item["id"],
            ),
        )
    else:
        choice = max(
            candidates,
            key=lambda item: (
                item["score"],
                -utc_seconds(item["ts"]),
                item["id"],
            ),
        )

    trace = {
        "rule_choice": choice["id"],
        "model_status": "not_requested",
    }
    if planner != "ollama":
        return choice, trace

    allowed = [item["id"] for item in candidates]
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "enum": allowed},
            "reason": {"type": "string"},
        },
        "required": ["id", "reason"],
        "additionalProperties": False,
    }
    public_candidates = [
        {
            "id": item["id"],
            "score": item["score"],
            "quality": item["quality"],
            "quality_source": item["quality_source"],
        }
        for item in candidates
    ]

    try:
        reply = ollama_chat(
            model,
            [
                {
                    "role": "system",
                    "content": (
                        "Recommend one listed imaging opportunity to map "
                        "flooded area. Output JSON. Never issue spacecraft commands."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"eligible": public_candidates},
                        sort_keys=True,
                    ),
                },
            ],
            schema,
            url,
        )
        if (
            reply.get("id") not in allowed
            or not isinstance(reply.get("reason"), str)
        ):
            raise ValueError("Model selected an unlisted opportunity")
        choice = next(
            item for item in candidates
            if item["id"] == reply["id"]
        )
        trace.update(
            model_status="valid",
            model_choice=choice["id"],
            model_reason=reply["reason"],
        )
    except Exception as error:
        trace.update(
            model_status="fallback_to_rule",
            model_error=str(error),
        )

    return choice, trace


def evaluate(case: dict, captured: list[str]) -> dict:
    total = found = 0
    for tile in case["tiles"]:
        if "truth_water_mask" not in tile:
            continue
        count = mask_observation(
            image_path(case, tile, "truth_water_mask")
        )["flood_pixels"]
        total += count
        if tile["id"] in captured:
            found += count

    return {
        "captured_water_pixels": found,
        "catalog_water_pixels": total,
        "fraction": round(found / total, 4) if total else None,
        "meaning": (
            "Held-out labeled water pixels in listed tiles; "
            "not geodesic area."
        ),
    }


def run(
    case: dict,
    planner: str,
    vision_mode: str,
    model: str,
    vision_model: str,
    url: str,
) -> dict:
    if planner not in ("adaptive", "static", "ollama"):
        raise ValueError("Unknown planner")

    catalog = {tile["id"]: tile for tile in case["tiles"]}
    active = set(case["initial_targets"])
    captured = []
    frontier = set()
    now = utc_seconds(case["now"])
    resources = case["resources"]
    battery = float(resources["battery_pct"])
    storage = float(resources["storage_mb"])
    events = []

    for _ in range(int(resources["max_captures"])):
        candidates = options(
            case,
            active,
            set(captured),
            now,
            battery,
            storage,
            frontier,
        )
        if not candidates:
            events.append({
                "status": "no_eligible_opportunity",
                "active_targets": sorted(active),
            })
            break

        chosen, decision = select(
            candidates, case, planner, model, url
        )
        tile = catalog[chosen["tile_id"]]

        now = utc_seconds(chosen["ts"])
        battery -= resources["battery_cost_pct"]
        storage -= resources["image_size_mb"]
        captured.append(tile["id"])

        # The next decision cannot inspect this image until the downlink.
        now += resources["downlink_delay_s"]

        event = {
            "status": "capture_replay",
            "tile_id": tile["id"],
            "capture_ts": chosen["ts"],
            "image_available_ts": datetime.fromtimestamp(
                now, timezone.utc
            ).isoformat(),
            "score": chosen["score"],
            "decision": decision,
        }

        try:
            with Image.open(
                image_path(case, tile, "post_image")
            ) as image:
                image.verify()

            observation = observe(
                case, tile, vision_mode, vision_model, url
            )
            event["observation"] = observation

            if (
                planner != "static"
                and observation["flood_observed"]
                and observation["confidence"] >= 0.65
            ):
                proposed = []
                for neighbor in case["tiles"]:
                    if (
                        neighbor["id"] == tile["id"]
                        or neighbor["id"] in captured
                    ):
                        continue

                    if any(
                        touching_on_edge(tile, neighbor, edge)
                        for edge in observation["touches_edges"]
                    ):
                        active.add(neighbor["id"])
                        frontier.add(neighbor["id"])
                        proposed.append(neighbor["id"])

                event["new_targets"] = sorted(proposed)

        except Exception as error:
            event["observation"] = {
                "status": "unavailable",
                "error": str(error),
            }
            event["new_targets"] = []

        events.append(event)

    return {
        "case": case["name"],
        "image_source": case["image_source"],
        "track_source": case["track_source"],
        "planner": planner,
        "vision_mode": vision_mode,
        "recommendation_only": True,
        "captured": captured,
        "events": events,
        "evaluation": evaluate(case, captured),
        "battery_pct_remaining": round(battery, 3),
        "storage_mb_remaining": round(storage, 3),
    }


def make_demo(folder: Path) -> Path:
    """Generate a small synthetic case to check the complete decision loop."""
    folder.mkdir(parents=True, exist_ok=True)

    tiles = [
        (
            "A",
            [-90.000, 29.995, -89.990, 30.005],
            4,
            (64, 35, 99, 65),
        ),
        (
            "B",
            [-89.990, 29.995, -89.980, 30.005],
            2,
            (0, 35, 40, 65),
        ),
        (
            "C",
            [-90.300, 29.995, -90.290, 30.005],
            2,
            None,
        ),
    ]

    records = []
    for tile_id, bbox, priority, flood in tiles:
        pre = Image.new("RGB", (100, 100), (58, 99, 55))
        post = pre.copy()
        mask = Image.new("L", (100, 100), 0)

        if flood:
            ImageDraw.Draw(post).rectangle(
                flood, fill=(22, 76, 160)
            )
            ImageDraw.Draw(mask).rectangle(
                flood, fill=255
            )

        pre.save(folder / f"{tile_id}_pre.png")
        post.save(folder / f"{tile_id}_post.png")
        mask.save(folder / f"{tile_id}_truth_water.png")

        records.append({
            "id": tile_id,
            "bbox": bbox,
            "priority": priority,
            "pre_image": f"{tile_id}_pre.png",
            "post_image": f"{tile_id}_post.png",
            "truth_water_mask": f"{tile_id}_truth_water.png",
        })

    case = {
        "name": "synthetic_flood_frontier",
        "image_source": "generated_synthetic",
        "track_source": "generated_synthetic",
        "now": "2026-10-21T00:00:00Z",
        "camera_fov_degrees": 9,
        "initial_targets": ["A", "C"],
        "tiles": records,
        "track": [
            {
                "ts": "2026-10-21T00:01:00Z",
                "lat": 30.0,
                "lon": -89.995,
                "alt_km": 500,
                "quality": 0.8,
            },
            {
                "ts": "2026-10-21T00:02:00Z",
                "lat": 30.0,
                "lon": -90.295,
                "alt_km": 500,
                "quality": 0.9,
            },
            {
                "ts": "2026-10-21T00:03:00Z",
                "lat": 30.0,
                "lon": -89.985,
                "alt_km": 500,
                "quality": 0.9,
            },
        ],
        "resources": {
            "max_captures": 2,
            "battery_pct": 80,
            "min_battery_pct": 20,
            "battery_cost_pct": 5,
            "storage_mb": 500,
            "min_storage_mb": 30,
            "image_size_mb": 20,
            "command_lead_s": 10,
            "downlink_delay_s": 20,
        },
    }

    path = folder / "case.json"
    path.write_text(json.dumps(case, indent=2) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(
        dest="command", required=True
    )

    demo = sub.add_parser(
        "demo",
        help="Generate and replay a synthetic labeled case",
    )
    demo.add_argument(
        "--output",
        type=Path,
        default=Path("flood_hunter_demo"),
    )

    trial = sub.add_parser(
        "run",
        help="Replay a supplied image manifest",
    )
    trial.add_argument("case", type=Path)
    trial.add_argument(
        "--output",
        type=Path,
        default=Path("flood_hunter_result.json"),
    )
    trial.add_argument(
        "--planner",
        choices=["adaptive", "static", "ollama"],
        default="adaptive",
    )
    trial.add_argument(
        "--vision-mode",
        choices=["oracle", "mask", "ollama"],
        required=True,
    )
    trial.add_argument(
        "--model",
        default="qwen2.5:1.5b",
    )
    trial.add_argument(
        "--vision-model",
        default="qwen2.5vl:3b",
    )
    trial.add_argument(
        "--ollama-url",
        default="http://localhost:11434/api/chat",
    )

    args = parser.parse_args()

    if args.command == "demo":
        case = load_case(make_demo(args.output))
        result = {
            "adaptive": run(
                case, "adaptive", "oracle", "", "", ""
            ),
            "fixed_priority": run(
                case, "static", "oracle", "", "", ""
            ),
        }

        assert result["adaptive"]["captured"] == [
            "A", "B"
        ], "Adaptive frontier replay failed"
        assert result["fixed_priority"]["captured"] == [
            "A", "C"
        ], "Fixed priority replay failed"
        assert (
            result["adaptive"]["evaluation"]["fraction"]
            > result["fixed_priority"]["evaluation"]["fraction"]
        )
        assert result["adaptive"]["events"][0][
            "new_targets"
        ] == ["B"]

        output = args.output / "results.json"

    else:
        case = load_case(args.case)
        result = {
            "adaptive": run(
                case,
                args.planner,
                args.vision_mode,
                args.model,
                args.vision_model,
                args.ollama_url,
            ),
            "fixed_priority": run(
                case,
                "static",
                args.vision_mode,
                args.model,
                args.vision_model,
                args.ollama_url,
            ),
        }
        output = args.output

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print(json.dumps({
        "adaptive_captures": result["adaptive"]["captured"],
        "fixed_captures": result["fixed_priority"]["captured"],
        "adaptive_coverage": result["adaptive"]["evaluation"],
        "fixed_coverage": result["fixed_priority"]["evaluation"],
        "result": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()