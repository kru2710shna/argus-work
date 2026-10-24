"""Offline, synthetic-only ground imaging decision replay.

The model chooses an opportunity ID; this program never sends commands.
Policy thresholds in the input file are experimental, not flight rules.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def eligible_windows(case: dict) -> tuple[list[dict], dict[str, list[str]]]:
    telemetry = case["telemetry"]
    policy = case["demo_policy"]
    rejected: dict[str, list[str]] = {}
    eligible: list[dict] = []
    seen: set[str] = set()

    for window in case["windows"]:
        window_id = window["id"]
        if not isinstance(window_id, str) or not window_id or window_id in seen:
            raise ValueError("Window IDs must be unique nonempty strings")
        seen.add(window_id)
        reasons: list[str] = []

        if telemetry["experiment_queue_depth"] != 0:
            reasons.append("experiment already queued")
        if telemetry["battery_pct"] < policy["min_battery_pct"]:
            reasons.append("battery below demo threshold")
        if telemetry["storage_free_mb"] < policy["min_storage_free_mb"]:
            reasons.append("storage below demo threshold")
        if window["ts_unix_s"] <= case["now_unix_s"]:
            reasons.append("window is not in the future")
        if not 1 <= window["camera_bit_flag"] <= 15:
            reasons.append("camera mask must select one of four camera bits")
        if window["duration_s"] <= 0 or window["capture_rate_hz"] <= 0:
            reasons.append("invalid duration or capture rate")
        if not 0 <= window["land_fraction"] <= 1:
            reasons.append("land_fraction outside [0, 1]")
        elif window["land_fraction"] < policy["min_land_fraction"]:
            reasons.append("insufficient land in demo forecast")
        if not 0 <= window["cloud_fraction"] <= 1:
            reasons.append("cloud_fraction outside [0, 1]")
        elif window["cloud_fraction"] > policy["max_cloud_fraction"]:
            reasons.append("cloud cover above demo threshold")

        if reasons:
            rejected[window_id] = reasons
        else:
            eligible.append(window)

    return eligible, rejected


def rules_choice(windows: list[dict]) -> str:
    if not windows:
        return "SKIP"
    return max(
        windows,
        key=lambda w: (w["land_fraction"] * (1 - w["cloud_fraction"]), -w["ts_unix_s"]),
    )["id"]


def ask_ollama(model: str, case: dict, windows: list[dict], url: str) -> dict:
    allowed = [w["id"] for w in windows] + ["SKIP"]
    schema = {
        "type": "object",
        "properties": {
            "window_id": {"type": "string", "enum": allowed},
            "reason": {"type": "string"},
        },
        "required": ["window_id", "reason"],
        "additionalProperties": False,
    }
    prompt = {
        "telemetry": case["telemetry"],
        "demo_policy": case["demo_policy"],
        "eligible_windows": windows,
        "allowed_window_ids": allowed,
    }
    body = {
        "model": model,
        "stream": False,
        "format": schema,
        "options": {"temperature": 0, "num_ctx": 2048},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You advise a human ground operator on the next imaging window. "
                    "Select only a listed window ID, or SKIP. Prefer clear land imagery. "
                    "Return JSON matching the provided schema. No spacecraft commands."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, sort_keys=True)},
        ],
    }
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        reply = json.load(response)
    result = json.loads(reply["message"]["content"])
    if not isinstance(result, dict) or not isinstance(result.get("reason"), str):
        raise ValueError("Model returned invalid decision JSON")
    if result.get("window_id") not in allowed:
        raise ValueError(f"Model selected an unlisted window: {result.get('window_id')!r}")
    return result


def run(case: dict, models: list[str], url: str) -> dict:
    if case.get("source") != "synthetic":
        raise ValueError("This first replay accepts only explicitly synthetic inputs")
    windows, rejected = eligible_windows(case)
    rule = rules_choice(windows)
    report = {
        "case": case["name"],
        "source": case["source"],
        "note": "Offline demonstration only; policy thresholds are not flight rules.",
        "eligible_ids": [w["id"] for w in windows],
        "rejected": rejected,
        "rules_baseline": rule,
        "models": [],
    }
    for model in models:
        start = time.perf_counter()
        try:
            decision = ask_ollama(model, case, windows, url)
            row = {"model": model, "status": "valid", **decision}
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError) as error:
            row = {"model": model, "status": "rejected", "window_id": "SKIP", "error": str(error)}
        row["elapsed_s"] = round(time.perf_counter() - start, 3)
        report["models"].append(row)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="Synthetic replay JSON")
    parser.add_argument("--models", nargs="*", default=[], help="Locally pulled Ollama model tags")
    parser.add_argument("--output", type=Path, default=Path("/tmp/ground_imaging_replay.json"))
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/chat")
    args = parser.parse_args()
    report = run(json.loads(args.case.read_text()), args.models, args.ollama_url)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
