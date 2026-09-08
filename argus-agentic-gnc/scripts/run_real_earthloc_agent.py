"""Run a transparent first agent over the real EarthLoc mini-episode."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_gnc.real_earthloc_env import EarthLocAction, RealEarthLocCaptureEnv


class ConservativeCaptureAgent:
    """Initial policy: warm up with two captures, then capture every other frame."""

    def act(self, observation: dict) -> EarthLocAction:
        if observation["frame"] < 2 or observation["frames_since_capture"] >= 1:
            return EarthLocAction.CAPTURE
        return EarthLocAction.SKIP


def main() -> None:
    environment = RealEarthLocCaptureEnv()
    agent = ConservativeCaptureAgent()
    observation = environment.reset()
    total_reward = 0.0
    history = []

    while True:
        action = agent.act(observation)
        observation, reward, done, info = environment.step(action)
        total_reward += reward
        history.append({**info, "reward": reward})

        label = Path(info["query_path"]).name.split("@")[9]
        if info["action"] == "CAPTURE":
            best = Path(info["top_paths"][0]).name.split("@")[9]
            print(f"{label}: CAPTURE | top tile={best} | positive rank={info['positive_rank']} | score={info['top_scores'][0]:.3f}")
        else:
            print(f"{label}: SKIP")

        if done:
            break

    captures = [row for row in history if row["action"] == "CAPTURE"]
    successes = sum(row["positive_rank"] is not None for row in captures)
    output_path = Path("argus-agentic-gnc/outputs/real_earthloc_agent_history.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(history, indent=2) + "\n")

    print("\n=== Real EarthLoc Agent Result ===")
    print("Images captured:", len(captures), "/", len(history))
    print("Top-5 positive retrievals:", successes, "/", len(captures))
    print("Total reward:", round(total_reward, 3))
    print("History:", output_path)


if __name__ == "__main__":
    main()
