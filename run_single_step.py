from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from navigation_pipeline import ModelWrapper, NavigationSystem, GoalParser


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y"}


def main() -> None:
    goal = os.getenv("NAV_GOAL", "B0.004")
    image_path = os.getenv("LIVE_IMAGE_PATH", "live_frames/current_frame.jpg")
    output_dir = Path(os.getenv("SINGLE_STEP_OUTPUT_DIR", "navigation_outputs/live_single_step"))
    keep_memory = env_bool("KEEP_MEMORY", False)
    use_rule_goal_parser = env_bool("USE_RULE_GOAL_PARSER", True)

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {Path(image_path).resolve()}")

    model_name = os.getenv("LLAMA_MODEL_ID", os.getenv("MODEL_NAME", "llm"))

    print(f"[SingleStep] Goal: {goal}")
    print(f"[SingleStep] Image: {image_path}")
    print(f"[SingleStep] Model: {model_name}")
    print("[SingleStep] Robot movement: disabled")

    model = ModelWrapper(model_name).load()
    nav = NavigationSystem(model)

    # Faster for live testing: use rule-based goal parsing instead of asking VLM to parse B0.004.
    if use_rule_goal_parser:
        nav.goal_parser = GoalParser(None)
        print("[SingleStep] Using rule-based goal parser for speed.")

    nav.start(goal, keep_memory=keep_memory)

    action, done = nav.step(image_path, execute=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "goal": goal,
        "image_path": image_path,
        "done": done,
        "action": asdict(action),
        "status": nav.current_status(),
    }

    output_path = output_dir / "single_step_result.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("[SingleStep] Action:")
    print(action.display())
    print(f"[SingleStep] Done: {done}")
    print(f"[SingleStep] Saved: {output_path}")


if __name__ == "__main__":
    main()