from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os
import re

from navigation_pipeline import ModelWrapper, NavigationSystem


def natural_key(path):
    name = Path(path).stem
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_goal_name(goal):
    """Matches thinkpad_robot_capture_to_amd.py's safe_goal_name so replay
    output folders use the same naming convention as live runs."""
    return (
        str(goal)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


def _find_camera_image(step_folder: Path, camera: str) -> str | None:
    for ext in ("png", "jpg", "jpeg"):
        candidate = step_folder / f"{camera}.{ext}"
        if candidate.exists():
            return str(candidate)
    return None


def _build_separate_camera_sequence(image_dir: Path):
    """Read a robot_live_frames-style folder of per-step subfolders, each
    containing front/left/right images (matching the live pipeline's
    camera_mode="separate"), into navigate()'s image_sequence +
    image_paths_sequence pair."""
    step_folders = sorted(
        (p for p in image_dir.iterdir() if p.is_dir()),
        key=lambda p: natural_key(str(p)),
    )
    if not step_folders:
        raise RuntimeError(
            f"No step subfolders found in {image_dir.resolve()}"
        )

    image_sequence = []
    image_paths_sequence = []
    for step_folder in step_folders:
        front = _find_camera_image(step_folder, "front")
        left = _find_camera_image(step_folder, "left")
        right = _find_camera_image(step_folder, "right")
        if front is None:
            print(f"[SKIP] {step_folder}: no front.png/.jpg found")
            continue
        image_sequence.append(front)
        image_paths_sequence.append({
            key: value
            for key, value in (
                ("LEFT", left),
                ("FRONT", front),
                ("RIGHT", right),
            )
            if value is not None
        })

    if not image_sequence:
        raise RuntimeError(
            f"No usable step subfolders (with at least front.png/.jpg) "
            f"found in {image_dir.resolve()}"
        )

    return image_sequence, image_paths_sequence


def main():
    # For full precision/HF testing, keep default MODEL_BACKEND=transformers.
    # For GGUF/Q8 testing, run llama-server first and set MODEL_BACKEND=llama_cpp_server.
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
    model = ModelWrapper(model_name).load()
    nav = NavigationSystem(model)

    goal = os.getenv("NAV_GOAL", "C0.004")
    image_dir = Path(os.getenv("IMAGE_DIR", "robot_live_frames/step_test_C0.004"))
    keep_memory = _env_bool("KEEP_MEMORY", False)
    max_images_env = os.getenv("MAX_IMAGES", "").strip()
    # "separate" (default, matches IMAGE_DIR's default robot_live_frames
    # layout): image_dir is a folder of per-step subfolders, each with
    # front/left/right images. "single": image_dir is a flat folder of one
    # image per step (the original behavior, for a flat IMAGE_DIR like
    # images/seq2).
    camera_mode = os.getenv("CAMERA_MODE", "separate").strip().lower()

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir.resolve()}")

    if camera_mode == "separate":
        image_sequence, image_paths_sequence = _build_separate_camera_sequence(
            image_dir
        )
    else:
        image_sequence = sorted(
            [str(p) for ext in ("*.jpg", "*.jpeg", "*.png") for p in image_dir.glob(ext)],
            key=natural_key,
        )
        image_paths_sequence = None

        if not image_sequence:
            raise RuntimeError(f"No image files found in {image_dir.resolve()}")

    if max_images_env:
        limit = int(max_images_env)
        image_sequence = image_sequence[:limit]
        if image_paths_sequence is not None:
            image_paths_sequence = image_paths_sequence[:limit]

    for img in image_sequence[:10]:
        print(img)

    print(f"Found {len(image_sequence)} images")
    print(f"Camera mode: {camera_mode}")
    print(f"Goal: {goal}")
    print(f"Image dir: {image_dir}")

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    # Deliberately NOT navigation_outputs/live_runs/: that's the directory
    # scripts/aggregate_success_rates.py scans by default, grouping rows by
    # goal alone (no mode filter). Writing offline replay runs in there
    # would silently blend them into the same success-rate bucket as real
    # hardware runs for the same goal. Same run_<timestamp>_<goal> naming
    # and step_XXX_result.json/memory.json/navigation_log.json contents as
    # a live run, just under a sibling tree -- pass --live-runs-dir
    # explicitly to compare/aggregate scripts when you want to look at
    # these runs together.
    save_dir = os.getenv(
        "SAVE_DIR",
        str(
            Path("navigation_outputs/offline_replays")
            / f"run_{timestamp}_{safe_goal_name(goal)}"
        ),
    )

    log = nav.navigate(
        raw_goal=goal,
        image_sequence=image_sequence,
        image_paths_sequence=image_paths_sequence,
        save_dir=save_dir,
        keep_memory=keep_memory,
        save_step_results=True,
        run_settings={
            "goal": goal,
            "mode": "offline_replay",
            "camera_mode": camera_mode,
            "memory": keep_memory,
            "execute": "replay",
            "model_name": model_name,
            "model_backend": model.backend,
        },
    )
    print(f"Navigation success: {log.success}")


if __name__ == "__main__":
    main()
