from pathlib import Path
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


def main():
    # For full precision/HF testing, keep default MODEL_BACKEND=transformers.
    # For GGUF/Q8 testing, run llama-server first and set MODEL_BACKEND=llama_cpp_server.
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
    model = ModelWrapper(model_name).load()
    nav = NavigationSystem(model)

    goal = os.getenv("NAV_GOAL", "B0.004")
    image_dir = Path(os.getenv("IMAGE_DIR", "images/seq2"))
    keep_memory = _env_bool("KEEP_MEMORY", False)
    max_images_env = os.getenv("MAX_IMAGES", "").strip()

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir.resolve()}")

    image_sequence = sorted(
        [str(p) for ext in ("*.jpg", "*.jpeg", "*.png") for p in image_dir.glob(ext)],
        key=natural_key,
    )

    if max_images_env:
        image_sequence = image_sequence[: int(max_images_env)]

    if not image_sequence:
        raise RuntimeError(f"No image files found in {image_dir.resolve()}")

    for img in image_sequence[:10]:
        print(img)

    print(f"Found {len(image_sequence)} images")
    print(f"Goal: {goal}")
    print(f"Image dir: {image_dir}")

    log = nav.navigate(
        raw_goal=goal,
        image_sequence=image_sequence,
        save_dir=f"navigation_outputs/{image_dir.name}_{goal.replace('.', '_')}",
        keep_memory=keep_memory,
    )
    print(f"Navigation success: {log.success}")


if __name__ == "__main__":
    main()
