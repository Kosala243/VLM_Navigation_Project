from pathlib import Path
import re

from navigation_pipeline import ModelWrapper, NavigationSystem

def natural_key(path):
    name = Path(path).stem
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]

def main():
    model = ModelWrapper("Qwen/Qwen3-VL-8B-Instruct").load() # Load Qwen3-VL-8B
    nav = NavigationSystem(model)

    goal = "B0.004" # Target room/location
    image_dir = Path("images/seq2")

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir.resolve()}")

    image_sequence = sorted(
        [str(p) for ext in ("*.jpg", "*.jpeg", "*.png") for p in image_dir.glob(ext)],
        key = natural_key
    )

    if not image_sequence:
        raise RuntimeError(f"No .jpg images found in {image_dir.resolve()}")

    for img in image_sequence[:10]:
        print(img)

    print(f"Found {len(image_sequence)} images")

    # Run navigation over image sequence
    log = nav.navigate(
        raw_goal=goal,
        image_sequence=image_sequence,
        save_dir=f"navigation_outputs/{image_dir.name}_{goal.replace('.', '_')}",
        keep_memory=False
    )
    print(f"Navigation success: {log.success}")

if __name__ == "__main__":
    main()