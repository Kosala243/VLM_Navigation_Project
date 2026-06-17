from pathlib import Path
import re

from navigation_pipeline import ModelWrapper, NavigationSystem


# 1. Load Qwen3-VL-8B
model = ModelWrapper("Qwen/Qwen3-VL-8B-Instruct").load()

# 2. Create navigation system
nav = NavigationSystem(model)

# 3. Give target room/location
goal = "B0.004"

def natural_key(path):
    name = Path(path).stem
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]

# 4. Load image sequence
image_dir = Path("images/seq2")

if not image_dir.exists():
    raise FileNotFoundError(f"Image folder not found: {image_dir.resolve()}")

image_sequence = sorted(
    [str(p) for p in image_dir.glob("*.jpg")],
    key = natural_key
)

if not image_sequence:
    raise RuntimeError(f"No .jpg images found in {image_dir.resolve()}")

for img in image_sequence[:10]:
    print(img)

print(f"Found {len(image_sequence)} images")

# 5. Run navigation over image sequence
log = nav.navigate(
    raw_goal=goal,
    image_sequence=image_sequence,
    save_dir="navigation_outputs",
    keep_memory=False
)