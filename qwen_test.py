#source .venv/bin/activate

import base64
import requests

MODEL = "qwen3-vl:2b"
IMAGE_PATH = "images/img5_seq2.jpg"
instruction = "The goal is to reach stairs"

with open(IMAGE_PATH, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": MODEL,
    "prompt": (
        f"Instruction: {instruction}\n"
        "Reply with exactly one action label only: "
        "go_straight, turn_left, turn_right, or stop."
    ),
    "images": [image_b64],
    "stream": False
}

r = requests.post("http://localhost:11434/api/generate", json=payload)

print("status:", r.status_code)
data = r.json()
# print("full json:", data)

if "error" in data:
    raise RuntimeError(data["error"])

print("model output:", data["response"])