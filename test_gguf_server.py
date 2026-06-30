from pathlib import Path
from navigation_pipeline import ModelWrapper

image = Path("images/seq2/img1_seq2.jpg")
model = ModelWrapper(backend="llama_cpp_server").load()

if image.exists():
    print(model.query("Describe this image in one sentence.", image_path=str(image), max_new_tokens=100))
else:
    print(model.query("Reply with exactly: server ok", max_new_tokens=20))
