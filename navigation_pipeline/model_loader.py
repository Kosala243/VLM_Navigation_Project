"""model_loader.py
Qwen3-VL wrapper used by the generalized navigation pipeline.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import torch

warnings.filterwarnings("ignore")


class ModelWrapper:
    """Small deterministic wrapper around Qwen3-VL style HuggingFace models."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._process_vision_info = None
        self._loaded = False

    def load(self) -> "ModelWrapper":
        if self._loaded:
            print(f"[ModelWrapper] Already loaded: {self.model_name}")
            return self

        print(f"[ModelWrapper] Loading: {self.model_name}")
        try:
            from qwen_vl_utils import process_vision_info
            self._process_vision_info = process_vision_info
            print("  qwen-vl-utils found")
        except Exception:
            self._process_vision_info = None
            print("  qwen-vl-utils not found; using PIL fallback")

        last_error: Optional[Exception] = None
        loaded_via = None

        # Qwen3-VL support depends on transformers version; keep robust fallbacks.
        for class_name in (
            "Qwen3VLForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration",
            "AutoModelForVision2Seq",
        ):
            try:
                import transformers
                AutoProcessor = transformers.AutoProcessor
                ModelClass = getattr(transformers, class_name)
                self.model = ModelClass.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                loaded_via = class_name
                break
            except Exception as exc:
                last_error = exc

        if self.model is None or self.processor is None:
            raise RuntimeError(
                f"Could not load {self.model_name}. Install/upgrade transformers and qwen-vl-utils. "
                f"Last error: {last_error}"
            )

        self._loaded = True
        device = next(self.model.parameters()).device
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Loaded via: {loaded_via}")
        print(f"  Device: {device}")
        print(f"  Parameters: ~{n_params / 1e9:.1f}B")
        return self

    def query(self, prompt: str, image_path: str | None = None, max_new_tokens: int = 600) -> str:
        if not self._loaded:
            raise RuntimeError("Call ModelWrapper.load() before query().")

        if image_path is not None:
            p = Path(image_path)
            if not p.exists():
                return f"[ERROR] Image not found: {image_path}"
            try:
                from PIL import Image
                Image.open(p).verify()
            except Exception as exc:
                return f"[ERROR] Cannot open image: {exc}"

        content = []
        if image_path is not None:
            content.append({"type": "image", "image": str(image_path)})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        try:
            if image_path and self._process_vision_info is not None:
                image_inputs, video_inputs = self._process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs if image_inputs else None,
                    videos=video_inputs if video_inputs else None,
                    padding=True,
                    return_tensors="pt",
                )
            elif image_path:
                from PIL import Image
                img = Image.open(image_path).convert("RGB")
                inputs = self.processor(text=[text], images=[img], padding=True, return_tensors="pt")
            else:
                inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        except Exception as exc:
            return f"[ERROR] Input processing failed: {exc}"

        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
