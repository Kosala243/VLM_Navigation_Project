"""model_loader.py
Qwen3-VL wrapper used by the generalized navigation pipeline.

Two backends are supported:
1) transformers          -> full precision / HF model, as before
2) llama_cpp_server      -> GGUF model served by llama.cpp OpenAI-compatible server

For GGUF/Q8 on the AMD server, start llama-server separately, then run:
    export MODEL_BACKEND=llama_cpp_server
    export LLAMA_SERVER_URL=http://127.0.0.1:8080/v1
    export LLAMA_MODEL_ID=Qwen3-VL-8B-Instruct-GGUF-Q8_0
    python run_navigation.py
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Optional

warnings.filterwarnings("default")


class ModelWrapper:
    """Small deterministic wrapper around Qwen3-VL style models.

    Keep the public API stable for the rest of the navigation pipeline:
        - load()
        - query(prompt, image_path=None, max_new_tokens=...)

    This lets navigator/memory/verifier/action_generator stay unchanged.
    """

    LLAMA_SERVER_BACKENDS = {"llama_cpp_server", "llama-server", "gguf_server", "gguf"}

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        backend: str | None = None,
        llama_server_url: str | None = None,
        llama_model_id: str | None = None,
    ):
        self.model_name = model_name
        self.backend = (backend or os.getenv("MODEL_BACKEND", "transformers")).strip().lower()

        # Transformers backend state
        self.model = None
        self.processor = None
        self._process_vision_info = None

        # llama.cpp server backend state
        self.llama_server_url = (llama_server_url or os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080/v1")).rstrip("/")
        self.llama_model_id = llama_model_id or os.getenv("LLAMA_MODEL_ID", model_name)
        self.api_key = os.getenv("LLAMA_SERVER_API_KEY", "")
        self.request_timeout_s = float(os.getenv("MODEL_TIMEOUT_S", "600"))
        self.temperature = float(os.getenv("MODEL_TEMPERATURE", "0.0"))
        self.top_p = float(os.getenv("MODEL_TOP_P", "1.0"))

        self._loaded = False
        self.last_response = ""

    def load(self) -> "ModelWrapper":
        if self._loaded:
            print(f"[ModelWrapper] Already loaded: {self.model_name}")
            return self

        if self.backend in self.LLAMA_SERVER_BACKENDS:
            return self._load_llama_cpp_server()

        return self._load_transformers()

    # ------------------------------------------------------------------
    # GGUF backend via llama.cpp OpenAI-compatible server
    # ------------------------------------------------------------------

    def _load_llama_cpp_server(self) -> "ModelWrapper":
        print("[ModelWrapper] Using GGUF backend: llama.cpp server")
        print(f"  Server: {self.llama_server_url}")
        print(f"  Model id sent to server: {self.llama_model_id}")

        # Check that the server is reachable. If /models is unavailable but the
        # server is running, we continue; if connection is refused, fail early.
        try:
            data = self._http_json("GET", f"{self.llama_server_url}/models")
            models = data.get("data", []) if isinstance(data, dict) else []
            if models and not os.getenv("LLAMA_MODEL_ID"):
                first_id = str(models[0].get("id", "")).strip()
                if first_id:
                    self.llama_model_id = first_id
                    print(f"  Auto-detected server model id: {self.llama_model_id}")
        except urllib.error.HTTPError as exc:
            print(f"  Warning: /models returned HTTP {exc.code}; continuing anyway.")
        except Exception as exc:
            raise RuntimeError(
                "Could not reach llama.cpp server. Start llama-server first, then run the pipeline. "
                f"Server URL: {self.llama_server_url}. Error: {exc}"
            ) from exc

        self._loaded = True
        return self

    def _collect_image_inputs(
        self,
        image_path: str | None = None,
        image_paths: dict[str, str] | None = None,
    ) -> tuple[list[tuple[str, Path]], str | None]:
        images: list[tuple[str, Path]] = []

        if image_paths:
            ordered_keys = [
                ("LEFT", ["LEFT", "left"]),
                ("FRONT", ["FRONT", "front"]),
                ("RIGHT", ["RIGHT", "right"]),
            ]

            for label, keys in ordered_keys:
                raw_path = None
                for key in keys:
                    if key in image_paths:
                        raw_path = image_paths[key]
                        break

                if raw_path:
                    images.append((label, Path(raw_path)))

        elif image_path is not None:
            images.append(("IMAGE", Path(image_path)))

        if not images:
            return [], None

        for label, path in images:
            if not path.exists():
                return [], f"[ERROR] Image not found for {label}: {path}"

        return images, None

    def _query_llama_cpp_server(
        self,
        prompt: str,
        image_path: str | None = None,
        image_paths: dict[str, str] | None = None,
        max_new_tokens: int = 600,
    ) -> str:
        content: list[dict[str, Any]] = []

        images, error = self._collect_image_inputs(
            image_path=image_path,
            image_paths=image_paths,
        )
        if error:
            return error

        for label, path in images:
            if label != "IMAGE":
                content.append({
                    "type": "text",
                    "text": f"{label} camera image:",
                })

            content.append({
                "type": "image_url",
                "image_url": {"url": self._image_to_data_url(path)},
            })

        content.append({"type": "text", "text": prompt})

        payload = {
            "model": self.llama_model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(max_new_tokens),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }

        try:
            data = self._http_json("POST", f"{self.llama_server_url}/chat/completions", payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            return f"[ERROR] llama.cpp server HTTP {exc.code}: {detail}"
        except Exception as exc:
            return f"[ERROR] llama.cpp server request failed: {exc}"

        try:
            output = data["choices"][0]["message"]["content"]
        except Exception:
            output = json.dumps(data, ensure_ascii=False)

        output = str(output).strip()
        self.last_response = output
        return output

    def _http_json(self, method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(request, timeout=self.request_timeout_s) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    @staticmethod
    def _image_to_data_url(path: Path) -> str:
        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            mime = "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    # ------------------------------------------------------------------
    # Original Transformers backend
    # ------------------------------------------------------------------

    def _load_transformers(self) -> "ModelWrapper":
        print(f"[ModelWrapper] Loading with Transformers: {self.model_name}")
        try:
            from qwen_vl_utils import process_vision_info
            self._process_vision_info = process_vision_info
            print("  qwen-vl-utils found")
        except Exception:
            self._process_vision_info = None
            print("  qwen-vl-utils not found; using PIL fallback")

        import torch

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

        self.model.eval()

        if hasattr(self.model, "generation_config"):
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None

        self._loaded = True

        device = next(self.model.parameters()).device
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Loaded via: {loaded_via}")
        print(f"  Device: {device}")
        print(f"  Parameters: ~{n_params / 1e9:.1f}B")
        return self

    def _input_device(self):
        """Return a safe device for Transformers input tensors."""
        import torch

        if self.model is None:
            return torch.device("cpu")

        for p in self.model.parameters():
            if p.device.type != "meta":
                return p.device

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _query_transformers(
        self,
        prompt: str,
        image_path: str | None = None,
        image_paths: dict[str, str] | None = None,
        max_new_tokens: int = 600,
    ) -> str:
        import torch

        images, error = self._collect_image_inputs(
            image_path=image_path,
            image_paths=image_paths,
        )
        if error:
            return error

        for _, path in images:
            try:
                from PIL import Image
                Image.open(path).verify()
            except Exception as exc:
                return f"[ERROR] Cannot open image {path}: {exc}"

        content = []

        for label, path in images:
            if label != "IMAGE":
                content.append({"type": "text", "text": f"{label} camera image:"})
            content.append({"type": "image", "image": str(path)})

        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)
        except Exception as exc:
            return f"[ERROR] Input processing failed: {exc}"

        inputs = inputs.to(self._input_device())

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }

        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)

        if pad_token_id is None and eos_token_id is not None:
            generation_kwargs["pad_token_id"] = eos_token_id
        elif pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]

        output = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        self.last_response = output
        return output

    # ------------------------------------------------------------------
    # Public query API used by the rest of the pipeline
    # ------------------------------------------------------------------

    def query(
        self,
        prompt: str,
        image_path: str | None = None,
        image_paths: dict[str, str] | None = None,
        max_new_tokens: int = 600,
    ) -> str:
        if not self._loaded:
            raise RuntimeError("Call ModelWrapper.load() before query().")

        if self.backend in self.LLAMA_SERVER_BACKENDS:
            return self._query_llama_cpp_server(
                prompt,
                image_path=image_path,
                image_paths=image_paths,
                max_new_tokens=max_new_tokens,
            )

        return self._query_transformers(
            prompt,
            image_path=image_path,
            image_paths=image_paths,
            max_new_tokens=max_new_tokens,
        )