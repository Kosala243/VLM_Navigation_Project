from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from navigation_pipeline import ModelWrapper, NavigationSystem, GoalParser


app = FastAPI(title="VLM Navigation Pipeline API")

MODEL = None
NAV: NavigationSystem | None = None
CURRENT_GOAL = os.getenv("NAV_GOAL", "B0.004")

LIVE_DIR = Path("live_frames")
OUTPUT_DIR = Path("navigation_outputs/http_api")
LIVE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y"}


def get_model():
    global MODEL
    if MODEL is None:
        model_name = os.getenv("LLAMA_MODEL_ID", os.getenv("MODEL_NAME", "llm"))
        print(f"[API] Loading model wrapper: {model_name}")
        MODEL = ModelWrapper(model_name).load()
    return MODEL


def new_navigation_system(goal: str, keep_memory: bool = False) -> NavigationSystem:
    model = get_model()
    nav = NavigationSystem(model)

    if env_bool("USE_RULE_GOAL_PARSER", True):
        nav.goal_parser = GoalParser(None)
        print("[API] Using rule-based goal parser for speed.")

    nav.start(goal, keep_memory=keep_memory)
    return nav


async def save_uploaded_image(image: UploadFile) -> Path:
    timestamp = int(time.time() * 1000)
    suffix = Path(image.filename or "frame.jpg").suffix or ".jpg"

    saved_path = LIVE_DIR / f"frame_{timestamp}{suffix}"
    current_path = LIVE_DIR / "current_frame.jpg"

    with saved_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)

    shutil.copyfile(saved_path, current_path)
    return current_path


def action_response(
    *,
    goal: str,
    image_path: Path,
    action,
    done: bool,
    nav: NavigationSystem,
    mode: str,
) -> dict[str, Any]:
    result = {
        "mode": mode,
        "goal": goal,
        "image_path": str(image_path),
        "done": done,
        "action": asdict(action),
        "action_display": action.display(),
        "status": nav.current_status(),
    }

    output_path = OUTPUT_DIR / f"{mode}_latest_result.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    result["saved_result"] = str(output_path)
    return result


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_backend": os.getenv("MODEL_BACKEND", ""),
        "llama_server_url": os.getenv("LLAMA_SERVER_URL", ""),
        "goal": CURRENT_GOAL,
    }


@app.post("/single_step")
async def single_step(
    image: UploadFile = File(...),
    goal: str = Form(default=None),
):
    """
    Single-frame test:
    - receives one image
    - starts fresh memory
    - runs one pipeline step
    - does NOT move robot
    """
    use_goal = goal or CURRENT_GOAL
    image_path = await save_uploaded_image(image)

    nav = new_navigation_system(use_goal, keep_memory=False)
    action, done = nav.step(str(image_path), execute=False)

    return JSONResponse(
        action_response(
            goal=use_goal,
            image_path=image_path,
            action=action,
            done=done,
            nav=nav,
            mode="single_step",
        )
    )


@app.post("/autonomous/start")
def autonomous_start(goal: str = Form(default=None)):
    """
    Starts/reset a memory-preserving autonomous session.
    Movement is still disabled; this only keeps memory across frames.
    """
    global NAV, CURRENT_GOAL

    CURRENT_GOAL = goal or CURRENT_GOAL
    NAV = new_navigation_system(CURRENT_GOAL, keep_memory=False)

    return {
        "status": "started",
        "goal": CURRENT_GOAL,
        "movement_enabled": False,
    }


@app.post("/autonomous/step")
async def autonomous_step(
    image: UploadFile = File(...),
):
    """
    Autonomous loop step:
    - receives one image
    - keeps memory from previous autonomous steps
    - returns next high-level action
    - does NOT move robot
    """
    global NAV

    if NAV is None:
        NAV = new_navigation_system(CURRENT_GOAL, keep_memory=False)

    image_path = await save_uploaded_image(image)
    action, done = NAV.step(str(image_path), execute=False)

    return JSONResponse(
        action_response(
            goal=CURRENT_GOAL,
            image_path=image_path,
            action=action,
            done=done,
            nav=NAV,
            mode="autonomous_step",
        )
    )