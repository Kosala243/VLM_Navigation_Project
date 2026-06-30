from __future__ import annotations

import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from navigation_pipeline import ModelWrapper, NavigationSystem
from navigation_pipeline.action_generator import Action


class DummyExecutor:
    """Safe executor: prints action only, no robot movement."""

    def execute(self, action: Action) -> bool:
        print("[DUMMY EXECUTOR]", action.display())
        return True


app = FastAPI(title="VLM Navigation Server")

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
FRAME_DIR = Path(os.getenv("FRAME_DIR", "live_robot_frames"))
FRAME_DIR.mkdir(exist_ok=True)

model: Optional[ModelWrapper] = None
nav: Optional[NavigationSystem] = None
current_goal: Optional[str] = None


@app.on_event("startup")
def startup() -> None:
    global model, nav
    print("[SERVER] Loading model...")
    model = ModelWrapper(MODEL_NAME).load()
    nav = NavigationSystem(model, executor=DummyExecutor())
    print("[SERVER] Ready.")


@app.post("/start")
def start_navigation(goal: str = Form(...), keep_memory: bool = Form(False)):
    global current_goal, nav

    if nav is None:
        return JSONResponse({"ok": False, "error": "Navigation system not loaded."}, status_code=500)

    current_goal = goal
    parsed = nav.start(goal, keep_memory=keep_memory)

    return {
        "ok": True,
        "goal": parsed.to_dict(),
        "message": f"Started navigation for goal: {goal}",
    }


@app.post("/step")
async def step_navigation(image: UploadFile = File(...)):
    global nav, current_goal

    if nav is None:
        return JSONResponse({"ok": False, "error": "Navigation system not loaded."}, status_code=500)

    if current_goal is None:
        return JSONResponse({"ok": False, "error": "Call /start first."}, status_code=400)

    frame_path = FRAME_DIR / "current_frame.jpg"

    with frame_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)

    action, done = nav.step(str(frame_path), execute=True)

    return {
        "ok": True,
        "done": done,
        "action": asdict(action),
        "status": nav.current_status(),
    }


@app.get("/status")
def status():
    if nav is None:
        return {"ok": False, "status": "not loaded"}

    return {
        "ok": True,
        "goal": current_goal,
        "status": nav.current_status(),
    }
