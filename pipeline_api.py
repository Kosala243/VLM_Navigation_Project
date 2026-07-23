from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
EXTERNAL_EXECUTION_ENABLED = False


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


async def save_upload(upload: UploadFile, label: str) -> Path:
    timestamp = int(time.time() * 1000)
    suffix = Path(upload.filename or f"{label}.png").suffix or ".png"

    saved_path = LIVE_DIR / f"frame_{timestamp}_{label}{suffix}"

    with saved_path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)

    return saved_path


async def prepare_observation(
    *,
    image: Optional[UploadFile],
    left_image: Optional[UploadFile],
    front_image: Optional[UploadFile],
    right_image: Optional[UploadFile],
) -> tuple[Path, dict[str, str] | None, str]:
    """
    Returns:
        primary_image_path, image_paths_or_None, observation_mode

    observation_mode:
        single_or_stitched = one uploaded image
        separate = LEFT/FRONT/RIGHT uploaded separately
    """
    has_single = image is not None
    has_separate = left_image is not None or front_image is not None or right_image is not None

    if has_separate:
        if left_image is None or front_image is None or right_image is None:
            raise HTTPException(
                status_code=400,
                detail="For separate mode, provide left_image, front_image, and right_image.",
            )

        left_path = await save_upload(left_image, "left")
        front_path = await save_upload(front_image, "front")
        right_path = await save_upload(right_image, "right")

        current_path = LIVE_DIR / "current_frame.png"
        shutil.copyfile(front_path, current_path)

        return (
            front_path,
            {
                "LEFT": str(left_path),
                "FRONT": str(front_path),
                "RIGHT": str(right_path),
            },
            "separate",
        )

    if has_single:
        saved_path = await save_upload(image, "image")
        current_path = LIVE_DIR / "current_frame.png"
        shutil.copyfile(saved_path, current_path)
        return current_path, None, "single_or_stitched"

    raise HTTPException(
        status_code=400,
        detail="Provide either image OR left_image + front_image + right_image.",
    )


def action_response(
    *,
    goal: str,
    image_path: Path,
    image_paths: dict[str, str] | None,
    observation_mode: str,
    action,
    done: bool,
    nav: NavigationSystem,
    mode: str,
) -> dict[str, Any]:
    last_record = (
        nav.records[-1]
        if nav.records
        else None
    )
    result = {
        "mode": mode,
        "goal": goal,
        "observation_mode": observation_mode,
        "image_path": str(image_path),
        "image_paths": image_paths,
        "done": done,
        "action": asdict(action),
        "action_display": action.display(),
        "status": nav.current_status(),
        "action_step": (last_record.image_num if last_record else None),
        "perception_ok": (last_record.memory_update.parse_ok if last_record else None),
        "perception_error": (last_record.memory_update.error if last_record else None),
        "execution_ack_required": (nav.pending_step_number is not None and last_record is not None and nav.pending_step_number == last_record.image_num),
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
    image: Optional[UploadFile] = File(default=None),
    left_image: Optional[UploadFile] = File(default=None),
    front_image: Optional[UploadFile] = File(default=None),
    right_image: Optional[UploadFile] = File(default=None),
    goal: str = Form(default=None),
):
    """
    Single-frame test:
    - accepts either one image OR three separate images
    - starts fresh memory
    - runs one pipeline step
    - does NOT move robot
    """
    use_goal = goal or CURRENT_GOAL

    primary_path, image_paths, observation_mode = await prepare_observation(
        image=image,
        left_image=left_image,
        front_image=front_image,
        right_image=right_image,
    )

    nav = new_navigation_system(use_goal, keep_memory=False)
    action, done = nav.step(
        str(primary_path),
        image_paths=image_paths,
        execute=False,
    )

    return JSONResponse(
        action_response(
            goal=use_goal,
            image_path=primary_path,
            image_paths=image_paths,
            observation_mode=observation_mode,
            action=action,
            done=done,
            nav=nav,
            mode="single_step",
        )
    )


@app.post("/autonomous/start")
def autonomous_start(goal: str = Form(default=None), execution_enabled: bool = Form(default=False)):
    """
    Starts/reset a memory-preserving autonomous session.
    Movement is still disabled on AMD; ThinkPad handles movement separately.
    """
    global NAV, CURRENT_GOAL, EXTERNAL_EXECUTION_ENABLED

    CURRENT_GOAL = goal or CURRENT_GOAL
    EXTERNAL_EXECUTION_ENABLED = bool(execution_enabled)
    NAV = new_navigation_system(CURRENT_GOAL, keep_memory=False)

    return {
        "status": "started",
        "goal": CURRENT_GOAL,
        "movement_enabled": False,
        "external_execution_enabled": EXTERNAL_EXECUTION_ENABLED,
    }

@app.post("/autonomous/step")
async def autonomous_step(
    image: Optional[UploadFile] = File(default=None),
    left_image: Optional[UploadFile] = File(default=None),
    front_image: Optional[UploadFile] = File(default=None),
    right_image: Optional[UploadFile] = File(default=None),
):
    """
    Autonomous loop step:
    - accepts either one image OR three separate images
    - keeps memory from previous autonomous steps
    - returns next high-level action
    - does NOT move robot on AMD
    """
    global NAV

    if NAV is None:
        NAV = new_navigation_system(CURRENT_GOAL, keep_memory=False)

    primary_path, image_paths, observation_mode = await prepare_observation(
        image=image,
        left_image=left_image,
        front_image=front_image,
        right_image=right_image,
    )

    action, done = NAV.step(
        str(primary_path),
        image_paths=image_paths,
        execute=False,
        expect_external_execution=EXTERNAL_EXECUTION_ENABLED,
    )

    return JSONResponse(
        action_response(
            goal=CURRENT_GOAL,
            image_path=primary_path,
            image_paths=image_paths,
            observation_mode=observation_mode,
            action=action,
            done=done,
            nav=NAV,
            mode="autonomous_step",
        )
    )

@app.post("/autonomous/ack")
def autonomous_ack(
    step_number: int = Form(...),
    executed: bool = Form(...),
    status: str = Form(default=""),
    reason: str = Form(default=""),
):
    global NAV

    if NAV is None:
        raise HTTPException(
            status_code=400,
            detail="No active autonomous session.",
        )

    acknowledgement = (
        NAV.acknowledge_execution(
            step_number=step_number,
            executed=executed,
            reason=(
                reason
                or status
            ),
        )
    )

    if not acknowledgement.get(
        "accepted",
        False,
    ):
        raise HTTPException(
            status_code=409,
            detail=acknowledgement,
        )

    return acknowledgement

@app.get("/autonomous/export")
def autonomous_export():
    """
    Export the current live autonomous session:
    - parsed goal;
    - full cumulative navigation memory;
    - current perception status;
    - complete step-by-step navigation records.
    """
    global NAV

    if NAV is None or NAV.goal is None:
        raise HTTPException(
            status_code=400,
            detail="No active autonomous session.",
        )

    last_record = (
        NAV.records[-1]
        if NAV.records
        else None
    )

    latest_perception = {
        "parse_ok": (
            last_record.memory_update.parse_ok
            if last_record
            else None
        ),
        "error": (
            last_record.memory_update.error
            if last_record
            else ""
        ),
        "parse_attempts": (
            last_record.memory_update.parse_attempts
            if last_record
            else 0
        ),
        "raw_response": (
            last_record.memory_update.raw_response
            if last_record
            else ""
        ),
    }

    memory_data = {
        "image_count": NAV.memory.image_count,
        "landmarks": [
            asdict(landmark)
            for landmark in NAV.memory.landmarks
        ],
        "observation_summaries": list(
            NAV.memory.observation_summaries
        ),
        "hypotheses": list(
            NAV.memory.hypotheses
        ),
        "failed_actions": list(
            NAV.memory.failed_actions
        ),
        "latest_perception": latest_perception,
    }

    navigation_records = []

    for record in NAV.records:
        navigation_records.append({
            "image_num": record.image_num,
            "image_path": record.image_path,
            "memory_update": {
                "useful": (
                    record.memory_update.useful
                ),
                "summary": (
                    record.memory_update.summary
                ),
                "landmarks": [
                    asdict(landmark)
                    for landmark
                    in record.memory_update.landmarks
                ],
                "hypotheses": list(
                    record.memory_update.hypotheses
                ),
                "parse_ok": (
                    record.memory_update.parse_ok
                ),
                "error": (
                    record.memory_update.error
                ),
                "parse_attempts": (
                    record.memory_update.parse_attempts
                ),
                "raw_response": (
                    record.memory_update.raw_response
                ),
            },
            "verification": (
                record.verification.to_dict()
                if record.verification
                else None
            ),
            "action": asdict(record.action),
            "executed": record.executed,
        })

    navigation_log_data = {
        "goal": NAV.goal.to_dict(),
        "status": NAV.current_status(),
        "total_steps": len(NAV.records),
        "records": navigation_records,
    }

    return {
        "goal": NAV.goal.to_dict(),
        "memory": memory_data,
        "navigation_log": navigation_log_data,
    }