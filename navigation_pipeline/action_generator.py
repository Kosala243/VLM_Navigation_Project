"""action_generator.py
High-level action generator for generalized indoor navigation.

The VLM selects skill primitives, not low-level robot control.
This version matches the generalized goal parser and updated memory design:
- no random-person interaction;
- official help only through reception/front-desk/security/help-counter evidence;
- prefer current/local evidence over old used evidence;
- validate directions, vertical movement, and stopping against recent memory.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .model_loader import ModelWrapper
    from .goal_parser import NavigationGoal
    from .memory import Landmark, NavigationMemory


@dataclass
class Action:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: str = "low"
    evidence_score: float = 0.0
    evidence_breakdown: dict[str, float] = field(default_factory=dict)
    goal_reached: bool = False
    needs_verification: bool = False
    raw: dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    invalid_reason: str = ""

    def display(self) -> str:
        status = "valid" if self.is_valid else f"INVALID: {self.invalid_reason}"
        return f"{self.name}({self.params}) [{self.confidence}, score={self.evidence_score:.2f}] {status} - {self.reason}"


class ActionGenerator:
    """Generate and validate the next high-level navigation skill."""

    VALID_ACTIONS = {
        "READ_SIGN": "Approach/read a visible sign, directory, map, room-range sign, or arrow sign.",
        "CHECK_DOOR_LABEL": "Approach/read a visible door plate or room label.",
        "NAVIGATE_TO_LANDMARK": "Navigate to a known landmark id from memory.",
        "NAVIGATE_TO_FRONTIER": "Move to an unexplored reachable path/frontier.",
        "FOLLOW_DIRECTION": (
            "Follow a direction from current/local evidence such as a sign, directory, "
            "room-range sign, or official staff/reception instruction."
        ),
        "ASK_RECEPTION_OR_STAFF": (
            "Ask only a clearly visible official help source: reception, front desk, "
            "information desk, security desk, or staff/help counter."
        ),
        "USE_ELEVATOR_OR_STAIRS": "Use lift/stairs when recent evidence says a floor transition is needed.",
        "SEARCH_FOR_CUE": "Search for a useful cue: sign, directory, room label, elevator, stairs, target, reception/help desk.",
        "WAIT_OR_RECOVER": "Use when input is unclear, blocked, unsafe, or the previous action failed.",
    }

    _PROMPT = """\
You are the high-level planner for a mobile robot navigating an unknown office/university/hospital/airport/public building.
You must select ONE skill primitive. Do NOT output low-level wheel velocities.

Goal:
{goal_context}

Structured memory:
{memory_context}

Current image: provided separately.

Visual input note:
- The current image may be a normal single front-camera image, a stitched LEFT/FRONT/RIGHT image, or three separate LEFT/FRONT/RIGHT images.
- If stitched, treat the panels as separate camera views, not one continuous scene.
- Use LEFT evidence for left-side actions, FRONT evidence for forward/stop/check actions, and RIGHT evidence for right-side actions.
- If choosing a direction, include the evidence view in params as "evidence_view": "LEFT | FRONT | RIGHT | NONE".

Available actions:
{action_list}

Decision rules:
- Prefer reading useful signs/directories/maps over blind exploration.
- If a visible door label may confirm the target, an alias, or a room-number pattern, use CHECK_DOOR_LABEL.
- If a current/local sign, directory, room-range sign, or official reception/staff memory gives a direction, use FOLLOW_DIRECTION or NAVIGATE_TO_LANDMARK.
- Prefer the newest relevant local evidence over old evidence whose status is already "used" or "visited".
- Do not ask random people, students, visitors, or people in corridors/classrooms/labs.
- Use ASK_RECEPTION_OR_STAFF only when a reception/front desk/information desk/security desk/help counter is clearly visible.
- If no useful cue is visible, use NAVIGATE_TO_FRONTIER or SEARCH_FOR_CUE.
- If a signboard, room-range sign, directory, or door label is visible but the text is too small, blurry, overexposed, or unreadable, do NOT guess the text or direction signs(arrow marks).
- In that case, choose an action that moves closer to the visible cue.
- If the unclear cue is in the FRONT view, choose READ_SIGN, CHECK_DOOR_LABEL, or FOLLOW_DIRECTION with direction="forward".
- If the unclear cue is in the LEFT view, choose FOLLOW_DIRECTION with direction="left" to face/approach it.
- If the unclear cue is in the RIGHT view, choose FOLLOW_DIRECTION with direction="right" to face/approach it.
- In the reason, explicitly say that the text is not clearly readable and moving closer may help decide the direction.
- Always include "direction" and "evidence_view" in params when approaching an unclear sign or label.
- Use USE_ELEVATOR_OR_STAIRS only when recent evidence shows a lift/stairs and the goal/floor evidence suggests a floor transition.
- Never assume a room-code structure is true until signs/labels/directories confirm it.
- Do not repeatedly read/check a landmark whose status is already "used" or "visited"; choose a newer cue, frontier, or search action instead.
- Treat building/zone-only markers such as "B" for goal "B0.004" as navigation cues, not strong target evidence.
- FOLLOW_DIRECTION must include landmark_id of the sign/directory/reception/stairs/elevator evidence that supports the direction.
- For debugging and evaluation, every action must report where the strongest current visual evidence came from.
- Include "evidence_view" inside params.
- evidence_view must be one of: "LEFT", "FRONT", "RIGHT", "STITCHED_UNKNOWN", or "NONE".
- Use "LEFT" if the strongest cue is in the left camera/panel.
- Use "FRONT" if the strongest cue is in the front camera/panel.
- Use "RIGHT" if the strongest cue is in the right camera/panel.
- Use "STITCHED_UNKNOWN" if the image is stitched but the panel/source is unclear.
- Use "NONE" if no useful visual cue is visible.

Return ONLY valid JSON:
{
  "action": "READ_SIGN | CHECK_DOOR_LABEL | NAVIGATE_TO_LANDMARK | NAVIGATE_TO_FRONTIER | FOLLOW_DIRECTION | ASK_RECEPTION_OR_STAFF | USE_ELEVATOR_OR_STAIRS | SEARCH_FOR_CUE | WAIT_OR_RECOVER",
  "params": {
    "landmark_id": null,
    "direction": null,
    "target": null,
    "floor": null,
    "search_for": null,
    "evidence_view": "LEFT | FRONT | RIGHT | STITCHED_UNKNOWN | NONE"
  },
  "reason": "one sentence",
  "confidence": "high | medium | low",
  "goal_reached": true/false,
  "needs_verification": true/false
}
"""

    def __init__(self, model: "ModelWrapper"):
        self.model = model

    def generate(
        self,
        image_path: str,
        goal: "NavigationGoal",
        memory: "NavigationMemory",
        image_paths: dict[str, str] | None = None,
    ) -> Action:
        prompt = (
            self._PROMPT
            .replace("{goal_context}", goal.compact())
            .replace("{memory_context}", memory.context_for_planner())
            .replace(
                "{action_list}",
                "\n".join(f"- {k}: {v}" for k, v in self.VALID_ACTIONS.items()),
            )
        )
        response = self.model.query(prompt, image_path=image_path, image_paths=image_paths, max_new_tokens=500)
        data = _extract_json(response) or {}

        raw_action_name = str(data.get("action", "WAIT_OR_RECOVER")).strip().upper()
        # Backward compatibility: if an older prompt/model returns ASK_PERSON, map it to
        # the stricter action, then validation will allow it only with official-help evidence.
        if raw_action_name == "ASK_PERSON":
            raw_action_name = "ASK_RECEPTION_OR_STAFF"

        action = Action(
            name=raw_action_name,
            params=data.get("params", {}) if isinstance(data.get("params", {}), dict) else {},
            reason=str(data.get("reason", "No parseable reason returned.")),
            confidence=str(data.get("confidence", "low")).lower(),
            goal_reached=bool(data.get("goal_reached", False)),
            needs_verification=bool(data.get("needs_verification", False)),
            raw=data,
        )
        validated = self._validate(action, memory, goal)
        self._attach_action_evidence_score(validated, memory, goal)
        return validated

    def _validate(self, action: Action, memory: "NavigationMemory", goal: "NavigationGoal") -> Action:
        """Block unsafe or hallucinated high-level actions before execution."""
        if action.confidence not in {"high", "medium", "low"}:
            action.confidence = "low"
        
        if action.name == "STOP_AND_VERIFY":
            action.is_valid = False
            action.goal_reached = False
            action.invalid_reason = (
                "STOP_AND_VERIFY must come from TargetVerifier using current-frame "
                "target door/entrance evidence, not from ActionGenerator."
            )
            return action

        if action.name not in self.VALID_ACTIONS:
            action.is_valid = False
            action.invalid_reason = f"Unknown action: {action.name}"
            return action

        recent = _recent_landmarks(memory, n=12)

        if action.name in {"READ_SIGN", "CHECK_DOOR_LABEL"}:
            lm_id = _clean_param(action.params.get("landmark_id"))

            if not lm_id:
                action.is_valid = False
                action.invalid_reason = f"{action.name} requires landmark_id."
                return action

            lm = _get_landmark(memory, lm_id)
            if lm is None:
                action.is_valid = False
                action.invalid_reason = f"Unknown landmark_id: {lm_id}"
                return action

            if str(getattr(lm, "status", "")).lower() in {"used", "visited"}:
                action.is_valid = False
                action.invalid_reason = f"Landmark {lm_id} was already {lm.status}; choose a newer cue."
                return action

        if action.name == "NAVIGATE_TO_LANDMARK":
            lm_id = _clean_param(action.params.get("landmark_id"))
            if not lm_id:
                action.is_valid = False
                action.invalid_reason = "NAVIGATE_TO_LANDMARK requires landmark_id."
                return action
            if not _landmark_exists(memory, lm_id):
                action.is_valid = False
                action.invalid_reason = f"Unknown landmark_id: {lm_id}"
                return action
            lm = _get_landmark(memory, lm_id)
            if lm is not None and str(getattr(lm, "status", "")).lower() in {"used", "visited"}:
                action.is_valid = False
                action.invalid_reason = f"Landmark {lm_id} was already {lm.status}; choose a newer cue."
                return action

        if action.name == "NAVIGATE_TO_FRONTIER":
            lm_id = _clean_param(action.params.get("landmark_id"))
            if lm_id and not _landmark_exists(memory, lm_id):
                action.is_valid = False
                action.invalid_reason = f"Unknown frontier landmark_id: {lm_id}"
                return action
            if lm_id and not _landmark_has_category(memory, lm_id, {"frontier"}):
                action.is_valid = False
                action.invalid_reason = f"Landmark {lm_id} is not a frontier."
                return action
            # If no ID is given, allow visual/local frontier selection; ROS/Gazebo can
            # still provide frontier candidates in memory via add_frontiers().

        if action.name == "ASK_RECEPTION_OR_STAFF":
            if not _has_official_help_evidence(recent):
                action.is_valid = False
                action.invalid_reason = (
                    "ASK_RECEPTION_OR_STAFF requires recent visible evidence of reception, "
                    "front desk, information desk, security desk, help desk, or staff counter."
                )
                return action

        if action.name == "FOLLOW_DIRECTION":
            direction = _clean_param(action.params.get("direction"))
            target = _clean_param(action.params.get("target"))
            lm_id = _clean_param(action.params.get("landmark_id"))

            if not (direction or target or lm_id):
                action.is_valid = False
                action.invalid_reason = "FOLLOW_DIRECTION requires direction, target, or landmark_id."
                return action
            
            # If VLM forgot landmark_id, attach the best recent supporting landmark.
            if not lm_id:
                supporting_lm = _find_best_direction_landmark(recent, direction)
                if supporting_lm is not None:
                    lm_id = str(getattr(supporting_lm, "id", ""))
                    action.params["landmark_id"] = lm_id

            if not lm_id:
                action.is_valid = False
                action.invalid_reason = "FOLLOW_DIRECTION requires landmark_id for robot execution."
                return action

            if not _landmark_exists(memory, lm_id):
                action.is_valid = False
                action.invalid_reason = f"Unknown direction landmark_id: {lm_id}"
                return action

            if not _has_recent_direction_evidence(recent, landmark_id=lm_id):
                action.is_valid = False
                action.invalid_reason = (
                    f"FOLLOW_DIRECTION landmark {lm_id} does not contain recent direction evidence."
                )
                return action

        if action.name == "USE_ELEVATOR_OR_STAIRS":
            if not _has_vertical_transition_evidence(recent):
                action.is_valid = False
                action.invalid_reason = "USE_ELEVATOR_OR_STAIRS requires recent elevator or stairs evidence."
                return action

        return action

    def _attach_action_evidence_score(self, action: Action, memory: "NavigationMemory", goal: "NavigationGoal") -> None:
        """Attach a deterministic evidence score to the chosen action.

        This keeps the VLM's action choice, but replaces vague confidence with
        a robot-side score based on action type, validity, and supporting memory.
        """
        action_base_scores = {
            "CHECK_DOOR_LABEL": 0.75,
            "READ_SIGN": 0.70,
            "FOLLOW_DIRECTION": 0.65,
            "NAVIGATE_TO_LANDMARK": 0.60,
            "USE_ELEVATOR_OR_STAIRS": 0.60,
            "ASK_RECEPTION_OR_STAFF": 0.55,
            "NAVIGATE_TO_FRONTIER": 0.40,
            "SEARCH_FOR_CUE": 0.35,
            "WAIT_OR_RECOVER": 0.20,
        }
        
        # To prevent invalid robot actions being medium/high confidence.
        if not action.is_valid:
            action.evidence_score = 0.0
            action.evidence_breakdown = {
                "invalid_action": 1.0,
                "final_score": 0.0,
            }
            action.confidence = "low"
            return

        base = action_base_scores.get(action.name, 0.20)
        lm_id = _clean_param(action.params.get("landmark_id"))

        landmark_score = 0.0
        landmark_bonus = 0.0
        if lm_id:
            lm = _get_landmark(memory, lm_id)
            if lm is not None:
                landmark_score = float(getattr(lm, "evidence_score", 0.0) or 0.0)
                landmark_bonus = min(0.15, landmark_score * 0.15)

        partial_marker_penalty = 0.0
        used_landmark_penalty = 0.0
        mismatched_room_code_penalty = 0.0

        if lm_id:
            lm = _get_landmark(memory, lm_id)
            if lm is not None:
                extra = getattr(lm, "extra", {}) if isinstance(getattr(lm, "extra", {}), dict) else {}

                if extra.get("partial_goal_marker"):
                    partial_marker_penalty = 0.25

                if str(getattr(lm, "status", "")).lower() in {"used", "visited"}:
                    used_landmark_penalty = 0.30
                
                mismatched_room_code_penalty = _room_code_mismatch_penalty(
                    lm=lm,
                    goal=goal,
                    action_name=action.name,
                )

        no_landmark_penalty = 0.0
        if action.name in {"FOLLOW_DIRECTION", "CHECK_DOOR_LABEL", "READ_SIGN", "NAVIGATE_TO_LANDMARK"} and not lm_id:
            no_landmark_penalty = 0.15

        invalid_penalty = 0.40 if not action.is_valid else 0.0

        score = base + landmark_bonus - no_landmark_penalty - invalid_penalty - partial_marker_penalty - used_landmark_penalty - mismatched_room_code_penalty
        score = max(0.0, min(1.0, score))

        action.evidence_score = score
        action.evidence_breakdown = {
            "action_base_score": base,
            "supporting_landmark_score": landmark_score,
            "supporting_landmark_bonus": landmark_bonus,
            "no_landmark_penalty": no_landmark_penalty,
            "partial_marker_penalty": partial_marker_penalty,
            "used_landmark_penalty": used_landmark_penalty,
            "invalid_penalty": invalid_penalty,
            "mismatched_room_code_penalty": mismatched_room_code_penalty,
            "final_score": score,
        }
        action.confidence = _confidence_from_score(score)


# ── Validation helpers ──────────────────────────────────────────────────────


def _recent_landmarks(memory: "NavigationMemory", n: int = 12) -> list["Landmark"]:
    return list(getattr(memory, "landmarks", [])[-n:])


def _landmark_exists(memory: "NavigationMemory", landmark_id: str) -> bool:
    return any(str(lm.id) == str(landmark_id) for lm in getattr(memory, "landmarks", []))


def _landmark_has_category(memory: "NavigationMemory", landmark_id: str, categories: set[str]) -> bool:
    cats = {c.lower() for c in categories}
    return any(
        str(lm.id) == str(landmark_id) and str(lm.category).lower() in cats
        for lm in getattr(memory, "landmarks", [])
    )


def _clean_param(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", ""} else text


def _landmark_text(lm: "Landmark") -> str:
    extra = getattr(lm, "extra", {})
    try:
        extra_text = json.dumps(extra, ensure_ascii=False)
    except TypeError:
        extra_text = str(extra)
    return " ".join([
        str(getattr(lm, "category", "")),
        str(getattr(lm, "description", "")),
        str(getattr(lm, "text", "")),
        extra_text,
    ]).lower()


def _has_official_help_evidence(landmarks: list["Landmark"]) -> bool:
    official_categories = {
        "reception",
        "front_desk",
        "information_desk",
        "security_desk",
        "staff_help",
        "help_desk",
    }
    official_words = [
        "reception",
        "front desk",
        "information desk",
        "security desk",
        "security",
        "help desk",
        "staff counter",
        "service desk",
        "information counter",
        "receptionist",
    ]
    for lm in landmarks:
        category = str(getattr(lm, "category", "")).lower()
        text = _landmark_text(lm)
        if category in official_categories or any(word in text for word in official_words):
            return True
    return False


def _has_recent_direction_evidence(landmarks: list["Landmark"], landmark_id: str = "") -> bool:
    direction_categories = {
        "sign",
        "directory",
        "reception",
        "front_desk",
        "information_desk",
        "security_desk",
        "staff_help",
        "stairs",
        "elevator",
        "observation",
    }
    direction_words = [
        "left",
        "right",
        "straight",
        "ahead",
        "forward",
        "back",
        "behind",
        "up",
        "down",
        "arrow",
        "turn",
        "north",
        "south",
        "east",
        "west",
        "←",
        "→",
        "↑",
        "↓",
    ]
    for lm in landmarks:
        if landmark_id and str(getattr(lm, "id", "")) != str(landmark_id):
            continue
        category = str(getattr(lm, "category", "")).lower()
        text = _landmark_text(lm)
        extra = getattr(lm, "extra", {}) if isinstance(getattr(lm, "extra", {}), dict) else {}
        has_extra_direction = any(k in extra and extra.get(k) not in (None, "", [], {}) for k in ("direction", "arrow"))
        if category in direction_categories and (has_extra_direction or any(word in text for word in direction_words)):
            return True
    return False

def _find_best_direction_landmark(
    landmarks: list["Landmark"],
    direction: str = "",
) -> "Landmark | None":
    """Find recent landmark that can justify a FOLLOW_DIRECTION action."""
    direction = direction.lower().strip()

    best = None
    best_score = -1.0

    for lm in reversed(landmarks):
        category = str(getattr(lm, "category", "")).lower()
        if category not in {
            "sign",
            "directory",
            "reception",
            "front_desk",
            "information_desk",
            "security_desk",
            "staff_help",
            "stairs",
            "elevator",
            "observation",
        }:
            continue

        text = _landmark_text(lm)
        extra = getattr(lm, "extra", {}) if isinstance(getattr(lm, "extra", {}), dict) else {}

        has_direction = any(
            extra.get(k) not in (None, "", [], {})
            for k in ("direction", "arrow")
        ) or any(
            word in text
            for word in ["left", "right", "straight", "forward", "ahead", "up", "down", "←", "→", "↑", "↓"]
        )

        if not has_direction:
            continue

        # Prefer landmarks whose text/extra matches requested direction.
        match_bonus = 0.0
        combined = f"{text} {extra}".lower()
        if direction and any(part in combined for part in direction.replace("and", " ").split()):
            match_bonus = 0.25

        score = float(getattr(lm, "evidence_score", 0.0) or 0.0) + match_bonus

        if score > best_score:
            best = lm
            best_score = score

    return best

def _has_vertical_transition_evidence(landmarks: list["Landmark"]) -> bool:
    vertical_words = ["elevator", "lift", "stairs", "staircase", "niveau", "level", "floor", "escalator"]
    for lm in landmarks:
        category = str(getattr(lm, "category", "")).lower()
        text = _landmark_text(lm)
        if category in {"elevator", "stairs"} or any(word in text for word in vertical_words):
            return True
    return False

def _get_landmark(memory: "NavigationMemory", landmark_id: str) -> "Landmark | None":
    for lm in getattr(memory, "landmarks", []):
        if str(getattr(lm, "id", "")) == str(landmark_id):
            return lm
    return None

def _confidence_from_score(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"

def _normalise_room_code(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _goal_room_code_norms(goal: "NavigationGoal") -> set[str]:
    labels = [
        getattr(goal, "raw_goal", ""),
        *getattr(goal, "known_tokens", []),
        *getattr(goal, "aliases", []),
    ]
    return {
        _normalise_room_code(label)
        for label in labels
        if _normalise_room_code(label)
    }


def _extract_room_codes(text: str) -> list[str]:
    text = str(text)
    codes = []

    patterns = [
        r"\b[A-Za-z]+\d+[.\-_]\d+[A-Za-z]*\b",   # C0.004, B0-004
        r"\b[A-Za-z]\d{3,5}[A-Za-z]?\b",         # C004, C0004
        r"\b\d+[.\-_]\d+[A-Za-z]*\b",            # 0.004, 3.14
    ]

    for pat in patterns:
        codes.extend(re.findall(pat, text))

    return _unique_room_codes(codes)


def _unique_room_codes(codes: list[str]) -> list[str]:
    out = []
    seen = set()
    for code in codes:
        key = _normalise_room_code(code)
        if key and key not in seen:
            out.append(code)
            seen.add(key)
    return out


def _room_code_prefix(code: str) -> str:
    norm = _normalise_room_code(code)
    m = re.match(r"([a-z]+)", norm)
    return m.group(1) if m else ""


def _room_code_mismatch_penalty(
    lm: "Landmark",
    goal: "NavigationGoal",
    action_name: str,
) -> float:
    """Penalize actions supported by visible room codes that conflict with the goal."""
    text = _landmark_text(lm)
    codes = _extract_room_codes(text)
    if not codes:
        return 0.0

    goal_codes = _goal_room_code_norms(goal)

    # If landmark contains the actual target/alias, do not penalize.
    if any(_normalise_room_code(code) in goal_codes for code in codes):
        return 0.0

    constraints = getattr(goal, "constraints", {}) or {}
    goal_building = str(constraints.get("possible_building") or "").strip().lower()

    code_prefixes = {
        _room_code_prefix(code)
        for code in codes
        if _room_code_prefix(code)
    }

    # Strong penalty: target is B0.004 but landmark is C0.004/C0.008.
    if goal_building and code_prefixes and goal_building not in code_prefixes:
        if action_name in {"CHECK_DOOR_LABEL", "NAVIGATE_TO_LANDMARK"}:
            return 0.55
        if action_name == "FOLLOW_DIRECTION":
            return 0.45
        return 0.35

    # Same building/zone but wrong specific room: useful as context, not strong target evidence.
    if action_name in {"CHECK_DOOR_LABEL", "NAVIGATE_TO_LANDMARK"}:
        return 0.35

    return 0.15

# JSON parsing

def _extract_json(text: str) -> dict[str, Any] | None:
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None