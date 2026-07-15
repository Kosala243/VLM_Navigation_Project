"""memory.py
Structured navigation memory bank for generalized indoor navigation.

The memory stores navigation-useful evidence only: signs, door labels,
directories/maps, stairs/elevators, reachable frontiers, reception/help desks,
room-number trends, and target-relevant observations.

It deliberately avoids treating random people as navigation landmarks. The robot
may store/ask only official help sources such as reception, information desk,
front desk, or security desk.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .model_loader import ModelWrapper
    from .goal_parser import NavigationGoal


@dataclass
class Landmark:
    id: str
    category: str  # sign | door | frontier | reception | directory | stairs | elevator | observation | junction
    description: str
    text: str = ""
    pose: dict[str, Any] = field(default_factory=dict)
    status: str = "unvisited"  # unvisited | visited | used | ignored
    confidence: str = "medium"
    evidence_score: float = 0.0
    evidence_breakdown: dict[str, float] = field(default_factory=dict)
    image_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    # Number of separate observations in which this landmark was detected.
    observation_count: int = 1
    # Number of times the planner selected an action supported by this landmark.
    selection_count: int = 0

@dataclass
class MemoryUpdate:
    useful: bool
    summary: str = ""
    landmarks: list[Landmark] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)


class NavigationMemory:
    """Structured memory bank updated from every robot camera image."""

    VALID_CATEGORIES = {
        "sign",
        "door",
        "frontier",
        "reception",
        "directory",
        "stairs",
        "elevator",
        "observation",
        "junction",
    }

    _PROMPT = """\
        You are updating a robot's structured navigation memory from the current camera image.
        The robot must navigate to this goal:
        {goal_context}

        The robot may provide visual input in one of these formats:

        FORMAT A — SINGLE FRONT CAMERA IMAGE:
        - The image is a normal single camera view.
        - Treat it as the robot's front-facing view.

        FORMAT B — STITCHED MULTI-CAMERA IMAGE:
        - The image contains three side-by-side panels.
        - LEFT panel = robot's left camera view.
        - FRONT panel = robot's front camera view.
        - RIGHT panel = robot's right camera view.
        - Use the panel labels to decide direction.
        - Do not treat the stitched image as one continuous real scene.

        FORMAT C — THREE SEPARATE CAMERA IMAGES:
        - The model may receive three separate images.
        - They are ordered or labelled as LEFT, FRONT, RIGHT.
        - LEFT image shows what is on the robot's left side.
        - FRONT image shows what is directly ahead.
        - RIGHT image shows what is on the robot's right side.

        Navigation interpretation rules:
        - If the target room/sign/landmark is visible in the FRONT view, prefer moving forward or stopping to verify.
        - If the target room/sign/landmark is visible in the LEFT view, prefer turning left.
        - If the target room/sign/landmark is visible in the RIGHT view, prefer turning right.
        - If useful navigation cues are visible only on one side, mention which side.
        - If no target or useful cue is visible, choose SEARCH_FOR_CUE.
        - Do not guess room numbers or signs that are not clearly visible.
        - Be careful with blurry text, reflections, glass, and overexposed regions.
        - If a signboard, room-range sign, directory, or door label is visible but the text is too small/blurry/unclear to read, still create a landmark for it.
        - Set "text" to an empty string if the exact text is unreadable.
        - Mention in the description that the cue is visible but unreadable, and include whether it is in LEFT, FRONT, or RIGHT view.
        - Such landmarks are useful because the robot can move closer to read them.
        - Prefer current visual evidence over old memory.
        - For every landmark, set extra.source_view to LEFT, FRONT, RIGHT, STITCHED_UNKNOWN, or NONE.
        - Use LEFT when the landmark/cue is visible in the left image/panel.
        - Use FRONT when the landmark/cue is visible in the front image/panel.
        - Use RIGHT when the landmark/cue is visible in the right image/panel.
        - Use STITCHED_UNKNOWN when the stitched panel/source is unclear.
        - Use NONE only when no current visual cue supports the landmark.

        Extract ONLY navigation-useful evidence. Ignore furniture, ceiling, wall colour, general room appearance, and random people unless they are part of an official help desk.

        Look for:
        - directional signs, arrows, room ranges, building/zone/floor signs
        - door labels and room plates
        - directories, maps, "you are here" boards
        - elevators, stairs, floor indicators
        - corridor junctions and reachable unexplored paths/frontiers
        - reception desks, information desks, front desks, security desks, or official help counters
        - visible evidence that confirms/rejects the target goal

        Important rule about people:
        - Do NOT create a landmark for students, visitors, pedestrians, or random people in corridors/classrooms/labs.
        - Only create category "reception" when there is clearly an official reception/front-desk/information/security/help-desk context.

        Return ONLY valid JSON:
        {
        "useful": true/false,
        "summary": "one short note, or empty string",
        "landmarks": [
            {
            "category": "sign | door | frontier | reception | directory | stairs | elevator | observation | junction",
            "description": "what it is and where it is in view",
            "text": "exact readable text if any, else empty",
            "confidence": "high | medium | low",
            "pose": {},
            "extra": {
                "direction": null,
                "room_range": null,
                "arrow": null,
                "target_relevance": "high | medium | low | none",
                "floor": null,
                "zone": null,
                "source_view": "LEFT | FRONT | RIGHT | STITCHED_UNKNOWN | NONE"
            }
            }
        ],
        "hypotheses": ["short useful inference, e.g. room numbers increase forward"]
        }

        If no useful navigation evidence is visible, return exactly:
        {"useful": false, "summary": "", "landmarks": [], "hypotheses": []}
        """

    def __init__(
        self,
        model: "ModelWrapper",
        max_landmarks: int = 120,
        max_summaries: int = 80,
        max_hypotheses: int = 40,
        max_failed_actions: int = 20,
    ):
        self.model = model
        self.max_landmarks = max_landmarks
        self.max_summaries = max_summaries
        self.max_hypotheses = max_hypotheses
        self.max_failed_actions = max_failed_actions

        self.landmarks: list[Landmark] = []
        self.observation_summaries: list[str] = []
        self.hypotheses: list[str] = []
        self.failed_actions: list[str] = []
        self.image_count = 0
        self._next_id = 1

    def update_from_image(
        self,
        image_path: str,
        goal: "NavigationGoal",
        image_paths: dict[str, str] | None = None,
    ) -> MemoryUpdate:
        self.image_count += 1
        if not Path(image_path).exists():
            return MemoryUpdate(False, f"Image not found: {image_path}")

        prompt = self._PROMPT.replace("{goal_context}", goal.compact())
        response = self.model.query(prompt, image_path=image_path, image_paths=image_paths, max_new_tokens=800)
        data = _extract_json(response)
        if not data:
            return MemoryUpdate(False, "Could not parse memory JSON from model.")

        useful = bool(data.get("useful", False))
        summary = str(data.get("summary", "")).strip()
        new_landmarks: list[Landmark] = []

        for raw_lm in data.get("landmarks", []) or []:
            landmark = self._build_landmark(raw_lm, image_path, goal)
            if landmark is None:
                continue
            existing = self._find_duplicate_landmark(landmark)
            if existing is not None:
                existing.description = landmark.description
                existing.text = landmark.text or existing.text
                existing.image_path = landmark.image_path
                existing.confidence = landmark.confidence
                existing.evidence_score = landmark.evidence_score
                existing.evidence_breakdown = landmark.evidence_breakdown
                existing.extra.update(landmark.extra)
                existing.observation_count += 1
                continue
            new_landmarks.append(landmark)

        new_hypotheses = [
            str(h).strip()
            for h in (data.get("hypotheses", []) or [])
            if str(h).strip()
        ]

        if useful or new_landmarks or summary or new_hypotheses:
            useful = True
            if summary:
                self.observation_summaries.append(f"Image {self.image_count}: {summary}")
            self.landmarks.extend(new_landmarks)
            for h in new_hypotheses:
                self._add_hypothesis(h)
            self._add_room_sequence_hypotheses()
            self._trim_memory()

        return MemoryUpdate(
            useful=useful,
            summary=summary,
            landmarks=new_landmarks,
            hypotheses=new_hypotheses,
        )

    def add_frontiers(self, frontiers: list[dict[str, Any]]) -> None:
        """Optional hook for ROS/Gazebo frontier detector.

        Each frontier dict may contain: description, pose, confidence, score,
        information_gain, distance_m, source.
        """
        for frontier in frontiers:
            confidence = str(frontier.get("confidence", "medium")).lower()
            if confidence not in {"high", "medium", "low"}:
                confidence = "medium"
            extra = {
                k: v
                for k, v in frontier.items()
                if k not in {"description", "pose", "confidence"}
            }
            evidence_score, evidence_breakdown = _score_landmark(
                category="frontier",
                text="",
                confidence=confidence,
                extra=extra,
            )
            landmark = Landmark(
                id=self._new_id(),
                category="frontier",
                description=str(frontier.get("description", "reachable unexplored path/frontier")),
                pose=frontier.get("pose", {}) if isinstance(frontier.get("pose", {}), dict) else {},
                confidence=_confidence_from_score(evidence_score),
                evidence_score=evidence_score,
                evidence_breakdown=evidence_breakdown,
                extra=extra,
            )
            if self._find_duplicate_landmark(landmark) is None:
                self.landmarks.append(landmark)
        self._trim_memory()

    def mark_landmark(self, landmark_id: str, status: str) -> None:
        valid_status = {"unvisited", "visited", "used", "ignored"}
        if status not in valid_status:
            return

        for lm in self.landmarks:
            if lm.id == landmark_id:
                lm.status = status
                lm.selection_count += 1
                return

    def add_failed_action(self, reason: str) -> None:
        if reason:
            self.failed_actions.append(reason)
            self.failed_actions = self.failed_actions[-self.max_failed_actions:]

    def context_for_planner(self, n_recent: int = 8, n_relevant: int = 8) -> str:
        """Return compact structured context for the action planner.

        The planner gets recent landmarks plus target-relevant landmarks. This is
        better than sending only the last few observations, because old signs or
        door labels can still be important.
        """
        recent = self.landmarks[-n_recent:]
        relevant = self._target_relevant_landmarks(n_relevant)

        # Avoid repeating the same landmark if it is both recent and relevant.
        seen_ids = {lm.id for lm in recent}
        relevant = [lm for lm in relevant if lm.id not in seen_ids]

        data = {
            "recent_observations": self.observation_summaries[-n_recent:],
            "recent_landmarks": [asdict(lm) for lm in recent],
            "target_relevant_landmarks": [asdict(lm) for lm in relevant],
            "current_beliefs": self.hypotheses[-10:],
            "failed_actions": self.failed_actions[-5:],
            "memory_limits": {
                "stored_landmarks": len(self.landmarks),
                "max_landmarks": self.max_landmarks,
                "planner_recent_landmarks": n_recent,
                "planner_relevant_landmarks": n_relevant,
            },
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def save(self, path: str) -> None:
        data = {
            "image_count": self.image_count,
            "landmarks": [asdict(lm) for lm in self.landmarks],
            "observation_summaries": self.observation_summaries,
            "hypotheses": self.hypotheses,
            "failed_actions": self.failed_actions,
            "limits": {
                "max_landmarks": self.max_landmarks,
                "max_summaries": self.max_summaries,
                "max_hypotheses": self.max_hypotheses,
                "max_failed_actions": self.max_failed_actions,
            },
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def clear(self) -> None:
        self.landmarks.clear()
        self.observation_summaries.clear()
        self.hypotheses.clear()
        self.failed_actions.clear()
        self.image_count = 0
        self._next_id = 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_id(self) -> str:
        landmark_id = f"L{self._next_id:03d}"
        self._next_id += 1
        return landmark_id

    def _build_landmark(self, raw_lm: dict[str, Any], image_path: str, goal: "NavigationGoal") -> Landmark | None:
        category = str(raw_lm.get("category", "observation")).strip().lower()
        description = str(raw_lm.get("description", "")).strip()
        text = str(raw_lm.get("text", "")).strip()
        confidence = str(raw_lm.get("confidence", "medium")).strip().lower()
        extra = raw_lm.get("extra", {}) if isinstance(raw_lm.get("extra", {}), dict) else {}
        extra = dict(extra)
        extra["source_image_index"] = self.image_count
        pose = raw_lm.get("pose", {}) if isinstance(raw_lm.get("pose", {}), dict) else {}

        category = self._normalise_category(category, description, text)
        if category is None:
            return None

        extra = _correct_target_relevance(category, description, text, extra, goal)

        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"

        evidence_score, evidence_breakdown = _score_landmark(
            category=category,
            text=text,
            confidence=confidence,
            extra=extra,
        )

        return Landmark(
            id=self._new_id(),
            category=category,
            description=description,
            text=text,
            pose=pose,
            confidence=_confidence_from_score(evidence_score),
            evidence_score=evidence_score,
            evidence_breakdown=evidence_breakdown,
            image_path=str(image_path),
            extra=extra,
        )

    def _normalise_category(self, category: str, description: str, text: str) -> str | None:
        combined = f"{category} {description} {text}".lower()

        category_map = {
            "map": "directory",
            "you_are_here": "directory",
            "helpdesk": "reception",
            "front_desk": "reception",
            "information_desk": "reception",
            "security": "reception",
            "staff": "reception",
            "room_label": "door",
            "door_label": "door",
            "room_plate": "door",
            "corridor": "frontier",
            "path": "frontier",
            "intersection": "junction",
        }
        category = category_map.get(category, category)

        # Do not store generic people. Only official help sources are allowed.
        if category in {"person", "people", "human"}:
            if _looks_like_official_help_source(combined):
                return "reception"
            return None

        if category == "reception" and not _looks_like_official_help_source(combined):
            # Be conservative: reception/front desk must be explicit.
            return None

        if category not in self.VALID_CATEGORIES:
            return "observation"
        return category

    def _find_duplicate_landmark(self, candidate: Landmark) -> Landmark | None:
        candidate_key = _landmark_key(candidate)
        for existing in reversed(self.landmarks[-30:]):
            if _landmark_key(existing) == candidate_key:
                return existing
        return None

    def _add_hypothesis(self, hypothesis: str) -> None:
        if hypothesis and hypothesis not in self.hypotheses:
            self.hypotheses.append(hypothesis)

    def _add_room_sequence_hypotheses(self) -> None:
        door_numbers = []
        for lm in self.landmarks:
            if lm.category != "door":
                continue
            number = _extract_last_number(lm.text or lm.description)
            if number is not None:
                door_numbers.append(number)

        if len(door_numbers) < 2:
            return

        recent = door_numbers[-5:]
        if len(recent) < 2:
            return

        diffs = [b - a for a, b in zip(recent, recent[1:])]
        if all(d > 0 for d in diffs):
            self._add_hypothesis("Recently observed door numbers are increasing along the travelled direction.")
        elif all(d < 0 for d in diffs):
            self._add_hypothesis("Recently observed door numbers are decreasing along the travelled direction.")

    def _target_relevant_landmarks(self, limit: int) -> list[Landmark]:
        relevant: list[Landmark] = []
        for lm in reversed(self.landmarks):
            relevance = str(lm.extra.get("target_relevance", "")).lower()
            if relevance in {"high", "medium"}:
                relevant.append(lm)
            elif lm.category in {"directory", "sign", "door", "reception"} and lm.text:
                relevant.append(lm)
            if len(relevant) >= limit:
                break
        return list(reversed(relevant))

    def _trim_memory(self) -> None:
        self.landmarks = self.landmarks[-self.max_landmarks:]
        self.observation_summaries = self.observation_summaries[-self.max_summaries:]
        self.hypotheses = self.hypotheses[-self.max_hypotheses:]
        self.failed_actions = self.failed_actions[-self.max_failed_actions:]


def _score_landmark(
    category: str,
    text: str,
    confidence: str,
    extra: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Compute robot-side evidence quality for a memory landmark."""
    category_scores = {
        "door": 0.80,
        "sign": 0.70,
        "directory": 0.65,
        "elevator": 0.60,
        "stairs": 0.60,
        "reception": 0.55,
        "junction": 0.45,
        "frontier": 0.35,
        "observation": 0.25,
    }
    confidence_scores = {"high": 0.10, "medium": 0.05, "low": 0.00}

    base = category_scores.get(category, 0.25)
    confidence_bonus = confidence_scores.get(str(confidence).lower(), 0.05)
    text_bonus = 0.10 if str(text).strip() else 0.0

    target_relevance = str(extra.get("target_relevance", "")).lower() if isinstance(extra, dict) else ""
    target_bonus = 0.10 if target_relevance == "high" else 0.05 if target_relevance == "medium" else 0.0
    direction_bonus = 0.05 if isinstance(extra, dict) and (extra.get("direction") or extra.get("arrow")) else 0.0
    partial_marker_penalty = 0.35 if isinstance(extra, dict) and extra.get("partial_goal_marker") else 0.0
    irrelevant_sign_penalty = _irrelevant_sign_penalty(category, text, extra)

    if partial_marker_penalty and target_bonus > 0.05:
        target_bonus = 0.05

    missing_text_penalty = 0.0
    if category in {"door", "sign", "directory"} and not str(text).strip():
        missing_text_penalty = 0.35

    score = base + confidence_bonus + text_bonus + target_bonus + direction_bonus - missing_text_penalty - partial_marker_penalty - irrelevant_sign_penalty
    score = max(0.0, min(1.0, score))

    breakdown = {
        "category_score": base,
        "confidence_bonus": confidence_bonus,
        "text_bonus": text_bonus,
        "target_relevance_bonus": target_bonus,
        "direction_bonus": direction_bonus,
        "missing_text_penalty": missing_text_penalty,
        "partial_marker_penalty": partial_marker_penalty,
        "irrelevant_sign_penalty": irrelevant_sign_penalty,
        "final_score": score,
    }
    return score, breakdown

def _looks_like_room_or_range(text: str) -> bool:
    t = str(text)
    return bool(
        re.search(r"\b[A-Za-z]+\d+[.\-_]?\d+[A-Za-z]*\b", t)
        or re.search(r"\b\d+[.\-_]\d+[A-Za-z]*\b", t)
        or re.search(r"\broom\s+[A-Za-z]*\d+[A-Za-z]*\b", t, re.I)
        or re.search(r"\d+\s*[-–]\s*\d+", t)
    )

def _irrelevant_sign_penalty(category: str, text: str, extra: dict[str, Any]) -> float:
    if category != "sign":
        return 0.0

    t = str(text).lower()
    target_relevance = str(extra.get("target_relevance", "")).lower() if isinstance(extra, dict) else ""
    room_range = extra.get("room_range") if isinstance(extra, dict) else None
    zone = extra.get("zone") if isinstance(extra, dict) else None

    # Hard irrelevant signs: penalize even if the VLM wrongly says target_relevance=medium.
    hard_irrelevant_words = [
        "exit",
        "lost and found",
        "lecture may be recorded",
        "recorded for educational purposes",
        "promotional purposes",
        "self tour",
        "keep this door closed",
        "please use the other door",
        "thank you",
    ]

    if any(word in t for word in hard_irrelevant_words):
        return 0.45

    # Real navigation signs should stay high.
    if room_range or zone:
        return 0.0

    if _looks_like_room_or_range(text):
        return 0.0

    if any(word in t for word in ["directory", "map", "you are here", "reception", "information desk"]):
        return 0.0

    if target_relevance == "high":
        return 0.0

    if target_relevance == "medium":
        return 0.10

    return 0.20

def _confidence_from_score(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"



def _looks_like_official_help_source(text: str) -> bool:
    keywords = [
        "reception",
        "front desk",
        "front-desk",
        "information desk",
        "info desk",
        "help desk",
        "helpdesk",
        "security desk",
        "security",
        "staff desk",
        "service desk",
        "concierge",
        "counter",
    ]
    return any(k in text for k in keywords)


def _extract_last_number(text: str) -> int | None:
    matches = re.findall(r"\d+", text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def _landmark_key(landmark: Landmark) -> tuple[str, str, str, str]:
    text = re.sub(r"\s+", " ", landmark.text.lower()).strip()
    desc = re.sub(r"\s+", " ", landmark.description.lower()).strip()[:80]

    pose = getattr(landmark, "pose", {}) or {}
    pose_key = ""
    if isinstance(pose, dict) and "x" in pose and "y" in pose:
        try:
            pose_key = f"{round(float(pose['x']), 1)}_{round(float(pose['y']), 1)}"
        except Exception:
            pose_key = ""

    return landmark.category, text, desc, pose_key


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

def _normalise_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _is_partial_goal_marker(text: str, goal: "NavigationGoal") -> bool:
    constraints = getattr(goal, "constraints", {}) or {}
    building = str(constraints.get("possible_building") or "").strip()
    room = str(constraints.get("possible_room") or "").strip()
    raw_goal = str(getattr(goal, "raw_goal", "")).strip()

    if not building or not room:
        return False

    text_raw = str(text)
    text_norm = _normalise_label(text_raw)
    raw_norm = _normalise_label(raw_goal)

    if raw_norm and raw_norm in text_norm:
        return False

    has_building = bool(re.search(rf"\b{re.escape(building)}\b", text_raw, re.I))
    has_room = room and room in text_raw

    return has_building and not has_room


def _correct_target_relevance(
    category: str,
    description: str,
    text: str,
    extra: dict[str, Any],
    goal: "NavigationGoal",
) -> dict[str, Any]:
    combined = f"{description} {text}"

    if _is_partial_goal_marker(combined, goal):
        extra["partial_goal_marker"] = True
        extra["full_target_match"] = False

        if str(extra.get("target_relevance", "")).lower() == "high":
            extra["target_relevance"] = "medium"

    return extra