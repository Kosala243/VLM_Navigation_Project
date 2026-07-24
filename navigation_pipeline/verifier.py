"""verifier.py
Target verification module for generalized indoor navigation.

This module is intentionally separate from action generation.
Its only job is to decide whether the robot has actually reached the target.

Key rule:
- A building/tower/zone marker such as a large "B" or "C" is NOT a target room.
- For room goals like B0.004, the verifier must see a current-frame door/room
  label matching B0.004 or an accepted full-label alias. A partial building
  letter match is rejected.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .model_loader import ModelWrapper
    from .goal_parser import NavigationGoal
    from .memory import Landmark, MemoryUpdate
    from .action_generator import Action


@dataclass
class VerificationResult:
    """Result returned by the target verifier."""

    target_visible: bool = False
    target_reached: bool = False
    matched_label: str = ""
    evidence_type: str = "none"  # target_door_label | target_entrance | directional_sign | directory | room_range | nearby_room | zone_marker | none | unclear
    confidence: str = "low"      # high | medium | low
    evidence_score: float = 0.0
    evidence_breakdown: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    landmark_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_action(self) -> "Action":
        """Convert a positive verification into a STOP_AND_VERIFY action."""
        from .action_generator import Action

        evidence_view = "NONE"
        raw_landmark = self.raw.get("landmark", {})
        if isinstance(raw_landmark, dict):
            extra = raw_landmark.get("extra", {})
            if isinstance(extra, dict):
                candidate = str(
                    extra.get("source_view", "")
                ).upper()

                if candidate in {
                    "LEFT",
                    "FRONT",
                    "RIGHT",
                    "STITCHED_UNKNOWN",
                }:
                    evidence_view = candidate

        return Action(
            name="STOP_AND_VERIFY",
            params={
                "landmark_id": self.landmark_id,
                "target": self.matched_label,
                "evidence_type": self.evidence_type,
                "evidence_view": evidence_view,
            },
            reason=self.reason or f"Target verified: {self.matched_label}",
            confidence=self.confidence,
            evidence_score=self.evidence_score,
            evidence_breakdown=self.evidence_breakdown,
            goal_reached=True,
            needs_verification=False,
            raw={"verification": self.to_dict()},
            is_valid=True,
        )


class TargetVerifier:
    """Verify whether the current image shows the actual target door/entrance."""

    STOP_EVIDENCE_TYPES = {"target_door_label", "target_entrance"}

    _PROMPT = """\
    You are the final target verifier for an indoor mobile robot.
    Your task is NOT to choose the next navigation action.
    Your task is ONLY to decide whether the robot has reached the actual target.

    Goal:
    {goal_context}

    Current-frame memory extracted from the same image:
    {current_memory_context}

    Current image: provided separately.

    Visual input note:
    - The current image may be a normal single front-camera image, a stitched LEFT/FRONT/RIGHT image, or three separate LEFT/FRONT/RIGHT images.
    - If stitched, treat the panels as separate camera views, not one continuous scene.
    - Verify target_reached=true only when the actual target door/entrance/room label is clearly visible in the current visual input.
    - If the target appears only in LEFT or RIGHT view, target_visible may be true, but target_reached should be false unless the robot is actually facing/reached the target entrance.

    Strict verification rules:
    - target_reached=true ONLY if the actual target door, room plate, entrance, office label, suite label, gate label, or target facility entrance is visible/readable in the CURRENT image.
    - For room-code goals such as B0.004 or C0.008, a single large building/tower/zone letter such as "B" or "C" is NOT the target room. It is only a navigation cue.
    - For room-code goals, the matched_label must be the full visible room label, e.g. "B0.004", "B0-004", "B0004", or an explicitly accepted full alias. Do not use only "B", "C", "0", or "004" as a reached target.
    - A directional sign pointing to the target is NOT reached.
    - A directory/map listing the target is NOT reached unless the actual target entrance/door is also visible.
    - A room range sign containing the target is NOT reached.
    - A nearby/adjacent room label is NOT reached.
    - Old memory from previous frames is NOT enough. Use only the current image and current-frame memory shown above.
    - If the text is too small/unclear, return target_reached=false with evidence_type="unclear".
    - FRONT visibility alone is not enough unless the exact full target door or entrance label is readable.
    - For an exact room-code goal, target_reached=true requires:
        1. the full exact target label or accepted full alias;
        2. evidence from the current frame;
        3. source_view=FRONT;
        4. horizontal_position=center;
        5. evidence_type=target_door_label or target_entrance.
    - proximity is advisory visual metadata. It must not block target verification once the exact full label is readable on the current centred FRONT target.
    - If the target is only in LEFT or RIGHT, or is off-centre in FRONT, return target_visible=true and target_reached=false.
    - LiDAR and robot-state safety determine whether additional physical forward movement is safe; the visual verifier does not require the robot to approach an already verified target door.

    Return ONLY valid JSON:
    {
    "target_visible": true/false,
    "target_reached": true/false,
    "matched_label": "exact visible label, or empty string",
    "evidence_type": "target_door_label | target_entrance | directional_sign | directory | room_range | nearby_room | zone_marker | none | unclear",
    "confidence": "high | medium | low",
    "reason": "one short sentence explaining the decision",
    "landmark_id": null
    }
    """

    def __init__(self, model: "ModelWrapper"):
        self.model = model

    def verify(
        self,
        image_path: str,
        goal: "NavigationGoal",
        memory_update: "MemoryUpdate | None" = None,
        image_paths: dict[str, str] | None = None,
    ) -> VerificationResult:
        """Verify whether the current image reaches the target.

        Only current-frame evidence can stop the robot. The deterministic memory
        check is conservative. The VLM verifier is then guarded with exact-label
        checks so it cannot stop on a building letter such as "B".
        """
        deterministic = self._verify_from_current_memory(goal, memory_update)
        if deterministic.target_reached:
            return deterministic

        prompt = (
            self._PROMPT
            .replace("{goal_context}", goal.compact())
            .replace("{current_memory_context}", self._current_memory_context(memory_update))
        )
        response = self.model.query(prompt, image_path=image_path, image_paths=image_paths, max_new_tokens=350)
        data = _extract_json(response)
        if not data:
            if deterministic.target_visible:
                return deterministic
            score, breakdown = _score_verification(
                evidence_type="unclear", 
                matched_label="",
                goal=goal,
                current_frame=True,
                is_stop_candidate=False,
                visual_confidence="low",
            )
            return VerificationResult(
                target_visible=False,
                target_reached=False,
                evidence_type="unclear",
                confidence=_confidence_from_score(score),
                evidence_score=score,
                evidence_breakdown=breakdown,
                reason="Verifier could not parse model response.",
                raw={"response": response},
            )

        result = VerificationResult(
            target_visible=bool(data.get("target_visible", False)),
            target_reached=bool(data.get("target_reached", False)),
            matched_label=str(data.get("matched_label", "")).strip(),
            evidence_type=str(data.get("evidence_type", "none")).strip().lower() or "none",
            confidence=_normalise_confidence(data.get("confidence", "low")),
            reason=str(data.get("reason", "")).strip(),
            landmark_id=_clean_optional_id(data.get("landmark_id")),
            raw=data,
        )
        guarded = self._apply_strict_guards(result, goal, memory_update)

        # If the VLM gave a false positive but deterministic current memory had
        # useful non-stop evidence, prefer the deterministic explanation.
        if not guarded.target_reached and deterministic.target_visible:
            return deterministic
        return guarded

    def _verify_from_current_memory(
        self,
        goal: "NavigationGoal",
        memory_update: "MemoryUpdate | None",
    ) -> VerificationResult:
        """Deterministically verify matching current-frame door/entrance landmarks."""
        labels = _stop_labels(goal)
        if not memory_update or not labels:
            return VerificationResult()

        best_non_stop = VerificationResult()

        for lm in getattr(memory_update, "landmarks", []) or []:
            category = str(getattr(lm, "category", "")).lower()
            combined = _landmark_combined_text(lm)
            matched = _find_stop_label(labels, combined)
            extra = getattr(lm, "extra", {}) if isinstance(getattr(lm, "extra", {}), dict) else {}
            source_view = _landmark_source_view(lm)
            target_relevance = str(extra.get("target_relevance", "")).lower()

            if matched and category == "door":
                score, breakdown = _score_verification(
                    evidence_type="target_door_label",
                    matched_label=matched,
                    goal=goal,
                    current_frame=True,
                    is_stop_candidate=True,
                    visual_confidence=getattr(lm, "confidence", "high"),
                )

                if _landmark_ready_for_stop(lm):
                    return VerificationResult(
                        target_visible=True,
                        target_reached=score >= 0.85,
                        matched_label=matched,
                        evidence_type="target_door_label",
                        confidence=_confidence_from_score(score),
                        evidence_score=score,
                        evidence_breakdown=breakdown,
                        reason=f"Target verified in current FRONT image: door label '{matched}' matches the goal.",
                        landmark_id=str(getattr(lm, "id", "")) or None,
                        raw={"source": "current_memory", "landmark": _safe_asdict(lm)},
                    )

                # Exact target is visible, but it is still a side-view cue.
                side_score = min(score, 0.80)
                breakdown["side_view_alignment_required"] = 1.0
                breakdown["final_score"] = side_score

                candidate = VerificationResult(
                    target_visible=True,
                    target_reached=False,
                    matched_label=matched,
                    evidence_type="target_door_label",
                    confidence=_confidence_from_score(side_score),
                    evidence_score=side_score,
                    evidence_breakdown=breakdown,
                    reason=_target_not_ready_reason(
                        lm,
                        matched,
                        "target door",
                    ),
                    landmark_id=str(
                        getattr(lm, "id", "")
                    ) or None,
                    raw={
                        "source": "current_memory_side_view",
                        "landmark": _safe_asdict(lm),
                    },
                )

                best_non_stop = _choose_better_non_stop(
                    best_non_stop,
                    candidate,
                )

                continue

            if (matched and category in {"sign", "observation"} and _looks_like_actual_entrance(combined)):
                score, breakdown = _score_verification(
                    evidence_type="target_entrance",
                    matched_label=matched,
                    goal=goal,
                    current_frame=True,
                    is_stop_candidate=True,
                    visual_confidence=getattr(lm, "confidence", "medium",),
                )
                if _landmark_ready_for_stop(lm):
                    return VerificationResult(
                        target_visible=True,
                        target_reached=score >= 0.85,
                        matched_label=matched,
                        evidence_type="target_entrance",
                        confidence=_confidence_from_score(score),
                        evidence_score=score,
                        evidence_breakdown=breakdown,
                        reason=(
                            f"Target verified in current FRONT image: "
                            f"entrance label '{matched}' matches the goal."
                        ),
                        landmark_id=str(
                            getattr(lm, "id", "")
                        ) or None,
                        raw={
                            "source": "current_memory",
                            "landmark": _safe_asdict(lm),
                        },
                    )

                side_score = min(score, 0.80)
                breakdown["side_view_alignment_required"] = 1.0
                breakdown["final_score"] = side_score

                candidate = VerificationResult(
                    target_visible=True,
                    target_reached=False,
                    matched_label=matched,
                    evidence_type="target_entrance",
                    confidence=_confidence_from_score(
                        side_score
                    ),
                    evidence_score=side_score,
                    evidence_breakdown=breakdown,
                    reason=(
                        f"Target entrance '{matched}' is visible in "
                        f"{source_view}, but the robot must align with "
                        "the entrance before confirming arrival."
                    ),
                    landmark_id=str(
                        getattr(lm, "id", "")
                    ) or None,
                    raw={
                        "source": "current_memory_side_view",
                        "landmark": _safe_asdict(lm),
                    },
                )
                best_non_stop = _choose_better_non_stop(
                    best_non_stop,
                    candidate,
                )
                continue

            # Current-frame target-related evidence that is useful but cannot stop.
            if matched and category in {"sign", "directory", "observation"}:
                e_type = _non_stop_evidence_type(category, combined)
                score, breakdown = _score_verification(
                    evidence_type=e_type,
                    matched_label=matched,
                    goal=goal,
                    current_frame=True,
                    is_stop_candidate=False,
                    visual_confidence=getattr(lm, "confidence", "medium"),
                )
                candidate = VerificationResult(
                    target_visible=True,
                    target_reached=False,
                    matched_label=matched,
                    evidence_type=e_type,
                    confidence=_confidence_from_score(score),
                    evidence_score=score,
                    evidence_breakdown=breakdown,
                    reason=(
                        f"Current image contains target-related "
                        f"{e_type.replace('_', ' ')} for '{matched}', "
                        "but not an actual target door/entrance."
                    ),
                    landmark_id=str(
                        getattr(lm, "id", "")
                    ) or None,
                    raw={
                        "source": "current_memory_non_stop",
                        "landmark": _safe_asdict(lm),
                    },
                )
                best_non_stop = _choose_better_non_stop(
                    best_non_stop,
                    candidate,
                )
            elif _looks_like_zone_marker_for_goal(goal, combined):
                matched_zone = _goal_building_or_zone(goal)
                score, breakdown = _score_verification(
                    evidence_type="zone_marker",
                    matched_label=matched_zone,
                    goal=goal,
                    current_frame=True,
                    is_stop_candidate=False,
                    visual_confidence=getattr(lm, "confidence", "medium"),
                )
                candidate = VerificationResult(
                    target_visible=True,
                    target_reached=False,
                    matched_label=matched_zone,
                    evidence_type="zone_marker",
                    confidence=_confidence_from_score(score),
                    evidence_score=score,
                    evidence_breakdown=breakdown,
                    reason=(
                        "Current image shows only the building/tower/"
                        "zone marker, not the target room label."
                    ),
                    landmark_id=str(
                        getattr(lm, "id", "")
                    ) or None,
                    raw={
                        "source": "current_memory_zone_marker",
                        "landmark": _safe_asdict(lm),
                    },
                )

                best_non_stop = _choose_better_non_stop(
                    best_non_stop,
                    candidate,
                )
            elif target_relevance in {"high", "medium"} and category in {"sign", "directory"}:
                e_type = _non_stop_evidence_type(category, combined)
                score, breakdown = _score_verification(
                    evidence_type=e_type,
                    matched_label="",
                    goal=goal,
                    current_frame=True,
                    is_stop_candidate=False,
                    visual_confidence=getattr(lm, "confidence", "medium"),
                )
                candidate = VerificationResult(
                    target_visible=True,
                    target_reached=False,
                    matched_label="",
                    evidence_type=e_type,
                    confidence=_confidence_from_score(score),
                    evidence_score=score,
                    evidence_breakdown=breakdown,
                    reason=(
                        "Current image contains target-relevant navigation "
                        "evidence, but not the actual target door/entrance."
                    ),
                    landmark_id=str(
                        getattr(lm, "id", "")
                    ) or None,
                    raw={
                        "source": "current_memory_relevant_non_stop",
                        "landmark": _safe_asdict(lm),
                    },
                )

                best_non_stop = _choose_better_non_stop(
                    best_non_stop,
                    candidate,
                )
        return best_non_stop

    def _apply_strict_guards(
        self,
        result: VerificationResult,
        goal: "NavigationGoal",
        memory_update: "MemoryUpdate | None",
    ) -> VerificationResult:
        """Prevent false positive stopping from signs, ranges, directories, or building letters."""
        result.confidence = _normalise_confidence(result.confidence)
        result.evidence_type = (result.evidence_type or "none").lower()

        valid_evidence = {
            "target_door_label", "target_entrance", "directional_sign", "directory",
            "room_range", "nearby_room", "zone_marker", "none", "unclear",
        }
        if result.evidence_type not in valid_evidence:
            result.evidence_type = "unclear"
            result.target_reached = False

        if result.evidence_type not in self.STOP_EVIDENCE_TYPES:
            result.target_reached = False
            score, breakdown = _score_verification(
                evidence_type=result.evidence_type,
                matched_label=result.matched_label,
                goal=goal,
                current_frame=True,
                is_stop_candidate=False,
                visual_confidence=result.confidence,
            )
            result.evidence_score = score
            result.evidence_breakdown = breakdown
            result.confidence = _confidence_from_score(score)
            return result

        matched = _find_stop_label(_stop_labels(goal), result.matched_label)
        if not matched:
            result.target_reached = False
            result.target_visible = bool(result.target_visible)
            result.evidence_type = "zone_marker" if _is_building_only_match(goal, result.matched_label) else "unclear"
            result.reason = (
                result.reason
                + " Verification rejected: matched_label is not a full target room/entrance label."
            ).strip()
            result.confidence = "low"
            result.evidence_score = 0.0
            result.evidence_breakdown = {
                "invalid_matched_label": 1.0,
                "final_score": 0.0,
            }
            return result

        # If the goal contains a room-code pattern, reject a building/zone-only label.
        if _is_building_only_match(goal, result.matched_label):
            result.target_reached = False
            result.evidence_type = "zone_marker"
            result.reason = (
                result.reason
                + " Verification rejected: visible label is only a building/tower marker, not the room label."
            ).strip()
            result.confidence = "low"
            result.evidence_score = 0.0
            result.evidence_breakdown = {
                "building_only_match": 1.0,
                "final_score": 0.0,
            }
            return result
        
        if not _current_memory_supports_stop(goal, memory_update, result.evidence_type, result.matched_label):
            result.target_reached = False
            result.target_visible = bool(result.target_visible)
            result.evidence_type = "unclear"
            result.reason = (
                result.reason
                + " Verification rejected: current-frame memory does not support the stop evidence."
            ).strip()
            result.confidence = "low"
            result.evidence_score = 0.0
            result.evidence_breakdown = {
                "missing_current_memory_support": 1.0,
                "final_score": 0.0,
            }
            return result

        # If a landmark id is provided, it must belong to the current frame's memory update.
        if result.landmark_id and memory_update:
            current_ids = {str(getattr(lm, "id", "")) for lm in getattr(memory_update, "landmarks", []) or []}
            if str(result.landmark_id) not in current_ids:
                result.target_reached = False
                result.reason = (
                    result.reason + " Verification rejected: landmark_id is not from the current frame."
                ).strip()
                result.confidence = "low"
                result.evidence_score = 0.0
                result.evidence_breakdown = {
                    "landmark_not_current_frame": 1.0,
                    "final_score": 0.0,
                }
                return result

        score, breakdown = _score_verification(
            evidence_type=result.evidence_type,
            matched_label=result.matched_label,
            goal=goal,
            current_frame=True,
            is_stop_candidate=True,
            visual_confidence=result.confidence,
        )

        result.evidence_score = score
        result.evidence_breakdown = breakdown
        result.confidence = _confidence_from_score(score)

        if score < 0.85:
            result.target_reached = False
            result.reason = result.reason + " Verification rejected: evidence score below stop threshold."
            return result

        result.target_visible = True
        result.target_reached = True
        result.matched_label = result.matched_label or matched
        if not result.reason:
            result.reason = f"Target verified: visible current-frame {result.evidence_type} matches '{matched}'."
        return result

    @staticmethod
    def _current_memory_context(memory_update: "MemoryUpdate | None") -> str:
        if memory_update is None:
            return "{}"
        data = {
            "useful": getattr(memory_update, "useful", False),
            "summary": getattr(memory_update, "summary", ""),
            "current_frame_landmarks": [
                _safe_asdict(lm) for lm in (getattr(memory_update, "landmarks", []) or [])
            ],
            "current_frame_hypotheses": getattr(memory_update, "hypotheses", []) or [],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)


# ── Helper functions ───────────────────────────────────────────────────────


def _stop_labels(goal: "NavigationGoal") -> list[str]:
    """Return labels that are allowed to stop navigation.

    This intentionally excludes partial building/floor/room fragments like "B",
    "0", or "004" for room-code goals. Only full labels such as B0.004,
    B0-004, B0004, or Room B0004 are accepted.
    """
    raw = str(getattr(goal, "raw_goal", "")).strip()
    labels: list[Any] = [raw]
    labels.extend(getattr(goal, "aliases", []) or [])

    constraints = getattr(goal, "constraints", {}) or {}
    target_name = constraints.get("target_name")
    if target_name:
        labels.append(target_name)

    cleaned: list[str] = []
    raw_norm = _normalise_label(raw)
    raw_has_letter_and_digit = bool(re.search(r"[A-Za-z]", raw) and re.search(r"\d", raw))
    raw_has_separator = bool(re.search(r"[.\-_]", raw))

    for label in labels:
        s = str(label).strip()
        norm = _normalise_label(s)
        if len(norm) < 3:
            continue
        if raw_has_letter_and_digit:
            # For room-code goals, full stop labels must include letters and digits.
            if not (re.search(r"[A-Za-z]", s) and re.search(r"\d", s)):
                continue
            # Reject aliases that are too short compared with the raw label.
            # Example: for B0.004 (b0004), reject B04 (b04) because it is ambiguous.
            if len(norm) < max(4, len(raw_norm) - 1):
                continue
        if raw_has_separator and len(norm) < len(raw_norm) - 1:
            continue
        cleaned.append(s)
    return _unique_strings(cleaned)


def _find_stop_label(labels: list[str], text: str) -> str:
    text_norm = _normalise_label(text)
    if not text_norm:
        return ""

    for label in labels:
        label_norm = _normalise_label(label)
        if not label_norm:
            continue

        if label_norm == text_norm:
            return label

        label_is_numeric_only = label_norm.isdigit()

        if label_is_numeric_only:
            # Avoid matching 314 inside 1314.
            pattern = rf"(?<!\d){re.escape(label_norm)}(?!\d)"
            if re.search(pattern, text_norm):
                return label
        else:
            # Allow C0.008 / C0-008 / C0008 inside longer visible door text.
            if label_norm in text_norm:
                return label

    return ""


def _normalise_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())

def _goal_building_or_zone(goal: "NavigationGoal") -> str:
    constraints = getattr(goal, "constraints", {}) or {}
    value = constraints.get("possible_building") or constraints.get("possible_zone") or ""
    return str(value).strip()

def _is_building_only_match(goal: "NavigationGoal", label: str) -> bool:
    building = _goal_building_or_zone(goal)
    if not building:
        return False
    return _normalise_label(label) == _normalise_label(building)

def _looks_like_zone_marker_for_goal(goal: "NavigationGoal", text: str) -> bool:
    building = _goal_building_or_zone(goal)
    if not building:
        return False
    t = text.lower()
    b = re.escape(str(building).lower())
    return bool(re.search(rf"\b{b}\b", t)) and any(w in t for w in ["tower", "building", "zone", "entrance", "door"])

def _landmark_combined_text(lm: "Landmark") -> str:
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
    ])

def _landmark_source_view(
    landmark: "Landmark",
) -> str:
    extra = (
        landmark.extra
        if isinstance(getattr(landmark, "extra", {}), dict)
        else {}
    )

    view = str(
        extra.get("source_view", "NONE")
    ).strip().upper()

    if view in {
        "LEFT",
        "FRONT",
        "RIGHT",
        "STITCHED_UNKNOWN",
        "NONE",
    }:
        return view

    return "NONE"

_NON_STOP_EVIDENCE_PRIORITY = {
    "target_door_label": 5,
    "target_entrance": 4,
    "room_range": 3,
    "directory": 2,
    "directional_sign": 2,
    "nearby_room": 2,
    "zone_marker": 1,
    "unclear": 0,
    "none": 0,
}

def _landmark_horizontal_position(
    landmark: "Landmark",
) -> str:
    extra = (
        landmark.extra
        if isinstance(
            getattr(landmark, "extra", {}),
            dict,
        )
        else {}
    )

    value = str(
        extra.get(
            "horizontal_position",
            "unknown",
        )
        or "unknown"
    ).strip().lower()

    if value == "centre":
        value = "center"

    return value


def _landmark_proximity(
    landmark: "Landmark",
) -> str:
    extra = (
        landmark.extra
        if isinstance(
            getattr(landmark, "extra", {}),
            dict,
        )
        else {}
    )

    return str(
        extra.get(
            "proximity",
            "unknown",
        )
        or "unknown"
    ).strip().lower()

def _landmark_ready_for_stop(
    landmark: "Landmark",
) -> bool:
    """
    Visual navigation-goal verification.

    Exact current target evidence must be in FRONT, centred,
    and not visually reported as "far". Fine-grained stand-off
    distance and obstacle clearance are handled separately by
    robot safety/LiDAR, but a "far" proximity reading means the
    target label is merely readable from a distance, not reached,
    and must not be treated as arrival.
    """
    return (
        _landmark_source_view(
            landmark
        )
        == "FRONT"
        and _landmark_horizontal_position(
            landmark
        )
        == "center"
        and _landmark_proximity(
            landmark
        )
        != "far"
    )

def _target_not_ready_reason(
    landmark: "Landmark",
    matched_label: str,
    target_kind: str,
) -> str:
    view = _landmark_source_view(
        landmark
    )

    horizontal = (
        _landmark_horizontal_position(
            landmark
        )
    )

    proximity = _landmark_proximity(
        landmark
    )

    return (
        "Exact {} '{}' is visible, but arrival is not "
        "confirmed: source_view={}, horizontal_position={}, "
        "proximity={}."
    ).format(
        target_kind,
        matched_label,
        view,
        horizontal,
        proximity,
    )

def _choose_better_non_stop(
    current: VerificationResult,
    candidate: VerificationResult,
) -> VerificationResult:
    """
    Keep the strongest non-stop verification result.

    Evidence type is considered before numerical score so that an exact
    target door visible from the side cannot be overwritten by a generic
    directional sign or zone marker.
    """
    current_priority = _NON_STOP_EVIDENCE_PRIORITY.get(
        current.evidence_type,
        0,
    )
    candidate_priority = _NON_STOP_EVIDENCE_PRIORITY.get(
        candidate.evidence_type,
        0,
    )

    if candidate_priority > current_priority:
        return candidate

    if candidate_priority < current_priority:
        return current

    if candidate.evidence_score > current.evidence_score:
        return candidate

    return current

def _looks_like_actual_entrance(text: str) -> bool:
    t = text.lower()
    entrance_words = ["entrance", "door", "room plate", "door plate", "office", "suite", "lab", "reception", "gate"]
    non_stop_words = ["arrow", "towards", "direction", "range", "directory", "map", "rooms", "tower", "building", "zone"]
    return any(w in t for w in entrance_words) and not any(w in t for w in non_stop_words)

def _non_stop_evidence_type(category: str, text: str) -> str:
    t = text.lower()
    if category == "directory" or "directory" in t or "map" in t:
        return "directory"
    if "range" in t or re.search(r"[a-z]\d+[.\-]?\d+\s*[-–]\s*[a-z]?\d+[.\-]?\d+", t):
        return "room_range"
    if any(w in t for w in ["tower", "building", "zone"]):
        return "zone_marker"
    if "arrow" in t or "←" in t or "→" in t or "left" in t or "right" in t or "straight" in t:
        return "directional_sign"
    return "directional_sign" if category == "sign" else "unclear"


def _normalise_confidence(value: Any) -> str:
    s = str(value).strip().lower()
    return s if s in {"high", "medium", "low"} else "low"


def _clean_optional_id(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"none", "null"}:
        return None
    return s


def _safe_asdict(obj: Any) -> dict[str, Any]:
    try:
        return asdict(obj)
    except Exception:
        return dict(getattr(obj, "__dict__", {}))


def _unique_strings(items: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        s = str(item).strip()
        if not s:
            continue
        key = s.lower()
        if key not in seen:
            out.append(s)
            seen.add(key)
    return out

def _extract_json(
    text: str,
) -> dict[str, Any] | None:
    cleaned = re.sub(
        r"```(?:json)?",
        "",
        str(text),
        flags=re.IGNORECASE,
    ).replace("```", "").strip()

    decoder = json.JSONDecoder()

    for match in re.finditer(
        r"\{",
        cleaned,
    ):
        candidate = cleaned[
            match.start():
        ]

        try:
            value, _ = decoder.raw_decode(
                candidate
            )
        except json.JSONDecodeError:
            continue

        if isinstance(value, dict):
            return value

    return None

def _confidence_from_score(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"
    
def _current_memory_supports_stop(
    goal: "NavigationGoal",
    memory_update: "MemoryUpdate | None",
    evidence_type: str,
    matched_label: str,
) -> bool:
    if memory_update is None:
        return False

    labels = _stop_labels(goal)
    matched = _find_stop_label(labels, matched_label)
    if not matched:
        return False

    for lm in getattr(memory_update, "landmarks", []) or []:
        category = str(getattr(lm, "category", "")).lower()
        combined = _landmark_combined_text(lm)

        if not _find_stop_label(labels, combined):
            continue
   
        if (evidence_type == "target_door_label" and category == "door"):
            if _landmark_ready_for_stop(lm):
                return True
            continue

        if (evidence_type == "target_entrance" and category in {"sign", "observation"}):
            if (
                _landmark_ready_for_stop(lm)
                and _looks_like_actual_entrance(
                    combined
                )
            ):
                return True
            continue

    return False

def _score_verification(
        evidence_type: str,
        matched_label: str,
        goal: "NavigationGoal",
        current_frame: bool = True,
        is_stop_candidate: bool = False,
        visual_confidence: str = "medium",
    ) -> tuple[float, dict[str, float]]:
        evidence_type_scores = {
            "target_door_label": 0.95,
            "target_entrance": 0.90,
            "directional_sign": 0.65,
            "directory": 0.55,
            "room_range": 0.50,
            "nearby_room": 0.40,
            "zone_marker": 0.25,
            "unclear": 0.10,
            "none": 0.00,
        }

        visual_scores = {
            "high": 0.10,
            "medium": 0.05,
            "low": 0.00,
        }

        base = evidence_type_scores.get(evidence_type, 0.10)
        visual_bonus = visual_scores.get(str(visual_confidence).lower(), 0.00)

        labels = _stop_labels(goal)
        label_match_bonus = 0.0
        if matched_label and _find_stop_label(labels, matched_label):
            label_match_bonus = 0.10

        current_frame_bonus = 0.05 if current_frame else -0.20

        ambiguity_penalty = 0.0
        if evidence_type in {"zone_marker", "nearby_room", "unclear"}:
            ambiguity_penalty += 0.20

        if is_stop_candidate and evidence_type not in {"target_door_label", "target_entrance"}:
            ambiguity_penalty += 0.30

        score = base + visual_bonus + label_match_bonus + current_frame_bonus - ambiguity_penalty
        score = max(0.0, min(1.0, score))

        breakdown = {
            "evidence_type_score": base,
            "visual_bonus": visual_bonus,
            "label_match_bonus": label_match_bonus,
            "current_frame_bonus": current_frame_bonus,
            "ambiguity_penalty": ambiguity_penalty,
            "final_score": score,
        }

        return score, breakdown