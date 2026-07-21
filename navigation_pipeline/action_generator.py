"""action_generator.py
    High-level action generator for generalized indoor navigation.

    The VLM selects skill primitives, not low-level robot control.

    The planner uses:
    - current semantic evidence such as signs, directories, room labels, and reception;
    - current structural evidence such as corridors, bends, junctions, and doorways;
    - recent remembered structural landmarks when the current view is visually sparse.

    Semantic evidence decides which route is relevant to the goal.
    Structural landmarks support execution of that route and must not override
    stronger or newer semantic evidence.
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
        "NAVIGATE_TO_LANDMARK": (
            "Navigate toward a current or recently remembered semantic/structural "
            "route landmark that is compatible with the active goal or direction."
        ),
        "NAVIGATE_TO_FRONTIER": (
            "Move toward a current reachable unexplored path only when no stronger "
            "semantic or remembered route evidence is available."
        ),
        "FOLLOW_DIRECTION": "Follow a direction from current/local evidence such as a sign, directory, room-range sign, or official staff/reception instruction.",
        "ASK_RECEPTION_OR_STAFF": "Ask only a clearly visible official help source: reception, front desk, information desk, security desk, or staff/help counter.",
        "USE_ELEVATOR_OR_STAIRS": "Use lift/stairs when recent evidence says a floor transition is needed.",
        "ALIGN_WITH_LANDMARK": "Rotate toward a currently visible landmark until it is approximately centred in the front camera.",
        "APPROACH_LANDMARK": "Move toward a currently visible target landmark while keeping it visible and stopping when the stated stop condition is reached.",
        "PASS_THROUGH_DOORWAY": "Approach and safely pass through a clearly visible open doorway that is relevant to the navigation goal.",
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

        The structured memory may contain:
        - recent_landmarks: landmarks from the latest observations;
        - target_relevant_landmarks: semantic evidence relevant to the goal;
        - remembered_route_landmarks: corridors, bends, junctions, passages, doorways, frontiers, and dead ends observed in current or recent images.

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
        - Prefer direct navigation cues (room labels, room-range signs, directional signs, directories, zone/building markers, corridors, doorways) over asking for help.
        - Use ASK_RECEPTION_OR_STAFF only when no reliable navigation cue is available, or the robot cannot make further progress after searching.
        - If both reception and a plausible navigation cue toward the goal are visible, continue navigation using the visual cue instead of asking for help.
        - If no useful cue is visible, use NAVIGATE_TO_FRONTIER or SEARCH_FOR_CUE.
        Structural route-memory rules:
        - Do not immediately use SEARCH_FOR_CUE or WAIT_OR_RECOVER merely because the current image contains plain walls or an visually empty corridor.
        - First check remembered_route_landmarks for a recent route continuation, corridor bend, junction, passage, or doorway that has not been passed, blocked, ignored, or contradicted.
        - A remembered structural landmark may support continued movement when:
          1. it was observed recently;
          2. it is still marked visible_now or remembered;
          3. it is traversable or not known to be blocked;
          4. its direction is compatible with the latest valid semantic direction;
          5. no newer current evidence contradicts it.
        - Use NAVIGATE_TO_LANDMARK for a remembered corridor continuation, corridor bend, junction, or passage when the current view is sparse and the remembered landmark remains the best route anchor.
        - Do not use APPROACH_LANDMARK, ALIGN_WITH_LANDMARK, or PASS_THROUGH_DOORWAY for a landmark that is not visible in the current observation. Those actions require current visual tracking.
        - A remembered route landmark is not proof that it leads to the final goal. It may only continue a route already supported by semantic evidence.
        - Current exact room labels, room-range signs, directories, directional signs, and floor evidence override remembered structural landmarks.
        - A generic open doorway must not override a corridor simply because the doorway has higher confidence or is visually prominent.
        - Prefer a corridor or passage with navigation_role="continue_route" when it matches the active direction.
        - Prefer a corridor_bend with navigation_role="turn_point" when the robot is progressing toward the remembered bend.
        - Use a doorway with navigation_role="entrance" only when current or remembered semantic evidence associates that doorway with the goal, tower, zone, or required route.
        - Never navigate toward a landmark with navigation_role="dead_end", route_state="blocked", route_state="passed", or traversable=false.
        - Do not choose landmarks solely by evidence_score. Apply goal relevance, directional compatibility, navigation role, traversability, route state, and recency before comparing confidence.
        Route-selection priority:
        1. Exact current target evidence, such as the target door label.
        2. Current semantic directional evidence relevant to the goal.
        3. Current structural landmark compatible with that semantic direction.
        4. Recent remembered structural landmark compatible with that direction.
        5. Current reachable frontier.
        6. SEARCH_FOR_CUE.
        7. WAIT_OR_RECOVER only when the input failed, the route is unsafe,
           blocked, contradictory, or no safe progress action exists.
        - If a signboard, room-range sign, directory, or door label is visible but the text is too small, blurry, overexposed, or unreadable, do NOT guess the text or direction signs(arrow marks).
        - In that case, consider moving closer only when the unclear cue is the most relevant available evidence toward the goal.
        - If an unreadable but potentially useful navigation cue is visible, choose an action that safely approaches the cue only if it is the most relevant current evidence toward the goal.
        - Do not ignore a clearer or more goal-relevant navigation cue elsewhere in the current views.
        - In the reason, explicitly say that the text is not clearly readable and moving closer may help decide the direction.
        - Include "direction" only when the chosen action requires movement toward the selected cue.
        - Always include "evidence_view" for the selected cue.
        - When multiple navigation cues are visible, choose the cue that makes the most progress toward the goal rather than the most visually prominent object.
        - Use USE_ELEVATOR_OR_STAIRS only when recent evidence shows a lift/stairs and the goal/floor evidence suggests a floor transition.
        - Never assume a room-code structure is true until signs/labels/directories confirm it.
        - A landmark marked "used" or "visited" may be reused when it is visible in the current images and remains relevant to the active navigation goal.
        - Prefer a newer cue only when the previous landmark is no longer visible, has already been passed, or no longer provides useful progress.
        - Treat building/zone-only markers such as "B" for goal "B0.004" as navigation cues, not strong target evidence.
        - Prefer navigating toward intermediate landmarks that lead closer to the goal (e.g., the correct tower entrance, corridor, doorway, or junction) before considering assistance from reception or staff.
        - FOLLOW_DIRECTION must include landmark_id of the sign/directory/reception/stairs/elevator evidence that supports the direction.
        - FOLLOW_DIRECTION must use directional evidence visible in the current observation. Do not repeatedly issue FOLLOW_DIRECTION from an old sign that is no longer visible.
        - For FOLLOW_DIRECTION, direction must be exactly "left", "right", or "forward". Put landmark names in target or target_description, not in direction.
        - Prefer target-oriented actions over vague standalone direction commands when a useful visible landmark can serve as the next subgoal.
        - Use ALIGN_WITH_LANDMARK when the selected landmark is visible in LEFT or RIGHT and the robot must first turn until it is centred in the FRONT view.
        - Use APPROACH_LANDMARK when the selected landmark is visible ahead and the robot should move toward it while keeping it visible.
        - Use PASS_THROUGH_DOORWAY only when a clearly open doorway is the intended route toward the goal.
        - Every target-oriented action must include landmark_id, target_description, stop_condition, evidence_view, and capture_after=true.
        - Stop conditions must be visually or sensor verifiable, such as "landmark centred in FRONT", "robot near doorway", or "doorway threshold reached".
        - Do not specify exact movement duration, wheel velocity, or number of robot steps.
        - For debugging and evaluation, every action must report where the strongest current visual evidence came from.
        - Include "evidence_view" inside params.
        - evidence_view must be one of: "LEFT", "FRONT", "RIGHT", "STITCHED_UNKNOWN", or "NONE".
        - Use "LEFT" if the strongest cue is in the left camera/panel.
        - Use "FRONT" if the strongest cue is in the front camera/panel.
        - Use "RIGHT" if the strongest cue is in the right camera/panel.
        - Use "STITCHED_UNKNOWN" if the image is stitched but the panel/source is unclear.
        - Use "NONE" if no useful visual cue is visible.
        - If the exact target door or target entrance is visible only in LEFT or RIGHT, choose ALIGN_WITH_LANDMARK first. Do not claim that the target has been reached.
        - Confirm the target only after a new observation shows the target in FRONT.

        When using a remembered structural landmark:
        - Choose NAVIGATE_TO_LANDMARK.
        - Include its exact landmark_id.
        - Include direction from continuation_direction when available.
        - Include target_description from the stored landmark description.
        - Use a stop_condition that requires a new observation, such as:
          "capture when near the corridor bend",
          "capture after advancing toward the remembered junction", or
          "capture before entering the passage".
        - Set capture_after=true.
        - Set evidence_view to the stored source_view.
        - In the reason, explicitly say that the current view is sparse and the
          recent remembered route landmark supports continued progress.

        Return ONLY valid JSON:
        {
        "action": "READ_SIGN | CHECK_DOOR_LABEL | NAVIGATE_TO_LANDMARK | NAVIGATE_TO_FRONTIER | FOLLOW_DIRECTION | ASK_RECEPTION_OR_STAFF | USE_ELEVATOR_OR_STAIRS | SEARCH_FOR_CUE | WAIT_OR_RECOVER | ALIGN_WITH_LANDMARK | APPROACH_LANDMARK | PASS_THROUGH_DOORWAY",
        "params": {
            "landmark_id": null,
            "direction": null,
            "target": null,
            "target_description": null,
            "floor": null,
            "search_for": null,
            "stop_condition": null,
            "capture_after": true,
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
        # An exact current target door visible in a side camera has
        # higher priority than a directional sign pointing toward it.
        side_target_lm = _best_current_side_target_landmark(memory, goal)
        if side_target_lm is not None:
            side_view = _infer_evidence_view_from_landmark(
                side_target_lm
            )
            validated = Action(
                name="ALIGN_WITH_LANDMARK",
                params={
                    "landmark_id": str(side_target_lm.id),
                    "direction": side_view.lower(),
                    "target": str(
                        getattr(goal, "raw_goal", "")
                    ),
                    "target_description": (
                        str(
                            getattr(side_target_lm, "description","")
                        ).strip()
                        or (
                            "Exact target door visible in "
                            f"{side_view}"
                        )
                    ),
                    "stop_condition": (
                        "target landmark centred in FRONT camera"
                    ),
                    "capture_after": True,
                    "evidence_view": side_view,
                },
                reason=(
                    "The exact target door is visible in the "
                    f"{side_view} view and must be centred in "
                    "the FRONT camera before arrival is verified."
                ),
                confidence=str(
                    getattr(
                        side_target_lm,
                        "confidence",
                        "medium",
                    )
                ).lower(),
                goal_reached=False,
                needs_verification=True,
            )

        # Do not let a directional sign replace an exact side-view target.
        semantic_lm = (
            None
            if side_target_lm is not None
            else _best_current_semantic_direction_landmark(
                memory
            )
        )
        if semantic_lm is not None:
            extra = (
                semantic_lm.extra
                if isinstance(getattr(semantic_lm, "extra", {}), dict)
                else {}
            )
            semantic_direction = _semantic_landmark_direction(semantic_lm)
            selected_lm_id = _clean_param(
                validated.params.get("landmark_id")
            )
            selected_direction = _normalise_direction(
                validated.params.get("direction")
            )
            already_using_current_semantic = (
                validated.name == "FOLLOW_DIRECTION"
                and selected_lm_id == str(semantic_lm.id)
                and selected_direction == semantic_direction
            )
            if semantic_direction and not already_using_current_semantic:
                validated = Action(
                    name="FOLLOW_DIRECTION",
                    params={
                        "landmark_id": str(semantic_lm.id),
                        "direction": semantic_direction,
                        "target": str(getattr(goal, "raw_goal", "")),
                        "target_description": str(
                            getattr(semantic_lm, "description", "")
                        ),
                        "stop_condition": (
                            f"robot aligned with the {semantic_direction} route "
                            "indicated by the current sign"
                        ),
                        "capture_after": True,
                        "evidence_view": extra.get(
                            "source_view",
                            "NONE",
                        ),
                    },
                    reason=(
                        "Current target-relevant directional evidence has priority "
                        "over structural or exploratory actions."
                    ),
                    confidence="high",
                )

                validated = self._validate(
                    validated,
                    memory,
                    goal,
                )
        validated = self._enforce_visual_alignment(validated, memory)
        validated = self._validate(validated, memory, goal)
        self._attach_action_evidence_score(
            validated,
            memory,
            goal,
        )
        self._ensure_evidence_view(
            validated,
            memory,
        )
        return validated

    def _ensure_evidence_view(self, action: Action, memory: "NavigationMemory") -> None:
        """
        Ensure every action has params["evidence_view"] for debugging/evaluation.

        Allowed:
        LEFT, FRONT, RIGHT, STITCHED_UNKNOWN, NONE
        """
        if not isinstance(action.params, dict):
            action.params = {}

        current = _normalise_evidence_view(action.params.get("evidence_view"))
        if current:
            action.params["evidence_view"] = current
            return

        inferred = ""

        lm_id = _clean_param(action.params.get("landmark_id"))
        if lm_id:
            lm = _get_landmark(memory, lm_id)
            inferred = _infer_evidence_view_from_landmark(lm)

        if not inferred:
            inferred = _infer_evidence_view_from_action(action)

        action.params["evidence_view"] = inferred or "NONE"

    def _enforce_visual_alignment(
        self,
        action: Action,
        memory: "NavigationMemory",
    ) -> Action:
        """
        Ensure physically reachable target actions.

        A landmark in LEFT/RIGHT must first be aligned into
        the FRONT camera before approach or doorway traversal.
        """
        if not action.is_valid:
            return action
        if action.name not in {
            "CHECK_DOOR_LABEL",
            "NAVIGATE_TO_LANDMARK",
            "APPROACH_LANDMARK",
            "PASS_THROUGH_DOORWAY",
        }:
            return action
        lm_id = _clean_param(action.params.get("landmark_id"))
        if not lm_id:
            return action
        lm = _get_landmark(memory, lm_id)
        if lm is None:
            return action
        if not _landmark_is_current(memory, lm):
            return action
        view = _infer_evidence_view_from_landmark(lm)
        if view == "FRONT":
            return action
        if view not in {"LEFT", "RIGHT"}:
            return action
        return Action(
            name="ALIGN_WITH_LANDMARK",
            params={
                "landmark_id": lm_id,
                "direction": view.lower(),
                "target_description": (
                    _clean_param(
                        action.params.get("target_description")
                    )
                    or str(
                        getattr(lm, "description", "")
                    ).strip()
                ),
                "stop_condition": (
                    "landmark centred in FRONT camera"
                ),
                "capture_after": True,
                "evidence_view": view,
            },
            reason=(
                "Selected landmark is visible in the "
                f"{view} view and must be aligned "
                "before approaching."
            ),
            confidence=action.confidence,
        )
    
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

            status = str(getattr(lm, "status", "")).lower()

            if status in {"used", "visited"} and not _landmark_is_current(memory, lm):
                action.is_valid = False
                action.invalid_reason = (
                    f"Landmark {lm_id} was already {lm.status} and is not visible "
                    "in the current observation."
                )
                return action

        if action.name == "NAVIGATE_TO_LANDMARK":
            lm_id = _clean_param(action.params.get("landmark_id"))
            if not lm_id:
                action.is_valid = False
                action.invalid_reason = (
                    "NAVIGATE_TO_LANDMARK requires landmark_id."
                )
                return action
            lm = _get_landmark(memory, lm_id)

            if lm is None:
                action.is_valid = False
                action.invalid_reason = f"Unknown landmark_id: {lm_id}"
                return action

            status = str(getattr(lm, "status", "")).lower()
            extra = (
                getattr(lm, "extra", {})
                if isinstance(getattr(lm, "extra", {}), dict)
                else {}
            )
            is_current = _landmark_is_current(memory, lm)
            is_structural = _is_structural_landmark(lm)
            if status in {"ignored", "used", "visited"} and not is_current:
                action.is_valid = False
                action.invalid_reason = (
                    f"Landmark {lm_id} is {status} and is no longer current."
                )
                return action
            if is_structural:
                valid_route, route_reason = _validate_structural_route_landmark(
                    memory=memory,
                    landmark=lm,
                    action=action,
                )

                if not valid_route:
                    action.is_valid = False
                    action.invalid_reason = route_reason
                    return action

                # Structural route actions should always stop and recapture.
                action.params["capture_after"] = True

                if not _clean_param(action.params.get("target_description")):
                    action.params["target_description"] = str(
                        getattr(lm, "description", "")
                    ).strip()

                if not _clean_param(action.params.get("stop_condition")):
                    action.params["stop_condition"] = (
                        "advance toward the route landmark, "
                        "then stop and capture a new observation"
                    )

                remembered_direction = _structural_landmark_direction(lm)
                if remembered_direction and not _normalise_direction(
                    action.params.get("direction")
                ):
                    action.params["direction"] = remembered_direction

            elif not is_current:
                # Do not approach stale semantic objects as physical navigation targets.
                action.is_valid = False
                action.invalid_reason = (
                    f"NAVIGATE_TO_LANDMARK semantic landmark {lm_id} "
                    "is not visible in the current observation."
                )
                return action

        if action.name in {
            "ALIGN_WITH_LANDMARK",
            "APPROACH_LANDMARK",
            "PASS_THROUGH_DOORWAY",
        }:
            lm_id = _clean_param(action.params.get("landmark_id"))
            target_description = _clean_param(
                action.params.get("target_description")
            )
            stop_condition = _clean_param(
                action.params.get("stop_condition")
            )

            if not lm_id:
                action.is_valid = False
                action.invalid_reason = (
                    f"{action.name} requires landmark_id."
                )
                return action

            lm = _get_landmark(memory, lm_id)
            if lm is None:
                action.is_valid = False
                action.invalid_reason = (
                    f"Unknown landmark_id: {lm_id}"
                )
                return action

            if not _landmark_is_current(memory, lm):
                action.is_valid = False
                action.invalid_reason = (
                    f"{action.name} requires landmark {lm_id} "
                    "to be visible in the current observation."
                )
                return action
            
            view = _infer_evidence_view_from_landmark(lm)
            if (
                action.name == "ALIGN_WITH_LANDMARK"
                and view not in {"LEFT", "RIGHT"}
            ):
                action.is_valid = False
                action.invalid_reason = (
                    "ALIGN_WITH_LANDMARK requires a landmark "
                    "visible in LEFT or RIGHT."
                )
                return action
            if (
                action.name in {
                    "APPROACH_LANDMARK",
                    "PASS_THROUGH_DOORWAY",
                }
                and view != "FRONT"
            ):
                action.is_valid = False
                action.invalid_reason = (
                    f"{action.name} requires landmark "
                    "to be visible in FRONT."
                )
                return action

            if not target_description:
                action.is_valid = False
                action.invalid_reason = (
                    f"{action.name} requires target_description."
                )
                return action

            if not stop_condition:
                action.is_valid = False
                action.invalid_reason = (
                    f"{action.name} requires stop_condition."
                )
                return action

            action.params["capture_after"] = True

            if action.name == "PASS_THROUGH_DOORWAY":
                allowed_categories = {
                    "doorway",
                    "passage",
                    "door",      # backward compatibility
                    "frontier",  # backward compatibility
                }
                category = str(getattr(lm, "category", "")).lower()
                if category not in allowed_categories:
                    action.is_valid = False
                    action.invalid_reason = (
                        "PASS_THROUGH_DOORWAY requires a doorway, passage, "
                        "door, or compatible frontier landmark."
                    )
                    return action
                extra = (
                    getattr(lm, "extra", {})
                    if isinstance(getattr(lm, "extra", {}), dict)
                    else {}
                )
                if extra.get("traversable") is False:
                    action.is_valid = False
                    action.invalid_reason = (
                        f"PASS_THROUGH_DOORWAY landmark {lm_id} "
                        "is not traversable."
                    )
                    return action
                route_state = str(
                    extra.get("route_state", "visible_now")
                ).lower()

                if route_state in {"blocked", "passed"}:
                    action.is_valid = False
                    action.invalid_reason = (
                        f"PASS_THROUGH_DOORWAY landmark {lm_id} "
                        f"has route_state={route_state}."
                    )
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
            direction = _normalise_direction(
                action.params.get("direction")
            )

            if not direction:
                action.is_valid = False
                action.invalid_reason = (
                    "FOLLOW_DIRECTION direction must be "
                    "left, right, or forward."
                )
                return action

            action.params["direction"] = direction

            target = _clean_param(
                action.params.get("target")
            )
            lm_id = _clean_param(
                action.params.get("landmark_id")
            )
            lm = None

            if not lm_id:
                supporting_lm = (
                    _best_current_semantic_direction_landmark(memory)
                )

                if (
                    supporting_lm is not None
                    and _landmark_supports_direction(
                        supporting_lm,
                        direction,
                    )
                ):
                    lm_id = str(
                        getattr(supporting_lm, "id", "")
                    )
                    action.params["landmark_id"] = lm_id
                    lm = supporting_lm

            if not lm_id:
                action.is_valid = False
                action.invalid_reason = (
                    "FOLLOW_DIRECTION requires a current "
                    "semantic landmark_id."
                )
                return action

            if not _landmark_exists(memory, lm_id):
                supporting_lm = (
                    _best_current_semantic_direction_landmark(memory)
                )

                if (
                    supporting_lm is not None
                    and _landmark_supports_direction(
                        supporting_lm,
                        direction,
                    )
                ):
                    lm_id = str(
                        getattr(supporting_lm, "id", "")
                    )
                    action.params["landmark_id"] = lm_id
                    lm = supporting_lm
                else:
                    action.is_valid = False
                    action.invalid_reason = (
                        "The supplied direction landmark ID is unknown, "
                        "and no current semantic landmark supports "
                        f"direction={direction}."
                    )
                    return action

            if lm is None:
                lm = _get_landmark(memory, lm_id)

            if lm is None:
                action.is_valid = False
                action.invalid_reason = (
                    f"Unknown direction landmark_id: {lm_id}"
                )
                return action

            if not _landmark_is_current(memory, lm):
                action.is_valid = False
                action.invalid_reason = (
                    f"FOLLOW_DIRECTION landmark {lm_id} is not "
                    "visible in the current observation."
                )
                return action

            if not _landmark_supports_direction(
                lm,
                direction,
            ):
                action.is_valid = False
                action.invalid_reason = (
                    f"FOLLOW_DIRECTION landmark {lm_id} does not "
                    f"support requested direction={direction}."
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
            "ALIGN_WITH_LANDMARK": 0.65,
            "APPROACH_LANDMARK": 0.65,
            "PASS_THROUGH_DOORWAY": 0.60,
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
        structural_route_penalty = 0.0

        if lm_id:
            lm = _get_landmark(memory, lm_id)
            if lm is not None:
                extra = getattr(lm, "extra", {}) if isinstance(getattr(lm, "extra", {}), dict) else {}

                if extra.get("partial_goal_marker"):
                    partial_marker_penalty = 0.25

                if str(getattr(lm, "status", "")).lower() in {"used", "visited"}:
                    if _landmark_is_current(memory, lm):
                        used_landmark_penalty = 0.0
                    else:
                        used_landmark_penalty = 0.30
                
                mismatched_room_code_penalty = _room_code_mismatch_penalty(
                    lm=lm,
                    goal=goal,
                    action_name=action.name,
                )
                if _is_structural_landmark(lm):
                    extra = (
                        getattr(lm, "extra", {})
                        if isinstance(getattr(lm, "extra", {}), dict)
                        else {}
                    )

                    navigation_role = str(
                        extra.get("navigation_role", "")
                    ).lower()

                    goal_support = str(
                        extra.get("goal_support", "unknown")
                    ).lower()

                    if (
                        navigation_role == "entrance"
                        and goal_support not in {"direct", "indirect"}
                    ):
                        structural_route_penalty = 0.25

        no_landmark_penalty = 0.0
        if action.name in {"FOLLOW_DIRECTION", "CHECK_DOOR_LABEL", "READ_SIGN", "NAVIGATE_TO_LANDMARK", "ALIGN_WITH_LANDMARK", "APPROACH_LANDMARK", "PASS_THROUGH_DOORWAY",} and not lm_id:
            no_landmark_penalty = 0.15

        invalid_penalty = 0.40 if not action.is_valid else 0.0

        score = (
            base
            + landmark_bonus
            - no_landmark_penalty
            - invalid_penalty
            - partial_marker_penalty
            - used_landmark_penalty
            - mismatched_room_code_penalty
            - structural_route_penalty
        )
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
            "structural_route_penalty": structural_route_penalty,
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

_STRUCTURAL_CATEGORIES = {
    "corridor",
    "corridor_bend",
    "junction",
    "doorway",
    "passage",
    "frontier",
    "dead_end",
}

def _is_structural_landmark(lm: "Landmark | None") -> bool:
    if lm is None:
        return False
    category = str(getattr(lm, "category", "")).lower()
    extra = (
        getattr(lm, "extra", {})
        if isinstance(getattr(lm, "extra", {}), dict)
        else {}
    )
    return (
        category in _STRUCTURAL_CATEGORIES
        or str(extra.get("landmark_type", "")).lower() == "structural"
    )

def _structural_landmark_direction(
    lm: "Landmark | None",
) -> str:
    if lm is None:
        return ""
    extra = (
        getattr(lm, "extra", {})
        if isinstance(getattr(lm, "extra", {}), dict)
        else {}
    )
    raw_direction = (
        extra.get("continuation_direction")
        or extra.get("direction")
        or ""
    )
    text = str(raw_direction).strip().lower()
    if text in {"left", "left_forward", "forward_left"}:
        return "left"
    if text in {"right", "right_forward", "forward_right"}:
        return "right"
    if text in {"forward", "straight", "ahead", "continue"}:
        return "forward"
    return ""

def _structural_landmark_bearing(
    lm: "Landmark | None",
) -> str:
    """
    Return where the structural landmark currently appears relative to the robot.

    This is different from continuation_direction:
    - bearing='right' means turn right now to face the landmark;
    - continuation_direction='forward' means continue forward after alignment.
    """
    if lm is None:
        return ""

    extra = (
        getattr(lm, "extra", {})
        if isinstance(getattr(lm, "extra", {}), dict)
        else {}
    )

    explicit_bearing = _normalise_direction(
        extra.get("bearing")
        or extra.get("relative_direction")
    )

    if explicit_bearing:
        return explicit_bearing

    source_view = _normalise_evidence_view(
        extra.get("source_view")
        or extra.get("evidence_view")
    )

    if source_view == "LEFT":
        return "left"

    if source_view == "RIGHT":
        return "right"

    if source_view == "FRONT":
        return "forward"

    return ""

def _validate_structural_route_landmark(
    memory: "NavigationMemory",
    landmark: "Landmark",
    action: Action,
    max_age_images: int = 3,
) -> tuple[bool, str]:
    extra = (
        getattr(landmark, "extra", {})
        if isinstance(getattr(landmark, "extra", {}), dict)
        else {}
    )

    category = str(getattr(landmark, "category", "")).lower()
    landmark_id = str(getattr(landmark, "id", ""))
    status = str(getattr(landmark, "status", "")).lower()

    route_state = str(
        extra.get("route_state", "visible_now")
    ).lower()

    navigation_role = str(
        extra.get("navigation_role", "")
    ).lower()

    traversable = extra.get("traversable")
    source_index = extra.get("source_image_index")
    current_index = int(getattr(memory, "image_count", 0) or 0)

    if status == "ignored":
        return False, f"Structural landmark {landmark_id} is ignored."

    if route_state in {"blocked", "passed"}:
        return (
            False,
            f"Structural landmark {landmark_id} has "
            f"route_state={route_state}.",
        )

    if traversable is False:
        return (
            False,
            f"Structural landmark {landmark_id} is not traversable.",
        )

    if category == "dead_end" or navigation_role == "dead_end":
        return (
            False,
            f"Structural landmark {landmark_id} is a dead end.",
        )

    # Remembered route landmarks must be recent.
    if source_index is not None:
        try:
            age = current_index - int(source_index)
        except (TypeError, ValueError):
            age = max_age_images + 1

        if age > max_age_images:
            return (
                False,
                f"Remembered structural landmark {landmark_id} is stale "
                f"({age} observations old).",
            )

    action_direction = _normalise_direction(
        action.params.get("direction")
    )
    landmark_direction = _structural_landmark_direction(landmark)

    if (
        action_direction
        and landmark_direction
        and action_direction != landmark_direction
    ):
        return (
            False,
            f"Action direction {action_direction} conflicts with "
            f"structural landmark direction {landmark_direction}.",
        )

    if navigation_role == "entrance":
        goal_support = str(
            extra.get("goal_support", "unknown")
        ).lower()

        target_relevance = str(
            extra.get("target_relevance", "none")
        ).lower()

        # Generic doorways should not become route targets without support.
        if (
            goal_support not in {"direct", "indirect"}
            and target_relevance not in {"high", "medium"}
        ):
            return (
                False,
                f"Doorway landmark {landmark_id} is not linked to "
                "the active goal or route.",
            )
    
    current_semantic = (_best_current_semantic_direction_landmark(memory))

    if current_semantic is not None:
        semantic_direction = (_semantic_landmark_direction(current_semantic))

        if semantic_direction:
            return (
                False,
                f"Current semantic landmark {current_semantic.id} "
                f"must be handled before structural landmark "
                f"{landmark_id}.",
            )

    return True, ""

def _latest_semantic_direction(
    memory: "NavigationMemory",
    max_age_images: int = 3,
) -> str:
    current_index = int(getattr(memory, "image_count", 0) or 0)

    semantic_categories = {
        "sign",
        "directory",
        "reception",
        "stairs",
        "elevator",
        "observation",
        "door",
    }

    for lm in reversed(getattr(memory, "landmarks", [])):
        category = str(getattr(lm, "category", "")).lower()

        if category not in semantic_categories:
            continue

        extra = (
            getattr(lm, "extra", {})
            if isinstance(getattr(lm, "extra", {}), dict)
            else {}
        )
        source_index = extra.get("source_image_index")
        if source_index is not None:
            try:
                age = current_index - int(source_index)
            except (TypeError, ValueError):
                continue

            if age > max_age_images:
                continue
        target_relevance = str(
            extra.get("target_relevance", "none")
        ).lower()

        if category not in {"stairs", "elevator"}:
            if target_relevance not in {"high", "medium"}:
                continue
        direction = _semantic_landmark_direction(lm)
        if direction:
            return direction

    return ""

def _best_current_semantic_direction_landmark(
    memory: "NavigationMemory",
) -> "Landmark | None":
    current_index = getattr(memory, "image_count", None)

    best = None
    best_score = -1.0

    for lm in getattr(memory, "landmarks", []):
        extra = (
            lm.extra
            if isinstance(getattr(lm, "extra", {}), dict)
            else {}
        )

        if extra.get("source_image_index") != current_index:
            continue

        if str(getattr(lm, "category", "")).lower() not in {
            "sign",
            "directory",
            "observation",
        }:
            continue

        relevance = str(
            extra.get("target_relevance", "none")
        ).lower()

        direction = _semantic_landmark_direction(lm)

        if relevance not in {"high", "medium"} or not direction:
            continue

        score = float(
            getattr(lm, "evidence_score", 0.0) or 0.0
        )

        if score > best_score:
            best = lm
            best_score = score

    return best

def _best_current_side_target_landmark(
    memory: "NavigationMemory",
    goal: "NavigationGoal",
) -> "Landmark | None":
    """
    Return an exact current target door/entrance visible
    in LEFT or RIGHT.

    A side-view target must be aligned into FRONT before
    the verifier is allowed to stop.
    """
    current_index = getattr(
        memory,
        "image_count",
        None,
    )

    raw_goal = str(
        getattr(goal, "raw_goal", "")
    ).strip()

    raw_goal_norm = _normalise_room_code(raw_goal)

    goal_code_norms = _goal_room_code_norms(goal)

    # Determine whether this is a room-code goal such as C0.004.
    raw_goal_codes = {
        _normalise_room_code(code)
        for code in _extract_room_codes(raw_goal)
        if _normalise_room_code(code)
    }

    best: "Landmark | None" = None
    best_score = -1.0

    for lm in getattr(memory, "landmarks", []):
        extra = (
            lm.extra
            if isinstance(
                getattr(lm, "extra", {}),
                dict,
            )
            else {}
        )

        if (
            extra.get("source_image_index")
            != current_index
        ):
            continue

        category = str(
            getattr(lm, "category", "")
        ).lower()

        # Labelled doors should normally be category "door".
        # A direct entrance observation is also accepted.
        if category not in {"door", "observation"}:
            continue

        view = _infer_evidence_view_from_landmark(lm)

        if view not in {"LEFT", "RIGHT"}:
            continue

        if category == "observation":
            navigation_role = str(
                extra.get("navigation_role", "")
            ).lower()

            goal_support = str(
                extra.get("goal_support", "")
            ).lower()

            if (
                navigation_role != "entrance"
                or goal_support != "direct"
            ):
                continue

        visible_label_source = (
            str(getattr(lm, "text", "")).strip()
            or str(
                getattr(lm, "description", "")
            ).strip()
        )

        exact_match = False

        if raw_goal_codes:
            visible_code_norms = {
                _normalise_room_code(code)
                for code in _extract_room_codes(
                    visible_label_source
                )
                if _normalise_room_code(code)
            }

            # A target door should contain one exact room code,
            # not merely a room-range sign.
            exact_match = (
                len(visible_code_norms) == 1
                and bool(
                    visible_code_norms
                    & goal_code_norms
                )
            )

        elif raw_goal_norm:
            visible_norm = _normalise_room_code(
                visible_label_source
            )

            exact_match = (
                len(raw_goal_norm) >= 3
                and raw_goal_norm in visible_norm
                and str(
                    extra.get(
                        "target_relevance",
                        "",
                    )
                ).lower() == "high"
            )

        if not exact_match:
            continue

        score = float(
            getattr(
                lm,
                "evidence_score",
                0.0,
            )
            or 0.0
        )

        if score > best_score:
            best = lm
            best_score = score

    return best

def _has_explicit_direction_metadata(
    extra: dict[str, Any],
) -> bool:
    return bool(_semantic_direction_from_extra(extra))

def _clean_param(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", ""} else text

_ALLOWED_EVIDENCE_VIEWS = {
    "LEFT",
    "FRONT",
    "RIGHT",
    "STITCHED_UNKNOWN",
    "NONE",
}

def _normalise_direction(value: Any) -> str:
    text = _clean_param(value).lower()

    if not text:
        return ""

    # Supported explicit compound forms.
    if text in {
        "left",
        "turn left",
        "left_forward",
        "forward_left",
    }:
        return "left"

    if text in {
        "right",
        "turn right",
        "right_forward",
        "forward_right",
    }:
        return "right"

    if text in {
        "forward",
        "front",
        "straight",
        "go straight",
        "ahead",
        "continue",
    }:
        return "forward"

    normalized = re.sub(r"[_-]+", " ", text)

    detected: set[str] = set()

    if (
        re.search(r"\bleft\b", normalized)
        or "←" in normalized
    ):
        detected.add("left")

    if (
        re.search(r"\bright\b", normalized)
        or "→" in normalized
    ):
        detected.add("right")

    if (
        re.search(
            r"\b(forward|front|straight|ahead|continue)\b",
            normalized,
        )
        or "↑" in normalized
    ):
        detected.add("forward")

    # Reject ambiguous values such as "left, right" or "← →".
    if len(detected) != 1:
        return ""

    return next(iter(detected))

def _semantic_direction_from_extra(extra: dict[str, Any]) -> str:
    if not isinstance(extra, dict):
        return ""

    for key in (
        "target_direction",
        "arrow",
        "continuation_direction",
        "direction",
    ):
        direction = _normalise_direction(extra.get(key))
        if direction:
            return direction

    return ""

def _semantic_landmark_direction(
    landmark: "Landmark | None",
) -> str:
    if landmark is None:
        return ""

    extra = (
        landmark.extra
        if isinstance(getattr(landmark, "extra", {}), dict)
        else {}
    )

    return _semantic_direction_from_extra(extra)

def _landmark_supports_direction(
    landmark: "Landmark | None",
    requested_direction: str,
) -> bool:
    requested = _normalise_direction(requested_direction)
    if not requested or landmark is None:
        return False

    metadata_direction = _semantic_landmark_direction(landmark)
    if metadata_direction:
        return metadata_direction == requested

    # Conservative fallback for older landmarks with missing metadata.
    # Accept text only when it contains one unambiguous direction.
    text = _visible_landmark_content(landmark)

    visible_directions: set[str] = set()

    if re.search(r"\bleft\b", text) or "←" in text:
        visible_directions.add("left")

    if re.search(r"\bright\b", text) or "→" in text:
        visible_directions.add("right")

    if (
        re.search(
            r"\b(forward|straight|ahead)\b",
            text,
        )
        or "↑" in text
    ):
        visible_directions.add("forward")

    return visible_directions == {requested}

def _normalise_evidence_view(value: Any) -> str:
    text = _clean_param(value).upper()

    if text in _ALLOWED_EVIDENCE_VIEWS:
        return text

    if "LEFT" in text:
        return "LEFT"
    if "RIGHT" in text:
        return "RIGHT"
    if "FRONT" in text or "FORWARD" in text or "AHEAD" in text:
        return "FRONT"
    if "STITCH" in text or "UNKNOWN" in text:
        return "STITCHED_UNKNOWN"
    if "NONE" in text or "NO" == text:
        return "NONE"

    return ""


def _infer_evidence_view_from_landmark(lm: "Landmark | None") -> str:
    if lm is None:
        return ""

    extra = getattr(lm, "extra", {})
    if isinstance(extra, dict):
        for key in (
            "evidence_view",
            "source_view",
            "view",
            "camera",
            "panel",
            "source_camera",
        ):
            view = _normalise_evidence_view(extra.get(key))
            if view:
                return view

    text = _landmark_text(lm)

    if "left view" in text or "left panel" in text or "left camera" in text or " on the left" in text:
        return "LEFT"

    if "right view" in text or "right panel" in text or "right camera" in text or " on the right" in text:
        return "RIGHT"

    if "front view" in text or "front panel" in text or "front camera" in text or "directly ahead" in text or "ahead" in text:
        return "FRONT"

    return ""


def _infer_evidence_view_from_action(action: Action) -> str:
    params = action.params if isinstance(action.params, dict) else {}

    direction = _clean_param(params.get("direction")).lower()
    target = _clean_param(params.get("target")).lower()
    search_for = _clean_param(params.get("search_for")).lower()
    reason = str(getattr(action, "reason", "") or "").lower()

    combined = " ".join([direction, target, search_for, reason])

    if action.name in {"SEARCH_FOR_CUE", "WAIT_OR_RECOVER"}:
        return "NONE"

    if "left" in combined:
        return "LEFT"

    if "right" in combined:
        return "RIGHT"

    if (
        "front" in combined
        or "forward" in combined
        or "ahead" in combined
        or "straight" in combined
        or direction in {"forward", "front", "ahead", "straight"}
    ):
        return "FRONT"

    return "STITCHED_UNKNOWN"


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
        text = _visible_landmark_content(lm)
        extra = getattr(lm, "extra", {}) if isinstance(getattr(lm, "extra", {}), dict) else {}
        has_extra_direction = _has_explicit_direction_metadata(extra)
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

        text = _visible_landmark_content(lm)
        extra = getattr(lm, "extra", {}) if isinstance(getattr(lm, "extra", {}), dict) else {}

        has_direction = _has_explicit_direction_metadata(extra) or any(
            word in text
            for word in ["left", "right", "straight", "forward", "ahead", "up", "down", "←", "→", "↑", "↓"]
        )

        if not has_direction:
            continue

        # Prefer landmarks whose text/extra matches requested direction.
        match_bonus = 0.0
        direction_metadata = " ".join([
            _clean_param(extra.get("direction")),
            _clean_param(extra.get("arrow")),
        ]).lower()
        combined = f"{text} {direction_metadata}"
        if direction and any(part in combined for part in direction.replace("and", " ").split()):
            match_bonus = 0.25

        score = float(getattr(lm, "evidence_score", 0.0) or 0.0) + match_bonus

        if score > best_score:
            best = lm
            best_score = score

    return best

def _landmark_is_current(
    memory: "NavigationMemory",
    landmark: "Landmark | None",
) -> bool:
    if landmark is None:
        return False

    extra = getattr(landmark, "extra", {})
    if not isinstance(extra, dict):
        return False

    source_index = extra.get("source_image_index")
    return source_index == getattr(memory, "image_count", None)

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

def _visible_landmark_content(lm: "Landmark") -> str:
    return " ".join([
        str(getattr(lm, "category", "")),
        str(getattr(lm, "description", "")),
        str(getattr(lm, "text", "")),
    ]).lower()

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