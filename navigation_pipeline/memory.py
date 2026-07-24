"""memory.py
    Structured navigation memory bank for generalized indoor navigation.

    The memory stores two kinds of navigation-useful evidence:

    1. Semantic landmarks:
    signs, door labels, directories, elevators, stairs, reception/help desks,
    room-number trends, and target-relevant observations.

    2. Structural navigation landmarks:
    corridors, corridor bends, junctions, passages, open doorways,
    reachable frontiers, dead ends, and visible route continuations.

    Structural landmarks describe where the robot can move. They do not automatically
    override semantic directional evidence. The action planner must use them only when
    their direction and role are compatible with the active goal or subgoal.

    Random people are never stored as navigation landmarks. Official help sources such
    as reception, information desks, front desks, and security desks may be stored.
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

    # True means the model response was successfully parsed and
    # validated as a current-frame memory payload.
    parse_ok: bool = True

    # Populated only when current-frame perception failed.
    error: str = ""
    parse_attempts: int = 1

    # Truncated diagnostic response. Do not use this for planning.
    raw_response: str = ""

@dataclass
class NavigationSubgoal:
    description: str = ""
    landmark_id: str = ""
    landmark_category: str = ""
    direction: str = ""
    source: str = ""          # semantic | structural
    status: str = "inactive"  # inactive | active | completed
    image_index: int = -1

class NavigationMemory:
    """Structured memory bank updated from every robot camera image."""

    VALID_CATEGORIES = {
        # Semantic landmarks
        "sign",
        "door",
        "reception",
        "directory",
        "stairs",
        "elevator",
        "observation",

        # Structural navigation landmarks
        "corridor",
        "corridor_bend",
        "junction",
        "doorway",
        "passage",
        "frontier",
        "dead_end",
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
        - Keep camera position and route direction strictly separate.
        - extra.source_view and extra.horizontal_position describe where the landmark appears in the cameras. They do NOT determine the route direction.
        - For an exact target door or entrance, source_view determines whether visual alignment is required.
        - For a directional sign or directory, target_direction must come only from the arrow or directional text associated with the active navigation goal.
        - A sign visible in FRONT may direct the robot left, right, or forward.
        - A sign visible in LEFT or RIGHT may still contain an arrow pointing in a different route direction.
        - Never set target_direction=forward merely because source_view=FRONT.
        - Never set target_direction=left or right merely because the sign appears in the LEFT or RIGHT camera.
        - If useful navigation cues are visible only on one side, record that side in source_view.
        - If no useful navigation evidence is visible, return useful=false with no landmarks.
        - Do not guess room numbers or signs that are not clearly visible.
        - Be careful with blurry text, reflections, glass, and overexposed regions.
        - If a signboard, room-range sign, directory, or door label is visible but the text is too small/blurry/unclear to read, still create a landmark for it.
        - Set "text" to an empty string if the exact text is unreadable.
        - Mention in the description that the cue is visible but unreadable, and include whether it is in LEFT, FRONT, or RIGHT view.
        - Such landmarks are useful because the robot can move closer to read them.
        - Prefer current visual evidence over old memory.
        - For every landmark, set extra.source_view to LEFT, FRONT, RIGHT, STITCHED_UNKNOWN, or NONE.
        - For every visible landmark, set extra.horizontal_position to "left", "center", "right", or "unknown" relative to its own source image.
        - Use "center" only when the landmark is approximately within the middle third of the image.
        - For every visible landmark, set extra.proximity to "far", "medium", "near", "reached", or "unknown".
        - Use "reached" only when the robot is immediately at the target/entrance and should not move forward again.
        - A target that is merely readable in FRONT is not automatically reached.
        - Set extra.path_clear_visual to true, false, or null.
        - Visual path clearance is advisory only and does not replace LiDAR.
        - For a sign containing multiple destinations or multiple arrows, identify the single direction associated with the active navigation goal.
        - Store that goal-specific direction in extra.target_direction.
        - extra.target_direction must be exactly left, right, forward, none, or unknown.
        - Do not store combined values such as "left, right" in target_direction.
        - extra.arrow may preserve the raw visible arrow information.
        - For a multi-destination sign, identify the arrow associated specifically with the active goal, room range, building, tower, zone, or floor.
        - If the target-associated arrow cannot be identified confidently, set target_direction="unknown" or "none".
        - extra.arrow should be normalized to left, right, forward, none, or unknown whenever possible.
        - For semantic signs and directories, do not copy source_view into continuation_direction.
        - continuation_direction is primarily for structural landmarks such as corridors, bends, junctions, doorways, and passages.
        - A room range or destination name without a visible associated arrow does not establish a route direction.
        - Do not create category="observation" for an ordinary wall, floor, corridor, doorway, passage, or decorative feature. Use the correct structural category or omit it.

        Structural navigation landmark rules:
        - Store visible navigable structures even when they do not contain readable text.
        - Useful structural landmarks include:
            - a corridor continuing forward;
            - a corridor bending left or right;
            - an open doorway or passage;
            - a corridor junction or intersection;
            - a reachable unexplored path;
            - a visible dead end;
            - an opening at the end of a corridor.
        - Do not store ordinary walls, furniture, floor texture, or decorative openings.
        - A structure is useful only when it can affect future robot movement.
        - Create separate landmarks for distinct useful structures visible in LEFT, FRONT, and RIGHT.
        - Do not return only one "best" structure. Return all clearly visible, navigation-relevant structures.
        - If the same camera view contains both a doorway and a corridor, store both when they represent different possible routes.
        - For doorways and passages, set extra.doorway_state to "open", "closed", "blocked", or "unknown".
        - For doorways and passages, set extra.threshold_state to "before", "at", "passed", or "unknown".
        
        For every landmark, set extra.landmark_type:
        - "semantic" for signs, labelled doors, directories, reception, stairs, elevators, and target observations.
        - "structural" for corridors, bends, junctions, doorways, passages, frontiers, and dead ends.
        
        For structural landmarks, set:
        - extra.navigation_role: "continue_route | turn_point | branch | entrance | exit | exploration | dead_end"
        - extra.traversable: true, false, or null when uncertain
        - extra.continuation_direction: "left | right | forward | left_forward | right_forward | none | unknown"
        - extra.route_state: "visible_now | remembered | reached | passed | blocked"
        - extra.goal_support: "direct | indirect | none | unknown"

        Important route-selection rule:
        - Do not treat structural attractiveness as proof that a route leads toward the goal.
        - An open glass door is not automatically better than a corridor.
        - Semantic directional evidence, such as a sign arrow, takes priority over structural landmarks.
        - Structural landmarks support execution of an already justified direction or help when no semantic cue is currently visible.
        - If a directional sign says left and both a corridor and a glass doorway are visible on the left, record both landmarks separately. Do not decide which one is correct in memory extraction.
        - Use LEFT when the landmark/cue is visible in the left image/panel.
        - Use FRONT when the landmark/cue is visible in the front image/panel.
        - Use RIGHT when the landmark/cue is visible in the right image/panel.
        - Use STITCHED_UNKNOWN when the stitched panel/source is unclear.
        - Use NONE only when no current visual cue supports the landmark.

        Extract ONLY navigation-useful evidence. Ignore furniture, ceiling, wall colour, general room appearance, and random people unless they are part of an official help desk.

        Look for semantic landmarks:
        - directional signs, arrows, room ranges, building/zone/floor signs
        - labelled doors, door labels, and room plates
        - directories, maps, and "you are here" boards
        - elevators, stairs, and floor indicators
        - reception desks, information desks, front desks, security desks, or official help counters
        - visible evidence that confirms or rejects the target goal

        Look for structural navigation landmarks:
        - corridors continuing forward, left, or right
        - corridor bends visible near or at the end of a hallway
        - junctions, intersections, and branching paths
        - open doorways and passable entrances
        - open passages between areas
        - reachable unexplored paths or frontiers
        - visible dead ends or blocked continuations
        - openings at the end of otherwise visually empty corridors

        Important rule about people:
        - Do NOT create a landmark for students, visitors, pedestrians, or random people in corridors/classrooms/labs.
        - Only create category "reception" when there is clearly an official reception/front-desk/information/security/help-desk context.

        - Return at most 8 landmarks.
        - Keep every description under 30 words.
        - Include only navigation-relevant landmarks.
        - Do not include landmark IDs. Python assigns IDs after parsing.
        - Return one JSON object only, with no markdown, explanation, or trailing text.
        Return ONLY valid JSON:
        {
        "useful": true/false,
        "summary": "one short note, or empty string",
        "landmarks": [
            {
            "category": "sign | door | reception | directory | stairs | elevator | observation | corridor | corridor_bend | junction | doorway | passage | frontier | dead_end",
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
                "source_view": "LEFT | FRONT | RIGHT | STITCHED_UNKNOWN | NONE",
                "horizontal_position": "left | center | right | unknown",
                "proximity": "far | medium | near | reached | unknown",
                "path_clear_visual": true,
                "doorway_state": "open | closed | blocked | unknown",
                "threshold_state": "before | at | passed | unknown",
                "landmark_type": "semantic | structural",
                "navigation_role": "continue_route | turn_point | branch | entrance | exit | exploration | dead_end | none",
                "traversable": true,
                "continuation_direction": "left | right | forward | left_forward | right_forward | none | unknown",
                "route_state": "visible_now | remembered | reached | passed | blocked",
                "goal_support": "direct | indirect | none | unknown",
                "target_direction": "left | right | forward | none | unknown"
            }
            }
        ],
        "hypotheses": ["short useful inference, e.g. room numbers increase forward"]
        }

        - For semantic landmarks, structural fields may use null, "none", or "unknown".
        - For structural landmarks, text should normally be empty unless readable text is physically attached to that structure.

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
        self.current_subgoal = NavigationSubgoal()

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
        """
        Extract and store current-frame navigation memory.

        A failed parse is explicitly reported through parse_ok=False.
        The caller must not plan movement from stale memory when this occurs.
        """
        self.image_count += 1

        if not Path(image_path).exists():
            message = f"Image not found: {image_path}"

            return MemoryUpdate(
                useful=False,
                summary=message,
                parse_ok=False,
                error=message,
                parse_attempts=0,
            )

        prompt = self._PROMPT.replace(
            "{goal_context}",
            goal.compact(),
        )

        retry_suffix = """
            IMPORTANT RETRY REQUIREMENTS:
            - Return one complete, compact JSON object only.
            - Return at most 6 landmarks.
            - Keep descriptions short.
            - Do not use markdown fences.
            - Do not add commentary before or after the JSON.
            - Ensure every opened brace, bracket, and quote is closed.
        """

        responses: list[str] = []
        data: dict[str, Any] | None = None
        parse_attempts = 0

        # First perform the normal extraction. Retry once with stricter,
        # shorter-output instructions if parsing or schema validation fails.
        for extra_prompt in ("", retry_suffix):
            parse_attempts += 1

            response = self.model.query(
                prompt + extra_prompt,
                image_path=image_path,
                image_paths=image_paths,
                max_new_tokens=1600,
            )

            responses.append(str(response))

            candidate = _extract_json(response)

            if _valid_memory_payload(candidate):
                data = candidate
                break

        if data is None:
            last_response = (
                responses[-1]
                if responses
                else ""
            )

            message = (
                "Current-frame memory extraction failed after "
                f"{parse_attempts} parse attempt(s)."
            )

            return MemoryUpdate(
                useful=False,
                summary=message,
                parse_ok=False,
                error=message,
                parse_attempts=parse_attempts,
                raw_response=last_response[:4000],
            )

        useful = bool(data.get("useful", False))
        summary = str(
            data.get("summary", "")
        ).strip()

        # New landmarks are appended to long-term memory.
        new_landmarks: list[Landmark] = []

        # Observed landmarks includes both newly created landmarks and
        # existing landmarks re-observed in this current frame.
        observed_landmarks: list[Landmark] = []

        raw_landmarks = data.get("landmarks", [])

        if not isinstance(raw_landmarks, list):
            raw_landmarks = []

        for raw_lm in raw_landmarks[:8]:
            if not isinstance(raw_lm, dict):
                continue

            landmark = self._build_landmark(
                raw_lm,
                image_path,
                goal,
            )

            if landmark is None:
                continue

            existing = self._find_duplicate_landmark(
                landmark
            )

            if existing is not None:
                existing.description = landmark.description
                existing.text = (
                    landmark.text
                    or existing.text
                )
                existing.image_path = landmark.image_path
                existing.confidence = landmark.confidence
                existing.evidence_score = (
                    landmark.evidence_score
                )
                existing.evidence_breakdown = dict(
                    landmark.evidence_breakdown
                )
                existing.extra.update(
                    landmark.extra
                )
                existing.observation_count += 1

                # Important: the verifier must receive a re-observed
                # existing landmark as current-frame evidence.
                observed_landmarks.append(existing)
                continue

            new_landmarks.append(landmark)
            observed_landmarks.append(landmark)

        raw_hypotheses = data.get(
            "hypotheses",
            [],
        )

        if not isinstance(raw_hypotheses, list):
            raw_hypotheses = []

        new_hypotheses = [
            str(item).strip()
            for item in raw_hypotheses
            if str(item).strip()
        ]

        if (
            useful
            or observed_landmarks
            or summary
            or new_hypotheses
        ):
            useful = True

            if summary:
                self.observation_summaries.append(
                    f"Image {self.image_count}: {summary}"
                )

            self.landmarks.extend(new_landmarks)

            for hypothesis in new_hypotheses:
                self._add_hypothesis(
                    hypothesis
                )

            self._add_room_sequence_hypotheses()
            self._trim_memory()

        self.update_current_subgoal()

        return MemoryUpdate(
            useful=useful,
            summary=summary,
            landmarks=observed_landmarks,
            hypotheses=new_hypotheses,
            parse_ok=True,
            error="",
            parse_attempts=parse_attempts,
            raw_response="",
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

    def record_landmark_selection(
        self,
        landmark_id: str,
    ) -> None:
        """
        Record that the planner selected the landmark.

        This does not mean the robot successfully acted on it.
        """
        for lm in self.landmarks:
            if str(lm.id) == str(landmark_id):
                lm.selection_count += 1
                return


    def mark_landmark(
        self,
        landmark_id: str,
        status: str,
    ) -> None:
        """
        Change route/use status only after confirmed execution.
        """
        valid_status = {
            "unvisited",
            "visited",
            "used",
            "ignored",
        }

        if status not in valid_status:
            return

        for lm in self.landmarks:
            if str(lm.id) == str(landmark_id):
                lm.status = status
                return

    def add_failed_action(self, reason: str) -> None:
        if reason:
            self.failed_actions.append(reason)
            self.failed_actions = self.failed_actions[-self.max_failed_actions:]

    def update_current_subgoal(self) -> None:
        """
        Select the best navigation subgoal.

        Priority:
        1. Current target-relevant semantic directional landmark.
        2. Current traversable structural landmark.
        3. Keep the previous active subgoal when it is still usable.
        4. Recent remembered structural landmark.
        """
        current_index = self.image_count
        current_semantic: list[Landmark] = []
        current_structural: list[Landmark] = []
        remembered_structural: list[Landmark] = []
        structural_categories = {
            "corridor",
            "corridor_bend",
            "junction",
            "doorway",
            "passage",
        }
        for lm in self.landmarks:
            extra = (
                lm.extra
                if isinstance(getattr(lm, "extra", {}), dict)
                else {}
            )
            category = str(getattr(lm, "category", "")).lower()
            status = str(getattr(lm, "status", "")).lower()
            source_index = extra.get("source_image_index")
            route_state = str(
                extra.get("route_state", "visible_now")
            ).lower()
            is_current = source_index == current_index
            if status == "ignored":
                continue
            if route_state in {"blocked", "passed"}:
                continue
            # Current semantic directional evidence.
            if (
                is_current
                and category in {"sign", "directory"}
            ):
                relevance = str(
                    extra.get("target_relevance", "none")
                ).lower()

                direction = _semantic_direction_from_extra(extra)
                if (
                    relevance in {"high", "medium"}
                    and direction
                ):
                    current_semantic.append(lm)
            # Current structural evidence.
            if (
                is_current
                and category in structural_categories
                and extra.get("traversable") is not False
            ):
                current_structural.append(lm)
            # Recent remembered structural evidence.
            if (
                not is_current
                and category in structural_categories
                and status not in {"visited", "used"}
                and extra.get("traversable") is not False
            ):
                try:
                    age = current_index - int(source_index)
                except (TypeError, ValueError):
                    continue

                if 0 < age <= 3:
                    remembered_structural.append(lm)
        if current_semantic:
            selected = max(
                current_semantic,
                key=lambda lm: float(
                    getattr(lm, "evidence_score", 0.0) or 0.0
                ),
            )
            extra = selected.extra
            direction = _semantic_direction_from_extra(extra)
            self.current_subgoal = NavigationSubgoal(
                description=str(selected.description),
                landmark_id=str(selected.id),
                landmark_category=str(selected.category),
                direction=direction,
                source="semantic",
                status="active",
                image_index=current_index,
            )
            return
        if current_structural:
            selected = max(
                current_structural,
                key=lambda lm: _structural_subgoal_score(lm),
            )
            extra = selected.extra
            self.current_subgoal = NavigationSubgoal(
                description=str(selected.description),
                landmark_id=str(selected.id),
                landmark_category=str(selected.category),
                direction=_normalise_subgoal_direction(
                    extra.get("continuation_direction")
                    or extra.get("direction")
                ),
                source="structural",
                status="active",
                image_index=current_index,
            )
            return

        # Keep the previous subgoal only when its landmark is still usable.
        previous_id = self.current_subgoal.landmark_id

        if (
            self.current_subgoal.status == "active"
            and previous_id
        ):
            previous = next(
                (
                    lm
                    for lm in self.landmarks
                    if str(lm.id) == str(previous_id)
                ),
                None,
            )
            if previous is not None:
                previous_extra = (
                    previous.extra
                    if isinstance(previous.extra, dict)
                    else {}
                )

                previous_category = str(
                    previous.category
                ).lower()

                previous_is_structural = (
                    previous_category in structural_categories
                    or str(
                        previous_extra.get(
                            "landmark_type",
                            "",
                        )
                    ).lower() == "structural"
                )

                # Only retain a previous structural route anchor.
                # Old semantic signs must not remain active after they
                # disappear from the current observation.
                if (
                    previous_is_structural
                    and self._structural_landmark_usable_for_planner(
                        previous
                    )
                ):
                    return
                
        if remembered_structural:
            selected = max(
                remembered_structural,
                key=lambda lm: _structural_subgoal_score(lm),
            )

            extra = selected.extra
            self.current_subgoal = NavigationSubgoal(
                description=str(selected.description),
                landmark_id=str(selected.id),
                landmark_category=str(selected.category),
                direction=_normalise_subgoal_direction(
                    extra.get("continuation_direction")
                    or extra.get("direction")
                ),
                source="structural",
                status="active",
                image_index=current_index,
            )
            return
        self.current_subgoal = NavigationSubgoal()

    def context_for_planner(
        self,
        n_recent: int = 4,
        n_relevant: int = 4,
    ) -> str:
        """Return compact navigation memory for the action planner."""

        recent: list[Landmark] = []
        structural_categories = {
            "corridor",
            "corridor_bend",
            "junction",
            "doorway",
            "passage",
            "frontier",
            "dead_end",
        }
        for lm in reversed(self.landmarks):
            extra = lm.extra if isinstance(lm.extra, dict) else {}
            is_structural = (
                lm.category in structural_categories
                or extra.get("landmark_type") == "structural"
            )
            if (
                is_structural
                and not self._structural_landmark_usable_for_planner(lm)
            ):
                continue
            recent.append(lm)
            if len(recent) >= n_recent:
                break
        recent = list(reversed(recent))
        relevant = self._target_relevant_landmarks(n_relevant)
        structural = self._recent_structural_landmarks(limit=6)

        seen_ids = {lm.id for lm in recent}
        relevant = [lm for lm in relevant if lm.id not in seen_ids]

        def compact_landmark(lm: Landmark) -> dict[str, Any]:
            extra = lm.extra if isinstance(lm.extra, dict) else {}

            return {
                "id": lm.id,
                "category": lm.category,
                "description": lm.description[:240],
                "text": lm.text[:160],
                "status": lm.status,
                "confidence": lm.confidence,
                "evidence_score": round(float(lm.evidence_score), 3),
                "observation_count": lm.observation_count,
                "source_view": extra.get("source_view", "NONE"),
                "horizontal_position": extra.get("horizontal_position", "unknown"),
                "proximity": extra.get("proximity", "unknown"),
                "path_clear_visual": extra.get("path_clear_visual"),
                "doorway_state": extra.get("doorway_state", "unknown"),
                "threshold_state": extra.get("threshold_state", "unknown"),
                "direction": extra.get("direction"),
                "arrow": extra.get("arrow"),
                "target_direction": extra.get("target_direction"),
                "room_range": extra.get("room_range"),
                "target_relevance": extra.get("target_relevance"),
                "floor": extra.get("floor"),
                "zone": extra.get("zone"),
                "landmark_type": extra.get("landmark_type", "semantic"),
                "navigation_role": extra.get("navigation_role", "none"),
                "traversable": extra.get("traversable"),
                "continuation_direction": extra.get(
                    "continuation_direction",
                    extra.get("direction"),
                ),
                "route_state": extra.get("route_state", "visible_now"),
                "goal_support": extra.get("goal_support", "unknown"),
                "source_image_index": extra.get("source_image_index"),
            }

        data = {
            "recent_observations": [
                str(item)[:300]
                for item in self.observation_summaries[-4:]
            ],
            "recent_landmarks": [
                compact_landmark(lm)
                for lm in recent
            ],
            "target_relevant_landmarks": [
                compact_landmark(lm)
                for lm in relevant
            ],
            "current_beliefs": [
                str(item)[:240]
                for item in self.hypotheses[-4:]
            ],
            "failed_actions": [
                str(item)[:240]
                for item in self.failed_actions[-3:]
            ],
            "remembered_route_landmarks": [
                compact_landmark(lm)
                for lm in structural
            ],
            "current_subgoal": (
                {
                    "description": self.current_subgoal.description,
                    "landmark_id": self.current_subgoal.landmark_id,
                    "category": self.current_subgoal.landmark_category,
                    "direction": self.current_subgoal.direction,
                    "source": self.current_subgoal.source,
                    "status": self.current_subgoal.status,
                    "image_index": self.current_subgoal.image_index,
                }
                if self.current_subgoal.status == "active"
                else None
            ),
        }
        return json.dumps(
            data,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def save(self, path: str) -> None:
        data = {
            "image_count": self.image_count,
            "landmarks": [asdict(lm) for lm in self.landmarks],
            "observation_summaries": self.observation_summaries,
            "hypotheses": self.hypotheses,
            "failed_actions": self.failed_actions,
            "current_subgoal": asdict(self.current_subgoal),
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
        self.current_subgoal = NavigationSubgoal()
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
        extra["source_view"] = _normalise_source_view(extra.get("source_view"))
        extra["horizontal_position"] = _normalise_horizontal_position(extra.get("horizontal_position"))
        extra["proximity"] = _normalise_proximity(extra.get("proximity"))
        extra["doorway_state"] = _normalise_doorway_state(extra.get("doorway_state"))
        extra["threshold_state"] = _normalise_threshold_state(extra.get("threshold_state"))

        if extra.get("path_clear_visual") not in {
            True,
            False,
            None,
        }:
            extra["path_clear_visual"] = None

        pose = raw_lm.get("pose", {}) if isinstance(raw_lm.get("pose", {}), dict) else {}

        category = self._normalise_category(category, description, text)
        if category is None:
            return None
        
        structural_categories = {
            "corridor",
            "corridor_bend",
            "junction",
            "doorway",
            "passage",
            "frontier",
            "dead_end",
        }

        semantic_categories = {
            "sign",
            "door",
            "reception",
            "directory",
            "stairs",
            "elevator",
            "observation",
        }

        if category in structural_categories:
            extra.setdefault("landmark_type", "structural")
            extra.setdefault("navigation_role", _default_navigation_role(category))
            extra.setdefault("traversable", None)
            extra.setdefault(
                "continuation_direction",
                extra.get("direction") or "unknown",
            )
            extra.setdefault("route_state", "visible_now")
            extra.setdefault("goal_support", "unknown")

        elif category in semantic_categories:
            extra.setdefault("landmark_type", "semantic")
            extra.setdefault("navigation_role", "none")
            extra.setdefault("traversable", None)
            extra.setdefault("continuation_direction", "none")
            extra.setdefault("route_state", "visible_now")
            extra.setdefault("goal_support", "unknown")

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
            # Semantic aliases
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

            # Structural aliases
            "hallway": "corridor",
            "hall": "corridor",
            "corridor_turn": "corridor_bend",
            "hallway_bend": "corridor_bend",
            "bend": "corridor_bend",
            "intersection": "junction",
            "crossroad": "junction",
            "branch": "junction",
            "open_door": "doorway",
            "open_doorway": "doorway",
            "entrance": "doorway",
            "exit": "doorway",
            "opening": "passage",
            "open_passage": "passage",
            "path": "frontier",
            "unexplored_path": "frontier",
            "blocked_path": "dead_end",
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

    def _target_relevant_landmarks(
        self,
        limit: int,
    ) -> list[Landmark]:
        relevant: list[Landmark] = []

        structural_categories = {
            "corridor",
            "corridor_bend",
            "junction",
            "doorway",
            "passage",
            "frontier",
            "dead_end",
        }
        for lm in reversed(self.landmarks):
            extra = (
                lm.extra
                if isinstance(lm.extra, dict)
                else {}
            )
            is_structural = (
                lm.category in structural_categories
                or str(
                    extra.get("landmark_type", "")
                ).lower() == "structural"
            )
            if (
                is_structural
                and not self._structural_landmark_usable_for_planner(
                    lm
                )
            ):
                continue
            relevance = str(
                extra.get("target_relevance", "")
            ).lower()
            if relevance in {"high", "medium"}:
                relevant.append(lm)
            elif (
                lm.category
                in {
                    "directory",
                    "sign",
                    "door",
                    "reception",
                }
                and lm.text
            ):
                relevant.append(lm)
            if len(relevant) >= limit:
                break

        return list(reversed(relevant))
    
    def _structural_landmark_usable_for_planner(
        self,
        landmark: Landmark,
        max_age_images: int = 3,
    ) -> bool:
        extra = (
            landmark.extra
            if isinstance(landmark.extra, dict)
            else {}
        )

        status = str(landmark.status).lower()
        route_state = str(
            extra.get("route_state", "visible_now")
        ).lower()

        source_index = extra.get("source_image_index")
        is_current = source_index == self.image_count

        # A landmark that is currently visible may be reused even when it
        # previously had a used/visited status. A stale one may not.
        if status == "ignored":
            return False
        if (
            not is_current
            and status in {"visited", "used"}
        ):
            return False

        if route_state in {"passed", "blocked"}:
            return False

        if extra.get("traversable") is False:
            return False

        if source_index is not None:
            try:
                age = self.image_count - int(source_index)
            except (TypeError, ValueError):
                return False

            if age > max_age_images:
                return False

        return True

    def _recent_structural_landmarks(
        self,
        limit: int = 6,
    ) -> list[Landmark]:
        structural_categories = {
            "corridor",
            "corridor_bend",
            "junction",
            "doorway",
            "passage",
            "frontier",
            "dead_end",
        }

        result: list[Landmark] = []

        for lm in reversed(self.landmarks):
            extra = lm.extra if isinstance(lm.extra, dict) else {}

            is_structural = (
                lm.category in structural_categories
                or extra.get("landmark_type") == "structural"
            )

            if not is_structural:
                continue

            if not self._structural_landmark_usable_for_planner(lm):
                continue

            result.append(lm)

            if len(result) >= limit:
                break

        return list(reversed(result))

    def _trim_memory(self) -> None:
        self.landmarks = self.landmarks[-self.max_landmarks:]
        self.observation_summaries = self.observation_summaries[-self.max_summaries:]
        self.hypotheses = self.hypotheses[-self.max_hypotheses:]
        self.failed_actions = self.failed_actions[-self.max_failed_actions:]

def _normalise_source_view(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {
        "LEFT",
        "FRONT",
        "RIGHT",
        "STITCHED_UNKNOWN",
        "NONE",
    }:
        return text

    return "NONE"

def _normalise_horizontal_position(
    value: Any,
) -> str:
    text = str(value or "").strip().lower()

    if text in {
        "left",
        "center",
        "right",
        "unknown",
    }:
        return text

    if text == "centre":
        return "center"

    return "unknown"

def _normalise_proximity(value: Any) -> str:
    text = str(value or "").strip().lower()

    if text in {
        "far",
        "medium",
        "near",
        "reached",
        "unknown",
    }:
        return text

    return "unknown"

def _normalise_doorway_state(
    value: Any,
) -> str:
    text = str(value or "").strip().lower()

    if text in {
        "open",
        "closed",
        "blocked",
        "unknown",
    }:
        return text

    return "unknown"

def _normalise_threshold_state(
    value: Any,
) -> str:
    text = str(value or "").strip().lower()

    if text in {
        "before",
        "at",
        "passed",
        "unknown",
    }:
        return text

    return "unknown"

def _score_landmark(
    category: str,
    text: str,
    confidence: str,
    extra: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Compute robot-side evidence quality for a memory landmark."""
    category_scores = {
        # Semantic evidence
        "door": 0.80,
        "sign": 0.70,
        "directory": 0.65,
        "elevator": 0.60,
        "stairs": 0.60,
        "reception": 0.55,
        "observation": 0.25,

        # Structural navigation evidence
        "corridor_bend": 0.52,
        "junction": 0.50,
        "corridor": 0.48,
        "doorway": 0.45,
        "passage": 0.45,
        "frontier": 0.35,
        "dead_end": 0.55,
    }
    confidence_scores = {"high": 0.10, "medium": 0.05, "low": 0.00}

    base = category_scores.get(category, 0.25)
    confidence_bonus = confidence_scores.get(str(confidence).lower(), 0.05)
    text_bonus = 0.10 if str(text).strip() else 0.0

    target_relevance = str(extra.get("target_relevance", "")).lower() if isinstance(extra, dict) else ""
    target_bonus = 0.10 if target_relevance == "high" else 0.05 if target_relevance == "medium" else 0.0
    direction_bonus = 0.0 
    if isinstance(extra, dict):
        if _semantic_direction_from_extra(extra):
            direction_bonus = 0.05
    traversable_bonus = 0.0
    if isinstance(extra, dict):
        if extra.get("traversable") is True:
            traversable_bonus = 0.05
        elif extra.get("traversable") is False:
            traversable_bonus = -0.15

    route_role_bonus = 0.0
    if isinstance(extra, dict):
        navigation_role = str(extra.get("navigation_role", "")).lower()

        if navigation_role in {"continue_route", "turn_point"}:
            route_role_bonus = 0.05
        elif navigation_role == "dead_end":
            # Dead ends are useful to remember, but must never be selected
            # as approach targets.
            route_role_bonus = 0.0
    partial_marker_penalty = 0.35 if isinstance(extra, dict) and extra.get("partial_goal_marker") else 0.0
    irrelevant_sign_penalty = _irrelevant_sign_penalty(category, text, extra)

    if partial_marker_penalty and target_bonus > 0.05:
        target_bonus = 0.05

    missing_text_penalty = 0.0
    if category in {"door", "sign", "directory"} and not str(text).strip():
        missing_text_penalty = 0.35

    score = (
        base
        + confidence_bonus
        + text_bonus
        + target_bonus
        + direction_bonus
        + traversable_bonus
        + route_role_bonus
        - missing_text_penalty
        - partial_marker_penalty
        - irrelevant_sign_penalty
    )
    score = max(0.0, min(1.0, score))

    breakdown = {
        "category_score": base,
        "confidence_bonus": confidence_bonus,
        "text_bonus": text_bonus,
        "target_relevance_bonus": target_bonus,
        "direction_bonus": direction_bonus,
        "traversable_bonus": traversable_bonus,
        "route_role_bonus": route_role_bonus,
        "missing_text_penalty": missing_text_penalty,
        "partial_marker_penalty": partial_marker_penalty,
        "irrelevant_sign_penalty": irrelevant_sign_penalty,
        "final_score": score,
    }
    return score, breakdown

def _normalise_subgoal_direction(
    value: Any,
) -> str:
    text = str(value or "").strip().lower()

    if not text:
        return ""

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
        "ahead",
        "continue",
        "go straight",
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

    if len(detected) != 1:
        return ""

    return next(iter(detected))

def _structural_subgoal_score(lm: Landmark) -> float:
    extra = (
        lm.extra
        if isinstance(getattr(lm, "extra", {}), dict)
        else {}
    )
    score = float(
        getattr(lm, "evidence_score", 0.0) or 0.0
    )
    role = str(
        extra.get("navigation_role", "")
    ).lower()

    if role in {"continue_route", "turn_point"}:
        score += 0.20
    elif role == "branch":
        score += 0.10
    elif role == "entrance":
        goal_support = str(
            extra.get("goal_support", "unknown")
        ).lower()

        if goal_support in {"direct", "indirect"}:
            score += 0.10
        else:
            score -= 0.20
    if extra.get("traversable") is True:
        score += 0.10
    return score

def _default_navigation_role(category: str) -> str:
    roles = {
        "corridor": "continue_route",
        "corridor_bend": "turn_point",
        "junction": "branch",
        "doorway": "entrance",
        "passage": "continue_route",
        "frontier": "exploration",
        "dead_end": "dead_end",
    }
    return roles.get(category, "none")

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

def _semantic_direction_from_extra(
    extra: dict[str, Any],
) -> str:
    """
    Return a semantic route direction only when the explicit
    semantic fields do not contradict one another.

    continuation_direction is intentionally excluded because it
    belongs primarily to structural route landmarks.
    """
    if not isinstance(extra, dict):
        return ""

    directions = []

    for key in (
        "target_direction",
        "arrow",
        "direction",
    ):
        direction = _normalise_subgoal_direction(
            extra.get(key)
        )

        if direction:
            directions.append(direction)

    if not directions:
        return ""

    unique = set(directions)

    if len(unique) != 1:
        return ""

    return directions[0]

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


def _extract_json(
    text: str,
) -> dict[str, Any] | None:
    """
    Find and decode the first complete JSON object.

    This tolerates markdown fences and accidental text before or
    after the object, but it does not fabricate missing JSON.
    """
    cleaned = re.sub(
        r"```(?:json)?",
        "",
        str(text),
        flags=re.IGNORECASE,
    ).replace("```", "").strip()

    decoder = json.JSONDecoder()

    for match in re.finditer(r"\{", cleaned):
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


def _valid_memory_payload(
    data: Any,
) -> bool:
    if not isinstance(data, dict):
        return False

    landmarks = data.get(
        "landmarks",
        [],
    )
    hypotheses = data.get(
        "hypotheses",
        [],
    )

    return (
        isinstance(landmarks, list)
        and isinstance(hypotheses, list)
    )

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