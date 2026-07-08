"""navigator.py
Connects parser, memory, verifier, planner, and optional robot executor.
Works in image-sequence testing and real robot/Gazebo stepping.

Updated behaviour:
- Supports continuous-goal navigation with keep_memory=True.
- Verifies target reach using verifier.py before asking the action generator.
- Stops only when verifier/action produces a valid STOP_AND_VERIFY.
- Marks acted-on landmarks as visited/used while keeping session memory.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .action_generator import Action, ActionGenerator
from .goal_parser import GoalParser, NavigationGoal
from .memory import MemoryUpdate, NavigationMemory
from .verifier import TargetVerifier, VerificationResult


class RobotExecutor(Protocol):
    """Optional interface for ROS/Nav2/Gazebo integration."""
    def execute(self, action: Action) -> bool: ...


@dataclass
class StepRecord:
    image_num: int
    image_path: str
    memory_update: MemoryUpdate
    action: Action
    executed: bool = False
    verification: VerificationResult | None = None


@dataclass
class NavigationLog:
    raw_goal: str
    started_at: str
    ended_at: str
    success: bool
    records: list[StepRecord] = field(default_factory=list)
    final_goal: dict[str, Any] = field(default_factory=dict)
    memory_path: str = ""
    memory_kept_from_previous_goal: bool = False

    def save(self, path: str) -> None:
        data = {
            "raw_goal": self.raw_goal,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "success": self.success,
            "final_goal": self.final_goal,
            "memory_path": self.memory_path,
            "memory_kept_from_previous_goal": self.memory_kept_from_previous_goal,
            "records": [
                {
                    "image_num": r.image_num,
                    "image_path": r.image_path,
                    "memory_update": {
                        "useful": r.memory_update.useful,
                        "summary": r.memory_update.summary,
                        "landmarks": [asdict(lm) for lm in r.memory_update.landmarks],
                        "hypotheses": r.memory_update.hypotheses,
                    },
                    "verification": r.verification.to_dict() if r.verification else None,
                    "action": asdict(r.action),
                    "executed": r.executed,
                }
                for r in self.records
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def summary(self) -> str:
        lines = [
            "=" * 70,
            f"Navigation goal: {self.raw_goal}",
            f"Success: {self.success}",
            f"Memory kept from previous goal: {self.memory_kept_from_previous_goal}",
            f"Images processed: {len(self.records)}",
            "-" * 70,
        ]
        for r in self.records:
            valid = "valid" if r.action.is_valid else f"INVALID: {r.action.invalid_reason}"
            executed = "executed" if r.executed else "not-executed"
            ver = ""
            if r.verification:
                ver = (
                    f" | verified={r.verification.target_reached}"
                    f"/{r.verification.evidence_type}"
                    f"/{r.verification.matched_label or '-'}"
                    f"/v_score={getattr(r.verification, 'evidence_score', 0.0):.2f}"
                )
            lines.append(
                f"#{r.image_num} {Path(r.image_path).name}: "
                f"memory={'yes' if r.memory_update.useful else 'no'} | "
                f"action={r.action.name} | conf={r.action.confidence} "
                f"| a_score={getattr(r.action, 'evidence_score', 0.0):.2f} | "
                f"{valid} | {executed}{ver} | {r.action.reason}"
            )
        lines.append("=" * 70)
        return "\n".join(lines)


class NavigationSystem:
    def __init__(self, model, executor: RobotExecutor | None = None):
        self.model = model
        self.executor = executor
        self.goal_parser = GoalParser(model)
        self.memory = NavigationMemory(model)
        self.verifier = TargetVerifier(model)
        self.action_generator = ActionGenerator(model)
        self.goal: NavigationGoal | None = None
        self.records: list[StepRecord] = []
        self.started_at = ""
        self._memory_kept_for_current_goal = False

    def start(self, raw_goal: str, keep_memory: bool = False) -> NavigationGoal:
        """Start a new navigation goal.

        Args:
            raw_goal: New target, e.g. "C0.008".
            keep_memory: Keep existing session memory when continuing from the
                current physical location in the same building/context. Use False
                for a new session, changed start point, simulation reset, or new building.
        """
        if not keep_memory:
            self.memory.clear()

        self.records.clear()
        self.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.goal = self.goal_parser.parse(raw_goal)
        self._memory_kept_for_current_goal = keep_memory

        print("[NavigationSystem] Goal parsed:")
        print(json.dumps(self.goal.to_dict(), indent=2, ensure_ascii=False))
        print(
            "[NavigationSystem] Continuing with existing session memory."
            if keep_memory else
            "[NavigationSystem] Started with cleared memory."
        )
        return self.goal

    def clear_memory(self) -> None:
        """Explicitly clear all session memory."""
        self.memory.clear()
        print("[NavigationSystem] Memory cleared.")

    def step(
        self,
        image_path: str,
        image_paths: dict[str, str] | None = None,
        frontiers: list[dict[str, Any]] | None = None,
        execute: bool = False,
    ) -> tuple[Action, bool]:

        if self.goal is None:
            raise RuntimeError("Call start(raw_goal) before step().")

        if frontiers:
            self.memory.add_frontiers(frontiers)

        update = self.memory.update_from_image(image_path, self.goal, image_paths=image_paths)

        # Proper target verification happens before planning the next action.
        verification = self.verifier.verify(image_path, self.goal, update, image_paths=image_paths)
        if verification.target_reached:
            action = verification.to_action()
        else:
            action = self.action_generator.generate(image_path, self.goal, self.memory, image_paths=image_paths)

        if not verification.target_reached and action.name == "STOP_AND_VERIFY":
            action.is_valid = False
            action.goal_reached = False
            action.needs_verification = True
            action.confidence = "low"
            action.evidence_score = 0.0
            action.invalid_reason = (
                "STOP_AND_VERIFY rejected: verifier did not confirm current-frame "
                "target door/entrance evidence."
            )

        executed = False
        if action.name == "STOP_AND_VERIFY" and action.is_valid:
            executed = True
        elif execute and self.executor is not None and action.is_valid:
            executed = bool(self.executor.execute(action))
            if not executed:
                self.memory.add_failed_action(
                    f"Executor failed for {action.name}: {action.reason}"
                )

        if not action.is_valid:
            self.memory.add_failed_action(action.invalid_reason)

        if action.is_valid:
            self._mark_action_landmark_status(action)

        rec = StepRecord(
            image_num=len(self.records) + 1,
            image_path=str(image_path),
            memory_update=update,
            verification=verification,
            action=action,
            executed=executed,
        )
        self.records.append(rec)

        navigation_complete = (
            action.name == "STOP_AND_VERIFY"
            and action.goal_reached
            and action.is_valid
            and verification.target_reached
        )

        print(
            f"[Step {rec.image_num}] "
            f"memory={'yes' if update.useful else 'no'} | "
            f"verify={'yes' if verification.target_reached else 'no'} "
            f"({verification.evidence_type}, {verification.matched_label or '-'}, "
            f"score={getattr(verification, 'evidence_score', 0.0):.2f}) | "
            f"action={action.display()} | "
            f"complete={'yes' if navigation_complete else 'no'}"
        )
        return action, navigation_complete

    def navigate(
        self,
        raw_goal: str,
        image_sequence: list[str],
        save_dir: str = "navigation_outputs",
        keep_memory: bool = False,
    ) -> NavigationLog:
        """Run navigation over an ordered sequence of images."""
        self.start(raw_goal, keep_memory=keep_memory)
        success = False
        for image_path in image_sequence:
            if not Path(image_path).exists():
                print(f"[SKIP] Missing image: {image_path}")
                continue
            _, success = self.step(image_path)
            if success:
                break

        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        memory_path = str(out_dir / "memory.json")
        self.memory.save(memory_path)

        log = NavigationLog(
            raw_goal=raw_goal,
            started_at=self.started_at,
            ended_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            success=success,
            records=list(self.records),
            final_goal=self.goal.to_dict() if self.goal else {},
            memory_path=memory_path,
            memory_kept_from_previous_goal=keep_memory,
        )
        log_path = str(out_dir / "navigation_log.json")
        log.save(log_path)
        print(log.summary())
        print(f"Saved: {log_path}")
        return log

    def _mark_action_landmark_status(self, action: Action) -> None:
        """Mark action-related landmarks without deleting old memory."""
        lm_id = action.params.get("landmark_id")
        if not lm_id:
            return

        if action.name in {"NAVIGATE_TO_LANDMARK", "NAVIGATE_TO_FRONTIER"}:
            self.memory.mark_landmark(str(lm_id), "visited")
        elif action.name in {
            "READ_SIGN",
            "CHECK_DOOR_LABEL",
            "FOLLOW_DIRECTION",
            "ASK_RECEPTION_OR_STAFF",
            "USE_ELEVATOR_OR_STAIRS",
            "STOP_AND_VERIFY",
        }:
            self.memory.mark_landmark(str(lm_id), "used")

    def current_status(self) -> str:
        if self.goal is None:
            return "Not started."
        return (
            f"Goal={self.goal.raw_goal} | "
            f"records={len(self.records)} | "
            f"memory_landmarks={len(self.memory.landmarks)} | "
            f"keep_memory={self._memory_kept_for_current_goal}"
        )