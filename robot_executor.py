#!/usr/bin/env python3

"""
robot_executor.py

ThinkPad-side safe movement executor for Unitree Go1.

This file converts high-level VLM actions into very small /cmd_vel commands.

IMPORTANT:
- Run this only on the ThinkPad, not AMD server, not inside Docker.
- roscore must be running.
- unitree_legged_real twist_sub must be running.
- The robot must be in a safe open area.
- Every movement is tiny and followed by stop.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time

from robot_safety import RobotSafetyMonitor

VALID_EXECUTE_MODES = {"false", "confirm", "true"}

class SafeCmdVelExecutor(object):
    """
    Map one high-level action to one small motion pulse.

    The outer ThinkPad loop is responsible for:
    capture -> plan -> pulse -> stop -> recapture.
    """

    def __init__(
        self,
        topic="/cmd_vel",
        forward_speed=0.12,
        turn_speed=0.20,
        duration=0.40,
        forward_pulse_duration=None,
        turn_pulse_duration=None,
        approach_pulse_duration=None,
        doorway_pulse_duration=None,
        search_pulse_duration=None,
        invert_turn=False,
        execute_mode="confirm",
        min_evidence_score=0.30,
        allow_low_confidence=False,
        safety_monitor=None,
    ):
        self.topic = topic
        self.forward_speed = float(forward_speed)
        self.turn_speed = float(turn_speed)

        fallback_duration = float(duration)

        self.forward_pulse_duration = float(
            forward_pulse_duration
            if forward_pulse_duration is not None
            else fallback_duration
        )
        self.turn_pulse_duration = float(
            turn_pulse_duration
            if turn_pulse_duration is not None
            else fallback_duration
        )
        self.approach_pulse_duration = float(
            approach_pulse_duration
            if approach_pulse_duration is not None
            else self.forward_pulse_duration
        )
        self.doorway_pulse_duration = float(
            doorway_pulse_duration
            if doorway_pulse_duration is not None
            else self.forward_pulse_duration
        )
        self.search_pulse_duration = float(
            search_pulse_duration
            if search_pulse_duration is not None
            else self.turn_pulse_duration
        )
        self.invert_turn = bool(invert_turn)
        self.execute_mode = execute_mode
        self.min_evidence_score = float(
            min_evidence_score
        )
        self.allow_low_confidence = bool(
            allow_low_confidence
        )
        self.safety_monitor = (
            safety_monitor
            if safety_monitor is not None
            else RobotSafetyMonitor()
        )
        self.last_result = {
            "executed": False,
            "status": "not_started",
            "reason": "",
            "command": None,
            "safety": None,
        }
        if self.execute_mode not in VALID_EXECUTE_MODES:
            raise ValueError(
                "execute_mode must be one of: false, confirm, true"
            )

    def _set_result(
        self,
        executed,
        status,
        reason,
        command=None,
        safety=None,
    ):
        self.last_result = {
            "executed": bool(executed),
            "status": str(status),
            "reason": str(reason),
            "command": command,
            "safety": safety,
        }
        return bool(executed)

    def stop(self):
        print("[EXECUTOR] STOP")

        return_code = self._publish_once(
            0.0,
            0.0,
        )
        if return_code not in {0, 124}:
            print(
                "[EXECUTOR] Warning: stop command returned "
                "code {}.".format(return_code)
            )

    def execute_action_response(self, response):
        action = response.get("action", {})

        if not isinstance(action, dict):
            self.stop()
            return self._set_result(
                False,
                "invalid_response",
                "No valid action dictionary in API response.",
            )
        return self.execute_action(action)

    def execute_action(self, action):
        name = str(
            action.get("name", "")
        ).strip().upper()

        params = action.get("params", {})
        if not isinstance(params, dict):
            params = {}
        reason = str(
            action.get("reason", "")
        ).strip()
        confidence = str(
            action.get("confidence", "low")
        ).strip().lower()
        try:
            evidence_score = float(
                action.get("evidence_score", 0.0)
                or 0.0
            )
        except (TypeError, ValueError):
            evidence_score = 0.0

        is_valid = bool(
            action.get("is_valid", True)
        )
        if not is_valid:
            self.stop()
            return self._set_result(
                False,
                "invalid_action",
                "Action was marked invalid.",
            )

        # Phase 2.6: make allow_low_confidence functional.
        if (
            confidence == "low"
            and not self.allow_low_confidence
        ):
            self.stop()
            return self._set_result(
                False,
                "low_confidence_blocked",
                (
                    "Low-confidence action blocked. Use "
                    "--allow-low-confidence only during "
                    "controlled testing."
                ),
            )

        if evidence_score < self.min_evidence_score:
            self.stop()
            return self._set_result(
                False,
                "evidence_score_blocked",
                (
                    "Evidence score {:.3f} is below "
                    "minimum {:.3f}."
                ).format(
                    evidence_score,
                    self.min_evidence_score,
                ),
            )

        command = self._action_to_cmd(
            name,
            params,
        )

        print("\n========== EXECUTOR DECISION ==========")
        print("Action:", name)
        print("Params:", params)
        print("Reason:", reason)
        print("Mapped command:", command)
        print("Execute mode:", self.execute_mode)
        print("=======================================\n")

        if command is None:
            self.stop()
            return self._set_result(
                False,
                "no_motion_mapping",
                (
                    "Action has no safe movement mapping "
                    "or its visual prerequisites are not met."
                ),
            )
        if self.execute_mode == "false":
            return self._set_result(
                False,
                "dry_run",
                "Dry run only; no robot motion.",
                command=command,
            )
        if self.execute_mode == "confirm":
            answer = input(
                "Execute this small movement pulse? [y/N]: "
            ).strip().lower()

            if answer not in {"y", "yes"}:
                self.stop()
                return self._set_result(
                    False,
                    "user_rejected",
                    "User rejected movement.",
                    command=command,
                )
        preflight_ok, preflight_reason = (
            self._preflight_check()
        )

        if not preflight_ok:
            self.stop()
            return self._set_result(
                False,
                "ros_preflight_failed",
                preflight_reason,
                command=command,
            )
        linear_x = float(
            command.get("linear_x", 0.0)
        )
        angular_z = float(
            command.get("angular_z", 0.0)
        )
        pulse_duration = float(
            command.get("duration", 0.0)
        )
        safety = self.safety_monitor.check_motion(
            action_name=name,
            linear_x=linear_x,
            angular_z=angular_z,
            motion_kind=command.get(
                "motion_kind",
                "unknown",
            ),
        )
        if not safety.get("allowed", False):
            self.stop()
            return self._set_result(
                False,
                "safety_blocked",
                safety.get(
                    "reason",
                    "Motion blocked by safety monitor.",
                ),
                command=command,
                safety=safety,
            )
        try:
            moved = self._move_for_duration(
                linear_x,
                angular_z,
                pulse_duration,
            )
        except Exception as exc:
            self.stop()
            return self._set_result(
                False,
                "movement_exception",
                str(exc),
                command=command,
                safety=safety,
            )
        if not moved:
            return self._set_result(
                False,
                "movement_command_failed",
                "The ROS movement command failed.",
                command=command,
                safety=safety,
            )
        return self._set_result(
            True,
            "pulse_executed",
            "One small movement pulse was executed.",
            command=command,
            safety=safety,
        )

    def _action_to_cmd(self, name, params):
        direction = str(
            params.get("direction", "") or ""
        ).strip().lower()

        evidence_view = str(
            params.get("evidence_view", "") or ""
        ).strip().upper()

        horizontal_position = str(
            params.get(
                "horizontal_position",
                "unknown",
            )
            or "unknown"
        ).strip().lower()

        doorway_state = str(
            params.get(
                "doorway_state",
                "unknown",
            )
            or "unknown"
        ).strip().lower()

        threshold_state = str(
            params.get(
                "threshold_state",
                "unknown",
            )
            or "unknown"
        ).strip().lower()

        traversable = params.get("traversable")

        if name in {
            "STOP_AND_VERIFY",
            "WAIT_OR_RECOVER",
            "ASK_RECEPTION_OR_STAFF",
            "USE_ELEVATOR_OR_STAIRS",
        }:
            return None

        # Phase 3.2: one small alignment pulse.
        if name == "ALIGN_WITH_LANDMARK":
            if direction not in {"left", "right"}:
                return None

            return self._turn_cmd(
                direction,
                self.turn_pulse_duration,
                "visual_alignment",
            )

        # Phase 3.3: approach only when centred in FRONT.
        if name == "APPROACH_LANDMARK":
            if evidence_view != "FRONT":
                return None

            if horizontal_position != "center":
                return None

            return self._forward_cmd(
                self.approach_pulse_duration,
                "landmark_approach",
            )

        # Phase 3.4: doorway pulse requires explicit open/traversable evidence.
        if name == "PASS_THROUGH_DOORWAY":
            if evidence_view != "FRONT":
                return None
            if horizontal_position != "center":
                return None
            if traversable is not True:
                return None
            if doorway_state != "open":
                return None
            if threshold_state == "passed":
                return None
            return self._forward_cmd(
                self.doorway_pulse_duration,
                "doorway_traversal",
            )

        if name in {
            "READ_SIGN",
            "CHECK_DOOR_LABEL",
        }:
            if evidence_view == "LEFT":
                return self._turn_cmd(
                    "left",
                    self.turn_pulse_duration,
                    "cue_alignment",
                )
            if evidence_view == "RIGHT":
                return self._turn_cmd(
                    "right",
                    self.turn_pulse_duration,
                    "cue_alignment",
                )
            if (
                evidence_view == "FRONT"
                and horizontal_position in {
                    "left",
                    "right",
                }
            ):
                return self._turn_cmd(
                    horizontal_position,
                    self.turn_pulse_duration,
                    "cue_alignment",
                )
            if (
                evidence_view == "FRONT"
                and horizontal_position == "center"
            ):
                return self._forward_cmd(
                    self.approach_pulse_duration,
                    "cue_approach",
                )
            return None

        if name in {
            "NAVIGATE_TO_FRONTIER",
            "NAVIGATE_TO_LANDMARK",
        }:
            if direction in {"left", "right"}:
                return self._turn_cmd(
                    direction,
                    self.turn_pulse_duration,
                    "route_alignment",
                )

            if direction == "forward":
                return self._forward_cmd(
                    self.forward_pulse_duration,
                    "route_advance",
                )
            return None

        if name == "FOLLOW_DIRECTION":
            if direction in {"left", "right"}:
                return self._turn_cmd(
                    direction,
                    self.turn_pulse_duration,
                    "direction_turn",
                )

            if direction == "forward":
                return self._forward_cmd(
                    self.forward_pulse_duration,
                    "direction_advance",
                )
            return None

        if name == "SEARCH_FOR_CUE":
            return self._turn_cmd(
                "left",
                self.search_pulse_duration,
                "visual_search",
            )
        return None

    def _forward_cmd(
        self,
        duration,
        motion_kind,
    ):
        return {
            "linear_x": self.forward_speed,
            "angular_z": 0.0,
            "duration": float(duration),
            "motion_kind": str(motion_kind),
        }
    def _turn_cmd(
        self,
        direction,
        duration,
        motion_kind,
    ):
        sign = (
            1.0
            if direction == "left"
            else -1.0
        )
        if self.invert_turn:
            sign *= -1.0
        return {
            "linear_x": 0.0,
            "angular_z": sign * self.turn_speed,
            "duration": float(duration),
            "motion_kind": str(motion_kind),
        }

    def _preflight_check(self):
        if shutil.which("rostopic") is None:
            return (
                False,
                "rostopic command is not available.",
            )
        try:
            result = subprocess.run(
                [
                    "rostopic",
                    "type",
                    self.topic,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=3,
            )
        except Exception as exc:
            return (
                False,
                "ROS topic preflight failed: {}".format(
                    exc
                ),
            )
        if result.returncode != 0:
            return (
                False,
                (
                    "Could not resolve ROS topic {}: {}"
                ).format(
                    self.topic,
                    result.stderr.strip(),
                ),
            )
        topic_type = result.stdout.strip()
        if topic_type != "geometry_msgs/Twist":
            return (
                False,
                (
                    "ROS topic {} has unexpected type {}."
                ).format(
                    self.topic,
                    topic_type or "unknown",
                ),
            )
        return True, ""

    def _move_for_duration(
        self,
        linear_x,
        angular_z,
        duration,
    ):
        if duration <= 0.0:
            raise ValueError(
                "Movement duration must be positive."
            )
        print(
            "[EXECUTOR] Pulse: linear.x={}, "
            "angular.z={}, duration={}s".format(
                linear_x,
                angular_z,
                duration,
            )
        )
        command = self._rostopic_pub_cmd(
            linear_x,
            angular_z,
            rate=10,
        )
        try:
            return_code = subprocess.call(
                [
                    "timeout",
                    str(duration),
                ]
                + command
            )
            # timeout normally returns 124 because rostopic pub -r
            # is intentionally terminated after the pulse duration.
            return return_code in {0, 124}
        finally:
            self.stop()

    def _publish_once(
        self,
        linear_x,
        angular_z,
    ):
        command = self._rostopic_pub_cmd(
            linear_x,
            angular_z,
            once=True,
        )
        return subprocess.call(
            ["timeout", "2"] + command
        )

    def _rostopic_pub_cmd(
        self,
        linear_x,
        angular_z,
        rate=None,
        once=False,
    ):
        message = (
            "{linear: {x: %.4f, y: 0.0, z: 0.0}, "
            "angular: {x: 0.0, y: 0.0, z: %.4f}}"
        ) % (
            linear_x,
            angular_z,
        )
        command = [
            "rostopic",
            "pub",
        ]
        if once:
            command.append("-1")
        else:
            command.extend([
                "-r",
                str(rate or 10),
            ])
        command.extend([
            self.topic,
            "geometry_msgs/Twist",
            message,
        ])
        return command
    
def load_json_file(path):
    with open(path, "r") as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Safe /cmd_vel executor for VLM navigation actions."
    )
    parser.add_argument("--action-json", default=None, help="Path to AMD API response JSON or action JSON file.")
    parser.add_argument(
        "--execute",
        choices=["false", "confirm", "true"],
        default="confirm",
        help="false=dry run, confirm=ask before moving, true=execute directly",
    )
    parser.add_argument("--topic", default="/cmd_vel")
    parser.add_argument("--forward-speed", type=float, default=0.12)
    parser.add_argument("--turn-speed", type=float, default=0.20)
    parser.add_argument("--duration", type=float, default=0.40, help="Fallback pulse duration in seconds.")
    parser.add_argument("--forward-pulse-duration", type=float, default=0.40)
    parser.add_argument("--turn-pulse-duration", type=float, default=0.35)
    parser.add_argument("--approach-pulse-duration", type=float, default=0.30)
    parser.add_argument("--doorway-pulse-duration", type=float, default=0.25)
    parser.add_argument("--search-pulse-duration",type=float,default=0.25)
    parser.add_argument("--safety-mode", choices=["placeholder", "disabled"], default="placeholder")
    parser.add_argument(
        "--allow-motion-without-safety-sensor",
        action="store_true",
        help=(
            "Explicitly bypass the unavailable LiDAR safety "
            "check. Use only for controlled manual testing."
        ),
    )
    parser.add_argument(
        "--invert-turn",
        action="store_true",
        help="Use this if +angular.z turns the robot right instead of left.",
    )
    parser.add_argument("--stop", action="store_true", help="Only send stop command and exit.")
    parser.add_argument("--test-forward", action="store_true", help="Send tiny forward command, then stop.")
    parser.add_argument("--test-left", action="store_true", help="Send tiny left turn command, then stop.")
    parser.add_argument("--test-right", action="store_true", help="Send tiny right turn command, then stop.")
    parser.add_argument(
        "--min-evidence-score",
        type=float,
        default=0.30,
        help="Minimum evidence score required for movement",
    )
    parser.add_argument("--allow-low-confidence", action="store_true", help="Allow low-confidence actions to move the robot")
    return parser.parse_args()

def main():
    args = parse_args()

    safety_monitor = RobotSafetyMonitor(
        sensor_mode=args.safety_mode,
        allow_without_sensor=(
            args.allow_motion_without_safety_sensor
        ),
    )

    executor = SafeCmdVelExecutor(
        topic=args.topic,
        forward_speed=args.forward_speed,
        turn_speed=args.turn_speed,
        duration=args.duration,
        forward_pulse_duration=(
            args.forward_pulse_duration
        ),
        turn_pulse_duration=(
            args.turn_pulse_duration
        ),
        approach_pulse_duration=(
            args.approach_pulse_duration
        ),
        doorway_pulse_duration=(
            args.doorway_pulse_duration
        ),
        search_pulse_duration=(
            args.search_pulse_duration
        ),
        invert_turn=args.invert_turn,
        execute_mode=args.execute,
        min_evidence_score=args.min_evidence_score,
        allow_low_confidence=args.allow_low_confidence,
        safety_monitor=safety_monitor,
    )

    if args.stop:
        executor.stop()
        return

    if args.test_forward:
        executor.execute_action({
            "name": "NAVIGATE_TO_FRONTIER",
            "params": {"direction": "forward"},
            "reason": "Manual tiny forward test.",
            "confidence": "high", #manual safe tests not be blocked by VLM confidence/evidence checks.
            "evidence_score": 1.0,
            "is_valid": True,
        })
        return

    if args.test_left:
        executor.execute_action({
            "name": "FOLLOW_DIRECTION",
            "params": {"direction": "left"},
            "reason": "Manual tiny left turn test.",
            "confidence": "high", #manual safe tests not be blocked by VLM confidence/evidence checks.
            "evidence_score": 1.0,
            "is_valid": True,
        })
        return

    if args.test_right:
        executor.execute_action({
            "name": "FOLLOW_DIRECTION",
            "params": {"direction": "right"},
            "reason": "Manual tiny right turn test.",
            "confidence": "high", #manual safe tests not be blocked by VLM confidence/evidence checks.
            "evidence_score": 1.0,
            "is_valid": True,
        })
        return

    if args.action_json:
        data = load_json_file(args.action_json)

        if "action" in data:
            executor.execute_action_response(data)
        else:
            executor.execute_action(data)
        return

    print("[INFO] No action supplied. Sending stop for safety.")
    executor.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] Ctrl+C pressed.")
        try:
            SafeCmdVelExecutor(execute_mode="false").stop()
        except Exception:
            pass
        sys.exit(130)
    except Exception as exc:
        print("[ERROR]", exc)
        sys.exit(1)