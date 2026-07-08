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
import subprocess
import sys
import time

VALID_EXECUTE_MODES = {"false", "confirm", "true"}


class SafeCmdVelExecutor(object):
    def __init__(
        self,
        topic="/cmd_vel",
        forward_speed=0.05,
        turn_speed=0.20,
        duration=1.0,
        invert_turn=False,
        execute_mode="confirm",
        min_evidence_score=0.30,
        allow_low_confidence=False,
    ):
        self.topic = topic
        self.forward_speed = float(forward_speed)
        self.turn_speed = float(turn_speed)
        self.duration = float(duration)
        self.invert_turn = bool(invert_turn)
        self.execute_mode = execute_mode
        self.min_evidence_score = float(min_evidence_score)
        self.allow_low_confidence = bool(allow_low_confidence)

        if self.execute_mode not in VALID_EXECUTE_MODES:
            raise ValueError("execute_mode must be one of: false, confirm, true")

    def stop(self):
        print("[EXECUTOR] STOP")
        self._publish_once(0.0, 0.0)

    def execute_action_response(self, response):
        """
        Execute action from AMD API response.

        Expected response shape:
        {
          "action": {
            "name": "FOLLOW_DIRECTION",
            "params": {...},
            "reason": "...",
            "confidence": "low|medium|high",
            "evidence_score": 0.0
          }
        }
        """
        action = response.get("action", {})
        if not isinstance(action, dict):
            print("[EXECUTOR] No valid action dict in response. Sending stop.")
            self.stop()
            return False

        return self.execute_action(action)

    def execute_action(self, action):
        name = str(action.get("name", "")).strip().upper()
        params = action.get("params", {})
        reason = str(action.get("reason", "")).strip()
        confidence = str(action.get("confidence", "low")).strip().lower()
        evidence_score = float(action.get("evidence_score", 0.0) or 0.0)
        is_valid = bool(action.get("is_valid", True))

        if not is_valid:
            print("[EXECUTOR] Action is invalid. Sending stop.")
            self.stop()
            return False

        if confidence == "low" and not self.allow_low_confidence:
            print("[EXECUTOR] Low-confidence action blocked. Sending stop.")
            self.stop()
            return False

        if evidence_score < self.min_evidence_score:
            print(
                "[EXECUTOR] Evidence score too low: {:.2f} < {:.2f}. Sending stop.".format(
                    evidence_score,
                    self.min_evidence_score,
                )
            )
            self.stop()
            return False

        if not isinstance(params, dict):
            params = {}

        cmd = self._action_to_cmd(name, params)

        print("\n========== EXECUTOR DECISION ==========")
        print("Action:", name)
        print("Params:", params)
        print("Reason:", reason)
        print("Mapped command:", cmd)
        print("Execute mode:", self.execute_mode)
        print("=======================================\n")

        if cmd is None:
            print("[EXECUTOR] No movement command. Sending stop.")
            self.stop()
            return False

        if self.execute_mode == "false":
            print("[EXECUTOR] Dry run only. Not moving robot.")
            return False

        if self.execute_mode == "confirm":
            answer = input("Execute this tiny movement? [y/N]: ").strip().lower()
            if answer not in {"y", "yes"}:
                print("[EXECUTOR] User rejected movement. Sending stop.")
                self.stop()
                return False

        linear_x = float(cmd.get("linear_x", 0.0))
        angular_z = float(cmd.get("angular_z", 0.0))
        duration = float(cmd.get("duration", self.duration))

        self._move_for_duration(linear_x, angular_z, duration)
        self.stop()
        return True

    def _action_to_cmd(self, name, params):
        direction = str(params.get("direction", "") or "").strip().lower()
        target = str(params.get("target", "") or "").strip().lower()
        search_for = str(params.get("search_for", "") or "").strip().lower()

        combined = " ".join([direction, target, search_for]).lower()

        # Always stop for final verification / unsafe / unclear actions.
        if name in {"STOP_AND_VERIFY", "WAIT_OR_RECOVER"}:
            return None

        # If target is reached or action says check/read, do not move automatically.
        # Reading/checking should happen by capturing another frame after manual/assisted positioning.
        if name in {"READ_SIGN", "CHECK_DOOR_LABEL", "ASK_RECEPTION_OR_STAFF"}:
            return None

        # Move forward only for frontier/landmark when no turn direction is specified.
        if name in {"NAVIGATE_TO_FRONTIER", "NAVIGATE_TO_LANDMARK"}:
            if "left" in combined:
                return self._turn_cmd("left")
            if "right" in combined:
                return self._turn_cmd("right")
            return self._forward_cmd()

        # Follow explicit direction.
        if name == "FOLLOW_DIRECTION":
            if "left" in combined:
                return self._turn_cmd("left")
            if "right" in combined:
                return self._turn_cmd("right")
            if "back" in combined or "behind" in combined:
                return self._turn_cmd("left")
            if "straight" in combined or "front" in combined or "forward" in combined or "ahead" in combined:
                return self._forward_cmd()
            return None

        # Search by rotating a tiny amount.
        if name == "SEARCH_FOR_CUE":
            return self._turn_cmd("left")

        # Elevator/stairs should not be executed automatically in first version.
        if name == "USE_ELEVATOR_OR_STAIRS":
            return None

        return None

    def _forward_cmd(self):
        return {
            "linear_x": self.forward_speed,
            "angular_z": 0.0,
            "duration": self.duration,
        }

    def _turn_cmd(self, direction):
        sign = 1.0 if direction == "left" else -1.0

        if self.invert_turn:
            sign *= -1.0

        return {
            "linear_x": 0.0,
            "angular_z": sign * self.turn_speed,
            "duration": self.duration,
        }

    def _move_for_duration(self, linear_x, angular_z, duration):
        print(
            "[EXECUTOR] Moving: linear.x={}, angular.z={}, duration={}s".format(
                linear_x,
                angular_z,
                duration,
            )
        )

        # Publish at 10 Hz for duration seconds.
        cmd = self._rostopic_pub_cmd(linear_x, angular_z, rate=10)

        try:
            subprocess.call(["timeout", str(duration)] + cmd)
        except KeyboardInterrupt:
            print("[EXECUTOR] Interrupted. Sending stop.")
            self.stop()
            raise

    def _publish_once(self, linear_x, angular_z):
        cmd = self._rostopic_pub_cmd(linear_x, angular_z, once=True)
        subprocess.call(cmd)

    def _rostopic_pub_cmd(self, linear_x, angular_z, rate=None, once=False):
        msg = (
            "{linear: {x: %.4f, y: 0.0, z: 0.0}, "
            "angular: {x: 0.0, y: 0.0, z: %.4f}}"
        ) % (linear_x, angular_z)

        cmd = [
            "rostopic",
            "pub",
        ]

        if once:
            cmd.append("-1")
        else:
            cmd.extend(["-r", str(rate or 10)])

        cmd.extend([
            self.topic,
            "geometry_msgs/Twist",
            msg,
        ])

        return cmd


def load_json_file(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Safe /cmd_vel executor for VLM navigation actions."
    )

    parser.add_argument(
        "--action-json",
        default=None,
        help="Path to AMD API response JSON or action JSON file.",
    )

    parser.add_argument(
        "--execute",
        choices=["false", "confirm", "true"],
        default="confirm",
        help="false=dry run, confirm=ask before moving, true=execute directly",
    )

    parser.add_argument("--topic", default="/cmd_vel")
    parser.add_argument("--forward-speed", type=float, default=0.05)
    parser.add_argument("--turn-speed", type=float, default=0.20)
    parser.add_argument("--duration", type=float, default=1.0)

    parser.add_argument(
        "--invert-turn",
        action="store_true",
        help="Use this if +angular.z turns the robot right instead of left.",
    )

    parser.add_argument(
        "--stop",
        action="store_true",
        help="Only send stop command and exit.",
    )

    parser.add_argument(
        "--test-forward",
        action="store_true",
        help="Send tiny forward command, then stop.",
    )

    parser.add_argument(
        "--test-left",
        action="store_true",
        help="Send tiny left turn command, then stop.",
    )

    parser.add_argument(
        "--test-right",
        action="store_true",
        help="Send tiny right turn command, then stop.",
    )

    parser.add_argument(
        "--min-evidence-score",
        type=float,
        default=0.30,
        help="Minimum evidence score required for movement",
    )

    parser.add_argument(
        "--allow-low-confidence",
        action="store_true",
        help="Allow low-confidence actions to move the robot",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    executor = SafeCmdVelExecutor(
        topic=args.topic,
        forward_speed=args.forward_speed,
        turn_speed=args.turn_speed,
        duration=args.duration,
        invert_turn=args.invert_turn,
        execute_mode=args.execute,
        min_evidence_score=args.min_evidence_score,
        allow_low_confidence=args.allow_low_confidence,
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