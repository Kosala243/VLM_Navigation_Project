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
import atexit
import json
import math
import os
import subprocess
import sys
import time

from robot_safety import RobotSafetyMonitor, LIDAR_STOP_DISTANCE_M

VALID_EXECUTE_MODES = {"false", "confirm", "true"}

# LiDAR-scaled forward pulses (only for genuine forward-advance motion
# kinds, not the deliberately-small precision pulses like landmark
# approach or doorway traversal, which should keep re-imaging often as
# they close in rather than lunge toward their target in one step).
# PROVISIONAL ceiling -- how far a single pulse may travel even when
# the LiDAR sees a fully open path ahead; tune once this has been
# tried on real hardware.
MAX_FORWARD_PULSE_DISTANCE_M = 0.60
MIN_SCALED_PULSE_DURATION_S = 0.05
LIDAR_SCALABLE_MOTION_KINDS = {"route_advance", "direction_advance"}

# A scaled forward pulse longer than this is chunked into sub-pulses of
# this size, with a fresh LiDAR clearance check before each one, instead
# of one long blind cmd_vel hold -- so something entering the path
# mid-pulse gets caught at the next chunk boundary instead of only after
# the whole pulse finishes. Matches the existing forward_pulse_duration
# default, i.e. the same "one blind increment" size already used
# elsewhere in this file, not a new arbitrary number.
MOTION_WATCH_CHUNK_SECONDS = 0.40

# Turn magnitude tiers (#1). Every turn used to share one fixed duration
# regardless of whether it was a fine alignment nudge or a "reorient
# toward a new heading" turn, which was implicated in the left/right
# hunting behavior observed during real runs: a fixed pulse either
# overshoots a small needed correction or undershoots a large one.
# "Fine" turns (ALIGN_WITH_LANDMARK, cue_alignment) keep using
# turn_pulse_duration unchanged. "Exploratory" turns (route_alignment,
# direction_turn, and by extension visual_search) get their own, larger
# duration. PROVISIONAL: ~15 degrees at the default turn_speed=0.20 rad/s
# (0.2618 rad / 0.20 rad/s), not yet validated on hardware.
EXPLORATION_TURN_PULSE_DURATION_DEFAULT_S = 1.30

# Vision-grounded turning (#2). Converts the VLM's own estimate of a
# landmark/cue's horizontal position within its source camera image
# (params.horizontal_offset_fraction, -1.0=left edge of that image to
# +1.0=right edge, 0.0=center) into an actual turn angle, instead of the
# fixed magnitude-tier duration. Only applies to landmark/cue alignment
# turns (ALIGN_WITH_LANDMARK, cue_alignment) -- there's nothing to
# ground against for route/direction/search turns, which stay on the
# fixed tiers from #1.
#
# This pipeline captures three separate cameras in --camera-mode
# separate (the mode actually used for real runs) -- front/left/right
# images sent to the model individually, at native capture resolution,
# no stitching/resizing. params.evidence_view (LEFT/FRONT/RIGHT) says
# which of the three images the offset fraction is relative to.
#
# PROVISIONAL: every value below is a placeholder pending physical
# measurement on the real hardware (wall-distance method for FOV,
# known-bearing-marker method for boresight -- see the lab checklist).
# Sign convention matches the LiDAR azimuth convention already used
# elsewhere in this repo: positive degrees = left, negative = right.
CAMERA_HORIZONTAL_FOV_DEG = {
    "FRONT": 60.0,
    "LEFT": 60.0,
    "RIGHT": 60.0,
}
CAMERA_BORESIGHT_DEG = {
    "FRONT": 0.0,
    "LEFT": 90.0,
    "RIGHT": -90.0,
}
# Clamp against a wildly-off single estimate (miscalibration or a bad
# model guess) turning into an oversized blind turn.
MAX_VISION_TURN_ANGLE_DEG = 90.0

BRIDGE_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cmd_vel_bridge.py",
)
BRIDGE_PYTHON2_BIN = "python2"
BRIDGE_READY_TIMEOUT_SECONDS = 8.0
BRIDGE_REPLY_TIMEOUT_SECONDS = 6.0
DEFAULT_PUBLISH_RATE_HZ = 10.0

# One bridge subprocess per topic, shared across every SafeCmdVelExecutor
# instance in this process. ROS node registration is the expensive part
# of talking to /cmd_vel (on the order of a second, sometimes more under
# load); paying it once per run instead of once per pulse is the entire
# point of the bridge. Callers construct a new SafeCmdVelExecutor per
# step/action, so this cache is what makes those all share one node.
_SHARED_BRIDGES = {}


class PersistentCmdVelBridge(object):
    """Owns one long-lived python2 rospy publisher subprocess.

    robot_executor.py runs under Python 3, but this ROS Melodic install
    only has rospy for Python 2, so the actual rospy.Publisher lives in
    a separate cmd_vel_bridge.py child process. Requests/replies are
    newline-delimited JSON over the child's stdin/stdout. Because the
    child's rospy node is created exactly once at startup, every pulse
    after that is just a fast local IPC round-trip plus in-process
    rospy.Rate() timing in the child -- no more per-pulse ROS
    registration, and no more racing an external `timeout` kill against
    rospy.init_node() completing.
    """

    def __init__(self, topic="/cmd_vel", rate=DEFAULT_PUBLISH_RATE_HZ):
        self.topic = topic
        self.rate = rate
        self._ready = False
        self._ready_error = None
        self._closed = False

        self._proc = subprocess.Popen(
            [
                BRIDGE_PYTHON2_BIN,
                "-u",
                BRIDGE_SCRIPT_PATH,
                "--topic",
                topic,
                "--rate",
                str(rate),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        )
        self._wait_ready()

    def _wait_ready(self):
        deadline = time.time() + BRIDGE_READY_TIMEOUT_SECONDS
        while time.time() < deadline:
            if self._proc.poll() is not None:
                stderr = self._proc.stderr.read()
                self._ready_error = (
                    "bridge process exited early (code {}): {}".format(
                        self._proc.returncode, stderr.strip()
                    )
                )
                return
            line = self._proc.stdout.readline()
            if not line:
                continue
            try:
                msg = json.loads(line.strip())
            except Exception:
                continue
            if msg.get("ready"):
                self._ready = True
                self._connections = msg.get("connections", 0)
                if self._connections == 0:
                    print(
                        "[BRIDGE] WARNING: no subscriber connected to "
                        "{} yet; is twist_sub running?".format(self.topic)
                    )
                return
            self._ready_error = msg.get("error", "unknown bridge error")
            return
        self._ready_error = "timed out waiting for bridge readiness"

    @property
    def is_ready(self):
        return self._ready and self._proc.poll() is None

    @property
    def ready_error(self):
        return self._ready_error

    def _send(self, request):
        if self._proc.poll() is not None:
            return {"ok": False, "error": "bridge process is not running"}
        try:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
        except Exception as exc:
            return {
                "ok": False,
                "error": "failed to send to bridge: {}".format(exc),
            }

        line = self._proc.stdout.readline()
        if not line:
            return {"ok": False, "error": "bridge process closed its output"}
        try:
            return json.loads(line.strip())
        except Exception as exc:
            return {
                "ok": False,
                "error": "bad reply from bridge: {}".format(exc),
            }

    def pulse(self, linear_x, angular_z, duration):
        return self._send({
            "cmd": "pulse",
            "linear_x": linear_x,
            "angular_z": angular_z,
            "duration": duration,
            "rate": self.rate,
        })

    def stop(self):
        return self._send({"cmd": "stop"})

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        if self._proc.poll() is None:
            try:
                self._send({"cmd": "quit"})
            except Exception:
                pass
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=3.0)
            except Exception:
                self._proc.kill()


def _get_shared_bridge(topic, rate=DEFAULT_PUBLISH_RATE_HZ):
    bridge = _SHARED_BRIDGES.get(topic)
    if bridge is not None and bridge.is_ready:
        return bridge
    bridge = PersistentCmdVelBridge(topic=topic, rate=rate)
    _SHARED_BRIDGES[topic] = bridge
    atexit.register(bridge.shutdown)
    return bridge


def _coerce_bool(value, default=True):
    """Coerce a possibly-string action field into a real bool.

    Actions should carry real JSON booleans, but a malformed or
    partially string-coerced action must not let a string like "false"
    evaluate to True via a bare bool(value) call.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"false", "no", "0", ""}:
            return False
        if s in {"true", "yes", "1"}:
            return True
        return default
    return bool(value)


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
        exploration_turn_pulse_duration=None,
        approach_pulse_duration=None,
        doorway_pulse_duration=None,
        search_pulse_duration=None,
        invert_turn=False,
        execute_mode="confirm",
        min_evidence_score=0.30,
        allow_low_confidence=True,
        safety_monitor=None,
    ):
        self.topic = topic
        self.publish_rate = DEFAULT_PUBLISH_RATE_HZ
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
        self.exploration_turn_pulse_duration = float(
            exploration_turn_pulse_duration
            if exploration_turn_pulse_duration is not None
            else EXPLORATION_TURN_PULSE_DURATION_DEFAULT_S
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
            # SEARCH_FOR_CUE is an exploratory scan, not a fine
            # correction, so it defaults alongside route/direction
            # turns rather than the fine-alignment tier.
            else self.exploration_turn_pulse_duration
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

    def _bridge(self):
        return _get_shared_bridge(self.topic, self.publish_rate)

    def stop(self):
        """Publish a zero-velocity command and report whether it succeeded."""
        print("[EXECUTOR] STOP")

        bridge = self._bridge()
        if not bridge.is_ready:
            print(
                "[EXECUTOR] CRITICAL: cmd_vel bridge is not ready ({}); "
                "cannot confirm the robot has stopped.".format(
                    bridge.ready_error
                )
            )
            return False

        result = bridge.stop()
        stop_ok = bool(result.get("ok"))
        if not stop_ok:
            print(
                "[EXECUTOR] CRITICAL: stop command failed ({}); "
                "the robot may still be moving.".format(
                    result.get("error")
                )
            )
        return stop_ok

    def _abort(self, status, reason, command=None, safety=None):
        """Stop the robot and report an abort result.

        If the stop command itself fails to confirm a halt, the reported
        status/reason is escalated so callers cannot mistake this for a
        normal, safely-stopped rejection.
        """
        print("[EXECUTOR] ABORT ({}): {}".format(status, reason))
        stop_ok = self.stop()
        if not stop_ok:
            status = "stop_command_failed"
            reason = (
                "{} STOP command also failed to confirm a halt; the "
                "robot may still be executing its last motion command."
            ).format(reason)
        return self._set_result(
            False,
            status,
            reason,
            command=command,
            safety=safety,
        )

    def execute_action_response(self, response):
        action = response.get("action", {})

        if not isinstance(action, dict):
            return self._abort(
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
        if confidence not in {"high", "medium", "low"}:
            # Fail closed: an unrecognised/garbled confidence value must
            # not be treated as sufficiently confident to move.
            confidence = "low"
        try:
            evidence_score = float(
                action.get("evidence_score", 0.0)
                or 0.0
            )
        except (TypeError, ValueError):
            evidence_score = 0.0

        is_valid = _coerce_bool(
            action.get("is_valid", True),
            default=True,
        )
        if not is_valid:
            return self._abort(
                "invalid_action",
                "Action was marked invalid.",
            )

        # Phase 2.6: make allow_low_confidence functional.
        if (
            confidence == "low"
            and not self.allow_low_confidence
        ):
            return self._abort(
                "low_confidence_blocked",
                (
                    "Low-confidence action blocked. Use "
                    "--allow-low-confidence only during "
                    "controlled testing."
                ),
            )

        if evidence_score < self.min_evidence_score:
            return self._abort(
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
            return self._abort(
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
                return self._abort(
                    "user_rejected",
                    "User rejected movement.",
                    command=command,
                )
        bridge = self._bridge()
        if not bridge.is_ready:
            return self._abort(
                "ros_preflight_failed",
                "cmd_vel bridge is not ready: {}".format(
                    bridge.ready_error
                ),
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
            return self._abort(
                "safety_blocked",
                safety.get(
                    "reason",
                    "Motion blocked by safety monitor.",
                ),
                command=command,
                safety=safety,
            )

        motion_kind = command.get("motion_kind", "unknown")
        lidar_scaled = (
            self.safety_monitor.sensor_mode == "lidar"
            and motion_kind in LIDAR_SCALABLE_MOTION_KINDS
            and abs(linear_x) > 1e-6
        )
        if lidar_scaled:
            nearest = safety.get("nearest_range_m")
            safe_distance = (
                MAX_FORWARD_PULSE_DISTANCE_M
                if nearest is None
                else min(
                    MAX_FORWARD_PULSE_DISTANCE_M,
                    max(0.0, nearest - LIDAR_STOP_DISTANCE_M),
                )
            )
            pulse_duration = max(
                MIN_SCALED_PULSE_DURATION_S,
                safe_distance / abs(linear_x),
            )
            # Keep the logged/returned command in sync with what actually
            # runs, not the pre-scaling nominal duration it started as.
            command["duration"] = pulse_duration

        # Any pulse longer than one chunk is watched -- run in short,
        # re-checked segments rather than one long blind hold. For a
        # LiDAR-scaled forward pulse this is a real obstacle recheck each
        # chunk; for a turn (vision-grounded or the fixed exploration
        # tier) there's no sensor that can cancel it early today, but
        # each chunk still ends in a confirmed stop(), so the worst-case
        # single uninterrupted blind segment is bounded to one chunk
        # instead of the whole multi-second turn. Anything at or under
        # one chunk keeps the original single-shot behavior unchanged.
        if pulse_duration > MOTION_WATCH_CHUNK_SECONDS + 1e-6:
            try:
                watch_result = self._move_for_duration_watched(
                    linear_x,
                    angular_z,
                    pulse_duration,
                    motion_kind,
                )
            except Exception as exc:
                return self._abort(
                    "movement_exception",
                    str(exc),
                    command=command,
                    safety=safety,
                )
            command["elapsed_duration"] = watch_result["elapsed"]
            watch_safety = watch_result.get("safety") or safety
            if watch_result["pulse_failed"]:
                return self._set_result(
                    False,
                    "movement_command_failed",
                    "The ROS movement command failed mid-pulse.",
                    command=command,
                    safety=watch_safety,
                )
            if watch_result["cut_short"]:
                return self._set_result(
                    True,
                    "pulse_cut_short",
                    "Pulse stopped early after {:.2f}s of "
                    "{:.2f}s planned -- {}".format(
                        watch_result["elapsed"],
                        pulse_duration,
                        watch_safety.get("reason", "obstacle detected"),
                    ),
                    command=command,
                    safety=watch_safety,
                )
            return self._set_result(
                True,
                "pulse_executed",
                "One watched pulse was executed.",
                command=command,
                safety=watch_safety,
            )

        try:
            moved = self._move_for_duration(
                linear_x,
                angular_z,
                pulse_duration,
            )
        except Exception as exc:
            return self._abort(
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
        horizontal_offset_fraction = params.get("horizontal_offset_fraction")

        if name in {
            "STOP_AND_VERIFY",
            "WAIT_OR_RECOVER",
            "ASK_RECEPTION_OR_STAFF",
            "USE_ELEVATOR_OR_STAIRS",
        }:
            return None

        # Phase 3.2: one small alignment pulse. Vision-grounded (#2) when
        # the model supplied a usable horizontal_offset_fraction, else
        # falls back to the fixed fine-alignment tier from #1.
        if name == "ALIGN_WITH_LANDMARK":
            vision_turn = self._vision_grounded_turn(
                evidence_view, horizontal_offset_fraction
            )
            if vision_turn is not None:
                vg_direction, vg_duration = vision_turn
                return self._turn_cmd(
                    vg_direction,
                    vg_duration,
                    "visual_alignment",
                )

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
            # Vision-grounded (#2) when the model supplied a usable
            # horizontal_offset_fraction for this evidence_view, else
            # falls back to the fixed fine-alignment tier from #1.
            vision_turn = self._vision_grounded_turn(
                evidence_view, horizontal_offset_fraction
            )
            if evidence_view == "LEFT":
                if vision_turn is not None:
                    vg_direction, vg_duration = vision_turn
                    return self._turn_cmd(vg_direction, vg_duration, "cue_alignment")
                return self._turn_cmd(
                    "left",
                    self.turn_pulse_duration,
                    "cue_alignment",
                )
            if evidence_view == "RIGHT":
                if vision_turn is not None:
                    vg_direction, vg_duration = vision_turn
                    return self._turn_cmd(vg_direction, vg_duration, "cue_alignment")
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
                if vision_turn is not None:
                    vg_direction, vg_duration = vision_turn
                    return self._turn_cmd(vg_direction, vg_duration, "cue_alignment")
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
                    self.exploration_turn_pulse_duration,
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
                    self.exploration_turn_pulse_duration,
                    "direction_turn",
                )

            if direction == "forward":
                return self._forward_cmd(
                    self.forward_pulse_duration,
                    "direction_advance",
                )
            return None

        if name == "SEARCH_FOR_CUE":
            search_direction = (
                direction
                if direction in {"left", "right"}
                else "left"
            )
            return self._turn_cmd(
                search_direction,
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

    def _vision_grounded_turn(self, evidence_view, horizontal_offset_fraction):
        """Convert a VLM-estimated in-frame offset into (direction, duration).

        Returns None if evidence_view/horizontal_offset_fraction aren't
        usable (missing field, wrong type, NaN, or an evidence_view this
        repo has no camera calibration for) -- callers fall back to their
        fixed magnitude-tier duration in that case.
        """
        if evidence_view not in CAMERA_HORIZONTAL_FOV_DEG:
            return None
        try:
            offset = float(horizontal_offset_fraction)
        except (TypeError, ValueError):
            return None
        if offset != offset:  # NaN
            return None
        offset = max(-1.0, min(1.0, offset))

        fov_deg = CAMERA_HORIZONTAL_FOV_DEG[evidence_view]
        boresight_deg = CAMERA_BORESIGHT_DEG[evidence_view]
        # offset=-1.0 is the left edge of that camera's own image, which
        # needs a *more positive* (more left, in this repo's left=positive
        # convention) turn to center it -- hence the negation here.
        target_relative_angle_deg = boresight_deg - offset * (fov_deg / 2.0)

        magnitude_deg = min(
            MAX_VISION_TURN_ANGLE_DEG,
            abs(target_relative_angle_deg),
        )
        direction = "left" if target_relative_angle_deg >= 0 else "right"
        duration = max(
            MIN_SCALED_PULSE_DURATION_S,
            math.radians(magnitude_deg) / self.turn_speed,
        )
        return direction, duration

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

        bridge = self._bridge()
        if not bridge.is_ready:
            raise RuntimeError(
                "cmd_vel bridge is not ready: {}".format(
                    bridge.ready_error
                )
            )

        result = bridge.pulse(linear_x, angular_z, duration)
        pulse_ok = bool(result.get("ok"))
        if not pulse_ok:
            print(
                "[EXECUTOR] Pulse failed: {}".format(
                    result.get("error")
                )
            )

        # The bridge always zeroes velocity itself at the end of a pulse,
        # including on its own internal failure, but this explicit stop()
        # is what this method relies on to *confirm* a halt: if it fails,
        # the robot may still be executing the last nonzero /cmd_vel, so
        # that must not be treated as a normal successful pulse.
        stop_ok = self.stop()

        if not stop_ok:
            raise RuntimeError(
                "STOP command failed to confirm halt after the movement "
                "pulse; the robot may still be moving."
            )
        if not pulse_ok:
            raise RuntimeError(
                result.get("error") or "Pulse command failed."
            )
        return pulse_ok

    def _move_for_duration_watched(
        self,
        linear_x,
        angular_z,
        total_duration,
        motion_kind,
    ):
        """Run a long pulse (forward or turn) as short, re-checked chunks.

        A single blind cmd_vel hold for several seconds means nothing
        changing mid-pulse gets noticed until the whole pulse finishes.
        Instead this re-queries the safety monitor before every
        MOTION_WATCH_CHUNK_SECONDS-sized chunk and stops early if it
        stops being allowed -- the robot is already halted (each chunk
        ends in a confirmed stop()) the moment a chunk decides not to
        start the next one.

        For a forward/backward pulse under "lidar" mode this is a real
        obstacle recheck (see RobotSafetyMonitor._check_lidar_clearance).
        For a pure turn, check_motion() always allows it by design (pure
        turns aren't LiDAR-gated), so this degrades to bounding the
        maximum blind turn segment and creating pause points -- not an
        automatic cancel condition, since no sensor exists today that
        can tell a turn to stop early. Still meaningfully safer than one
        uninterrupted multi-second blind spin.
        """
        remaining = total_duration
        elapsed = 0.0
        last_safety = None

        while remaining > 1e-6:
            last_safety = self.safety_monitor.check_motion(
                action_name="MOTION_WATCH_RECHECK",
                linear_x=linear_x,
                angular_z=angular_z,
                motion_kind=motion_kind,
            )
            if not last_safety.get("allowed", False):
                print(
                    "[EXECUTOR] Motion watch: cutting pulse short after "
                    "{:.2f}s of {:.2f}s planned -- {}".format(
                        elapsed,
                        total_duration,
                        last_safety.get("reason", "obstacle detected"),
                    )
                )
                return {
                    "completed": False,
                    "cut_short": True,
                    "pulse_failed": False,
                    "elapsed": elapsed,
                    "safety": last_safety,
                }

            chunk = min(MOTION_WATCH_CHUNK_SECONDS, remaining)
            moved = self._move_for_duration(linear_x, angular_z, chunk)
            if not moved:
                return {
                    "completed": False,
                    "cut_short": False,
                    "pulse_failed": True,
                    "elapsed": elapsed,
                    "safety": last_safety,
                }
            elapsed += chunk
            remaining -= chunk

        return {
            "completed": True,
            "cut_short": False,
            "pulse_failed": False,
            "elapsed": elapsed,
            "safety": last_safety,
        }

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
    parser.add_argument(
        "--exploration-turn-pulse-duration",
        type=float,
        default=EXPLORATION_TURN_PULSE_DURATION_DEFAULT_S,
        help=(
            "Turn duration for reorienting turns (route/direction turns, "
            "and by default SEARCH_FOR_CUE) as opposed to fine alignment "
            "corrections, which keep using --turn-pulse-duration."
        ),
    )
    parser.add_argument("--approach-pulse-duration", type=float, default=0.30)
    parser.add_argument("--doorway-pulse-duration", type=float, default=0.25)
    parser.add_argument(
        "--search-pulse-duration",
        type=float,
        default=EXPLORATION_TURN_PULSE_DURATION_DEFAULT_S,
    )
    parser.add_argument("--safety-mode", choices=["placeholder", "disabled", "lidar"], default="placeholder")
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
        exploration_turn_pulse_duration=(
            args.exploration_turn_pulse_duration
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