"""
robot_safety.py

ThinkPad-side motion-safety interface.

"placeholder" and "disabled" remain fail-closed / bypassed as before.
"lidar" is the real obstacle-safety adapter: it spawns a persistent
Python 2 rospy bridge subprocess (lidar_bridge.py, same pattern as
robot_executor.py's PersistentCmdVelBridge for /cmd_vel) that stays
subscribed to /rslidar_points and answers on-demand "nearest range in
the forward cone, floor excluded" queries.

Odometry and full robot-state checks are still not connected here.
"""

import atexit
import json
import os
import subprocess
import sys
import time

BRIDGE_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "lidar_bridge.py",
)
BRIDGE_PYTHON2_BIN = "python2"
BRIDGE_READY_TIMEOUT_SECONDS = 10.0

# --- LiDAR calibration constants ---
# Cone half-angle: validated against real scans in temp_files/lidar_probe.py
# (good point density, correctly separated near/far real objects at +/-20 deg).
LIDAR_CONE_HALF_ANGLE_DEG = 20.0

# Floor-height cutoff (meters, in the LiDAR's own frame -- negative is
# below the sensor). PROVISIONAL: derived from the photo-measured LiDAR
# mounting height (38-43.5cm above the floor, temp_files/20260727_144133.jpg),
# not yet from a direct floor-point z reading. Replace with a value just
# above the actually-observed floor z once temp_files/lidar_probe.py's
# floor-band section has been run and read (a flat floor should show
# z roughly constant around -0.38 to -0.435 regardless of range).
LIDAR_FLOOR_HEIGHT_CUTOFF_M = -0.30

# Stop-distance threshold (meters): block a forward/backward pulse if
# the nearest in-cone, floor-excluded point is closer than this.
# PROVISIONAL placeholder -- replace with measured single-pulse travel
# distance (mark the nose position, run one forward pulse, measure the
# gap) plus a safety margin, once that measurement exists.
LIDAR_STOP_DISTANCE_M = 0.50

LIDAR_MAX_SCAN_AGE_SECONDS = 1.0


class PersistentLidarBridge(object):
    """Owns one long-lived python2 rospy subscriber subprocess.

    Mirrors robot_executor.py's PersistentCmdVelBridge: newline-delimited
    JSON requests/replies over the child's stdin/stdout, so the rospy
    node and topic subscription are paid for once, not once per
    check_motion() call.
    """

    def __init__(
        self,
        topic="/rslidar_points",
        cone_half_angle_deg=LIDAR_CONE_HALF_ANGLE_DEG,
        floor_height_cutoff_m=LIDAR_FLOOR_HEIGHT_CUTOFF_M,
        max_scan_age_seconds=LIDAR_MAX_SCAN_AGE_SECONDS,
    ):
        self.topic = topic
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
                "--cone-half-angle-deg",
                str(cone_half_angle_deg),
                "--floor-height-cutoff-m",
                str(floor_height_cutoff_m),
                "--max-scan-age-seconds",
                str(max_scan_age_seconds),
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
                    "lidar bridge process exited early (code {}): {}".format(
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
                return
            self._ready_error = msg.get("error", "unknown lidar bridge error")
            return
        self._ready_error = "timed out waiting for lidar bridge readiness"

    @property
    def is_ready(self):
        return self._ready and self._proc.poll() is None

    @property
    def ready_error(self):
        return self._ready_error

    def _send(self, request):
        if self._proc.poll() is not None:
            return {"ok": False, "error": "lidar bridge process is not running"}
        try:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
        except Exception as exc:
            return {
                "ok": False,
                "error": "failed to send to lidar bridge: {}".format(exc),
            }

        line = self._proc.stdout.readline()
        if not line:
            return {"ok": False, "error": "lidar bridge process closed its output"}
        try:
            return json.loads(line.strip())
        except Exception as exc:
            return {
                "ok": False,
                "error": "bad reply from lidar bridge: {}".format(exc),
            }

    def get_nearest_range(self):
        return self._send({"cmd": "get_range"})

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


class RobotSafetyMonitor(object):
    VALID_MODES = {
        "placeholder",
        "disabled",
        "lidar",
    }

    def __init__(
        self,
        sensor_mode="placeholder",
        allow_without_sensor=False,
        lidar_topic="/rslidar_points",
    ):
        self.sensor_mode = str(
            sensor_mode or "placeholder"
        ).strip().lower()

        if self.sensor_mode not in self.VALID_MODES:
            raise ValueError(
                "sensor_mode must be one of: {}".format(
                    ", ".join(sorted(self.VALID_MODES))
                )
            )

        self.allow_without_sensor = bool(
            allow_without_sensor
        )

        self._lidar_bridge = None
        if self.sensor_mode == "lidar":
            try:
                self._lidar_bridge = PersistentLidarBridge(topic=lidar_topic)
                atexit.register(self._lidar_bridge.shutdown)
            except Exception as exc:
                # Spawn failure is not fatal here -- it surfaces through
                # sensor_ready/check_motion the same way an unready
                # bridge does, so allow_without_sensor still governs
                # whether motion is permitted.
                self._lidar_bridge = None
                self._lidar_spawn_error = str(exc)
            else:
                self._lidar_spawn_error = None

    @property
    def sensor_ready(self):
        """
        True once a real sensor adapter is connected and has usable data.

        "placeholder" and "disabled" never claim the route is clear.
        "lidar" is ready once its bridge subprocess is alive and reported
        a successful scan at startup.
        """
        if self.sensor_mode == "lidar":
            return (
                self._lidar_bridge is not None
                and self._lidar_bridge.is_ready
            )
        return False

    def _check_lidar_clearance(self, action_name, motion_kind):
        if self._lidar_bridge is None or not self._lidar_bridge.is_ready:
            error = (
                getattr(self, "_lidar_spawn_error", None)
                or (
                    self._lidar_bridge.ready_error
                    if self._lidar_bridge is not None
                    else "lidar bridge not started"
                )
            )
            if self.allow_without_sensor:
                return {
                    "allowed": True,
                    "sensor_ready": False,
                    "sensor_mode": self.sensor_mode,
                    "bypassed": True,
                    "reason": (
                        "Motion allowed through explicit manual override. "
                        "LiDAR bridge is not ready: {}".format(error)
                    ),
                    "action_name": str(action_name),
                    "motion_kind": str(motion_kind),
                }
            return {
                "allowed": False,
                "sensor_ready": False,
                "sensor_mode": self.sensor_mode,
                "bypassed": False,
                "reason": "Motion blocked: LiDAR bridge is not ready: {}".format(
                    error
                ),
                "action_name": str(action_name),
                "motion_kind": str(motion_kind),
            }

        reply = self._lidar_bridge.get_nearest_range()
        if not reply.get("ok"):
            if self.allow_without_sensor:
                return {
                    "allowed": True,
                    "sensor_ready": True,
                    "sensor_mode": self.sensor_mode,
                    "bypassed": True,
                    "reason": (
                        "Motion allowed through explicit manual override. "
                        "LiDAR read failed: {}".format(reply.get("error"))
                    ),
                    "action_name": str(action_name),
                    "motion_kind": str(motion_kind),
                }
            return {
                "allowed": False,
                "sensor_ready": True,
                "sensor_mode": self.sensor_mode,
                "bypassed": False,
                "reason": "Motion blocked: LiDAR read failed: {}".format(
                    reply.get("error")
                ),
                "action_name": str(action_name),
                "motion_kind": str(motion_kind),
            }

        nearest = reply.get("range")
        if nearest is not None and nearest < LIDAR_STOP_DISTANCE_M:
            return {
                "allowed": False,
                "sensor_ready": True,
                "sensor_mode": self.sensor_mode,
                "bypassed": False,
                "reason": (
                    "Motion blocked: nearest obstacle at {:.2f}m, "
                    "below the {:.2f}m stop distance.".format(
                        nearest, LIDAR_STOP_DISTANCE_M
                    )
                ),
                "action_name": str(action_name),
                "motion_kind": str(motion_kind),
                "nearest_range_m": nearest,
            }

        return {
            "allowed": True,
            "sensor_ready": True,
            "sensor_mode": self.sensor_mode,
            "bypassed": False,
            "reason": (
                "Nearest obstacle at {:.2f}m, clear.".format(nearest)
                if nearest is not None
                else "No obstacle detected within sensor range."
            ),
            "action_name": str(action_name),
            "motion_kind": str(motion_kind),
            "nearest_range_m": nearest,
        }

    def check_motion(
        self,
        action_name,
        linear_x,
        angular_z,
        motion_kind="unknown",
    ):
        """
        Return a dictionary describing whether motion is allowed.
        """
        linear_x = float(linear_x)
        angular_z = float(angular_z)

        movement_requested = (
            abs(linear_x) > 1e-6
            or abs(angular_z) > 1e-6
        )

        if not movement_requested:
            return {
                "allowed": True,
                "sensor_ready": self.sensor_ready,
                "sensor_mode": self.sensor_mode,
                "bypassed": False,
                "reason": "No movement requested.",
            }

        # LiDAR only gates forward/backward pulses (nonzero linear_x).
        # Pure turns are skipped by the minimal design -- the forward
        # cone doesn't characterize what a turn sweeps through, and
        # gating turns too was deferred until proven necessary. This
        # must be its own branch, not a fall-through into the
        # placeholder/disabled logic below, otherwise a pure turn under
        # "lidar" mode would get blocked by the "sensor not connected"
        # message even though the sensor is connected and simply isn't
        # consulted for turns.
        if self.sensor_mode == "lidar":
            if abs(linear_x) > 1e-6:
                return self._check_lidar_clearance(action_name, motion_kind)
            return {
                "allowed": True,
                "sensor_ready": self.sensor_ready,
                "sensor_mode": self.sensor_mode,
                "bypassed": False,
                "reason": "Pure turn - not gated by the LiDAR forward-cone check.",
                "action_name": str(action_name),
                "motion_kind": str(motion_kind),
            }

        if self.allow_without_sensor:
            return {
                "allowed": True,
                "sensor_ready": False,
                "sensor_mode": self.sensor_mode,
                "bypassed": True,
                "reason": (
                    "Motion allowed through explicit manual override. "
                    "No LiDAR or robot-state safety check was performed."
                ),
                "action_name": str(action_name),
                "motion_kind": str(motion_kind),
            }

        return {
            "allowed": False,
            "sensor_ready": False,
            "sensor_mode": self.sensor_mode,
            "bypassed": False,
            "reason": (
                "Motion blocked because the real obstacle-safety "
                "sensor adapter is not connected."
            ),
            "action_name": str(action_name),
            "motion_kind": str(motion_kind),
        }
