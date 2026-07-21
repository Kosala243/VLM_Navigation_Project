"""
robot_safety.py

ThinkPad-side motion-safety interface.

Today this provides a fail-closed placeholder. Tomorrow the real LiDAR,
odometry and robot-state checks will be connected through this interface.
"""

class RobotSafetyMonitor(object):
    VALID_MODES = {
        "placeholder",
        "disabled",
    }

    def __init__(
        self,
        sensor_mode="placeholder",
        allow_without_sensor=False,
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

    @property
    def sensor_ready(self):
        """
        This becomes True tomorrow when a real LiDAR adapter
        is connected and has received a valid recent scan.
        """
        return False

    def check_motion(
        self,
        action_name,
        linear_x,
        angular_z,
        motion_kind="unknown",
    ):
        """
        Return a dictionary describing whether motion is allowed.

        This placeholder does not claim that the route is clear.
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