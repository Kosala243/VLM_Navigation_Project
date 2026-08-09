#!/usr/bin/env python2

"""
lidar_probe.py

One-shot diagnostic tool for /rslidar_points (RSHELIOS_16P via rslidar_sdk).
`rostopic echo /rslidar_points` is unusable because PointCloud2 packs every
point into an opaque byte blob -- this script decodes it with
sensor_msgs.point_cloud2.read_points and prints summary numbers instead.

Answers, from one captured scan:
  - blind zone / self-occlusion: closest point in the forward cone
  - vertical FOV: elevation angle spread (from z)
  - horizontal FOV: azimuth angle spread (from atan2(y, x))
  - floor-band check: z (height, in the LiDAR's own frame) for points on
    the steepest-downward channel, i.e. almost certainly floor -- use this
    to pick the height-band cutoff for the obstacle filter

Run on the ThinkPad with the ROS env sourced, same as cmd_vel_bridge.py:
    source /opt/ros/melodic/setup.bash
    export ROS_MASTER_URI=http://192.168.12.212:11311
    export ROS_IP=192.168.12.212
    python2 temp_files/lidar_probe.py
"""

import math

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

FORWARD_CONE_HALF_ANGLE_DEG = 20.0
FLOOR_BAND_DEG = 2.0  # width of the "steepest downward channel" band, in degrees


def main():
    rospy.init_node("lidar_probe", anonymous=True, disable_signals=True)
    print("Waiting for one message on /rslidar_points ...")
    msg = rospy.wait_for_message(
        "/rslidar_points", PointCloud2, timeout=10.0
    )

    points = list(
        point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
    )
    n = len(points)
    print("Total points in this scan: {}".format(n))
    if n == 0:
        print("No points received - check the sensor/topic before trusting anything else.")
        return

    ranges = []
    azimuths_deg = []
    elevations_deg = []
    forward_cone_points = []  # (range, x, y, z, elevation_deg)
    all_points_full = []  # (range, x, y, z, elevation_deg, azimuth_deg)

    for x, y, z in points:
        r = math.sqrt(x * x + y * y + z * z)
        if r < 1e-6:
            continue
        azimuth_deg = math.degrees(math.atan2(y, x))
        elevation_deg = math.degrees(math.asin(z / r))
        ranges.append(r)
        azimuths_deg.append(azimuth_deg)
        elevations_deg.append(elevation_deg)
        all_points_full.append((r, x, y, z, elevation_deg, azimuth_deg))
        if abs(azimuth_deg) <= FORWARD_CONE_HALF_ANGLE_DEG:
            forward_cone_points.append((r, x, y, z, elevation_deg))

    print("\n--- Range, all points (meters) ---")
    print("min: {:.3f}   max: {:.3f}".format(min(ranges), max(ranges)))

    print("\n--- Horizontal spread / azimuth, atan2(y,x) (degrees) ---")
    print(
        "min: {:.1f}   max: {:.1f}   (should span close to "
        "-180..180 for a full 360 deg mechanical scan)".format(
            min(azimuths_deg), max(azimuths_deg)
        )
    )

    print("\n--- Vertical spread / elevation, asin(z/r) (degrees) ---")
    print(
        "min: {:.1f}   max: {:.1f}   (this is your usable vertical FOV "
        "for height-band filtering)".format(
            min(elevations_deg), max(elevations_deg)
        )
    )

    print(
        "\n--- Forward cone (+/-{:.0f} deg of straight ahead) ---".format(
            FORWARD_CONE_HALF_ANGLE_DEG
        )
    )
    if forward_cone_points:
        closest = min(forward_cone_points, key=lambda p: p[0])
        r, x, y, z, elevation_deg = closest
        print("point count: {}".format(len(forward_cone_points)))
        print("closest point: range={:.3f} m  x={:.3f}  y={:.3f}  z={:.3f}  elevation={:.1f} deg".format(
            r, x, y, z, elevation_deg
        ))
        print(
            "If z is strongly negative and elevation is near the most "
            "downward angle seen above, this is almost certainly the "
            "FLOOR being picked up by a low channel, not the robot's own "
            "body -- expected until the height-band filter excludes it. "
            "If z is instead close to 0 (near the LiDAR's own height) "
            "with a shallow elevation angle, that would point to real "
            "self-occlusion from the robot's structure instead."
        )
    else:
        print(
            "No points fell inside the forward cone at all -- check the "
            "LiDAR's mounting/orientation before trusting the azimuth "
            "convention."
        )

    min_elevation_deg = min(elevations_deg)
    floor_band_all = [
        p for p in all_points_full if p[4] <= min_elevation_deg + FLOOR_BAND_DEG
    ]
    floor_band_forward = [
        p for p in floor_band_all if abs(p[5]) <= FORWARD_CONE_HALF_ANGLE_DEG
    ]

    print(
        "\n--- Floor-band check (points within {:.1f} deg of the steepest "
        "downward elevation, {:.1f} deg) ---".format(
            FLOOR_BAND_DEG, min_elevation_deg
        )
    )
    if floor_band_all:
        z_vals = [p[3] for p in floor_band_all]
        r_vals = [p[0] for p in floor_band_all]
        print(
            "all points in band: count={}  z: min={:.3f} max={:.3f} "
            "mean={:.3f}   range: min={:.3f} max={:.3f}".format(
                len(floor_band_all), min(z_vals), max(z_vals),
                sum(z_vals) / len(z_vals), min(r_vals), max(r_vals)
            )
        )
        print(
            "(z here should be close to constant across range if this is "
            "a flat floor -- that constant is your height-band cutoff, "
            "and should be roughly -1x the measured LiDAR mounting height "
            "above the floor)"
        )
    else:
        print("No points found in the steepest-downward band.")

    if floor_band_forward:
        z_vals = [p[3] for p in floor_band_forward]
        r_vals = [p[0] for p in floor_band_forward]
        print(
            "forward-cone-only: count={}  z: min={:.3f} max={:.3f} "
            "mean={:.3f}   range: min={:.3f} max={:.3f}".format(
                len(floor_band_forward), min(z_vals), max(z_vals),
                sum(z_vals) / len(z_vals), min(r_vals), max(r_vals)
            )
        )
    else:
        print("No steepest-downward points fell inside the forward cone.")


if __name__ == "__main__":
    main()
