#!/usr/bin/env python2

"""
lidar_bridge.py

Persistent ROS subscriber child process for /rslidar_points.

WHY THIS FILE EXISTS: same reason as cmd_vel_bridge.py -- ROS Melodic
only ships rospy for Python 2, but robot_safety.py runs under Python 3.
This script is spawned once, as a long-lived subprocess, by
robot_safety.py's PersistentLidarBridge. It performs rospy.init_node()
and creates its Subscriber exactly once for the whole run, keeps the
most recently received PointCloud2 message in memory, and answers
"nearest forward-cone range" requests over stdin/stdout by filtering
that stored message on demand -- no per-check ROS registration cost,
no per-check wait for a fresh scan.

Filtering, mirrors temp_files/lidar_probe.py's validated geometry:
  - forward cone: |atan2(y, x)| <= cone-half-angle-deg
  - floor exclusion: points with z <= floor-height-cutoff-m are dropped
    (z is height in the LiDAR's own frame; a flat floor sits at an
    ~constant negative z regardless of range, which is why this is a
    height cutoff and not an elevation-angle cutoff)

Protocol: newline-delimited JSON on stdin (requests) and stdout
(replies), identical style to cmd_vel_bridge.py.

Requests:
    {"cmd": "get_range"}
    {"cmd": "quit"}

On startup, before reading any request, this script writes exactly one
line: {"ready": true} on success, or {"ready": false, "error": "..."}
on failure (then exits non-zero).
"""

import sys
import json
import time
import math
import argparse

_STATE = {"msg": None, "stamp": 0.0}


def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _nearest_forward_range(msg, cone_half_angle_deg, floor_height_cutoff_m):
    from sensor_msgs import point_cloud2

    nearest = None
    for x, y, z in point_cloud2.read_points(
        msg, field_names=("x", "y", "z"), skip_nans=True
    ):
        if z <= floor_height_cutoff_m:
            continue
        r = math.sqrt(x * x + y * y + z * z)
        if r < 1e-6:
            continue
        azimuth_deg = math.degrees(math.atan2(y, x))
        if abs(azimuth_deg) > cone_half_angle_deg:
            continue
        if nearest is None or r < nearest:
            nearest = r
    return nearest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/rslidar_points")
    parser.add_argument("--cone-half-angle-deg", type=float, default=20.0)
    parser.add_argument("--floor-height-cutoff-m", type=float, default=-0.30)
    parser.add_argument("--max-scan-age-seconds", type=float, default=1.0)
    parser.add_argument(
        "--ready-timeout-seconds",
        type=float,
        default=8.0,
        help="How long to wait for the first scan before declaring not-ready.",
    )
    args = parser.parse_args()

    try:
        import rospy
        from sensor_msgs.msg import PointCloud2
    except Exception as exc:
        _emit({"ready": False, "error": "import failed: {}".format(exc)})
        sys.exit(1)

    def _callback(msg):
        _STATE["msg"] = msg
        _STATE["stamp"] = time.time()

    try:
        # disable_signals: same reasoning as cmd_vel_bridge.py -- this
        # process's lifecycle is controlled by its parent over stdin.
        rospy.init_node(
            "vlm_lidar_bridge",
            anonymous=True,
            disable_signals=True,
        )
        rospy.Subscriber(args.topic, PointCloud2, _callback, queue_size=1)

        deadline = time.time() + args.ready_timeout_seconds
        while _STATE["msg"] is None and time.time() < deadline:
            time.sleep(0.05)
    except Exception as exc:
        _emit({"ready": False, "error": "init failed: {}".format(exc)})
        sys.exit(1)

    if _STATE["msg"] is None:
        _emit({
            "ready": False,
            "error": "no scan received on {} within {}s".format(
                args.topic, args.ready_timeout_seconds
            ),
        })
        sys.exit(1)

    _emit({"ready": True})

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except Exception as exc:
            _emit({"ok": False, "error": "bad json: {}".format(exc)})
            continue

        cmd = request.get("cmd")
        try:
            if cmd == "get_range":
                msg = _STATE["msg"]
                age = time.time() - _STATE["stamp"]
                if msg is None:
                    _emit({"ok": False, "cmd": cmd, "error": "no scan received yet"})
                    continue
                if age > args.max_scan_age_seconds:
                    _emit({
                        "ok": False,
                        "cmd": cmd,
                        "error": "stale scan, age={:.2f}s".format(age),
                    })
                    continue
                nearest = _nearest_forward_range(
                    msg, args.cone_half_angle_deg, args.floor_height_cutoff_m
                )
                _emit({
                    "ok": True,
                    "cmd": cmd,
                    "range": nearest,
                    "age": age,
                })

            elif cmd == "quit":
                _emit({"ok": True, "cmd": "quit"})
                break

            else:
                _emit({
                    "ok": False,
                    "cmd": cmd,
                    "error": "unknown cmd: {}".format(cmd),
                })

        except Exception as exc:
            _emit({"ok": False, "cmd": cmd, "error": str(exc)})


if __name__ == "__main__":
    main()
