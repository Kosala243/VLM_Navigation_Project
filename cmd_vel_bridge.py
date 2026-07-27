#!/usr/bin/env python2

"""
cmd_vel_bridge.py

Persistent ROS publisher child process for the Unitree Go1 /cmd_vel topic.

WHY THIS FILE EXISTS:
ROS Melodic only ships rospy for Python 2, but the rest of the
navigation pipeline (robot_executor.py, thinkpad_robot_capture_to_amd.py)
runs under Python 3. This script is spawned once, as a long-lived
subprocess, by robot_executor.py's PersistentCmdVelBridge. It performs
rospy.init_node() and creates its Publisher exactly once for the whole
run, then services many pulse/stop requests over stdin/stdout without
ever paying ROS master registration cost again. This removes the
registration-vs-timeout race that a per-pulse "spawn rostopic pub,
kill it with `timeout`" approach was subject to.

Protocol: newline-delimited JSON on stdin (requests) and stdout
(replies). One reply is written for every request, in order.

Requests:
    {"cmd": "pulse", "linear_x": <float>, "angular_z": <float>,
     "duration": <float seconds>, "rate": <float Hz>}
    {"cmd": "stop"}
    {"cmd": "quit"}

On startup, before reading any request, this script writes exactly one
line: {"ready": true, "connections": <int>} on success, or
{"ready": false, "error": "..."} on failure (then exits non-zero).
"""

import sys
import json
import time
import argparse


def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/cmd_vel")
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument(
        "--subscriber-wait-seconds",
        type=float,
        default=5.0,
        help=(
            "How long to wait for a subscriber to connect before "
            "declaring ready anyway. This is a one-time cost paid "
            "once at startup, not per pulse."
        ),
    )
    args = parser.parse_args()

    try:
        import rospy
        from geometry_msgs.msg import Twist
    except Exception as exc:
        _emit({"ready": False, "error": "import failed: {}".format(exc)})
        sys.exit(1)

    try:
        # disable_signals: this process's lifecycle is controlled by
        # its parent over stdin (an explicit "quit" request, or the
        # parent closing/killing it), not by delivering SIGINT/SIGTERM
        # to rospy directly.
        rospy.init_node(
            "vlm_cmd_vel_bridge",
            anonymous=True,
            disable_signals=True,
        )
        pub = rospy.Publisher(args.topic, Twist, queue_size=1)

        deadline = time.time() + args.subscriber_wait_seconds
        while (
            pub.get_num_connections() == 0
            and time.time() < deadline
        ):
            time.sleep(0.05)
    except Exception as exc:
        _emit({"ready": False, "error": "init failed: {}".format(exc)})
        sys.exit(1)

    _emit({"ready": True, "connections": pub.get_num_connections()})

    zero = Twist()

    while True:
        line = sys.stdin.readline()
        if not line:
            # Parent closed stdin; publish a final stop and exit.
            try:
                pub.publish(zero)
            except Exception:
                pass
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
            if cmd == "pulse":
                linear_x = float(request.get("linear_x", 0.0))
                angular_z = float(request.get("angular_z", 0.0))
                duration = float(request.get("duration", 0.0))
                rate_hz = float(request.get("rate", args.rate))
                if duration <= 0.0:
                    raise ValueError("duration must be positive")

                msg = Twist()
                msg.linear.x = linear_x
                msg.angular.z = angular_z

                rate = rospy.Rate(rate_hz)
                start = time.time()
                published = 0
                while (
                    time.time() - start < duration
                    and not rospy.is_shutdown()
                ):
                    pub.publish(msg)
                    published += 1
                    rate.sleep()

                pub.publish(zero)
                _emit({
                    "ok": True,
                    "cmd": "pulse",
                    "published": published,
                    "elapsed": time.time() - start,
                    "connections": pub.get_num_connections(),
                })

            elif cmd == "stop":
                pub.publish(zero)
                _emit({
                    "ok": True,
                    "cmd": "stop",
                    "connections": pub.get_num_connections(),
                })

            elif cmd == "quit":
                pub.publish(zero)
                _emit({"ok": True, "cmd": "quit"})
                break

            else:
                _emit({
                    "ok": False,
                    "cmd": cmd,
                    "error": "unknown cmd: {}".format(cmd),
                })

        except Exception as exc:
            try:
                pub.publish(zero)
            except Exception:
                pass
            _emit({"ok": False, "cmd": cmd, "error": str(exc)})


if __name__ == "__main__":
    main()
