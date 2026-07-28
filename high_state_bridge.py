#!/usr/bin/env python2

"""
high_state_bridge.py

Persistent ROS subscriber child process for /high_state.

WHY THIS FILE EXISTS: same reason as lidar_bridge.py and cmd_vel_bridge.py
-- ROS Melodic only ships rospy for Python 2, but robot_executor.py runs
under Python 3. This script is spawned once, as a long-lived subprocess,
by robot_executor.py's PersistentHighStateBridge. It performs
rospy.init_node() and creates its Subscriber exactly once for the whole
run, keeps the most recently received HighState message in memory, and
answers "current yaw" requests over stdin/stdout by reading that stored
message on demand -- no per-check ROS registration cost, no per-check
wait for a fresh message.

Confirmed via temp_files/ROS_related.txt: /high_state publishes at
~50Hz, unitree_legged_msgs/HighState, with imu.rpy = [roll, pitch, yaw]
in radians.

Protocol: newline-delimited JSON on stdin (requests) and stdout
(replies), identical style to lidar_bridge.py / cmd_vel_bridge.py.

Requests:
    {"cmd": "get_yaw"}
    {"cmd": "quit"}

On startup, before reading any request, this script writes exactly one
line: {"ready": true} on success, or {"ready": false, "error": "..."}
on failure (then exits non-zero).
"""

import sys
import json
import time
import argparse

_STATE = {"msg": None, "stamp": 0.0}


def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/high_state")
    parser.add_argument(
        "--max-state-age-seconds",
        type=float,
        default=0.3,
        help=(
            "/high_state publishes at ~50Hz, so 0.3s is generous "
            "compared to normal jitter (observed max ~0.041s) while "
            "still catching a genuinely stalled feed quickly."
        ),
    )
    parser.add_argument(
        "--ready-timeout-seconds",
        type=float,
        default=8.0,
        help="How long to wait for the first state before declaring not-ready.",
    )
    args = parser.parse_args()

    try:
        import rospy
        from unitree_legged_msgs.msg import HighState
    except Exception as exc:
        _emit({"ready": False, "error": "import failed: {}".format(exc)})
        sys.exit(1)

    def _callback(msg):
        _STATE["msg"] = msg
        _STATE["stamp"] = time.time()

    try:
        # disable_signals: same reasoning as the other bridges -- this
        # process's lifecycle is controlled by its parent over stdin.
        rospy.init_node(
            "vlm_high_state_bridge",
            anonymous=True,
            disable_signals=True,
        )
        rospy.Subscriber(args.topic, HighState, _callback, queue_size=1)

        deadline = time.time() + args.ready_timeout_seconds
        while _STATE["msg"] is None and time.time() < deadline:
            time.sleep(0.02)
    except Exception as exc:
        _emit({"ready": False, "error": "init failed: {}".format(exc)})
        sys.exit(1)

    if _STATE["msg"] is None:
        _emit({
            "ready": False,
            "error": "no state received on {} within {}s".format(
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
            if cmd == "get_yaw":
                msg = _STATE["msg"]
                age = time.time() - _STATE["stamp"]
                if msg is None:
                    _emit({"ok": False, "cmd": cmd, "error": "no state received yet"})
                    continue
                if age > args.max_state_age_seconds:
                    _emit({
                        "ok": False,
                        "cmd": cmd,
                        "error": "stale state, age={:.2f}s".format(age),
                    })
                    continue
                _emit({
                    "ok": True,
                    "cmd": cmd,
                    "yaw": float(msg.imu.rpy[2]),
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
