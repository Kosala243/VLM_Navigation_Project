import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, check=True):
    print("[CMD]", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if check and result.returncode != 0:
        print("[STDOUT]", result.stdout)
        print("[STDERR]", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result


def capture_robot_frame(robot_user, robot_ip, camera_device, remote_image_path):
    cmd = (
        "timeout 8 ffmpeg -y -f v4l2 "
        "-video_size 960x720 "
        "-i {} "
        "-frames:v 1 {}"
    ).format(camera_device, remote_image_path)

    run(["ssh", "{}@{}".format(robot_user, robot_ip), cmd])
    print("[Robot] Captured frame:", remote_image_path)


def copy_from_robot(robot_user, robot_ip, remote_image_path, local_image_path):
    local_image_path.parent.mkdir(parents=True, exist_ok=True)

    run([
        "scp",
        "{}@{}:{}".format(robot_user, robot_ip, remote_image_path),
        str(local_image_path),
    ])

    print("[ThinkPad] Copied image:", local_image_path)


def post_to_amd(api, endpoint, image_path, goal=None):
    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        api.rstrip("/") + "/" + endpoint.lstrip("/"),
    ]

    if goal:
        cmd.extend(["-F", "goal={}".format(goal)])

    cmd.extend(["-F", "image=@{}".format(image_path)])

    result = run(cmd)

    try:
        return json.loads(result.stdout)
    except Exception:
        print("[ERROR] Could not parse response:")
        print(result.stdout)
        raise


def start_autonomous(api, goal):
    result = run([
        "curl",
        "-sS",
        "-X",
        "POST",
        api.rstrip("/") + "/autonomous/start",
        "-F",
        "goal={}".format(goal),
    ])
    print(result.stdout)


def print_result(response):
    print("\n========== ACTION RESULT ==========")
    print("Mode:", response.get("mode"))
    print("Goal:", response.get("goal"))
    print("Done:", response.get("done"))
    print("Action:", response.get("action_display"))

    action = response.get("action", {})
    if isinstance(action, dict):
        print("Action name:", action.get("name"))
        print("Params:", action.get("params"))
        print("Confidence:", action.get("confidence"))
        print("Evidence score:", action.get("evidence_score"))

    print("Saved:", response.get("saved_result"))
    print("===================================\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["single", "auto"], default="single")
    parser.add_argument("--goal", default="B0.004")
    parser.add_argument("--api", default="http://ernis-amd395:8001")

    parser.add_argument("--robot-user", default="unitree")
    parser.add_argument("--robot-ip", default="192.168.123.14")
    parser.add_argument("--camera-device", default="/dev/video1")

    parser.add_argument("--remote-image-path", default="/tmp/vlm_current_frame.jpg")
    parser.add_argument("--local-dir", default="robot_live_frames")

    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=10)

    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        local_image = local_dir / "current_frame.jpg"

        capture_robot_frame(args.robot_user, args.robot_ip, args.camera_device, args.remote_image_path)
        copy_from_robot(args.robot_user, args.robot_ip, args.remote_image_path, local_image)

        response = post_to_amd(args.api, "/single_step", local_image, args.goal)
        print_result(response)
        return

    if args.mode == "auto":
        start_autonomous(args.api, args.goal)

        for step in range(1, args.max_steps + 1):
            print("\n[AUTO] Step", step)
            local_image = local_dir / "frame_{:04d}.jpg".format(step)

            capture_robot_frame(args.robot_user, args.robot_ip, args.camera_device, args.remote_image_path)
            copy_from_robot(args.robot_user, args.robot_ip, args.remote_image_path, local_image)

            response = post_to_amd(args.api, "/autonomous/step", local_image)
            print_result(response)

            if response.get("done") is True:
                print("[AUTO] Goal reached.")
                break

            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print("\n[COMMAND FAILED]")
        print("Return code:", e.returncode)
        print("STDOUT:", e.output)
        print("STDERR:", e.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print("\n[ERROR]", e)
        sys.exit(1)