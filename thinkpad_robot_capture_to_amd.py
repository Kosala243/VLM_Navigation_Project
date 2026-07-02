from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("[CMD]", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def ssh_cmd(robot_user: str, robot_ip: str, remote_cmd: str) -> subprocess.CompletedProcess:
    return run(["ssh", f"{robot_user}@{robot_ip}", remote_cmd])


def capture_on_robot(
    *,
    robot_user: str,
    robot_ip: str,
    camera_device: str,
    remote_image_path: str,
) -> None:
    """
    Captures one image on the robot from USB webcam.
    Tries OpenCV first, then ffmpeg, then fswebcam.
    """

    python_capture = f"""
python3 - <<'PY'
import sys
import time

device = {camera_device!r}
out = {remote_image_path!r}

try:
    import cv2
except Exception as e:
    print("OpenCV not available:", e, file=sys.stderr)
    sys.exit(10)

cap = cv2.VideoCapture(device)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

time.sleep(0.5)
ok, frame = cap.read()
cap.release()

if not ok or frame is None:
    print(f"Could not capture image from {{device}}", file=sys.stderr)
    sys.exit(11)

cv2.imwrite(out, frame)
print(out)
PY
""".strip()

    methods = [
        python_capture,
        f"ffmpeg -y -f v4l2 -video_size 1280x720 -i {shlex.quote(camera_device)} -frames:v 1 {shlex.quote(remote_image_path)}",
        f"fswebcam -r 1280x720 --no-banner {shlex.quote(remote_image_path)}",
    ]

    last_error = ""
    for method in methods:
        result = ssh_cmd(robot_user, robot_ip, method)
        if result.returncode == 0:
            print("[Robot] Image captured successfully.")
            return
        last_error = result.stderr
        print("[Robot] Capture method failed. Trying next method...")
        print(last_error)

    raise RuntimeError(f"All robot capture methods failed. Last error:\n{last_error}")


def copy_image_from_robot(
    *,
    robot_user: str,
    robot_ip: str,
    remote_image_path: str,
    local_image_path: Path,
) -> None:
    local_image_path.parent.mkdir(parents=True, exist_ok=True)

    run([
        "scp",
        f"{robot_user}@{robot_ip}:{remote_image_path}",
        str(local_image_path),
    ])

    if not local_image_path.exists():
        raise FileNotFoundError(f"Image was not copied to {local_image_path}")

    print(f"[ThinkPad] Copied image to {local_image_path}")


def post_image_to_amd(
    *,
    api_base_url: str,
    endpoint: str,
    image_path: Path,
    goal: str | None = None,
) -> dict:
    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        f"{api_base_url.rstrip('/')}/{endpoint.lstrip('/')}",
    ]

    if goal is not None:
        cmd += ["-F", f"goal={goal}"]

    cmd += ["-F", f"image=@{image_path}"]

    result = run(cmd)

    if result.stderr:
        print(result.stderr)

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print("[ERROR] Could not parse AMD response as JSON.")
        print(result.stdout)
        raise


def start_autonomous_session(api_base_url: str, goal: str) -> None:
    result = run([
        "curl",
        "-sS",
        "-X",
        "POST",
        f"{api_base_url.rstrip('/')}/autonomous/start",
        "-F",
        f"goal={goal}",
    ])

    print("[AMD] Autonomous start response:")
    print(result.stdout)


def print_action(response: dict) -> None:
    print("\n========== AMD ACTION RESULT ==========")
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

    print("Saved result:", response.get("saved_result"))
    print("======================================\n")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["single", "auto"], default="single")
    parser.add_argument("--goal", default="B0.004")

    parser.add_argument("--api", default="http://ernis-amd395:8001")

    parser.add_argument("--robot-user", default="unitree")
    parser.add_argument("--robot-ip", default="192.168.123.14")
    parser.add_argument("--camera-device", default="/dev/video0")

    parser.add_argument("--remote-image-path", default="/tmp/vlm_current_frame.jpg")
    parser.add_argument("--local-dir", default="robot_live_frames")

    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=10)

    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        local_image_path = local_dir / "current_frame.jpg"

        capture_on_robot(
            robot_user=args.robot_user,
            robot_ip=args.robot_ip,
            camera_device=args.camera_device,
            remote_image_path=args.remote_image_path,
        )

        copy_image_from_robot(
            robot_user=args.robot_user,
            robot_ip=args.robot_ip,
            remote_image_path=args.remote_image_path,
            local_image_path=local_image_path,
        )

        response = post_image_to_amd(
            api_base_url=args.api,
            endpoint="/single_step",
            image_path=local_image_path,
            goal=args.goal,
        )

        print_action(response)
        return

    if args.mode == "auto":
        start_autonomous_session(args.api, args.goal)

        step = 0
        while True:
            step += 1
            local_image_path = local_dir / f"frame_{step:04d}.jpg"

            print(f"\n[AUTO] Step {step}")

            capture_on_robot(
                robot_user=args.robot_user,
                robot_ip=args.robot_ip,
                camera_device=args.camera_device,
                remote_image_path=args.remote_image_path,
            )

            copy_image_from_robot(
                robot_user=args.robot_user,
                robot_ip=args.robot_ip,
                remote_image_path=args.remote_image_path,
                local_image_path=local_image_path,
            )

            response = post_image_to_amd(
                api_base_url=args.api,
                endpoint="/autonomous/step",
                image_path=local_image_path,
            )

            print_action(response)

            if response.get("done") is True:
                print("[AUTO] Goal reached according to pipeline.")
                break

            if args.max_steps > 0 and step >= args.max_steps:
                print("[AUTO] Reached max steps.")
                break

            time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print("\n[COMMAND FAILED]")
        print("Command:", e.cmd)
        print("Return code:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        sys.exit(e.returncode)
    except Exception as e:
        print("\n[ERROR]", e)
        sys.exit(1)