#!/usr/bin/env python3

import argparse
import json
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from robot_executor import SafeCmdVelExecutor


def run(cmd, check=True):
    print("[CMD]", " ".join(cmd))
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    if result.stdout.strip():
        print("[STDOUT]")
        print(result.stdout)

    if result.stderr.strip():
        print("[STDERR]")
        print(result.stderr)

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )

    return result


def ssh_run(robot_user, robot_ip, remote_cmd):
    return run([
        "ssh",
        "{}@{}".format(robot_user, robot_ip),
        remote_cmd,
    ])


def scp_from_robot(robot_user, robot_ip, remote_path, local_path):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    run([
        "scp",
        "{}@{}:{}".format(robot_user, robot_ip, remote_path),
        str(local_path),
    ])

    return local_path


def ffmpeg_capture_command(device, output_path, capture_size, timeout_s):
    return (
        "timeout {timeout_s} ffmpeg -y -loglevel error "
        "-f v4l2 "
        "-video_size {capture_size} "
        "-i {device} "
        "-frames:v 1 "
        "-q:v 2 "
        "{output_path}"
    ).format(
        timeout_s=int(timeout_s),
        capture_size=shlex.quote(capture_size),
        device=shlex.quote(device),
        output_path=shlex.quote(output_path),
    )


def capture_single_camera(args, local_step_dir):
    local_step_dir.mkdir(parents=True, exist_ok=True)

    remote_image = args.remote_image_path
    local_image = local_step_dir / "current_frame.jpg"

    cmd = ffmpeg_capture_command(
        device=args.camera_device,
        output_path=remote_image,
        capture_size=args.capture_size,
        timeout_s=args.capture_timeout,
    )

    ssh_run(args.robot_user, args.robot_ip, cmd)
    scp_from_robot(args.robot_user, args.robot_ip, remote_image, local_image)

    print("[OK] Single camera image:", local_image)
    return local_image


def capture_three_cameras(args, local_step_dir):
    local_step_dir.mkdir(parents=True, exist_ok=True)

    remote_front = "/tmp/vlm_front.jpg"
    remote_left = "/tmp/vlm_left.jpg"
    remote_right = "/tmp/vlm_right.jpg"
    remote_tar = "/tmp/vlm_three_cameras.tar"

    cmd_front = ffmpeg_capture_command(
        device=args.front_device,
        output_path=remote_front,
        capture_size=args.capture_size,
        timeout_s=args.capture_timeout,
    )

    cmd_left = ffmpeg_capture_command(
        device=args.left_device,
        output_path=remote_left,
        capture_size=args.capture_size,
        timeout_s=args.capture_timeout,
    )

    cmd_right = ffmpeg_capture_command(
        device=args.right_device,
        output_path=remote_right,
        capture_size=args.capture_size,
        timeout_s=args.capture_timeout,
    )

    remote_cmd = """
    set -e
    rm -f /tmp/vlm_front.jpg /tmp/vlm_left.jpg /tmp/vlm_right.jpg /tmp/vlm_three_cameras.tar
    {cmd_front}
    {cmd_left}
    {cmd_right}
    tar -cf {remote_tar} -C /tmp vlm_front.jpg vlm_left.jpg vlm_right.jpg
    ls -lh /tmp/vlm_front.jpg /tmp/vlm_left.jpg /tmp/vlm_right.jpg /tmp/vlm_three_cameras.tar
    """.format(
            cmd_front=cmd_front,
            cmd_left=cmd_left,
            cmd_right=cmd_right,
            remote_tar=shlex.quote(remote_tar),
        )

    ssh_run(args.robot_user, args.robot_ip, remote_cmd)

    local_tar = local_step_dir / "vlm_three_cameras.tar"
    scp_from_robot(args.robot_user, args.robot_ip, remote_tar, local_tar)

    with tarfile.open(str(local_tar), "r") as tar:
        tar.extractall(str(local_step_dir))

    rename_map = {
        "vlm_left.jpg": "left.jpg",
        "vlm_front.jpg": "front.jpg",
        "vlm_right.jpg": "right.jpg",
    }

    for old_name, new_name in rename_map.items():
        old_path = local_step_dir / old_name
        new_path = local_step_dir / new_name

        if new_path.exists():
            new_path.unlink()

        old_path.rename(new_path)

    left = local_step_dir / "left.jpg"
    front = local_step_dir / "front.jpg"
    right = local_step_dir / "right.jpg"

    print("[OK] Left image:", left)
    print("[OK] Front image:", front)
    print("[OK] Right image:", right)

    stitched = stitch_three_images(
        left_path=left,
        front_path=front,
        right_path=right,
        output_path=local_step_dir / "stitched_left_front_right.jpg",
        width=args.stitch_width,
        height=args.stitch_height,
    )

    return stitched


def load_font(size):
    try:
        from PIL import ImageFont
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size,
        )
    except Exception:
        try:
            from PIL import ImageFont
            return ImageFont.load_default()
        except Exception:
            return None


def draw_label(img, label):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    font = load_font(28)

    draw.rectangle([0, 0, 190, 45], fill=(255, 255, 255))
    draw.text((12, 8), label, fill=(0, 0, 0), font=font)

    return img


def stitch_three_images(left_path, front_path, right_path, output_path, width, height):
    try:
        from PIL import Image
    except Exception:
        print("[ERROR] PIL is not installed.")
        print("Install it on ThinkPad using:")
        print("sudo apt install python3-pil -y")
        raise

    images = []

    for label, path in [
        ("LEFT", left_path),
        ("FRONT", front_path),
        ("RIGHT", right_path),
    ]:
        img = Image.open(str(path)).convert("RGB")
        img = img.resize((width, height))
        img = draw_label(img, label)
        images.append(img)

    stitched = Image.new("RGB", (width * 3, height))

    for i, img in enumerate(images):
        stitched.paste(img, (i * width, 0))

    output_path = Path(output_path)
    stitched.save(str(output_path), quality=95)

    print("[OK] Stitched image:", output_path)
    return output_path


def get_observation_image(args, step_index=None):
    local_dir = Path(args.local_dir)

    if step_index is None:
        local_step_dir = local_dir / "single"
    else:
        local_step_dir = local_dir / "step_{:04d}".format(step_index)

    if args.camera_mode == "single":
        return capture_single_camera(args, local_step_dir)

    if args.camera_mode == "stitched":
        return capture_three_cameras(args, local_step_dir)

    raise ValueError("Unknown camera mode: {}".format(args.camera_mode))


def post_to_amd(api, endpoint, image_path, goal=None):
    url = api.rstrip("/") + "/" + endpoint.lstrip("/")

    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        url,
    ]

    if goal is not None:
        cmd.extend(["-F", "goal={}".format(goal)])

    cmd.extend(["-F", "image=@{}".format(str(image_path))])

    result = run(cmd)

    try:
        return json.loads(result.stdout)
    except Exception:
        print("[ERROR] AMD response was not valid JSON:")
        print(result.stdout)
        raise


def start_autonomous_session(api, goal):
    url = api.rstrip("/") + "/autonomous/start"

    result = run([
        "curl",
        "-sS",
        "-X",
        "POST",
        url,
        "-F",
        "goal={}".format(goal),
    ])

    try:
        response = json.loads(result.stdout)
    except Exception:
        print(result.stdout)
        return None

    print("[OK] Autonomous session started")
    print(json.dumps(response, indent=2))
    return response


def print_action_result(response):
    print("\n========== ACTION RESULT ==========")

    print("Mode:", response.get("mode"))
    print("Goal:", response.get("goal"))
    print("Done:", response.get("done"))

    if response.get("action_display") is not None:
        print("Action:", response.get("action_display"))

    action = response.get("action", {})

    if isinstance(action, dict):
        print("Action name:", action.get("name"))
        print("Params:", action.get("params"))
        print("Reason:", action.get("reason"))
        print("Confidence:", action.get("confidence"))
        print("Evidence score:", action.get("evidence_score"))

    print("Saved:", response.get("saved_result"))
    print("===================================\n")

def maybe_execute_robot_action(args, response):
    """
    Optionally execute the VLM action using /cmd_vel.

    --execute false   = no movement
    --execute confirm = ask before movement
    --execute true    = move directly
    """
    if args.execute == "false":
        print("[EXECUTE] Dry run only. Robot will not move.")
        return False

    executor = SafeCmdVelExecutor(
        topic=args.cmd_vel_topic,
        forward_speed=args.forward_speed,
        turn_speed=args.turn_speed,
        duration=args.move_duration,
        invert_turn=args.invert_turn,
        execute_mode=args.execute,
    )

    return executor.execute_action_response(response)

def run_single_step(args):
    image_path = get_observation_image(args, step_index=None)

    response = post_to_amd(
        api=args.api,
        endpoint="/single_step",
        image_path=image_path,
        goal=args.goal,
    )

    print_action_result(response)
    maybe_execute_robot_action(args, response)


def run_auto_mode(args):
    start_autonomous_session(args.api, args.goal)

    for step in range(1, args.max_steps + 1):
        print("\n========== AUTO STEP {} ==========".format(step))

        image_path = get_observation_image(args, step_index=step)

        response = post_to_amd(
            api=args.api,
            endpoint="/autonomous/step",
            image_path=image_path,
            goal=None,
        )

        print_action_result(response)
        maybe_execute_robot_action(args, response)

        if response.get("done") is True:
            print("[DONE] Goal reached according to pipeline.")
            break

        time.sleep(args.interval)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ThinkPad-side robot image capture and AMD VLM API client."
    )

    parser.add_argument(
        "--mode",
        choices=["single", "auto"],
        default="single",
        help="single = one step without memory, auto = repeated steps with memory",
    )

    parser.add_argument(
        "--camera-mode",
        choices=["single", "stitched"],
        default="single",
        help="single = one camera image, stitched = left/front/right stitched image",
    )

    parser.add_argument("--goal", default="B0.004")
    parser.add_argument("--api", default="http://ernis-amd395:8001")

    parser.add_argument("--robot-user", default="unitree")
    parser.add_argument("--robot-ip", default="192.168.123.14")

    parser.add_argument(
        "--camera-device",
        default="/dev/video3",
        help="Used only for --camera-mode single",
    )

    parser.add_argument(
        "--front-device",
        default="/dev/video2",
        help="Front camera device for stitched mode",
    )

    parser.add_argument(
        "--left-device",
        default="/dev/video4",
        help="Left camera device for stitched mode",
    )

    parser.add_argument(
        "--right-device",
        default="/dev/video3",
        help="Right camera device for stitched mode",
    )

    parser.add_argument(
        "--capture-size",
        default="960x720",
        help="ffmpeg capture size. Use 960x720 because your cameras accepted it.",
    )

    parser.add_argument(
        "--capture-timeout",
        type=int,
        default=8,
        help="Timeout in seconds for each ffmpeg capture",
    )

    parser.add_argument(
        "--remote-image-path",
        default="/tmp/vlm_current_frame.jpg",
        help="Robot-side image path for single camera mode",
    )

    parser.add_argument(
        "--local-dir",
        default="robot_live_frames",
        help="ThinkPad folder where copied/stiched images are saved",
    )

    parser.add_argument("--stitch-width", type=int, default=640)
    parser.add_argument("--stitch-height", type=int, default=480)

    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=10)

    parser.add_argument(
        "--execute",
        choices=["false", "confirm", "true"],
        default="false",
        help="false=no movement, confirm=ask before movement, true=execute movement directly",
    )

    parser.add_argument(
        "--cmd-vel-topic",
        default="/cmd_vel",
        help="ROS cmd_vel topic used for robot movement",
    )

    parser.add_argument(
        "--forward-speed",
        type=float,
        default=0.05,
        help="Safe forward speed for tiny movement step",
    )

    parser.add_argument(
        "--turn-speed",
        type=float,
        default=0.20,
        help="Safe angular speed for tiny turn step",
    )

    parser.add_argument(
        "--move-duration",
        type=float,
        default=1.0,
        help="Duration in seconds for each tiny movement",
    )

    parser.add_argument(
        "--invert-turn",
        action="store_true",
        help="Use this if left/right turning direction is reversed",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] Mode:", args.mode)
    print("[INFO] Camera mode:", args.camera_mode)
    print("[INFO] Goal:", args.goal)
    print("[INFO] AMD API:", args.api)
    print("[INFO] Robot:", "{}@{}".format(args.robot_user, args.robot_ip))

    if args.camera_mode == "single":
        print("[INFO] Single camera device:", args.camera_device)
    else:
        print("[INFO] Front camera:", args.front_device)
        print("[INFO] Left camera:", args.left_device)
        print("[INFO] Right camera:", args.right_device)

    if args.mode == "single":
        run_single_step(args)
    elif args.mode == "auto":
        run_auto_mode(args)
    else:
        raise ValueError("Unsupported mode: {}".format(args.mode))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print("\n[COMMAND FAILED]")
        print("Return code:", e.returncode)
        print("STDOUT:", e.output)
        print("STDERR:", e.stderr)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n[STOPPED] User interrupted with Ctrl+C")
        sys.exit(130)
    except Exception as e:
        print("\n[ERROR]", e)
        sys.exit(1)