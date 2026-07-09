#!/usr/bin/env python3

import argparse
import json
import shlex
import subprocess
import sys
import tarfile
import time
from datetime import datetime
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
    local_image = local_step_dir / "current_frame.png"

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

def capture_three_camera_files(args, local_step_dir):
    local_step_dir.mkdir(parents=True, exist_ok=True)

    remote_front = "/tmp/vlm_front.png"
    remote_left = "/tmp/vlm_left.png"
    remote_right = "/tmp/vlm_right.png"
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
    rm -f /tmp/vlm_front.png /tmp/vlm_left.png /tmp/vlm_right.png /tmp/vlm_three_cameras.tar
    {cmd_front}
    {cmd_left}
    {cmd_right}
    tar -cf {remote_tar} -C /tmp vlm_front.png vlm_left.png vlm_right.png
    ls -lh /tmp/vlm_front.png /tmp/vlm_left.png /tmp/vlm_right.png /tmp/vlm_three_cameras.tar
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
        "vlm_left.png": "left.png",
        "vlm_front.png": "front.png",
        "vlm_right.png": "right.png",
    }

    for old_name, new_name in rename_map.items():
        old_path = local_step_dir / old_name
        new_path = local_step_dir / new_name

        if new_path.exists():
            new_path.unlink()

        old_path.rename(new_path)

    image_paths = {
        "LEFT": local_step_dir / "left.png",
        "FRONT": local_step_dir / "front.png",
        "RIGHT": local_step_dir / "right.png",
    }

    print("[OK] Left image:", image_paths["LEFT"])
    print("[OK] Front image:", image_paths["FRONT"])
    print("[OK] Right image:", image_paths["RIGHT"])

    return image_paths


def capture_three_cameras_stitched(args, local_step_dir):
    image_paths = capture_three_camera_files(args, local_step_dir)

    stitched = stitch_three_images(
        left_path=image_paths["LEFT"],
        front_path=image_paths["FRONT"],
        right_path=image_paths["RIGHT"],
        output_path=local_step_dir / "stitched_left_front_right.png",
        width=args.stitch_width,
        height=args.stitch_height,
    )

    return stitched, image_paths

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


def get_observation(args, step_index=None):
    local_dir = Path(args.local_dir)

    if step_index is None:
        local_step_dir = local_dir / "single"
    else:
        local_step_dir = local_dir / "step_{:04d}".format(step_index)

    if args.camera_mode == "single":
        primary = capture_single_camera(args, local_step_dir)
        return {
            "camera_mode": "single",
            "primary_path": primary,
            "image_paths": None,
            "saved_files": {
                "image": str(primary),
            },
        }

    if args.camera_mode == "stitched":
        stitched, raw_paths = capture_three_cameras_stitched(args, local_step_dir)
        return {
            "camera_mode": "stitched",
            "primary_path": stitched,
            "image_paths": None,
            "saved_files": {
                "left": str(raw_paths["LEFT"]),
                "front": str(raw_paths["FRONT"]),
                "right": str(raw_paths["RIGHT"]),
                "stitched": str(stitched),
            },
        }

    if args.camera_mode == "separate":
        raw_paths = capture_three_camera_files(args, local_step_dir)
        return {
            "camera_mode": "separate",
            "primary_path": raw_paths["FRONT"],
            "image_paths": raw_paths,
            "saved_files": {
                "left": str(raw_paths["LEFT"]),
                "front": str(raw_paths["FRONT"]),
                "right": str(raw_paths["RIGHT"]),
            },
        }

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

def post_observation_to_amd(api, endpoint, observation, goal=None):
    """
    Send either:
    - one image field for single/stitched mode
    - left_image/front_image/right_image fields for separate mode
    """
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

    camera_mode = observation.get("camera_mode")

    if camera_mode == "separate":
        image_paths = observation.get("image_paths") or {}

        cmd.extend([
            "-F", "left_image=@{}".format(image_paths["LEFT"]),
            "-F", "front_image=@{}".format(image_paths["FRONT"]),
            "-F", "right_image=@{}".format(image_paths["RIGHT"]),
        ])
    else:
        cmd.extend(["-F", "image=@{}".format(observation["primary_path"])])

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
        params = action.get("params", {})
        if not isinstance(params, dict):
            params = {}

        print("Params:", params)
        print("Evidence view:", params.get("evidence_view", "UNKNOWN"))
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
        min_evidence_score=args.min_evidence_score,
        allow_low_confidence=args.allow_low_confidence,
    )

    return executor.execute_action_response(response)

def safe_goal_name(goal):
    return (
        str(goal)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


def create_run_dir(args):
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        run_dir = Path("navigation_outputs/live_runs") / (
            "run_{}_{}".format(timestamp, safe_goal_name(args.goal))
        )

    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "goal": args.goal,
        "mode": args.mode,
        "camera_mode": args.camera_mode,
        "api": args.api,
        "robot_user": args.robot_user,
        "robot_ip": args.robot_ip,
        "camera_device": args.camera_device,
        "front_device": args.front_device,
        "left_device": args.left_device,
        "right_device": args.right_device,
        "capture_size": args.capture_size,
        "execute": args.execute,
        "cmd_vel_topic": args.cmd_vel_topic,
        "forward_speed": args.forward_speed,
        "turn_speed": args.turn_speed,
        "move_duration": args.move_duration,
        "invert_turn": args.invert_turn,
        "memory": args.memory,
        "interval": args.interval,
        "max_steps": args.max_steps,
        "stitch_width": args.stitch_width,
        "stitch_height": args.stitch_height,
        "min_evidence_score": args.min_evidence_score,
        "allow_low_confidence": args.allow_low_confidence,
    }

    args.run_config = config
    (run_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )

    print("[LOG] Run directory:", run_dir)
    return run_dir


def save_step_log(
    run_dir,
    step_index,
    observation,
    response,
    executed,
    run_config=None,
    step_metric=None,
):
    step_result = {
        "run_settings": run_config or {},
        "step": step_index,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "camera_mode": observation.get("camera_mode"),
        "primary_path": str(observation.get("primary_path")),
        "saved_files": observation.get("saved_files", {}),
        "executed": executed,
        "step_metric": step_metric or {},
        "evidence_view": (step_metric or {}).get("evidence_view", "UNKNOWN"),
        "response": response,
    }

    result_path = run_dir / "step_{:03d}_result.json".format(step_index)
    result_path.write_text(json.dumps(step_result, indent=2, ensure_ascii=False))

    print("[LOG] Step result saved:", result_path)
    return result_path


def save_final_summary(run_dir, goal, steps, success, run_config=None, metrics=None):
    summary = {
        "run_settings": run_config or {},
        "ended_at": datetime.now().isoformat(timespec="seconds"),
        "goal": goal,
        "steps": steps,
        "success": success,
        "metrics": metrics or {},
    }

    summary_path = run_dir / "final_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("[LOG] Final summary saved:", summary_path)
    return summary_path

def extract_step_metric(response, executed):
    action = response.get("action", {})
    if not isinstance(action, dict):
        action = {}

    params = action.get("params", {})
    if not isinstance(params, dict):
        params = {}

    try:
        evidence_score = float(action.get("evidence_score", 0.0) or 0.0)
    except Exception:
        evidence_score = 0.0

    return {
        "done": bool(response.get("done")),
        "executed": bool(executed),
        "mode": response.get("mode"),
        "observation_mode": response.get("observation_mode"),
        "action_name": action.get("name"),
        "confidence": action.get("confidence"),
        "evidence_score": evidence_score,
        "is_valid": bool(action.get("is_valid", True)),
        "evidence_view": params.get("evidence_view"),
    }


def compute_run_metrics(step_metrics, success):
    total_steps = len(step_metrics)

    action_counts = {}
    evidence_view_counts = {}

    evidence_scores = []
    low_confidence_count = 0
    invalid_action_count = 0
    executed_steps = 0

    for step in step_metrics:
        action_name = step.get("action_name") or "UNKNOWN"
        action_counts[action_name] = action_counts.get(action_name, 0) + 1

        evidence_view = step.get("evidence_view") or "UNKNOWN"
        evidence_view_counts[evidence_view] = evidence_view_counts.get(evidence_view, 0) + 1

        evidence_scores.append(float(step.get("evidence_score", 0.0) or 0.0))

        if step.get("confidence") == "low":
            low_confidence_count += 1

        if not step.get("is_valid", True):
            invalid_action_count += 1

        if step.get("executed"):
            executed_steps += 1

    avg_evidence_score = (
        sum(evidence_scores) / len(evidence_scores)
        if evidence_scores else 0.0
    )

    return {
        "success": bool(success),
        "success_rate_single_run": 1.0 if success else 0.0,
        "failure_rate_single_run": 0.0 if success else 1.0,
        "total_steps": total_steps,
        "steps_to_success": total_steps if success else None,
        "executed_steps": executed_steps,
        "non_executed_steps": total_steps - executed_steps,
        "low_confidence_actions": low_confidence_count,
        "invalid_actions": invalid_action_count,
        "average_action_evidence_score": avg_evidence_score,
        "action_counts": action_counts,
        "evidence_view_counts": evidence_view_counts,
        "obstacle_avoided": None,
        "obstacle_note": "Obstacle avoidance is not measured automatically yet. Add manual annotations or sensor-based obstacle logging later.",
    }

def run_single_step(args):
    run_dir = create_run_dir(args)

    observation = get_observation(args, step_index=1)

    response = post_observation_to_amd(
        api=args.api,
        endpoint="/single_step",
        observation=observation,
        goal=args.goal,
    )

    print_action_result(response)
    executed = maybe_execute_robot_action(args, response)

    run_config = getattr(args, "run_config", {})
    step_metric = extract_step_metric(response, executed)
    step_metrics = [step_metric]
    success = bool(response.get("done"))
    metrics = compute_run_metrics(step_metrics, success)

    save_step_log(
        run_dir=run_dir,
        step_index=1,
        observation=observation,
        response=response,
        executed=executed,
        run_config=run_config,
        step_metric=step_metric,
    )

    save_final_summary(
        run_dir=run_dir,
        goal=args.goal,
        steps=1,
        success=success,
        run_config=run_config,
        metrics=metrics,
    )

def run_auto_mode(args):
    run_dir = create_run_dir(args)
    run_config = getattr(args, "run_config", {})

    use_memory = args.memory == "true"

    if use_memory:
        print("[INFO] Memory enabled: using /autonomous/start and /autonomous/step")
        start_autonomous_session(args.api, args.goal)
    else:
        print("[INFO] Memory disabled: using fresh /single_step for every auto step")

    success = False
    completed_steps = 0
    step_metrics = []

    for step in range(1, args.max_steps + 1):
        completed_steps = step
        print("\n========== AUTO STEP {} ==========".format(step))

        observation = get_observation(args, step_index=step)

        if use_memory:
            endpoint = "/autonomous/step"
            goal_for_request = None
        else:
            endpoint = "/single_step"
            goal_for_request = args.goal

        response = post_observation_to_amd(
            api=args.api,
            endpoint=endpoint,
            observation=observation,
            goal=goal_for_request,
        )

        print_action_result(response)
        executed = maybe_execute_robot_action(args, response)

        step_metric = extract_step_metric(response, executed)
        step_metrics.append(step_metric)

        save_step_log(
            run_dir=run_dir,
            step_index=step,
            observation=observation,
            response=response,
            executed=executed,
            run_config=run_config,
            step_metric=step_metric,
        )

        if response.get("done") is True:
            print("[DONE] Goal reached according to pipeline.")
            success = True
            break

        time.sleep(args.interval)

    metrics = compute_run_metrics(step_metrics, success)

    save_final_summary(
        run_dir=run_dir,
        goal=args.goal,
        steps=completed_steps,
        success=success,
        run_config=run_config,
        metrics=metrics,
    )

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
        choices=["single", "stitched", "separate"],
        default="single",
        help="single = one camera image, stitched = left/front/right stitched image, separate = send left/front/right as separate model images",
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
        default="/tmp/vlm_current_frame.png",
        help="Robot-side image path for single camera mode",
    )

    parser.add_argument(
        "--local-dir",
        default="robot_live_frames",
        help="ThinkPad folder where copied/stiched images are saved",
    )

    parser.add_argument("--stitch-width", type=int, default=960)
    parser.add_argument("--stitch-height", type=int, default=720)

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
        default=0.12,
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
        default=5.0,
        help="Duration in seconds for each tiny movement",
    )

    parser.add_argument(
        "--invert-turn",
        action="store_true",
        help="Use this if left/right turning direction is reversed",
    )

    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional output folder for this live run. Default creates navigation_outputs/live_runs/run_<timestamp>_<goal>",
    )

    parser.add_argument(
        "--memory",
        choices=["true", "false"],
        default="true",
        help="true = keep memory across auto steps, false = fresh memory every step",
    )

    parser.add_argument(
        "--min-evidence-score",
        type=float,
        default=0.30,
        help="Minimum action evidence score required for robot movement",
    )

    parser.add_argument(
        "--allow-low-confidence",
        action="store_true",
        help="Allow low-confidence VLM actions to move the robot",
    )

    return parser.parse_args()

 
def main():
    args = parse_args()

    print("[INFO] Mode:", args.mode)
    print("[INFO] Camera mode:", args.camera_mode)
    print("[INFO] Memory:", args.memory)
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