from __future__ import annotations

import argparse
import sys
import socket
from pathlib import Path

from .config import load_config
from .controller import Controls
from .controller import create_controller
from .learning import DRIVING_MODES, OnlineDrivingPolicy, resolve_driving_mode
from .paths import DEFAULT_MODEL_TYPE, DEFAULT_NAME, data_path, model_path, online_model_path
from .policy import CautiousFallbackPolicy, LearnedPolicy, SmoothPolicy
from .redline import RedlineEstimator
from .terminal_ui import DashboardState, TerminalDashboard, normalize_command
from .telemetry import TelemetryReceiver, append_frame, is_driving_frame
from .terrain import TERRAIN_PREFERENCES, enrich_terrain, resolve_terrain_preference
from .trainer import train_model
from .transmission import TRANSMISSION_MODES, ShiftAdvisor, normalize_transmission_mode
from .vision import VisionWorker


def record(args: argparse.Namespace) -> int:
    output_path = args.out or data_path(args.name, args.type)
    track = args.track or args.name
    config = load_config(args.config)
    transmission_mode = normalize_transmission_mode(args.transmission or config.drive.transmission_mode)
    terrain_preference = resolve_terrain_preference(args.type, args.terrain_preference)
    receiver = TelemetryReceiver(
        config.telemetry.host,
        config.telemetry.port,
        config.telemetry.profile,
        config.telemetry.timeout_seconds,
    )
    vision = VisionWorker()
    vision.start()
    seen = 0
    saved = 0
    dashboard = TerminalDashboard(
        DashboardState(
            mode="record",
            target=f"{config.telemetry.profile} UDP {config.telemetry.host}:{config.telemetry.port}",
            transmission_mode=transmission_mode,
            terrain_preference=terrain_preference,
            message=f"Writing {output_path}",
        ),
        enabled=not args.no_ui,
    )
    dashboard.start()
    if args.no_ui:
        print(
            f"Listening on UDP {config.telemetry.host}:{config.telemetry.port}; "
            f"transmission={transmission_mode}; terrain={terrain_preference}; writing {output_path}"
        )
    previous_frame = None
    redline_estimator = RedlineEstimator()
    try:
        for frame in receiver.frames(track):
            vision_state = vision.get_state()
            if vision_state.active:
                frame.values.update({
                    "vision_is_menu": int(vision_state.is_menu),
                    "vision_skill_score": vision_state.skill_score,
                    "vision_lane_offset": vision_state.lane_offset,
                })

            redline_estimator.enrich(frame)
            enrich_terrain(frame, previous_frame)
            seen += 1
            should_quit = False
            for command_text in dashboard.poll_commands():
                command = normalize_command(command_text)
                if command == "quit":
                    should_quit = True
                elif not dashboard.apply_common_command(command):
                    dashboard.update(message=f"Unknown command: {command_text}")
            if should_quit:
                break
            if not dashboard.state.paused:
                append_frame(output_path, frame)
                saved += 1
            dashboard.update(frame=frame, frames_seen=seen, frames_saved=saved)
            if args.no_ui and saved % 300 == 0:
                track_ordinal = frame.values.get("track_ordinal")
                suffix = f", track_ordinal={track_ordinal}" if track_ordinal is not None else ""
                print(f"recorded {saved} frames{suffix}")
            if args.limit and saved >= args.limit:
                break
            previous_frame = frame
    except socket.timeout:
        dashboard.stop("Timed out waiting for telemetry")
        print("Timed out waiting for telemetry.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        dashboard.stop(f"Stopped recording after {saved} saved frames.")
        if args.no_ui:
            print(f"Stopped recording after {saved} frames.")
        return 0
    finally:
        vision.stop()
    dashboard.stop(f"Stopped recording after {saved} saved frames.")
    return 0


def train(args: argparse.Namespace) -> int:
    input_path = Path(args.input) if args.input else data_path(args.name, args.type)
    output_path = Path(args.model) if args.model else model_path(args.name, args.type)
    track = args.track if args.track is not None else (args.name if args.input is None else None)
    result = train_model(input_path, output_path, track, args.track_ordinal, args.min_samples)
    print(f"trained {result['samples']} frames -> {result['model']}")
    return 0


def drive(args: argparse.Namespace) -> int:
    base_model_path = Path(args.model) if args.model else model_path(args.name, args.type)
    online_path = Path(args.online_model) if args.online_model else online_model_path(args.name, args.type)
    track = args.track if args.track is not None else args.name
    score_weight = args.score_weight
    if score_weight is None:
        score_weight = 2.0 if args.type.lower() in {"skill", "skills"} else 1.0
    driving_mode = resolve_driving_mode(args.type, getattr(args, "driving_mode", "auto"))
    config = load_config(args.config)
    transmission_mode = normalize_transmission_mode(args.transmission or config.drive.transmission_mode)
    terrain_preference = resolve_terrain_preference(args.type, args.terrain_preference)
    receiver = TelemetryReceiver(
        config.telemetry.host,
        config.telemetry.port,
        config.telemetry.profile,
        config.telemetry.timeout_seconds,
    )
    base = LearnedPolicy(base_model_path) if base_model_path.exists() else CautiousFallbackPolicy()
    explore = getattr(args, "explore", True)
    online_policy = None
    if args.self_train:
        online_policy = OnlineDrivingPolicy(
            base,
            online_path,
            autosave_frames=args.autosave_frames,
            online_weight=args.online_weight,
            score_weight=score_weight,
            terrain_preference=terrain_preference,
            driving_mode=driving_mode,
            # Flatten both exploration knobs to minimum when disabled
            epsilon=0.15 if explore else 0.0,
            epsilon_min=0.05 if explore else 0.0,
            exploration_std=0.18 if explore else 0.0,
            min_exploration_std=0.04 if explore else 0.0,
        )
        base = online_policy
    policy = SmoothPolicy(
        base,
        max_steer_delta=config.drive.max_steer_delta,
        max_throttle_delta=config.drive.max_throttle_delta,
        max_brake_delta=config.drive.max_brake_delta,
    )
    shift_advisor = ShiftAdvisor(transmission_mode)
    controller_kind = "dry-run" if args.dry_run else config.drive.controller
    controller = create_controller(controller_kind)
    vision = VisionWorker()
    vision.start()
    dashboard = TerminalDashboard(
        DashboardState(
            mode="drive",
            target=f"{config.telemetry.profile} UDP {config.telemetry.host}:{config.telemetry.port} -> {controller_kind}",
            transmission_mode=transmission_mode,
            terrain_preference=terrain_preference,
        ),
        enabled=not args.no_ui,
    )
    dashboard.start()
    if args.no_ui:
        print(
            f"Driving from telemetry with {transmission_mode} transmission mode "
            f"and {terrain_preference} terrain preference. Press Ctrl+C to stop."
        )
    seen = 0
    previous_frame = None
    redline_estimator = RedlineEstimator()
    previous_learning_frame = None
    previous_learning_controls = None
    try:
        for frame in receiver.frames(track):
            vision_state = vision.get_state()
            if vision_state.active:
                frame.values.update({
                    "vision_is_menu": int(vision_state.is_menu),
                    "vision_skill_score": vision_state.skill_score,
                    "vision_lane_offset": vision_state.lane_offset,
                })

            redline_estimator.enrich(frame)
            enrich_terrain(frame, previous_frame)
            seen += 1
            should_quit = False
            force_neutral = False
            for command_text in dashboard.poll_commands():
                command = normalize_command(command_text)
                if command == "quit":
                    should_quit = True
                elif command == "neutral":
                    force_neutral = True
                    dashboard.update(message="Neutral sent")
                elif not dashboard.apply_common_command(command):
                    dashboard.update(message=f"Unknown command: {command_text}")
            if should_quit:
                break
            if force_neutral or dashboard.state.paused:
                controller.neutral()
                dashboard.update(frame=frame, controls=Controls(), frames_seen=seen)
                previous_learning_frame = None
                previous_learning_controls = None
                continue
            if is_driving_frame(frame):
                if online_policy is not None and previous_learning_frame is not None and previous_learning_controls is not None:
                    reward = online_policy.learn(previous_learning_frame, frame, previous_learning_controls)
                    if args.no_ui and args.autosave_frames > 0 and online_policy.updates % args.autosave_frames == 0:
                        print(f"self-trained {online_policy.updates} updates; last reward {reward.total:+.3f}")
                controls = shift_advisor.apply(policy.predict(frame), frame)
                controller.apply(controls)
                reward_message = None
                if online_policy is not None and online_policy.last_reward is not None:
                    reward_message = (
                        f"Self-train updates {online_policy.updates}; "
                        f"reward {online_policy.last_reward.total:+.3f}"
                    )
                dashboard.update(frame=frame, controls=controls, frames_seen=seen, message=reward_message)
                previous_learning_frame = frame
                previous_learning_controls = controls
            else:
                controller.neutral()
                dashboard.update(frame=frame, controls=Controls(), frames_seen=seen, message="Waiting for Horizon driving telemetry")
                previous_learning_frame = None
                previous_learning_controls = None
            previous_frame = frame
    except socket.timeout:
        controller.neutral()
        dashboard.stop("Timed out waiting for telemetry; controller neutral")
        print("Timed out waiting for telemetry.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        if args.no_ui:
            print("Stopping and returning controller to neutral.")
    finally:
        controller.neutral()
        vision.stop()
        if online_policy is not None and online_policy.updates:
            online_policy.save()
        dashboard.stop("Stopped; controller neutral")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forza-ai")
    sub = parser.add_subparsers(required=True)

    record_parser = sub.add_parser("record", help="Record Forza Data Out telemetry as JSONL training data.")
    record_parser.add_argument("--config", default="configs/horizon.toml")
    record_parser.add_argument("--name", default=DEFAULT_NAME, help="Model/route name used for automatic file paths.")
    record_parser.add_argument("--type", "--model-type", default=DEFAULT_MODEL_TYPE, help="Model family such as driving, skills, or racing.")
    record_parser.add_argument("--out", help="Optional explicit recording path override.")
    record_parser.add_argument("--track")
    record_parser.add_argument("--terrain-preference", choices=TERRAIN_PREFERENCES, default="auto", help="Terrain preference for this recording.")
    record_parser.add_argument("--transmission", choices=TRANSMISSION_MODES, help="Transmission mode used for this recording.")
    record_parser.add_argument("--limit", type=int)
    record_parser.add_argument("--no-ui", action="store_true")
    record_parser.set_defaults(func=record)

    train_parser = sub.add_parser("train", help="Train a driving model from recorded telemetry.")
    train_parser.add_argument("--name", default=DEFAULT_NAME, help="Model/route name used for automatic file paths.")
    train_parser.add_argument("--type", "--model-type", default=DEFAULT_MODEL_TYPE, help="Model family such as driving, skills, or racing.")
    train_parser.add_argument("--in", dest="input", help="Optional explicit recording path override.")
    train_parser.add_argument("--model", help="Optional explicit model path override.")
    train_parser.add_argument("--track")
    train_parser.add_argument("--track-ordinal", type=int)
    train_parser.add_argument("--min-samples", type=int, default=120)
    train_parser.set_defaults(func=train)

    drive_parser = sub.add_parser("drive", help="Drive with a trained model or cautious fallback policy.")
    drive_parser.add_argument("--config", default="configs/horizon.toml")
    drive_parser.add_argument("--name", default=DEFAULT_NAME, help="Model/route name used for automatic file paths.")
    drive_parser.add_argument("--type", "--model-type", default=DEFAULT_MODEL_TYPE, help="Model family such as driving, skills, or racing.")
    drive_parser.add_argument("--model", help="Optional explicit model path override.")
    drive_parser.add_argument("--self-train", action="store_true", help="Continuously learn from reward-scored telemetry while driving.")
    drive_parser.add_argument("--online-model", help="Optional explicit self-learning model path override.")
    drive_parser.add_argument("--autosave-frames", type=int, default=300, help="Save the self-learning model after this many updates.")
    drive_parser.add_argument("--online-weight", type=float, default=0.35, help="Blend weight for the self-learning model when it has learned enough.")
    drive_parser.add_argument("--score-weight", type=float, help="Reward weight for skill score/points gains when those fields are available.")
    drive_parser.add_argument("--track")
    drive_parser.add_argument("--terrain-preference", choices=TERRAIN_PREFERENCES, default="auto", help="Terrain reward preference while self-training.")
    drive_parser.add_argument("--driving-mode", choices=list(DRIVING_MODES) + ["auto"], default="auto", help="Driving mode: road, racing, drift, offroad, or mixed. Controls which rewards and penalties are active.")
    drive_parser.add_argument("--explore", action="store_true", default=True, help="Enable curiosity-driven exploration (default on).")
    drive_parser.add_argument("--no-explore", dest="explore", action="store_false", help="Disable all random exploration; model runs purely on what it has learned.")
    drive_parser.add_argument("--transmission", choices=TRANSMISSION_MODES, help="Transmission mode to track while driving.")
    drive_parser.add_argument("--dry-run", action="store_true")
    drive_parser.add_argument("--no-ui", action="store_true")
    drive_parser.set_defaults(func=drive)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except TimeoutError as exc:
        print(f"Timed out waiting for telemetry: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
