from __future__ import annotations

import argparse
import sys
import socket
from pathlib import Path

from .config import load_config
from .controller import Controls
from .controller import create_controller
from .learning import DRIVING_MODES, OnlineDrivingPolicy, _TORCH_AVAILABLE, resolve_driving_mode
from .paths import DEFAULT_MODEL_TYPE, DEFAULT_NAME, data_path, model_path, online_model_path
from .policy import CautiousFallbackPolicy, LearnedPolicy, SmoothPolicy
from .redline import RedlineEstimator
from .reward_config import default_reward_profile_path, load_reward_profile
from .terminal_ui import DashboardState, TerminalDashboard, normalize_command
from .telemetry import TelemetryReceiver, append_frame, is_driving_frame
from .terrain import TERRAIN_PREFERENCES, enrich_terrain, resolve_terrain_preference
from .session_log import SessionLogger
from .trainer import train_model
from .transmission import TRANSMISSION_MODES, ShiftAdvisor, normalize_transmission_mode
from .user_input import KeyboardOverrideReader, UserOverride, telemetry_user_override
from .vision import create_visual_cue_reader, default_vision_profile_path, list_vision_screens
from .vision_training import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_LABELS_PATH,
    DEFAULT_SAMPLE_ROOT,
    VisionTrainingSampler,
    annotate_vision_samples,
)


def _resolve_vision_enabled(args: argparse.Namespace, config_value: bool | None) -> bool | None:
    requested = getattr(args, "vision_enabled", None)
    if requested is not None:
        return bool(requested)
    return config_value


def _resolve_training_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "train_enabled", True))


def _select_user_override(
    *,
    args: argparse.Namespace,
    keyboard_reader: KeyboardOverrideReader,
    frame,
    last_program_controls: Controls | None,
) -> UserOverride | None:
    if not getattr(args, "user_override", True):
        return None
    if getattr(args, "keyboard_override", True):
        keyboard = keyboard_reader.poll()
        if keyboard is not None:
            return keyboard
    if getattr(args, "telemetry_override", True):
        return telemetry_user_override(
            frame,
            last_program_controls,
            difference_threshold=getattr(args, "override_difference_threshold", 0.08),
        )
    return None


def record(args: argparse.Namespace) -> int:
    output_path = args.out or data_path(args.name, args.type)
    track = args.track or args.name
    config = load_config(args.config)
    transmission_mode = normalize_transmission_mode(args.transmission or config.drive.transmission_mode)
    terrain_preference = resolve_terrain_preference(args.type, args.terrain_preference)
    vision_path = args.vision_profile or config.learning.vision_profile or default_vision_profile_path(config.telemetry.profile)
    vision_reader = create_visual_cue_reader(
        vision_path,
        enabled=_resolve_vision_enabled(args, config.learning.vision_enabled),
        target_mode=getattr(args, "vision_target", None),
        screen_index=getattr(args, "vision_screen", None),
        window_title=getattr(args, "vision_window_title", None),
    )
    receiver = TelemetryReceiver(
        config.telemetry.host,
        config.telemetry.port,
        config.telemetry.profile,
        config.telemetry.timeout_seconds,
    )
    seen = 0
    saved = 0
    dashboard = TerminalDashboard(
        DashboardState(
            mode="record",
            target=f"{config.telemetry.profile} UDP {config.telemetry.host}:{config.telemetry.port}",
            transmission_mode=transmission_mode,
            terrain_preference=terrain_preference,
            message=f"Writing {output_path}; {vision_reader.status}",
        ),
        enabled=not args.no_ui,
    )
    dashboard.start()
    if args.no_ui:
        print(
            f"Listening on UDP {config.telemetry.host}:{config.telemetry.port}; "
            f"transmission={transmission_mode}; terrain={terrain_preference}; "
            f"{vision_reader.status}; writing {output_path}"
        )
    previous_frame = None
    redline_estimator = RedlineEstimator()
    try:
        for frame in receiver.frames(track):
            redline_estimator.enrich(frame)
            seen += 1
            vision_reader.enrich(frame, seen)
            enrich_terrain(frame, previous_frame)
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
    dashboard.stop(f"Stopped recording after {saved} saved frames.")
    return 0


def train(args: argparse.Namespace) -> int:
    input_path = Path(args.input) if args.input else data_path(args.name, args.type)
    output_path = Path(args.model) if args.model else model_path(args.name, args.type)
    track = args.track if args.track is not None else (args.name if args.input is None else None)
    result = train_model(input_path, output_path, track, args.track_ordinal, args.min_samples)
    print(f"trained {result['samples']} frames -> {result['model']}")
    return 0


def vision_screens(args: argparse.Namespace) -> int:
    screens = list_vision_screens()
    if not screens:
        print("No vision screens detected. On Windows, make sure the app can access the desktop.")
        return 1
    print("Vision screen indexes:")
    for screen in screens:
        print(
            "  {index}: {width}x{height} at left={left}, top={top}".format(**screen)
        )
    print("Use one with: --vision-target screen --vision-screen <index>")
    return 0


def annotate_vision(args: argparse.Namespace) -> int:
    return annotate_vision_samples(
        args.session,
        root=args.root,
        labels_path=args.labels,
        calibration_path=args.calibration,
        use_ui=not args.no_ui,
    )


def drive(args: argparse.Namespace) -> int:
    base_model_path = Path(args.model) if args.model else model_path(args.name, args.type)
    online_path = Path(args.online_model) if args.online_model else online_model_path(args.name, args.type)
    track = args.track if args.track is not None else args.name
    score_weight = args.score_weight
    if score_weight is None:
        score_weight = 2.0 if args.type.lower() in {"skill", "skills"} else 1.0
    driving_mode = resolve_driving_mode(args.type, getattr(args, "driving_mode", "auto"))
    config = load_config(args.config)
    reward_profile_path = args.reward_profile or config.learning.reward_profile or default_reward_profile_path(config.telemetry.profile)
    reward_profile = load_reward_profile(reward_profile_path)
    vision_path = args.vision_profile or config.learning.vision_profile or default_vision_profile_path(config.telemetry.profile)
    vision_enabled = _resolve_vision_enabled(args, config.learning.vision_enabled)
    vision_reader = create_visual_cue_reader(
        vision_path,
        enabled=vision_enabled,
        target_mode=getattr(args, "vision_target", None),
        screen_index=getattr(args, "vision_screen", None),
        window_title=getattr(args, "vision_window_title", None),
    )
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
    train_enabled = _resolve_training_enabled(args)
    online_policy = None
    if train_enabled:
        if not _TORCH_AVAILABLE:
            print(
                "Learning is enabled by default, but PyTorch is not installed. "
                "Install it with `python -m pip install torch`, or run with --no-train.",
                file=sys.stderr,
            )
            return 1
        steering_weight = (
            args.steering_weight
            if args.steering_weight is not None
            else reward_profile.path_weight("steering", 1.5)
        )
        speed_weight = args.speed_weight if args.speed_weight is not None else reward_profile.path_weight("speed", 0.8)
        terrain_weight = (
            args.terrain_weight
            if args.terrain_weight is not None
            else reward_profile.path_weight("terrain", 1.0)
        )
        achievement_weight = (
            args.achievement_weight
            if args.achievement_weight is not None
            else reward_profile.path_weight("achievement", 1.0)
        )
        online_policy = OnlineDrivingPolicy(
            base,
            online_path,
            autosave_frames=args.autosave_frames,
            online_weight=args.online_weight,
            score_weight=score_weight,
            terrain_preference=terrain_preference,
            driving_mode=driving_mode,
            reward_profile=reward_profile,
            steering_weight=steering_weight,
            speed_weight=speed_weight,
            terrain_weight=terrain_weight,
            achievement_weight=achievement_weight,
            # Flatten exploration/entropy knobs when disabled
            exploration_enabled=explore,
            epsilon=reward_profile.number("online.epsilon", 0.15) if explore else 0.0,
            epsilon_min=reward_profile.number("online.epsilon_min", 0.05) if explore else 0.0,
            exploration_std=reward_profile.number("online.exploration_std", 0.18) if explore else 0.0,
            min_exploration_std=reward_profile.number("online.min_exploration_std", 0.04) if explore else 0.0,
        )
        base = online_policy
    session_logger = None
    if online_policy is not None:
        session_logger = SessionLogger(
            online_path=online_path,
            driving_mode=driving_mode,
            terrain_preference=terrain_preference,
            transmission=transmission_mode,
            n_features=len(online_policy.features),
            interval_frames=args.autosave_frames,
        )
    vision_sampler = VisionTrainingSampler(
        vision_reader,
        enabled=bool(getattr(args, "vision_sampling", True)) and bool(vision_reader.enabled),
        root=getattr(args, "vision_sample_dir", DEFAULT_SAMPLE_ROOT),
        min_interval_seconds=getattr(args, "vision_sample_min_seconds", 3.0),
        max_interval_seconds=getattr(args, "vision_sample_max_seconds", 7.0),
        session_log=session_logger.path if session_logger is not None else None,
    )
    if session_logger is not None and vision_sampler.session_dir is not None:
        try:
            with session_logger.path.open("a", encoding="utf-8") as handle:
                handle.write(f"\nVISION SAMPLES  dir={vision_sampler.session_dir}\n")
        except OSError:
            pass
    policy = SmoothPolicy(
        base,
        max_steer_delta=config.drive.max_steer_delta,
        max_throttle_delta=config.drive.max_throttle_delta,
        max_brake_delta=config.drive.max_brake_delta,
    )
    shift_advisor = ShiftAdvisor(transmission_mode)
    controller_kind = "dry-run" if args.dry_run else config.drive.controller
    controller = create_controller(controller_kind)
    keyboard_reader = KeyboardOverrideReader(
        enabled=getattr(args, "user_override", True) and getattr(args, "keyboard_override", True)
    )
    dashboard = TerminalDashboard(
        DashboardState(
            mode="drive",
            target=f"{config.telemetry.profile} UDP {config.telemetry.host}:{config.telemetry.port} -> {controller_kind}",
            transmission_mode=transmission_mode,
            terrain_preference=terrain_preference,
            message=(
                f"Learning into {online_path}"
                if online_policy is not None
                else "Learning disabled for this run"
            ),
        ),
        enabled=not args.no_ui,
    )
    dashboard.start()
    if args.no_ui:
        print(
            f"Driving from telemetry with {transmission_mode} transmission mode "
            f"and {terrain_preference} terrain preference. "
            f"Rewards={reward_profile.name}; "
            f"learning={'enabled -> ' + str(online_path) if online_policy is not None else 'disabled'}; "
            f"{vision_reader.status}; "
            f"vision samples={vision_sampler.session_dir if vision_sampler.session_dir is not None else 'disabled'}. "
            "Press Ctrl+C to stop."
        )
    seen = 0
    previous_frame = None
    last_program_controls: Controls | None = None
    override_hold_frames = 0
    last_override_source = ""
    redline_estimator = RedlineEstimator()
    previous_learning_frame = None
    previous_learning_controls = None
    try:
        for frame in receiver.frames(track):
            redline_estimator.enrich(frame)
            seen += 1
            vision_reader.enrich(frame, seen)
            enrich_terrain(frame, previous_frame)
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
                last_program_controls = Controls()
                dashboard.update(frame=frame, controls=Controls(), frames_seen=seen)
                previous_learning_frame = None
                previous_learning_controls = None
                continue
            if is_driving_frame(frame):
                sample = vision_sampler.maybe_capture(frame, seen)
                if sample is not None and args.no_ui:
                    print(f"vision sample {sample['sample_id']} -> {sample['roi_path']}")
                if online_policy is not None and previous_learning_frame is not None and previous_learning_controls is not None:
                    reward = online_policy.learn(previous_learning_frame, frame, previous_learning_controls)
                    if session_logger is not None:
                        session_logger.record(
                            frame_num=seen,
                            reward=reward,
                            policy=online_policy,
                            speed_ms=float(frame.values.get("speed", 0.0) or 0.0),
                            terrain_state=str(frame.values.get("terrain_state", "unknown")),
                        )
                    if args.no_ui and args.autosave_frames > 0 and online_policy.updates % args.autosave_frames == 0:
                        print(f"learned {online_policy.updates} updates; last reward {reward.total:+.3f}")
                override = _select_user_override(
                    args=args,
                    keyboard_reader=keyboard_reader,
                    frame=frame,
                    last_program_controls=last_program_controls,
                )
                holding_override = False
                if override is not None:
                    override_hold_frames = max(0, int(getattr(args, "override_release_frames", 24) or 0))
                    last_override_source = override.source
                    controls = override.controls.clipped()
                    frame.values["user_override_active"] = 1
                    frame.values["user_override_source"] = override.source
                    if override.apply_to_controller:
                        controller.apply(controls)
                        last_program_controls = controls
                    else:
                        controller.neutral()
                        last_program_controls = Controls()
                    if hasattr(policy, "previous"):
                        policy.previous = last_program_controls
                    learn_this_frame = True
                elif override_hold_frames > 0:
                    override_hold_frames -= 1
                    holding_override = True
                    controls = Controls()
                    frame.values["user_override_active"] = 1
                    frame.values["user_override_source"] = f"{last_override_source} hold" if last_override_source else "user override hold"
                    controller.neutral()
                    last_program_controls = Controls()
                    if hasattr(policy, "previous"):
                        policy.previous = Controls()
                    learn_this_frame = False
                else:
                    frame.values["user_override_active"] = 0
                    controls = shift_advisor.apply(policy.predict(frame), frame)
                    controller.apply(controls)
                    last_program_controls = controls
                    learn_this_frame = True
                reward_message = None
                if online_policy is not None and online_policy.last_reward is not None:
                    reward_message = (
                        f"Self-train updates {online_policy.updates}; "
                        f"reward {online_policy.last_reward.total:+.3f}"
                    )
                if override is not None:
                    suffix = f"User override: {override.source}"
                    reward_message = f"{reward_message}; {suffix}" if reward_message else suffix
                elif holding_override:
                    suffix = f"User override hold: {last_override_source}"
                    reward_message = f"{reward_message}; {suffix}" if reward_message else suffix
                dashboard.update(frame=frame, controls=controls, frames_seen=seen, message=reward_message)
                if learn_this_frame:
                    previous_learning_frame = frame
                    previous_learning_controls = controls
                else:
                    previous_learning_frame = None
                    previous_learning_controls = None
            else:
                controller.neutral()
                last_program_controls = Controls()
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
        if online_policy is not None and online_policy.updates:
            online_policy.save()
        if session_logger is not None:
            session_logger.close(total_frames=seen, policy=online_policy)
            if args.no_ui:
                print(f"Session log written to {session_logger.path}")
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
    record_parser.add_argument("--vision-profile", help="JSON screen-cue/OCR profile. Defaults to the game config.")
    record_parser.add_argument("--vision", dest="vision_enabled", action="store_true", help="Enable configured visual cues while recording.")
    record_parser.add_argument("--no-vision", dest="vision_enabled", action="store_false", help="Disable visual cues while recording.")
    record_parser.add_argument("--vision-target", choices=("desktop", "screen", "window"), help="What OCR/vision follows for this run.")
    record_parser.add_argument("--vision-screen", type=int, help="Screen number to follow when --vision-target screen is used.")
    record_parser.add_argument("--vision-window-title", "--vision-app", help="Window/app title to follow when --vision-target window is used.")
    record_parser.add_argument("--limit", type=int)
    record_parser.add_argument("--no-ui", action="store_true")
    record_parser.set_defaults(vision_enabled=None)
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

    screens_parser = sub.add_parser("vision-screens", help="List detected screen indexes for OCR/vision capture.")
    screens_parser.set_defaults(func=vision_screens)

    annotate_parser = sub.add_parser("annotate-vision", help="Label saved road/dirt vision samples and rebuild calibration.")
    annotate_parser.add_argument("--session", default="latest", help="Vision sample session directory, or latest.")
    annotate_parser.add_argument("--root", default=str(DEFAULT_SAMPLE_ROOT), help="Vision sample root directory.")
    annotate_parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH), help="Global JSONL label store to update.")
    annotate_parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION_PATH), help="Calibration JSON file to rebuild.")
    annotate_parser.add_argument("--no-ui", action="store_true", help="Use terminal prompts instead of the small labeling window.")
    annotate_parser.set_defaults(func=annotate_vision)

    drive_parser = sub.add_parser("drive", help="Drive with a trained model or cautious fallback policy.")
    drive_parser.add_argument("--config", default="configs/horizon.toml")
    drive_parser.add_argument("--name", default=DEFAULT_NAME, help="Model/route name used for automatic file paths.")
    drive_parser.add_argument("--type", "--model-type", default=DEFAULT_MODEL_TYPE, help="Model family such as driving, skills, or racing.")
    drive_parser.add_argument("--model", help="Optional explicit model path override.")
    drive_parser.add_argument(
        "--train",
        "--self-train",
        dest="train_enabled",
        action="store_true",
        help="Continuously learn from reward-scored telemetry while driving (default on).",
    )
    drive_parser.add_argument(
        "--no-train",
        dest="train_enabled",
        action="store_false",
        help="Do not update or save the online self-learning model during this run.",
    )
    drive_parser.add_argument("--online-model", help="Optional explicit self-learning model path override.")
    drive_parser.add_argument("--autosave-frames", type=int, default=300, help="Save the self-learning model after this many updates.")
    drive_parser.add_argument("--online-weight", type=float, default=0.35, help="Blend weight for the self-learning model when it has learned enough.")
    drive_parser.add_argument("--reward-profile", help="JSON reward/punishment profile. Defaults to the game config.")
    drive_parser.add_argument("--vision-profile", help="JSON screen-cue/OCR profile. Defaults to the game config.")
    drive_parser.add_argument("--vision", dest="vision_enabled", action="store_true", help="Enable configured visual cues while driving.")
    drive_parser.add_argument("--no-vision", dest="vision_enabled", action="store_false", help="Disable visual cues while driving.")
    drive_parser.add_argument("--vision-target", choices=("desktop", "screen", "window"), help="What OCR/vision follows for this run.")
    drive_parser.add_argument("--vision-screen", type=int, help="Screen number to follow when --vision-target screen is used.")
    drive_parser.add_argument("--vision-window-title", "--vision-app", help="Window/app title to follow when --vision-target window is used.")
    drive_parser.add_argument("--vision-sampling", dest="vision_sampling", action="store_true", default=True, help="Save periodic road-region screenshots for manual labeling (default on).")
    drive_parser.add_argument("--no-vision-sampling", dest="vision_sampling", action="store_false", help="Do not save manual-label vision samples this run.")
    drive_parser.add_argument("--vision-sample-dir", default=str(DEFAULT_SAMPLE_ROOT), help="Directory for periodic vision training screenshots.")
    drive_parser.add_argument("--vision-sample-min-seconds", type=float, default=3.0, help="Minimum seconds between vision training screenshots.")
    drive_parser.add_argument("--vision-sample-max-seconds", type=float, default=7.0, help="Maximum seconds between vision training screenshots.")
    drive_parser.add_argument("--score-weight", type=float, help="Reward weight for skill score/points gains when those fields are available.")
    drive_parser.add_argument("--track")
    drive_parser.add_argument("--terrain-preference", choices=TERRAIN_PREFERENCES, default="auto", help="Terrain reward preference while learning.")
    drive_parser.add_argument("--driving-mode", choices=list(DRIVING_MODES) + ["auto"], default="auto", help="Driving mode: road, racing, drift, offroad, or mixed. Controls which rewards and penalties are active.")
    drive_parser.add_argument("--steering-weight", type=float, help="Override the reward profile's Steering path multiplier.")
    drive_parser.add_argument("--speed-weight", type=float, help="Override the reward profile's Speed path multiplier.")
    drive_parser.add_argument("--terrain-weight", type=float, help="Override the reward profile's Terrain path multiplier.")
    drive_parser.add_argument("--achievement-weight", type=float, help="Override the reward profile's Achievement path multiplier.")
    drive_parser.add_argument("--explore", action="store_true", default=True, help="Enable curiosity-driven exploration (default on).")
    drive_parser.add_argument("--no-explore", dest="explore", action="store_false", help="Disable all random exploration; model runs purely on what it has learned.")
    drive_parser.add_argument("--user-override", dest="user_override", action="store_true", default=True, help="Let keyboard or human telemetry input override AI control (default on).")
    drive_parser.add_argument("--no-user-override", dest="user_override", action="store_false", help="Disable human-input override while driving.")
    drive_parser.add_argument("--keyboard-override", dest="keyboard_override", action="store_true", default=True, help="Let WASD/arrow keyboard input override AI control (default on).")
    drive_parser.add_argument("--no-keyboard-override", dest="keyboard_override", action="store_false", help="Disable keyboard polling override.")
    drive_parser.add_argument("--telemetry-override", dest="telemetry_override", action="store_true", default=True, help="Let human controller telemetry override AI control when it differs from program output (default on).")
    drive_parser.add_argument("--no-telemetry-override", dest="telemetry_override", action="store_false", help="Disable telemetry-based human override detection.")
    drive_parser.add_argument("--override-difference-threshold", type=float, default=0.08, help="How different telemetry input must be from program output before it counts as human override.")
    drive_parser.add_argument("--override-release-frames", type=int, default=24, help="Neutral frames after user override ends so the model does not fight back immediately.")
    drive_parser.add_argument("--transmission", choices=TRANSMISSION_MODES, help="Transmission mode to track while driving.")
    drive_parser.add_argument("--dry-run", action="store_true")
    drive_parser.add_argument("--no-ui", action="store_true")
    drive_parser.set_defaults(vision_enabled=None)
    drive_parser.set_defaults(train_enabled=True)
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
