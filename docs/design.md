# Forza AI Design

## Primary Target

The primary game target is Forza Horizon on Windows PC. The program expects the official Data Out UDP telemetry feature and uses the Horizon Dash packet profile by default.

The Horizon profile is first-class because it exposes enough structured data for useful learning:

- Car motion: speed, local velocity, acceleration, yaw, pitch, roll, angular velocity.
- Driver inputs: steer, accel, brake, clutch, handbrake, gear.
- Race/free-roam state: race-on flag, lap number, race position, lap timing when available.
- Route state: manual `--track` labels and position.
- Tire behavior: slip ratio, slip angle, combined slip, and temperatures.

## Training Strategy

The first model is imitation learning. You drive clean routes, the recorder saves telemetry and your controller inputs, then the trainer learns to predict those inputs from the telemetry frame.

Dash packets expose `accel` and `brake` as 0-255 driver-input values. The program normalizes these into throttle/brake values from 0.0 to 1.0 for labels, dashboard display, and controller comparison.

Horizon free roam can report `is_race_on = 0` even when usable telemetry is arriving. For Horizon, the program treats a packet as drivable when it has speed/control fields and the speed or live controls indicate real driving activity.

The standard Horizon UDP packet does not appear to expose on-screen skill score or unspent skill points. The reward system still supports score fields from another reader by looking for aliases such as `skill_score`, `skill_points`, `skill_chain`, `score`, and `points`. The Horizon vision profile can add those fields from OCR when the optional screen-capture dependencies are installed.

Car identity is included in learning through telemetry fields such as `car_ordinal`, engine RPM range, cylinder count, car class, performance index, and drivetrain type.

Redline is learned per car during telemetry. Frames are enriched with `learned_redline_rpm`, `learned_redline_confidence`, and `max_observed_rpm`; reward logic uses the learned estimate once confidence is high enough and otherwise falls back to `engine_max_rpm`.

The modes are:

- Named Horizon model: train with a name such as `open-road`, `airport-drift`, or `highway-loop`.
- Typed Horizon model: group different behavior families with `--type`, such as `driving`, `skills`, or `racing`.
- Motorsport fallback model: available with `configs/motorsport.toml` and optional `track_ordinal` filtering.
- Online self-training model: enabled by default during `drive`, scores the previous action after the next telemetry frame arrives, and can be disabled for a run with `--no-train`.

Named paths follow the project structure:

- Recordings: `data/<type>/<name>.jsonl`
- Offline models: `models/<type>/<name>.joblib`
- Online self-training models: `models/<type>/<name>-online.joblib`; each drive session loads this file at startup, autosaves during learning, and saves again on exit.

Transmission mode is configured per run as `automatic`, `manual`, or `manual-clutch`. Horizon telemetry exposes gear and clutch input, but not the assist-menu transmission setting, so the configured mode is the source of truth. Telemetry is used as a sanity check, especially to notice clutch input that suggests manual-with-clutch behavior.

Terrain is inferred from telemetry instead of a direct Horizon road flag. Each frame can be enriched with `terrain_state`, `terrain_confidence`, `terrain_offroad_score`, `terrain_road_score`, `terrain_is_road`, and `terrain_is_offroad`. Telemetry off-road detection deliberately uses suspension spring travel only; tire slip and rumble are not treated as telemetry off-road evidence anymore. Suspension scoring is speed-aware: low-speed travel and chatter use lower thresholds because the car will not bounce as much, while high-speed bumps need a stronger suspension signal before they are treated as offroad. When vision is enabled, the configured forward-view road ROI adds object-recognition-like cues such as `vision_road_score`, `vision_offroad_score`, `vision_surface_is_road`, and `vision_surface_is_offroad`. It also adds lane and road-direction cues such as `vision_lane_center_offset`, `vision_road_center_offset`, and `vision_road_heading`. Forward road and lane evidence dampens nearby dirt/grass so a road shoulder does not overpower the road surface. The terminal dashboard prints both the combined terrain result and a separate `Vision surface` line for the active visual detector. The CLI accepts `--terrain-preference {auto,road,offroad,mixed}` for recording and driving. `auto` resolves `racing` to `road`; `skills` and other types resolve to `mixed`.

Human override has highest priority in the drive loop. Keyboard polling can turn `W/A/S/D`, arrows, `Space`, `Q`, `E`, and Shift into live control targets. Telemetry input that differs from the program's last output is treated as physical user control, so the virtual controller is neutralized and the online learner trains on the user's action instead of the model's action. A short neutral hold follows user release so the model does not immediately counter-steer against the user.

Vision sampling runs during driving when vision is enabled. It saves the full screenshot and masked road ROI at randomized intervals, writes a manifest that includes the session log path and telemetry snapshot, and appends sample ids back into the session log. `forza-ai annotate-vision --session latest` presents a small road/dirt/mixed/crash/skip labeling UI. Labels are upserted by sample id, so manual corrections overwrite older labels, and the surface calibration JSON is rebuilt for the next run.

## Rewards and Punishments

Reward and punishment values are read from JSON profiles at startup. Horizon currently uses `configs/rewards/horizon.json`; Motorsport keeps a fallback profile at `configs/rewards/motorsport.json`. The JSON owns path weights, score multipliers, speed/progress rewards, redline and throttle punishments, terrain punishments, drift bonuses, target-adjustment behavior, and online exploration defaults.

The online learner rewards movement through the world, forward motion, speed gain, throttle that actually produces acceleration, fast RPM climb through the useful band below redline, small sustained high-speed bonuses, and clean upshifts that land below redline. The Horizon profile intentionally gives acceleration a high reward weight so clear-road behavior trends toward commitment instead of caution. Horizon can leave `distance_traveled` at zero, so movement falls back to position delta when needed. Launch throttle has a low-speed grace path so the learner does not become timid just because the first few frames of throttle have not moved the car much yet. Clear road in the visual ROI is also treated as forward intent: moving into it gets a progress bonus, while crawling or braking with clear road ahead gets a timidity penalty and a higher learned throttle target.

Terrain rewards are preference-based: road preference rewards clean road movement and penalizes off-road, but Horizon penalties are capped so a single offroad frame cannot overwhelm many useful acceleration/progress rewards. Offroad preference rewards controlled off-road movement; mixed preference leaves terrain neutral while existing slip/stall penalties still apply. Lane holding is a separate steering reward in road/racing-style modes; it blends Forza driving-line offset, visual lane-marking offset, lateral velocity, and yaw rate so the learner favors steady lane-centered travel instead of wandering across the road. Visual road-direction offset and heading also influence steering rewards and target correction, so the learner can associate road shape with steering before the car leaves the lane.

Road preference also blocks wreckage-style skill chasing. When OCR detects wreckage/destruction skill text in road mode, the score delta is ignored and a wreckage penalty is applied so the model does not learn to farm object hits.

It punishes signals that usually mean the car is being mishandled:

- High tire combined slip.
- Lateral sliding or spinning instead of stable forward motion.
- Large driving-line error.
- Redlining while still applying throttle, with stronger punishment over max RPM.
- Throttle while the AI brake signal suggests braking.
- Throttle/brake overlap.
- Meaningful throttle that does not produce speed gain or world movement.
- Applying throttle while the car is stalled or barely moving.
- Crash-like transitions from sudden speed loss, impact acceleration, or reset prompts.

The reward score changes how strongly the online learner trains on the last action. Negative transitions also adjust the target by reducing throttle, trimming steering, or adding a little braking before the sample is learned. Exploration has two layers: curiosity-driven action bursts for discovering broader behavior, and a low-amplitude entropy layer that continuously samples nearby steering/throttle choices. Saved online models restore a configurable exploration floor at startup so they keep testing unless `--no-explore` is used.

When skill score fields are available, positive score deltas become the strongest reward. This lets the same online learner pivot from "drive cleanly" to "go for skill points" without replacing the control model.

## Runtime Loop

```mermaid
flowchart LR
    A["Forza Horizon Data Out"] --> B["UDP Receiver"]
    B --> C["Dash Packet Parser"]
    S["Screen Capture and OCR"] --> V["Visual Cue Enrichment"]
    C --> V
    V --> D["Driving Policy"]
    D --> E["Smoothing and Safety Clamp"]
    E --> F["Virtual Xbox Controller"]
    E --> H["Reward Scorer and Online Learner"]
    V --> H
    H --> D
    V --> G["Terminal Dashboard"]
    G --> E
```

## Why Telemetry First

Telemetry is the foundation because it is already numeric, stable, and high frequency. Vision is now a helper stream for screen-only information such as skill score, route prompts, reset prompts, wrong-way prompts, road/off-road surface recognition, and region motion. Visual cues are converted into numeric frame fields before learning; the driving policy still avoids raw pixel control. The vision profile can follow the full desktop, a numbered screen, or a window/app title. Horizon defaults to the `Forza Horizon 5` window and falls back to desktop capture if the window cannot be found.

## Safety

- The controller is neutralized when the drive command exits.
- Outputs are clipped to valid controller ranges.
- Steering, throttle, and brake changes are smoothed frame-to-frame.
- `--dry-run` lets the loop run without creating controller input.
- The terminal dashboard can pause/resume, send neutral controls, show status, or stop the run.
