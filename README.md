# Forza AI

Experimental Python program for learning Forza Horizon driving behavior from Data Out telemetry and replaying it through a virtual Xbox controller.

The primary target is Forza Horizon using the Data Out `Dash` packet profile. Motorsport support is still available through `configs/motorsport.toml`, but the default commands now favor Horizon. This is a training scaffold, not a finished self-driving driver. Start with `record` while you drive clean routes, train a named model from those runs, then run the model in assisted mode on the same area. Keep the game in a private/free-roam environment while testing.

## Game Setup

Forza Horizon:

1. Turn `Data Out` on in the HUD/gameplay settings.
2. Set IP to `127.0.0.1`.
3. Set port to `9876`.
4. Use the default Horizon config when recording or driving.

Forza Motorsport, fallback:

1. Open `Settings > Gameplay & HUD > UDP Race Telemetry`.
2. Turn `Data Out` on.
3. Set IP to `127.0.0.1`.
4. Set port to `9876`.
5. Set packet format to `Dash` / `Car Dash`.
6. Pass `--config configs/motorsport.toml`.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

`vgamepad` requires the ViGEmBus/virtual gamepad driver on Windows.

OCR and visual screen cues are optional helper inputs. To use them, install the
vision extra and install the Tesseract OCR app on Windows:

```powershell
python -m pip install -e ".[vision]"
```

The program still runs without those pieces; it will keep telemetry learning active and mark vision as unavailable.

## Record Training Data

Record a route or area by name:

```powershell
forza-ai record --name open-road
```

That writes to `data/driving/open-road.jsonl`.

While recording, the terminal dashboard accepts commands:

```text
p pause | r resume | s status | h help | q quit
```

Use model types to keep separate behavior families:

```powershell
forza-ai record --name open-road --type racing
forza-ai record --name airport-drift --type skills
```

Choose how road/off-road behavior should be treated:

```powershell
forza-ai record --name highway-loop --type racing --terrain-preference road
forza-ai record --name field-route --type skills --terrain-preference offroad
```

`--terrain-preference auto` is the default. It resolves `racing` to `road`; `skills` and other model types use `mixed`.

The Horizon config enables visual cue capture by default through `configs/vision/horizon.json`. These cues are saved into recordings as numeric fields, including screen-region brightness/motion and OCR-derived skill score values when OCR is available. Disable it for a run with:

```powershell
forza-ai record --name open-road --no-vision
```

By default, the Horizon vision profile follows a window titled `Forza Horizon 5` and falls back to the desktop if that window is not found. You can point OCR/vision at a specific screen, app window, or the full desktop:

```powershell
forza-ai vision-screens
forza-ai record --name open-road --vision-target screen --vision-screen 1
forza-ai record --name open-road --vision-target window --vision-app "Forza Horizon 5"
forza-ai record --name open-road --vision-target desktop
```

Screen indexes are sorted by desktop position: leftmost screens first, then top to bottom. `vision-screens` prints the exact detected index and bounds.

Record the transmission style with the run:

```powershell
forza-ai record --name highway-manual --type racing --transmission manual
forza-ai record --name clutch-skills --type skills --transmission manual-clutch
```

Motorsport training can also use the official track ordinal saved in Dash packets:

```powershell
forza-ai train --in data/general.jsonl --model models/track-812.joblib --track-ordinal 812
```

## Train

```powershell
forza-ai train --name open-road
forza-ai train --name airport-drift --type skills
forza-ai train --name open-road --type racing
```

Named training uses `data/<type>/<name>.jsonl` and writes `models/<type>/<name>.joblib`.

## Drive

```powershell
forza-ai drive --name open-road
forza-ai drive --name airport-drift --type skills
forza-ai drive --name clutch-skills --type skills --transmission manual-clutch
forza-ai drive --name highway-loop --type racing --terrain-preference road
forza-ai drive --name field-route --type skills --terrain-preference offroad
```

The drive screen accepts commands while it runs:

```text
p pause | r resume | n neutral | s status | h help | q quit
```

Human input override is on by default while driving. If you press `W/A/S/D`, arrow keys, `Space`, `Q`, `E`, or Shift, those controls override the model immediately and become the action used for online learning. If a physical controller/wheel input appears in telemetry and differs from what the program just sent, the program neutralizes its virtual controller output and learns from your telemetry input instead.

Disable override for pure AI-only runs:

```powershell
forza-ai drive --name fh5-driving --type driving --no-user-override
```

Press `Ctrl+C` to stop. The driver sends neutral controls on exit.

The dashboard shows the configured transmission mode and checks telemetry for gear/clutch behavior. Horizon telemetry does not expose the assist menu directly, so `--transmission` is the source of truth; clutch input is used as a sanity check for manual-with-clutch behavior.

The dashboard also shows inferred terrain as `road`, `offroad`, `mixed`, or `unknown`. The inference uses wheel rumble, surface rumble, puddle depth, tire slip, speed, movement, and optional vision surface recognition. It also shows a separate `Vision surface` line so you can see what the screen/object detector actively sees apart from the telemetry-based terrain result. When vision is enabled, configured forward-view regions classify asphalt/road markings versus grass/dirt and add `vision_road_score`, `vision_offroad_score`, lane-marking offset, and related surface fields. Forward road and lane cues dampen nearby shoulder dirt so it is less likely to call a paved road offroad. Recordings include derived terrain metadata such as `terrain_state` and `terrain_confidence`.

Test without sending controller input:

```powershell
forza-ai drive --name open-road --dry-run
```

Learning is enabled by default while driving. Each run loads the existing online model from `models/<type>/<name>-online.joblib`, keeps updating it during the session, autosaves periodically, and saves again when the run exits. Disable learning for a one-off run with:

```powershell
forza-ai drive --name open-road --no-train
```

`--train` and the older `--self-train` flag still work, but they are no longer required.

Reward and punishment tuning is read from JSON at drive startup. Horizon uses `configs/rewards/horizon.json` by default; Motorsport uses `configs/rewards/motorsport.json` when `configs/motorsport.toml` is selected. Override it for experiments with:

```powershell
forza-ai drive --name open-road --reward-profile configs/rewards/horizon.json
```

Self-training scores each action against the next telemetry frame. It rewards movement through the world, forward motion, speed gain, fast RPM climb through the useful band below redline, small high-speed bonuses, clean upshifts that stay below redline, and steady lane holding on road/racing runs. It strongly punishes holding throttle near or over redline, along with tire slip, lateral sliding, spinning, lane drift, driving-line error, throttle/brake conflict, wasted throttle that does not increase speed or movement, and stalled throttle. The online model is saved periodically and again when the drive loop exits.

Terrain preference changes self-training rewards: racing/road preference rewards clean road movement and heavily punishes off-road. In road mode, the off-road penalty is weighted higher than acceleration, progress, and speed rewards. Offroad preference rewards controlled off-road movement; mixed preference avoids strong terrain rewards or penalties.

Road preference also disables wreckage-style skill chasing. If OCR sees wreckage/destruction skill text while road mode is active, the learner ignores the skill-score increase and applies a penalty instead.

Learning also includes car identity and engine context from telemetry, including `car_ordinal`, max/idle RPM, cylinders, class, performance index, and drivetrain type, so separate cars can learn different behavior.

The program also estimates redline per car while telemetry streams. It records `learned_redline_rpm`, `learned_redline_confidence`, and `max_observed_rpm`; redline punishment uses the learned value once confidence is high enough and falls back to `engine_max_rpm` while warming up.

Horizon note: free roam can report `is_race_on = 0`, so the program treats Horizon packets as drivable when speed or live inputs show the car is actually moving or being controlled.

Skill score note: the standard Horizon UDP packet does not appear to include on-screen skill score or skill points. The learner is ready for them anyway: if a frame contains `skill_score`, `skill_points`, `skill_chain`, `score`, or `points`, self-training rewards increases heavily so the driver learns to chase them. The Horizon vision profile can populate `skill_score`, `horizon_skill_score`, `skill_multiplier`, and prompt flags from OCR. Use `--score-weight` to make that reward stronger:

```powershell
forza-ai drive --name airport-drift --type skills
```

For `--type skills`, the default score reward is stronger automatically. The online learner saves to `models/<type>/<name>-online.joblib`.

Disable live visual cues while driving:

```powershell
forza-ai drive --name open-road --no-vision
```

Choose what live OCR/vision follows while driving:

```powershell
forza-ai drive --name fh5-driving --type driving --vision-target screen --vision-screen 1
forza-ai drive --name fh5-driving --type driving --vision-target window --vision-app "Forza Horizon 5"
forza-ai drive --name fh5-driving --type driving --vision-target desktop
```

Disable the interactive terminal dashboard:

```powershell
forza-ai drive --name open-road --no-ui
```

Explicit paths still work as overrides for imports or one-off experiments:

```powershell
forza-ai train --in data/old-recording.jsonl --model models/imported.joblib --name imported
forza-ai drive --model models/imported.joblib
```

## Project Shape

- `forza_ai.telemetry`: UDP listener and Forza packet parsing.
- `forza_ai.policy`: baseline and learned driving policies.
- `forza_ai.learning`: online self-training and reward/punishment scoring.
- `forza_ai.reward_config`: JSON reward profile loading for Horizon and Motorsport.
- `forza_ai.vision`: optional screen capture, OCR, visual cue enrichment, and road/off-road surface recognition.
- `forza_ai.paths`: name-based recording and model paths.
- `forza_ai.terrain`: road/off-road inference, metadata, and terrain reward preferences.
- `forza_ai.transmission`: configured transmission mode and clutch/gear telemetry checks.
- `forza_ai.controller`: virtual Xbox controller output.
- `forza_ai.terminal_ui`: interactive terminal dashboard and runtime commands.
- `forza_ai.trainer`: model training from recorded telemetry.
- `forza_ai.cli`: commands for recording, training, and driving.
- `docs/design.md`: design notes for the Horizon-first architecture.
