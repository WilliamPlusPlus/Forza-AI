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
forza-ai drive --name highway-loop --type racing --self-train --terrain-preference road
forza-ai drive --name field-route --type skills --self-train --terrain-preference offroad
```

The drive screen accepts commands while it runs:

```text
p pause | r resume | n neutral | s status | h help | q quit
```

Press `Ctrl+C` to stop. The driver sends neutral controls on exit.

The dashboard shows the configured transmission mode and checks telemetry for gear/clutch behavior. Horizon telemetry does not expose the assist menu directly, so `--transmission` is the source of truth; clutch input is used as a sanity check for manual-with-clutch behavior.

The dashboard also shows inferred terrain as `road`, `offroad`, `mixed`, or `unknown`. The inference uses wheel rumble, surface rumble, puddle depth, tire slip, speed, and movement. Recordings include derived terrain metadata such as `terrain_state` and `terrain_confidence`.

Test without sending controller input:

```powershell
forza-ai drive --name open-road --dry-run
```

Let the driver keep learning while it runs:

```powershell
forza-ai drive --name open-road --self-train
```

Self-training scores each action against the next telemetry frame. It rewards movement through the world, forward motion, speed gain, fast RPM climb through the useful band below redline, small high-speed bonuses, and clean upshifts that stay below redline. It strongly punishes holding throttle near or over redline, along with tire slip, lateral sliding, spinning, driving-line error, throttle/brake conflict, wasted throttle that does not increase speed or movement, and stalled throttle. The online model is saved periodically and again when the drive loop exits.

Terrain preference changes self-training rewards: racing/road preference rewards clean road movement and heavily punishes off-road. In road mode, the off-road penalty is weighted higher than acceleration, progress, and speed rewards. Offroad preference rewards controlled off-road movement; mixed preference avoids strong terrain rewards or penalties.

Learning also includes car identity and engine context from telemetry, including `car_ordinal`, max/idle RPM, cylinders, class, performance index, and drivetrain type, so separate cars can learn different behavior.

The program also estimates redline per car while telemetry streams. It records `learned_redline_rpm`, `learned_redline_confidence`, and `max_observed_rpm`; redline punishment uses the learned value once confidence is high enough and falls back to `engine_max_rpm` while warming up.

Horizon note: free roam can report `is_race_on = 0`, so the program treats Horizon packets as drivable when speed or live inputs show the car is actually moving or being controlled.

Skill score note: the standard Horizon UDP packet does not appear to include on-screen skill score or skill points. The learner is ready for them anyway: if a frame contains `skill_score`, `skill_points`, `skill_chain`, `score`, or `points`, self-training rewards increases heavily so the driver learns to chase them. Use `--score-weight` to make that reward stronger:

```powershell
forza-ai drive --name airport-drift --type skills --self-train
```

For `--type skills`, the default score reward is stronger automatically. The online learner saves to `models/<type>/<name>-online.joblib`.

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
- `forza_ai.paths`: name-based recording and model paths.
- `forza_ai.terrain`: road/off-road inference, metadata, and terrain reward preferences.
- `forza_ai.transmission`: configured transmission mode and clutch/gear telemetry checks.
- `forza_ai.controller`: virtual Xbox controller output.
- `forza_ai.terminal_ui`: interactive terminal dashboard and runtime commands.
- `forza_ai.trainer`: model training from recorded telemetry.
- `forza_ai.cli`: commands for recording, training, and driving.
- `docs/design.md`: design notes for the Horizon-first architecture.
