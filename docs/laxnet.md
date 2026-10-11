# LacrosseNet

HockeyNet-style player tracking for NLL broadcasts, plus ball tracking and the stats that come from it.
Players are segmented, tracked, split into teams and identified by jersey number. They are mapped onto
the floor and smoothed through occlusions. On top of that sit the ball, possession, passes (who to whom),
shots with their context (location, open gap, lane, screens, assist), and player-only stats that the box
score cannot give: speed, sprints, transition speed, defensive gaps, matchups, picks and space control.

```
nllcv run (nllcv/, runs/nllcv/<game>): live shots, scoreboard clock, goals, floor homographies
        │
prep ─► track ─► teams ─► numbers ─► floor ─► ball ─► plays ─► advanced ─► analytics ─► render
JPEGs   SAM 3     SigLIP    Claude     homog.   RF-DETR   holder,    speed, shape,  game_report  video +
per     masks +   + UMAP +  vision,    + Kalman (or SAM 3 passes,    matchups,      .json        radar,
shot    IDs       KMeans    voted      / RTS    zero-shot) shots     space control               ball layer
                                                + tracker
```

| Piece | Here | Based on |
|---|---|---|
| SAM 3 detection + tracking | `track.py` (`sam3`): text prompts "lacrosse player" and "referee", one session per live shot | facebookresearch/sam3 video predictor |
| Team clustering by jersey colour | `teams.py`: SigLIP embeddings → UMAP(3) → KMeans; per-track vote | roboflow/sports `TeamClassifier` |
| Jersey numbers with a VLM | `numbers.py`: 4 clearest torso crops per track → Claude, voted, roster names | Roboflow basketball; Koshkina et al. |
| Floor coordinates + Kalman | `floor.py`: nllcv homographies, constant-velocity Kalman + RTS smoother, χ² gating | — |
| Ball detector | `ball_detect.py`: RF-DETR fine-tuned on our labels, run on full-resolution 640 px tiles at 30 fps; SAM 3 zero-shot until it is trained | the volleyball pipeline (RF-DETR at high resolution on own labels) |
| Camera motion | `ball_detect.py`: phase correlation between frames, so ball speed is measured against the floor, not the screen | the volleyball pipeline |
| Ball tracker | `ball_track.py`: tracklets in camera-stabilised pixels, static clutter dropped, one ball chosen by weighted interval scheduling, short gaps interpolated | — |
| Rules layer | `plays.py`: holder (ball in a stick zone, slow), possessions, passes, turnovers, shots and outcomes, shot context, pass network | the volleyball rules layer |
| Player-only stats | `advanced.py` | — |

## Run on Rivanna (GPU)

```bash
# once, on a login node, from this folder
bash slurm/setup_env.sh                 # conda env 'laxnet': torch cu128, SAM 3, RF-DETR, transformers, umap
#   request access at https://huggingface.co/facebook/sam3, then:
conda activate laxnet && hf auth login
echo 'export ANTHROPIC_API_KEY=sk-...' > ~/.laxnet_env && chmod 600 ~/.laxnet_env

# each game (needs an nllcv run of the same video in runs/nllcv/<game>, done through `homography`)
mkdir -p logs
sbatch --account=<allocation> slurm/laxnet_gpu.slurm runs/laxnet/hfx_at_ga_2026-05-02
```

The job runs every stage on one A100-80GB. Outputs land in `runs/<game>/outputs/`. Until the ball
detector is trained, the ball stage uses SAM 3 zero-shot ("lacrosse ball" on tiles, every 3rd frame).

### Train the ball detector (once per arena / broadcast look; more games make it better)

1. **Frames to label**:
   `python -m laxnet ball-sample --run runs/<game>` picks about 600 frames across all live shots and adds
   ball proposals (motion, and the ball track if one exists).
2. **SAM 3 proposals** (optional, GPU): `sbatch --account=<allocation> slurm/ball_prelabel.slurm runs/<game>`.
3. **Review on your Mac** (needs `pip install opencv-python pandas pyyaml`):
   `python -m laxnet ball-review --run runs/<game>`. On each frame:
   - ENTER accepts the highlighted proposal, TAB or 1-9 picks another.
   - A click puts the ball where you click; click again for a second ball.
   - 0 means no ball visible, S skips an unsure frame, U undoes the last ball.
   - D/→ moves on, A/← goes back, and Q saves and quits. It saves after every action, so you can stop any time.
   - The 4× zoom follows the mouse.
   - Expect roughly 1 to 2 hours for 600 frames. Mark frames as "no ball" too, since they teach the model.
4. **Export**: `python -m laxnet ball-export --run runs/<game>` writes `datasets/ball_<game>/` in Roboflow COCO
   layout. It cuts 640 px tiles at full resolution and splits them by camera shot. As an alternative to
   step 3, upload that folder to Roboflow, label there, and export COCO into the same layout.
5. **Train**: copy `datasets/ball_<game>/` to Rivanna, then
   `sbatch --account=<allocation> slurm/ball_train.slurm datasets/ball_<game>`. The weights go to
   `models/ball_rfdetr/checkpoint_best_total.pth`, and the next `laxnet_gpu.slurm` run uses them automatically.

To re-run only the ball part after training:
`python -m laxnet run --run runs/<game> --from-stage ball` (or inside an `srun --gres=gpu:1` session).

## Run on a Mac / CPU (development)

```bash
python -m laxnet run --run runs/<game> --backend cpu --embedder hsv --detector motion
```

`cpu` replaces SAM 3 with YOLOX plus a Kalman/IoU tracker, and `motion` replaces the ball detector with
camera-compensated frame differencing. Both stand in so the rest of the pipeline can be developed. The motion ball
detector mostly finds stick heads, hands and shoes, so ball-based outputs from it are **not real
results**: the report marks them `reliable: false` and the render leaves the ball layer out.
Single stages: `python -m laxnet <stage> --run runs/<game>`; `--shots 7 10` limits work to some shots.

## Outputs

`runs/<game>/outputs/`

| File | Contents |
|---|---|
| `video/laxnet_render.mp4` | annotated video: outlines, name tags, radar; with a real ball track also the ball with trail, a ring under the carrier, pass arrows and shot lines with the result |
| `player_stats.csv`, `team_stats.csv`, `heatmap_<team>.png` | time, distance, speed (ft/s and mph), spacing, depth, offensive-half share |
| `advanced/player_motion.csv` | per player: top speed and best 1-s speed (mph), distance per minute, high-speed and sprint distance, sprints, accelerations |
| `advanced/stints.csv` | identified players' on-floor stints from sightings (experimental) |
| `advanced/transitions.csv` | team moving end to end: seconds from its defensive zone to the offensive zone, centroid speed |
| `advanced/defense_shape.csv` | while defending: spacing, area, widest seam between neighbouring defenders, slot-to-nearest-defender, how often the slot is open (> 8 ft) |
| `advanced/matchups.csv` | who guarded whom (optimal assignment per frame), time together, mean separation, switches |
| `advanced/picks.csv` | heuristic picks: screener, user, defender, separation before/after |
| `advanced/space_control.csv`, `space_control_<shot>.png` | share of the visible floor and of the scoring area each team reaches first |
| `ball/possessions.csv` | who had the ball, from when to when |
| `ball/passes.csv` | passer → receiver, flight time, distance, forward/back/lateral, defenders bypassed, cross-crease, into the slot |
| `ball/turnovers.csv` | interceptions and loose balls, who lost and who won it, where |
| `ball/shots.csv` | shooter, result (goal / saved / blocked / not seen), distance and angle, zone, nearest defender, **open gap** around the shooter (ft and degrees), defenders in the lane, screens, catch-to-release, shooter speed, assist and whether it crossed the crease, transition shot |
| `ball/carrier_pressure.csv` | per ball carrier: mean distance to the nearest defender, share of time under pressure (≤ 5 ft) |
| `ball/pass_network_<team>.png`, `pass_network_edges.csv`, `shot_map.png` | pass network on the floor; shots with lines to goal, sized by the open gap |
| `game_report.json` | all of it with definitions, for the Coach Biscuit-style chat (next step) |

Per-frame data at the run root: `tracks.csv`, `teams.csv`, `numbers.csv`, `trajectories.csv`,
`ball_candidates.csv`, `ball_track.csv`, `ball_possession_frames.csv`.

## Settings worth knowing (`configs/laxnet.yaml`)

- `prep.track_fps`: frames per second tracked and rendered (15; 30 matches broadcast smoothness).
- `ball.detector`: `rfdetr` | `sam3` | `motion`. The fallback order is rfdetr (if weights exist) → sam3 (on a
  GPU) → motion. `ball.fps` is 30; `ball.sam3_fps` is 10.
- `plays.*`: the rules, i.e. the stick zone, how fast a ball must move to count as in flight, shot speed and
  cone, assist window, lane width, pressure distance. They are measured in player heights per second where they
  are about the image, so they hold at any zoom.
- `advanced.*`: speed bands (14 ft/s high speed, 20 ft/s sprint), pick heuristic, space-control grid.
- `numbers.model`, `numbers.roster`, `render.team_colors`, `render.watermark`, `render.ball`.

## Limits

- Floor positions exist only where the nllcv stage verified a floor homography (left-end views on the Gas
  South test clip), so the floor-based stats cover those views. The radar says when a view has none.
- A broadcast shows part of the floor and part of each team. Defensive shape, matchups and space control
  use the players in view, and each table reports how many were tracked.
- Speeds come from homography + Kalman smoothing: good for comparing players and spotting sprints, roughly
  ±10–15 % in absolute terms.
- The ball is often hidden in a pocket. Possession is carried through those stretches, and a pass is only
  counted when the ball leaves one stick and reaches a teammate's.
- Body pose (RTMPose in the volleyball pipeline) is not used yet. Stick-side, shooting form and dodges
  would need it, and it is the natural next addition.
