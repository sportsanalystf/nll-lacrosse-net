# nllcv — NLL broadcast computer-vision pipeline

Turns a full NLL broadcast into a clock-indexed game record: segmentation, scoreboard state,
player positions on a floor map, goals/shots/possession, and auto-clipped goal highlights.

```
broadcast ─► 1 ingest ─► 2 shots ─► 3 scoreboard ─► 4 detect ─► 5 homography ─► 6 events ─► 7 outputs
            (yt-dlp)    (cuts +     (OCR → game     (YOLOX +    (floor feet)     (goals,     (clips, maps,
                         labels)     state spine)    jersey k-means)              possession,  event table)
                                                                                  shots)
```

## Setup (macOS)

```bash
brew install ffmpeg tesseract
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install onnxruntime        # optional: uses CoreML on Apple silicon for detection
```

`models/yolox_tiny.onnx` (default) and `models/yolox_s.onnx` are the official YOLOX ONNX exports
(Apache-2.0, github.com/Megvii-BaseDetection/YOLOX). No PyTorch needed.

## Run

```bash
# everything, from a URL (NLL YouTube channel etc.) or a local file
python -m nllcv run --source "https://www.youtube.com/watch?v=..." --run runs/nllcv/hfx_at_ga_2026-05-02

# or stage by stage (each reads/writes files in the run folder, so any stage can be re-run)
python -m nllcv ingest     --run runs/game --source broadcast.mp4
python -m nllcv shots      --run runs/game
python -m nllcv scoreboard --run runs/game
python -m nllcv detect     --run runs/game            # --budget 600 to stop/resume in chunks
python -m nllcv homography --run runs/game
python -m nllcv events     --run runs/game
python -m nllcv outputs    --run runs/game
```

`--force` recomputes cached intermediates (decoded features, per-shot detections).

## What each stage does

| Stage | Method | Output |
|---|---|---|
| 1 ingest | yt-dlp best ≤1080p video + audio, or symlink a local file; mono WAV for audio | `source.mp4`, `audio.wav`, `manifest.json` |
| 2 shots | one 10 fps decode; cuts from structural change (1−NCC of thumbnails) + colour-histogram jumps, dissolves from moderate joint jumps. Each shot is labelled from floor-turf fraction, scorebug presence (learned template), edge density and flat-area fraction | `shots.csv` (`live / replay / closeup / graphics / crowd / studio`) |
| 3 scoreboard | full-res crop of the fixed bug at 2 fps. Digits are read by a self-trained glyph matcher, bootstrapped by Tesseract (~100 Tesseract calls per game instead of thousands). Scores/period need 3 identical reads; clocks must agree with a neighbour under "stopped or running at 1 s/s" | `scoreboard.csv`, `spine.csv` (broadcast second → period, clock, game seconds, score) |
| 4 detect | YOLOX on live wide shots (2 overlapping square tiles), kept only if turf is under the feet; torso colour histogram → k-means (2 teams + officials), clusters named from configured jersey colours | `detections.csv`, `debug/teams.jpg` |
| 5 homography | per-arena reference frames calibrated once (landmarks + chamfer alignment, or a pan-tilt-zoom camera model); every live frame registered to the best reference with ORB on the floor (players masked), then chamfer-polished | `detections_floor.csv` (feet → floor feet), `homographies.npz` |
| 6 events | goals = scoreboard increments placed at the audio spike (horn + crowd) inside the footage that last showed the old score; possession from which end the play is at + which team's goalie defends it; shots from 30 s shot-clock resets | `events.csv`, `possession.csv` |
| 7 outputs | goal clips with pre-roll (never opening on a replay), shot map, possession map + timeline, HTML report | `outputs/` |

## Calibrating a new arena (stage 5)

Floor dimensions come from the config (NLL rulebook: goal 12 ft from the end boards, crease radius
9′3″, restraining lines 42′6″ from centre, corner face-off dots 15 ft inside the restraining line and
15 ft off the boards). Calibration is stored in `calibration/<arena>/` and reused for every game there.

1. **Anchor frame (clicked landmarks).** Pick a wide main-camera frame showing a crease and a
   restraining line:
   ```bash
   python -m nllcv calibrate --run runs/game --at 40
   ```
   Click a point in the window, type its landmark number in the terminal (goal centre, both ends of
   the goal line across the crease, restraining line at the far boards, face-off dots `@-57.5,27.5`,
   …), press `q`. Or pass them non-interactively:
   `--points '{"left_goal":[199,518], "@-57.5,-27.5":[484,866], ...}'`.
2. **More references, automatically.** With the anchor in place, the main camera's position is
   recovered from it and other frames only need pan/tilt/roll/zoom:
   ```bash
   python -m nllcv calibrate --run runs/game --auto --at 12 58 258
   ```
3. Check `calibration/<arena>/*_overlay.jpg`; delete any reference whose lines don't sit on the floor.

Aim for at least one good reference per end plus one centre view per camera that shows play.

## Config

`configs/nll_2026_broadcast.yaml`: team abbreviations and jersey colours, scorebug field boxes
(at 1920×1080, rescaled automatically), shot-detection thresholds, detector model/fps/tiles, floor
dimensions, audio spike thresholds, clip pre/post-roll. A different broadcaster's bug only needs new
boxes under `scorebug.fields`.

## Hand corrections

- `runs/<game>/shot_labels.csv` (`shot_id,label`) overrides shot labels; with ≥10 rows a random
  forest relabels the remaining shots from them.
- Debug images in `runs/<game>/debug/` show the learned bug template, detections, team split and
  homography overlays.

## Floor registration: how it works and what to check

`homography.method: ptz` (default) tracks the main camera's pan / tilt / roll / zoom shot by shot,
with the camera position recovered from the landmark-calibrated anchor reference. Each frame is
fitted against the painted lines, crease and board base and is **rejected** (no floor coordinates)
unless the projected lines sit within `max_line_cost_px` of the image lines, so a bad fit never turns
into a wrong map point. `method: orb` (feature matching to references) is kept as an option but
aliases badly on repeated board ads.

On the Gas South test clip the left end registers well, but right-end views did not fit the
single-camera model: the crease and far boards line up, the right restraining line does not. Two
independent right-end frames agree, so the cause is real (lens distortion at that zoom, a second
camera, or a non-standard floor), not noise. Fix it once per arena by clicking an anchor at the
right end (`calibrate --at <t>` on a wide right-end frame). The tracker then uses the closest anchor.
Frames without floor coordinates still get possession through the image-space fallback (whose goalie
stands in the visible crease).

## Known limits

- The ball is not tracked; shot/goal locations are the attacking player nearest the goal just before
  the event.
- Possession is inferred from player layout and goalie colours; it is unreliable during line changes
  and in close-ups.
- Highlight packages compress audio heavily, so the goal-horn check confirms few goals there; a full
  broadcast with natural crowd sound gives a much clearer rise. Tune `audio.goal_rise` if needed.
- In highlight packages several goals can happen between clips; they are logged at the same moment
  with the note "earlier goal not shown on air" and get a single clip.
- Frames from cameras without a calibrated reference (e.g. a tight centre cam) get no floor coordinates.
