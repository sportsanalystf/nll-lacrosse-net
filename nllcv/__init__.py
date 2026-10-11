"""nllcv - computer-vision pipeline for NLL box-lacrosse broadcasts.

Stages (each reads/writes files in a run directory, so any stage can be re-run alone):
    ingest      -> source video, audio.wav, manifest.json
    shots       -> shots.csv, frame_features.npz
    scoreboard  -> scoreboard.csv, spine.csv
    detect      -> detections.csv (+ team labels)
    homography  -> homographies.npz, detections_floor.csv
    events      -> events.csv, possession.csv
    outputs     -> clips/, maps, report.html
"""

__version__ = "0.1.0"
