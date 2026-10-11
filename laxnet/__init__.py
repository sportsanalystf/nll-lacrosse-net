"""LacrosseNet - player tracking, teams, jersey numbers and floor trajectories from NLL broadcast video.

Modelled on HockeyNet (SAM 3 tracking, jersey-colour teams, VLM jersey numbers, homography, Kalman
filtering) and Roboflow's open basketball pipeline. It builds on an `nllcv` run of the same video for
live-play segments, the scoreboard clock, goals and the floor homographies.

Stages: prep -> track -> teams -> numbers -> floor -> analytics -> render
"""

__version__ = "0.1.0"
