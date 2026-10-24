# Play tracker

Times the key moments of a batted ball (pitch release, bat contact, fielder arrival) from Baseball Savant clips and MLB tracking data, and charts each play as a timeline.

## How it works

1. **Sample.** Pull Statcast pitch data for a date range and draw a stratified sample of balls in play (grounders, fly balls, line drives, popups).
2. **Clips.** Match each pitch to the MLB game feed to get its play ID and MLB's plate time (time from release to the front of the plate), then download and shrink the broadcast clip.
3. **Auto marks.** Find the bat crack in the clip audio (the sharpest rise in 1.5 to 7 kHz energy). Release is contact minus plate time.
4. **Annotate.** A frame-stepping page opens each clip at the auto contact time. The one manual step is marking the frame where the fielder gets the ball.
5. **Report.** Each play gets a timeline with release at zero, and a CSV of every event time.

## Validation

On 15 plays marked by hand, auto contact and release were within about one to two frames of the hand marks (median error near 0 ms, mean absolute error under 30 ms). The hand-marked release-to-contact interval also matched MLB's plate time to within about 13 ms on average across 24 plays. `compare` reproduces the check.

## Setup

    pip install -r requirements.txt

`imageio-ffmpeg` bundles ffmpeg, so nothing else is needed.

## Run

    python pipeline.py sample --start 2026-09-01 --end 2026-09-14 --n 25 --team DET
    python pipeline.py clips
    python pipeline.py auto
    python pipeline.py annotate
    # open annotate.html in Chrome, press F at each fielder, click Export annotations.json,
    # and save it as data/annotations.json
    python pipeline.py compare     # optional: auto marks vs your hand marks
    python pipeline.py clean       # optional: delete clips you have marked
    python pipeline.py report

In a Jupyter notebook, call `pipeline.main([...])` with the same arguments, for example
`pipeline.main(["clips", "--limit", "10"])`. Always pass a list.

Outputs: `data/events.csv` (one row per play) and `report.html` (a timeline per play).

## Notes

- Fielder arrival is the one manual mark. Public data does not time it, and tracking a small, blurred ball in broadcast video needs a trained model.
- In the report, filled markers are measured from video or audio, hollow markers are MLB timing data or a ball-flight estimate.
- Hand marks always override auto marks.
- Clips shrink to about 0.5 MB each at 360p with audio kept (`--height 0` keeps originals).
- Broadcast cuts away from the fielder on some plays, and the auto step needs a clip with usable audio.
