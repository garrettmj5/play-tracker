#!/usr/bin/env python3
"""Play tracker: Statcast balls in play -> clips -> release/contact/fielder timeline.

Subcommands:
  sample    pull Statcast for a date range, draw a stratified sample of balls in play
  clips     match pitches to the MLB game feed, download and shrink the Savant clips
  auto      find bat contact from the clip audio, derive release from MLB plate time
  annotate  build annotate.html, a frame-stepping tool with auto marks preloaded
  compare   check auto marks against hand marks
  report    combine data, video marks, and estimates into events.csv and report.html
  clean     delete clips that have already been marked, to free disk space

Event times are reported in seconds after ball release (release = 0).
"""
import argparse
import html
import io
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "Mozilla/5.0 (play-tracker research script)"}
DATA = Path("data")
CLIPS = DATA / "clips"
FEEDS = DATA / "feeds"
PLAYS_CSV = DATA / "plays.csv"
AUTO_JSON = DATA / "auto_marks.json"
USABLE = ("ok", "cleaned")  # clip_status values that count as a real play

# Minimum size of the bat-crack onset (log10 jump in 1.5-7 kHz energy, 1.0 = 10x).
# Clips with a weaker onset get no auto marks. Real clips scored 2.3 to 4.6 in testing.
AUTO_MIN_STRENGTH = 0.8

SAVANT_URL = (
    "https://baseballsavant.mlb.com/statcast_search/csv?all=true"
    "&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones="
    "&hfGT={gt}&hfSea={season}%7C&hfSit=&player_type=pitcher&hfOuts=&opponent="
    "&pitcher_throws=&batter_stands=&hfSA=&game_date_gt={day}&game_date_lt={day}"
    "&team=&position=&hfRO=&home_road=&hfFlag=&metric_1=&hfInn=&min_pitches=0"
    "&min_results=0&group_by=name&sort_col=pitches&player_event_sort=h_launch_speed"
    "&sort_order=desc&min_abs=0&type=details"
)

KEEP_COLS = [
    "game_date", "game_pk", "at_bat_number", "pitch_number", "inning",
    "home_team", "away_team", "player_name", "batter", "pitcher", "events",
    "description", "bb_type", "launch_speed", "launch_angle", "hit_distance_sc",
    "hc_x", "hc_y", "release_speed", "release_extension", "pitch_type", "hit_location",
]

# Share of the sample drawn from each batted-ball type.
BB_SHARE = {"ground_ball": 0.40, "fly_ball": 0.25, "line_drive": 0.25, "popup": 0.10}


# ---------------------------------------------------------------- helpers

def load_json(p):
    p = Path(p)
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def get_ffmpeg():
    """System ffmpeg if present, else the copy bundled with the imageio-ffmpeg package."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def clip_fps(path):
    """Frame rate of a clip, read from ffmpeg's stream report (30 if it cannot be read)."""
    ff = get_ffmpeg()
    if not ff:
        return 30.0
    r = subprocess.run([ff, "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"(\d+(?:\.\d+)?) fps", r.stderr)
    return float(m.group(1)) if m else 30.0


def audio_contact(m):
    """Detected bat-crack time on the clip clock, or None. Earlier runs stored it as contact_raw."""
    if not m:
        return None
    return m.get("contact", m.get("contact_raw"))


def auto_for_play(m, plate):
    """Auto marks in video seconds for one play, or {} if missing or low confidence."""
    t = audio_contact(m)
    if t is None or m.get("strength", 0) < AUTO_MIN_STRENGTH:
        return {}
    out = {"contact": float(t)}
    if plate is not None:
        out["release"] = out["contact"] - float(plate)
    return out


# ---------------------------------------------------------------- sample

def savant_day(day, gametypes):
    gt = requests.utils.quote(gametypes, safe="")
    url = SAVANT_URL.format(day=day, gt=gt, season=day[:4])
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=180)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text.lstrip("﻿")), low_memory=False)
        except Exception as exc:
            print(f"  retry {attempt + 1} for {day}: {exc}")
            time.sleep(3 * (attempt + 1))
    return pd.DataFrame()


def cmd_sample(a):
    DATA.mkdir(exist_ok=True)
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    team = a.team.upper() if a.team else None
    frames = []
    d = d0
    while d <= d1:
        day = d.isoformat()
        df = savant_day(day, a.gametypes)
        if team and len(df):
            df = df[(df["home_team"] == team) | (df["away_team"] == team)]
        print(f"{day}: {len(df)} pitches")
        if len(df):
            frames.append(df)
        d += timedelta(days=1)
        time.sleep(1.0)
    if not frames:
        sys.exit("No Statcast rows came back. Check dates, game types, team code, and network access.")
    allp = pd.concat(frames, ignore_index=True)
    bip = allp[(allp["type"] == "X") & allp["bb_type"].notna()].copy()
    if not a.include_hr:
        bip = bip[bip["events"] != "home_run"]
    print(f"{len(bip)} balls in play in range (home runs excluded: {not a.include_hr})")

    rng = np.random.default_rng(a.seed)
    picks = []
    for bb, share in BB_SHARE.items():
        pool = bip[bip["bb_type"] == bb]
        k = min(len(pool), int(round(a.n * share)))
        picks.append(pool.sample(k, random_state=int(rng.integers(1e9))))
    out = pd.concat(picks)
    short = a.n - len(out)
    if short > 0:
        rest = bip.drop(out.index)
        out = pd.concat([out, rest.sample(min(short, len(rest)), random_state=a.seed)])
    cols = [c for c in KEEP_COLS if c in out.columns]
    out = out[cols].sort_values(["game_date", "game_pk", "at_bat_number"]).reset_index(drop=True)
    out.to_csv(PLAYS_CSV, index=False)
    print(f"Wrote {len(out)} plays to {PLAYS_CSV}")
    print(out["bb_type"].value_counts().to_string())


# ----------------------------------------------------------------- clips

def slim_feed(feed):
    """Keep only what match_pitch and cmd_clips need. Full feeds are several MB each."""
    plays = []
    for p in feed["liveData"]["plays"]["allPlays"]:
        evs = []
        for e in p.get("playEvents", []):
            if not e.get("isPitch"):
                continue
            evs.append({
                "isPitch": True,
                "pitchNumber": e.get("pitchNumber"),
                "playId": e.get("playId"),
                "pitchData": {"plateTime": (e.get("pitchData") or {}).get("plateTime")},
            })
        plays.append({"atBatIndex": p.get("atBatIndex"), "playEvents": evs})
    return {"liveData": {"plays": {"allPlays": plays}}}


def load_feed(pk):
    FEEDS.mkdir(parents=True, exist_ok=True)
    p = FEEDS / f"{pk}.json"
    if p.exists():
        return json.loads(p.read_text())
    r = requests.get(
        f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live", headers=UA, timeout=90
    )
    r.raise_for_status()
    slim = slim_feed(r.json())
    p.write_text(json.dumps(slim))
    return slim


def match_pitch(feed, at_bat_number, pitch_number):
    """Find the feed pitch event for a Statcast row (atBatIndex is 0-based)."""
    plays = feed["liveData"]["plays"]["allPlays"]
    pa = next((p for p in plays if p.get("atBatIndex") == int(at_bat_number) - 1), None)
    if pa is None:
        return None
    pitches = [e for e in pa.get("playEvents", []) if e.get("isPitch")]
    for e in pitches:
        if e.get("pitchNumber") == int(pitch_number):
            return e
    # A ball in play is always the last pitch of the plate appearance.
    return pitches[-1] if pitches else None


def find_mp4(play_id):
    r = requests.get(
        f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}",
        headers=UA, timeout=60,
    )
    r.raise_for_status()
    m = re.search(r"https?://[^\"'\s>]+\.mp4", r.text)
    return html.unescape(m.group(0)) if m else None


def download(url, dest):
    with requests.get(url, headers=UA, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        tmp.rename(dest)


def compress(path, height):
    """Shrink a clip in place: scaled to `height` pixels tall, same frame rate, audio kept
    (the auto step needs the audio)."""
    ff = get_ffmpeg()
    if not ff or height <= 0:
        return
    tmp = path.with_suffix(".small.mp4")
    r = subprocess.run(
        [ff, "-y", "-loglevel", "error", "-i", str(path),
         "-map", "0:v:0", "-map", "0:a:0?",
         "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-crf", "30",
         "-preset", "veryfast", "-c:a", "aac", "-b:a", "64k", str(tmp)]
    )
    if r.returncode == 0 and tmp.exists() and tmp.stat().st_size < path.stat().st_size:
        tmp.replace(path)
    else:
        tmp.unlink(missing_ok=True)


def cmd_clips(a):
    CLIPS.mkdir(parents=True, exist_ok=True)
    if a.height > 0 and not get_ffmpeg():
        print("ffmpeg not found, so clips will be kept at full size. "
              "Run: pip install imageio-ffmpeg  (also needed for the auto step)")
    df = pd.read_csv(PLAYS_CSV)
    for c in ["play_id", "clip_path", "clip_status"]:
        df[c] = df[c].fillna("").astype(str) if c in df else ""
    if "plate_time_s" not in df:
        df["plate_time_s"] = np.nan

    fetched = 0
    for i, row in df.iterrows():
        if row["clip_status"] in USABLE:
            continue
        if a.limit and fetched >= a.limit:
            print(f"Reached --limit {a.limit}. Mark these, run clean, then run clips again.")
            break
        try:
            feed = load_feed(int(row["game_pk"]))
            ev = match_pitch(feed, row["at_bat_number"], row["pitch_number"])
            if ev is None or not ev.get("playId"):
                df.at[i, "clip_status"] = "no feed match"
                continue
            pid = ev["playId"]
            df.at[i, "play_id"] = pid
            pt = (ev.get("pitchData") or {}).get("plateTime")
            df.at[i, "plate_time_s"] = pt if pt is not None else np.nan

            dest = CLIPS / f"{pid}.mp4"
            if not dest.exists():
                url = find_mp4(pid)
                if not url:
                    df.at[i, "clip_status"] = "no mp4 on page"
                    continue
                download(url, dest)
                compress(dest, a.height)
                time.sleep(a.delay)
            df.at[i, "clip_path"] = str(dest)
            df.at[i, "clip_status"] = "ok"
            fetched += 1
            print(f"[{i + 1}/{len(df)}] ok {pid} ({dest.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:
            df.at[i, "clip_status"] = f"error: {exc}"[:200]
            print(f"[{i + 1}/{len(df)}] {df.at[i, 'clip_status']}")
        if (i + 1) % 10 == 0:
            df.to_csv(PLAYS_CSV, index=False)
    df.to_csv(PLAYS_CSV, index=False)
    print(df["clip_status"].value_counts().to_string())


# ------------------------------------------------------------------ auto

def load_audio(path, sr=16000):
    """Decode a clip's audio to mono float samples. Returns (None, sr) if there is none."""
    ff = get_ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found. Run: pip install imageio-ffmpeg")
    r = subprocess.run(
        [ff, "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
         "-f", "s16le", "pipe:1"],
        capture_output=True,
    )
    if r.returncode != 0 or len(r.stdout) < 2 * sr // 2:
        return None, sr
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0, sr


def detect_contact(x, sr, lo=1500.0, hi=7000.0, win_ms=8.0, hop_ms=2.0):
    """Find the bat crack: the sharpest rise in 1.5-7 kHz energy against the local background.

    Returns {'t', 'strength'} or None. `t` is seconds on the clip clock; strength is the
    log10 energy jump at the onset (1.0 = 10x).
    """
    n = int(sr * win_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    if len(x) < n * 8:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    idx = np.arange(0, len(x) - n + 1, hop)[: len(frames)]
    spec = np.fft.rfft(frames * np.hanning(n), axis=1)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    band = (freqs >= lo) & (freqs <= hi)
    e = np.log10(np.sum(np.abs(spec[:, band]) ** 2, axis=1) + 1e-10)
    t = (idx + n / 2) / sr

    bg = int(150 / hop_ms)   # background window in frames
    gap = int(10 / hop_ms)   # keep the onset itself out of the background
    onset = np.zeros_like(e)
    for i in range(bg + gap, len(e)):
        onset[i] = e[i] - np.median(e[i - gap - bg:i - gap])

    i = int(np.argmax(onset))
    strength = float(onset[i])
    if strength <= 0:
        return None
    for _ in range(10):  # walk back to where the rise begins
        if i > 0 and onset[i - 1] >= 0.5 * strength:
            i -= 1
        else:
            break
    return {"t": float(t[i]), "strength": strength}


def cmd_auto(a):
    df = pd.read_csv(PLAYS_CSV)
    marks = load_json(AUTO_JSON)
    done = 0
    for _, r in df.iterrows():
        if r.get("clip_status") != "ok":
            continue
        pid = str(r["play_id"])
        p = Path(str(r["clip_path"]))
        if not p.exists() or (pid in marks and not a.force):
            continue
        x, sr = load_audio(p)
        d = detect_contact(x, sr) if x is not None else None
        if d is None:
            marks[pid] = {"error": "no usable audio or no onset found"}
            print(f"{pid}: no contact found")
            continue
        marks[pid] = {"contact": round(d["t"], 4), "strength": round(d["strength"], 3)}
        ok = d["strength"] >= AUTO_MIN_STRENGTH
        print(f"{pid}: contact at {d['t']:.3f}s, strength {d['strength']:.2f} "
              f"({'ok' if ok else 'LOW CONFIDENCE'})")
        done += 1
    AUTO_JSON.write_text(json.dumps(marks, indent=1))
    good = sum(1 for m in marks.values() if m.get("strength", 0) >= AUTO_MIN_STRENGTH)
    print(f"Processed {done} clips. {good} of {len(marks)} total auto marks pass the confidence check.")


def cmd_compare(a):
    """Compare auto marks with hand marks on plays that have both. Errors are auto minus hand."""
    df = pd.read_csv(PLAYS_CSV)
    hand = load_json(a.annotations)
    marks = load_json(AUTO_JSON)
    plate = {str(r["play_id"]): r.get("plate_time_s") for _, r in df.iterrows()}
    rows = []
    for pid, h in hand.items():
        m = marks.get(pid)
        t = audio_contact(m)
        if t is None or "contact" not in h or m.get("strength", 0) < AUTO_MIN_STRENGTH:
            continue
        row = {"play_id": pid, "hand_contact": h["contact"], "auto_contact": t,
               "contact_error_s": t - h["contact"], "strength": m.get("strength")}
        pl = plate.get(pid)
        if "release" in h and pl is not None and pd.notna(pl):
            row["release_error_s"] = (t - float(pl)) - h["release"]
        rows.append(row)
    if not rows:
        sys.exit("No plays have both hand marks and auto marks yet. Run auto on clips you have hand-marked.")
    res = pd.DataFrame(rows)
    res.to_csv(DATA / "auto_vs_hand.csv", index=False)

    for col, label in [("contact_error_s", "Contact"), ("release_error_s", "Release")]:
        e = res[col].dropna()
        if len(e):
            print(f"{label}: n={len(e)}, median {e.median() * 1000:+.0f} ms, mean abs {e.abs().mean() * 1000:.0f} ms, "
                  f"within 17 ms: {(e.abs() <= 0.0167).sum()}, within 33 ms: {(e.abs() <= 0.0334).sum()}, "
                  f"worst {e.abs().max() * 1000:.0f} ms")
    print("Details in data/auto_vs_hand.csv.")


# -------------------------------------------------------------- annotate

ANNOTATE_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Play annotator</title>
<style>
body{font-family:system-ui,sans-serif;color:#000;margin:0;display:flex;height:100vh}
#list{width:300px;overflow:auto;border-right:1px solid #999}
#list div{padding:6px 8px;border-bottom:1px solid #ddd;cursor:pointer;font-size:13px}
#list div.sel{background:#e8e8e8;font-weight:600}
#main{flex:1;padding:12px;overflow:auto}
video{width:100%;max-width:960px;background:#000}
button{margin:2px;padding:6px 10px}
.mono{font-family:ui-monospace,monospace}
</style></head><body>
<div id="list"></div>
<div id="main">
<h3 id="title"></h3>
<video id="v" controls></video>
<div class="mono" id="clock"></div>
<div>
<button onclick="mark('release')">R: release</button>
<button onclick="mark('contact')">C: contact</button>
<button onclick="mark('fielder')">F: fielder arrives</button>
<button onclick="seekAuto('auto_release')">1: go to auto release</button>
<button onclick="seekAuto('auto_contact')">2: go to auto contact</button>
<button onclick="clearMarks()">Clear this play</button>
<button onclick="exportJson()">Export annotations.json</button>
</div>
<p class="mono" id="marks"></p>
<p class="mono" id="auto"></p>
<p>Each clip opens at the auto contact time when there is one. If auto marks exist you only need
F (the frame where the fielder gets the ball); mark R and C yourself only to override a bad auto
mark. Keys: left/right = 1 frame, shift+left/right = 5 frames, space = play/pause, R / C / F = mark,
1 / 2 = jump to auto release / contact, N / P = next / previous play. Marks save in this browser as
you go. Export when done and put annotations.json in the data folder.</p>
</div>
<script>
const PLAYS = __PLAYS__;
const store = JSON.parse(localStorage.getItem("ann") || "{}");
const v = document.getElementById("v");
let idx = 0;
const fps = () => PLAYS[idx].fps || 30;
function save(){ localStorage.setItem("ann", JSON.stringify(store)); }
function load(i){
  idx = i; v.src = PLAYS[i].clip;
  v.onloadedmetadata = () => { const a = PLAYS[idx].auto_contact; if (a !== null && a !== undefined) v.currentTime = Math.max(0, a); };
  v.load(); render();
}
function seekAuto(k){ const a = PLAYS[idx][k]; if (a !== null && a !== undefined) { v.pause(); v.currentTime = Math.max(0, a); } }
function step(n){ v.pause(); v.currentTime = Math.max(0, Math.min(v.duration || 1e9, v.currentTime + n / fps())); }
function mark(k){ const id = PLAYS[idx].play_id; store[id] = store[id] || {}; store[id][k] = v.currentTime; save(); render(); }
function clearMarks(){ delete store[PLAYS[idx].play_id]; save(); render(); }
function exportJson(){
  const b = new Blob([JSON.stringify(store, null, 2)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = "annotations.json"; a.click();
}
function render(){
  const p = PLAYS[idx];
  document.getElementById("title").textContent = `${p.player_name} | ${p.events} | ${p.bb_type} | ${p.game_date} (${idx + 1}/${PLAYS.length})`;
  const m = store[p.play_id] || {};
  const f = x => x === undefined ? "not set" : `${x.toFixed(3)}s (frame ${Math.round(x * fps())})`;
  document.getElementById("marks").textContent = `your marks  release: ${f(m.release)}   contact: ${f(m.contact)}   fielder: ${f(m.fielder)}`;
  const has = p.auto_contact !== null && p.auto_contact !== undefined;
  const rel = (p.auto_release === null || p.auto_release === undefined) ? "n/a" : p.auto_release.toFixed(3) + "s";
  document.getElementById("auto").textContent = has ? `auto marks  release: ${rel}   contact: ${p.auto_contact.toFixed(3)}s` : "auto marks  none (low confidence or not run)";
  const L = document.getElementById("list"); L.innerHTML = "";
  PLAYS.forEach((q, i) => {
    const s = store[q.play_id] || {};
    const flags = ["release", "contact", "fielder"].map(k => s[k] !== undefined ? k[0].toUpperCase() : "-").join("");
    const d = document.createElement("div");
    d.className = i === idx ? "sel" : "";
    d.textContent = `${flags}  ${q.player_name} ${q.events}`;
    d.onclick = () => load(i);
    L.appendChild(d);
  });
}
setInterval(() => {
  document.getElementById("clock").textContent = `t = ${v.currentTime.toFixed(3)}s   frame ${Math.round(v.currentTime * fps())}   fps ${fps().toFixed(2)}`;
}, 50);
document.addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  if (e.key === "ArrowRight") { step(e.shiftKey ? 5 : 1); e.preventDefault(); }
  else if (e.key === "ArrowLeft") { step(e.shiftKey ? -5 : -1); e.preventDefault(); }
  else if (e.key === " ") { v.paused ? v.play() : v.pause(); e.preventDefault(); }
  else if (k === "r") mark("release");
  else if (k === "c") mark("contact");
  else if (k === "f") mark("fielder");
  else if (k === "1") seekAuto("auto_release");
  else if (k === "2") seekAuto("auto_contact");
  else if (k === "n" && idx < PLAYS.length - 1) load(idx + 1);
  else if (k === "p" && idx > 0) load(idx - 1);
});
if (PLAYS.length) load(0); else document.getElementById("title").textContent = "No clips on disk. Run: python pipeline.py clips";
</script></body></html>
"""


def cmd_annotate(a):
    df = pd.read_csv(PLAYS_CSV)
    marks = load_json(AUTO_JSON)
    plays = []
    for _, r in df.iterrows():
        if r.get("clip_status") != "ok" or not Path(str(r["clip_path"])).exists():
            continue
        plate = r.get("plate_time_s")
        am = auto_for_play(marks.get(str(r["play_id"])), float(plate) if pd.notna(plate) else None)
        plays.append({
            "play_id": r["play_id"],
            "clip": "data/clips/" + Path(r["clip_path"]).name,
            "fps": clip_fps(r["clip_path"]),
            "player_name": str(r.get("player_name", "")),
            "events": str(r.get("events", "")),
            "bb_type": str(r.get("bb_type", "")),
            "game_date": str(r.get("game_date", "")),
            "auto_contact": am.get("contact"),
            "auto_release": am.get("release"),
        })
    Path("annotate.html").write_text(ANNOTATE_HTML.replace("__PLAYS__", json.dumps(plays)))
    n_auto = sum(1 for p in plays if p["auto_contact"] is not None)
    print(f"Wrote annotate.html with {len(plays)} plays ({n_auto} with auto marks). Open it in Chrome from this folder.")


# ----------------------------------------------------------------- clean

def cmd_clean(a):
    """Delete clips for plays that have marks in annotations.json (or all clips with --all)."""
    ap = Path(a.annotations)
    if not ap.exists() and not a.all:
        sys.exit(f"{ap} not found. Export your marks from annotate.html first, or pass --all.")
    ann = load_json(ap)
    df = pd.read_csv(PLAYS_CSV)
    freed = 0
    n = 0
    for i, r in df.iterrows():
        if r.get("clip_status") != "ok":
            continue
        if not a.all and str(r["play_id"]) not in ann:
            continue
        p = Path(str(r["clip_path"]))
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
            n += 1
        df.at[i, "clip_status"] = "cleaned"
    df.to_csv(PLAYS_CSV, index=False)
    print(f"Deleted {n} clips, freed {freed / 1e6:.1f} MB. Plays stay in plays.csv and the report.")


# ---------------------------------------------------------------- report

def hang_estimate(row):
    """Rough ball flight time (s) from launch angle and landing distance. Air balls only."""
    la, d = row.get("launch_angle"), row.get("hit_distance_sc")
    if pd.isna(la) or pd.isna(d) or la < 10 or d <= 0:
        return np.nan
    g = 32.174
    th = np.radians(la)
    v = np.sqrt(d * g / np.sin(2 * th))
    return float(2 * v * np.sin(th) / g)


def build_events(df, ann, auto=None):
    """Hand marks win over auto marks, key by key. Release is the time origin."""
    auto = auto or {}
    rows = []
    for _, r in df.iterrows():
        pid = str(r.get("play_id"))
        h = ann.get(pid, {})
        plate = r.get("plate_time_s")
        plate_ok = pd.notna(plate)
        am = auto_for_play(auto.get(pid), float(plate) if plate_ok else None)
        hang = hang_estimate(r)

        rel = h["release"] if "release" in h else am.get("release")
        rel_src = "hand" if "release" in h else ("auto" if "release" in am else "")
        con = h["contact"] if "contact" in h else am.get("contact")
        con_src = "hand" if "contact" in h else ("auto" if "contact" in am else "")
        fld = h.get("fielder")

        if rel is not None and con is not None:
            contact_t = con - rel
            if rel_src == "hand" and con_src == "hand":
                contact_src = "video"
            elif rel_src == "auto" and con_src == "auto":
                contact_src = "data"  # equals MLB plate time by construction
            else:
                contact_src = "auto"
        elif plate_ok:
            contact_t, contact_src = float(plate), "data"
        else:
            contact_t, contact_src = np.nan, ""

        if rel is not None and fld is not None:
            fielder_t, fielder_src = fld - rel, "video"
        elif pd.notna(contact_t) and pd.notna(hang):
            fielder_t, fielder_src = contact_t + hang, "est"
        else:
            fielder_t, fielder_src = np.nan, ""

        check = np.nan
        if contact_src == "video" and plate_ok:
            check = contact_t - float(plate)

        rows.append({
            "play_id": r.get("play_id"), "game_date": r.get("game_date"),
            "player_name": r.get("player_name"), "events": r.get("events"),
            "bb_type": r.get("bb_type"), "launch_speed": r.get("launch_speed"),
            "launch_angle": r.get("launch_angle"),
            "release_s": 0.0, "release_src": rel_src,
            "contact_s": contact_t, "contact_src": contact_src,
            "fielder_s": fielder_t, "fielder_src": fielder_src,
            "ball_in_play_s": fielder_t - contact_t if pd.notna(fielder_t) and pd.notna(contact_t) else np.nan,
            "feed_plate_time_s": plate,
            "video_vs_feed_contact_diff_s": check,
            "est_hang_time_s": hang,
        })
    return pd.DataFrame(rows)


def timeline_svg(row, tmax):
    w, h, pad = 760, 70, 30

    def x(t):
        return pad + (w - 2 * pad) * min(t, tmax) / tmax

    def fill(src):
        return "#000" if src in ("video", "auto") else "#fff"

    parts = [f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" font-size="11">']
    parts.append(f'<line x1="{pad}" y1="40" x2="{w - pad}" y2="40" stroke="#000"/>')
    for s in range(int(tmax) + 1):
        parts.append(f'<line x1="{x(s)}" y1="36" x2="{x(s)}" y2="44" stroke="#000"/>')
        parts.append(f'<text x="{x(s)}" y="60" text-anchor="middle" fill="#000">{s}s</text>')
    if pd.notna(row["contact_s"]):
        parts.append(f'<line x1="{x(0)}" y1="25" x2="{x(row["contact_s"])}" y2="25" stroke="#000" stroke-width="3"/>')
    if pd.notna(row["fielder_s"]) and pd.notna(row["contact_s"]):
        parts.append(f'<line x1="{x(row["contact_s"])}" y1="25" x2="{x(row["fielder_s"])}" y2="25" stroke="#777" stroke-width="3"/>')

    parts.append(f'<circle cx="{x(0)}" cy="25" r="6" fill="#000" stroke="#000"/>')
    parts.append(f'<text x="{x(0)}" y="14" text-anchor="middle" fill="#000">release</text>')
    if pd.notna(row["contact_s"]):
        cx = x(row["contact_s"])
        parts.append(f'<rect x="{cx - 6}" y="19" width="12" height="12" fill="{fill(row["contact_src"])}" stroke="#000"/>')
        parts.append(f'<text x="{cx}" y="14" text-anchor="middle" fill="#000">contact {row["contact_s"]:.2f}s</text>')
    if pd.notna(row["fielder_s"]):
        fx = x(row["fielder_s"])
        pts = f"{fx},18 {fx - 7},32 {fx + 7},32"
        parts.append(f'<polygon points="{pts}" fill="{fill(row["fielder_src"])}" stroke="#000"/>')
        parts.append(f'<text x="{fx}" y="14" text-anchor="middle" fill="#000">fielder {row["fielder_s"]:.2f}s</text>')
    parts.append("</svg>")
    return "".join(parts)


def cmd_report(a):
    df = pd.read_csv(PLAYS_CSV)
    df = df[df["clip_status"].isin(USABLE)].copy() if "clip_status" in df else df
    ev = build_events(df, load_json(a.annotations), load_json(AUTO_JSON))
    ev.to_csv(DATA / "events.csv", index=False)

    tmax = max(2.0, float(np.ceil(np.nanmax(ev[["contact_s", "fielder_s"]].to_numpy(dtype=float)))) if len(ev) else 2.0)
    tmax = min(tmax, 12.0)
    blocks = []
    for _, r in ev.iterrows():
        title = f'{html.escape(str(r["player_name"]))} | {html.escape(str(r["events"]))} | {html.escape(str(r["bb_type"]))} | {r["game_date"]}'
        blocks.append(f"<div class='p'><b>{title}</b><br>{timeline_svg(r, tmax)}</div>")
    n_video = int((ev["contact_src"] == "video").sum()) if len(ev) else 0
    n_auto = int((ev["release_src"] == "auto").sum()) if len(ev) else 0
    n_fld = int((ev["fielder_src"] == "video").sum()) if len(ev) else 0
    page = (
        "<!doctype html><meta charset='utf-8'><title>Play timelines</title>"
        "<style>body{font-family:sans-serif;color:#000;max-width:820px;margin:20px auto}"
        ".p{border-bottom:1px solid #ccc;padding:8px 0}</style>"
        "<h2>Play timelines</h2>"
        f"<p>{len(ev)} plays. Release and contact both hand-marked: {n_video}. Release found automatically "
        f"from the bat-crack sound and MLB plate time: {n_auto}. Fielder arrival marked on video: {n_fld}. "
        "Filled marker = measured from video or audio. Hollow marker = taken from MLB timing data or "
        "estimated from ball flight physics. Time is seconds after release.</p>" + "".join(blocks)
    )
    Path("report.html").write_text(page)
    print(f"Wrote {DATA / 'events.csv'} and report.html ({len(ev)} plays)")


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--start", required=True, help="YYYY-MM-DD")
    s.add_argument("--end", required=True, help="YYYY-MM-DD")
    s.add_argument("--n", type=int, default=25)
    s.add_argument("--team", help="only games involving this team, e.g. DET")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--gametypes", default="R|", help="Savant game types, e.g. 'R|' or 'R|F|D|L|W|'")
    s.add_argument("--include-hr", action="store_true")
    s.set_defaults(fn=cmd_sample)

    c = sub.add_parser("clips")
    c.add_argument("--delay", type=float, default=1.0, help="seconds between downloads")
    c.add_argument("--height", type=int, default=360, help="shrink clips to this pixel height (0 = keep original)")
    c.add_argument("--limit", type=int, default=0, help="download at most this many clips this run (0 = all)")
    c.set_defaults(fn=cmd_clips)

    u = sub.add_parser("auto")
    u.add_argument("--force", action="store_true", help="redo clips that already have auto marks")
    u.set_defaults(fn=cmd_auto)

    n = sub.add_parser("annotate")
    n.set_defaults(fn=cmd_annotate)

    m = sub.add_parser("compare")
    m.add_argument("--annotations", default="data/annotations.json")
    m.set_defaults(fn=cmd_compare)

    k = sub.add_parser("clean")
    k.add_argument("--annotations", default="data/annotations.json")
    k.add_argument("--all", action="store_true", help="delete every clip, marked or not")
    k.set_defaults(fn=cmd_clean)

    r = sub.add_parser("report")
    r.add_argument("--annotations", default="data/annotations.json")
    r.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
