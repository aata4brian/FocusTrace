#!/usr/bin/env python3
"""
TraceFokus - Preprocessing final 18 partisipan.

Pipeline:
- Final cohort dikunci: 18 peserta.
- P08 dan P17 dikeluarkan; P20 menggantikan P17.
- Hanya state BLOCK dengan binary_label 0/1.
- Calibration dan transition TIDAK masuk dataset ML.
- Setiap block diproses terpisah agar window tidak menyeberangi kondisi.
- Data di-resample ke 10 Hz.
- Window = 2 detik = 20 timestep.
- Stride = 1 detik = 10 timestep.
- Hanya gap pendek <= 0.5 detik yang boleh diinterpolasi.
- Sisa missing diisi secara konservatif menggunakan baseline calibration
  bila tersedia; fitur delta/velocity/motion yang tidak punya baseline diisi 0.
- TIDAK melakukan standardisasi global. Scaling harus dilakukan di dalam
  training fold untuk menghindari data leakage.
- Output mengikuti format dataset.npz yang kompatibel dengan pipeline model.

Contoh:
    python preprocessing/preprocess.py --input data --output data/processed_final
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


FINAL_PARTICIPANTS = [
    "P01","P02","P03","P04","P05","P06","P07","P09","P10",
    "P11","P12","P13","P14","P15","P16","P18","P19","P20"
]
EXCLUDED = {"P08", "P17"}

FEATURES = [
    "gaze_horizontal","gaze_vertical","gaze_deviation","gaze_away_duration",
    "gaze_valid","gaze_CENTER","gaze_LEFT","gaze_RIGHT","gaze_DOWN","gaze_UNKNOWN",
    "head_yaw","head_pitch","head_roll","head_yaw_smooth","head_pitch_smooth",
    "head_roll_smooth","head_angular_speed","head_valid","shoulder_motion",
    "shoulder_angle","torso_motion","wrist_motion","body_motion",
    "normalized_velocity","motion_variance","body_valid","delta_yaw",
    "delta_pitch","delta_roll","delta_gaze_h","delta_gaze_v","delta_body_motion"
]

WINDOW_SEC = 2.0
TARGET_HZ = 10.0
WINDOW_STEPS = int(WINDOW_SEC * TARGET_HZ)
STRIDE_SEC = 1.0
MAX_GAP_SEC = 0.5
MIN_VALID_FRACTION = 0.5

BASELINE_MAP = {
    "head_yaw": "head_yaw",
    "head_pitch": "head_pitch",
    "head_roll": "head_roll",
    "head_yaw_smooth": "head_yaw",
    "head_pitch_smooth": "head_pitch",
    "head_roll_smooth": "head_roll",
    "gaze_horizontal": "gaze_horizontal",
    "gaze_vertical": "gaze_vertical",
}

def canonical_pid(value: str) -> str:
    m = re.search(r"(\d+)", str(value))
    return f"P{int(m.group(1)):02d}" if m else str(value)

def locate(root: Path, kind: str, pid: str) -> Path | None:
    candidates = [
        root / kind / f"{pid}_{'features' if kind=='features' else 'events' if kind=='events' else 'responses' if kind=='responses' else 'frame_timestamps' if kind=='raw_video' else 'baseline'}.csv",
    ]
    if kind == "features":
        candidates = [root/"features"/f"{pid}_features.csv", root/"features"/f"P{int(pid[1:]):03d}_features.csv"]
    elif kind == "events":
        candidates = [root/"events"/f"{pid}_events.csv", root/"events"/f"P{int(pid[1:]):03d}_events.csv"]
    elif kind == "responses":
        candidates = [root/"responses"/f"{pid}_responses.csv", root/"responses"/f"P{int(pid[1:]):03d}_responses.csv"]
    elif kind == "profiles":
        candidates = [root/"profiles"/f"{pid}_baseline.json", root/"profiles"/f"P{int(pid[1:]):03d}_baseline.json"]
    return next((p for p in candidates if p.exists()), None)

def load_baseline(root: Path, pid: str) -> dict:
    p = locate(root, "profiles", pid)
    if not p:
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj.get("features", {})
    except Exception:
        return {}

def fill_remaining(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    out = df.copy()

    # Sensor features: baseline median from calibration, if valid.
    for col in FEATURES:
        if col not in out:
            out[col] = np.nan

        if col in BASELINE_MAP:
            b = baseline.get(BASELINE_MAP[col], {})
            med = b.get("median") if isinstance(b, dict) else None
            valid = b.get("valid", False) if isinstance(b, dict) else False
            if valid and med is not None and np.isfinite(med):
                out[col] = out[col].fillna(float(med))

    # Validity flags and one-hot category: missing means unavailable/unknown.
    for col in ["gaze_valid","head_valid","body_valid"]:
        if col in out:
            out[col] = out[col].fillna(0.0)

    for col in ["gaze_CENTER","gaze_LEFT","gaze_RIGHT","gaze_DOWN","gaze_UNKNOWN"]:
        if col in out:
            out[col] = out[col].fillna(0.0)
    if "gaze_UNKNOWN" in out:
        # If all gaze category one-hot values are zero after missingness, mark UNKNOWN.
        cats = out[["gaze_CENTER","gaze_LEFT","gaze_RIGHT","gaze_DOWN","gaze_UNKNOWN"]]
        missing_cat = cats.sum(axis=1) == 0
        out.loc[missing_cat, "gaze_UNKNOWN"] = 1.0

    # Derived temporal/motion quantities have a natural neutral value of zero.
    zero_fill = [
        "gaze_deviation","gaze_away_duration","head_angular_speed",
        "shoulder_motion","shoulder_angle","torso_motion","wrist_motion",
        "body_motion","normalized_velocity","motion_variance",
        "delta_yaw","delta_pitch","delta_roll","delta_gaze_h","delta_gaze_v",
        "delta_body_motion",
    ]
    for col in zero_fill:
        if col in out:
            out[col] = out[col].fillna(0.0)

    # Last-resort fallback: median within this BLOCK, then zero.
    # This is only a fallback for rare unsupported/derived values.
    for col in FEATURES:
        if out[col].isna().any():
            med = out[col].median()
            out[col] = out[col].fillna(0.0 if pd.isna(med) else float(med))

    return out

def resample_block(block: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X_grid: [T, F]
      observed_fraction: [T] — whether original data support each grid point
      grid_time: [T]
    """
    b = block.sort_values("timestamp_sec").copy()
    t = b["timestamp_sec"].to_numpy(dtype=float)
    if len(b) < 2:
        return np.empty((0, len(FEATURES))), np.empty(0), np.empty(0)

    start = float(t[0])
    end = float(t[-1])
    step = 1.0 / TARGET_HZ
    grid = np.arange(start, end + 1e-9, step)

    # Numeric interpolation only over short gaps.
    tmp = b[["timestamp_sec"] + FEATURES].copy()
    tmp = tmp.drop_duplicates("timestamp_sec").set_index("timestamp_sec")
    tmp = tmp[FEATURES].apply(pd.to_numeric, errors="coerce")
    full_index = pd.Index(grid, name="timestamp_sec")
    joined = tmp.reindex(full_index)

    # Linear interpolation, then invalidate points whose nearest original
    # observations are farther than MAX_GAP_SEC/2.
    interp = joined.interpolate(method="index", limit_area="inside")
    original_t = np.asarray(tmp.index, dtype=float)
    pos = np.searchsorted(original_t, grid, side="left")
    left = np.clip(pos - 1, 0, len(original_t)-1)
    right = np.clip(pos, 0, len(original_t)-1)
    nearest = np.minimum(np.abs(grid-original_t[left]), np.abs(grid-original_t[right]))
    observed = nearest <= (MAX_GAP_SEC / 2.0)

    arr = interp[FEATURES].to_numpy(dtype=float)
    arr[~observed, :] = np.nan
    return arr, observed.astype(float), grid

def make_windows(X_grid: np.ndarray, observed: np.ndarray, grid: np.ndarray,
                 label: int, pid: str, block_code: str):
    out_X, out_y, out_groups, out_blocks, out_starts, out_verified = [], [], [], [], [], []
    if len(X_grid) < WINDOW_STEPS:
        return out_X, out_y, out_groups, out_blocks, out_starts, out_verified

    stride_steps = int(STRIDE_SEC * TARGET_HZ)
    for s in range(0, len(X_grid) - WINDOW_STEPS + 1, stride_steps):
        e = s + WINDOW_STEPS
        frac = float(np.mean(observed[s:e]))
        if frac < MIN_VALID_FRACTION:
            continue

        w = X_grid[s:e].copy()
        out_X.append(w)
        out_y.append(int(label))
        out_groups.append(pid)
        out_blocks.append(block_code)
        out_starts.append(float(grid[s]))
        out_verified.append(1)

    return out_X, out_y, out_groups, out_blocks, out_starts, out_verified

def process_participant(root: Path, pid: str):
    fp = locate(root, "features", pid)
    if not fp:
        raise FileNotFoundError(f"{pid}: features tidak ditemukan: {fp}")

    df = pd.read_csv(fp)
    required = {"timestamp_sec","system_state","block_code","binary_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{pid}: kolom wajib hilang: {sorted(missing)}")

    # Only experimental, labeled BLOCK rows.
    df = df[
        (df["system_state"].astype(str).str.upper() == "BLOCK") &
        (df["binary_label"].isin([0, 1]))
    ].copy()

    if df.empty:
        return [], [], [], [], [], []

    # Ensure all model features exist.
    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan

    baseline = load_baseline(root, pid)

    all_X, all_y, all_g, all_b, all_s, all_v = [], [], [], [], [], []

    # Process each experimental block separately.
    for block_code, b in df.groupby("block_code", sort=False):
        label_values = b["binary_label"].dropna().unique()
        if len(label_values) != 1:
            continue
        label = int(label_values[0])

        # Do not allow a window to cross blocks.
        Xg, observed, grid = resample_block(b)
        if len(Xg) == 0:
            continue

        # Fill only after interpolation so real gaps are handled first.
        temp = pd.DataFrame(Xg, columns=FEATURES)
        temp = fill_remaining(temp, baseline)
        Xg = temp.to_numpy(dtype=np.float32)

        w = make_windows(Xg, observed, grid, label, pid, str(block_code))
        all_X.extend(w[0]); all_y.extend(w[1]); all_g.extend(w[2])
        all_b.extend(w[3]); all_s.extend(w[4]); all_v.extend(w[5])

    return all_X, all_y, all_g, all_b, all_s, all_v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder data, misalnya data")
    ap.add_argument("--output", required=True, help="Folder output, misalnya data/processed_final")
    args = ap.parse_args()

    root = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    X_all, y_all, groups, blocks, starts, verified = [], [], [], [], [], []
    per_participant = {}

    for pid in FINAL_PARTICIPANTS:
        result = process_participant(root, pid)
        Xp, yp, gp, bp, sp, vp = result
        X_all.extend(Xp); y_all.extend(yp); groups.extend(gp)
        blocks.extend(bp); starts.extend(sp); verified.extend(vp)
        per_participant[pid] = {
            "windows": len(Xp),
            "class_0": int(sum(v == 0 for v in yp)),
            "class_1": int(sum(v == 1 for v in yp)),
        }
        print(f"{pid}: {len(Xp):5d} windows | class0={sum(v==0 for v in yp):4d} | class1={sum(v==1 for v in yp):4d}")

    if not X_all:
        raise RuntimeError("Tidak ada window yang dihasilkan.")

    X = np.asarray(X_all, dtype=np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    groups_arr = np.asarray(groups, dtype=str)
    blocks_arr = np.asarray(blocks, dtype=str)
    starts_arr = np.asarray(starts, dtype=np.float32)
    verified_arr = np.asarray(verified, dtype=np.int8)

    np.savez_compressed(
        out / "dataset.npz",
        X=X,
        y=y,
        groups=groups_arr,
        blocks=blocks_arr,
        starts=starts_arr,
        verified=verified_arr,
        features=np.asarray(FEATURES, dtype=str),
    )

    metadata = {
        "pipeline": "TraceFokus final 18",
        "included_participants": FINAL_PARTICIPANTS,
        "excluded_participants": {
            "P08": "eksperimen tidak selesai",
            "P17": "digantikan oleh P20",
        },
        "replacement": "P17 -> P20",
        "windowing": {
            "seconds": WINDOW_SEC,
            "target_hz": TARGET_HZ,
            "timesteps": WINDOW_STEPS,
            "stride_seconds": STRIDE_SEC,
            "maximum_gap_sec": MAX_GAP_SEC,
            "minimum_valid_fraction": MIN_VALID_FRACTION,
        },
        "preprocessing": [
            "exclude CALIBRATION and TRANSITION",
            "keep only binary_label 0/1",
            "process each block separately",
            "resample to 10 Hz",
            "interpolate only short gaps",
            "baseline-aware missing-value fallback",
            "no global normalization before cross-validation",
        ],
        "features": FEATURES,
        "n_windows": int(len(y)),
        "class_distribution": {
            "0": int((y == 0).sum()),
            "1": int((y == 1).sum()),
        },
        "per_participant": per_participant,
    }
    (out / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    pd.DataFrame([
        {"participant_id": pid, **stats}
        for pid, stats in per_participant.items()
    ]).to_csv(out / "participant_window_summary.csv", index=False)

    print("\n" + "=" * 72)
    print("PREPROCESSING SELESAI")
    print("=" * 72)
    print(f"Participants : {len(FINAL_PARTICIPANTS)}")
    print(f"Windows      : {len(y)}")
    print(f"X shape      : {X.shape}")
    print(f"Class 0      : {(y == 0).sum()}")
    print(f"Class 1      : {(y == 1).sum()}")
    print(f"Saved        : {(out / 'dataset.npz').resolve()}")
    print("\nCatatan: scaling/standardisasi jangan dilakukan di sini.")
    print("Lakukan fit scaler HANYA pada training fold saat cross-validation.")

if __name__ == "__main__":
    main()
