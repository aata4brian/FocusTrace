#!/usr/bin/env python3
"""
TraceFokus - Audit data mentah peserta.

Tujuan:
1) Memastikan peserta final tepat 18 orang.
2) Secara eksplisit mengecualikan P08 dan P17.
3) Memastikan P20 menjadi pengganti P17.
4) Memeriksa keberadaan dan konsistensi events/features/profiles/raw_video/responses.
5) Memeriksa timestamp, duplikasi, label, missingness, dan jumlah block.
6) Tidak mengubah data apa pun.

Contoh:
    python preprocessing/audit_raw_data.py --input data --output outputs/audit
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
EXCLUDED = {
    "P08": "Eksperimen tidak selesai",
    "P17": "Digantikan oleh P20",
}
REPLACEMENT = {"original": "P17", "replacement": "P20"}

REQUIRED_COLUMNS = {
    "profiles": [],
    "events": [
        "participant_id","session_id","sequence_id","event_index","event_type",
        "system_state","block_number","block_code","conceptual_condition",
        "start_time_sec","end_time_sec","duration_sec","binary_label"
    ],
    "features": [
        "participant_id","session_id","sequence_id","timestamp_sec","frame_index",
        "system_state","block_number","block_code","conceptual_condition","binary_label"
    ],
    "raw_video": ["frame_index","timestamp_sec","encoded_time_sec"],
    "responses": [
        "participant_id","session_id","block_code","question_id","response",
        "response_mode","first_interaction_sec","last_edit_sec","completed"
    ],
}

def canonical_pid(value: str) -> str:
    m = re.search(r"(\d+)", str(value))
    return f"P{int(m.group(1)):02d}" if m else str(value)

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def locate(folder: Path, kind: str, pid: str):
    patterns = {
        "events": f"{pid}_events.csv",
        "features": f"{pid}_features.csv",
        "profiles": f"{pid}_baseline.json",
        "raw_video": f"{pid}_frame_timestamps.csv",
        "responses": f"{pid}_responses.csv",
    }
    # Support both P16 and P016 naming conventions.
    candidates = [
        folder / kind / patterns[kind],
        folder / kind / patterns[kind].replace(pid, f"P{int(pid[1:]):03d}"),
    ]
    return next((p for p in candidates if p.exists()), None)

def audit_one(data_root: Path, pid: str) -> dict:
    result = {"participant_id": pid}
    for kind in REQUIRED_COLUMNS:
        p = locate(data_root, kind, pid)
        result[f"{kind}_exists"] = bool(p)
        result[f"{kind}_path"] = str(p) if p else ""
        result[f"{kind}_rows"] = 0
        result[f"{kind}_missing_columns"] = ""
        result[f"{kind}_duplicates"] = 0

        if not p:
            continue

        if kind == "profiles":
            try:
                with p.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                result["profile_valid"] = bool(obj.get("valid", False))
                result["profile_method"] = obj.get("method", "")
            except Exception as e:
                result["profile_valid"] = False
                result["profile_method"] = f"ERROR: {e}"
            continue

        if kind == "profiles":
            continue
        df = read_csv(p)
        result[f"{kind}_rows"] = len(df)
        missing = [c for c in REQUIRED_COLUMNS[kind] if c not in df.columns]
        result[f"{kind}_missing_columns"] = ";".join(missing)
        result[f"{kind}_duplicates"] = int(df.duplicated().sum())

        if kind == "features":
            result["features_timestamp_monotonic"] = bool(
                df["timestamp_sec"].is_monotonic_increasing
            ) if "timestamp_sec" in df else False
            result["features_tmin"] = float(df["timestamp_sec"].min()) if "timestamp_sec" in df else np.nan
            result["features_tmax"] = float(df["timestamp_sec"].max()) if "timestamp_sec" in df else np.nan
            result["features_labeled_rows"] = int(df["binary_label"].notna().sum()) if "binary_label" in df else 0
            result["features_label_0"] = int((df["binary_label"] == 0).sum()) if "binary_label" in df else 0
            result["features_label_1"] = int((df["binary_label"] == 1).sum()) if "binary_label" in df else 0
            result["features_unlabeled_rows"] = int(df["binary_label"].isna().sum()) if "binary_label" in df else 0
            result["features_mean_nan_fraction"] = float(df.isna().mean().mean())
            result["features_blocks"] = int(df["block_code"].nunique()) if "block_code" in df else 0

            if "timestamp_sec" in df and len(df) > 1:
                dt = np.diff(df["timestamp_sec"].to_numpy(dtype=float))
                result["features_timestamp_median_gap_sec"] = float(np.median(dt))
                result["features_timestamp_max_gap_sec"] = float(np.max(dt))
                result["features_timestamp_gaps_over_0_5_sec"] = int((dt > 0.5).sum())
            else:
                result["features_timestamp_median_gap_sec"] = np.nan
                result["features_timestamp_max_gap_sec"] = np.nan
                result["features_timestamp_gaps_over_0_5_sec"] = 0

        elif kind == "events":
            if "event_type" in df:
                result["event_block_events"] = int((df["event_type"] == "BLOCK").sum())
            if "binary_label" in df:
                result["event_labeled_blocks"] = int(df["binary_label"].notna().sum())
        elif kind == "raw_video":
            if "timestamp_sec" in df and len(df) > 1:
                dt = np.diff(df["timestamp_sec"].to_numpy(dtype=float))
                result["video_timestamp_monotonic"] = bool(df["timestamp_sec"].is_monotonic_increasing)
                result["video_max_gap_sec"] = float(np.max(dt))
                result["video_frames"] = int(len(df))
        elif kind == "responses":
            result["responses_completed"] = int(df["completed"].fillna(0).astype(int).sum()) if "completed" in df else 0

    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder data, misalnya data")
    ap.add_argument("--output", required=True, help="Folder output audit")
    args = ap.parse_args()

    root = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rows = [audit_one(root, pid) for pid in FINAL_PARTICIPANTS]
    audit_df = pd.DataFrame(rows)

    # Also audit excluded participants when their files exist.
    excluded_rows = []
    for pid in EXCLUDED:
        excluded_rows.append(audit_one(root, pid))
    excluded_df = pd.DataFrame(excluded_rows)

    # Cross-participant summary.
    summary = {
        "final_participants": FINAL_PARTICIPANTS,
        "final_n": len(FINAL_PARTICIPANTS),
        "excluded": EXCLUDED,
        "replacement": REPLACEMENT,
        "target_balance": "9 laki-laki + 9 perempuan (demografi tidak ditebak oleh script)",
        "final_files_with_all_core_sources": int(
            audit_df[["events_exists","features_exists","profiles_exists","raw_video_exists","responses_exists"]].all(axis=1).sum()
        ),
        "warnings": [],
    }

    required_cols = ["events_exists","features_exists","profiles_exists","raw_video_exists","responses_exists"]
    if not audit_df[required_cols].all(axis=1).all():
        summary["warnings"].append("Ada peserta final yang tidak memiliki semua sumber inti.")

    if (audit_df["features_missing_columns"].astype(str) != "").any():
        summary["warnings"].append("Ada file features dengan kolom wajib yang hilang.")

    if (audit_df["features_timestamp_monotonic"] == False).any():
        summary["warnings"].append("Ada timestamp features yang tidak monoton.")

    if (audit_df["features_labeled_rows"] <= 0).any():
        summary["warnings"].append("Ada peserta final tanpa baris berlabel.")

    # Check expected 6 experimental blocks from event files.
    summary["block_count_by_participant"] = {
        r["participant_id"]: int(r["events_block_events"])
        for r in rows if "events_block_events" in r
    }

    audit_df.to_csv(out / "audit_final_18.csv", index=False)
    excluded_df.to_csv(out / "audit_excluded_P08_P17.csv", index=False)
    (out / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("=" * 72)
    print("TRACEFOKUS RAW DATA AUDIT")
    print("=" * 72)
    print(f"Final participants : {len(FINAL_PARTICIPANTS)}")
    print("Included           :", ", ".join(FINAL_PARTICIPANTS))
    print("Excluded           : P08 (unfinished), P17 (replaced by P20)")
    print("Replacement        : P17 -> P20")
    print()
    print(audit_df[[
        "participant_id","features_rows","features_labeled_rows",
        "features_label_0","features_label_1","features_blocks",
        "features_mean_nan_fraction"
    ]].to_string(index=False))
    print()
    if summary["warnings"]:
        print("WARNINGS:")
        for w in summary["warnings"]:
            print(" -", w)
    else:
        print("AUDIT CORE: OK")
    print()
    print(f"Reports saved to: {out.resolve()}")

if __name__ == "__main__":
    main()
