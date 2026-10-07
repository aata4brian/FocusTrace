"""Predeclared RF feature ablations on frozen participant folds."""
import argparse
import ctypes
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
from tracefokus.research.common import digest, read_json, write_json
from tracefokus.research.models import fit_baseline, load_model, save_model
from tracefokus.research.runner import inputs, run_context, reserve, evaluate_artifact, summarize

SUBSETS = {
    "head_only": list(range(0, 8)) + [14],
    "gaze_only": list(range(8, 14)) + list(range(15, 21)),
    "without_explicit_relative": [0, 1, 2, 3, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20],
}


def keep_awake():
    if sys.platform == "win32" and not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
        raise OSError("Could not request Windows to remain awake")


def release_awake():
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--until", required=True, help="ISO cutoff with timezone")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    stop_at = datetime.fromisoformat(args.until)
    if stop_at.tzinfo is None:
        raise ValueError("--until requires timezone")

    d, manifest, meta, folds = inputs(args.dataset, args.folds)
    names = [str(x) for x in d["features"]]
    if len(names) != 21 or names[0] != "head_yaw_sin" or names[20] != "gaze_UNKNOWN":
        raise ValueError("Feature schema changed; inspect ablation indices")
    keep_awake()
    try:
        for label, indices in SUBSETS.items():
            subset = dict(d)
            subset["X"] = d["X"][:, :, indices]
            subset["observed"] = d["observed"][:, :, indices]
            subset["features"] = d["features"][indices]
            context = run_context(args.dataset, folds, meta, "rf_feature_ablation")
            context["experiment_script_sha256"] = digest(__file__)
            context["subset"] = label
            context["feature_indices"] = indices
            context["feature_names"] = [names[i] for i in indices]
            context["limitation"] = "Fixed feature-set ablation; each test participant evaluated once per subset"
            output = reserve(args.output / label, context, args.resume)
            write_json(output / "execution_cutoff.json", dict(stop_at=args.until,
                       checked_at=datetime.now().astimezone().isoformat()))
            for split in folds["splits"]:
                folder = output / f"fold_{split['fold']:02}"
                folder.mkdir(exist_ok=True)
                local_context = dict(run=context, split=split)
                file = folder / "model.joblib"
                if file.exists():
                    artifact, saved, threshold = load_model(file)
                    if saved != local_context:
                        raise ValueError("Saved ablation model context mismatch")
                else:
                    if datetime.now().astimezone() >= stop_at:
                        print(f"Cutoff reached before {label} fold {split['fold']}", flush=True)
                        print(summarize(output, len(folds["splits"]), "rf_feature_ablation"), flush=True)
                        return
                    ti, vi = [np.isin(d["groups"], split[role]) for role in ("train", "validation")]
                    artifact = fit_baseline("rf", subset["X"][ti], subset["observed"][ti],
                                            d["y"][ti], d["groups"][ti], split["train"],
                                            meta["config"]["rf"], meta["config"]["seed"] + split["fold"])
                    from tracefokus.research.models import predict, validation_threshold
                    probability = predict(artifact, subset["X"][vi], subset["observed"][vi])
                    threshold = validation_threshold(d["y"][vi], probability)
                    temp = file.with_suffix(".joblib.tmp")
                    save_model(artifact, temp, local_context, threshold)
                    temp.replace(file)
                evaluate_artifact(artifact, subset, manifest, split, threshold, folder, local_context)
                print(f"Completed {label} fold {split['fold']}", flush=True)
            summary = summarize(output, len(folds["splits"]), "rf_feature_ablation")
            print(f"{label}: {summary}", flush=True)
    finally:
        release_awake()


if __name__ == "__main__":
    main()
