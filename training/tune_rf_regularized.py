"""Time-bounded, resumable RF regularization search on frozen participant folds."""
import argparse
import ctypes
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tracefokus.dataset import aggregate
from tracefokus.research.common import digest, read_json, write_json
from tracefokus.research.models import (TrainNormalizer, metrics, predict, save_model,
                                        load_model, validation_threshold)
from tracefokus.research.runner import inputs, run_context, reserve, evaluate_artifact, summarize


# The baseline is always a candidate. These alternatives mainly reduce tree complexity.
PLAN = [
    dict(name="leaf5_depth10", n_estimators=200, max_depth=10, min_samples_leaf=5, max_features="sqrt"),
    dict(name="leaf10_depth8", n_estimators=200, max_depth=8, min_samples_leaf=10, max_features="sqrt"),
    dict(name="leaf10_depth12", n_estimators=200, max_depth=12, min_samples_leaf=10, max_features="sqrt"),
    dict(name="leaf20_depth8", n_estimators=200, max_depth=8, min_samples_leaf=20, max_features="sqrt"),
    dict(name="leaf20_depth12", n_estimators=200, max_depth=12, min_samples_leaf=20, max_features="sqrt"),
    dict(name="leaf5_depth8_half", n_estimators=160, max_depth=8, min_samples_leaf=5, max_features=0.5),
    dict(name="leaf10_depth10_half", n_estimators=160, max_depth=10, min_samples_leaf=10, max_features=0.5),
    dict(name="leaf20_depth10_half", n_estimators=160, max_depth=10, min_samples_leaf=20, max_features=0.5),
]


def atomic_save(artifact, path, context, threshold):
    temp = path.with_suffix(path.suffix + ".tmp")
    save_model(artifact, temp, context, threshold)
    temp.replace(path)


def score_validation(artifact, d, split):
    vi = np.isin(d["groups"], split["validation"])
    probability = predict(artifact, d["X"][vi], d["observed"][vi])
    threshold = validation_threshold(d["y"][vi], probability)
    return threshold, metrics(d["y"][vi], probability, threshold)["macro_f1"]


def prevent_idle_sleep():
    # Process-scoped Windows request; reset when this process exits.
    if sys.platform != "win32":
        return
    if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001):
        raise OSError("Could not request Windows to remain awake")


def release_idle_sleep():
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--until", required=True, help="ISO local cutoff with timezone, e.g. 2026-10-06T07:20:00+07:00")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    stop_at = datetime.fromisoformat(args.until)
    if stop_at.tzinfo is None:
        raise ValueError("--until must include a timezone offset")
    d, manifest, meta, fold_data = inputs(args.dataset, args.folds)
    baseline = read_json(args.baseline / "run.json")
    context = run_context(args.dataset, fold_data, meta, "rf_regularized")
    if baseline["kind"] != "rf" or baseline["dataset_fingerprint"] != context["dataset_fingerprint"] or baseline["split_fingerprint"] != context["split_fingerprint"]:
        raise ValueError("Baseline does not match frozen dataset and splits")
    context["experiment_script_sha256"] = digest(__file__)
    context["candidate_plan"] = PLAN
    context["baseline_run_sha256"] = digest(args.baseline / "run.json")
    context["baseline_model_sha256"] = {
        str(s["fold"]): digest(args.baseline / f"fold_{s['fold']:02}" / "model.joblib")
        for s in fold_data["splits"]
    }
    context["selection_rule"] = "Validation macro F1 with validation-only threshold; baseline eligible"
    output = reserve(args.output, context, args.resume)
    write_json(output / "execution_cutoff.json", dict(stop_at=args.until,
               checked_at=datetime.now().astimezone().isoformat(),
               note="Cutoff limits new RF fits; baseline fallback completes remaining folds."))
    prevent_idle_sleep()
    try:
        for index, split in enumerate(fold_data["splits"]):
            folder = output / f"fold_{split['fold']:02}"
            folder.mkdir(exist_ok=True)
            local_context = dict(run=context, split=split)
            model_file = folder / "model.joblib"
            if model_file.exists():
                artifact, saved, threshold = load_model(model_file)
                if saved != local_context:
                    raise ValueError("Saved selected model context mismatch")
            else:
                baseline_artifact, base_saved, _ = load_model(
                    args.baseline / f"fold_{split['fold']:02}" / "model.joblib")
                if base_saved["split"] != split or base_saved["run"] != baseline:
                    raise ValueError("Baseline fold model context mismatch")
                threshold, score = score_validation(baseline_artifact, d, split)
                best_artifact, best_threshold = baseline_artifact, threshold
                best = dict(name="baseline_rf", validation_macro_f1=score, threshold=threshold)
                scores = [best]
                ti = np.isin(d["groups"], split["train"])
                norm = TrainNormalizer().fit(d["X"][ti], d["observed"][ti],
                                              d["groups"][ti], split["train"])
                zt = aggregate(norm.transform(d["X"][ti], d["observed"][ti]))
                remaining_folds = len(fold_data["splits"]) - index
                seconds_left = max(0.0, (stop_at - datetime.now().astimezone()).total_seconds())
                fold_end = datetime.now().astimezone().timestamp() + seconds_left / remaining_folds
                for number, spec in enumerate(PLAN):
                    trial_file = folder / f"trial_{number:02}.joblib"
                    trial_context = dict(**local_context, trial=spec)
                    if trial_file.exists():
                        artifact, saved, _ = load_model(trial_file)
                        if saved != trial_context:
                            raise ValueError("Saved RF candidate context mismatch")
                    else:
                        if datetime.now().astimezone().timestamp() >= fold_end:
                            break
                        model = RandomForestClassifier(
                            n_estimators=spec["n_estimators"], max_depth=spec["max_depth"],
                            min_samples_leaf=spec["min_samples_leaf"],
                            max_features=spec["max_features"], class_weight="balanced",
                            n_jobs=4, random_state=meta["config"]["seed"] + split["fold"])
                        model.fit(zt, d["y"][ti])
                        artifact = dict(kind="rf", model=model, normalizer=norm)
                        candidate_threshold, _ = score_validation(artifact, d, split)
                        atomic_save(artifact, trial_file, trial_context, candidate_threshold)
                    candidate_threshold, candidate_score = score_validation(artifact, d, split)
                    row = dict(name=spec["name"], validation_macro_f1=candidate_score,
                               threshold=candidate_threshold)
                    scores.append(row)
                    if candidate_score > best["validation_macro_f1"]:
                        best, best_artifact, best_threshold = row, artifact, candidate_threshold
                    write_json(folder / "progress.json", dict(candidates=scores, selected=best))
                    print(f"fold {split['fold']} {spec['name']}: validation macro F1={candidate_score:.4f}", flush=True)
                atomic_save(best_artifact, model_file, local_context, best_threshold)
                write_json(folder / "selection.json", dict(best=best, candidates=scores,
                           searched=len(scores) - 1, fold_deadline_timestamp=fold_end,
                           limitation="Single fixed validation partition; test used only after selection"))
                artifact, threshold = best_artifact, best_threshold
            evaluate_artifact(artifact, d, manifest, split, threshold, folder, local_context)
            print(f"Completed regularized RF fold {split['fold']}", flush=True)
        print(summarize(output, len(fold_data["splits"]), "rf_regularized"), flush=True)
    finally:
        release_idle_sleep()


if __name__ == "__main__":
    main()
