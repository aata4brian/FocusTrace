"""Exploratory grouped comparison: fold-local selection of simple tabular models."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tracefokus.dataset import aggregate
from tracefokus.research.common import digest, read_json, write_json
from tracefokus.research.models import TrainNormalizer, metrics, validation_threshold
from tracefokus.research.runner import inputs, run_context, reserve, summarize


PLAN = [
    {"name": "logreg_c0.1", "family": "logreg", "C": 0.1},
    {"name": "logreg_c1", "family": "logreg", "C": 1.0},
    {"name": "histgb_15", "family": "histgb", "max_leaf_nodes": 15},
    {"name": "histgb_31", "family": "histgb", "max_leaf_nodes": 31},
]


def make_candidate(spec, seed):
    if spec["family"] == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=spec["C"], class_weight="balanced",
                               max_iter=1000, random_state=seed),
        )
    return HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.05, max_leaf_nodes=spec["max_leaf_nodes"],
        l2_regularization=1.0, class_weight="balanced",
        early_stopping=False, random_state=seed,
    )


def evaluate(model, norm, d, manifest, split, threshold, folder, context):
    reports = {}
    for role in ("train", "validation", "test"):
        ix = np.isin(d["groups"], split[role])
        z = aggregate(norm.transform(d["X"][ix], d["observed"][ix]))
        probability = model.predict_proba(z)[:, 1]
        reports[role] = metrics(d["y"][ix], probability, threshold)
        if role == "test":
            frame = manifest.loc[ix].copy()
            frame["probability"] = probability
            frame["prediction"] = (probability >= threshold).astype(int)
            frame["fold"] = split["fold"]
            frame["threshold"] = threshold
            frame.to_csv(folder / "predictions.csv", index=False)
    write_json(folder / "metrics.json", dict(fold=split["fold"], split=split,
                                               metrics=reports, context=context))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    d, manifest, meta, fold_data = inputs(args.dataset, args.folds)
    context = run_context(args.dataset, fold_data, meta, "exploratory_tabular")
    context["experiment_script_sha256"] = digest(__file__)
    context["candidate_plan"] = PLAN
    context["selection_rule"] = "Best validation macro F1 after validation-only threshold"
    output = reserve(args.output, context, args.resume)
    for split in fold_data["splits"]:
        folder = output / f"fold_{split['fold']:02}"
        folder.mkdir(exist_ok=True)
        local_context = dict(run=context, split=split)
        model_file = folder / "model.joblib"
        if model_file.exists():
            payload = joblib.load(model_file)
            if payload["context"] != local_context:
                raise ValueError("Saved fold context mismatch")
            model, norm, threshold = (payload[k] for k in ("model", "normalizer", "threshold"))
        else:
            ti, vi = [np.isin(d["groups"], split[role]) for role in ("train", "validation")]
            norm = TrainNormalizer().fit(d["X"][ti], d["observed"][ti],
                                          d["groups"][ti], split["train"])
            zt = aggregate(norm.transform(d["X"][ti], d["observed"][ti]))
            zv = aggregate(norm.transform(d["X"][vi], d["observed"][vi]))
            scored = []
            for spec in PLAN:
                candidate = make_candidate(spec, meta["config"]["seed"] + split["fold"])
                candidate.fit(zt, d["y"][ti])
                probability = candidate.predict_proba(zv)[:, 1]
                threshold_candidate = validation_threshold(d["y"][vi], probability)
                score = metrics(d["y"][vi], probability, threshold_candidate)["macro_f1"]
                scored.append(dict(name=spec["name"], family=spec["family"],
                                   validation_macro_f1=score, threshold=threshold_candidate))
                if len(scored) == 1 or score > max(x["validation_macro_f1"] for x in scored[:-1]):
                    model, threshold = candidate, threshold_candidate
                print(f"fold {split['fold']} {spec['name']}: validation macro F1={score:.4f}", flush=True)
            best = max(scored, key=lambda x: x["validation_macro_f1"])
            write_json(folder / "selection.json", dict(best=best, candidates=scored,
                       limitation="Same validation partition chooses model and threshold; exploratory comparison"))
            temp = model_file.with_suffix(".joblib.tmp")
            joblib.dump(dict(model=model, normalizer=norm, threshold=threshold,
                             context=local_context), temp)
            temp.replace(model_file)
        evaluate(model, norm, d, manifest, split, threshold, folder, local_context)
        print(f"Completed exploratory tabular fold {split['fold']}", flush=True)
    result = summarize(output, len(fold_data["splits"]), "exploratory_tabular")
    print(result, flush=True)


if __name__ == "__main__":
    main()
