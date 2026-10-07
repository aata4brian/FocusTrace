"""Fixed 50:50 RF/GRU probability blend on frozen participant folds."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
from tracefokus.research.common import digest, read_json, write_json
from tracefokus.research.models import load_model, metrics, predict, validation_threshold
from tracefokus.research.runner import inputs, run_context, reserve, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--rf", type=Path, required=True)
    parser.add_argument("--gru", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    d, manifest, meta, folds = inputs(args.dataset, args.folds)
    rf_run, gru_run = read_json(args.rf / "run.json"), read_json(args.gru / "run.json")
    context = run_context(args.dataset, folds, meta, "blend_rf_gru")
    for name, source, expected in (("rf", args.rf, rf_run), ("gru", args.gru, gru_run)):
        if expected["kind"] != name or expected["dataset_fingerprint"] != context["dataset_fingerprint"] or expected["split_fingerprint"] != context["split_fingerprint"]:
            raise ValueError(f"{name} model is from different data or folds")
    context["experiment_script_sha256"] = digest(__file__)
    context["blend_weight_rf"] = 0.5
    context["blend_weight_gru"] = 0.5
    context["source_model_sha256"] = {
        name: {str(s["fold"]): digest(source / f"fold_{s['fold']:02}" / ("model.pt" if name == "gru" else "model.joblib"))
               for s in folds["splits"]}
        for name, source in (("rf", args.rf), ("gru", args.gru))
    }
    context["selection_rule"] = "Fixed 50:50 probabilities; threshold selected on validation only"
    output = reserve(args.output, context, args.resume)

    for split in folds["splits"]:
        folder = output / f"fold_{split['fold']:02}"
        folder.mkdir(exist_ok=True)
        local_context = dict(run=context, split=split)
        if (folder / "metrics.json").exists():
            if read_json(folder / "metrics.json")["context"] != local_context:
                raise ValueError("Saved fold context mismatch")
            continue
        rf, saved_rf, _ = load_model(args.rf / f"fold_{split['fold']:02}" / "model.joblib")
        gru, saved_gru, _ = load_model(args.gru / f"fold_{split['fold']:02}" / "model.pt")
        if saved_rf != dict(run=rf_run, split=split) or saved_gru != dict(run=gru_run, split=split):
            raise ValueError("Source fold model provenance mismatch")
        predictions = {}
        for role in ("validation", "train", "test"):
            ix = np.isin(d["groups"], split[role])
            probabilities = 0.5 * predict(rf, d["X"][ix], d["observed"][ix]) + 0.5 * predict(gru, d["X"][ix], d["observed"][ix])
            predictions[role] = (ix, probabilities)
            if role == "validation":
                threshold = validation_threshold(d["y"][ix], probabilities)
                write_json(folder / "selection.json", dict(weight_rf=0.5, weight_gru=0.5,
                           threshold=threshold, validation_macro_f1=metrics(d["y"][ix], probabilities, threshold)["macro_f1"]))
        reports = {}
        for role in ("train", "validation", "test"):
            ix, probabilities = predictions[role]
            reports[role] = metrics(d["y"][ix], probabilities, threshold)
            if role == "test":
                frame = manifest.loc[ix].copy()
                frame["probability"] = probabilities
                frame["prediction"] = (probabilities >= threshold).astype(int)
                frame["fold"], frame["threshold"] = split["fold"], threshold
                frame.to_csv(folder / "predictions.csv", index=False)
        write_json(folder / "metrics.json", dict(fold=split["fold"], split=split,
                                                  metrics=reports, context=local_context))
        print(f"Completed blend fold {split['fold']}", flush=True)
    print(summarize(output, len(folds["splits"]), "blend_rf_gru"), flush=True)


if __name__ == "__main__":
    main()
