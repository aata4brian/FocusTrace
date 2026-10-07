"""Predeclared TraceFokus experiments; participant-local validation and hard budgets.

Run inside the existing project. Frozen input files and existing runs are read-only.
No participant, task, block, timestamp, or label metadata enters model features.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from tracefokus.research.common import digest, identity, read_json, write_json, environment
from tracefokus.research.models import TrainNormalizer
from tracefokus.research.runner import inputs
from tracefokus.research.splits import make_splits


def clock():
    from datetime import datetime
    return datetime.now().astimezone().isoformat()


def atomic_joblib(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, tmp)
    tmp.replace(path)


def make_sixfold(dataset, output):
    from tracefokus.research.common import load_dataset
    d, manifest, meta = load_dataset(dataset)
    cfg = dict(meta["config"], folds=6, seed=42, validation_participants=3)
    payload = dict(method="KFold on participants; 6 folds, seed 42, fixed 3 inner validation participants",
                   seed=42, gender_stratified=False,
                   dataset_fingerprint=digest(Path(dataset) / "dataset.json"),
                   splits=make_splits(d["groups"], cfg), group_overlap=0, test_once=True,
                   limitation="Predeclared secondary evaluation on the same exploratory cohort; not a new independent test cohort")
    payload["split_fingerprint"] = identity(payload)
    output = Path(output)
    if output.exists():
        if read_json(output / "folds.json") != payload:
            raise ValueError("Existing secondary splits differ")
        return
    output.mkdir(parents=True)
    write_json(output / "folds.json", payload)
    rows = []
    for s in payload["splits"]:
        for role in ("train", "validation", "test"):
            for pid in s[role]:
                p = manifest[manifest.participant_id.eq(pid)]
                rows.append(dict(fold=s["fold"], role=role, participant_id=pid, windows=len(p),
                                 **{c: int(p.condition.eq(c).sum()) for c in ("DTE", "TREE", "TIE")}))
    pd.DataFrame(rows).to_csv(output / "fold_balance.csv", index=False)


def causal_context(x, mask, manifest, seconds):
    """Add only earlier samples. Group metadata is used only to reset at boundaries."""
    if seconds == 2:
        return x.copy(), mask.copy()
    if seconds != 6 or x.shape[1] != 10:
        raise ValueError("Only audited 2 s/5 Hz windows and causal 6 s context supported")
    n, _, f = x.shape
    out = np.zeros((n, 30, f), dtype=np.float32)
    observed = np.zeros_like(out, dtype=bool)
    # At time t, previous 2 s windows ending t-4,...,t-1 add 5 unique samples each.
    # Missing history stays zero+false, never filled using the future or another block.
    for _, g in manifest.groupby(["participant_id", "session_id", "block_number"], sort=False):
        lookup = {round(float(manifest.loc[i, "end_sec"]), 4): int(i) for i in g.index}
        for i in g.index:
            end = float(manifest.loc[i, "end_sec"])
            for lag in range(4, 0, -1):
                j = lookup.get(round(end - lag, 4))
                if j is not None:
                    assert float(manifest.loc[j, "end_sec"]) < end
                    k = (4 - lag) * 5
                    out[i, k:k+5] = x[j, :5]
                    observed[i, k:k+5] = mask[j, :5]
            out[i, -10:] = x[i]
            observed[i, -10:] = mask[i]
    return out, observed


def feature_view(x, mask, names, view):
    if view == "all":
        return x, mask
    keep = [i for i, n in enumerate(names) if n not in {
        "head_yaw_sin", "head_yaw_cos", "head_pitch_sin", "head_pitch_cos",
        "gaze_horizontal", "gaze_vertical"}]
    return x[..., keep], mask[..., keep]


def motion_features(z):
    """Head/gaze level, variability, displacement and changes; no protocol features."""
    delta = np.diff(z, axis=1)
    t = np.linspace(-1, 1, z.shape[1], dtype=np.float32)
    slope = (z * t[None, :, None]).sum(axis=1) / (t*t).sum()
    parts = [z.mean(1), z.std(1), np.quantile(z, .1, axis=1), np.quantile(z, .5, axis=1),
             np.quantile(z, .9, axis=1), z[:, -1] - z[:, 0], slope,
             np.abs(delta).mean(1), delta.std(1), np.abs(delta).max(1)]
    result = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite motion features")
    return result


def macro(y, pred, classes):
    cm = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y, int), np.asarray(pred, int)), 1)
    den = cm.sum(0) + cm.sum(1)
    return float(np.divide(2*np.diag(cm), den, out=np.zeros(classes), where=den > 0).mean())


def validation_score(y, p, groups, classes):
    """Equal participant influence; thresholds and all selections use validation only."""
    subsets = [np.flatnonzero(groups == pid) for pid in sorted(set(groups))]
    if classes == 3:
        pred = p.argmax(1)
        return float(np.mean([macro(y[ix], pred[ix], classes) for ix in subsets])), .5
    candidates = np.linspace(.2, .8, 31)
    scores = [float(np.mean([macro(y[ix], p[ix, 1] >= t, 2) for ix in subsets])) for t in candidates]
    j = max(range(len(scores)), key=lambda i: (scores[i], -abs(candidates[i]-.5), -candidates[i]))
    return scores[j], float(candidates[j])


def sample_weights(y, groups, balanced, classes):
    counts = {pid: int(np.sum(groups == pid)) for pid in set(groups)}
    weights = np.array([1.0/counts[pid] for pid in groups], np.float32)
    if balanced:
        totals = np.bincount(y, weights=weights, minlength=classes)
        factors = totals.sum() / (classes * np.maximum(totals, 1e-8))
        weights *= factors[y]
    return weights / weights.mean()


def tabular_plan():
    plan = []
    for view in ("all", "relative"):
        for depth, leaf in ((10, 8), (18, 8), (None, 20)):
            plan.append(dict(family="extra", view=view, depth=depth, leaf=leaf, balanced=True))
        for depth, leaf in ((12, 8), (18, 16)):
            plan.append(dict(family="rf", view=view, depth=depth, leaf=leaf, balanced=True))
        for depth, regularization in ((3, 3.), (5, 5.), (3, 10.)):
            plan.append(dict(family="hist", view=view, depth=depth, regularization=regularization, balanced=True))
        for c in (.3, 1., 3.):
            plan.append(dict(family="svm", view=view, C=c, balanced=True))
        plan.append(dict(family="logistic", view=view, C=.3, balanced=True))
    return plan


def neural_plan(wave=0):
    plan = []
    families = ("gru_pool", "lstm_pool", "tcn", "mlp")
    for i, family in enumerate(families):
        for j, (view, hidden, dropout, lr, balanced) in enumerate([
            ("all", 32, .35, .001, False),
            ("relative", 32, .35, .001, False),
            ("all", 64, .4, .0005, True),
            ("relative", 48, .25, .0007, True),
        ]):
            plan.append(dict(family=family, view=view, hidden=hidden, dropout=dropout,
                             learning_rate=lr * (.7 if wave else 1), balanced=balanced,
                             epochs=55, patience=9, batch_size=128, weight_decay=.003,
                             focal_gamma=0., augmentation=.025 if wave else 0.))
    # Fixed regularized alternatives, not selected based on outer-test findings.
    plan.extend([
        dict(family="gru_pool", view="relative", hidden=32, dropout=.5, learning_rate=.0005,
             balanced=True, epochs=65, patience=10, batch_size=128, weight_decay=.01, focal_gamma=1., augmentation=.025),
        dict(family="tcn", view="all", hidden=48, dropout=.4, learning_rate=.001,
             balanced=False, epochs=65, patience=10, batch_size=128, weight_decay=.01, focal_gamma=0., augmentation=.025),
    ])
    return plan


def tabular_model(spec, seed, threads):
    family = spec["family"]
    if family in ("extra", "rf"):
        cls = ExtraTreesClassifier if family == "extra" else RandomForestClassifier
        return cls(n_estimators=spec.get("n_estimators", 300), max_depth=spec["depth"], min_samples_leaf=spec["leaf"],
                   max_features=.7, n_jobs=threads, random_state=seed)
    if family == "hist":
        return HistGradientBoostingClassifier(max_iter=180, max_depth=spec["depth"],
            max_leaf_nodes=2**spec["depth"], learning_rate=.05,
            l2_regularization=spec["regularization"], early_stopping=False, random_state=seed)
    if family == "svm":
        return make_pipeline(StandardScaler(), SVC(C=spec["C"], gamma="scale", probability=False,
                                                  cache_size=256, random_state=seed))
    return make_pipeline(StandardScaler(), LogisticRegression(C=spec["C"], max_iter=1200, random_state=seed))


def tabular_probability(model, z):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(z)
    score = model.decision_function(z)
    if score.ndim == 1:
        pos = 1/(1+np.exp(-np.clip(score, -30, 30)))
        return np.stack([1-pos, pos], axis=1)
    score = score - score.max(1, keepdims=True)
    p = np.exp(score)
    return p/p.sum(1, keepdims=True)


def make_neural(dim, classes, spec):
    import torch
    from torch import nn

    class TemporalNet(nn.Module):
        def __init__(self):
            super().__init__()
            h, drop, family = spec["hidden"], spec["dropout"], spec["family"]
            self.family = family
            self.project = nn.Sequential(nn.Linear(dim, h), nn.LayerNorm(h), nn.GELU())
            if family in ("gru_pool", "lstm_pool"):
                cls = nn.GRU if family == "gru_pool" else nn.LSTM
                self.temporal = cls(h, h, batch_first=True)
                self.attention = nn.Linear(h, 1)
                size = 3*h
            elif family == "tcn":
                self.temporal = nn.Sequential(
                    nn.Conv1d(h, h, 3, padding=1), nn.GELU(), nn.Dropout(drop),
                    nn.Conv1d(h, h, 3, padding=2, dilation=2), nn.GELU(),
                    nn.Conv1d(h, h, 3, padding=4, dilation=4), nn.GELU())
                size = 3*h
            else:
                size = 3*h
            self.head = nn.Sequential(nn.LayerNorm(size), nn.Dropout(drop), nn.Linear(size, h),
                                      nn.GELU(), nn.Dropout(drop), nn.Linear(h, classes))

        def forward(self, z):
            z = self.project(z)
            if self.family in ("gru_pool", "lstm_pool"):
                z, _ = self.temporal(z)
                pooled = (z * torch.softmax(self.attention(z), dim=1)).sum(1)
            elif self.family == "tcn":
                z = self.temporal(z.transpose(1, 2)).transpose(1, 2)
                pooled = z.mean(1)
            else:
                pooled = z.mean(1)
            return self.head(torch.cat([pooled, z.max(1).values, z[:, -1]], dim=1))
    return TemporalNet()


def neural_probability(model, z, device, batch=512):
    import torch
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(z), batch):
            chunks.append(torch.softmax(model(torch.as_tensor(z[i:i+batch], device=device)), dim=1).cpu().numpy())
    return np.concatenate(chunks)


def save_torch(value, path):
    import torch
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def train_neural(zt, yt, gt, zv, yv, gv, classes, spec, seed, folder, signature, device, deadline, threads):
    import torch
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = make_neural(zt.shape[-1], classes, spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    weights = sample_weights(yt, gt, spec["balanced"], classes)
    state = dict(epoch=0, best=-1., stale=0, curve=[], completed=False, elapsed_seconds=0.)
    best_state = None
    last = folder / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location=device, weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("Neural resume provenance mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state, best_state = saved["progress"], saved["best_model"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        generator.set_state(saved["loader_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
    ds = torch.utils.data.TensorDataset(torch.from_numpy(zt), torch.as_tensor(yt, dtype=torch.long), torch.from_numpy(weights))
    loader = torch.utils.data.DataLoader(ds, batch_size=spec["batch_size"], shuffle=True,
                                        generator=generator, num_workers=0)
    while not state["completed"] and time.monotonic() < deadline:
        started = time.monotonic()
        model.train()
        total = 0.
        interrupted = False
        for bx, by, bw in loader:
            if time.monotonic() >= deadline:
                interrupted = True
                break
            bx, by, bw = bx.to(device), by.to(device), bw.to(device)
            if spec["augmentation"]:
                # Mask channels (second half) remain exact binary observations.
                half = bx.shape[-1] // 2
                noise = torch.randn_like(bx[..., :half]) * spec["augmentation"]
                bx = bx.clone()
                bx[..., :half] += noise * bx[..., half:]
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            losses = torch.nn.functional.cross_entropy(logits, by, reduction="none")
            if spec["focal_gamma"]:
                pt = torch.softmax(logits, 1).gather(1, by[:, None]).squeeze(1)
                losses = losses * (1-pt)**spec["focal_gamma"]
            loss = (losses*bw).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite neural loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach().item()) * len(by)
        if interrupted:
            # The prior full-epoch checkpoint stays valid; no half-epoch selected model.
            break
        p = neural_probability(model, zv, device)
        score, threshold = validation_score(yv, p, gv, classes)
        state["epoch"] += 1
        state["curve"].append(dict(epoch=state["epoch"], train_loss=total/len(yt),
                                     validation_participant_macro_f1=score, threshold=threshold))
        if score > state["best"] + 1e-6:
            state["best"], state["stale"] = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            state["stale"] += 1
        state["elapsed_seconds"] += time.monotonic() - started
        state["completed"] = state["epoch"] >= spec["epochs"] or state["stale"] >= spec["patience"]
        save_torch(dict(signature=signature, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        best_model=best_state, progress=state, torch_rng=torch.get_rng_state(),
                        loader_rng=generator.get_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else None), last)
        write_json(folder / "history.json", state)
        print(f"epoch {state['epoch']}: validation participant macro-F1={score:.4f}", flush=True)
    if not state["completed"]:
        return None, state
    model.load_state_dict(best_state)
    return model, state


def report(y, p, groups, classes, threshold):
    pred = p[:, 1] >= threshold if classes == 2 else p.argmax(1)
    c = classification_report(y, pred, labels=list(range(classes)), zero_division=0, output_dict=True)
    by_participant = {str(g): macro(y[groups == g], pred[groups == g], classes) for g in sorted(set(groups))}
    return dict(samples=len(y), accuracy=float(np.mean(y == pred)), macro_f1=c["macro avg"]["f1-score"],
                participant_mean_macro_f1=float(np.mean(list(by_participant.values()))),
                participant_macro_f1=by_participant, classification_report=c,
                confusion_matrix=confusion_matrix(y, pred, labels=list(range(classes))).tolist()), pred.astype(int)


def summarize(output, splits, manifest, classes):
    reports, frames = [], []
    for s in splits:
        f = output / f"fold_{s['fold']:02}"
        if (f / "metrics.json").exists():
            reports.append(read_json(f / "metrics.json"))
            frames.append(pd.read_csv(f / "predictions.csv"))
    payload = dict(status="COMPLETE" if len(reports) == len(splits) else "INCOMPLETE",
                   completed_folds=len(reports), expected_folds=len(splits), updated_at=clock())
    if not frames:
        write_json(output / "summary.json", payload)
        return payload
    frame = pd.concat(frames, ignore_index=True)
    if frame.window_id.duplicated().any():
        raise ValueError("Duplicate out-of-fold window IDs")
    if len(reports) == len(splits) and set(frame.window_id) != set(manifest.window_id):
        raise ValueError("OOF coverage differs from frozen dataset")
    frame.to_csv(output / "oof_predictions.csv", index=False)
    fold_values = [r["metrics"]["test"]["macro_f1"] for r in reports]
    payload["fold_metrics"] = dict(macro_f1=dict(mean=float(np.mean(fold_values)),
        std_sample=float(np.std(fold_values, ddof=1)) if len(fold_values) > 1 else None),
        accuracy=dict(mean=float(np.mean([r["metrics"]["test"]["accuracy"] for r in reports]))))
    payload["pooled_oof"] = classification_report(frame.model_label, frame.prediction,
        labels=list(range(classes)), zero_division=0, output_dict=True)
    payload["overfit_warning_folds"] = [r["fold"] for r in reports if
        r["metrics"]["train"]["participant_mean_macro_f1"] - r["metrics"]["validation"]["participant_mean_macro_f1"] > .2]
    payload["suspicious_validation_folds"] = [r["fold"] for r in reports if r["metrics"]["validation"]["macro_f1"] >= .98]
    if payload["suspicious_validation_folds"]:
        payload["status"] = "REVIEW"
    payload["group_overlap"] = 0
    payload["limitation"] = "Exploratory cohort; repeated model-family comparisons require a new participant cohort for confirmation. Fold SD is not a confidence interval. Labels largely reflect assigned tasks."
    rows = []
    for field in ("participant_id", "condition", "block_code", "label_source"):
        for value, g in frame.groupby(field):
            rows.append(dict(grouping=field, group=str(value), windows=len(g),
                             accuracy=float(np.mean(g.model_label == g.prediction)),
                             macro_f1=macro(g.model_label.values, g.prediction.values, classes)))
    pd.DataFrame(rows).to_csv(output / "error_analysis.csv", index=False)
    write_json(output / "summary.json", payload)
    return payload


def run(args):
    from datetime import datetime
    from contextlib import nullcontext
    deadline = time.monotonic() + min(args.hours*3600, max(0.,
        (datetime.fromisoformat(args.until) - datetime.now().astimezone()).total_seconds()))
    d, manifest, meta, fold_data = inputs(args.dataset, args.folds)
    classes = 2 if args.target == "binary" else 3
    y = d["y"] if classes == 2 else manifest.condition.map({"DTE": 0, "TREE": 1, "TIE": 2}).to_numpy()
    if not np.isin(y, np.arange(classes)).all():
        raise ValueError("Unknown three-class condition label")
    x, mask = causal_context(d["X"], d["observed"], manifest, args.context_seconds)
    plan = tabular_plan() if args.suite == "tabular" else neural_plan(args.wave)
    if args.smoke:
        plan = [plan[0]]
        if args.suite == "neural":
            plan[0] = dict(plan[0], epochs=2, patience=2)
        else:
            plan[0] = dict(plan[0], n_estimators=20)
    plan = plan[:args.max_candidates]
    device = "cpu"
    if args.suite == "neural":
        import torch
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device == "auto":
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
    context = dict(dataset_fingerprint=digest(args.dataset / "dataset.json"),
        split_fingerprint=fold_data["split_fingerprint"], script_sha256=digest(__file__),
        target=args.target, label_mapping={"DTE": 0, "TREE": 1, "TIE": 2} if classes == 3 else {"on_task": 0, "TIE": 1},
        suite=args.suite, context_seconds=args.context_seconds, plan=plan, environment=environment(),
        device=device, selection="Highest validation mean participant macro-F1; threshold validation only; ties favor earliest trial",
        train_weighting="Equal participant weight, optional inverse class frequency using train only",
        forbidden_model_inputs=["participant_id", "session_id", "condition", "assigned_condition", "block_code", "block_number", "start_sec", "end_sec", "label"],
        context_boundary="participant/session/block boundaries reset history; no boundary metadata enters features",
        code_hashes={p.name:digest(p) for p in (ROOT / "backend/tracefokus/research").glob("*.py")})
    output = args.output
    if output.exists():
        if not args.resume or read_json(output / "run.json") != context:
            raise ValueError("Existing run has different provenance or needs --resume")
    else:
        output.mkdir(parents=True)
        write_json(output / "run.json", context)
    write_json(output / "execution_budget.json", dict(until=args.until, hours=args.hours, started_at=clock()))
    splits = fold_data["splits"][:1] if args.smoke else fold_data["splits"]
    for si, split in enumerate(splits):
        folder = output / f"fold_{split['fold']:02}"
        folder.mkdir(exist_ok=True)
        if (folder / "metrics.json").exists():
            if read_json(folder / "metrics.json")["run_fingerprint"] != identity(context):
                raise ValueError("Fold provenance differs")
            continue
        roles = {role:np.isin(d["groups"], split[role]) for role in ("train", "validation", "test")}
        for role in roles:
            if set(y[roles[role]]) != set(range(classes)):
                raise ValueError("Missing class in participant split")
        ti, vi = roles["train"], roles["validation"]
        # Equal remaining-fold budgets prevent exhausting all compute on fold one.
        fold_deadline = time.monotonic() + max(0., deadline-time.monotonic())/(len(splits)-si)
        scored, cache = [], {}
        for number, spec in enumerate(plan):
            candidate_folder = folder / f"trial_{number:02}"
            candidate_folder.mkdir(exist_ok=True)
            score_file = candidate_folder / "score.json"
            if score_file.exists():
                scored.append(read_json(score_file))
                continue
            if time.monotonic() >= fold_deadline:
                break
            view = spec["view"]
            if view not in cache:
                xx, mm = feature_view(x, mask, d["features"], view)
                norm = TrainNormalizer().fit(xx[ti], mm[ti], d["groups"][ti], split["train"])
                zt, zv = norm.transform(xx[ti], mm[ti]), norm.transform(xx[vi], mm[vi])
                if args.suite == "tabular":
                    zt, zv = motion_features(zt), motion_features(zv)
                cache[view] = (norm, zt, zv)
            norm, zt, zv = cache[view]
            signature = identity(dict(run=context, split=split, trial=number, normalizer=norm.export()))
            seed = 42000 + args.wave*10000 + split["fold"]*100 + number
            print(f"START fold {split['fold']} trial {number}: {json.dumps(spec)}", flush=True)
            started = time.monotonic()
            if args.suite == "tabular":
                model = tabular_model(spec, seed, args.threads)
                weights = sample_weights(y[ti], d["groups"][ti], spec["balanced"], classes)
                with threadpool_limits(limits=args.threads):
                    if isinstance(model, type(make_pipeline(StandardScaler(), SVC()))):
                        model.fit(zt, y[ti], **{model.steps[-1][0] + "__sample_weight":weights})
                    else:
                        model.fit(zt, y[ti], sample_weight=weights)
                p = tabular_probability(model, zv)
                atomic_joblib(dict(model=model, normalizer=norm.export(), spec=spec, signature=signature), candidate_folder / "model.joblib")
                progress = dict(completed=True)
            else:
                model, progress = train_neural(zt, y[ti], d["groups"][ti], zv, y[vi], d["groups"][vi],
                    classes, spec, seed, candidate_folder, signature, device, fold_deadline, args.threads)
                if model is None:
                    print(f"PAUSED fold {split['fold']} trial {number}", flush=True)
                    break
                p = neural_probability(model, zv, device)
                save_torch(dict(state_dict=model.state_dict(), normalizer=norm.export(), spec=spec,
                                input_dim=zt.shape[-1], classes=classes, signature=signature), candidate_folder / "model.pt")
            score, threshold = validation_score(y[vi], p, d["groups"][vi], classes)
            result = dict(trial=number, spec=spec, validation_participant_macro_f1=score, threshold=threshold,
                          elapsed_seconds=time.monotonic()-started, completed=True, progress=progress)
            write_json(score_file, result)
            scored.append(result)
            print(f"DONE fold {split['fold']} trial {number}: validation={score:.4f}", flush=True)
        if not scored:
            print("No completed candidate in fold budget; saved checkpoints retained", flush=True)
            continue
        best = max(scored, key=lambda r:(r["validation_participant_macro_f1"], -r["trial"]))
        selected_folder = folder / f"trial_{best['trial']:02}"
        spec, threshold = best["spec"], best["threshold"]
        suffix = "joblib" if args.suite == "tabular" else "pt"
        src = selected_folder / f"model.{suffix}"
        shutil.copyfile(src, folder / f"selected_model.{suffix}")
        if args.suite == "tabular":
            saved = joblib.load(src)
            model = saved["model"]
        else:
            import torch
            saved = torch.load(src, map_location=device, weights_only=False)
            model = make_neural(saved["input_dim"], classes, saved["spec"]).to(device)
            model.load_state_dict(saved["state_dict"])
        norm = TrainNormalizer.restore(saved["normalizer"])
        xx, mm = feature_view(x, mask, d["features"], spec["view"])
        fold_reports = {}
        for role, ix in roles.items():
            z = norm.transform(xx[ix], mm[ix])
            if args.suite == "tabular":
                p = tabular_probability(model, motion_features(z))
            else:
                p = neural_probability(model, z, device)
            fold_reports[role], pred = report(y[ix], p, d["groups"][ix], classes, threshold)
            if role == "test":
                frame = manifest.loc[ix].copy()
                frame["model_label"] = y[ix]
                frame["prediction"], frame["fold"], frame["threshold"] = pred, split["fold"], threshold
                for c in range(classes):
                    frame[f"probability_{c}"] = p[:, c]
                frame.to_csv(folder / "predictions.csv", index=False)
        write_json(folder / "selection.json", dict(best=best, candidates=scored,
                   test_used_for_selection=False, completed_candidate_count=len(scored), planned_candidates=len(plan)))
        write_json(folder / "metrics.json", dict(fold=split["fold"], split=split, metrics=fold_reports,
                                                  run_fingerprint=identity(context)))
        print(f"FOLD COMPLETE {split['fold']}; test evaluated once after validation selection", flush=True)
        if not args.smoke:
            summarize(output, fold_data["splits"], manifest, classes)
    result = summarize(output, fold_data["splits"], manifest, classes)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--folds", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--init-sixfold", action="store_true")
    p.add_argument("--suite", choices=["tabular", "neural"], default="tabular")
    p.add_argument("--target", choices=["binary", "three"], default="binary")
    p.add_argument("--context-seconds", type=int, choices=[2, 6], default=2)
    p.add_argument("--until", default="2026-10-07T15:00:00+07:00")
    p.add_argument("--hours", type=float, default=20.)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-candidates", type=int, default=100)
    p.add_argument("--wave", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.init_sixfold:
        make_sixfold(args.dataset, args.folds)
        print("Secondary six-fold split frozen", flush=True)
        return
    if args.output is None or not math.isfinite(args.hours) or args.hours <= 0:
        p.error("output and finite positive hours required")
    run(args)


if __name__ == "__main__":
    main()
