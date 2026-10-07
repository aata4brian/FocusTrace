"""Resumable, time-bounded Temporal CNN on frozen participant folds."""
import argparse
import ctypes
from datetime import datetime
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from tracefokus.research.common import digest, identity, write_json
from tracefokus.research.models import (TrainNormalizer, metrics, validation_threshold,
                                        save_torch)
from tracefokus.research.runner import inputs, run_context, reserve, summarize


CONFIG = dict(channels=48, dropout=0.2, epochs=40, patience=6,
              batch_size=64, learning_rate=0.001, weight_decay=0.0001,
              class_weight=True)


class TemporalCNN(nn.Module):
    def __init__(self, input_dim, channels=48, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, channels, 3, padding=1),
            nn.GroupNorm(4, channels), nn.GELU(),
            nn.Conv1d(channels, channels, 3, padding=2, dilation=2),
            nn.GroupNorm(4, channels), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * channels, 1))

    def forward(self, x):
        h = self.features(x.transpose(1, 2))
        pooled = torch.cat((h.mean(dim=-1), h.amax(dim=-1)), dim=1)
        return self.head(pooled).squeeze(-1)


def probability(model, z, batch_size=512):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            logits = model(torch.from_numpy(z[start:start + batch_size]))
            out.append(torch.sigmoid(logits).numpy())
    return np.concatenate(out)


def keep_awake():
    if sys.platform == "win32" and not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
        raise OSError("Could not request Windows to remain awake")


def release_awake():
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def fit_fold(d, split, folder, context, deadline):
    seed = context["run"]["config"]["seed"] + split["fold"] + 7000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ti, vi = [np.isin(d["groups"], split[k]) for k in ("train", "validation")]
    norm = TrainNormalizer().fit(d["X"][ti], d["observed"][ti], d["groups"][ti], split["train"])
    zt = norm.transform(d["X"][ti], d["observed"][ti])
    zv = norm.transform(d["X"][vi], d["observed"][vi])
    model = TemporalCNN(zt.shape[-1], CONFIG["channels"], CONFIG["dropout"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                                   weight_decay=CONFIG["weight_decay"])
    ratio = float((d["y"][ti] == 0).sum() / (d["y"][ti] == 1).sum()) if CONFIG["class_weight"] else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(zt),
                        torch.as_tensor(d["y"][ti], dtype=torch.float32)),
                        batch_size=CONFIG["batch_size"], shuffle=True,
                        generator=generator, num_workers=0)
    signature = identity(dict(context=context, config=CONFIG, seed=seed,
                              normalizer=norm.export(), device="cpu"))
    state = dict(epoch=0, best=-1.0, best_epoch=0, stale=0, curve=[],
                 completed=False, elapsed_seconds=0.0)
    best_state = None
    last = folder / "training" / "last.pt"
    last.parent.mkdir(exist_ok=True)
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise ValueError("TCN resume mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state, best_state = saved["progress"], saved["best_model"]
        torch.set_rng_state(saved["torch_rng"])
        generator.set_state(saved["loader_rng"])
    started = time.monotonic()
    while not state["completed"] and state["epoch"] < CONFIG["epochs"]:
        if datetime.now().astimezone().timestamp() >= deadline:
            break
        model.train()
        total = 0.0
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite TCN loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(bx)
        pv = probability(model, zv)
        score = float(f1_score(d["y"][vi], pv >= 0.5, labels=[0, 1],
                               average="macro", zero_division=0))
        state["epoch"] += 1
        state["curve"].append(dict(epoch=state["epoch"],
                                   train_loss=total / ti.sum(),
                                   validation_macro_f1_at_05=score))
        if score > state["best"] + 1e-6:
            state["best"], state["best_epoch"], state["stale"] = score, state["epoch"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_torch(folder / "training" / "best.pt",
                       dict(state_dict=best_state, normalizer=norm.export(),
                            config=CONFIG, input_dim=zt.shape[-1], context=context,
                            epoch=state["epoch"], signature=signature))
        else:
            state["stale"] += 1
        state["completed"] = state["stale"] >= CONFIG["patience"] or state["epoch"] >= CONFIG["epochs"]
        now = time.monotonic()
        state["elapsed_seconds"] += now - started
        started = now
        save_torch(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                             best_model=best_state, progress=state, signature=signature,
                             torch_rng=torch.get_rng_state(), loader_rng=generator.get_state()))
        write_json(folder / "training" / "history.json", state)
        print(f"fold {split['fold']} epoch {state['epoch']}: validation macro F1={score:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError("No TCN epoch completed before cutoff; resume with a later cutoff")
    model.load_state_dict(best_state)
    model.eval()
    return model, norm, state, zt.shape[-1]


def evaluate(model, norm, d, manifest, split, threshold, folder, context):
    reports = {}
    for role in ("train", "validation", "test"):
        ix = np.isin(d["groups"], split[role])
        p = probability(model, norm.transform(d["X"][ix], d["observed"][ix]))
        reports[role] = metrics(d["y"][ix], p, threshold)
        if role == "test":
            frame = manifest.loc[ix].copy()
            frame["probability"] = p
            frame["prediction"] = (p >= threshold).astype(int)
            frame["fold"], frame["threshold"] = split["fold"], threshold
            frame.to_csv(folder / "predictions.csv", index=False)
    write_json(folder / "metrics.json", dict(fold=split["fold"], split=split,
                                              metrics=reports, context=context))


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
        raise ValueError("--until requires a timezone offset")
    d, manifest, meta, fold_data = inputs(args.dataset, args.folds)
    context = run_context(args.dataset, fold_data, meta, "tcn")
    context["experiment_script_sha256"] = digest(__file__)
    context["architecture"] = "2 Conv1d layers, GroupNorm/GELU, mean+max pooling"
    context["tcn_config"] = CONFIG
    output = reserve(args.output, context, args.resume)
    write_json(output / "execution_cutoff.json", dict(stop_at=args.until,
               checked_at=datetime.now().astimezone().isoformat()))
    keep_awake()
    try:
        for index, split in enumerate(fold_data["splits"]):
            folder = output / f"fold_{split['fold']:02}"
            folder.mkdir(exist_ok=True)
            local_context = dict(run=context, split=split)
            model_file = folder / "model.pt"
            if model_file.exists():
                saved = torch.load(model_file, map_location="cpu", weights_only=True)
                if saved["context"] != local_context:
                    raise ValueError("Saved TCN model context mismatch")
                model = TemporalCNN(saved["input_dim"], CONFIG["channels"], CONFIG["dropout"])
                model.load_state_dict(saved["state_dict"])
                model.eval()
                norm = TrainNormalizer.restore(saved["normalizer"])
                threshold = saved["threshold"]
            else:
                remaining_folds = len(fold_data["splits"]) - index
                seconds_left = max(0.0, (stop_at - datetime.now().astimezone()).total_seconds())
                deadline = datetime.now().astimezone().timestamp() + seconds_left / remaining_folds
                model, norm, state, input_dim = fit_fold(d, split, folder, local_context, deadline)
                vi = np.isin(d["groups"], split["validation"])
                pv = probability(model, norm.transform(d["X"][vi], d["observed"][vi]))
                threshold = validation_threshold(d["y"][vi], pv)
                save_torch(model_file, dict(kind="tcn", state_dict=model.state_dict(),
                           normalizer=norm.export(), input_dim=input_dim, config=CONFIG,
                           context=local_context, threshold=threshold))
                write_json(folder / "selection.json", dict(best_epoch=state["best_epoch"],
                           epochs=state["epoch"], completed=state["completed"],
                           fold_deadline_timestamp=deadline))
            evaluate(model, norm, d, manifest, split, threshold, folder, local_context)
            print(f"Completed TCN fold {split['fold']}", flush=True)
        print(summarize(output, len(fold_data["splits"]), "tcn"), flush=True)
    finally:
        release_awake()


if __name__ == "__main__":
    main()
