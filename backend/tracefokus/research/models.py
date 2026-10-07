"""Train-only transformations, expanded metrics, resumable one-layer GRU."""
import copy
import random
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score, f1_score)
from ..dataset import aggregate
from ..ml import make_gru
from .common import identity, write_json


class TrainNormalizer:
    def fit(self, x, observed, groups, expected_train):
        if set(groups) != set(expected_train):
            raise ValueError('Normalizer.fit must receive exactly training participants')
        raw = np.where(observed, x, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            median = np.nanmedian(raw.reshape(-1, raw.shape[-1]), axis=0)
        self.empty_features = np.isnan(median).tolist()
        self.median = np.nan_to_num(median, nan=0.)
        filled = np.where(observed, x, self.median)
        self.mean = filled.mean(axis=(0, 1))
        self.std = filled.std(axis=(0, 1))
        self.std[self.std < 1e-6] = 1.
        self.train_ids = sorted({str(g) for g in groups})
        return self

    def transform(self, x, observed):
        filled = np.where(observed, x, self.median)
        result = np.concatenate([(filled-self.mean)/self.std, observed.astype(float)], axis=-1).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite transformed input')
        return result

    def export(self):
        return dict(median=self.median.tolist(), mean=self.mean.tolist(), std=self.std.tolist(),
                    train_ids=self.train_ids, empty_features=self.empty_features)

    @classmethod
    def restore(cls, data):
        result = cls()
        for k in ('median', 'mean', 'std'):
            setattr(result, k, np.asarray(data[k]))
        result.train_ids, result.empty_features = data['train_ids'], data['empty_features']
        return result


def metrics(y, probability, threshold=.5):
    y, probability = np.asarray(y), np.asarray(probability)
    if probability.shape != y.shape or not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError('Invalid probability output shape/range')
    pred = (probability >= threshold).astype(int)
    report = classification_report(y, pred, labels=[0, 1], output_dict=True, zero_division=0)
    return dict(samples=len(y), accuracy=float(accuracy_score(y, pred)),
        macro_f1=float(report['macro avg']['f1-score']), weighted_f1=float(report['weighted avg']['f1-score']),
        per_class={str(k): report[str(k)] for k in (0, 1)},
        confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(), threshold=float(threshold),
        roc_auc=float(roc_auc_score(y, probability)) if len(set(y)) == 2 else None,
        pr_auc_average_precision=float(average_precision_score(y, probability)) if len(set(y)) == 2 else None)


def validation_threshold(y, p):
    candidates = np.linspace(.2, .8, 61)
    scores = [f1_score(y, p >= t, labels=[0, 1], average='macro', zero_division=0) for t in candidates]
    best = max(scores)
    return float(min((t for t, score in zip(candidates, scores) if score == best), key=lambda t: (abs(t-.5), t)))


def predict_gru(model, z, device='cpu', batch_size=512):
    import torch
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            logits = model(torch.as_tensor(z[start:start+batch_size], device=device))
            if logits.shape != (len(z[start:start+batch_size]),):
                raise ValueError('GRU output must be [batch]')
            output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output)


def save_torch(path, payload):
    import torch
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def train_gru(x, observed, y, groups, train_ids, xv, mv, yv, cfg, output, context, seed=42, device='auto', deadline=None):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    if set(y) != {0, 1} or set(yv) != {0, 1}:
        raise ValueError('Train/validation need both classes')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device
    if target.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    norm = TrainNormalizer().fit(x, observed, groups, train_ids)
    z, zv = norm.transform(x, observed), norm.transform(xv, mv)
    model = make_gru(z.shape[-1], cfg['hidden'], cfg['dropout']).to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    ratio = float((y == 0).sum()/(y == 1).sum()) if cfg['class_weight'] else 1.
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio, device=target))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(z), torch.as_tensor(y, dtype=torch.float32)),
        batch_size=cfg['batch_size'], shuffle=True, generator=generator, num_workers=0)
    signature = identity(dict(context=context, config=cfg, seed=seed, normalizer=norm.export(), device=target))
    state = dict(epoch=0, best=-1., stale=0, curve=[], completed=False, elapsed_seconds=0.)
    best_state = None
    last = output/'last.pt'
    if last.exists():
        saved = torch.load(last, map_location=target, weights_only=True)
        if saved['signature'] != signature:
            raise ValueError('Resume mismatch: dataset/split/config/device changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        state, best_state = saved['progress'], saved['best_model']
        torch.set_rng_state(saved['torch_rng'].cpu())
        generator.set_state(saved['loader_rng'].cpu())
        if saved['cuda_rng'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
    started = time.monotonic()
    while not state['completed'] and state['epoch'] < cfg['epochs']:
        if deadline is not None and time.monotonic() >= deadline:
            break
        model.train()
        total = 0.
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx.to(target)), by.to(target))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.item())*len(bx)
        pv = predict_gru(model, zv, target)
        score = float(f1_score(yv, pv >= .5, labels=[0, 1], average='macro', zero_division=0))
        state['epoch'] += 1
        state['curve'].append(dict(epoch=state['epoch'], train_loss=total/len(y), validation_macro_f1_at_05=score))
        if score > state['best']+1e-6:
            state['best'], state['stale'] = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_torch(output/'best.pt', dict(state_dict=best_state, normalizer=norm.export(), config=cfg,
                input_dim=z.shape[-1], context=context, epoch=state['epoch'], signature=signature))
        else:
            state['stale'] += 1
        state['completed'] = state['stale'] >= cfg['patience'] or state['epoch'] >= cfg['epochs']
        now = time.monotonic()
        state['elapsed_seconds'] += now-started
        started = now
        save_torch(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), best_model=best_state,
            normalizer=norm.export(), progress=state, signature=signature, context=context,
            torch_rng=torch.get_rng_state(), loader_rng=generator.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
        write_json(output/'history.json', state)
        print(f"epoch {state['epoch']}: validation macro F1={score:.4f}", flush=True)
    if best_state is None:
        return None, state
    model.load_state_dict(best_state)
    model.cpu().eval()
    artifact = dict(kind='gru', model=model, normalizer=norm, config=cfg, input_dim=z.shape[-1])
    return artifact, state


def fit_baseline(kind, x, mask, y, groups, ids, cfg, seed):
    norm = TrainNormalizer().fit(x, mask, groups, ids)
    if kind == 'dummy':
        return dict(kind=kind, probability=float(np.mean(y)), normalizer=norm)
    if kind != 'rf':
        raise ValueError('Unknown baseline kind')
    # Reuse the existing source's temporal aggregation, after train-only imputation.
    z = norm.transform(x, mask)
    model = RandomForestClassifier(**cfg, random_state=seed)
    model.fit(aggregate(z), y)
    return dict(kind=kind, model=model, normalizer=norm)


def predict(artifact, x, mask):
    if artifact['kind'] == 'dummy':
        return np.full(len(x), artifact['probability'])
    z = artifact['normalizer'].transform(x, mask)
    if artifact['kind'] == 'rf':
        return artifact['model'].predict_proba(aggregate(z))[:, 1]
    return predict_gru(artifact['model'], z)


def save_model(artifact, path, context, threshold):
    path = Path(path)
    if artifact['kind'] == 'gru':
        save_torch(path, dict(kind='gru', state_dict=artifact['model'].state_dict(),
            normalizer=artifact['normalizer'].export(), config=artifact['config'], input_dim=artifact['input_dim'],
            context=context, threshold=float(threshold)))
    else:
        joblib.dump(dict(artifact=artifact, context=context, threshold=float(threshold)), path)


def load_model(path):
    path = Path(path)
    if path.suffix == '.pt':
        import torch
        payload = torch.load(path, map_location='cpu', weights_only=True)
        model = make_gru(payload['input_dim'], payload['config']['hidden'], payload['config']['dropout'])
        model.load_state_dict(payload['state_dict']); model.eval()
        artifact = dict(kind='gru', model=model, normalizer=TrainNormalizer.restore(payload['normalizer']),
                        config=payload['config'], input_dim=payload['input_dim'])
    else:
        payload = joblib.load(path)
        artifact = payload['artifact']
    return artifact, payload['context'], payload['threshold']
