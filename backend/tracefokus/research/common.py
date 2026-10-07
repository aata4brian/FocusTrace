"""Provenance, atomic derived outputs, and finite dataset validation."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def environment():
    versions = {}
    for name in ('numpy', 'pandas', 'scikit-learn', 'torch', 'joblib'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions)


def validate_arrays(d):
    x, mask, y, groups = (d[k] for k in ('X', 'observed', 'y', 'groups'))
    if x.ndim != 3 or not all(x.shape) or mask.shape != x.shape:
        raise ValueError('X/observed must have the same nonempty [N,T,F] shape')
    if y.shape != (len(x),) or groups.shape != (len(x),):
        raise ValueError('X, y and participant groups have inconsistent lengths')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('NaN/Inf in finite storage representation')
    if mask.dtype != np.bool_ or not np.isin(y, [0, 1]).all():
        raise ValueError('Observed mask must be boolean and labels binary 0/1')
    if any(not str(g).strip() for g in groups):
        raise ValueError('Empty participant ID')
    if len(d['features']) != x.shape[2] or len(set(d['features'])) != x.shape[2]:
        raise ValueError('Feature names do not match X or are duplicated')
    if (x[~mask] != 0).any():
        raise ValueError('Unobserved storage positions must be neutral zero + false mask')


def load_dataset(folder):
    import pandas as pd
    folder = Path(folder)
    meta = read_json(folder/'dataset.json')
    for name, expected in meta['artifact_hashes'].items():
        if digest(folder/name) != expected:
            raise ValueError('Dataset artifact changed: ' + name)
    with np.load(folder/'windows.npz', allow_pickle=False) as archive:
        d = {k: archive[k] for k in archive.files}
    validate_arrays(d)
    manifest = pd.read_csv(folder/'manifest.csv', keep_default_na=False)
    if len(manifest) != len(d['y']) or not np.array_equal(manifest.participant_id, d['groups']):
        raise ValueError('Manifest and arrays differ')
    if not np.array_equal(manifest.label, d['y']):
        raise ValueError('Manifest and array labels differ')
    if meta.get('training_authorized') is not True or meta.get('synthetic') is not False:
        raise ValueError('Dataset is not authorized human research data')
    return d, manifest, meta
