"""K-fold on participant records, never on windows; frozen inner validation."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from .common import digest, identity, load_dataset, write_json


def assert_split(groups, split):
    sets = [set(split[k]) for k in ('train', 'validation', 'test')]
    a, b, c = sets
    if any(not s for s in sets) or a & b or a & c or b & c:
        raise ValueError('Empty split or participant leakage')
    if set.union(*sets) != set(groups):
        raise ValueError('Split must account for every participant exactly once')


def make_splits(ids, config):
    ids = np.asarray(sorted(set(ids)))
    folds, seed = config['folds'], config['seed']
    gender = config.get('gender_by_participant', {})
    if gender and (set(gender) != set(ids) or any(v not in ('male', 'female') for v in gender.values())):
        raise ValueError('Gender mapping must cover exact cohort using male/female, or remain empty')
    if len(ids) - int(np.ceil(len(ids)/folds)) - config['validation_participants'] < 2:
        raise ValueError('Too few participant groups for train/validation/test')
    if gender:
        labels = np.array([gender[p] for p in ids])
        if min(np.unique(labels, return_counts=True)[1]) < folds:
            raise ValueError('Insufficient gender stratum count for requested folds')
        iterator = StratifiedKFold(folds, shuffle=True, random_state=seed).split(ids, labels)
    else:
        iterator = KFold(folds, shuffle=True, random_state=seed).split(ids)
    results = []
    for number, (pool_index, test_index) in enumerate(iterator, 1):
        pool = ids[pool_index]
        stratify = [gender[p] for p in pool] if gender else None
        train, val = train_test_split(pool, test_size=config['validation_participants'], random_state=seed+number, stratify=stratify)
        split = dict(fold=number, train=sorted(train), validation=sorted(val), test=sorted(ids[test_index]))
        assert_split(ids, split)
        results.append(split)
    test_ids = [pid for s in results for pid in s['test']]
    if sorted(test_ids) != sorted(ids.tolist()):
        raise ValueError('Every participant must be tested exactly once')
    return results


def create_folds(dataset, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Fold output already exists; do not overwrite frozen splits')
    d, manifest, meta = load_dataset(dataset)
    config = meta['config']
    splits = make_splits(d['groups'], config)
    balance = []
    for split in splits:
        for role in ('train', 'validation', 'test'):
            part = manifest.loc[manifest.participant_id.isin(split[role])]
            if set(part.label) != {0, 1}:
                raise ValueError('Each split requires both classes: '+str(split['fold'])+'/'+role)
            for pid in split[role]:
                sub = part.loc[part.participant_id.eq(pid)]
                balance.append(dict(fold=split['fold'], role=role, participant_id=pid,
                    gender=config.get('gender_by_participant', {}).get(pid, 'NOT_PROVIDED'),
                    windows=len(sub), class_0=int(sub.label.eq(0).sum()), class_1=int(sub.label.eq(1).sum()),
                    nominal_order=' '.join(sub[['block_number', 'block_code']].drop_duplicates().sort_values('block_number').block_code),
                    effective_conditions=' '.join(sorted(set(sub.condition)))))
    payload = dict(method='KFold over one record per participant; optional gender StratifiedKFold',
        gender_stratified=bool(config.get('gender_by_participant')), seed=config['seed'],
        dataset_fingerprint=digest(Path(dataset)/'dataset.json'), splits=splits,
        group_overlap=0, test_once=True, limitation='Counterbalancing recorded in fold_balance.csv; no claim of exact balance in each fold')
    payload['split_fingerprint'] = identity(payload)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'folds.json', payload)
    pd.DataFrame(balance).to_csv(output/'fold_balance.csv', index=False)
    return payload
