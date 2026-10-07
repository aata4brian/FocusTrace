"""Fold-local training/tuning and checkpoint-only evaluation."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .common import digest, identity, load_dataset, read_json, write_json, environment
from .models import (fit_baseline, train_gru, predict, metrics, validation_threshold,
                     save_model, load_model)
from .splits import assert_split


def inputs(dataset, folds):
    d, manifest, meta = load_dataset(dataset)
    fold_data = read_json(Path(folds)/'folds.json')
    if fold_data['dataset_fingerprint'] != digest(Path(dataset)/'dataset.json'):
        raise ValueError('Fold file belongs to a different dataset')
    verify = {k: v for k, v in fold_data.items() if k != 'split_fingerprint'}
    if identity(verify) != fold_data['split_fingerprint']:
        raise ValueError('Fold file changed after freezing')
    test_ids = []
    for split in fold_data['splits']:
        assert_split(d['groups'], split)
        test_ids.extend(split['test'])
        for role in ('train', 'validation', 'test'):
            if set(d['y'][np.isin(d['groups'], split[role])]) != {0, 1}:
                raise ValueError('Split must contain both classes')
    if sorted(test_ids) != sorted(set(d['groups'])):
        raise ValueError('Each participant must be tested once')
    return d, manifest, meta, fold_data


def run_context(dataset, fold_data, meta, kind):
    source = Path(__file__).parent
    files = list(source.glob('*.py')) + [source.parent/'dataset.py', source.parent/'ml.py']
    return dict(dataset_fingerprint=digest(Path(dataset)/'dataset.json'),
        split_fingerprint=fold_data['split_fingerprint'], kind=kind, config=meta['config'],
        code_hashes={p.name: digest(p) for p in files}, environment=environment())


def reserve(output, context, resume):
    output = Path(output)
    if output.exists():
        if not resume:
            raise FileExistsError('Existing run: use --resume with identical inputs or choose a new run name')
        if read_json(output/'run.json') != context:
            raise ValueError('Resume rejected: inputs/config/code/environment changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output/'run.json', context)
    return output


def evaluate_artifact(artifact, d, manifest, split, threshold, output, context):
    reports, predictions = {}, []
    for role in ('train', 'validation', 'test'):
        ix = np.isin(d['groups'], split[role])
        p = predict(artifact, d['X'][ix], d['observed'][ix])
        reports[role] = metrics(d['y'][ix], p, threshold)
        if role == 'test':
            frame = manifest.loc[ix].copy()
            frame['probability'], frame['prediction'] = p, (p >= threshold).astype(int)
            frame['fold'], frame['threshold'] = split['fold'], threshold
            predictions.append(frame)
    result = dict(fold=split['fold'], split=split, metrics=reports, context=context)
    pd.concat(predictions).to_csv(output/'predictions.csv', index=False)
    write_json(output/'metrics.json', result)
    return result


def summarize(output, fold_count, model_kind):
    output = Path(output)
    files = sorted(output.glob('fold_*/metrics.json'))
    if len(files) != fold_count:
        return dict(status='INCOMPLETE', completed_folds=len(files), expected_folds=fold_count)
    reports = [read_json(p) for p in files]
    predictions = pd.concat([pd.read_csv(p.parent/'predictions.csv') for p in files], ignore_index=True)
    if predictions.window_id.duplicated().any():
        raise ValueError('OOF predictions contain duplicate window IDs')
    predictions.to_csv(output/'oof_predictions.csv', index=False)
    summary = {}
    for key in ('accuracy', 'macro_f1', 'weighted_f1', 'roc_auc', 'pr_auc_average_precision'):
        values = [r['metrics']['test'][key] for r in reports if r['metrics']['test'][key] is not None]
        summary[key] = dict(mean=float(np.mean(values)), std_sample=float(np.std(values, ddof=1)) if len(values) > 1 else None)
    # Stored class decisions use each fold's own validation-selected threshold.
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(predictions.label, predictions.prediction, labels=[0, 1], output_dict=True, zero_division=0)
    pooled = dict(classification_report=report, confusion_matrix=confusion_matrix(predictions.label, predictions.prediction, labels=[0, 1]).tolist())
    errors = []
    predictions['missing_band'] = pd.cut(predictions.observed_fraction, [-.001, .5, .8, .95, 1.001], labels=['>50%', '20-50%', '5-20%', '<5%']).astype(str)
    for field in ('participant_id', 'condition', 'block_code', 'label_source', 'missing_band'):
        for value, rows in predictions.groupby(field):
            pred_report = classification_report(rows.label, rows.prediction, labels=[0, 1], output_dict=True, zero_division=0)
            errors.append(dict(grouping=field, group=value, windows=len(rows),
                accuracy=float((rows.label == rows.prediction).mean()),
                macro_f1=pred_report['macro avg']['f1-score'], weighted_f1=pred_report['weighted avg']['f1-score'],
                errors=int((rows.label != rows.prediction).sum()), classes_present=' '.join(map(str, sorted(set(rows.label))))))
    table = pd.DataFrame(errors).sort_values(['grouping', 'macro_f1'])
    table.to_csv(output/'error_analysis.csv', index=False)
    # Test/validation results do not choose hyperparameters. Suspiciously perfect validation pauses tuning.
    flags = []
    if model_kind != 'dummy' and any(r['metrics']['validation']['macro_f1'] >= .98 for r in reports):
        flags.append('Near-perfect validation: inspect direct-label features/duplicates/protocol cues before tuning')
    overfit = [r['fold'] for r in reports if r['metrics']['train']['macro_f1']-r['metrics']['validation']['macro_f1'] > .2]
    payload = dict(status='REVIEW' if flags else 'PASS_STRUCTURAL_CHECKS',
        fold_metrics=summary, pooled_oof=pooled,
        participant_mean_macro_f1=float(table.loc[table.grouping.eq('participant_id'), 'macro_f1'].mean()),
        suspicious_flags=flags, overfit_warning_folds=overfit, group_overlap=0,
        limitation='Fold SD is descriptive, not a confidence interval. Overlapping windows are not independent participants. Single-class condition macro F1 includes both class slots.')
    write_json(output/'checkpoint_C.json', payload)
    return payload


def run(dataset, folds, output, kind, resume=False, device='auto'):
    d, manifest, meta, fold_data = inputs(dataset, folds)
    context = run_context(dataset, fold_data, meta, kind)
    output = reserve(output, context, resume)
    cfg = meta['config']
    for split in fold_data['splits']:
        folder = output/f"fold_{split['fold']:02}"
        folder.mkdir(exist_ok=True)
        local_context = dict(run=context, split=split)
        suffix = '.pt' if kind == 'gru' else '.joblib'
        artifact_file = folder/('model'+suffix)
        if artifact_file.exists():
            artifact, saved, threshold = load_model(artifact_file)
            if saved != local_context:
                raise ValueError('Saved fold model context mismatch')
        else:
            ti, vi = [np.isin(d['groups'], split[r]) for r in ('train', 'validation')]
            if kind == 'gru':
                artifact, progress = train_gru(d['X'][ti], d['observed'][ti], d['y'][ti], d['groups'][ti], split['train'],
                    d['X'][vi], d['observed'][vi], d['y'][vi], cfg['gru'], folder/'training', local_context,
                    cfg['seed']+split['fold'], device)
                if not progress['completed']:
                    raise RuntimeError('Training paused; rerun with --resume')
            else:
                artifact = fit_baseline(kind, d['X'][ti], d['observed'][ti], d['y'][ti], d['groups'][ti], split['train'], cfg['rf'], cfg['seed']+split['fold'])
            threshold = .5 if kind == 'dummy' else validation_threshold(d['y'][vi], predict(artifact, d['X'][vi], d['observed'][vi]))
            save_model(artifact, artifact_file, local_context, threshold)
        evaluate_artifact(artifact, d, manifest, split, threshold, folder, local_context)
        print(f"Completed {kind}, fold {split['fold']}", flush=True)
    return summarize(output, len(fold_data['splits']), kind)


def evaluate_only(dataset, folds, run_directory, output):
    d, manifest, meta, fold_data = inputs(dataset, folds)
    run_directory, output = Path(run_directory), Path(output)
    if output.exists():
        raise FileExistsError('Evaluation requires a new output folder')
    original = read_json(run_directory/'run.json')
    if original['dataset_fingerprint'] != digest(Path(dataset)/'dataset.json') or original['split_fingerprint'] != fold_data['split_fingerprint']:
        raise ValueError('Evaluation inputs differ from training inputs')
    output.mkdir(parents=True, exist_ok=False)
    for split in fold_data['splits']:
        name = f"fold_{split['fold']:02}"
        folder = output/name
        folder.mkdir()
        suffix = '.pt' if original['kind'] in ('gru', 'tuned_gru') else '.joblib'
        artifact, context, threshold = load_model(run_directory/name/('model'+suffix))
        if context['split'] != split or context['run'] != original:
            raise ValueError('Checkpoint fold provenance mismatch')
        evaluate_artifact(artifact, d, manifest, split, threshold, folder, context)
    return summarize(output, len(fold_data['splits']), original['kind'])


def candidates(config, fold):
    rng = np.random.default_rng(config['seed']+1000+fold)
    result = [dict(config['gru'], epochs=config['tuning']['epochs'], patience=config['tuning']['patience'])]
    while len(result) < config['tuning']['maximum_trials_per_fold']:
        trial = dict(result[0], hidden=int(rng.choice([16, 32, 64])), dropout=float(rng.choice([0., .2, .4])),
            learning_rate=float(10**rng.uniform(-4, -2.7)), batch_size=int(rng.choice([32, 64, 128])),
            weight_decay=float(rng.choice([0., .0001, .001])), class_weight=bool(rng.choice([True, False])))
        result.append(trial)
    return result


def tune(dataset, folds, output, baseline, resume=False, device='auto'):
    d, manifest, meta, fold_data = inputs(dataset, folds)
    baseline = Path(baseline)
    base_context = read_json(baseline/'run.json')
    if base_context['kind'] not in ('rf', 'gru') or base_context['dataset_fingerprint'] != digest(Path(dataset)/'dataset.json') or base_context['split_fingerprint'] != fold_data['split_fingerprint']:
        raise ValueError('Checkpoint C must belong to this dataset/fold baseline')
    c = read_json(baseline/'checkpoint_C.json')
    if c['status'] != 'PASS_STRUCTURAL_CHECKS':
        raise ValueError('Checkpoint C requires investigation before tuning; no automatic bypass')
    context = run_context(dataset, fold_data, meta, 'tuned_gru')
    context['baseline_checkpoint_C_sha256'] = digest(baseline/'checkpoint_C.json')
    output = reserve(output, context, resume)
    cfg = meta['config']
    budget_per_fold = cfg['tuning']['total_hours']*3600/len(fold_data['splits'])
    for split in fold_data['splits']:
        folder = output/f"fold_{split['fold']:02}"
        folder.mkdir(exist_ok=True)
        local_context = dict(run=context, split=split)
        if (folder/'model.pt').exists():
            artifact, saved, threshold = load_model(folder/'model.pt')
            if saved != local_context:
                raise ValueError('Tuned model context mismatch')
            evaluate_artifact(artifact, d, manifest, split, threshold, folder, local_context)
            continue
        configs = candidates(cfg, split['fold'])
        write_json(folder/'trial_plan.json', configs)
        spent = sum(read_json(p)['elapsed_seconds'] for p in folder.glob('trial_*/history.json'))
        deadline = time.monotonic()+max(0., budget_per_fold-spent)
        ti, vi = [np.isin(d['groups'], split[r]) for r in ('train', 'validation')]
        scores = []
        for number, trial in enumerate(configs):
            trial_folder = folder/f'trial_{number:02}'
            trial_folder.mkdir(exist_ok=True)
            score_file = trial_folder/'score.json'
            if score_file.exists():
                scores.append(read_json(score_file)); continue
            if time.monotonic() >= deadline and not (trial_folder/'last.pt').exists():
                break
            artifact, progress = train_gru(d['X'][ti], d['observed'][ti], d['y'][ti], d['groups'][ti], split['train'],
                d['X'][vi], d['observed'][vi], d['y'][vi], trial, trial_folder,
                dict(**local_context, trial=number), cfg['seed']+split['fold']*100+number, device, deadline)
            if artifact is None:
                break
            pv = predict(artifact, d['X'][vi], d['observed'][vi])
            threshold = validation_threshold(d['y'][vi], pv)
            score = dict(trial=number, validation_macro_f1=metrics(d['y'][vi], pv, threshold)['macro_f1'],
                         threshold=threshold, completed=progress['completed'], epochs=progress['epoch'])
            save_model(artifact, trial_folder/'candidate.pt', dict(**local_context, trial=number), threshold)
            write_json(score_file, score)
            scores.append(score)
            if time.monotonic() >= deadline:
                break
        if not scores:
            raise ValueError('No trained candidate within fold budget; benchmark a short GRU before changing the predeclared budget')
        best = max(scores, key=lambda r: (r['validation_macro_f1'], -r['trial']))
        artifact, _, threshold = load_model(folder/f"trial_{best['trial']:02}"/'candidate.pt')
        # Only this fold's validation determines the winner; outer test remains untouched until now.
        save_model(artifact, folder/'model.pt', local_context, threshold)
        write_json(folder/'selection.json', dict(best=best, candidates=scores, budget_seconds=budget_per_fold,
            limitation='Single fixed inner validation split; up to one epoch budget overrun; no final train+validation refit'))
        evaluate_artifact(artifact, d, manifest, split, threshold, folder, local_context)
    return summarize(output, len(fold_data['splits']), 'tuned_gru')
