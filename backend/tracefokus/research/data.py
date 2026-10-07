"""Immutable-source, causal, boundary-aware head/gaze dataset preparation."""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from .common import digest, identity, read_json, write_json, environment, validate_arrays

CONDITIONS = {'A1': 'DTE', 'A2': 'DTE', 'B1': 'TREE', 'B2': 'TREE', 'C1': 'TIE', 'C2': 'TIE'}
# Roll has opposite modes in actual P12/P13 calibration (R < .5).
# Exclude this unreliable channel for the entire cohort, before modeling.
ANGLES = ['head_yaw', 'head_pitch']
GAZE = ['gaze_horizontal', 'gaze_vertical', 'gaze_deviation', 'gaze_away_duration']
CATEGORIES = ['CENTER', 'LEFT', 'RIGHT', 'DOWN', 'UNKNOWN']
FEATURES = ([f'{k}_{trig}' for k in ANGLES for trig in ('sin', 'cos')]
            + [f'{k}_relative_{trig}' for k in ANGLES for trig in ('sin', 'cos')]
            + GAZE + ['gaze_horizontal_relative', 'gaze_vertical_relative']
            + ['head_valid', 'gaze_valid'] + ['gaze_'+k for k in CATEGORIES])


def audit_bundle(audit, config):
    audit = Path(audit)
    files = pd.read_csv(audit/'audit/source_files.csv', keep_default_na=False)
    checkpoint = read_json(audit/'audit/checkpoint_A.json')
    if not checkpoint.get('database_external_snapshot'):
        raise ValueError('A frozen database audit is required; do not use a live database')
    issues = read_json(audit/'audit/audit_issues.json')
    critical = [i for i in issues if i['severity'] in ('ERROR', 'FATAL')
                and (not i['participant_id'] or i['participant_id'] in config['included_ids'])]
    return files, checkpoint, critical, identity(dict(files=files.to_dict('records'), checkpoint=checkpoint, issues=issues))


def check_sources(files):
    for row in files.itertuples():
        p = Path(row.path)
        if not p.is_file() or p.stat().st_size != int(row.size_bytes):
            raise ValueError('Audited source absent/changed: ' + str(p))
        if len(str(row.sha256)) == 64 and digest(p) != row.sha256:
            raise ValueError('Audited source hash changed: ' + str(p))
        if row.sha256 == 'NOT_HASHED_VIDEO' and p.stat().st_mtime_ns != int(row.mtime_ns):
            raise ValueError('Audited video timestamp changed; re-audit: ' + str(p))


def init_decisions(audit, config, output):
    _, _, issues, stamp = audit_bundle(audit, config)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Decision records are never overwritten by initialization')
    write_json(output, dict(audit_fingerprint=stamp, config_fingerprint=identity(config),
        reviewed_by='', review_date='', overall_decision='PENDING',
        protocol_decisions={
            'P01_protocol': {'status': 'PENDING', 'evidence': ''},
            'P20_B2_playback': {'status': 'PENDING', 'evidence': ''},
            'head_gaze_only_and_circular_features': {'status': 'PENDING', 'evidence': ''},
            'hybrid_task_and_review_labels': {'status': 'PENDING', 'evidence': ''}},
        issues=[dict(issue_id=identity(i), issue=i, disposition='PENDING', evidence='') for i in issues]))


def gate(audit, config, decisions):
    files, checkpoint, issues, stamp = audit_bundle(audit, config)
    if decisions.get('audit_fingerprint') != stamp or decisions.get('config_fingerprint') != identity(config):
        raise ValueError('Audit/config changed after decision template; regenerate and review')
    if decisions.get('overall_decision') != 'READY_FOR_HEAD_GAZE_DATASET' or not decisions.get('reviewed_by') or not decisions.get('review_date'):
        raise ValueError('Checkpoint A unresolved. Complete factual audit decisions before final preprocessing')
    for name in ('P01_protocol', 'P20_B2_playback', 'head_gaze_only_and_circular_features', 'hybrid_task_and_review_labels'):
        r = decisions.get('protocol_decisions', {}).get(name, {})
        if r.get('status') != 'RESOLVED' or not r.get('evidence', '').strip():
            raise ValueError('Unresolved protocol decision: ' + name)
    records = {r['issue_id']: r for r in decisions.get('issues', [])}
    for issue in issues:
        r = records.get(identity(issue), {})
        if r.get('disposition') not in ('RESOLVED', 'ACCEPTED_WITH_LIMITATION') or not r.get('evidence', '').strip():
            raise ValueError('Unresolved audit issue: ' + issue['participant_id'] + '/' + issue['code'])
    check_sources(files)
    return files, checkpoint, stamp


def calibration_centers(frame, profile):
    calibration = frame.loc[frame.system_state.eq('CALIBRATION')]
    blocks = frame.loc[frame.system_state.eq('BLOCK')]
    if calibration.empty or blocks.empty or calibration.timestamp_sec.max() >= blocks.timestamp_sec.min():
        raise ValueError('Calibration must exist strictly before all experimental blocks')
    centers = {}
    for key in ANGLES + GAZE[:2]:
        saved = profile['features'][key]
        n = int(saved['n'])
        if not saved['valid'] or n < 50:
            raise ValueError('Invalid head/gaze calibration support: ' + key)
        values = pd.to_numeric(calibration[key], errors='coerce').to_numpy(float)
        values = values[np.isfinite(values)][:n]
        if len(values) != n or not np.isclose(np.median(values), saved['median'], atol=1e-6, rtol=1e-5):
            raise ValueError('Calibration prefix does not reproduce saved profile: ' + key)
        if key in ANGLES:
            vector = np.exp(1j * np.deg2rad(values)).mean()
            if abs(vector) < .5:
                raise ValueError('Diffuse circular calibration requires review: ' + key)
            centers[key] = float(np.rad2deg(np.angle(vector)))
        else:
            centers[key] = float(saved['median'])
    return centers


def numeric(frame, key):
    return pd.to_numeric(frame[key], errors='coerce').to_numpy(float)


def encode(frame, centers):
    out = {}
    flags = {}
    for key in ('head_valid', 'gaze_valid'):
        values = frame[key].replace({'True': 1, 'False': 0, 'true': 1, 'false': 0})
        flag = pd.to_numeric(values, errors='coerce').to_numpy(float)
        if not np.isin(flag, [0, 1]).all():
            raise ValueError('Malformed validity flag: ' + key)
        flags[key] = flag.astype(bool)
    for key in ANGLES:
        raw = numeric(frame, key)
        if np.isinf(raw).any():
            raise ValueError('Infinite head angle requires audit review: '+key)
        raw[~flags['head_valid']] = np.nan
        for suffix, angle in [('', raw), ('_relative', raw-centers[key])]:
            out[key+suffix+'_sin'] = np.sin(np.deg2rad(angle))
            out[key+suffix+'_cos'] = np.cos(np.deg2rad(angle))
    for key in GAZE:
        values = numeric(frame, key)
        values[~flags['gaze_valid']] = np.nan
        out[key] = values
        if key in GAZE[:2]:
            out[key+'_relative'] = values-centers[key]
    out.update({k: v.astype(float) for k, v in flags.items()})
    category = frame.gaze_category.fillna('UNKNOWN').astype(str).to_numpy()
    if not np.isin(category, CATEGORIES).all():
        raise ValueError('Unexpected gaze category')
    for cat in CATEGORIES:
        out['gaze_'+cat] = (category == cat).astype(float)
        out['gaze_'+cat][~flags['gaze_valid']] = np.nan
    values = np.column_stack([out[k] for k in FEATURES])
    if np.isinf(values).any():
        raise ValueError('Infinite source feature requires audit review')
    return values


def prepare_rows(frame, meta, reviews):
    frame = frame.copy().reset_index(drop=True)
    pid, sid = meta['participant_id'], meta['session_id']
    if frame.empty or set(frame.participant_id) != {pid} or set(frame.session_id) != {sid}:
        raise ValueError('Mixed/missing participant or session IDs')
    times = numeric(frame, 'timestamp_sec')
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('Feature timestamps must increase in acquisition order; sorting cannot hide resets')
    frame['timestamp_sec'] = times
    frame['source_csv_line'] = np.arange(len(frame))+2
    block = frame.system_state.eq('BLOCK')
    expected = frame.loc[block, 'block_code'].map(CONDITIONS)
    if expected.isna().any() or not (expected == frame.loc[block, 'conceptual_condition']).all():
        raise ValueError('Raw condition/block mismatch')
    if not (numeric(frame.loc[block], 'binary_label') == expected.eq('TIE').to_numpy(int)).all():
        raise ValueError('Raw binary/condition mismatch')
    actual = frame.loc[block, 'block_code']
    order = actual.loc[actual.ne(actual.shift())].tolist()
    if order != meta['sequence'] or len(order) != 6 or set(order) != set(CONDITIONS):
        raise ValueError('Observed order differs from six-block metadata')
    frame['condition'] = frame.conceptual_condition
    frame['review_id'] = 0
    frame['bound_start'] = np.nan
    frame['bound_end'] = np.nan
    for interval in meta['intervals']:
        if interval['state'] != 'BLOCK':
            continue
        sel = block & frame.block_code.eq(interval['block_code'])
        a, b = float(interval['start']), float(interval['end'])
        if not ((times[sel] >= a) & (times[sel] < b)).all() or not sel.any():
            raise ValueError('Block rows outside recorded interval')
        frame.loc[sel, ['bound_start', 'bound_end']] = [a, b]
    if frame.loc[block, 'bound_end'].isna().any():
        raise ValueError('Missing block interval')
    cuts = []
    for r in reviews.sort_values('id').itertuples():
        a, b = float(r.start_sec), float(r.end_sec)
        label = r.final_verified_label
        if not np.isfinite([a, b]).all() or a >= b or label not in (*CONDITIONS.values(), 'UNKNOWN', 'TRANSITION', 'CALIBRATION'):
            raise ValueError('Malformed label review')
        cuts.extend([a, b])
        sel = block & (times >= a) & (times < b)
        frame.loc[sel, 'condition'] = label
        frame.loc[sel, 'review_id'] = int(r.id)
    # Partition at *all* review endpoints, including changes occurring between samples.
    for cut in sorted(set(cuts)):
        left = block & (times < cut) & (frame.bound_end > cut)
        right = block & (times >= cut) & (frame.bound_start < cut)
        frame.loc[left, 'bound_end'] = cut
        frame.loc[right, 'bound_start'] = cut
    return frame


def build_windows(frame, values, cfg, source_name):
    steps = round(cfg['seconds']*cfg['hz'])
    if steps < 2 or not np.isclose(steps, cfg['seconds']*cfg['hz']):
        raise ValueError('Window length times Hz must be an integer >= 2')
    if cfg['stride_seconds'] <= 0 or cfg['maximum_age_sec'] <= 0 or cfg['maximum_gap_sec'] <= 0 or not 0 < cfg['minimum_valid_fraction'] <= 1:
        raise ValueError('Invalid window thresholds')
    valid = frame.system_state.eq('BLOCK') & frame.condition.isin(['DTE', 'TREE', 'TIE'])
    keys = ['participant_id', 'session_id', 'block_code', 'condition', 'review_id', 'bound_start', 'bound_end']
    boundary = frame[keys].ne(frame[keys].shift()).any(axis=1) | valid.ne(valid.shift()) | frame.timestamp_sec.diff().gt(cfg['maximum_gap_sec'])
    sequences, masks, manifest = [], [], []
    rejected = {'short_segment': 0, 'low_valid_fraction': 0}
    for _, seg in frame.groupby(boundary.cumsum(), sort=False):
        if not valid.loc[seg.index].all():
            continue
        t = seg.timestamp_sec.to_numpy(float)
        v = values[seg.index]
        stop = min(float(seg.bound_end.iloc[0]), float(t[-1]))
        starts = np.arange(t[0], stop-cfg['seconds']+1e-8, cfg['stride_seconds'])
        if not len(starts):
            rejected['short_segment'] += 1
        for start in starts:
            grid = start+np.arange(steps)/cfg['hz']
            indices = np.searchsorted(t, grid, side='right')-1
            fresh = ((indices >= 0) & (t[indices.clip(0)] >= start-1e-9)
                     & (grid-t[indices.clip(0)] <= cfg['maximum_age_sec']+1e-9))
            x = v[indices.clip(0)].copy()
            x[~fresh] = np.nan
            sensor = np.nan_to_num(x[:, [FEATURES.index('head_valid'), FEATURES.index('gaze_valid')]])
            fraction = float((sensor.max(axis=1) > .5).mean())
            if fraction < cfg['minimum_valid_fraction']:
                rejected['low_valid_fraction'] += 1
                continue
            observed = np.isfinite(x)
            x = np.where(observed, x, 0).astype(np.float32)
            row = seg.iloc[0]
            src = seg.iloc[indices[fresh]]
            sequences.append(x)
            masks.append(observed)
            manifest.append(dict(participant_id=row.participant_id, session_id=row.session_id,
                block_code=row.block_code, block_number=int(row.block_number),
                assigned_condition=row.conceptual_condition, condition=row.condition,
                label=int(row.condition == 'TIE'), start_sec=float(start), end_sec=float(start+cfg['seconds']),
                review_id=int(row.review_id), label_source='review' if row.review_id else 'task_assignment',
                source_file=source_name, source_first_line=int(src.source_csv_line.min()),
                source_last_line=int(src.source_csv_line.max()), source_first_sec=float(src.timestamp_sec.min()),
                source_last_sec=float(src.timestamp_sec.max()), sensor_valid_fraction=fraction,
                observed_fraction=float(observed.mean()),
                sequence_sha256=identity(dict(x=x.tolist(), observed=observed.tolist()))))
    return sequences, masks, manifest, rejected


def prepare(project, audit, config, decision_path, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    decisions = read_json(decision_path)
    files, checkpoint, stamp = gate(audit, config, decisions)
    if output.exists() or not output.is_relative_to(project/'outputs'):
        raise ValueError('Choose a NEW directory inside project/outputs; raw and existing results cannot be overwritten')
    ids = config['included_ids']
    if len(ids) < config['folds'] or len(set(ids)) != len(ids) or set(ids) & set(config['excluded']):
        raise ValueError('Cohort duplicate, overlap, or too few participants')
    database = Path(checkpoint['database_source']).resolve()
    if any(Path(str(database)+suffix).exists() for suffix in ('-wal', '-shm')):
        raise ValueError('Frozen database must be standalone without WAL/SHM')
    audited_paths = {str(Path(p).resolve()) for p in files.path}
    con = sqlite3.connect(database.as_uri()+'?mode=ro&immutable=1', uri=True)
    xs, masks, rows, centers_by_id, rejected = [], [], [], {}, {}
    try:
        for pid in ids:
            paths = [project/'data/features'/f'{pid}_features.csv', project/'data/profiles'/f'{pid}_baseline.json', project/'outputs/sessions'/f'{pid}_metadata.json']
            if any(str(p) not in audited_paths for p in paths):
                raise ValueError('Input file missing from audit manifest: ' + pid)
            feature_path, profile_path, meta_path = paths
            meta, profile = read_json(meta_path), read_json(profile_path)
            if meta['participant_id'] != pid:
                raise ValueError('Metadata participant mismatch: ' + pid)
            record = con.execute('SELECT participant_id,state,mode,sequence_id,sequence_json FROM sessions WHERE session_id=?', (meta['session_id'],)).fetchall()
            if len(record) != 1 or record[0][:3] != (pid, 'FINISHED', 'RESEARCH'):
                raise ValueError('Session must uniquely resolve to FINISHED RESEARCH: '+pid)
            import json
            if record[0][3] != meta['sequence_id'] or json.loads(record[0][4]) != meta['sequence']:
                raise ValueError('Database and metadata sequence mismatch')
            reviews = pd.read_sql_query('SELECT * FROM reviews WHERE session_id=? ORDER BY id', con, params=[meta['session_id']])
            frame = prepare_rows(pd.read_csv(feature_path), meta, reviews)
            centers = calibration_centers(frame, profile)
            x, m, r, reject = build_windows(frame, encode(frame, centers), config['window'], str(feature_path.relative_to(project)))
            if not x:
                raise ValueError('No eligible windows; do not silently drop participant '+pid)
            xs.extend(x); masks.extend(m); rows.extend(r)
            centers_by_id[pid], rejected[pid] = centers, reject
    finally:
        con.close()
    check_sources(files)
    manifest = pd.DataFrame(rows)
    manifest.insert(0, 'window_id', np.arange(len(manifest)))
    duplicates = manifest.groupby('sequence_sha256').participant_id.nunique()
    if (duplicates > 1).any():
        raise ValueError('Exact feature sequences shared by different participants; review before splitting')
    d = dict(X=np.stack(xs), observed=np.stack(masks), y=manifest.label.to_numpy(np.int64),
             groups=manifest.participant_id.to_numpy(str), features=np.asarray(FEATURES))
    validate_arrays(d)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'windows.npz', **d)
    manifest.to_csv(output/'manifest.csv', index=False)
    counts = manifest.groupby(['participant_id', 'block_code', 'condition', 'label']).size().reset_index(name='windows')
    counts.to_csv(output/'window_counts.csv', index=False)
    write_json(output/'decisions.json', decisions)
    health = dict(status='PASS', windows=len(rows), timesteps=d['X'].shape[1], features=FEATURES,
        nan_count=0, inf_count=0, participants=sorted(set(d['groups'])),
        classes={str(k): int(v) for k, v in manifest.label.value_counts().items()},
        windows_per_participant={str(k): int(v) for k, v in manifest.participant_id.value_counts().items()},
        groups_overlap_check='must pass again after fold construction',
        missing_handling='finite zero placeholders plus observed mask; fit imputation/scaling only on train')
    write_json(output/'checkpoint_B.json', health)
    write_json(output/'dataset.json', dict(synthetic=False, training_authorized=True,
        audit_fingerprint=stamp, config=config, decisions_sha256=digest(output/'decisions.json'),
        artifact_hashes={n: digest(output/n) for n in ('windows.npz', 'manifest.csv', 'decisions.json')},
        calibration_centers=centers_by_id, rejected=rejected, environment=environment(),
        method='research_head_gaze_v1; causal previous-sample resampling; conservative full duration coverage',
        limitation='Raw video bytes may be size/mtime checked only; header/sample probes are not full decoding'))
    return health
