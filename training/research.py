"""Research extension CLI. Run from the existing TraceFokus project root."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backend'))


def main():
    from tracefokus.research.common import read_json, load_dataset
    from tracefokus.research.data import init_decisions, prepare
    from tracefokus.research.splits import create_folds
    from tracefokus.research.runner import run, tune, evaluate_only
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('decisions', 'prepare'):
        p = sub.add_parser(name)
        p.add_argument('--config', type=Path, default=ROOT/'configs/research_18.json')
        p.add_argument('--audit', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        if name == 'prepare':
            p.add_argument('--project', type=Path, default=ROOT)
            p.add_argument('--decisions', type=Path, required=True)
    for name in ('sanity', 'folds', 'run', 'tune', 'evaluate'):
        p = sub.add_parser(name)
        p.add_argument('--dataset', type=Path, required=True)
        if name != 'sanity':
            p.add_argument('--output', type=Path, required=True)
        if name in ('run', 'tune', 'evaluate'):
            p.add_argument('--folds', type=Path, required=True)
        if name in ('run', 'tune'):
            p.add_argument('--resume', action='store_true')
            p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
        if name == 'run':
            p.add_argument('--kind', choices=['dummy', 'rf', 'gru'], required=True)
        if name == 'tune':
            p.add_argument('--baseline', type=Path, required=True)
        if name == 'evaluate':
            p.add_argument('--run', type=Path, required=True)
    a = parser.parse_args()
    try:
        if a.command == 'decisions':
            init_decisions(a.audit, read_json(a.config), a.output)
            result = dict(status='PENDING', decisions=str(a.output))
        elif a.command == 'prepare':
            result = prepare(a.project, a.audit, read_json(a.config), a.decisions, a.output)
        elif a.command == 'sanity':
            d, manifest, _ = load_dataset(a.dataset)
            result = dict(status='PASS', shape=list(d['X'].shape), participants=sorted(set(d['groups'])), windows=len(manifest))
        elif a.command == 'folds':
            result = create_folds(a.dataset, a.output)
        elif a.command == 'run':
            result = run(a.dataset, a.folds, a.output, a.kind, a.resume, a.device)
        elif a.command == 'tune':
            result = tune(a.dataset, a.folds, a.output, a.baseline, a.resume, a.device)
        else:
            result = evaluate_only(a.dataset, a.folds, a.run, a.output)
    except (ValueError, OSError, ImportError) as exc:
        print('Research pipeline stopped: '+str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
