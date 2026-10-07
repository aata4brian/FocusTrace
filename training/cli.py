import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from tracefokus.config import load_config
from tracefokus.dataset import prepare,synthetic_dataset,ABLATIONS
from tracefokus.ml import evaluate_dataset
from tracefokus.storage import json_write

def main(default='rf'):
    p=argparse.ArgumentParser(description='TraceFokus grouped ML pipeline — no frame-level random split')
    if default=='prepare':
        p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--output',type=Path,default=ROOT/'data/processed/dataset.npz');p.add_argument('--verified-only',action='store_true');p.add_argument('--allow-development',action='store_true')
    elif default=='synthetic':
        p.add_argument('--output',type=Path,default=ROOT/'outputs/synthetic/dataset.npz');p.add_argument('--participants',type=int,default=18);p.add_argument('--seconds-per-block',type=int,default=8)
    else:
        p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        p.add_argument('--model',choices=['rule','rf','gru'],default=default if default in ['rule','rf','gru'] else 'rf')
        p.add_argument('--mode',choices=['fixed','group'],default='group');p.add_argument('--folds',type=int);p.add_argument('--epochs',type=int);p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--ablation',choices=list(ABLATIONS),default='personalized')
    args=p.parse_args();cfg=load_config(ROOT)
    try:
        if default=='prepare':result=prepare(args.data,args.output,cfg,args.verified_only,args.allow_development);print(f'Prepared {len(result["y"])} windows from {len(set(result["groups"]))} participants.');return 0
        if default=='synthetic':
            if args.participants<4:raise ValueError('Use at least four synthetic participants.')
            result=synthetic_dataset(args.output,cfg,args.participants,args.seconds_per_block);print(f'SYNTHETIC — NOT RESEARCH DATA: {len(result["y"])} windows.');return 0
        if args.folds:cfg['evaluation']['folds']=args.folds
        if args.epochs:cfg['GRU']['epochs']=args.epochs
        if default=='ablation':
            results={}
            for name in ABLATIONS:
                report=evaluate_dataset(args.dataset,args.output/name,cfg,args.model,args.mode,name,args.device);results[name]=report['summary']
            json_write(args.output/'ablation_summary.json',results)
        else:
            report=evaluate_dataset(args.dataset,args.output,cfg,args.model,args.mode,args.ablation,args.device);print(json.dumps(report['summary'],indent=2));print(report['warning'])
        print('Results written to',args.output);return 0
    except (ValueError,OSError,ImportError) as exc:print('Cannot run pipeline:',exc,file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
