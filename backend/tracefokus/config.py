from pathlib import Path
import os
import yaml
ROOT = Path(__file__).resolve().parents[2]
def load_config(root=ROOT):
    root=Path(root)
    cfg=yaml.safe_load((root/'configs/config.yaml').read_text(encoding='utf-8'))
    cfg['root']=root
    cfg['paths']={k:root/v for k,v in cfg['paths'].items()}
    if base:=os.environ.get('TRACEFOKUS_DATA_ROOT'):
        for key in ('data','outputs','models'):cfg['paths'][key]=Path(base).resolve()/key
    return cfg
