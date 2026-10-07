import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
import cv2
from tracefokus.config import load_config
from tracefokus.domain import validate_id
from tracefokus.vision import Vision,draw_overlay

def main():
    p=argparse.ArgumentParser(description='Offline annotation; preserves raw file');p.add_argument('--participant',required=True);p.add_argument('--dev',action='store_true');a=p.parse_args();validate_id(a.participant);cfg=load_config();base=cfg['paths']['data']/('development' if a.dev else '');out=cfg['paths']['outputs']/('development' if a.dev else '')/'annotated_video';out.mkdir(parents=True,exist_ok=True)
    dest=out/f'{a.participant}_annotated.mp4'
    if dest.exists():print('Output already exists. Rename it before rendering again.');return 1
    raw=base/'raw_video'/f'{a.participant}_raw.mp4';sidecar=base/'raw_video'/f'{a.participant}_frame_timestamps.csv';meta_path=cfg['paths']['outputs']/('development' if a.dev else '')/'sessions'/f'{a.participant}_metadata.json'
    cap=cv2.VideoCapture(str(raw));writer=None;vision=None
    try:
        if not cap.isOpened():raise ValueError('Raw video unreadable.')
        rows=list(csv.DictReader(sidecar.open(encoding='utf-8')));meta=json.loads(meta_path.read_text(encoding='utf-8'));fps=cap.get(cv2.CAP_PROP_FPS);size=(int(cap.get(3)),int(cap.get(4)))
        writer=cv2.VideoWriter(str(dest),cv2.VideoWriter_fourcc(*cfg['recording']['codec']),fps,size)
        if not writer.isOpened():raise ValueError('Cannot open annotation writer.')
        vision=Vision(cfg)
        for record in rows:
            ok,frame=cap.read()
            if not ok:raise ValueError('Raw frame count does not match timestamp sidecar.')
            t=float(record['timestamp_sec']);row=vision.extract(frame,t);view=draw_overlay(frame,vision.overlay,row)
            state=next((i['block_code'] for i in meta['intervals'] if i['start']<=t<i['end']),'UNKNOWN')
            cv2.rectangle(view,(0,0),(size[0],36),(25,35,29),-1);cv2.putText(view,f'{a.participant} | {state} | session {t:.2f}s',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,245,235),1);writer.write(view)
        print(dest);return 0
    except (ValueError,OSError) as exc:print(str(exc));return 1
    finally:
        cap.release()
        if writer:writer.release()
        if vision:vision.close()
if __name__=='__main__':raise SystemExit(main())
