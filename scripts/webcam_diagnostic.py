import argparse,json,sys,time,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from tracefokus.config import load_config
from tracefokus.recording import Camera,probe_writer
from tracefokus.vision import Vision

def main():
    p=argparse.ArgumentParser(description='Check camera, frames, codec and MediaPipe validity');p.add_argument('--index',type=int);p.add_argument('--seconds',type=float,default=5);p.add_argument('--synthetic',action='store_true');a=p.parse_args();cfg=load_config()
    if a.index is not None:cfg['camera']['index']=a.index
    cam=Camera(cfg['camera'],a.synthetic);vision=None
    try:
        cam.start();deadline=time.monotonic()+a.seconds
        while time.monotonic()<deadline:time.sleep(.1)
        if not cam.ready:print(cam.error or 'No frames received.');return 1
        with tempfile.TemporaryDirectory() as folder:writer_ok=probe_writer(folder,cam.frame,cfg)
        vision=Vision(cfg);features=vision.extract(cam.frame,0)
        print(json.dumps(dict(synthetic=a.synthetic,resolution=list(cam.frame.shape[:2][::-1]),frames=cam.frames,fps=cam.fps,writer_ok=writer_ok,face_detected=features['face_detected'],pose_detected=features['pose_detected']),indent=2))
        return 0 if writer_ok else 1
    except Exception as exc:print(f'Diagnostic failed: {type(exc).__name__}: {exc}');return 1
    finally:
        cam.close()
        if vision:vision.close()
if __name__=='__main__':raise SystemExit(main())
