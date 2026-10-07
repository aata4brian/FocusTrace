import threading,time,platform
from pathlib import Path
import cv2
import numpy as np
from .storage import CSVStream
class Camera:
    def __init__(self,cfg,synthetic=False):
        self.cfg=cfg;self.synthetic=synthetic;self.frame=None;self.jpeg=None;self.error=None;self.callback=None;self.stop_event=threading.Event();self.thread=None;self.cap=None;self.fps=0.;self.frames=0;self.last_frame_at=0.
    def start(self):
        if not self.synthetic:
            self.cap=cv2.VideoCapture(self.cfg['index'],cv2.CAP_DSHOW if platform.system()=='Windows' else cv2.CAP_ANY)
            if not self.cap.isOpened():self.error='Kamera tidak tersedia. Tutup Zoom/Teams, periksa izin kamera dan camera.index.';return
            for key,prop in [('width',cv2.CAP_PROP_FRAME_WIDTH),('height',cv2.CAP_PROP_FRAME_HEIGHT),('fps',cv2.CAP_PROP_FPS)]:self.cap.set(prop,self.cfg[key])
        self.thread=threading.Thread(target=self._loop,daemon=True,name='camera');self.thread.start()
    def _loop(self):
        started=time.monotonic()
        while not self.stop_event.is_set():
            tick=time.monotonic()
            if self.synthetic:
                frame=np.full((self.cfg['height'],self.cfg['width'],3),35,np.uint8);cv2.putText(frame,'SYNTHETIC - NOT RESEARCH DATA',(15,40),cv2.FONT_HERSHEY_SIMPLEX,.6,(230,230,230),1);cv2.circle(frame,(60+int(self.frames*3)%max(1,self.cfg['width']-120),180),30,(90,155,100),-1);ok=True
            else:ok,frame=self.cap.read()
            if not ok:self.error='Frame kamera terputus. Periksa kabel/izin lalu abort sesi.';self.stop_event.wait(.1);continue
            self.frame=frame;self.frames+=1;self.last_frame_at=time.monotonic();self.fps=self.frames/max(self.last_frame_at-started,.001)
            ok,jpg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,75])
            if ok:self.jpeg=jpg.tobytes()
            if self.callback:
                try:self.callback(frame,self.last_frame_at)
                except Exception as exc:self.error=f'Perekaman gagal ({type(exc).__name__}). Abort sesi dan periksa disk.'
            self.stop_event.wait(max(0,1/self.cfg['fps']-(time.monotonic()-tick)))
    @property
    def ready(self):return self.frame is not None and time.monotonic()-self.last_frame_at<2 and self.error is None
    def close(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=5)
        if self.cap:self.cap.release()
class Recorder:
    def __init__(self,path,shape,fps,codec,sidecar):
        self.path=Path(path);self.fps=fps;self.count=0;self.last_t=None;self.max_gap=0.;self.first_t=None
        if self.path.exists() or Path(sidecar).exists():raise ValueError('Video sudah ada; penimpaan ditolak.')
        self.writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*codec),fps,(shape[1],shape[0]))
        if not self.writer.isOpened():raise ValueError('VideoWriter gagal. Periksa codec mp4v, izin folder, dan ruang disk.')
        self.sidecar=CSVStream(sidecar,['frame_index','timestamp_sec','encoded_time_sec'])
    def write(self,frame,t):
        self.writer.write(frame);self.sidecar.write(dict(frame_index=self.count,timestamp_sec=t,encoded_time_sec=self.count/self.fps))
        if self.last_t is not None:self.max_gap=max(self.max_gap,t-self.last_t)
        if self.first_t is None:self.first_t=t
        self.last_t=t;self.count+=1
    def close(self):
        self.writer.release();self.sidecar.close();cap=cv2.VideoCapture(str(self.path));ok,_=cap.read();reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        return dict(readable=bool(ok),frames_written=self.count,frames_reported=reported,fps_declared=self.fps,first_frame_sec=self.first_t,last_frame_sec=self.last_t,max_gap_sec=self.max_gap,time_mapping='frame_timestamps.csv defines canonical session time, not MP4 playback time.')
def probe_writer(folder,frame,cfg):
    path=Path(folder)/'.writer_probe.mp4';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*cfg['recording']['codec']),cfg['camera']['fps'],(frame.shape[1],frame.shape[0]))
    try:
        if not writer.isOpened():return False
        for _ in range(3):writer.write(frame)
    finally:writer.release()
    cap=cv2.VideoCapture(str(path));ok,_=cap.read();cap.release();path.unlink(missing_ok=True);return bool(ok)
