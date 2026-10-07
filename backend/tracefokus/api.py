import asyncio,csv,json,secrets,sqlite3,time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse
import cv2
from fastapi import FastAPI,Request,WebSocket,WebSocketDisconnect,HTTPException,Depends
from fastapi.responses import FileResponse,JSONResponse,Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field,ConfigDict
from .runtime import Runtime
from .domain import PRESETS,NAMES,validate_id
from .storage import utcnow
class StrictModel(BaseModel):model_config=ConfigDict(extra='forbid')
class Pair(StrictModel):role:str;pin:str=''
class Beat(StrictModel):visible:bool=True
class Setup(StrictModel):
    participant_id:str=Field(max_length=24)
    sequence:list[str]=Field(min_length=6,max_length=6)
    sequence_id:str=Field(default='CUSTOM',max_length=32)
class Start(Setup):
    command_id:str=Field(min_length=8,max_length=100)
    consent_confirmed:bool=False
class Command(StrictModel):action:str;command_id:str=Field(min_length=8,max_length=100);revision:int
class Answer(StrictModel):task_token:str;question_id:str;response:str=Field(max_length=20000);completed:bool=False
class Interaction(StrictModel):event:str=Field(max_length=60);payload:dict=Field(default_factory=dict)
class Review(StrictModel):
    start_sec:float=Field(ge=0,allow_inf_nan=False)
    end_sec:float=Field(gt=0,allow_inf_nan=False)
    final_verified_label:str
    reviewer:str=Field(min_length=1,max_length=100)
    reason:str=Field(min_length=1,max_length=2000)
class Demo(StrictModel):model:str=Field(max_length=100)

def create_app(cfg,dev=False,synthetic=False,runtime=None,launch=True):
    rt=runtime or Runtime(cfg,dev,synthetic)
    @asynccontextmanager
    async def lifespan(app):
        if launch:rt.launch()
        yield
        if launch:rt.close()
    app=FastAPI(title='TraceFokus local research API',version='2.0.0',lifespan=lifespan,docs_url=None,redoc_url=None)
    app.state.runtime=rt
    @app.exception_handler(ValueError)
    async def bad_input(request,exc):return JSONResponse({'detail':str(exc)},status_code=409)
    @app.exception_handler(PermissionError)
    async def forbidden(request,exc):return JSONResponse({'detail':str(exc)},status_code=403)
    @app.exception_handler(sqlite3.Error)
    async def db_error(request,exc):
        rt.fault='Database gagal ditulis. Hentikan pengumpulan dan periksa disk/izin.'
        return JSONResponse({'detail':rt.fault},status_code=503)
    @app.exception_handler(OSError)
    async def io_error(request,exc):
        return JSONResponse({'detail':'Berkas tidak tersedia atau disk tidak dapat ditulis. Periksa folder proyek dan izin.'},status_code=503)
    @app.middleware('http')
    async def headers(request,call_next):
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff';response.headers['Referrer-Policy']='no-referrer';response.headers['Cache-Control']='no-store'
        response.headers['Content-Security-Policy']="default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self' ws: wss:; frame-ancestors 'none'"
        return response
    def identity(request:Request):
        token=request.headers.get('Authorization','').removeprefix('Bearer ') or request.query_params.get('token','')
        if token not in rt.auth:raise HTTPException(401,'Hubungkan ulang perangkat; token sesi server sudah berakhir.')
        return token,rt.auth[token]
    def operator(auth=Depends(identity)):
        if auth[1]!='operator':raise HTTPException(403,'Akses peneliti diperlukan.')
        return auth
    @app.get('/api/health')
    def health():return {'ok':True,'version':'2.0.0'}
    @app.post('/api/pair')
    def pair(body:Pair,request:Request):
        # Limit brute-force attempts per client on this private LAN service.
        host=request.client.host;now=time.monotonic();attempts=getattr(app.state,'attempts',{})
        attempts[host]=[t for t in attempts.get(host,[]) if now-t<60]
        if body.role=='operator':
            if len(attempts[host])>=15:raise HTTPException(429,'Terlalu banyak percobaan; tunggu satu menit.')
            attempts[host].append(now);app.state.attempts=attempts
        return {'token':rt.register(body.role,body.pin),'role':body.role}
    @app.post('/api/heartbeat')
    def heartbeat(body:Beat,auth=Depends(identity)):rt.heartbeat(auth[0],body.visible);return {'ok':True}
    @app.get('/api/state')
    def state(auth=Depends(identity)):return rt.snapshot(auth[1])
    @app.websocket('/ws')
    async def sync(ws:WebSocket):
        token=ws.query_params.get('token','');origin=ws.headers.get('origin')
        if token not in rt.auth or origin and urlparse(origin).netloc!=ws.headers.get('host'):
            await ws.close(code=1008);return
        await ws.accept()
        try:
            while True:
                await ws.send_json(rt.snapshot(rt.auth[token]));await asyncio.sleep(.25)
        except (WebSocketDisconnect,RuntimeError):return
    @app.get('/api/setup')
    def setup(auth=Depends(operator)):return dict(presets=PRESETS,names=NAMES,content=rt.content.data)
    @app.post('/api/preflight')
    def preflight(body:Setup,auth=Depends(operator)):
        with rt.lock:return rt.preflight(body.participant_id,body.sequence)
    @app.post('/api/start')
    def start(body:Start,auth=Depends(operator)):
        if not body.consent_confirmed:raise ValueError('Konfirmasi persetujuan peserta sebelum START.')
        rt.stop_demo();return rt.start(body.participant_id,body.sequence,body.sequence_id,body.command_id)
    @app.post('/api/command')
    def command(body:Command,auth=Depends(operator)):return rt.command(body.action,body.command_id,body.revision)
    @app.post('/api/answer')
    def answer(body:Answer,auth=Depends(identity)):
        if auth[1]!='participant':raise HTTPException(403,'Hanya layar peserta dapat menyimpan jawaban.')
        return rt.save_response(body.task_token,body.question_id,body.response,body.completed)
    @app.post('/api/interaction')
    def interaction(body:Interaction,auth=Depends(identity)):
        allowed={'participant':{'play','pause','ended','media_error'},'stimulus':{'play','pause','ended','scroll','visible_post','like','dwell','media_error'},'operator':{'next_prompt','note'}}
        if body.event not in allowed[auth[1]] or len(json.dumps(body.payload))>8000:raise ValueError('Interaksi tidak valid.')
        with rt.lock:
            rt.interaction(auth[1],body.event,body.payload)
            if body.event=='next_prompt' and rt.machine.code=='C2':rt.prompt_index=min(rt.prompt_index+1,len(rt.content.data['C2']['prompts'])-1)
        return {'ok':True}
    @app.get('/api/preview.jpg')
    def preview(overlay:bool=False,auth=Depends(operator)):
        data=rt.overlay_jpeg if overlay and rt.overlay_jpeg else rt.camera.jpeg
        if not data:raise HTTPException(503,'Kamera belum mengirim gambar.')
        return Response(data,media_type='image/jpeg')
    @app.get('/api/sessions')
    def sessions(auth=Depends(operator)):return rt.db.query('SELECT participant_id,session_id,state,mode,started_at FROM sessions ORDER BY started_at DESC')
    def session(pid):
        validate_id(pid);rows=rt.db.query('SELECT * FROM sessions WHERE participant_id=?',(pid,))
        if not rows:raise HTTPException(404,'Sesi tidak ditemukan.')
        return rows[0]
    @app.get('/api/sessions/{pid}')
    def details(pid:str,auth=Depends(operator)):
        s=session(pid);m=json.loads(rt.path('metadata',pid).read_text(encoding='utf-8')) if rt.path('metadata',pid).exists() else json.loads(s['metadata_json'])
        reviews=rt.db.query('SELECT * FROM reviews WHERE session_id=? ORDER BY id',(s['session_id'],))
        responses=rt.db.query('SELECT * FROM responses WHERE session_id=?',(s['session_id'],))
        integrity=json.loads(rt.path('integrity',pid).read_text(encoding='utf-8')) if rt.path('integrity',pid).exists() else {'status':'NEEDS REVIEW','issues':['Sesi belum ditutup atau terputus.']}
        return dict(session=s,metadata=m,reviews=reviews,responses=responses,integrity=integrity)
    @app.get('/api/sessions/{pid}/video')
    def raw_video(pid:str,auth=Depends(operator)):
        session(pid);return FileResponse(rt.path('video',pid),media_type='video/mp4',filename=f'{pid}_raw.mp4')
    @app.get('/api/sessions/{pid}/frame.jpg')
    def frame(pid:str,t:float=0,auth=Depends(operator)):
        session(pid)
        rows=list(csv.DictReader(rt.path('frames',pid).open(encoding='utf-8')))
        if not rows:raise HTTPException(404,'Tidak ada frame.')
        index=min(range(len(rows)),key=lambda i:abs(float(rows[i]['timestamp_sec'])-t))
        cap=cv2.VideoCapture(str(rt.path('video',pid)));cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,im=cap.read();cap.release()
        if not ok:raise HTTPException(404,'Frame tidak dapat dibaca.')
        ok,jpg=cv2.imencode('.jpg',im)
        return Response(jpg.tobytes(),media_type='image/jpeg',headers={'X-Session-Time':rows[index]['timestamp_sec']})
    @app.post('/api/sessions/{pid}/review')
    def review(pid:str,body:Review,auth=Depends(operator)):
        s=session(pid)
        if s['state'] not in ('FINISHED','ABORTED','INTERRUPTED'):raise ValueError('Review hanya setelah sesi ditutup.')
        m=json.loads(rt.path('metadata',pid).read_text(encoding='utf-8'))
        if body.start_sec>=body.end_sec or body.end_sec>m.get('elapsed',0):raise ValueError('Rentang di luar durasi sesi.')
        if body.final_verified_label not in ('DTE','TREE','TIE','UNKNOWN'):raise ValueError('Label final tidak valid.')
        initial=sorted({i['conceptual_condition'] for i in m['intervals'] if i['start']<body.end_sec and i['end']>body.start_sec})
        rt.db.execute('INSERT INTO reviews(session_id,start_sec,end_sec,initial_label,final_verified_label,reviewer,review_timestamp,reason) VALUES(?,?,?,?,?,?,?,?)',(s['session_id'],body.start_sec,body.end_sec,json.dumps(initial),body.final_verified_label,body.reviewer.strip(),utcnow(),body.reason.strip()))
        return {'ok':True}
    @app.get('/api/content')
    def content(auth=Depends(operator)):return dict(content=rt.content.data,errors=rt.content.errors,hashes=rt.content.hashes)
    @app.post('/api/content/reload')
    def reload_content(auth=Depends(operator)):
        if rt.active:raise ValueError('Materi dikunci selama sesi aktif.')
        rt.content.reload();return dict(ok=not rt.content.errors,errors=rt.content.errors)
    @app.get('/api/models')
    def models(auth=Depends(operator)):return [p.name for p in cfg['paths']['models'].glob('*.joblib')]+[p.name for p in cfg['paths']['models'].glob('*.pt')]
    @app.post('/api/demo/start')
    def demo_start(body:Demo,auth=Depends(operator)):
        if Path(body.model).name!=body.model:raise ValueError('Nama model tidak valid.')
        rt.start_demo(cfg['paths']['models']/body.model);return {'ok':True}
    @app.post('/api/demo/stop')
    def demo_stop(auth=Depends(operator)):rt.stop_demo();return {'ok':True}
    # Only browser-safe media is served; experiment manifests and private data have separate access checks.
    @app.get('/media/{asset:path}')
    def media(asset:str):
        base=cfg['paths']['content'].resolve();path=(base/asset).resolve()
        if not path.is_relative_to(base) or path.suffix.lower() not in ('.mp4','.webm','.png','.jpg','.jpeg','.svg') or not path.is_file():raise HTTPException(404)
        return FileResponse(path)
    dist=cfg['root']/'frontend/dist'
    if (dist/'assets').is_dir():app.mount('/assets',StaticFiles(directory=dist/'assets'),name='assets')
    @app.get('/operator-protocol.js')
    def operator_protocol_script():
        path=cfg['root']/'frontend/public/operator-protocol.js'
        if not path.exists():raise HTTPException(404)
        return FileResponse(path,media_type='application/javascript')
    @app.get('/{route:path}')
    def frontend(route:str):
        if route not in ('','operator','participant','stimulus','monitor','review','content-preview','demo'):raise HTTPException(404)
        if not (dist/'index.html').exists():raise HTTPException(503,'Frontend belum dibangun: jalankan npm.cmd ci dan npm.cmd run build di frontend.')
        return FileResponse(dist/'index.html')
    return app
