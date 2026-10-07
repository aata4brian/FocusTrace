import csv,hashlib,json,queue,secrets,shutil,threading,time,uuid
from pathlib import Path
import cv2
from .domain import Machine,PRESETS,validate_id,validate_sequence,labels
from .storage import Database,CSVStream,EVENT_FIELDS,FEATURE_FIELDS,json_write,utcnow
from .recording import Camera,Recorder,probe_writer
from .vision import Vision,SyntheticVision,Baseline,draw_overlay
from .content import Content

class Runtime:
    def __init__(self,cfg,dev=False,synthetic=False,clock=time.monotonic):
        self.cfg=cfg;self.dev=dev;self.synthetic=synthetic;self.clock=clock
        if synthetic and not dev:raise ValueError('Kamera sintetis hanya tersedia dalam DEV MODE.')
        self.paths=cfg['paths'].copy()
        if dev:
            self.paths['data']=self.paths['data']/'development'
            self.paths['outputs']=self.paths['outputs']/'development'
        for name in ['raw_video','events','features','responses','profiles','processed','database']:(self.paths['data']/name).mkdir(parents=True,exist_ok=True)
        for name in ['sessions','annotated_video','plots','metrics']:(self.paths['outputs']/name).mkdir(parents=True,exist_ok=True)
        self.db=Database(self.paths['data']/'database/tracefokus.sqlite3');self.content=Content(self.paths['content'],cfg.get('content',{}),strict_media=not (dev or synthetic))
        self.lock=threading.RLock();self.machine=Machine(PRESETS['P01']);self.camera=Camera(cfg['camera'],synthetic)
        self.vision=None;self.vision_error=None;self.latest_features={};self.overlay_jpeg=None
        self.recorder=None;self.events=None;self.features=None;self.clients={};self.auth={}
        self.pin=secrets.token_hex(4);self.pid=None;self.sid=None;self.sequence_id=None;self.meta={};self.integrity=None;self.fault=None
        self.stopping=threading.Event();self.jobs=queue.Queue(maxsize=1);self.threads=[];self.last_feature=-1.;self.dropped_jobs=0;self.prompt_index=0
        self.last_vision_sid=None;self.interval_cursor=0;self.vision_rows=0;self.baseline=Baseline(cfg['calibration']['minimum_valid_samples'])
        self.baseline_result=None;self.disconnects=set();self.demo=None
    def launch(self):
        if self.synthetic:
            self.vision=SyntheticVision(self.cfg)
            self.vision_error=None
        else:
            try:self.vision=Vision(self.cfg)
            except Exception as exc:self.vision_error=f'CV gagal dimuat: {type(exc).__name__}: {exc}'
        for fn,name in [(self._features_loop,'features'),(self._timer_loop,'session-clock')]:
            t=threading.Thread(target=fn,daemon=True,name=name);t.start();self.threads.append(t)
        self.camera.callback=self.on_frame;self.camera.start()
    def close(self):
        with self.lock:
            if self.active:
                try:self.command('abort',str(uuid.uuid4()),self.machine.revision,'SYSTEM')
                except Exception:self.fault='Shutdown tidak selesai. Tinjau sesi terakhir.'
        self.stopping.set();self.camera.close()
        for t in self.threads:t.join(timeout=5)
        if self.vision:self.vision.close()
    @property
    def active(self):return self.machine.state in ('CALIBRATION','BLOCK','TRANSITION')
    def connected(self,role):
        # Connection health is based on heartbeat recency. Browser visibility is
        # tracked separately and must not be misreported as a network disconnect.
        return any(c['role']==role and self.clock()-c['seen']<self.cfg['server']['heartbeat_timeout_sec'] for c in self.clients.values())
    def visible(self,role):
        return any(c['role']==role and c.get('visible',True) and self.clock()-c['seen']<self.cfg['server']['heartbeat_timeout_sec'] for c in self.clients.values())
    def register(self,role,pin=''):
        if role not in ('participant','stimulus','operator'):raise ValueError('Peran tidak dikenal.')
        if role=='operator' and not secrets.compare_digest(pin,self.pin):raise PermissionError('PIN operator salah. Lihat terminal laptop.')
        token=secrets.token_urlsafe(32);self.auth[token]=role;self.clients[token]={'role':role,'seen':self.clock(),'visible':True}
        return token
    def heartbeat(self,token,visible=True):
        role=self.auth[token];self.clients[token]={'role':role,'seen':self.clock(),'visible':visible}
        self.db.execute('INSERT OR REPLACE INTO connections VALUES(?,?,?)',(hashlib.sha256(token.encode()).hexdigest()[:16],role,utcnow()))
    def preflight(self,pid,sequence):
        checks=[]
        def add(name,ok,detail):checks.append(dict(name=name,ok=bool(ok),detail=detail,critical=True))
        try:validate_id(pid,self.cfg['privacy']['anonymous_id_pattern']);validate_sequence(sequence);add('Identitas & urutan',True,'Enam blok unik; ID anonim.')
        except ValueError as exc:add('Identitas & urutan',False,str(exc))
        add('ID belum digunakan',not self.db.query('SELECT 1 FROM sessions WHERE participant_id=?',(pid,)) and not any(self.path(kind,pid).exists() for kind in ('video','events','features','baseline','metadata')),'Gunakan ID baru; satu peserta satu sesi, tanpa penimpaan.')
        self.content.reload();add('Materi lokal',not self.content.errors,'; '.join(self.content.errors) or 'Teks, pertanyaan, feed, prompt, dan dua video terbaca.')
        add('Kamera',self.camera.ready,self.camera.error or 'Kamera mengirim frame.')
        add('Computer vision',self.vision is not None and self.vision_error is None,self.vision_error or 'MediaPipe siap.')
        try:
            free=shutil.disk_usage(self.paths['data']).free/1024**2
            probe=self.paths['data']/'.write_probe';probe.write_text('probe');probe.unlink();self.db.execute('CREATE TABLE IF NOT EXISTS health_check(ok INTEGER)')
            add('Disk & database',free>=self.cfg['recording']['minimum_free_mb'],f'Ruang tersedia {free:.0f} MB.')
            writable=self.camera.ready and probe_writer(self.paths['data'],self.camera.frame,self.cfg)
            add('VideoWriter',writable,'Codec harus berhasil menulis dan membaca kembali frame uji.')
        except Exception as exc:add('Disk / video writer',False,f'{type(exc).__name__}: periksa izin dan ruang penyimpanan.')
        add('Layar peserta',self.connected('participant'),'Buka /participant dan biarkan tab aktif.')
        add('Perangkat stimulus',self.connected('stimulus'),'Buka /stimulus di perangkat eksternal dan biarkan tab aktif.')
        add('Tidak ada sesi aktif',not self.active,'Selesaikan atau abort sesi aktif.')
        return dict(ok=all(x['ok'] for x in checks),checks=checks,mode='SYNTHETIC — NOT RESEARCH DATA' if self.synthetic else 'DEV MODE' if self.dev else 'RESEARCH')
    def path(self,kind,pid=None):
        pid=pid or self.pid
        mapping={'video':('data','raw_video',f'{pid}_raw.mp4'),'frames':('data','raw_video',f'{pid}_frame_timestamps.csv'),
                 'events':('data','events',f'{pid}_events.csv'),'features':('data','features',f'{pid}_features.csv'),
                 'responses':('data','responses',f'{pid}_responses.csv'),'baseline':('data','profiles',f'{pid}_baseline.json'),
                 'metadata':('outputs','sessions',f'{pid}_metadata.json'),'integrity':('outputs','sessions',f'{pid}_integrity_report.json')}
        base,folder,name=mapping[kind];return self.paths[base]/folder/name
    def start(self,pid,sequence,sequence_id,command_id,source='OPERATOR_PHONE'):
        with self.lock:
            if command_id in self.machine.commands:return self.snapshot('operator')
            check=self.preflight(pid,sequence)
            if not check['ok']:raise ValueError('Pre-flight belum lolos. '+ '; '.join(c['name'] for c in check['checks'] if not c['ok']))
            self.pid=pid;self.sid=str(uuid.uuid4());self.sequence_id=sequence_id
            e=self.cfg['experiment'];self.machine=Machine(sequence,e['dev_calibration_sec'] if self.dev else e['calibration_sec'],e['dev_block_sec'] if self.dev else e['block_sec'],clock=self.clock)
            self.interval_cursor=0;self.integrity=None;self.fault=None;self.last_feature=-1.;self.vision_rows=0;self.prompt_index=0;self.disconnects=set()
            self.baseline=Baseline(min(10,self.cfg['calibration']['minimum_valid_samples']) if self.dev else self.cfg['calibration']['minimum_valid_samples']);self.baseline_result=None
            recorder=Recorder(self.path('video'),self.camera.frame.shape,self.cfg['camera']['fps'],self.cfg['recording']['codec'],self.path('frames'))
            try:
                self.events=CSVStream(self.path('events'),EVENT_FIELDS);self.features=CSVStream(self.path('features'),FEATURE_FIELDS)
                self.meta=dict(participant_id=pid,session_id=self.sid,sequence_id=sequence_id,sequence=sequence,mode=check['mode'],started_at=utcnow(),content_hashes=self.content.hashes,calibration_sec=self.machine.calibration_sec,block_sec=self.machine.block_sec,clock='time.monotonic, seconds from START',software_version='2.0.0',consent_confirmed=True)
                self.db.execute('INSERT INTO sessions VALUES(?,?,?,?,?,?,?,?)',(pid,self.sid,sequence_id,json.dumps(sequence),'CALIBRATION',check['mode'],self.meta['started_at'],json.dumps(self.meta)))
                self.machine.command('start',command_id,0);self.recorder=recorder
                self._event('SESSION_START',0.,source);self._save_meta()
            except Exception:
                recorder.close()
                if self.events:self.events.close()
                if self.features:self.features.close()
                raise
            return self.snapshot('operator')
    def _event(self,event,t,source='AUTO',notes='',interval=None):
        if not self.events:return
        i=interval or self.machine.at(t)
        self.events.write(dict(participant_id=self.pid,session_id=self.sid,sequence_id=self.sequence_id,event_index=self.events.rows,event_type=event,system_state=i['state'],block_number=i['block_number'],block_code=i['block_code'],conceptual_condition=i['conceptual_condition'],start_time_sec=i.get('start',t),end_time_sec=i.get('end',t),duration_sec=i.get('duration',0),source=source,notes=notes))
    def _sync_intervals(self):
        while self.interval_cursor<len(self.machine.intervals):
            i=self.machine.intervals[self.interval_cursor];self._event('INTERVAL',i['end'],interval=i);self.interval_cursor+=1
            if i['state']=='CALIBRATION' and self.baseline_result is None:
                self.baseline_result=self.baseline.finalize();json_write(self.path('baseline'),dict(participant_id=self.pid,session_id=self.sid,**self.baseline_result))
        if self.sid:
            self.db.execute('UPDATE sessions SET state=? WHERE session_id=?',(self.machine.state,self.sid));self._save_meta()
    def _save_meta(self):
        json_write(self.path('metadata'),dict(**self.meta,state=self.machine.state,intervals=self.machine.intervals,elapsed=self.machine.elapsed(),fault=self.fault,dropped_feature_jobs=self.dropped_jobs))
    def tick(self):
        with self.lock:
            if self.machine.advance():self._sync_intervals()
            if self.active:
                for role in ('participant','stimulus','operator'):
                    disconnected=not self.connected(role)
                    if disconnected and role not in self.disconnects:
                        self.disconnects.add(role);self.interaction('SYSTEM','device_disconnected',{'role':role})
                    elif not disconnected and role in self.disconnects:
                        self.disconnects.remove(role);self.interaction('SYSTEM','device_reconnected',{'role':role})
                if not self.camera.ready:self.fault=self.camera.error or 'Kamera tidak mengirim frame.'
    def _timer_loop(self):
        while not self.stopping.wait(.04):
            try:self.tick()
            except Exception as exc:self.fault=f'Penyimpanan/status gagal ({type(exc).__name__}). Abort dan periksa disk.'
    def on_frame(self,frame,capture_time):
        with self.lock:
            if self.recorder and self.active:
                t=max(0.,capture_time-self.machine.origin);index=self.recorder.count;self.recorder.write(frame,t)
                if t-self.last_feature>=1/self.cfg['vision']['feature_hz']:
                    job=(frame.copy(),t,index,self.sid);self.last_feature=t
                    try:self.jobs.put_nowait(job)
                    except queue.Full:self.dropped_jobs+=1
            elif self.demo and self.vision:
                t=time.monotonic()-self.demo['origin']
                if t-self.last_feature>=1/self.cfg['vision']['feature_hz']:
                    self.last_feature=t
                    try:self.jobs.put_nowait((frame.copy(),t,0,'DEMO'))
                    except queue.Full:self.dropped_jobs+=1
    def _features_loop(self):
        while not self.stopping.is_set():
            try:frame,t,index,sid=self.jobs.get(timeout=.2)
            except queue.Empty:continue
            try:
                if not self.vision:continue
                if sid!=self.last_vision_sid:self.vision.reset();self.last_vision_sid=sid
                row=self.vision.extract(frame,t)
                ok,jpg=cv2.imencode('.jpg',draw_overlay(frame,self.vision.overlay,row))
                if ok:self.overlay_jpeg=jpg.tobytes()
                with self.lock:
                    if sid=='DEMO' and self.demo:
                        self._demo_row(row,t);continue
                    if sid!=self.sid or not self.recorder:continue
                    phase=self.machine.at(t)
                    if phase['state']=='CALIBRATION' and self.baseline_result is None:self.baseline.add(row)
                    row.update(self.baseline.deviations(row));self.latest_features=row
                    self.features.write(dict(participant_id=self.pid,session_id=self.sid,sequence_id=self.sequence_id,timestamp_sec=t,frame_index=index,system_state=phase['state'],block_number=phase['block_number'],block_code=phase['block_code'],conceptual_condition=phase['conceptual_condition'],binary_label=phase['binary_label'],**row));self.vision_rows+=1
            except Exception as exc:self.vision_error=f'Ekstraksi fitur gagal ({type(exc).__name__}). Periksa diagnostik.';self.fault=self.vision_error
            finally:self.jobs.task_done()
    def command(self,action,command_id,revision,source='OPERATOR_PHONE'):
        with self.lock:
            self.tick()
            if self.machine.command(action,command_id,revision):
                self._sync_intervals()
                if action=='next':
                    self.prompt_index=0
                    if self.machine.code in ('B1','B2'):
                        self.save_response(self.task_token(),'paper','',False,mode='PAPER',internal=True)
                if action in ('stop','abort'):self.finish(source)
            return self.snapshot('operator')
    def finish(self,source):
        self._event('SESSION_STOP',self.machine.elapsed(),source,notes=self.machine.state)
        video=self.recorder.close();self.recorder=None;self.events.close();self.features.close()
        self.db.export_responses(self.sid,self.path('responses'))
        self.meta['video']=video;self.meta['stopped_at']=utcnow();self._save_meta()
        self.integrity=self.check_integrity(video);json_write(self.path('integrity'),self.integrity)
        self.db.execute('UPDATE sessions SET metadata_json=? WHERE session_id=?',(json.dumps(dict(**self.meta,intervals=self.machine.intervals,elapsed=self.machine.elapsed())),self.sid))
    def check_integrity(self,video):
        blocks=[i for i in self.machine.intervals if i['state']=='BLOCK'];rows=list(csv.DictReader(self.path('features').open(encoding='utf-8')))
        events=list(csv.DictReader(self.path('events').open(encoding='utf-8')));frames=list(csv.DictReader(self.path('frames').open(encoding='utf-8')))
        responses=self.db.query('SELECT * FROM responses WHERE session_id=?',(self.sid,))
        issues=[]
        def check(ok,message):
            if not ok:issues.append(message)
        check(self.machine.state=='FINISHED','Session aborted / interrupted')
        check(not self.dev and not self.synthetic,'Development or synthetic session; never research-valid')
        check(video['readable'] and video['frames_written']==video['frames_reported'] and self.path('video').stat().st_size>0,'Raw video missing, unreadable or frame count mismatch')
        check(video['max_gap_sec']<=self.cfg['recording']['maximum_gap_sec'],'Frame gap exceeds configured limit')
        check(video['first_frame_sec'] is not None and video['first_frame_sec']<1,'Late first frame')
        check(video['last_frame_sec'] is not None and self.machine.elapsed()-video['last_frame_sec']<1,'Video ended before session stop')
        check([b['block_code'] for b in blocks]==self.machine.sequence,'Six unique configured blocks not completed in order')
        check(all(abs(b['duration']-self.machine.block_sec)<1e-6 for b in blocks),'Incomplete block duration')
        check(any(i['state']=='CALIBRATION' and abs(i['duration']-self.machine.calibration_sec)<1e-6 for i in self.machine.intervals),'Calibration missing/incomplete')
        baseline_ok=bool(self.path('baseline').exists() and self.baseline_result and self.baseline_result['valid'])
        if not baseline_ok:
            details=[]
            if self.baseline_result:
                minimum=self.baseline_result.get('minimum_valid_samples',self.cfg['calibration']['minimum_valid_samples'])
                details=[f'{name}={item.get("n",0)}/{minimum}' for name,item in self.baseline_result.get('features',{}).items() if not item.get('valid')]
            issues.append('Baseline has insufficient valid samples'+(': '+', '.join(details) if details else ''))
        for name,data,col in [('features',rows,'timestamp_sec'),('frames',frames,'timestamp_sec'),('events',events,'end_time_sec')]:
            times=[float(r[col]) for r in data];check(bool(times) and all(a<=b for a,b in zip(times,times[1:])),f'{name}: missing/nonmonotonic timestamps')
        check(all(r['participant_id']==self.pid and r['session_id']==self.sid for r in rows+events+responses),'ID mismatch')
        for code in ('A1','A2'):
            for q in self.content.data[code]['questions']:check(any(r['block_code']==code and r['question_id']==q['id'] and r['response'].strip() for r in responses),f'{code}/{q["id"]}: answer missing')
        check(bool(rows),'Feature CSV is empty');check(not self.fault,self.fault or 'Runtime error')
        interaction_rows=self.db.query("SELECT * FROM interactions WHERE session_id=? AND event IN ('device_disconnected','device_reconnected') ORDER BY timestamp_sec",(self.sid,))
        open_disconnects={};disconnect_periods=[]
        for item in interaction_rows:
            try:role=json.loads(item.get('payload_json') or '{}').get('role')
            except json.JSONDecodeError:role=None
            if role not in ('participant','stimulus','operator'):continue
            t=float(item['timestamp_sec'])
            if item['event']=='device_disconnected' and role not in open_disconnects:
                open_disconnects[role]=(t,item.get('block_code'))
            elif item['event']=='device_reconnected' and role in open_disconnects:
                start,block=open_disconnects.pop(role);disconnect_periods.append(dict(role=role,start_sec=start,end_sec=t,duration_sec=max(0.,t-start),block_code=block))
        for role,(start,block) in open_disconnects.items():
            disconnect_periods.append(dict(role=role,start_sec=start,end_sec=self.machine.elapsed(),duration_sec=max(0.,self.machine.elapsed()-start),block_code=block))
        grace=float(self.cfg['server'].get('integrity_disconnect_grace_sec',15))
        serious=[d for d in disconnect_periods if d['role'] in ('participant','stimulus') and d['duration_sec']>=grace]
        for d in serious:issues.append(f"{d['role']} disconnected {d['duration_sec']:.1f}s during {d['block_code']}; inspect interaction log")
        check(all(sum(r['block_code']==b['block_code'] for r in rows)>=self.machine.block_sec*self.cfg['vision']['feature_hz']*.5 for b in blocks),'Feature coverage below 50%')
        return dict(participant_id=self.pid,session_id=self.sid,status='NEEDS REVIEW' if issues else 'VALID',issues=issues,video=video,feature_rows=len(rows),event_rows=len(events),response_rows=len(responses),disconnect_periods=disconnect_periods,checked_at=utcnow(),label_warning='Block labels are initial labels. Human verification is still required.')
    def task_token(self,code=None):return hashlib.sha256(f'{self.sid}:{code or self.machine.code}'.encode()).hexdigest()[:24]
    def save_response(self,token,qid,text,completed,mode='TYPED',internal=False):
        with self.lock:
            if not self.active:raise ValueError('Sesi sudah berakhir; jawaban tidak dapat diubah.')
            self.tick();code=next((c for c in self.machine.sequence if self.task_token(c)==token),None)
            recent=[i for i in self.machine.intervals if i['state']=='BLOCK' and i['block_code']==code and self.machine.elapsed()-i['end']<=3]
            if code is None or not(self.machine.state=='BLOCK' and self.machine.code==code or recent):raise ValueError('Tugas sudah ditutup. Draf terakhir yang diakui server tetap tersimpan.')
            if not internal and (code not in ('A1','A2') or qid not in [q['id'] for q in self.content.data[code]['questions']]):raise ValueError('Pertanyaan tidak valid.')
            now=self.machine.elapsed();row=dict(participant_id=self.pid,session_id=self.sid,block_code=code,question_id=qid,response=text,response_mode=mode,first_interaction_sec=now,last_edit_sec=now,completed=int(completed))
            self.db.response(row);self.db.export_responses(self.sid,self.path('responses'));return dict(saved=True,timestamp_sec=now)
    def interaction(self,role,event,payload):
        if not self.sid or not self.active:return
        self.db.execute('INSERT INTO interactions(session_id,role,block_code,timestamp_sec,event,payload_json) VALUES(?,?,?,?,?,?)',(self.sid,role,self.machine.code,self.machine.elapsed(),event,json.dumps(payload,ensure_ascii=False)))
    def snapshot(self,role):
        with self.lock:
            s=self.machine.snapshot();s.update(mode='SYNTHETIC — NOT RESEARCH DATA' if self.synthetic else 'DEV MODE' if self.dev else 'RESEARCH',connection='online')
            if role in ('participant','stimulus'):
                code=self.machine.code;phase='task' if s['state']=='BLOCK' else s['state'].lower()
                result=dict(phase=phase,task_token=self.task_token() if phase=='task' else None,remaining=s['remaining'],mode=s['mode'],content=self.content.calibration_public(role) if phase=='calibration' else self.content.public(code,role),task_kind={'A1':'text','A2':'video','B1':'paper','B2':'external','C1':'feed','C2':'conversation'}.get(code),revision=s['revision'])
                if phase=='task' and role=='participant':
                    result['answers']=self.db.query('SELECT question_id,response,completed FROM responses WHERE session_id=? AND block_code=?',(self.sid,code))
                    if code in ('B1','B2'):result['paper_code']=f'{self.pid}_{code}'
                return result
            s.update(participant_id=self.pid,session_id=self.sid,sequence_id=self.sequence_id,recording=self.recorder is not None,camera_ready=self.camera.ready,camera_error=self.camera.error,fps=round(self.camera.fps,1),feature_rows=self.vision_rows,features=self.latest_features,vision_error=self.vision_error,fault=self.fault,devices={r:self.connected(r) for r in ('participant','stimulus','operator')},integrity=self.integrity,prompt_index=self.prompt_index,prompts=self.content.data.get('C2',{}).get('prompts',[]),content_ready=not self.content.errors,demo=self.demo_snapshot())
            return s
    def _demo_row(self,row,t):
        self.demo['baseline'].add(row) if t<30 else None
        if t>=30 and not self.demo['baseline'].profile:self.demo['profile']=self.demo['baseline'].finalize()
        row.update(self.demo['baseline'].deviations(row));self.latest_features=row
        self.demo['rows'].append(dict(timestamp_sec=t,**row));self.demo['rows']=self.demo['rows'][-100:]
        if t>=30 and t-self.demo['last_inference']>=.5:
            self.demo['last_inference']=t
            from .ml import infer_live
            p=infer_live(self.demo['artifact'],self.demo['rows'],self.cfg)
            self.demo['probability']=p
            if p is not None:self.demo['episodes'].update(t,p)
    def start_demo(self,model_path):
        with self.lock:
            if self.active:raise ValueError('Selesaikan sesi eksperimen sebelum demo.')
            from .ml import load_artifact
            from .episodes import EpisodeDetector
            artifact=load_artifact(model_path)
            if artifact.get('synthetic') or artifact.get('development'):raise ValueError('Model sintetis tidak boleh digunakan untuk prediksi manusia.')
            self.demo=dict(origin=time.monotonic(),baseline=Baseline(self.cfg['calibration']['minimum_valid_samples']),rows=[],artifact=artifact,episodes=EpisodeDetector(self.cfg['inference']),probability=None,last_inference=0)
            self.last_feature=-1
    def stop_demo(self):
        with self.lock:
            if self.demo:
                t=time.monotonic()-self.demo['origin'];self.demo['episodes'].flush(t)
                path=self.paths['outputs']/'sessions'/f'demo_{int(time.time())}_episodes.json';json_write(path,self.demo['episodes'].episodes)
            self.demo=None
    def demo_snapshot(self):
        if not self.demo:return None
        return dict(elapsed=time.monotonic()-self.demo['origin'],probability=self.demo['probability'],episodes=self.demo['episodes'].episodes,active=self.demo['episodes'].active,calibration_remaining=max(0,30-(time.monotonic()-self.demo['origin'])))
