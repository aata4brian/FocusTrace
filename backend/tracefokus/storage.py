import csv,json,sqlite3
from pathlib import Path
from datetime import datetime,timezone
EVENT_FIELDS='participant_id session_id sequence_id event_index event_type system_state block_number block_code conceptual_condition start_time_sec end_time_sec duration_sec source notes'.split()
RESPONSE_FIELDS='participant_id session_id block_code question_id response response_mode first_interaction_sec last_edit_sec completed'.split()
FEATURE_FIELDS=('participant_id session_id sequence_id timestamp_sec frame_index system_state block_number block_code conceptual_condition binary_label face_detected pose_detected gaze_valid head_valid body_valid head_yaw head_pitch head_roll head_yaw_smooth head_pitch_smooth head_roll_smooth head_angular_speed gaze_horizontal gaze_vertical gaze_deviation gaze_category gaze_away_flag gaze_away_duration shoulder_motion shoulder_angle torso_motion wrist_motion body_motion normalized_velocity motion_variance delta_yaw delta_pitch delta_roll delta_gaze_h delta_gaze_v delta_body_motion').split()
def utcnow():return datetime.now(timezone.utc).isoformat()
def json_write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');temp.replace(path)
class CSVStream:
    def __init__(self,path,fields):
        self.file=Path(path).open('x',newline='',encoding='utf-8');self.writer=csv.DictWriter(self.file,fieldnames=fields,extrasaction='ignore');self.writer.writeheader();self.file.flush();self.rows=0
    def write(self,row):self.writer.writerow(row);self.file.flush();self.rows+=1
    def close(self):self.file.close()
class Database:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as con:
            con.executescript('''PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS sessions(participant_id TEXT PRIMARY KEY,session_id TEXT UNIQUE NOT NULL,sequence_id TEXT NOT NULL,sequence_json TEXT NOT NULL,state TEXT NOT NULL,mode TEXT NOT NULL,started_at TEXT NOT NULL,metadata_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS responses(participant_id TEXT NOT NULL,session_id TEXT NOT NULL,block_code TEXT NOT NULL,question_id TEXT NOT NULL,response TEXT NOT NULL,response_mode TEXT NOT NULL,first_interaction_sec REAL NOT NULL,last_edit_sec REAL NOT NULL,completed INTEGER NOT NULL,PRIMARY KEY(session_id,block_code,question_id));
            CREATE TABLE IF NOT EXISTS reviews(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT NOT NULL,start_sec REAL NOT NULL,end_sec REAL NOT NULL,initial_label TEXT NOT NULL,final_verified_label TEXT NOT NULL,reviewer TEXT NOT NULL,review_timestamp TEXT NOT NULL,reason TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS interactions(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT NOT NULL,role TEXT NOT NULL,block_code TEXT NOT NULL,timestamp_sec REAL NOT NULL,event TEXT NOT NULL,payload_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS connections(client_id TEXT PRIMARY KEY,role TEXT NOT NULL,last_seen_at TEXT NOT NULL);''')
            con.execute("UPDATE sessions SET state='INTERRUPTED' WHERE state IN ('CALIBRATION','BLOCK','TRANSITION')")
    def connect(self):
        con=sqlite3.connect(self.path,timeout=10);con.row_factory=sqlite3.Row;return con
    def query(self,sql,args=()):
        with self.connect() as con:return [dict(r) for r in con.execute(sql,args).fetchall()]
    def execute(self,sql,args=()):
        with self.connect() as con:con.execute(sql,args)
    def response(self,row):
        with self.connect() as con:con.execute('''INSERT INTO responses VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(session_id,block_code,question_id) DO UPDATE SET response=excluded.response,last_edit_sec=excluded.last_edit_sec,completed=excluded.completed''',tuple(row[k] for k in RESPONSE_FIELDS))
    def export_responses(self,sid,path):
        rows=self.query('SELECT * FROM responses WHERE session_id=? ORDER BY first_interaction_sec',(sid,))
        with Path(path).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=RESPONSE_FIELDS);w.writeheader();w.writerows(rows)
