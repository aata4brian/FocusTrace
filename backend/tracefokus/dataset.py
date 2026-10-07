import json,sqlite3,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from .storage import json_write
GAZE=['gaze_horizontal','gaze_vertical','gaze_deviation','gaze_away_duration','gaze_valid']+[f'gaze_{x}' for x in ['CENTER','LEFT','RIGHT','DOWN','UNKNOWN']]
HEAD=['head_yaw','head_pitch','head_roll','head_yaw_smooth','head_pitch_smooth','head_roll_smooth','head_angular_speed','head_valid']
BODY=['shoulder_motion','shoulder_angle','torso_motion','wrist_motion','body_motion','normalized_velocity','motion_variance','body_valid']
DELTA=['delta_yaw','delta_pitch','delta_roll','delta_gaze_h','delta_gaze_v','delta_body_motion']
ABLATIONS={'gaze':GAZE,'head':HEAD,'body':BODY,'gaze_head':GAZE+HEAD,'multimodal':GAZE+HEAD+BODY,'personalized':GAZE+HEAD+BODY+DELTA}
FEATURES=ABLATIONS['personalized']

def encode_frame(df):
    out=pd.DataFrame(index=df.index)
    for feature in FEATURES:
        if feature.startswith('gaze_') and feature[5:] in ('CENTER','LEFT','RIGHT','DOWN','UNKNOWN'):
            out[feature]=(df.get('gaze_category',pd.Series('UNKNOWN',index=df.index))==feature[5:]).astype(float)
        elif feature in df:
            series=df[feature].replace({'True':1,'False':0,'true':1,'false':0})
            out[feature]=pd.to_numeric(series,errors='coerce')
        else:out[feature]=np.nan
    for columns,flag in [(GAZE[:4]+['delta_gaze_h','delta_gaze_v'],'gaze_valid'),(HEAD[:-1]+DELTA[:3],'head_valid'),(BODY[:-1]+['delta_body_motion'],'body_valid')]:
        out.loc[out[flag].fillna(0)<.5,columns]=np.nan
    return out.replace([np.inf,-np.inf],np.nan)

def aggregate(x):
    """Temporal statistics plus explicit observed fraction, preserving all-missing columns."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        mean=np.nanmean(x,axis=1);std=np.nanstd(x,axis=1);low=np.nanpercentile(x,10,axis=1);high=np.nanpercentile(x,90,axis=1)
        change=x[:,-1]-x[:,0];fraction=np.isfinite(x).mean(axis=1)
    return np.concatenate([mean,std,low,high,change,fraction],axis=1)

def resample_window(times,values,start,cfg):
    grid=start+np.arange(round(cfg['seconds']*cfg['hz']))/cfg['hz']
    right=np.searchsorted(times,grid).clip(0,len(times)-1);left=(right-1).clip(0,len(times)-1)
    nearest=np.where(abs(times[left]-grid)<=abs(times[right]-grid),left,right)
    if np.max(abs(times[nearest]-grid))>cfg['maximum_gap_sec']:return None
    x=values[nearest].copy()
    return x

def windows(df,cfg,verified_only=False):
    seq=[];ys=[];groups=[];blocks=[];starts=[];verified=[]
    for (pid,sid),part in df.groupby(['participant_id','session_id'],sort=True):
        part=part.sort_values('timestamp_sec').copy()
        # Do not bridge transitions, review-label changes, unknown ranges, or camera outages.
        valid=part['system_state'].eq('BLOCK')&part['block_code'].isin(['A1','A2','B1','B2','C1','C2'])&part['final_label'].isin(['DTE','TREE','TIE'])
        if verified_only:valid &= part['verified'].astype(bool)
        boundaries=(part['block_code'].ne(part['block_code'].shift())|part['final_label'].ne(part['final_label'].shift())|valid.ne(valid.shift())|part['timestamp_sec'].diff().gt(cfg['maximum_gap_sec'])).cumsum()
        for _,segment in part.groupby(boundaries):
            if not valid.loc[segment.index].all():continue
            times=segment.timestamp_sec.to_numpy(float);values=encode_frame(segment).to_numpy(float)
            for start in np.arange(times[0],times[-1]-cfg['seconds']+1/cfg['hz']+1e-7,cfg['stride_seconds']):
                x=resample_window(times,values,start,cfg)
                if x is None:continue
                flag_indices=[FEATURES.index(k) for k in ['head_valid','gaze_valid','body_valid']]
                if (np.nan_to_num(x[:,flag_indices]).max(axis=1)>.5).mean()<cfg['minimum_valid_fraction']:continue
                seq.append(x);ys.append(int(segment.final_label.iloc[0]=='TIE'));groups.append(str(pid));blocks.append(str(segment.block_code.iloc[0]));starts.append(float(start));verified.append(bool(segment.verified.all()))
    if not seq:raise ValueError('No eligible temporal windows. Check labels, timestamps, validity flags and window duration.')
    return dict(X=np.asarray(seq,dtype=np.float32),y=np.asarray(ys,dtype=np.int64),groups=np.asarray(groups),blocks=np.asarray(blocks),starts=np.asarray(starts),verified=np.asarray(verified),features=np.asarray(FEATURES))

def prepare(data_root,output,cfg,verified_only=False,allow_development=False):
    data_root=Path(data_root);files=sorted((data_root/'features').glob('*_features.csv'))
    if not files:raise ValueError('No feature CSV files found.')
    if 'development' in data_root.parts and not allow_development:raise ValueError('Development data rejected. Use engineering-only --allow-development explicitly.')
    frames=[];source=[];database=data_root/'database/tracefokus.sqlite3'
    con=sqlite3.connect(database) if database.exists() else None
    try:
        for file in files:
            frame=pd.read_csv(file)
            if frame.empty:continue
            frame['final_label']=frame['conceptual_condition'];frame['verified']=False
            sid=str(frame.session_id.iloc[0])
            if con:
                state=con.execute('SELECT state,mode FROM sessions WHERE session_id=?',(sid,)).fetchone()
                if not state or state[0]!='FINISHED':continue
                if state[1]!='RESEARCH' and not allow_development:raise ValueError(f'{file.name}: not research data')
                for start,end,label in con.execute('SELECT start_sec,end_sec,final_verified_label FROM reviews WHERE session_id=? ORDER BY id',(sid,)):
                    mask=(frame.timestamp_sec>=start)&(frame.timestamp_sec<end);frame.loc[mask,'final_label']=label;frame.loc[mask,'verified']=True
            elif not allow_development:raise ValueError('Session database is required for real data provenance and verified labels.')
            frames.append(frame);source.append(str(file))
        if not frames:raise ValueError('No finished eligible sessions.')
        result=windows(pd.concat(frames,ignore_index=True),cfg['windowing'],verified_only)
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**result)
        json_write(output.with_suffix('.json'),dict(synthetic=False,development=allow_development,source_files=source,windowing=cfg['windowing'],verified_only=verified_only,verified_window_fraction=float(result['verified'].mean()),features=FEATURES,participants=sorted(set(result['groups'])),windows=len(result['y'])))
        return result
    finally:
        if con:con.close()

def synthetic_dataset(output,cfg,participants=18,seconds_per_block=8,seed=42):
    """Engineering fixture, not human data or evidence of model accuracy."""
    from .domain import PRESETS,labels
    rng=np.random.default_rng(seed);frames=[]
    for p in range(1,participants+1):
        pid=f'P{p:02}';offset=rng.normal(0,3,3);t=0
        for code in PRESETS.get(pid,PRESETS['P01']):
            condition,y=labels(code)
            for step in range(seconds_per_block*10):
                head=offset+rng.normal(0,2,3)+np.array([y*12,3 if code.startswith('B') else 0,0]);gh=rng.normal(y*.10,.05);gv=rng.normal(.04 if code.startswith('B') else 0,.03);body=abs(rng.normal(.15+y*.08,.05))
                row=dict(participant_id=pid,session_id='SYNTHETIC-'+pid,timestamp_sec=t,frame_index=int(t*10),system_state='BLOCK',block_number=list(PRESETS.get(pid,PRESETS['P01'])).index(code)+1,block_code=code,conceptual_condition=condition,final_label=condition,binary_label=y,verified=True,face_detected=1,pose_detected=1,gaze_valid=1,head_valid=1,body_valid=1,head_yaw=head[0],head_pitch=head[1],head_roll=head[2],head_yaw_smooth=head[0],head_pitch_smooth=head[1],head_roll_smooth=head[2],head_angular_speed=abs(rng.normal(5,2)),gaze_horizontal=gh,gaze_vertical=gv,gaze_deviation=np.hypot(gh,gv),gaze_category='RIGHT' if gh>.15 else 'CENTER',gaze_away_duration=abs(gh)*4,shoulder_motion=body,shoulder_angle=rng.normal(0,3),torso_motion=body*.8,wrist_motion=body*1.2,body_motion=body,normalized_velocity=body,motion_variance=.01,delta_yaw=head[0]-offset[0],delta_pitch=head[1]-offset[1],delta_roll=head[2]-offset[2],delta_gaze_h=gh,delta_gaze_v=gv,delta_body_motion=body-.15)
                if rng.random()<.04:row['gaze_valid']=0;row['gaze_category']='UNKNOWN'
                frames.append(row);t+=.1
            t+=1
    df=pd.DataFrame(frames);result=windows(df,cfg['windowing']);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**result)
    json_write(output.with_suffix('.json'),dict(synthetic=True,warning='SYNTHETIC — NOT RESEARCH DATA',participants=participants,windows=len(result['y']),windowing=cfg['windowing'],features=FEATURES,seed=seed))
    return result
