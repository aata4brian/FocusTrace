"""Approximate gaze/head/body features. No identity or internal-state inference."""
from collections import deque
import math
import numpy as np
import cv2
BASE_FIELDS={'head_yaw':'delta_yaw','head_pitch':'delta_pitch','head_roll':'delta_roll','gaze_horizontal':'delta_gaze_h','gaze_vertical':'delta_gaze_v','body_motion':'delta_body_motion'}
class Baseline:
    def __init__(self,minimum=50):self.rows=[];self.profile={};self.minimum=minimum
    def add(self,row):self.rows.append({k:row.get(k) for k in BASE_FIELDS})
    def finalize(self):
        for key in BASE_FIELDS:
            vals=[r[key] for r in self.rows if r.get(key) is not None and math.isfinite(r[key])]
            med=float(np.median(vals)) if vals else None
            self.profile[key]=dict(median=med,mad=float(np.median(np.abs(np.array(vals)-med))) if vals else None,n=len(vals),valid=len(vals)>=self.minimum)
        return dict(method='median/MAD',minimum_valid_samples=self.minimum,features=self.profile,valid=all(v['valid'] for v in self.profile.values()))
    def deviations(self,row):
        return {delta:float(row[key]-self.profile[key]['median']) if row.get(key) is not None and self.profile.get(key,{}).get('valid') else None for key,delta in BASE_FIELDS.items()}
class SyntheticVision:
    """Deterministic feature generator used only with DEV synthetic camera mode."""
    def __init__(self,cfg):
        self.cfg=cfg
        self.overlay={}
        self.reset()
    def reset(self):
        self.previous_t=None
        self.overlay={}
    def close(self):
        return None
    def extract(self,frame,t):
        # Values vary smoothly so calibration, windowing, overlays and persistence
        # can be exercised end-to-end without pretending to be human CV output.
        yaw=6.0*math.sin(t*0.7)
        pitch=4.0*math.sin(t*0.5+0.4)
        roll=2.0*math.sin(t*0.3)
        gh=0.05*math.sin(t*0.9)
        gv=0.04*math.cos(t*0.8)
        motion=0.15+0.05*abs(math.sin(t*1.1))
        dt=max(t-self.previous_t,1e-6) if self.previous_t is not None else None
        self.previous_t=t
        h,w=frame.shape[:2]
        self.overlay={
            'body': [[w*.42,h*.35],[w*.58,h*.35],[w*.38,h*.48],[w*.62,h*.48],[w*.34,h*.62],[w*.66,h*.62],[w*.45,h*.67],[w*.55,h*.67]],
            'axes': [[w*.5,h*.32],[w*.57,h*.32],[w*.5,h*.39],[w*.47,h*.27]],
        }
        return dict(
            face_detected=True,pose_detected=True,gaze_valid=True,head_valid=True,body_valid=True,
            head_yaw=yaw,head_pitch=pitch,head_roll=roll,head_yaw_smooth=yaw,head_pitch_smooth=pitch,head_roll_smooth=roll,
            head_angular_speed=0.0 if dt is None else abs(4.2*math.cos(t*0.7)),
            gaze_horizontal=gh,gaze_vertical=gv,gaze_deviation=float(math.hypot(gh,gv)),gaze_category='CENTER',gaze_away_flag=0,gaze_away_duration=0.0,
            shoulder_motion=motion*.8,shoulder_angle=0.0,torso_motion=motion*.7,wrist_motion=motion*1.2,body_motion=motion,normalized_velocity=motion,motion_variance=0.0025,
        )

class Vision:
    def __init__(self,cfg):
        import mediapipe as mp
        if not hasattr(mp,'solutions'):raise RuntimeError('Pasang mediapipe 0.10.21 melalui requirements.txt.')
        v=cfg['vision'];self.cfg=cfg
        self.face=mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=v['min_detection_confidence'],min_tracking_confidence=v['min_tracking_confidence'])
        self.pose=mp.solutions.pose.Pose(model_complexity=1,min_detection_confidence=v['min_detection_confidence'],min_tracking_confidence=v['min_tracking_confidence'])
        self.motions=deque(maxlen=cfg['body_motion']['variance_samples']);self.reset();self.overlay={}
    def reset(self):self.previous=None;self.smooth=None;self.away_start=None;self.motions.clear()
    def close(self):self.face.close();self.pose.close()
    def extract(self,frame,t):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);f=self.face.process(rgb);p=self.pose.process(rgb)
        row={k:None for k in 'head_yaw head_pitch head_roll head_yaw_smooth head_pitch_smooth head_roll_smooth head_angular_speed gaze_horizontal gaze_vertical gaze_deviation gaze_away_duration shoulder_motion shoulder_angle torso_motion wrist_motion body_motion normalized_velocity motion_variance'.split()}
        row.update(face_detected=bool(f.multi_face_landmarks),pose_detected=bool(p.pose_landmarks),gaze_valid=False,head_valid=False,body_valid=False,gaze_category='UNKNOWN',gaze_away_flag=None)
        self.overlay={};h,w=frame.shape[:2];body=None
        if f.multi_face_landmarks:
            lm=f.multi_face_landmarks[0].landmark;xy=np.array([(l.x*w,l.y*h) for l in lm],dtype=np.float64);self.overlay['face']=xy.tolist()
            model=np.array([(0,0,0),(0,330,-65),(-225,-170,-135),(225,-170,-135),(-150,150,-125),(150,150,-125)],np.float64)
            matrix=np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],np.float64)
            ok,rvec,tvec=cv2.solvePnP(model,xy[[1,152,33,263,61,291]],matrix,np.zeros((4,1)),flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                rotation,_=cv2.Rodrigues(rvec);pitch,yaw,roll=cv2.RQDecomp3x3(rotation)[0];angles=np.array([yaw,pitch,roll])
                if np.all(np.isfinite(angles)):
                    row.update(head_valid=True,head_yaw=float(yaw),head_pitch=float(pitch),head_roll=float(roll))
                    a=self.cfg['vision']['smoothing_alpha'];self.smooth=angles if self.smooth is None else a*angles+(1-a)*self.smooth
                    row.update(zip(['head_yaw_smooth','head_pitch_smooth','head_roll_smooth'],map(float,self.smooth)))
                    if self.previous and self.previous['angles'] is not None:row['head_angular_speed']=float(np.linalg.norm(angles-self.previous['angles'])/max(t-self.previous['t'],1e-6))
                    axes,_=cv2.projectPoints(np.array([[0,0,0],[100,0,0],[0,100,0],[0,0,100]],float),rvec,tvec,matrix,np.zeros((4,1)));self.overlay['axes']=axes.reshape(-1,2).tolist()
            if len(lm)>=478:
                estimates=[]
                for left,right,top,bottom,iris in [(33,133,159,145,468),(362,263,386,374,473)]:
                    width=np.linalg.norm(xy[right]-xy[left]);height=np.linalg.norm(xy[bottom]-xy[top])
                    if width>3 and height>1.5:
                        gh=np.dot(xy[iris]-(xy[left]+xy[right])/2,(xy[right]-xy[left])/width)/width
                        gv=(xy[iris][1]-(xy[top][1]+xy[bottom][1])/2)/height;estimates.append((gh,gv))
                if len(estimates)==2:
                    gh,gv=np.mean(estimates,axis=0);g=self.cfg['gaze'];category='DOWN' if gv>g['vertical_threshold'] else 'LEFT' if gh < -g['horizontal_threshold'] else 'RIGHT' if gh>g['horizontal_threshold'] else 'CENTER'
                    row.update(gaze_valid=True,gaze_horizontal=float(gh),gaze_vertical=float(gv),gaze_deviation=float(np.hypot(gh,gv)),gaze_category=category,gaze_away_flag=int(category!='CENTER'))
                    if category!='CENTER':
                        if self.away_start is None:self.away_start=t
                        row['gaze_away_duration']=t-self.away_start
                    else:self.away_start=None;row['gaze_away_duration']=0.
        if not row['gaze_valid']:self.away_start=None
        if p.pose_landmarks:
            lm=p.pose_landmarks.landmark;ids=[11,12,13,14,15,16,23,24];self.overlay['body']=[[lm[i].x*w,lm[i].y*h] for i in ids]
            if all(lm[i].visibility>=.5 for i in [11,12,23,24]):
                pts=np.array([[lm[i].x,lm[i].y] for i in ids]);scale=max(float(np.linalg.norm(pts[1]-pts[0])),.01)
                body=dict(shoulder=pts[:2].mean(axis=0),torso=pts[[0,1,6,7]].mean(axis=0),wrist=pts[4:6],scale=scale,wrist_valid=all(lm[i].visibility>=.5 for i in [15,16]))
                row.update(body_valid=True,shoulder_angle=float(np.degrees(np.arctan2(*(pts[1]-pts[0])[::-1]))))
                if self.previous and self.previous['body']:
                    old=self.previous['body'];dt=max(t-self.previous['t'],1e-6)
                    for key,feature in [('shoulder','shoulder_motion'),('torso','torso_motion')]:row[feature]=float(np.linalg.norm(body[key]-old[key])/scale/dt)
                    if body['wrist_valid'] and old['wrist_valid']:row['wrist_motion']=float(np.linalg.norm(body['wrist']-old['wrist'],axis=1).mean()/scale/dt)
                    row['body_motion']=float(np.mean([row[k] for k in ['shoulder_motion','torso_motion','wrist_motion'] if row[k] is not None]));row['normalized_velocity']=row['body_motion'];self.motions.append(row['body_motion']);row['motion_variance']=float(np.var(self.motions))
        self.previous={'t':t,'body':body,'angles':np.array([row['head_yaw'],row['head_pitch'],row['head_roll']]) if row['head_valid'] else None}
        return row

def draw_overlay(frame,overlay,row):
    out=frame.copy()
    for point in overlay.get('face',[]):cv2.circle(out,tuple(map(int,point)),1,(110,190,170),-1)
    body=overlay.get('body',[])
    for a,b in [(0,1),(0,2),(2,4),(1,3),(3,5),(0,6),(1,7),(6,7)]:
        if body:cv2.line(out,tuple(map(int,body[a])),tuple(map(int,body[b])),(220,175,90),2)
    axes=overlay.get('axes',[])
    if axes:
        for i,color in enumerate([(40,60,240),(60,220,60),(240,80,60)],1):cv2.line(out,tuple(map(int,axes[0])),tuple(map(int,axes[i])),color,2)
    if row.get('gaze_valid'):
        origin=(out.shape[1]//2,out.shape[0]//3);endpoint=(int(origin[0]+row['gaze_horizontal']*300),int(origin[1]+row['gaze_vertical']*150));cv2.arrowedLine(out,origin,endpoint,(255,220,0),2)
    return out
