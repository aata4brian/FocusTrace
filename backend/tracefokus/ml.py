"""Participant-grouped evaluation. All fitted preprocessing sees training groups only."""
import copy,json,random,warnings
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score,recall_score,f1_score,balanced_accuracy_score,confusion_matrix,average_precision_score,roc_auc_score
from .dataset import aggregate,FEATURES,ABLATIONS,encode_frame,resample_window
from .storage import json_write

def assert_disjoint(train,val,test):
    a,b,c=map(set,(train,val,test))
    if not a or not b or not c or a&b or a&c or b&c:raise ValueError('Participant leakage or empty split detected.')

def split_groups(groups,mode,cfg):
    unique=sorted(set(groups))
    if mode=='fixed':
        train,val,test=[cfg['evaluation'][k] for k in ('training_ids','validation_ids','testing_ids')]
        assert_disjoint(train,val,test)
        missing=(set(train)|set(val)|set(test))-set(unique)
        if missing:raise ValueError('Fixed split missing participants: '+', '.join(sorted(missing)))
        if set(unique)!=(set(train)|set(val)|set(test)):raise ValueError('Fixed split does not account for every participant; edit config explicitly.')
        yield train,val,test
    else:
        folds=cfg['evaluation']['folds']
        if len(unique)<max(folds,4):raise ValueError('Need at least max(folds,4) participant groups for outer test plus inner validation.')
        for outer,testing in GroupKFold(folds).split(np.zeros(len(groups)),groups=groups):
            pool=sorted(set(groups[outer]));rng=np.random.default_rng(cfg['seed']);rng.shuffle(pool)
            n_val=max(1,len(pool)//5);val=pool[:n_val];train=pool[n_val:];test=sorted(set(groups[testing]));assert_disjoint(train,val,test);yield train,val,test

def metrics(y,p,threshold=.5):
    pred=np.asarray(p)>=threshold
    result=dict(precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),pr_auc=None,roc_auc=None,samples=len(y),threshold=float(threshold))
    if len(set(y))==2:result.update(pr_auc=float(average_precision_score(y,p)),roc_auc=float(roc_auc_score(y,p)))
    return result

def threshold_from_validation(y,p):
    candidates=np.linspace(.2,.8,25);scores=[f1_score(y,np.asarray(p)>=t,zero_division=0) for t in candidates]
    best=max(scores);return float(min([t for t,s in zip(candidates,scores) if s==best],key=lambda t:abs(t-.5)))

class Normalizer:
    def fit(self,x):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);self.median=np.nanmedian(x.reshape(-1,x.shape[-1]),axis=0)
        # A train-fold feature with no observations has no estimated center: explicit neutral fallback + mask.
        self.median=np.nan_to_num(self.median,nan=0.)
        filled=np.where(np.isfinite(x),x,self.median);self.mean=filled.mean(axis=(0,1));self.std=filled.std(axis=(0,1));self.std[self.std<1e-6]=1
        return self
    def transform(self,x):
        mask=np.isfinite(x).astype(np.float32);filled=np.where(np.isfinite(x),x,self.median)
        return np.concatenate([(filled-self.mean)/self.std,mask],axis=-1).astype(np.float32)
    def export(self):return {k:getattr(self,k).tolist() for k in ('median','mean','std')}
    @classmethod
    def restore(cls,d):
        obj=cls()
        for k,v in d.items():setattr(obj,k,np.asarray(v))
        return obj

def make_gru(input_dim,hidden=32,dropout=.2):
    import torch
    class GRU(torch.nn.Module):
        def __init__(self):
            super().__init__();self.gru=torch.nn.GRU(input_dim,hidden,batch_first=True);self.head=torch.nn.Sequential(torch.nn.Dropout(dropout),torch.nn.Linear(hidden,16),torch.nn.ReLU(),torch.nn.Linear(16,1))
        def forward(self,x):
            _,h=self.gru(x);return self.head(h[-1]).squeeze(-1)
    return GRU()

def train_gru(x,y,xv,yv,cfg,device='auto'):
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()));torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    target='cuda' if device=='auto' and torch.cuda.is_available() else 'cpu' if device=='auto' else device
    if target=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable.')
    g=cfg['GRU'];norm=Normalizer().fit(x);tx=norm.transform(x);tv=norm.transform(xv);model=make_gru(tx.shape[-1],g['hidden'],g['dropout']).to(target)
    positives=int(y.sum());negatives=len(y)-positives
    if not positives or not negatives:raise ValueError('Training groups must contain both binary classes.')
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives/positives,device=target));optimizer=torch.optim.Adam(model.parameters(),lr=g['learning_rate'])
    generator=torch.Generator().manual_seed(seed);loader=DataLoader(TensorDataset(torch.tensor(tx),torch.tensor(y,dtype=torch.float32)),batch_size=g['batch_size'],shuffle=True,generator=generator)
    vx=torch.tensor(tv,device=target);vy=torch.tensor(yv,dtype=torch.float32,device=target);best=float('inf');best_state=None;patience=0;curve=[]
    for epoch in range(g['epochs']):
        model.train();total=0.
        for bx,by in loader:
            optimizer.zero_grad();loss=criterion(model(bx.to(target)),by.to(target));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=loss.item()*len(bx)
        model.eval()
        with torch.no_grad():val_loss=criterion(model(vx),vy).item()
        curve.append(dict(epoch=epoch+1,train_loss=total/len(y),validation_loss=val_loss))
        if val_loss<best-1e-6:best=val_loss;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};patience=0
        else:patience+=1
        if patience>=g['patience']:break
    model.load_state_dict(best_state);model.cpu().eval()
    return dict(kind='gru',model=model,normalizer=norm,hidden=g['hidden'],dropout=g['dropout'],input_dim=tx.shape[-1],curve=curve,device_used=target)

def fit(kind,x,y,xv,yv,names,cfg,device='auto'):
    if len(set(y))<2:raise ValueError('Training labels require both classes.')
    if kind=='rf':
        model=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('rf',RandomForestClassifier(**cfg['RF'],class_weight='balanced',random_state=cfg['seed'],n_jobs=-1))]);model.fit(aggregate(x),y)
        artifact=dict(kind='rf',model=model)
    elif kind=='gru':artifact=train_gru(x,y,xv,yv,cfg,device)
    elif kind=='rule':artifact=dict(kind='rule',config=cfg['head_pose'])
    else:raise ValueError('Unknown model')
    artifact['features']=list(names);artifact['windowing']=cfg['windowing'];artifact['threshold']=.5
    if kind!='rule':artifact['threshold']=threshold_from_validation(yv,predict(artifact,xv))
    return artifact

def predict(artifact,x):
    if artifact['kind']=='rf':return artifact['model'].predict_proba(aggregate(x))[:,1]
    if artifact['kind']=='gru':
        import torch
        artifact['model'].eval()
        with torch.no_grad():return torch.sigmoid(artifact['model'](torch.tensor(artifact['normalizer'].transform(x)))).cpu().numpy()
    names=artifact['features']
    def col(k):return x[:,:,names.index(k)] if k in names else np.full(x.shape[:2],np.nan)
    yaw=col('delta_yaw') if 'delta_yaw' in names else col('head_yaw');gh=col('gaze_horizontal')
    # Deliberately conservative engineering baseline: head AND gaze must deviate together.
    valid=np.isfinite(yaw)&np.isfinite(gh);flags=(abs(yaw)>artifact['config']['away_yaw_deg'])&(abs(gh)>.15)&valid
    return np.divide(flags.sum(1),valid.sum(1),out=np.full(len(x),.5),where=valid.sum(1)>0)

def save_artifact(artifact,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if artifact['kind']=='gru':
        import torch
        payload={k:v for k,v in artifact.items() if k not in ('model','normalizer')};payload['normalizer']=artifact['normalizer'].export();payload['state_dict']=artifact['model'].state_dict();torch.save(payload,path)
    else:joblib.dump(artifact,path)

def load_artifact(path):
    path=Path(path)
    if not path.is_file():raise ValueError('Model tidak ditemukan. Jalankan pelatihan dahulu.')
    if path.suffix=='.pt':
        import torch
        artifact=torch.load(path,map_location='cpu',weights_only=True);model=make_gru(artifact['input_dim'],artifact['hidden'],artifact['dropout']);model.load_state_dict(artifact.pop('state_dict'));model.eval();artifact['model']=model;artifact['normalizer']=Normalizer.restore(artifact['normalizer']);return artifact
    return joblib.load(path)

def evaluate_dataset(dataset,output,cfg,kind='rf',mode='group',ablation='personalized',device='auto'):
    dataset=Path(dataset);d=np.load(dataset,allow_pickle=False);provenance=json.loads(dataset.with_suffix('.json').read_text(encoding='utf-8'))
    names=ABLATIONS[ablation];indices=[list(d['features']).index(k) for k in names];x=d['X'][:,:,indices];y=d['y'];groups=d['groups'];output=Path(output);output.mkdir(parents=True,exist_ok=True);folds=[]
    for n,(train,val,test) in enumerate(split_groups(groups,mode,cfg),1):
        ti=np.isin(groups,train);vi=np.isin(groups,val);si=np.isin(groups,test)
        artifact=fit(kind,x[ti],y[ti],x[vi],y[vi],names,cfg,device)
        artifact.update(synthetic=bool(provenance.get('synthetic')),development=bool(provenance.get('development')),seed=cfg['seed'],train_participants=train,validation_participants=val,test_participants=test,ablation=ablation)
        suffix='.pt' if kind=='gru' else '.joblib';save_artifact(artifact,output/f'fold_{n}{suffix}')
        prob=predict(artifact,x[si]);report=metrics(y[si],prob,artifact['threshold']);report.update(fold=n,train_participants=train,validation_participants=val,test_participants=test,validation_metrics=metrics(y[vi],predict(artifact,x[vi]),artifact['threshold']))
        folds.append(report);json_write(output/f'fold_{n}_metrics.json',report)
        import pandas as pd
        pd.DataFrame({'participant_id':groups[si],'block_code':d['blocks'][si],'timestamp_sec':d['starts'][si],'label':y[si],'probability':prob}).to_csv(output/f'fold_{n}_predictions.csv',index=False)
        if kind=='rf':
            stats=[s+'__'+f for s in ('mean','std','p10','p90','change','observed_fraction') for f in names]
            imputer=artifact['model']['imputer'];features=imputer.get_feature_names_out(stats);importance=artifact['model']['rf'].feature_importances_
            pd.DataFrame({'feature':features,'importance':importance}).sort_values('importance',ascending=False).to_csv(output/f'fold_{n}_importance.csv',index=False)
        if kind=='gru':pd.DataFrame(artifact['curve']).to_csv(output/f'fold_{n}_training_curve.csv',index=False)
    summary={}
    for key in ('precision','recall','f1','balanced_accuracy','pr_auc','roc_auc'):
        vals=[f[key] for f in folds if f[key] is not None];summary[key]=dict(mean=float(np.mean(vals)),std=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.) if vals else None
    result=dict(kind=kind,mode=mode,ablation=ablation,synthetic=provenance.get('synthetic',False),development=provenance.get('development',False),warning='SYNTHETIC — NOT RESEARCH DATA' if provenance.get('synthetic') else 'Participant-grouped results; validate protocol and labels before scientific interpretation.',folds=folds,summary=summary,provenance=provenance)
    json_write(output/'summary.json',result);plot_results(result,output);return result

def plot_results(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    keys=['precision','recall','f1','balanced_accuracy'];axes[0].bar(keys,[result['summary'][k]['mean'] for k in keys],color='#436950');axes[0].set_ylim(0,1);axes[0].tick_params(axis='x',rotation=20)
    matrix=sum(np.array(f['confusion_matrix']) for f in result['folds']);axes[1].imshow(matrix,cmap='Greens')
    for i in range(2):
        for j in range(2):axes[1].text(j,i,str(matrix[i,j]),ha='center',va='center')
    axes[1].set(xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='Reference');fig.suptitle(('SYNTHETIC — NOT RESEARCH DATA | ' if result['synthetic'] else '')+result['kind'].upper());fig.savefig(Path(output)/'evaluation.png',dpi=160);plt.close(fig)
    if result['kind']=='gru':
        import pandas as pd
        fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
        for file in sorted(Path(output).glob('*_training_curve.csv')):
            d=pd.read_csv(file);ax.plot(d.epoch,d.validation_loss,label=file.stem.replace('_training_curve',''))
        ax.set(xlabel='Epoch',ylabel='Validation BCE loss');ax.legend();fig.savefig(Path(output)/'training_curves.png',dpi=160);plt.close(fig)

def infer_live(artifact,rows,cfg):
    import pandas as pd
    if len(rows)<2:return None
    df=pd.DataFrame(rows);times=df.timestamp_sec.to_numpy(float);w=artifact['windowing'];start=times[-1]-w['seconds']+1/w['hz']
    if times[0]>start:return None
    values=encode_frame(df)[artifact['features']].to_numpy(float);x=resample_window(times,values,start,w)
    if x is None:return None
    flags=[i for i,k in enumerate(artifact['features']) if k.endswith('_valid')]
    if flags and (np.nan_to_num(x[:,flags]).max(1)>.5).mean()<w['minimum_valid_fraction']:return None
    return float(predict(artifact,x[None])[0])
