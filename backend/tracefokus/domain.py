from dataclasses import dataclass,field
import re,time
CODES=('A1','A2','B1','B2','C1','C2')
NAMES=dict(zip(CODES,['DTE Text','DTE Video','TREE Paper','TREE External Video','TIE Scrolling','TIE Conversation']))
PRESETS=dict(zip([f'P{i:02}' for i in range(1,19)],[s.split() for s in [
'A1 B1 C1 A2 B2 C2','A2 B2 C2 A1 B1 C1','A1 B2 C1 B1 C2 A2','A2 B1 C2 C1 A1 B2',
'B1 C1 A1 B2 C2 A2','B2 C2 A2 C1 A1 B1','B1 C2 A1 A2 C1 B2','B2 C1 A2 C2 A1 B1',
'C1 A1 B1 C2 A2 B2','C2 A2 B2 A1 B1 C1','C1 A2 B1 B2 C2 A1','C2 A1 B2 B1 A2 C1',
'A1 C1 B2 B1 A2 C2','A2 C2 B1 C1 B2 A1','B1 A2 C1 C2 B2 A1','B2 A1 C2 A2 C1 B1',
'C1 B2 A1 A2 C2 B1','C2 B1 A2 B2 A1 C1']]))
def validate_sequence(seq):
    if len(seq)!=6 or set(seq)!=set(CODES):raise ValueError('Urutan harus memuat A1–C2 tepat satu kali.')
    return list(seq)
def validate_id(value,pattern=r'^[A-Z][A-Z0-9_-]{1,23}$'):
    if not re.fullmatch(pattern,value):raise ValueError('Gunakan ID anonim 2–24 karakter: huruf besar, angka, _ atau -.')
    return value
def labels(code):
    condition={'A':'DTE','B':'TREE','C':'TIE'}.get(code[:1],code) if code in CODES else code
    return condition,1 if condition=='TIE' else 0 if condition in ('DTE','TREE') else None
@dataclass
class Machine:
    sequence:list
    calibration_sec:float=30
    block_sec:float=120
    clock:object=time.monotonic
    state:str='READY'
    index:int=-1
    origin:float|None=None
    state_start:float=0
    revision:int=0
    intervals:list=field(default_factory=list)
    commands:set=field(default_factory=set)
    frozen_at:float|None=None
    def __post_init__(self):self.sequence=validate_sequence(self.sequence)
    def elapsed(self):return self.frozen_at if self.frozen_at is not None else max(0.,self.clock()-self.origin) if self.origin is not None else 0.
    @property
    def code(self):return self.sequence[self.index] if self.state=='BLOCK' else self.state
    def _change(self,state,at):
        if self.state!='READY':
            condition,binary=labels(self.code)
            self.intervals.append(dict(state=self.state,block_number=self.index+1 if self.state=='BLOCK' else 0,block_code=self.code,conceptual_condition=condition,binary_label=binary,start=self.state_start,end=at,duration=at-self.state_start))
        self.state,self.state_start=state,at;self.revision+=1
    def advance(self):
        if self.state in ('CALIBRATION','BLOCK'):
            deadline=self.state_start+(self.calibration_sec if self.state=='CALIBRATION' else self.block_sec)
            if self.elapsed()>=deadline:self._change('TRANSITION',deadline);return True
        return False
    def command(self,action,command_id,revision):
        if command_id in self.commands:return False
        self.advance()
        if revision!=self.revision:raise ValueError('Status berubah. Tunggu sinkronisasi lalu coba lagi.')
        now=self.elapsed()
        if action=='start' and self.state=='READY':self.origin=self.clock();self._change('CALIBRATION',0.)
        elif action=='next' and self.state=='TRANSITION' and self.index<5:self._change('BLOCK',now);self.index+=1
        elif action=='stop' and self.state=='TRANSITION' and self.index==5:self._change('FINISHED',now);self.frozen_at=now
        elif action=='abort' and self.state in ('CALIBRATION','BLOCK','TRANSITION'):self._change('ABORTED',now);self.frozen_at=now
        else:raise ValueError('Perintah tidak diizinkan pada tahap ini.')
        self.commands.add(command_id);return True
    def at(self,t):
        for i in self.intervals:
            if i['start']<=t<i['end']:return i.copy()
        return dict(state=self.state,block_number=self.index+1 if self.state=='BLOCK' else 0,block_code=self.code,conceptual_condition=labels(self.code)[0],binary_label=labels(self.code)[1])
    def snapshot(self):
        duration=self.calibration_sec if self.state=='CALIBRATION' else self.block_sec if self.state=='BLOCK' else 0
        return dict(state=self.state,block_code=self.code,block_number=self.index+1 if self.state=='BLOCK' else 0,next_block=self.sequence[self.index+1] if self.index<5 else None,elapsed=self.elapsed(),remaining=max(0.,self.state_start+duration-self.elapsed()) if duration else None,duration=duration,revision=self.revision,sequence=self.sequence)
