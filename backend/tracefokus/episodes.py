class EpisodeDetector:
    """Hysteresis with persistence; episodes describe model output, not mental states."""
    def __init__(self,cfg):
        self.cfg=cfg;self.smoothed=None;self.onset=None;self.recovery=None;self.active=False;self.start=None;self.values=[];self.episodes=[];self.last_t=None
    def update(self,t,p):
        if self.last_t is not None and t<=self.last_t:raise ValueError('Episode timestamps must strictly increase.')
        self.last_t=t;a=self.cfg['smoothing_alpha'];self.smoothed=p if self.smoothed is None else a*p+(1-a)*self.smoothed
        if not self.active:
            if self.smoothed>=self.cfg['onset_threshold']:
                if self.onset is None:self.onset=t;self.values=[]
                self.values.append(self.smoothed)
                if t-self.onset>=self.cfg['onset_sec']:self.active=True;self.start=self.onset;self.recovery=None
            else:self.onset=None;self.values=[]
        else:
            self.values.append(self.smoothed)
            if self.smoothed<=self.cfg['recovery_threshold']:
                if self.recovery is None:self.recovery=t
                if t-self.recovery>=self.cfg['recovery_sec']:self._end(self.recovery)
            else:self.recovery=None
        return self.active
    def _end(self,end):
        if self.start is not None and end-self.start>=self.cfg['minimum_episode_sec']:
            e=dict(start=self.start,end=end,duration=end-self.start,mean_probability=sum(self.values)/len(self.values),peak_probability=max(self.values),samples=len(self.values))
            if self.episodes and e['start']-self.episodes[-1]['end']<=self.cfg['merge_gap_sec']:
                old=self.episodes.pop();n=old['samples']+e['samples'];e.update(start=old['start'],duration=end-old['start'],mean_probability=(old['mean_probability']*old['samples']+e['mean_probability']*e['samples'])/n,peak_probability=max(old['peak_probability'],e['peak_probability']),samples=n)
            self.episodes.append(e)
        self.active=False;self.onset=None;self.start=None;self.recovery=None;self.values=[]
    def flush(self,t):
        if self.active:self._end(t)
