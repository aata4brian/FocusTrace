import hashlib,json
from pathlib import Path
import cv2


class Content:
    def __init__(self,path,quality=None,strict_media=True):
        self.path=Path(path);self.quality=quality or {};self.strict_media=strict_media
        self.data={};self.errors=[];self.hashes={};self.media_info={};self.reload()

    def _video_info(self,path):
        cap=cv2.VideoCapture(str(path));ok,_=cap.read()
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0);frames=float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0);height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0);cap.release()
        duration=frames/fps if fps>0 and frames>0 else 0.0
        # MP4 handler type "soun" is present in normal files with an audio track.
        # This avoids requiring ffprobe on the Windows collection laptop.
        has_audio=None
        if path.suffix.lower()=='.mp4':
            try:has_audio=b'soun' in path.read_bytes()
            except OSError:has_audio=False
        return dict(readable=bool(ok),duration_sec=duration,width=width,height=height,has_audio=has_audio)

    def _validate_research_media(self,code,entry):
        q=self.quality
        if code in ('A2','B2'):
            media=self.path/entry['video'];info=self._video_info(media);self.media_info[code]=info
            if not info['readable']:raise ValueError(f'{code}: video lokal tidak dapat dibaca')
            minimum=float(q.get('minimum_video_sec',20))
            if info['duration_sec']<minimum:raise ValueError(f'{code}: durasi video hanya {info["duration_sec"]:.1f} detik; minimal {minimum:.0f} detik')
            if q.get('require_video_audio',True) and info['has_audio'] is False:raise ValueError(f'{code}: video lokal tidak memiliki track audio; ganti dengan video final bersuara')
        if code=='C1':
            feed=entry.get('feed',[]);minimum=int(q.get('c1_min_posts',12))
            if len(feed)<minimum:raise ValueError(f'C1: minimal {minimum} posting')
            hashes=set()
            for post in feed:
                rel=post.get('image','');image=self.path/rel
                if not image.is_file():raise ValueError(f'C1: gambar tidak ditemukan: {rel}')
                raw=image.read_bytes();digest=hashlib.sha256(raw).hexdigest()
                if digest in hashes:raise ValueError(f'C1: gambar duplikat terdeteksi: {rel}')
                hashes.add(digest)
                im=cv2.imread(str(image))
                if im is None:raise ValueError(f'C1: gambar tidak dapat dibaca: {rel}')
                h,w=im.shape[:2]
                if w<int(q.get('c1_min_width',600)) or h<int(q.get('c1_min_height',750)):
                    raise ValueError(f'C1: resolusi gambar terlalu kecil: {rel} ({w}x{h})')
                # The bundled placeholders are deliberately near-uniform cards. Real feed
                # photos should have substantially more visual variation.
                if float(im.std())<float(q.get('c1_min_image_std',8.0)):
                    raise ValueError(f'C1: {rel} masih terdeteksi sebagai placeholder/nyaris kosong; ganti dengan foto final')

    def reload(self):
        self.errors=[];self.data={};self.hashes={};self.media_info={}
        try:self.data=json.loads((self.path/'experiment.json').read_text(encoding='utf-8'))
        except (OSError,ValueError) as exc:
            self.errors.append(str(exc));return False
        for code in ('A1','A2','B1','B2','C1','C2'):
            try:
                entry=self.data[code]
                if not entry.get('title'):raise ValueError(f'{code}: judul kosong')
                if code in ('A1','A2') and not entry.get('questions'):raise ValueError(f'{code}: pertanyaan kosong')
                if code in ('A2','B2'):
                    media=self.path/entry['video'];cap=cv2.VideoCapture(str(media));ok,_=cap.read();cap.release()
                    if not ok:raise ValueError(f'{code}: video lokal tidak dapat dibaca')
                if code=='C1':
                    if len(entry.get('feed',[]))<6:raise ValueError('C1: minimal enam posting')
                    for post in entry.get('feed',[]):
                        image=self.path/post.get('image','')
                        if not image.is_file():raise ValueError(f"C1: gambar tidak ditemukan: {post.get('image','')}")
                if code=='C2' and len(entry.get('prompts',[]))<5:raise ValueError('C2: minimal lima prompt')
                if self.strict_media:self._validate_research_media(code,entry)
            except (OSError,ValueError,KeyError) as exc:self.errors.append(str(exc))
        try:
            for path in self.path.rglob('*'):
                if path.is_file():self.hashes[str(path.relative_to(self.path))]=hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:self.errors.append(f'Hash materi gagal: {exc}')
        return not self.errors

    def calibration_public(self,role):
        if role!='participant':return {}
        entry=self.data.get('calibration_screen',{})
        return {k:v for k,v in entry.items() if k in ('title','instruction','material')}

    def public(self,code,role):
        entry=self.data.get(code,{})
        if role=='participant':
            if code in ('C1','C2'):return dict(title='Instruksi',instruction='Ikuti instruksi peneliti.')
            return {k:v for k,v in entry.items() if k in ('title','instruction','material','questions','video','response_mode') and not(code=='B2' and k=='video')}
        if role=='stimulus':return entry if code in ('B2','C1') else {}
        return entry
