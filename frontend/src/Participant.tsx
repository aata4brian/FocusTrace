import {useEffect,useRef,useState} from 'react';
import {ArrowRight,CheckCircle2,FilePenLine,Monitor} from 'lucide-react';
import {AnswerField,Button,MediaPlayer} from './components';
import {emit} from './api';
function Waiting({title,text,complete=false}:{title:string,text:string,complete?:boolean}){return <div className="waiting"><div className="waiting-symbol">{complete?<CheckCircle2 size={32} strokeWidth={1.3}/>:<Monitor size={32} strokeWidth={1.3}/>}</div><h1>{title}</h1><p>{text}</p>{!complete&&<span className="waiting-line"/>}</div>}
function CalibrationReading({content}:{content:any}){
 const title=content?.title||'Mengapa Langit Berubah Warna?';
 const instruction=content?.instruction||'Bacalah teks berikut secara wajar sampai instruksi berikutnya muncul. Tidak perlu terburu-buru atau harus menyelesaikan seluruh bacaan.';
 const material=content?.material||'';
 return <main className="calibration-reading"><div className="calibration-copy"><span className="eyebrow">BACAAN KALIBRASI</span><h1>{title}</h1><p className="calibration-instruction">{instruction}</p><article className="calibration-article">{material.split('\n\n').filter(Boolean).map((paragraph:string,i:number)=><p key={i}>{paragraph}</p>)}</article></div></main>
}
export function Participant({s}:{s:any}){
 const c=s.content||{};
 if(s.phase==='calibration')return <CalibrationReading content={c}/>;
 if(s.phase==='finished'||s.phase==='aborted')return <Waiting title="Kegiatan selesai" text="Terima kasih. Silakan ikuti arahan peneliti selanjutnya." complete/>;
 if(s.phase!=='task')return <Waiting title="Tunggu sebentar" text="Peneliti sedang menyiapkan kegiatan. Instruksi berikutnya akan muncul di sini."/>;
 if(['feed','conversation'].includes(s.task_kind))return <Waiting title="Ikuti instruksi peneliti." text=""/>;
 return <div className="participant-task" key={s.task_token}><div className="task-heading"><span className="eyebrow">KEGIATAN</span><h1>{c.title}</h1><p>{c.instruction}</p></div>{s.task_kind==='paper'||s.task_kind==='external'?<div className="paper-note"><FilePenLine size={26}/><div><b>Kerjakan pada kertas yang disediakan</b><p>Kode kertas: <strong>{s.paper_code}</strong></p>{s.task_kind==='external'&&<p>Materi ditampilkan pada perangkat di sampingmu.</p>}</div></div>:<div className="task-columns"><div>{c.material&&<article className="reading">{c.material.split('\n\n').map((p:string,i:number)=><p key={i}>{p}</p>)}</article>}{c.video&&<MediaPlayer src={c.video}/>}</div><section className="response-area"><span className="eyebrow">JAWABAN SINGKAT</span>{c.questions?.map((q:any)=><AnswerField key={s.task_token+q.id} question={q} taskToken={s.task_token} initial={s.answers?.find((a:any)=>a.question_id===q.id)}/>)}</section></div>}</div>
}
function Feed({posts}:{posts:any[]}){
 const area=useRef<HTMLDivElement>(null),last=useRef(0),distance=useRef(0),lastEmit=useRef(0),[liked,setLiked]=useState<string[]>([]);
 useEffect(()=>{
  const dwell=new Map<string,number>();const observer=new IntersectionObserver(entries=>{entries.forEach(e=>{const id=(e.target as HTMLElement).dataset.post!;if(e.isIntersecting){dwell.set(id,performance.now());void emit('visible_post',{post:id}).catch(()=>{})}else if(dwell.has(id)){void emit('dwell',{post:id,client_duration_ms:performance.now()-dwell.get(id)!}).catch(()=>{});dwell.delete(id)}})},{threshold:.6});
  area.current?.querySelectorAll('[data-post]').forEach(el=>observer.observe(el));
  return()=>{observer.disconnect();dwell.forEach((start,post)=>void emit('dwell',{post,client_duration_ms:performance.now()-start}).catch(()=>{}))}
 },[]);
 return <div className="feed" ref={area}><header><strong>Jeda</strong><span>Cerita sehari-hari</span></header><div className="feed-scroll" onScroll={e=>{const top=e.currentTarget.scrollTop;distance.current+=Math.abs(top-last.current);last.current=top;if(performance.now()-lastEmit.current>500){lastEmit.current=performance.now();void emit('scroll',{position:top,total_distance:distance.current}).catch(()=>{})}}}>{posts.map(p=><article key={p.id} data-post={p.id}><div className="post-author"><span>{p.author[0]}</span><b>{p.author}</b></div><img src={'/media/'+p.image} alt={'Ilustrasi '+p.theme}/><div className="post-body"><p>{p.caption}</p><button aria-pressed={liked.includes(p.id)} onClick={()=>{setLiked(v=>v.includes(p.id)?v.filter(x=>x!==p.id):[...v,p.id]);void emit('like',{post:p.id}).catch(()=>{})}}>{liked.includes(p.id)?'Disukai':'Suka'}</button></div></article>)}<div className="feed-end">Semua cerita sudah ditampilkan.</div></div></div>
}
export function Stimulus({s}:{s:any}){
 if(s.phase==='task'&&s.task_kind==='feed')return <Feed key={s.task_token} posts={s.content.feed||[]}/>;
 if(s.phase==='task'&&s.task_kind==='external')return <div className="stimulus-video"><span className="eyebrow">MATERI KEGIATAN</span><h1>{s.content.title}</h1><MediaPlayer src={s.content.video}/><p>Perhatikan materi, lalu kerjakan pada kertas yang disediakan.</p></div>;
 return <Waiting title={s.phase==='finished'?'Kegiatan selesai':'Perangkat siap'} text={s.phase==='finished'?'Terima kasih.':'Materi akan muncul sesuai instruksi peneliti.'} complete={s.phase==='finished'}/>;
}
