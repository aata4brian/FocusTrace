import {useEffect,useState} from 'react';
import {api,role,token,tokenKey} from './api';
export function useSession(){
 const [state,setState]=useState<any>(null),[online,setOnline]=useState(false),[paired,setPaired]=useState(Boolean(token())),[error,setError]=useState('');
 async function pair(pin=''){try{const r=await api('/pair',{role,pin});sessionStorage.setItem(tokenKey,r.token);setPaired(true);setError('')}catch(e){setError((e as Error).message)}}
 useEffect(()=>{if(!paired&&role!=='operator')void pair()},[paired]);
 useEffect(()=>{
  if(!paired)return;
  let closed=false,ws:WebSocket|undefined,retry:ReturnType<typeof setTimeout>;
  async function heartbeat(){try{await api('/heartbeat',{visible:!document.hidden})}catch(e){if(!token()){setPaired(false);setError((e as Error).message)}}}
  function connect(){if(closed)return;ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws?token=${encodeURIComponent(token())}`);ws.onmessage=e=>{if(!closed){setState(JSON.parse(e.data));setOnline(true)}};ws.onclose=()=>{setOnline(false);if(!closed)retry=setTimeout(connect,1000)};ws.onerror=()=>ws?.close()}
  void heartbeat();connect();const beat=setInterval(heartbeat,2000);document.addEventListener('visibilitychange',heartbeat);
  return()=>{closed=true;clearInterval(beat);clearTimeout(retry);ws?.close();document.removeEventListener('visibilitychange',heartbeat)};
 },[paired]);
 return {state,online,paired,error,pair};
}
