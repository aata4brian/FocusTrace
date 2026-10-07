export type Role='operator'|'participant'|'stimulus';
export const role:Role=location.pathname==='/participant'?'participant':location.pathname==='/stimulus'?'stimulus':'operator';
export const tokenKey=`tracefokus-${role}`;
export function token(){return sessionStorage.getItem(tokenKey)||''}
export async function api<T=any>(path:string,body?:unknown):Promise<T>{
 const response=await fetch('/api'+path,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json','Authorization':'Bearer '+token()},body:body===undefined?undefined:JSON.stringify(body)});
 if(!response.ok){let message='Server tidak merespons.';try{const r=await response.json();message=typeof r.detail==='string'?r.detail:'Data tidak valid.'}catch{} if(response.status===401)sessionStorage.removeItem(tokenKey);throw new Error(message)}
 return response.json();
}
export const id=()=>globalThis.crypto?.randomUUID?.()||`${Date.now()}-${Math.random().toString(36).slice(2)}`;
export function privateURL(path:string){return `/api${path}${path.includes('?')?'&':'?'}token=${encodeURIComponent(token())}`}
export function emit(event:string,payload:Record<string,unknown>={}){return api('/interaction',{event,payload})}
