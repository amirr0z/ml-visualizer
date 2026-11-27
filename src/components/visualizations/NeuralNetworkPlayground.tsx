"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { RotateCcw, ChevronLeft, ChevronRight, Play, Pause, SkipBack, SkipForward } from "lucide-react";
import type { DataPoint } from "@/lib/datasets/generateDataset";
type Activation = "tanh" | "relu" | "sigmoid" | "linear";
type Network = { weights:number[][][]; biases:number[][] };
type Frame = { network:Network; loss:number; epoch:number };
function act(x:number,t:Activation){if(t==="relu")return Math.max(0,x);if(t==="sigmoid")return 1/(1+Math.exp(-Math.max(-30,Math.min(30,x))));if(t==="linear")return Math.max(-10,Math.min(10,x));return Math.tanh(x);}
function deriv(z:number,a:number,t:Activation){if(t==="relu")return z>0?1:0;if(t==="sigmoid")return a*(1-a);if(t==="linear")return Math.abs(z)<10?1:0;return 1-a*a;}
function init(sizes:number[],seed:number):Network{let s=seed>>>0;const rnd=()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296;};return{weights:sizes.slice(1).map((n,l)=>Array.from({length:n},()=>Array.from({length:sizes[l]},()=>((rnd()*2)-1)*Math.sqrt(2/Math.max(1,sizes[l]))))),biases:sizes.slice(1).map(n=>Array(n).fill(0))};}
function forward(net:Network,x:number,y:number,t:Activation){const values:number[][]=[[x,y]],zs:number[][]=[];for(let l=0;l<net.weights.length;l++){const z=net.weights[l].map((row,j)=>row.reduce((s,w,i)=>s+w*values[l][i],net.biases[l][j]));zs.push(z);values.push(z.map(v=>l===net.weights.length-1?act(v,"sigmoid"):act(v,t)));}return{values,zs,out:values[values.length-1][0]};}
function trainOne(net:Network,data:{x:number;y:number;target:number}[],t:Activation,rate:number){
 const next:Network={weights:net.weights.map(l=>l.map(r=>[...r])),biases:net.biases.map(l=>[...l])};
 const gw=next.weights.map(l=>l.map(r=>r.map(()=>0))),gb=next.biases.map(l=>l.map(()=>0));let loss=0;
 for(const p of data){const f=forward(next,p.x,p.y,t),o=Math.max(1e-7,Math.min(1-1e-7,f.out));loss-=p.target*Math.log(o)+(1-p.target)*Math.log(1-o);let d=[o-p.target];
  for(let l=next.weights.length-1;l>=0;l--){for(let j=0;j<d.length;j++){gb[l][j]+=d[j];for(let i=0;i<next.weights[l][j].length;i++)gw[l][j][i]+=d[j]*f.values[l][i];}
   if(l>0){const prev=Array(next.weights[l][0].length).fill(0);for(let i=0;i<prev.length;i++){let sum=0;for(let j=0;j<d.length;j++)sum+=next.weights[l][j][i]*d[j];prev[i]=sum*deriv(f.zs[l-1][i],f.values[l][i],t);}d=prev;}
  }
 }
 const scale=Math.min(Math.max(rate,0),1)/Math.max(1,data.length);
 next.weights.forEach((l,a)=>l.forEach((r,b)=>r.forEach((_,c)=>{const g=Math.max(-5,Math.min(5,gw[a][b][c]));next.weights[a][b][c]=Math.max(-8,Math.min(8,next.weights[a][b][c]-scale*g));})));
 next.biases.forEach((l,a)=>l.forEach((_,b)=>{const g=Math.max(-5,Math.min(5,gb[a][b]));next.biases[a][b]=Math.max(-8,Math.min(8,next.biases[a][b]-scale*g));}));
 const avg=loss/Math.max(1,data.length);if(!Number.isFinite(avg)||next.weights.some(l=>l.some(r=>r.some(w=>!Number.isFinite(w)))))throw Error("Numerical instability. Reset the network or reduce the learning rate.");
 return{network:next,loss:avg};
}
export default function NeuralNetworkPlayground({points,learningRate,epochs,seed}:{points:DataPoint[];learningRate:number;epochs:number;seed:number}){
 const [activation,setActivation]=useState<Activation>("tanh"),[sizes,setSizes]=useState([4,3]),[arch,setArch]=useState(0),[history,setHistory]=useState<Frame[]>([]),[step,setStep]=useState(0),[playing,setPlaying]=useState(false),[error,setError]=useState(""),[mounted,setMounted]=useState(false);
 const framesRef=useRef<Frame[]>([]),stepRef=useRef(0),trainRef=useRef<()=>void>(()=>{});
 useEffect(()=>setMounted(true),[]);
 const labels=useMemo(()=>[...new Set(points.map(p=>p.label))].sort((a,b)=>a-b),[points]),positive=labels[labels.length-1]??1;
 const stats=useMemo(()=>{const mx=points.reduce((s,p)=>s+p.x,0)/Math.max(1,points.length),my=points.reduce((s,p)=>s+p.y,0)/Math.max(1,points.length);return{mx,my,sx:Math.sqrt(points.reduce((s,p)=>s+(p.x-mx)**2,0)/Math.max(1,points.length))||1,sy:Math.sqrt(points.reduce((s,p)=>s+(p.y-my)**2,0)/Math.max(1,points.length))||1};},[points]);
 const data=useMemo(()=>points.map(p=>({x:(p.x-stats.mx)/stats.sx,y:(p.y-stats.my)/stats.sy,target:p.label===positive?1:0})),[points,stats,positive]);
 function reset(){const f={network:init([2,...sizes,1],seed+arch),loss:Math.LN2,epoch:0};framesRef.current=[f];setHistory([f]);stepRef.current=0;setStep(0);setPlaying(false);setError("");}
 useEffect(()=>{reset();},[points,seed,arch]);
 const frame=history[Math.min(step,Math.max(0,history.length-1))]??{network:init([2,...sizes,1],seed+arch),loss:Math.LN2,epoch:0};
 function next(){setError("");if(stepRef.current<framesRef.current.length-1){const n=stepRef.current+1;stepRef.current=n;setStep(n);return;}if(stepRef.current>=epochs){setPlaying(false);return;}try{const result=trainOne(framesRef.current[framesRef.current.length-1].network,data,activation,learningRate);const f={...result,epoch:framesRef.current[framesRef.current.length-1].epoch+1};const all=[...framesRef.current,f];framesRef.current=all;stepRef.current=all.length-1;setHistory(all);setStep(all.length-1);}catch(e){setError(e instanceof Error?e.message:"Training failed");setPlaying(false);}}
 useEffect(()=>{trainRef.current=next;});
 useEffect(()=>{if(!playing)return;const id=window.setInterval(()=>trainRef.current(),120);return()=>window.clearInterval(id);},[playing]);
 useEffect(()=>{if(step>=epochs)setPlaying(false);},[step,epochs]);
 const layers=[2,...frame.network.weights.map(l=>l.length)],grid=useMemo(()=>{const cells:{x:number;y:number;v:number}[]=[];for(let iy=0;iy<30;iy++)for(let ix=0;ix<30;ix++){const rawX=-1.5+ix*3/29,rawY=1.5-iy*3/29;const nx=(rawX-stats.mx)/stats.sx,ny=(rawY-stats.my)/stats.sy;cells.push({x:ix,y:iy,v:forward(frame.network,nx,ny,activation).out});}return cells;},[frame,activation,stats]);
 const controls=<div className="space-y-4">
  <div><label className="control-label">Activation function</label><select className="control-select" value={activation} onChange={e=>{setActivation(e.target.value as Activation);reset();}}><option value="tanh">Tanh</option><option value="relu">ReLU</option><option value="sigmoid">Sigmoid</option><option value="linear">Linear</option></select></div>
  <div className="rounded-lg border p-3 space-y-3" style={{borderColor:"var(--border)"}}><div className="flex justify-between"><span className="text-xs font-semibold">Hidden layer configuration</span><span className="text-[10px]">{sizes.length} layers</span></div>
   {sizes.map((n,i)=><div key={i} className="flex items-center justify-between text-xs"><span>Hidden layer {i+1}</span><div className="flex items-center gap-2"><button className="icon-button !w-7 !h-7 border" onClick={()=>setSizes(v=>v.map((x,j)=>j===i?Math.max(1,x-1):x))}>−</button><b>{n}</b><button className="icon-button !w-7 !h-7 border" onClick={()=>setSizes(v=>v.map((x,j)=>j===i?Math.min(12,x+1):x))}>+</button></div></div>)}
   <div className="flex gap-3"><button className="text-button" disabled={sizes.length>=6} onClick={()=>setSizes(v=>v.length<6?[...v,3]:v)}>+ Add layer</button><button className="text-button" disabled={sizes.length<=1} onClick={()=>setSizes(v=>v.length>1?v.slice(0,-1):v)}>Remove last</button></div><button className="text-button" onClick={()=>setArch(v=>v+1)}>Apply architecture</button>
  </div><div className="grid grid-cols-2 gap-2"><div className="metric-card"><div className="metric-label">Epoch</div><div className="metric-value">{frame.epoch}</div></div><div className="metric-card"><div className="metric-label">Training loss</div><div className="metric-value">{frame.loss.toFixed(5)}</div></div></div>
 </div>;
 const execution=<div className="space-y-2"><div className="flex items-center justify-between"><div><div className="panel-title">Execution</div><div className="panel-description">Step {step+1} of {epochs+1} · {step===0?"Initialization":playing?"Training":"Epoch "+step}</div></div><button className="text-button" onClick={reset}><RotateCcw size={13}/>Reset</button></div>
  <input type="range" min="0" max={Math.max(0,history.length-1)} value={step} onChange={e=>{const n=Number(e.target.value);stepRef.current=n;setStep(n);setPlaying(false);}} className="w-full mt-3" aria-label="Training timeline"/>
  <div className="execution-compact-controls"><button className="execution-button" onClick={()=>{stepRef.current=0;setStep(0);setPlaying(false);}}><SkipBack size={16}/></button><button className="execution-button" disabled={step===0} onClick={()=>{const n=step-1;stepRef.current=n;setStep(n);setPlaying(false);}}><ChevronLeft size={17}/></button><button className="execution-button primary" onClick={()=>setPlaying(v=>!v)} disabled={step>=epochs&&history.length>epochs}>{playing?<Pause size={17}/>:<Play size={17} fill="currentColor"/>}</button><button className="execution-button" disabled={step>=epochs} onClick={()=>{setPlaying(false);next();}}><ChevronRight size={17}/></button><button className="execution-button" onClick={()=>setPlaying(true)} disabled={step>=epochs}><SkipForward size={16}/></button></div>
  <div className="execution-step-title" title={step===0?"Initialize weights and biases":"Epoch "+step+": forward propagation, loss calculation, backpropagation, and gradient descent"}>{step===0?"Initialize weights and biases": "Epoch "+step+": forward propagation, loss calculation, backpropagation, and gradient descent"}</div>{error&&<p className="text-xs text-red-500">{error}</p>}
 </div>;
 const width=Math.max(900,layers.length*230),height=Math.max(340,Math.max(...layers)*76+100),xAt=(l:number)=>80+l*(width-160)/(layers.length-1),yAt=(l:number,i:number)=>height/2-(layers[l]-1)*34+i*68;
 return <>
 {mounted&&document.getElementById("neural-network-controls-slot")&&createPortal(controls,document.getElementById("neural-network-controls-slot")!)}
 {mounted&&document.getElementById("neural-network-execution-slot")&&createPortal(execution,document.getElementById("neural-network-execution-slot")!)}
 <div className="space-y-5">
  <section className="analysis-card"><div className="analysis-card-title">Network architecture and live weights</div><p className="analysis-description">Connection labels and thickness represent the current weights and update at every epoch.</p><div className="overflow-auto rounded-lg mt-3" style={{maxHeight:560,background:"var(--plot-bg)"}}><svg width={width} height={height} viewBox={"0 0 "+width+" "+height} role="img" aria-label="Scrollable neural network architecture">
   {layers.slice(0,-1).map((count,l)=><g key={l}>{Array.from({length:count},(_,i)=>Array.from({length:layers[l+1]},(_,j)=>{const w=frame.network.weights[l][j][i],x1=xAt(l)+19,y1=yAt(l,i),x2=xAt(l+1)-19,y2=yAt(l+1,j);return <g key={i+"-"+j}><line x1={x1} y1={y1} x2={x2} y2={y2} stroke={w>=0?"#2788d8":"#ed8738"} strokeWidth={1+Math.min(4,Math.abs(w)*1.5)} opacity=".8"/><text x={(x1+x2)/2} y={(y1+y2)/2-3} fontSize="10" textAnchor="middle" fill="var(--text-muted)">{w.toFixed(2)}</text></g>;}))}</g>)}
   {layers.map((count,l)=><g key={l}><text x={xAt(l)} y="28" textAnchor="middle" fontSize="13" fill="var(--text-muted)">{l===0?"Input":l===layers.length-1?"Output":"Hidden "+l}</text>{Array.from({length:count},(_,i)=><g key={i}><circle cx={xAt(l)} cy={yAt(l,i)} r="19" fill="var(--panel-secondary)" stroke="var(--accent)" strokeWidth="2"/><text x={xAt(l)} y={yAt(l,i)+4} fontSize="10" textAnchor="middle" fill="var(--text-primary)">{l===0?"x"+(i+1):i+1}</text></g>)}</g>)}
  </svg></div><div className="plot-meta"><span>Blue: positive weights</span><span>Orange: negative weights</span><span>Epoch {frame.epoch} / {epochs}</span></div></section>
  <section className="analysis-card"><div className="flex justify-between gap-3"><div><div className="analysis-card-title">Learned decision boundary</div><p className="analysis-description">The output changes as you move through the training history.</p></div><span className="phase-pill">{playing?"Training":"Paused"} · Epoch {frame.epoch}</span></div>
   <svg viewBox="0 0 480 480" className="w-full max-h-[620px] mt-3"><rect x="60" y="60" width="360" height="360" fill="var(--plot-bg)"/>{grid.map((c,i)=><rect key={i} x={60+c.x*360/30} y={60+c.y*360/30} width="12.1" height="12.1" fill={c.v>=.5?"#3b82f6":"#f97316"} opacity={.08+Math.abs(c.v-.5)*.55}/>)}{points.map(p=><circle key={p.id} cx={240+p.x*150} cy={240-p.y*150} r="4" fill={p.label===positive?"#3b82f6":"#f97316"} stroke="var(--panel-bg)" strokeWidth="1.3"/>)}</svg><div className="plot-meta"><span>Blue: positive class</span><span>Orange: negative class</span><span>Binary cross-entropy</span></div>
  </section>
  <section className="analysis-card"><div className="analysis-card-title">Current training step</div><p className="analysis-description">{step===0?"The network is initialized. Advance one step to run forward propagation and backpropagation.":"This view shows the exact weights and loss saved at epoch "+step+"."}</p><div className="grid grid-cols-2 gap-3 mt-4"><div className="metric-card"><div className="metric-label">Training samples</div><div className="metric-value">{points.length}</div></div><div className="metric-card"><div className="metric-label">Parameters</div><div className="metric-value">{frame.network.weights.reduce((s,l)=>s+l.reduce((a,r)=>a+r.length,0),0)+frame.network.biases.reduce((s,l)=>s+l.length,0)}</div></div></div></section>
 </div></>;
}
