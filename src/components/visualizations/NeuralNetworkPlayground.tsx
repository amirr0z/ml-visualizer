"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { DataPoint } from "@/lib/datasets/generateDataset";
import { RotateCcw, ChevronLeft, ChevronRight, Play, Pause } from "lucide-react";

type Activation = "tanh" | "relu" | "sigmoid" | "linear";
type Network = { weights: number[][][]; biases: number[][] };

function activate(x: number, type: Activation) {
  if (type === "relu") return Math.max(0, x);
  if (type === "sigmoid") return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x))));
  if (type === "linear") return x;
  return Math.tanh(x);
}
function derivative(z: number, a: number, type: Activation) {
  if (type === "relu") return z > 0 ? 1 : 0;
  if (type === "sigmoid") return a * (1 - a);
  if (type === "linear") return 1;
  return 1 - a * a;
}
function initialize(sizes: number[], seed: number): Network {
  let state = seed >>> 0;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  return {
    weights: sizes.slice(1).map((size, layer) =>
      Array.from({ length: size }, () =>
        Array.from({ length: sizes[layer] }, () => (random() * 2 - 1) * Math.sqrt(2 / sizes[layer]))
      )
    ),
    biases: sizes.slice(1).map(size => Array(size).fill(0)),
  };
}
function forward(network: Network, x: number, y: number, activation: Activation) {
  const values: number[][] = [[x, y]];
  const zs: number[][] = [];
  for (let layer = 0; layer < network.weights.length; layer++) {
    const z = network.weights[layer].map((row, j) =>
      row.reduce((sum, weight, i) => sum + weight * values[layer][i], network.biases[layer][j])
    );
    zs.push(z);
    values.push(z.map(value => layer === network.weights.length - 1 ? activate(value, "sigmoid") : activate(value, activation)));
  }
  return { values, zs, output: values[values.length - 1][0] };
}

export default function NeuralNetworkPlayground({ points }: { points: DataPoint[] }) {
  const [activation, setActivation] = useState<Activation>("tanh");
  const [learningRate, setLearningRate] = useState(0.03);
  const [layerSizes, setLayerSizes] = useState([4, 3]);
  const [network, setNetwork] = useState<Network>(() => initialize([2, 4, 3, 1], 42));
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(0.69315);
  const [playing, setPlaying] = useState(false);
  const [seed, setSeed] = useState(42);
  const [mounted, setMounted] = useState(false);
  const trainRef = useRef<() => void>(() => {});

  useEffect(() => setMounted(true), []);

  const labels = useMemo(() => [...new Set(points.map(point => point.label))].sort((a, b) => a - b), [points]);
  const positive = labels[labels.length - 1] ?? 1;
  const normalized = useMemo(() => {
    const meanX = points.reduce((sum, p) => sum + p.x, 0) / Math.max(1, points.length);
    const meanY = points.reduce((sum, p) => sum + p.y, 0) / Math.max(1, points.length);
    const stdX = Math.sqrt(points.reduce((sum, p) => sum + (p.x - meanX) ** 2, 0) / Math.max(1, points.length)) || 1;
    const stdY = Math.sqrt(points.reduce((sum, p) => sum + (p.y - meanY) ** 2, 0) / Math.max(1, points.length)) || 1;
    return points.map(p => ({ x: (p.x - meanX) / stdX, y: (p.y - meanY) / stdY, target: p.label === positive ? 1 : 0 }));
  }, [points, positive]);

  function reset(nextSeed = seed) {
    setNetwork(initialize([2, ...layerSizes, 1], nextSeed));
    setEpoch(0);
    setLoss(0.69315);
    setPlaying(false);
  }

  function trainEpoch() {
    if (!normalized.length) return;
    const next: Network = {
      weights: network.weights.map(layer => layer.map(row => [...row])),
      biases: network.biases.map(layer => [...layer]),
    };
    const weightGradients = next.weights.map(layer => layer.map(row => row.map(() => 0)));
    const biasGradients = next.biases.map(layer => layer.map(() => 0));
    let totalLoss = 0;

    for (const point of normalized) {
      const pass = forward(next, point.x, point.y, activation);
      const output = Math.max(1e-7, Math.min(1 - 1e-7, pass.output));
      totalLoss += -(point.target * Math.log(output) + (1 - point.target) * Math.log(1 - output));
      let delta = [output - point.target];

      for (let layer = next.weights.length - 1; layer >= 0; layer--) {
        for (let neuron = 0; neuron < delta.length; neuron++) {
          biasGradients[layer][neuron] += delta[neuron];
          for (let input = 0; input < next.weights[layer][neuron].length; input++) {
            weightGradients[layer][neuron][input] += delta[neuron] * pass.values[layer][input];
          }
        }
        if (layer > 0) {
          const previousDelta = Array(next.weights[layer][0].length).fill(0);
          for (let input = 0; input < previousDelta.length; input++) {
            let propagated = 0;
            for (let neuron = 0; neuron < delta.length; neuron++) propagated += next.weights[layer][neuron][input] * delta[neuron];
            previousDelta[input] = propagated * derivative(pass.zs[layer - 1][input], pass.values[layer][input + 1], activation);
          }
          delta = previousDelta;
        }
      }
    }

    const scale = Math.min(learningRate, 0.08) / normalized.length;
    next.weights.forEach((layer, l) => layer.forEach((row, j) => row.forEach((_, i) => {
      const gradient = Math.max(-5, Math.min(5, weightGradients[l][j][i]));
      next.weights[l][j][i] = Math.max(-6, Math.min(6, next.weights[l][j][i] - scale * gradient));
    })));
    next.biases.forEach((layer, l) => layer.forEach((_, j) => {
      const gradient = Math.max(-5, Math.min(5, biasGradients[l][j]));
      next.biases[l][j] = Math.max(-6, Math.min(6, next.biases[l][j] - scale * gradient));
    }));

    const nextLoss = totalLoss / normalized.length;
    if (!Number.isFinite(nextLoss) || next.weights.some(layer => layer.some(row => row.some(w => !Number.isFinite(w))))) { setPlaying(false); return; }
    setNetwork(next);
    setEpoch(current => current + 1);
    setLoss(nextLoss);
  }

  useEffect(() => { trainRef.current = trainEpoch; });
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => trainRef.current(), 80);
    return () => window.clearInterval(timer);
  }, [playing]);

  const execution = (
    <div className="space-y-2">
      <div className="flex items-center justify-between"><div><div className="panel-title">Execution</div><div className="panel-description">Epoch {epoch} · {playing ? "Training" : "Paused"}</div></div><button className="text-button" onClick={() => reset()}><RotateCcw size={13} />Reset</button></div>
      <div className="execution-compact-controls">
        <button className="execution-button" onClick={() => reset()} aria-label="Reset"><RotateCcw size={15}/></button>
        <button className="execution-button" onClick={trainEpoch} disabled={playing} aria-label="One epoch"><ChevronLeft size={17}/></button>
        <button className="execution-button primary" onClick={() => setPlaying(v=>!v)} aria-label={playing?"Pause":"Train"}>{playing?<Pause size={17}/>:<Play size={17} fill="currentColor"/>}</button>
        <button className="execution-button" onClick={trainEpoch} disabled={playing} aria-label="Next epoch"><ChevronRight size={17}/></button>
      </div>
      <div className="execution-step-title">{epoch===0?"Initialize network weights":"Forward pass → backpropagation → weight update"}</div>
    </div>
  );
  const grid = useMemo(() => {
    const cells: { x: number; y: number; value: number }[] = [];
    for (let iy = 0; iy < 30; iy++) {
      for (let ix = 0; ix < 30; ix++) {
        const x = -1.5 + (ix / 29) * 3;
        const y = 1.5 - (iy / 29) * 3;
        cells.push({ x: ix, y: iy, value: forward(network, x, y, activation).output });
      }
    }
    return cells;
  }, [network, activation]);
  const layers = [2, ...network.weights.map(layer => layer.length)];

  const controls = (
    <div className="space-y-4">
      <div>
        <label className="control-label">Activation function</label>
        <select className="control-select" value={activation} onChange={event => { setActivation(event.target.value as Activation); setPlaying(false); }}>
          <option value="tanh">Tanh</option><option value="relu">ReLU</option><option value="sigmoid">Sigmoid</option><option value="linear">Linear</option>
        </select>
      </div>
      <div>
        <label className="control-label"><span>Learning rate</span><span className="control-value">{learningRate.toFixed(3)}</span></label>
        <input type="range" min="0.001" max="0.2" step="0.001" value={learningRate} onChange={event => setLearningRate(Number(event.target.value))} />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium">Hidden layers</span>
        {layerSizes.map((size, index) => <div key={index} className="flex items-center gap-1 text-xs">
          <span>{index + 1}:</span><button className="icon-button !w-7 !h-7 border" onClick={() => setLayerSizes(current => current.map((n, i) => i === index ? Math.max(1, n - 1) : n))}>−</button>
          <strong>{size}</strong><button className="icon-button !w-7 !h-7 border" onClick={() => setLayerSizes(current => current.map((n, i) => i === index ? Math.min(8, n + 1) : n))}>+</button>
        </div>)}
      </div>
      <div className="flex flex-wrap gap-2">
        <button className="text-button" onClick={() => setLayerSizes(current => current.length < 4 ? [...current, 3] : current)}>Add layer</button>
        {layerSizes.length > 1 && <button className="text-button" onClick={() => setLayerSizes(current => current.slice(0, -1))}>Remove layer</button>}
        <button className="text-button" onClick={() => reset()}>Apply architecture</button>
      </div>
      <div className="flex flex-wrap gap-2">
        <button className="execution-button primary !w-auto !h-9 !rounded-lg px-4 text-xs" onClick={() => setPlaying(current => !current)}>{playing ? "Pause training" : "Train"}</button>
        <button className="text-button" onClick={trainEpoch} disabled={playing}>One epoch</button>
        <button className="text-button" onClick={() => reset()}>Reset weights</button>
        <button className="text-button" onClick={() => { const nextSeed = seed + 1; setSeed(nextSeed); reset(nextSeed); }}>New initialization</button>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div className="metric-card"><div className="metric-label">Epoch</div><div className="metric-value">{epoch}</div></div>
        <div className="metric-card"><div className="metric-label">Training loss</div><div className="metric-value">{loss.toFixed(5)}</div></div>
      </div>
    </div>
  );

  return <>
    {mounted && typeof document !== "undefined" && document.getElementById("neural-network-controls-slot") && createPortal(controls, document.getElementById("neural-network-controls-slot")!)}
    {mounted && typeof document !== "undefined" && document.getElementById("neural-network-execution-slot") && createPortal(execution, document.getElementById("neural-network-execution-slot")!)}
    <div className="space-y-5">
      <section className="analysis-card">
        <div className="analysis-card-title">Network architecture and live weights</div>
        <p className="analysis-description">Blue connections are positive, orange connections are negative. Labels show current weight values.</p>
        <svg viewBox="0 0 900 340" className="w-full mt-4" role="img" aria-label="Network with live weights">
          {layers.slice(0,-1).map((count,l)=>{const next=layers[l+1],x1=90+l*240,x2=x1+240,s1=Math.min(62,220/Math.max(1,count-1)),s2=Math.min(62,220/Math.max(1,next-1)),y1=(340-(count-1)*s1)/2,y2=(340-(next-1)*s2)/2;return <g key={l}>{Array.from({length:count},(_,i)=>Array.from({length:next},(_,j)=>{const w=network.weights[l][j][i],sx=x1+18,sy=y1+i*s1,ex=x2-18,ey=y2+j*s2;return <g key={i+"-"+j}><line x1={sx} y1={sy} x2={ex} y2={ey} stroke={w>=0?"#2788d8":"#ed8738"} strokeWidth={1+Math.min(4,Math.abs(w)*1.5)} opacity=".75"/><text x={(sx+ex)/2} y={(sy+ey)/2-3} fontSize="9" textAnchor="middle" fill="var(--text-muted)">{w.toFixed(2)}</text></g>}))}</g>})}
          {layers.map((count,l)=>{const step=Math.min(62,220/Math.max(1,count-1)),x=90+l*240,y=(340-(count-1)*step)/2;return <g key={l}><text x={x} y="22" textAnchor="middle" fontSize="12" fill="var(--text-muted)">{l===0?"Input":l===layers.length-1?"Output":"Hidden "+l}</text>{Array.from({length:count},(_,i)=><g key={i}><circle cx={x} cy={y+i*step} r="17" fill="var(--panel-secondary)" stroke="var(--accent)" strokeWidth="2"/><text x={x} y={y+i*step+4} fontSize="10" textAnchor="middle" fill="var(--text-primary)">{l===0?"x"+(i+1):i+1}</text></g>)}</g>})}
        </svg><div className="plot-meta"><span>Weights update after each epoch</span><span>Hidden activation: {activation}</span><span>Output: sigmoid</span></div>
      </section>
      <section className="analysis-card">
        <div className="flex flex-wrap justify-between gap-3"><div><div className="analysis-card-title">Learned decision boundary</div><p className="analysis-description">The model output updates as gradient descent trains the network.</p></div><span className="phase-pill">{playing ? "Training" : "Paused"} · Epoch {epoch}</span></div>
        <svg viewBox="0 0 480 480" className="w-full max-h-[620px] mt-3" role="img" aria-label="Neural network decision boundary">
          <rect x="60" y="60" width="360" height="360" fill="var(--plot-bg)" />
          {grid.map((cell, index) => <rect key={index} x={60 + cell.x * 360 / 30} y={60 + cell.y * 360 / 30} width="12.1" height="12.1" fill={cell.value >= 0.5 ? "#3b82f6" : "#f97316"} opacity={0.08 + Math.abs(cell.value - 0.5) * 0.55} />)}
          {points.map(point => <circle key={point.id} cx={240 + point.x * 150} cy={240 - point.y * 150} r="4" fill={point.label === positive ? "#3b82f6" : "#f97316"} stroke="var(--panel-bg)" strokeWidth="1.3" />)}
          <rect x="60" y="60" width="360" height="360" fill="none" stroke="var(--plot-axis)" />
        </svg>
        <div className="plot-meta"><span>Blue: positive class</span><span>Orange: negative class</span><span>Binary cross-entropy</span></div>
      </section>
      <section className="analysis-card">
        <div className="analysis-card-title">How training works</div>
        <ol className="guide-list"><li><span>1</span>Forward propagation calculates weighted sums and activations.</li><li><span>2</span>Binary cross-entropy measures prediction error.</li><li><span>3</span>Backpropagation calculates gradients for each weight and bias.</li><li><span>4</span>Gradient descent updates parameters using the learning rate.</li></ol>
      </section>
    </div>
  </>;
}
