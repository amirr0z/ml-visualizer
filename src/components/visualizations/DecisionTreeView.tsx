"use client";
import { useMemo } from "react";
import type { DecisionTreeNode } from "@/lib/algorithms/types";
type Positioned={node:DecisionTreeNode;id:string;x:number;y:number;depth:number;parent?:string;branch?:"Yes"|"No"};
export default function DecisionTreeView({tree}:{tree?:DecisionTreeNode}){
 const layout=useMemo(()=>{if(!tree)return{nodes:[] as Positioned[],width:640,height:180};const nodes:Positioned[]=[];let leaves=0,maxDepth=0;
  const visit=(node:DecisionTreeNode,depth:number,id:string,parent?:string,branch?:"Yes"|"No"):number=>{maxDepth=Math.max(maxDepth,depth);const split=!!(node.feature&&node.threshold!==undefined&&node.left&&node.right);let x:number;if(split){const a=visit(node.left!,depth+1,id+"L",id,"Yes"),b=visit(node.right!,depth+1,id+"R",id,"No");x=(a+b)/2;}else{x=leaves++;}nodes.push({node,id,x,y:depth,parent,branch});return x;};
  visit(tree,0,"root");const width=Math.max(640,leaves*150+80),height=(maxDepth+1)*130+40;nodes.forEach(n=>{n.x=60+n.x*(width-120)/Math.max(1,leaves-1);n.y=55+n.depth*130;});return{nodes,width,height};},[tree]);
 if(!tree)return <p className="empty-note">The tree has not been constructed yet.</p>;const byId=new Map(layout.nodes.map(n=>[n.id,n]));
 return <div className="overflow-auto rounded-lg p-3" style={{background:"var(--plot-bg)",maxHeight:620}}><svg width={layout.width} height={layout.height} viewBox={"0 0 "+layout.width+" "+layout.height} role="img" aria-label="Connected decision tree">
 {layout.nodes.filter(n=>n.parent).map(n=>{const p=byId.get(n.parent!);return p?<g key={n.id}><line x1={p.x} y1={p.y+27} x2={n.x} y2={n.y-27} stroke="var(--border-strong)" strokeWidth="2"/><text x={(p.x+n.x)/2} y={(p.y+n.y)/2-4} fontSize="11" textAnchor="middle" fill="var(--text-muted)">{n.branch}</text></g>:null;})}
 {layout.nodes.map(n=>{const split=!!(n.node.feature&&n.node.threshold!==undefined&&n.node.left&&n.node.right);return <g key={n.id}><rect x={n.x-54} y={n.y-26} width="108" height="52" rx="9" fill="var(--panel-secondary)" stroke={split?"var(--accent)":"var(--border-strong)"} strokeWidth="1.5"/><text x={n.x} y={n.y-3} textAnchor="middle" fontSize="12" fontWeight="600" fill="var(--text-primary)">{split?n.node.feature!.toUpperCase()+" ≤ "+n.node.threshold!.toFixed(3):"Class "+n.node.prediction}</text><text x={n.x} y={n.y+13} textAnchor="middle" fontSize="10" fill="var(--text-muted)">{split?"Decision node":"Leaf prediction"}</text></g>;})}
 </svg></div>;
}
