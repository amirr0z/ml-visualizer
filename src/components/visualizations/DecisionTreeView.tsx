"use client";
import type { DecisionTreeNode } from "@/lib/algorithms/types";
function Branch({node}:{node:DecisionTreeNode}) {
 const split=node.feature&&node.threshold!==undefined&&node.left&&node.right;
 return <div className="flex flex-col items-center">
  <div className="rounded-lg border px-3 py-2 text-center min-w-24" style={{borderColor:split?"var(--accent)":"var(--border-strong)",background:"var(--panel-secondary)"}}>
   {split?<><div className="text-xs font-semibold">{node.feature!.toUpperCase()} ≤ {node.threshold!.toFixed(3)}?</div><div className="text-[10px]" style={{color:"var(--text-muted)"}}>Decision node</div></>:<><div className="text-xs font-semibold">Class {node.prediction}</div><div className="text-[10px]" style={{color:"var(--text-muted)"}}>Leaf prediction</div></>}
  </div>
  {split&&<div className="flex gap-8 mt-2"><div className="flex flex-col items-center"><span className="text-[10px] mb-1">Yes</span><div className="w-px h-3" style={{background:"var(--border-strong)"}}/><Branch node={node.left!}/></div><div className="flex flex-col items-center"><span className="text-[10px] mb-1">No</span><div className="w-px h-3" style={{background:"var(--border-strong)"}}/><Branch node={node.right!}/></div></div>}
 </div>;
}
export default function DecisionTreeView({tree}:{tree?:DecisionTreeNode}) {
 if(!tree)return <p className="empty-note">The tree has not been constructed yet.</p>;
 return <div className="overflow-x-auto rounded-lg p-4" style={{background:"var(--plot-bg)"}}><div className="min-w-max flex justify-center"><Branch node={tree}/></div></div>;
}
