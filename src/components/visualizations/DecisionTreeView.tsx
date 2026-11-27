"use client";
import { useMemo } from "react";
import type { DecisionTreeNode } from "@/lib/algorithms/types";

type Positioned = {
  node: DecisionTreeNode;
  id: string;
  x: number;
  y: number;
  parent?: string;
  branch?: "Yes" | "No";
};

function isSplit(node: DecisionTreeNode): boolean {
  return (node.feature === "x" || node.feature === "y")
    && typeof node.threshold === "number"
    && Number.isFinite(node.threshold)
    && !!node.left && !!node.right;
}

export default function DecisionTreeView({ tree }: { tree?: DecisionTreeNode }) {
  const layout = useMemo(() => {
    if (!tree || typeof tree !== "object") {
      return { nodes: [] as Positioned[], width: 640, height: 180 };
    }

    const nodes: Positioned[] = [];
    let leafCount = 0;
    let maxDepth = 0;

    // Assign each leaf a unique horizontal slot, then center every parent
    // between its children. Depth is computed locally, never read from data.
    const visit = (
      node: DecisionTreeNode,
      depth: number,
      id: string,
      parent?: string,
      branch?: "Yes" | "No",
      ancestors = new Set<DecisionTreeNode>()
    ): number => {
      const safeDepth = Number.isFinite(depth) ? Math.max(0, depth) : 0;
      maxDepth = Math.max(maxDepth, safeDepth);

      // Protect rendering from malformed/cyclic snapshots.
      if (!node || typeof node !== "object" || ancestors.has(node)) {
        const x = leafCount++;
        nodes.push({ node: { prediction: 0 }, id, x: 0, y: 0, parent, branch });
        return x;
      }

      const nextAncestors = new Set(ancestors);
      nextAncestors.add(node);
      let slot: number;

      if (isSplit(node)) {
        const left = visit(node.left!, safeDepth + 1, id + "L", id, "Yes", nextAncestors);
        const right = visit(node.right!, safeDepth + 1, id + "R", id, "No", nextAncestors);
        slot = (left + right) / 2;
      } else {
        slot = leafCount++;
      }

      nodes.push({
        node,
        id,
        x: slot,
        y: 55 + safeDepth * 130,
        parent,
        branch
      });
      return slot;
    };

    visit(tree, 0, "root");
    const width = Math.max(640, leafCount * 150 + 80);
    const height = (maxDepth + 1) * 130 + 40;
    const denominator = Math.max(1, leafCount - 1);

    const positioned = nodes.map(node => ({
      ...node,
      x: Number.isFinite(node.x) ? 60 + node.x * (width - 120) / denominator : 60,
      y: Number.isFinite(node.y) ? node.y : 55
    }));

    return { nodes: positioned, width, height };
  }, [tree]);

  if (!tree) return <p className="empty-note">The tree has not been constructed yet.</p>;

  const byId = new Map(layout.nodes.map(node => [node.id, node]));

  return (
    <div className="overflow-auto rounded-lg p-3" style={{ background: "var(--plot-bg)", maxHeight: 620 }}>
      <svg
        width={layout.width}
        height={layout.height}
        viewBox={`0 0 ${layout.width} ${layout.height}`}
        role="img"
        aria-label="Connected decision tree"
      >
        {layout.nodes.filter(node => node.parent).map(node => {
          const parent = byId.get(node.parent!);
          if (!parent || ![parent.x, parent.y, node.x, node.y].every(Number.isFinite)) return null;
          return (
            <g key={node.id}>
              <line
                x1={parent.x}
                y1={parent.y + 27}
                x2={node.x}
                y2={node.y - 27}
                stroke="var(--border-strong)"
                strokeWidth="2"
              />
              <text
                x={(parent.x + node.x) / 2}
                y={(parent.y + node.y) / 2 - 4}
                fontSize="11"
                textAnchor="middle"
                fill="var(--text-muted)"
              >
                {node.branch}
              </text>
            </g>
          );
        })}
        {layout.nodes.map(node => {
          const split = isSplit(node.node);
          const prediction = Number.isFinite(node.node.prediction) ? node.node.prediction : 0;
          return (
            <g key={node.id}>
              <rect
                x={node.x - 54}
                y={node.y - 26}
                width="108"
                height="52"
                rx="9"
                fill="var(--panel-secondary)"
                stroke={split ? "var(--accent)" : "var(--border-strong)"}
                strokeWidth="1.5"
              />
              <text
                x={node.x}
                y={node.y - 3}
                textAnchor="middle"
                fontSize="12"
                fontWeight="600"
                fill="var(--text-primary)"
              >
                {split ? `${node.node.feature!.toUpperCase()} ≤ ${node.node.threshold!.toFixed(3)}?` : `Class ${prediction}`}
              </text>
              <text x={node.x} y={node.y + 13} textAnchor="middle" fontSize="10" fill="var(--text-muted)">
                {split ? "Decision node" : "Leaf prediction"}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
