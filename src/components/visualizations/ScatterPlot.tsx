
"use client";

import { useEffect, useId, useState } from "react";
import type { DataPoint } from "@/lib/datasets/generateDataset";
import type {
    Centroid,
    RegressionLine,
    DecisionSplit,
} from "@/lib/algorithms/types";

interface Props {
    points: DataPoint[];
    showLabels?: boolean;
    title?: string;
    assignments?: number[];
    centroids?: Centroid[];
    regressionLine?: RegressionLine;
    decisionSplits?: DecisionSplit[];
    highlightedPointIds?: number[];
}

const COLORS = [
    "#22d3ee",
    "#a78bfa",
    "#34d399",
    "#fb923c",
    "#f472b6",
    "#818cf8",
    "#facc15",
    "#2dd4bf",
];

export default function ScatterPlot({
    points,
    showLabels = true,
    title = "2D Dataset",
    assignments,
    centroids = [],
    regressionLine,
    decisionSplits = [],
    highlightedPointIds = [],
}: Props) {
    const [mounted, setMounted] = useState(false);
    const id = useId().replace(/:/g, "");

    useEffect(() => {
        setMounted(true);
    }, []);

    const size = 1000;
    const padding = 65;

    const min = -1.5;
    const max = 1.5;

    const scale = (value: number) =>
        padding +
        ((value - min) / (max - min)) *
        (size - padding * 2);

    const yScale = (value: number) =>
        size - scale(value);

    const hasLine =
        regressionLine &&
        Number.isFinite(regressionLine.slope) &&
        Number.isFinite(regressionLine.intercept);

    const lineY1 = hasLine
        ? regressionLine.slope * min + regressionLine.intercept
        : 0;

    const lineY2 = hasLine
        ? regressionLine.slope * max + regressionLine.intercept
        : 0;

    const classes = Array.from(new Set(
        points.map(p => assignments?.[p.id] ?? p.label)
    )).sort((a, b) => a - b);

    if (!mounted) {
        return (
            <div className="plot-container">
                <div className="plot-heading">
                    <span>{title}</span>
                    <span>{points.length} samples</span>
                </div>
                <div className="plot-stage" />
            </div>
        );
    }

    return (
        <div className="plot-container">

            <div className="plot-heading">
                <span>{title}</span>
                <span>{points.length} samples</span>
            </div>

            <div className="plot-stage">

                <svg
                    viewBox={`0 0 ${size} ${size}`}
                    preserveAspectRatio="xMidYMid meet"
                    className="scatter-svg"
                    role="img"
                    aria-label={title}
                >

                    <defs>
                        <pattern
                            id={`grid-${id}`}
                            width="50"
                            height="50"
                            patternUnits="userSpaceOnUse"
                        >
                            <path
                                d="M 50 0 L 0 0 0 50"
                                fill="none"
                                stroke="var(--plot-grid)"
                                strokeWidth="1"
                            />
                        </pattern>
                    </defs>

                    <rect
                        width={size}
                        height={size}
                        fill={`url(#grid-${id})`}
                    />

                    <line
                        x1={scale(0)}
                        y1={padding}
                        x2={scale(0)}
                        y2={size - padding}
                        stroke="var(--plot-axis)"
                        strokeWidth="1.5"
                    />

                    <line
                        x1={padding}
                        y1={yScale(0)}
                        x2={size - padding}
                        y2={yScale(0)}
                        stroke="var(--plot-axis)"
                        strokeWidth="1.5"
                    />

                    {/* Regression / SVM line */}

                    {hasLine && (
                        <line
                            x1={scale(min)}
                            y1={yScale(lineY1)}
                            x2={scale(max)}
                            y2={yScale(lineY2)}
                            stroke="#f97316"
                            strokeWidth="4"
                            strokeLinecap="round"
                        />
                    )}

                    {/* Decision tree splits */}

                    {decisionSplits.map((split, index) => (
                        <line
                            key={index}
                            x1={split.feature === "x"
                                ? scale(split.threshold)
                                : padding}
                            x2={split.feature === "x"
                                ? scale(split.threshold)
                                : size - padding}
                            y1={split.feature === "x"
                                ? padding
                                : yScale(split.threshold)}
                            y2={split.feature === "x"
                                ? size - padding
                                : yScale(split.threshold)}
                            stroke="#f97316"
                            strokeWidth="2"
                            strokeDasharray="8 5"
                            opacity={Math.max(0.3, 1 - split.depth * 0.15)}
                        />
                    ))}

                    {/* Data points */}

                    {points.map(point => {
                        const cluster = assignments?.[point.id];

                        const color = cluster !== undefined
                            ? COLORS[cluster % COLORS.length]
                            : showLabels
                                ? COLORS[point.label % COLORS.length]
                                : "var(--plot-neutral)";

                        const highlighted =
                            highlightedPointIds.includes(point.id);

                        return (
                            <circle
                                key={point.id}
                                cx={scale(point.x)}
                                cy={yScale(point.y)}
                                r={highlighted ? 12 : 8}
                                fill={color}
                                stroke={highlighted
                                    ? "#f97316"
                                    : "var(--plot-point-stroke)"
                                }
                                strokeWidth={highlighted ? 4 : 2}
                                opacity={highlighted ? 1 : 0.88}
                            >
                                <title>
                                    Point {point.id}
                                    {"\n"}X: {point.x.toFixed(4)}
                                    {"\n"}Y: {point.y.toFixed(4)}
                                    {"\n"}Class: {point.label}
                                    {cluster !== undefined
                                        ? `\nCluster: ${cluster + 1}`
                                        : ""
                                    }
                                </title>
                            </circle>
                        );
                    })}

                    {/* Centroids */}

                    {centroids.map(center => (
                        <g
                            key={center.id}
                            transform={`translate(${scale(center.x)}, ${yScale(center.y)})`}
                        >
                            <circle
                                r="17"
                                fill={COLORS[center.id % COLORS.length]}
                                opacity="0.2"
                            />
                            <circle
                                r="10"
                                fill={COLORS[center.id % COLORS.length]}
                                stroke="var(--text-primary)"
                                strokeWidth="3"
                            />
                            <path
                                d="M -5 0 H 5 M 0 -5 V 5"
                                stroke="var(--text-primary)"
                                strokeWidth="2"
                            />
                        </g>
                    ))}

                    <text
                        x={size / 2}
                        y={size - 15}
                        fill="var(--text-muted)"
                        fontSize="14"
                        textAnchor="middle"
                    >
                        Feature 1
                    </text>

                    <text
                        x="18"
                        y={size / 2}
                        fill="var(--text-muted)"
                        fontSize="14"
                        textAnchor="middle"
                        transform={`rotate(-90 18 ${size / 2})`}
                    >
                        Feature 2
                    </text>

                </svg>

            </div>

            {/* Legend */}

            <div className="plot-legend">

                {classes.map(label => (
                    <div key={label} className="legend-item">
                        <span
                            className="legend-dot"
                            style={{
                                backgroundColor: COLORS[label % COLORS.length],
                            }}
                        />
                        {assignments
                            ? `Cluster ${label + 1}`
                            : `Class ${label}`}
                    </div>
                ))}

                {centroids.length > 0 && (
                    <div className="legend-item">
                        <span className="legend-dot" style={{
                            backgroundColor: "#f97316",
                        }} />
                        Centroids
                    </div>
                )}

            </div>

        </div>
    );
}
