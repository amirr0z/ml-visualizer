
"use client";

import { useEffect, useId, useState } from "react";

import type {
    DataPoint,
} from "@/lib/datasets/generateDataset";

interface ScatterPlotProps {
    points: DataPoint[];
    showLabels?: boolean;
    title?: string;
}

const COLORS = [
    "#22d3ee",
    "#a78bfa",
    "#34d399",
];

export default function ScatterPlot({
    points,
    showLabels = true,
    title = "2D Dataset",
}: ScatterPlotProps) {
    const [mounted, setMounted] = useState(false);

    const rawId = useId();

    const gridId = `plot-grid-${rawId.replace(/:/g, "")}`;

    useEffect(() => {
        setMounted(true);
    }, []);

    const size = 1000;
    const padding = 65;
    const center = size / 2;

    const scale = (value: number) => {
        return (
            padding +
            ((value + 1.5) / 3) *
            (size - padding * 2)
        );
    };

    const classes = Array.from(
        new Set(points.map(point => point.label))
    ).sort((a, b) => a - b);

    // =======================================
    // HYDRATION FALLBACK
    // =======================================

    if (!mounted) {
        return (
            <div className="plot-container">
                <div className="plot-heading">
                    <span>{title}</span>
                    <span>{points.length} samples</span>
                </div>

                <div className="plot-stage">
                    <div
                        className="w-full h-full rounded-lg"
                        style={{
                            backgroundColor: "var(--plot-bg)",
                        }}
                    />
                </div>
            </div>
        );
    }

    // =======================================
    // RENDER
    // =======================================

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
                            id={gridId}
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

                    {/* Background */}

                    <rect
                        width={size}
                        height={size}
                        fill="url(#${gridId})"
                    />

                    {/* Grid */}

                    <rect
                        width={size}
                        height={size}
                        fill={`url(#${gridId})`}
                    />

                    {/* Vertical axis */}

                    <line
                        x1={center}
                        y1={padding}
                        x2={center}
                        y2={size - padding}
                        stroke="var(--plot-axis)"
                        strokeWidth="1.5"
                    />

                    {/* Horizontal axis */}

                    <line
                        x1={padding}
                        y1={center}
                        x2={size - padding}
                        y2={center}
                        stroke="var(--plot-axis)"
                        strokeWidth="1.5"
                    />

                    {/* Points */}

                    {points.map(point => {
                        const color = showLabels
                            ? COLORS[point.label % COLORS.length]
                            : "var(--plot-neutral)";

                        return (
                            <circle
                                key={point.id}
                                cx={scale(point.x)}
                                cy={size - scale(point.y)}
                                r="8"
                                fill={color}
                                stroke="var(--plot-point-stroke)"
                                strokeWidth="2"
                                className="transition-all duration-150"
                            >
                                <title>
                                    Point {point.id}
                                    {"\n"}
                                    X: {point.x.toFixed(3)}
                                    {"\n"}
                                    Y: {point.y.toFixed(3)}
                                    {showLabels
                                        ? `\nClass: ${point.label}`
                                        : ""}
                                </title>
                            </circle>
                        );
                    })}

                    {/* Axis labels */}

                    <text
                        x={center}
                        y={size - 15}
                        fill="var(--text-muted)"
                        fontSize="14"
                        textAnchor="middle"
                    >
                        Feature 1
                    </text>

                    <text
                        x="18"
                        y={center}
                        fill="var(--text-muted)"
                        fontSize="14"
                        textAnchor="middle"
                        transform={`rotate(-90 18 ${center})`}
                    >
                        Feature 2
                    </text>

                </svg>
            </div>

            {/* Ground truth legend */}

            {showLabels && (
                <div className="plot-legend">
                    {classes.map(label => (
                        <div
                            key={label}
                            className="legend-item"
                        >
                            <span
                                className="legend-dot"
                                style={{
                                    backgroundColor:
                                        COLORS[label % COLORS.length],
                                }}
                            />

                            Class {label}
                        </div>
                    ))}
                </div>
            )}

        </div>
    );
}
