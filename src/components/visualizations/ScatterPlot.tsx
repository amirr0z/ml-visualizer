
"use client";

import { DataPoint } from "@/lib/datasets/generateDataset";

interface Props {
    points: DataPoint[];
    showLabels?: boolean;
    title?: string;
}

const COLORS = ["#22d3ee", "#a78bfa", "#34d399"];

export default function ScatterPlot({
    points,
    showLabels = true,
    title = "2D Dataset",
}: Props) {
    const size = 1000;
    const padding = 55;

    const scale = (value: number) =>
        padding + ((value + 1.5) / 3) * (size - padding * 2);

    const center = size / 2;

    return (
        <div className="plot-container">
            <div className="plot-heading">
                <span>{title}</span>
                <span>{points.length} samples</span>
            </div>

            <svg
                viewBox={`0 0 ${size} ${size}`}
                className="scatter-svg"
                role="img"
                aria-label={title}
            >
                <defs>
                    <pattern
                        id="plot-grid"
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
                    fill="url(#plot-grid)"
                />

                {/* Coordinate axes */}
                <line
                    x1={center}
                    y1={padding}
                    x2={center}
                    y2={size - padding}
                    stroke="var(--plot-axis)"
                    strokeWidth="1.5"
                />

                <line
                    x1={padding}
                    y1={center}
                    x2={size - padding}
                    y2={center}
                    stroke="var(--plot-axis)"
                    strokeWidth="1.5"
                />

                {/* Dataset points */}
                {points.map(point => (
                    <circle
                        key={point.id}
                        cx={scale(point.x)}
                        cy={size - scale(point.y)}
                        r="9"
                        fill={
                            showLabels
                                ? COLORS[point.label % COLORS.length]
                                : "var(--plot-neutral)"
                        }
                        stroke="var(--plot-point-stroke)"
                        strokeWidth="2"
                    >
                        <title>
                            Point {point.id}: ({point.x.toFixed(3)}, {point.y.toFixed(3)})
                            {showLabels ? ` | Class ${point.label}` : ""}
                        </title>
                    </circle>
                ))}
            </svg>

            {showLabels && (
                <div className="plot-legend">
                    {[0, 1, 2].map(label => (
                        points.some(p => p.label === label) && (
                            <div key={label} className="legend-item">
                                <span
                                    className="legend-dot"
                                    style={{
                                        background: COLORS[label],
                                    }}
                                />
                                Class {label}
                            </div>
                        )
                    ))}
                </div>
            )}
        </div>
    );
}
