
import type { DataPoint } from "@/lib/datasets/generateDataset";

export type AlgorithmName =
    | "CART"
    | "C4.5"
    | "KNN"
    | "SVM"
    | "Naive Bayes"
    | "K-Means"
    | "Linear Regression"
    | "PCA";

export type AlgorithmPhase =
    | "initialization"
    | "training"
    | "assignment"
    | "update"
    | "prediction"
    | "evaluation"
    | "converged";

export interface Centroid {
    id: number;
    x: number;
    y: number;
}

export interface RegressionLine {
    slope: number;
    intercept: number;
}

export interface DecisionSplit {
    feature: "x" | "y";
    threshold: number;
    depth: number;
}

export interface AlgorithmSnapshot {
    index: number;
    phase: AlgorithmPhase;
    title: string;
    description: string;
    equation?: string;

    assignments?: number[];
    predictions?: number[];

    centroids?: Centroid[];

    regressionLine?: RegressionLine;

    decisionSplits?: DecisionSplit[];

    highlightedPointIds?: number[];

    metrics: Record<string, number>;

    details?: {
        name: string;
        value: string;
    }[];
}

export interface AlgorithmResult {
    algorithm: AlgorithmName;
    snapshots: AlgorithmSnapshot[];
    finalMetrics: Record<string, number>;
    summary: string;
}

export interface AlgorithmParameters {
    clusters: number;
    neighbors: number;
    learningRate: number;
    epochs: number;
    maxDepth: number;
    regularization: number;
    tolerance: number;
    initialization: "random" | "kmeans++";
}

export const DEFAULT_PARAMETERS: AlgorithmParameters = {
    clusters: 3,
    neighbors: 5,
    learningRate: 0.03,
    epochs: 50,
    maxDepth: 4,
    regularization: 0.01,
    tolerance: 0.0001,
    initialization: "kmeans++",
};

export interface AlgorithmInput {
    algorithm: AlgorithmName;
    points: DataPoint[];
    parameters: AlgorithmParameters;
    seed: number;
}
