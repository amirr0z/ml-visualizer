
import type { DataPoint } from "@/lib/datasets/generateDataset";

import {
    AlgorithmInput,
    AlgorithmResult,
    AlgorithmSnapshot,
    AlgorithmParameters,
    Centroid,
    DecisionSplit,
} from "./types";

// =============================================
// UTILITIES
// =============================================

const COLORS_COUNT = 8;

function distance(a: DataPoint | Centroid, b: DataPoint | Centroid) {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function mean(values: number[]) {
    return values.length
        ? values.reduce((a, b) => a + b, 0) / values.length
        : 0;
}

function variance(values: number[]) {
    if (!values.length) return 0;
    const m = mean(values);
    return mean(values.map(v => (v - m) ** 2));
}

function createRandom(seed: number) {
    let state = seed >>> 0;

    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

function shuffle<T>(items: T[], seed: number): T[] {
    const result = [...items];
    const random = createRandom(seed);

    for (let i = result.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [result[i], result[j]] = [result[j], result[i]];
    }

    return result;
}

function snapshot(
    index: number,
    phase: AlgorithmSnapshot["phase"],
    title: string,
    description: string,
    extra: Partial<AlgorithmSnapshot> = {}
): AlgorithmSnapshot {
    return {
        index,
        phase,
        title,
        description,
        metrics: {},
        ...extra,
    };
}

function makeResult(
    algorithm: AlgorithmResult["algorithm"],
    snapshots: AlgorithmSnapshot[],
    summary: string
): AlgorithmResult {
    return {
        algorithm,
        snapshots,
        finalMetrics: snapshots[snapshots.length - 1]?.metrics ?? {},
        summary,
    };
}

function trainTestSplit(points: DataPoint[], seed: number) {
    const shuffled = shuffle(points, seed);
    const split = Math.max(1, Math.floor(shuffled.length * 0.8));

    return {
        train: shuffled.slice(0, split),
        test: shuffled.slice(split),
    };
}

function accuracy(actual: number[], predicted: number[]) {
    if (!actual.length) return 0;

    return actual.filter(
        (value, index) => value === predicted[index]
    ).length / actual.length;
}

function mse(actual: number[], predicted: number[]) {
    if (!actual.length) return 0;

    return mean(actual.map(
        (value, index) => (value - predicted[index]) ** 2
    ));
}

function safeParameters(parameters: AlgorithmParameters): AlgorithmParameters {
    return {
        ...parameters,
        clusters: Math.max(2, Math.min(8, Math.floor(parameters.clusters))),
        neighbors: Math.max(1, Math.floor(parameters.neighbors)),
        epochs: Math.max(1, Math.floor(parameters.epochs)),
        maxDepth: Math.max(1, Math.floor(parameters.maxDepth)),
        learningRate: Math.max(0.0001, parameters.learningRate),
        tolerance: Math.max(0.000001, parameters.tolerance),
    };
}

// =============================================
// K-MEANS INITIALIZATION
// =============================================

function initializeCentroids(
    points: DataPoint[],
    k: number,
    method: "random" | "kmeans++",
    seed: number
): Centroid[] {
    const random = createRandom(seed);

    if (!points.length) return [];

    const chosen: DataPoint[] = [];

    const first = points[Math.floor(random() * points.length)];
    chosen.push(first);

    while (chosen.length < k) {
        if (method === "random") {
            const remaining = points.filter(
                p => !chosen.some(c => c.id === p.id)
            );

            if (!remaining.length) break;

            chosen.push(
                remaining[Math.floor(random() * remaining.length)]
            );

            continue;
        }

        const distances = points.map(point => {
            const nearest = Math.min(
                ...chosen.map(center => distance(point, center))
            );

            return nearest ** 2;
        });

        const total = distances.reduce((a, b) => a + b, 0);

        if (total <= 0) break;

        let target = random() * total;
        let selected = points[points.length - 1];

        for (let i = 0; i < points.length; i++) {
            target -= distances[i];

            if (target <= 0) {
                selected = points[i];
                break;
            }
        }

        if (chosen.some(p => p.id === selected.id)) break;

        chosen.push(selected);
    }

    return chosen.map((p, id) => ({
        id,
        x: p.x,
        y: p.y,
    }));
}

// =============================================
// K-MEANS
// =============================================

function runKMeans(
    points: DataPoint[],
    params: AlgorithmParameters,
    seed: number
): AlgorithmResult {
    const k = Math.min(params.clusters, points.length);

    const snapshots: AlgorithmSnapshot[] = [];

    let centroids = initializeCentroids(
        points,
        k,
        params.initialization,
        seed
    );

    let previousAssignments: number[] = [];

    snapshots.push(snapshot(
        0,
        "initialization",
        "Initialize centroids",
        `Initialize ${k} centroids using ${params.initialization === "kmeans++" ? "K-Means++" : "random initialization"}.`,
        {
            centroids: [...centroids],
            equation: "μ₁, μ₂, ..., μₖ",
            metrics: { clusters: k, inertia: 0 },
        }
    ));

    const maxIterations = 50;

    for (let iteration = 0; iteration < maxIterations; iteration++) {
        const assignments = points.map(point => {
            let bestCluster = 0;
            let bestDistance = Infinity;

            centroids.forEach((center, index) => {
                const d = distance(point, center);

                if (d < bestDistance) {
                    bestDistance = d;
                    bestCluster = index;
                }
            });

            return bestCluster;
        });

        const changed = assignments.some(
            (value, index) => value !== previousAssignments[index]
        );

        const inertia = points.reduce((total, point, index) => {
            const center = centroids[assignments[index]];
            return total + distance(point, center) ** 2;
        }, 0);

        snapshots.push(snapshot(
            snapshots.length,
            "assignment",
            `Assign points — iteration ${iteration + 1}`,
            "Each observation is assigned to its nearest centroid using Euclidean distance.",
            {
                assignments: [...assignments],
                centroids: [...centroids],
                equation: "Cᵢ = argminⱼ ||xᵢ - μⱼ||²",
                metrics: {
                    iteration: iteration + 1,
                    inertia,
                    changedPoints: previousAssignments.length
                        ? assignments.filter(
                            (v, i) => v !== previousAssignments[i]
                        ).length
                        : points.length,
                },
            }
        ));

        const updated = centroids.map((center, cluster) => {
            const members = points.filter(
                (_, index) => assignments[index] === cluster
            );

            if (!members.length) return center;

            return {
                id: cluster,
                x: mean(members.map(p => p.x)),
                y: mean(members.map(p => p.y)),
            };
        });

        const movement = mean(centroids.map(
            (center, index) => distance(center, updated[index])
        ));

        centroids = updated;

        const updatedInertia = points.reduce((total, point, index) => {
            return total + distance(point, centroids[assignments[index]]) ** 2;
        }, 0);

        snapshots.push(snapshot(
            snapshots.length,
            "update",
            `Update centroids — iteration ${iteration + 1}`,
            "Recalculate every centroid as the arithmetic mean of its assigned observations.",
            {
                assignments: [...assignments],
                centroids: [...centroids],
                equation: "μⱼ = (1 / |Cⱼ|) Σ xᵢ",
                metrics: {
                    iteration: iteration + 1,
                    inertia: updatedInertia,
                    centroidMovement: movement,
                },
            }
        ));

        const converged =
            !changed ||
            movement < params.tolerance;

        previousAssignments = [...assignments];

        if (converged) {
            snapshots.push(snapshot(
                snapshots.length,
                "converged",
                "K-Means converged",
                "The cluster assignments no longer change, or centroid movement is below the configured tolerance.",
                {
                    assignments: [...assignments],
                    centroids: [...centroids],
                    equation: "J = Σᵢ ||xᵢ - μcᵢ||²",
                    metrics: {
                        iteration: iteration + 1,
                        inertia: updatedInertia,
                        clusters: k,
                        centroidMovement: movement,
                    },
                }
            ));

            break;
        }
    }

    return makeResult(
        "K-Means",
        snapshots,
        "K-Means groups observations by minimizing within-cluster squared distances."
    );
}

// =============================================
// CART / C4.5 DECISION TREE
// =============================================

interface TreeNode {
    prediction: number;
    feature?: "x" | "y";
    threshold?: number;
    left?: TreeNode;
    right?: TreeNode;
}

function gini(points: DataPoint[]) {
    if (!points.length) return 0;

    const counts = new Map<number, number>();

    points.forEach(p => {
        counts.set(p.label, (counts.get(p.label) ?? 0) + 1);
    });

    return 1 - [...counts.values()].reduce(
        (sum, count) => sum + (count / points.length) ** 2,
        0
    );
}

function entropy(points: DataPoint[]) {
    if (!points.length) return 0;

    const counts = new Map<number, number>();

    points.forEach(p => {
        counts.set(p.label, (counts.get(p.label) ?? 0) + 1);
    });

    return -[...counts.values()].reduce((sum, count) => {
        const probability = count / points.length;
        return sum + probability * Math.log2(probability);
    }, 0);
}

function majority(points: DataPoint[]) {
    const counts = new Map<number, number>();

    points.forEach(p => {
        counts.set(p.label, (counts.get(p.label) ?? 0) + 1);
    });

    return [...counts.entries()].sort(
        (a, b) => b[1] - a[1]
    )[0]?.[0] ?? 0;
}

function splitScore(
    left: DataPoint[],
    right: DataPoint[],
    parent: DataPoint[],
    criterion: "gini" | "gain-ratio"
) {
    if (!left.length || !right.length) return -Infinity;

    const n = parent.length;
    const weighted =
        (left.length / n) * entropy(left) +
        (right.length / n) * entropy(right);

    if (criterion === "gini") {
        return gini(parent) -
            (left.length / n) * gini(left) -
            (right.length / n) * gini(right);
    }

    const gain = entropy(parent) - weighted;

    const p = left.length / n;
    const q = right.length / n;

    const splitInfo =
        -(p * Math.log2(p) + q * Math.log2(q));

    return splitInfo === 0 ? 0 : gain / splitInfo;
}

function trainTree(
    points: DataPoint[],
    params: AlgorithmParameters,
    criterion: "gini" | "gain-ratio"
) {
    const splits: DecisionSplit[] = [];

    function build(
        subset: DataPoint[],
        depth: number
    ): TreeNode {
        const prediction = majority(subset);

        if (
            depth >= params.maxDepth ||
            subset.length < 4 ||
            gini(subset) < 1e-12
        ) {
            return { prediction };
        }

        let bestScore = 0;
        let best:
            | {
                feature: "x" | "y";
                threshold: number;
                left: DataPoint[];
                right: DataPoint[];
            }
            | undefined;

        for (const feature of ["x", "y"] as const) {
            const values = [...new Set(
                subset.map(p => p[feature])
            )].sort((a, b) => a - b);

            for (let i = 0; i < values.length - 1; i++) {
                const threshold = (values[i] + values[i + 1]) / 2;

                const left = subset.filter(
                    p => p[feature] <= threshold
                );

                const right = subset.filter(
                    p => p[feature] > threshold
                );

                const score = splitScore(
                    left,
                    right,
                    subset,
                    criterion
                );

                if (score > bestScore) {
                    bestScore = score;
                    best = { feature, threshold, left, right };
                }
            }
        }

        if (!best) return { prediction };

        splits.push({
            feature: best.feature,
            threshold: best.threshold,
            depth,
        });

        return {
            prediction,
            feature: best.feature,
            threshold: best.threshold,
            left: build(best.left, depth + 1),
            right: build(best.right, depth + 1),
        };
    }

    return {
        root: build(points, 0),
        splits,
    };
}

function predictTree(node: TreeNode, point: DataPoint): number {
    if (
        !node.feature ||
        node.threshold === undefined ||
        !node.left ||
        !node.right
    ) {
        return node.prediction;
    }

    return point[node.feature] <= node.threshold
        ? predictTree(node.left, point)
        : predictTree(node.right, point);
}

function runTree(
    points: DataPoint[],
    params: AlgorithmParameters,
    seed: number,
    algorithm: "CART" | "C4.5"
): AlgorithmResult {
    const { train, test } = trainTestSplit(points, seed);

    const criterion = algorithm === "CART"
        ? "gini"
        : "gain-ratio";

    const snapshots: AlgorithmSnapshot[] = [];

    snapshots.push(snapshot(
        0,
        "initialization",
        "Prepare training data",
        `Split the dataset into ${train.length} training and ${test.length} testing observations.`,
        {
            metrics: {
                trainingSamples: train.length,
                testingSamples: test.length,
            },
        }
    ));

    const { root, splits } = trainTree(
        train,
        params,
        criterion
    );

    const predictions = test.map(
        point => predictTree(root, point)
    );

    const actual = test.map(p => p.label);
    const score = accuracy(actual, predictions);

    for (let i = 0; i < splits.length; i++) {
        const split = splits[i];

        snapshots.push(snapshot(
            snapshots.length,
            "training",
            `Create decision split ${i + 1}`,
            `Split on feature ${split.feature.toUpperCase()} at ${split.threshold.toFixed(3)}.`,
            {
                decisionSplits: splits.slice(0, i + 1),
                equation: algorithm === "CART"
                    ? "Gini = 1 - Σ pᵢ²"
                    : "Gain Ratio = Information Gain / Split Information",
                metrics: {
                    depth: split.depth,
                    splits: i + 1,
                },
            }
        ));
    }

    snapshots.push(snapshot(
        snapshots.length,
        "evaluation",
        "Evaluate decision tree",
        `The tree classifies the held-out observations with ${(score * 100).toFixed(1)}% accuracy.`,
        {
            predictions,
            decisionSplits: splits,
            metrics: {
                accuracy: score,
                depth: Math.max(0, ...splits.map(s => s.depth + 1)),
                nodes: 1 + splits.length * 2,
            },
        }
    ));

    return makeResult(
        algorithm,
        snapshots,
        algorithm === "CART"
            ? "CART uses Gini impurity to select binary splits."
            : "C4.5 uses information gain ratio to select decision splits."
    );
}

// =============================================
// K NEAREST NEIGHBORS
// =============================================

function runKNN(
    points: DataPoint[],
    params: AlgorithmParameters,
    seed: number
): AlgorithmResult {
    const { train, test } = trainTestSplit(points, seed);

    const snapshots: AlgorithmSnapshot[] = [];

    const query = test[0] ?? train[0];

    const ranked = train.map(point => ({
        id: point.id,
        distance: distance(point, query),
        label: point.label,
    })).sort((a, b) => a.distance - b.distance);

    const k = Math.min(params.neighbors, ranked.length);

    snapshots.push(snapshot(
        0,
        "initialization",
        "Select query point",
        "Choose an unseen observation and calculate its distance to every training observation.",
        {
            highlightedPointIds: [query.id],
            metrics: {
                trainingSamples: train.length,
                queryX: query.x,
                queryY: query.y,
            },
        }
    ));

    for (let i = 1; i <= k; i++) {
        const neighbors = ranked.slice(0, i);

        snapshots.push(snapshot(
            snapshots.length,
            "assignment",
            `Find nearest neighbor ${i}`,
            `The ${i}${i === 1 ? "st" : i === 2 ? "nd" : i === 3 ? "rd" : "th"} nearest observation is ${ranked[i - 1].id}.`,
            {
                highlightedPointIds: [
                    query.id,
                    ...neighbors.map(p => p.id),
                ],
                metrics: {
                    neighborsFound: i,
                    distance: ranked[i - 1].distance,
                },
            }
        ));
    }

    const nearest = ranked.slice(0, k);
    const votes = new Map<number, number>();

    nearest.forEach(point => {
        votes.set(
            point.label,
            (votes.get(point.label) ?? 0) + 1
        );
    });

    const prediction = [...votes.entries()].sort(
        (a, b) => b[1] - a[1]
    )[0][0];

    const correct = prediction === query.label;

    snapshots.push(snapshot(
        snapshots.length,
        "evaluation",
        "Classify query point",
        `Majority voting predicts class ${prediction}. Actual class: ${query.label}.`,
        {
            predictions: [prediction],
            highlightedPointIds: [
                query.id,
                ...nearest.map(p => p.id),
            ],
            equation: "ŷ = mode(y₁, y₂, ..., yₖ)",
            metrics: {
                k,
                prediction,
                actual: query.label,
                correct: correct ? 1 : 0,
            },
            details: nearest.map((p, i) => ({
                name: `Neighbor ${i + 1}`,
                value: `Point ${p.id}, distance ${p.distance.toFixed(4)}, class ${p.label}`,
            })),
        }
    ));

    return makeResult(
        "KNN",
        snapshots,
        "KNN classifies an observation using the majority class of its nearest neighbors."
    );
}

// =============================================
// GAUSSIAN NAIVE BAYES
// =============================================

function runNaiveBayes(
    points: DataPoint[],
    seed: number
): AlgorithmResult {
    const { train, test } = trainTestSplit(points, seed);

    const snapshots: AlgorithmSnapshot[] = [];

    const classes = [...new Set(train.map(p => p.label))];

    const statistics = classes.map(label => {
        const subset = train.filter(p => p.label === label);

        return {
            label,
            prior: subset.length / train.length,
            meanX: mean(subset.map(p => p.x)),
            meanY: mean(subset.map(p => p.y)),
            varianceX: Math.max(variance(subset.map(p => p.x)), 1e-6),
            varianceY: Math.max(variance(subset.map(p => p.y)), 1e-6),
        };
    });

    snapshots.push(snapshot(
        0,
        "initialization",
        "Calculate class priors",
        "Estimate the prior probability of each class from the training data.",
        {
            equation: "P(C) = N(C) / N",
            metrics: {
                classes: classes.length,
                trainingSamples: train.length,
            },
            details: statistics.map(s => ({
                name: `Class ${s.label}`,
                value: `Prior: ${s.prior.toFixed(4)}`,
            })),
        }
    ));

    snapshots.push(snapshot(
        1,
        "training",
        "Estimate Gaussian distributions",
        "Calculate the mean and variance of each feature for every class.",
        {
            equation: "P(x|C) = Gaussian(x; μ, σ²)",
            metrics: {
                classes: classes.length,
            },
            details: statistics.map(s => ({
                name: `Class ${s.label}`,
                value:
                    `μ=(${s.meanX.toFixed(3)}, ${s.meanY.toFixed(3)}), ` +
                    `σ²=(${s.varianceX.toFixed(3)}, ${s.varianceY.toFixed(3)})`,
            })),
        }
    ));

    const gaussianLog = (
        value: number,
        meanValue: number,
        varianceValue: number
    ) => {
        return -0.5 * (
            Math.log(2 * Math.PI * varianceValue) +
            ((value - meanValue) ** 2) / varianceValue
        );
    };

    const predict = (point: DataPoint) => {
        const scores = statistics.map(s => ({
            label: s.label,
            score:
                Math.log(s.prior) +
                gaussianLog(point.x, s.meanX, s.varianceX) +
                gaussianLog(point.y, s.meanY, s.varianceY),
        }));

        scores.sort((a, b) => b.score - a.score);

        return {
            label: scores[0].label,
            scores,
        };
    };

    const predictions = test.map(point => predict(point).label);
    const actual = test.map(p => p.label);

    const score = accuracy(actual, predictions);

    const query = test[0];

    if (query) {
        const result = predict(query);

        snapshots.push(snapshot(
            2,
            "prediction",
            "Calculate posterior probabilities",
            `The maximum posterior score predicts class ${result.label} for the selected observation.`,
            {
                highlightedPointIds: [query.id],
                predictions: [result.label],
                equation: "P(C|x) ∝ P(C)P(x₁|C)P(x₂|C)",
                details: result.scores.map(s => ({
                    name: `Class ${s.label}`,
                    value: `Log posterior: ${s.score.toFixed(4)}`,
                })),
                metrics: {
                    prediction: result.label,
                    actual: query.label,
                },
            }
        ));
    }

    snapshots.push(snapshot(
        snapshots.length,
        "evaluation",
        "Evaluate Naive Bayes",
        `Classification accuracy on the test set is ${(score * 100).toFixed(1)}%.`,
        {
            predictions,
            metrics: {
                accuracy: score,
                testSamples: test.length,
            },
        }
    ));

    return makeResult(
        "Naive Bayes",
        snapshots,
        "Gaussian Naive Bayes applies Bayes' theorem with conditionally independent Gaussian features."
    );
}

// =============================================
// LINEAR REGRESSION
// =============================================

function runLinearRegression(
    points: DataPoint[],
    params: AlgorithmParameters
): AlgorithmResult {
    const snapshots: AlgorithmSnapshot[] = [];

    const xMean = mean(points.map(p => p.x));
    const yMean = mean(points.map(p => p.y));

    let slope = 0;
    let intercept = yMean;

    snapshots.push(snapshot(
        0,
        "initialization",
        "Initialize regression model",
        "Start with a horizontal line at the mean target value.",
        {
            regressionLine: { slope, intercept },
            equation: "ŷ = wx + b",
            metrics: {
                mse: mse(
                    points.map(p => p.y),
                    points.map(() => intercept)
                )
            },
        }
    ));

    const learningRate = params.learningRate;

    for (let epoch = 0; epoch < params.epochs; epoch++) {
        let gradientW = 0;
        let gradientB = 0;

        points.forEach(point => {
            const prediction = slope * point.x + intercept;
            const error = prediction - point.y;

            gradientW += error * point.x;
            gradientB += error;
        });

        gradientW = (2 / points.length) * gradientW;
        gradientB = (2 / points.length) * gradientB;

        slope -= learningRate * gradientW;
        intercept -= learningRate * gradientB;

        const predictions = points.map(
            p => slope * p.x + intercept
        );

        const error = mse(
            points.map(p => p.y),
            predictions
        );

        if (
            epoch % Math.max(1, Math.floor(params.epochs / 20)) === 0 ||
            epoch === params.epochs - 1
        ) {
            snapshots.push(snapshot(
                snapshots.length,
                "training",
                `Gradient descent — epoch ${epoch + 1}`,
                "Update slope and intercept using the gradients of mean squared error.",
                {
                    regressionLine: { slope, intercept },
                    equation: "w ← w - α(∂MSE/∂w)",
                    metrics: {
                        epoch: epoch + 1,
                        mse: error,
                        slope,
                        intercept,
                    },
                }
            ));
        }

        if (Math.abs(gradientW) + Math.abs(gradientB) < params.tolerance) {
            break;
        }
    }

    const predictions = points.map(
        p => slope * p.x + intercept
    );

    const actual = points.map(p => p.y);

    const residual = mse(actual, predictions);

    const total = variance(actual) * actual.length;

    const r2 = total === 0
        ? 0
        : 1 - (residual * actual.length) / total;

    snapshots.push(snapshot(
        snapshots.length,
        "evaluation",
        "Evaluate regression model",
        "Measure the final prediction error and coefficient of determination.",
        {
            regressionLine: { slope, intercept },
            equation: `ŷ = ${slope.toFixed(4)}x + ${intercept.toFixed(4)}`,
            metrics: {
                mse: residual,
                r2,
                slope,
                intercept,
            },
        }
    ));

    return makeResult(
        "Linear Regression",
        snapshots,
        "Linear regression learns a line by minimizing mean squared prediction error."
    );
}

// =============================================
// PCA — TWO DIMENSIONS
// =============================================

function runPCA(points: DataPoint[]): AlgorithmResult {
    const snapshots: AlgorithmSnapshot[] = [];

    const mx = mean(points.map(p => p.x));
    const my = mean(points.map(p => p.y));

    const centered = points.map(p => ({
        ...p,
        x: p.x - mx,
        y: p.y - my,
    }));

    snapshots.push(snapshot(
        0,
        "initialization",
        "Center the data",
        "Subtract each feature's mean so the dataset is centered around the origin.",
        {
            equation: "Xcentered = X - μ",
            metrics: { meanX: mx, meanY: my },
        }
    ));

    const xx = mean(centered.map(p => p.x ** 2));
    const yy = mean(centered.map(p => p.y ** 2));
    const xy = mean(centered.map(p => p.x * p.y));

    const trace = xx + yy;
    const discriminant = Math.sqrt(
        (xx - yy) ** 2 + 4 * xy ** 2
    );

    const lambda1 = (trace + discriminant) / 2;
    const lambda2 = (trace - discriminant) / 2;

    let vx = xy;
    let vy = lambda1 - xx;

    if (Math.abs(vx) + Math.abs(vy) < 1e-12) {
        vx = xx >= yy ? 1 : 0;
        vy = xx >= yy ? 0 : 1;
    }

    const norm = Math.hypot(vx, vy);

    vx /= norm;
    vy /= norm;

    const explained = lambda1 + lambda2 > 0
        ? lambda1 / (lambda1 + lambda2)
        : 0;

    snapshots.push(snapshot(
        1,
        "training",
        "Calculate covariance matrix",
        "Compute the covariance matrix to measure relationships between the two features.",
        {
            equation: "Σ = (1/n) XᵀX",
            metrics: {
                covarianceXX: xx,
                covarianceXY: xy,
                covarianceYY: yy,
            },
            details: [
                { name: "Variance X", value: xx.toFixed(4) },
                { name: "Covariance XY", value: xy.toFixed(4) },
                { name: "Variance Y", value: yy.toFixed(4) },
            ],
        }
    ));

    const projection = centered.map(p =>
        p.x * vx + p.y * vy
    );

    snapshots.push(snapshot(
        2,
        "update",
        "Calculate principal components",
        "Find the eigenvalues and eigenvectors of the covariance matrix. The first principal component has the largest eigenvalue.",
        {
            equation: "Σv = λv",
            regressionLine: {
                slope: Math.abs(vx) < 1e-12
                    ? Infinity
                    : vy / vx,
                intercept: my - (vy / vx) * mx,
            },
            metrics: {
                eigenvalue1: lambda1,
                eigenvalue2: lambda2,
                explainedVariance: explained,
            },
            details: [
                { name: "PC1", value: `(${vx.toFixed(4)}, ${vy.toFixed(4)})` },
                { name: "Eigenvalue 1", value: lambda1.toFixed(4) },
                { name: "Eigenvalue 2", value: lambda2.toFixed(4) },
            ],
        }
    ));

    snapshots.push(snapshot(
        3,
        "evaluation",
        "Project observations",
        "Project each centered observation onto the first principal component. This reduces the two-dimensional data to one dimension.",
        {
            equation: "z = Xcentered · v₁",
            metrics: {
                explainedVariance: explained,
                projectedDimensions: 1,
                originalDimensions: 2,
                projectionMean: mean(projection),
            },
            details: [
                { name: "Explained variance", value: `${(explained * 100).toFixed(2)}%` },
                { name: "Reduced dimensions", value: "2 → 1" },
            ],
        }
    ));

    return makeResult(
        "PCA",
        snapshots,
        "PCA transforms correlated features into orthogonal principal components ordered by explained variance."
    );
}

// =============================================
// LINEAR SVM
// =============================================

function runSVM(
    points: DataPoint[],
    params: AlgorithmParameters,
    seed: number
): AlgorithmResult {
    const { train, test } = trainTestSplit(points, seed);

    const snapshots: AlgorithmSnapshot[] = [];

    const labels = [...new Set(train.map(p => p.label))];

    if (labels.length !== 2) {
        return makeResult(
            "SVM",
            [snapshot(
                0,
                "evaluation",
                "SVM requires two classes",
                "This educational implementation uses a binary linear SVM. Select a two-class dataset.",
                { metrics: { classes: labels.length } }
            )],
            "Binary linear SVM requires exactly two classes."
        );
    }

    const positive = labels[1];

    let w1 = 0;
    let w2 = 0;
    let bias = 0;

    const signed = (p: DataPoint) =>
        p.label === positive ? 1 : -1;

    const lr = params.learningRate;
    const lambda = params.regularization;

    snapshots.push(snapshot(
        0,
        "initialization",
        "Initialize linear SVM",
        "Initialize the separating hyperplane and prepare binary class labels.",
        {
            equation: "f(x) = wᵀx + b",
            metrics: {
                trainingSamples: train.length,
                testingSamples: test.length,
            },
        }
    ));

    for (let epoch = 0; epoch < params.epochs; epoch++) {
        let loss = 0;

        train.forEach(point => {
            const y = signed(point);
            const margin = y * (w1 * point.x + w2 * point.y + bias);

            const oldW1 = w1;
            const oldW2 = w2;

            if (margin < 1) {
                w1 = oldW1 - lr * (lambda * oldW1 - y * point.x);
                w2 = oldW2 - lr * (lambda * oldW2 - y * point.y);
                bias += lr * y;
                loss += 1 - margin;
            } else {
                w1 = oldW1 * (1 - lr * lambda);
                w2 = oldW2 * (1 - lr * lambda);
            }
        });

        if (
            epoch % Math.max(1, Math.floor(params.epochs / 20)) === 0 ||
            epoch === params.epochs - 1
        ) {
            snapshots.push(snapshot(
                snapshots.length,
                "training",
                `Optimize margin — epoch ${epoch + 1}`,
                "Update the weight vector using the hinge-loss objective and L2 regularization.",
                {
                    equation: "L = max(0, 1 - y(wᵀx+b)) + λ||w||²/2",
                    regressionLine: {
                        slope: Math.abs(w2) < 1e-12 ? Infinity : -w1 / w2,
                        intercept: Math.abs(w2) < 1e-12 ? Infinity : -bias / w2,
                    },
                    metrics: {
                        epoch: epoch + 1,
                        hingeLoss: loss / train.length,
                        weightX: w1,
                        weightY: w2,
                        bias,
                    },
                }
            ));
        }
    }

    const predictions = test.map(p =>
        w1 * p.x + w2 * p.y + bias >= 0
            ? positive
            : labels[0]
    );

    const score = accuracy(
        test.map(p => p.label),
        predictions
    );

    snapshots.push(snapshot(
        snapshots.length,
        "evaluation",
        "Evaluate SVM",
        "Evaluate the linear decision boundary on unseen observations.",
        {
            regressionLine: {
                slope: Math.abs(w2) < 1e-12 ? Infinity : -w1 / w2,
                intercept: Math.abs(w2) < 1e-12 ? Infinity : -bias / w2,
            },
            predictions,
            equation: "w₁x₁ + w₂x₂ + b = 0",
            metrics: {
                accuracy: score,
                margin: 2 / Math.max(Math.hypot(w1, w2), 1e-12),
                weightNorm: Math.hypot(w1, w2),
            },
        }
    ));

    return makeResult(
        "SVM",
        snapshots,
        "Linear SVM learns a maximum-margin hyperplane using hinge loss and L2 regularization."
    );
}

// =============================================
// PUBLIC ENGINE ENTRY
// =============================================

export function runAlgorithm(
    input: AlgorithmInput
): AlgorithmResult {
    const params = safeParameters(input.parameters);

    if (!input.points.length) {
        throw new Error("The dataset is empty.");
    }

    switch (input.algorithm) {
        case "K-Means":
            return runKMeans(input.points, params, input.seed);

        case "CART":
            return runTree(input.points, params, input.seed, "CART");

        case "C4.5":
            return runTree(input.points, params, input.seed, "C4.5");

        case "KNN":
            return runKNN(input.points, params, input.seed);

        case "SVM":
            return runSVM(input.points, params, input.seed);

        case "Naive Bayes":
            return runNaiveBayes(input.points, input.seed);

        case "Linear Regression":
            return runLinearRegression(input.points, params);

        case "PCA":
            return runPCA(input.points);
    }
}
