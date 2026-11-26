
export type DatasetName =
    | "blobs"
    | "moons"
    | "circles"
    | "spiral"
    | "linear"
    | "xor";

export interface DataPoint {
    id: number;
    x: number;
    y: number;
    label: number;
}

export interface DatasetConfig {
    type: DatasetName;
    samples: number;
    noise: number;
    seed: number;
}

export interface Dataset {
    name: DatasetName;
    points: DataPoint[];
    samples: number;
    noise: number;
    seed: number;
}

// =========================================
// RANDOM GENERATOR
// =========================================

function createRandom(seed: number) {
    let state = seed >>> 0;

    return () => {
        state += 0x6D2B79F5;

        let t = state;

        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);

        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function gaussian(random: () => number): number {
    const u = Math.max(random(), 1e-10);
    const v = random();

    return (
        Math.sqrt(-2 * Math.log(u)) *
        Math.cos(2 * Math.PI * v)
    );
}

// =========================================
// DATASET GENERATOR
// =========================================

export function generateDataset(
    config: DatasetConfig
): Dataset {
    const { type, samples, noise, seed } = config;

    const random = createRandom(seed);

    const points: DataPoint[] = [];

    const safeSamples = Math.max(1, Math.floor(samples));
    const safeNoise = Math.max(0, noise);

    function addPoint(
        x: number,
        y: number,
        label: number
    ) {
        points.push({
            id: points.length,
            x: x + gaussian(random) * safeNoise,
            y: y + gaussian(random) * safeNoise,
            label,
        });
    }

    // =======================================
    // GAUSSIAN BLOBS
    // =======================================

    if (type === "blobs") {
        const centers = [
            [-0.55, -0.45],
            [0.55, 0.45],
            [-0.55, 0.55],
        ];

        for (let i = 0; i < safeSamples; i++) {
            const label = i % centers.length;
            const [cx, cy] = centers[label];

            addPoint(
                cx + gaussian(random) * 0.13,
                cy + gaussian(random) * 0.13,
                label
            );
        }
    }

    // =======================================
    // TWO MOONS
    // =======================================

    else if (type === "moons") {
        for (let i = 0; i < safeSamples; i++) {
            const label = i % 2;
            const t = random() * Math.PI;

            const x =
                label === 0
                    ? Math.cos(t) * 0.7 - 0.35
                    : 0.35 - Math.cos(t) * 0.7;

            const y =
                label === 0
                    ? Math.sin(t) * 0.5
                    : 0.15 - Math.sin(t) * 0.5;

            addPoint(x, y, label);
        }
    }

    // =======================================
    // CONCENTRIC CIRCLES
    // =======================================

    else if (type === "circles") {
        for (let i = 0; i < safeSamples; i++) {
            const label = i % 2;
            const angle = random() * Math.PI * 2;

            const radius =
                label === 0
                    ? 0.35
                    : 0.75;

            addPoint(
                Math.cos(angle) * radius,
                Math.sin(angle) * radius,
                label
            );
        }
    }

    // =======================================
    // SPIRAL
    // =======================================

    else if (type === "spiral") {
        for (let i = 0; i < safeSamples; i++) {
            const label = i % 2;

            const index = Math.floor(i / 2);
            const pairCount = Math.ceil(safeSamples / 2);

            const t = pairCount > 1
                ? index / (pairCount - 1)
                : 0;

            const radius = 0.08 + t * 0.78;
            const angle = t * Math.PI * 3 + label * Math.PI;

            addPoint(
                Math.cos(angle) * radius,
                Math.sin(angle) * radius,
                label
            );
        }
    }

    // =======================================
    // LINEAR CLASSIFICATION
    // =======================================

    else if (type === "linear") {
        for (let i = 0; i < safeSamples; i++) {
            const x = random() * 2 - 1;
            const y = random() * 2 - 1;

            const label = x + y > 0 ? 1 : 0;

            addPoint(x, y, label);
        }
    }

    // =======================================
    // XOR
    // =======================================

    else if (type === "xor") {
        for (let i = 0; i < safeSamples; i++) {
            const x = random() * 2 - 1;
            const y = random() * 2 - 1;

            const label = x * y > 0 ? 1 : 0;

            addPoint(x, y, label);
        }
    }

    // =======================================
    // RETURN DATASET
    // =======================================

    return {
        name: type,
        points,
        samples: safeSamples,
        noise: safeNoise,
        seed,
    };
}
