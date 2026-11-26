
export type DatasetName =
    | "blobs"
    | "moons"
    | "circles"
    | "spiral"
    | "linear"
    | "xor";

export interface DataPoint {
    x: number;
    y: number;
    label: number;
    id: number;
}

export interface DatasetConfig {
    type: DatasetName;
    samples: number;
    noise: number;
    seed: number;
}

export interface Dataset {
    points: DataPoint[];
    name: DatasetName;
    samples: number;
    noise: number;
    seed: number;
}

// Deterministic random number generator
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

function gaussian(random: () => number) {
    const u = Math.max(random(), 1e-10);
    const v = random();

    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function generateDataset(config: DatasetConfig): Dataset {
    const { type, samples, noise, seed } = config;

    const random = createRandom(seed);
    const points: DataPoint[] = [];

    const addPoint = (x: number, y: number, label: number) => {
        points.push({
            x: x + gaussian(random) * noise,
            y: y + gaussian(random) * noise,
            label,
            id: points.length,
        });
    };

    switch (type) {
        case "blobs": {
            const centers = [
                [-0.65, -0.55],
                [0.65, 0.55],
                [-0.65, 0.65],
            ];

            for (let i = 0; i < samples; i++) {
                const label = i % centers.length;
                const [cx, cy] = centers[label];

                addPoint(
                    cx + gaussian(random) * 0.13,
                    cy + gaussian(random) * 0.13,
                    label
                );
            }

            break;
        }

        case "moons": {
            for (let i = 0; i < samples; i++) {
                const label = i % 2;
                const t = random() * Math.PI;

                const x = label === 0
                    ? Math.cos(t) * 0.75
                    : 1 - Math.cos(t) * 0.75;

                const y = label === 0
                    ? Math.sin(t) * 0.55
                    : 0.5 - Math.sin(t) * 0.55;

                addPoint(
                    x - 0.5,
                    y - 0.2,
                    label
                );
            }

            break;
        }

        case "circles": {
            for (let i = 0; i < samples; i++) {
                const label = i % 2;
                const angle = random() * Math.PI * 2;

                const radius = label === 0
                    ? 0.3
                    : 0.75;

                addPoint(
                    Math.cos(angle) * radius,
                    Math.sin(angle) * radius,
                    label
                );
            }

            break;
        }

        case "spiral": {
            for (let i = 0; i < samples; i++) {
                const label = i % 2;
                const t = (i / samples) * 3 * Math.PI;
                const radius = 0.1 + (i / samples) * 0.8;
                const angle = t + label * Math.PI;

                addPoint(
                    Math.cos(angle) * radius,
                    Math.sin(angle) * radius,
                    label
                );
            }

            break;
        }

        case "linear": {
            for (let i = 0; i < samples; i++) {
                const x = random() * 2 - 1;
                const y = random() * 2 - 1;

                const label = x + y > 0 ? 1 : 0;

                addPoint(x, y, label);
            }

            break;
        }

        case "xor": {
            for (let i = 0; i < samples; i++) {
                const x = random() * 2 - 1;
                const y = random() * 2 - 1;

                const label = x * y > 0 ? 1 : 0;

                addPoint(x, y, label);
            }

            break;
        }
    }

    return {
        points,
        name: type,
        samples,
        noise,
        seed,
    };
}
