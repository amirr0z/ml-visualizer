
"use client";

import { useEffect, useMemo, useState } from "react";
import {
  BrainCircuit, Layers3, Activity, Sun, Moon,
  Play, Pause, SkipBack, SkipForward, RotateCcw,
  PanelLeftClose, Settings2, CircleHelp, Shuffle
} from "lucide-react";

import ScatterPlot from "@/components/visualizations/ScatterPlot";
import {
  generateDataset,
  DatasetName
} from "@/lib/datasets/generateDataset";

import { useTheme } from "next-themes";


const algorithms = [
  { name: "CART", group: "Classification" },
  { name: "C4.5", group: "Classification" },
  { name: "KNN", group: "Classification" },
  { name: "SVM", group: "Classification" },
  { name: "Naive Bayes", group: "Classification" },
  { name: "K-Means", group: "Clustering" },
  { name: "Linear Regression", group: "Regression" },
  { name: "PCA", group: "Dimensionality Reduction" },
];

const datasets: {
  name: DatasetName;
  label: string;
  description: string;
}[] = [
    { name: "blobs", label: "Blobs", description: "Gaussian clusters" },
    { name: "moons", label: "Moons", description: "Nonlinear separation" },
    { name: "circles", label: "Circles", description: "Concentric rings" },
    { name: "spiral", label: "Spiral", description: "Interlocking spirals" },
    { name: "linear", label: "Linear", description: "Linear separation" },
    { name: "xor", label: "XOR", description: "Exclusive OR pattern" },
  ];

const executionSteps = [
  "Initialize algorithm",
  "Process dataset",
  "Calculate model",
  "Generate predictions",
  "Evaluate results",
];

export default function Home() {
  const [algorithm, setAlgorithm] = useState("K-Means");
  const [dataset, setDataset] = useState<DatasetName>("blobs");

  const [samples, setSamples] = useState(200);
  const [noise, setNoise] = useState(0.05);
  const [seed, setSeed] = useState(42);

  const { theme, setTheme } = useTheme();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [playing, setPlaying] = useState(false);
  const [step, setStep] = useState(0);

  const [clusters, setClusters] = useState(3);

  useEffect(() => {
    const saved = localStorage.getItem("ml-theme");

    if (saved === "light" || saved === "dark") {
      setTheme(saved);
    }
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("ml-theme", theme);
  }, [theme]);

  const data = useMemo(
    () => generateDataset({
      type: dataset,
      samples,
      noise,
      seed,
    }),
    [dataset, samples, noise, seed]
  );

  const showLabels = !["K-Means", "PCA"].includes(algorithm);

  const resetExecution = () => {
    setStep(0);
    setPlaying(false);
  };

  const selectAlgorithm = (name: string) => {
    setAlgorithm(name);
    resetExecution();
  };

  const selectDataset = (name: DatasetName) => {
    setDataset(name);
    resetExecution();
  };

  const regenerate = () => {
    setSeed(s => s + 1);
    resetExecution();
  };

  const groups = [
    "Classification",
    "Clustering",
    "Regression",
    "Dimensionality Reduction",
  ];

  const currentStep = Math.min(step, executionSteps.length - 1);

  const surface = {
    backgroundColor: "var(--panel-bg)",
    borderColor: "var(--border)",
  };

  return (
    <main
      className="min-h-screen flex"
      style={{
        backgroundColor: "var(--app-bg)",
        color: "var(--text-primary)",
      }}
    >
      {/* Sidebar */}
      {sidebarOpen && (
        <aside
          className="w-[245px] shrink-0 border-r flex flex-col min-h-screen"
          style={{
            backgroundColor: "var(--sidebar-bg)",
            borderColor: "var(--border)",
          }}
        >
          <div
            className="h-[70px] flex items-center gap-3 px-5 border-b"
            style={{ borderColor: "var(--border)" }}
          >
            <div className="h-9 w-9 rounded-xl flex items-center justify-center"
              style={{ backgroundColor: "var(--accent-soft)" }}>
              <BrainCircuit size={21} style={{ color: "var(--accent)" }} />
            </div>

            <div>
              <div className="font-semibold">ML Laboratory</div>
              <div className="text-[10px] tracking-widest uppercase"
                style={{ color: "var(--text-muted)" }}>
                Interactive learning
              </div>
            </div>
          </div>

          <div className="px-4 pt-6">
            <div className="text-[10px] uppercase tracking-widest mb-3"
              style={{ color: "var(--text-muted)" }}>
              Workspace
            </div>

            <div
              className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm"
              style={{
                backgroundColor: "var(--accent-soft)",
                color: "var(--accent)",
              }}
            >
              <Layers3 size={17} />
              Algorithm Explorer
            </div>
          </div>

          <div className="px-4 pt-7 flex-1">
            <div className="text-[10px] uppercase tracking-widest mb-4"
              style={{ color: "var(--text-muted)" }}>
              Algorithms
            </div>

            {groups.map(group => (
              <div key={group} className="mb-5">
                <div className="text-xs mb-2 px-2"
                  style={{ color: "var(--text-muted)" }}>
                  {group}
                </div>

                {algorithms.filter(a => a.group === group).map(a => (
                  <button
                    key={a.name}
                    onClick={() => selectAlgorithm(a.name)}
                    className="w-full text-left px-3 py-2.5 rounded-lg text-sm mb-1 border transition-all"
                    style={{
                      backgroundColor: algorithm === a.name
                        ? "var(--accent-soft)"
                        : "transparent",
                      color: algorithm === a.name
                        ? "var(--accent)"
                        : "var(--text-secondary)",
                      borderColor: algorithm === a.name
                        ? "var(--accent)"
                        : "transparent",
                    }}
                  >
                    {a.name}
                  </button>
                ))}
              </div>
            ))}
          </div>

          <div className="p-4 border-t text-xs flex items-center gap-2"
            style={{
              borderColor: "var(--border)",
              color: "var(--text-muted)",
            }}>
            <Activity size={14} style={{ color: "#34d399" }} />
            Browser-side computation
          </div>
        </aside>
      )}

      {/* Main workspace */}
      <section className="flex-1 min-w-0 flex flex-col">
        {/* Header */}
        <header
          className="h-[70px] shrink-0 border-b flex items-center justify-between px-5 lg:px-7"
          style={{
            backgroundColor: "var(--panel-bg)",
            borderColor: "var(--border)",
          }}
        >
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              style={{ color: "var(--text-secondary)" }}
            >
              <PanelLeftClose size={19} />
            </button>

            <div>
              <div className="text-sm font-medium">{algorithm}</div>
              <div className="text-[11px]"
                style={{ color: "var(--text-muted)" }}>
                Interactive algorithm visualization
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="hidden sm:flex items-center gap-2 text-xs"
              style={{ color: "var(--text-muted)" }}>
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              Local execution
            </span>

            <button
              onClick={() => setTheme(t => t === "dark" ? "light" : "dark")}
              className="p-2 rounded-lg hover:bg-black/5"
              title="Toggle theme"
              style={{ color: "var(--text-secondary)" }}
            >
              {theme === "dark" ? <Sun size={19} /> : <Moon size={19} />}
            </button>

            <button
              className="p-2"
              style={{ color: "var(--text-secondary)" }}
            >
              <CircleHelp size={18} />
            </button>

            <button
              className="p-2"
              style={{ color: "var(--text-secondary)" }}
            >
              <Settings2 size={18} />
            </button>
          </div>
        </header>

        <div className="flex-1 p-5 lg:p-7 overflow-auto">
          <div className="max-w-[1500px] mx-auto">

            {/* Title */}
            <div className="mb-6">
              <h1 className="text-xl font-semibold">
                Algorithm Workspace
              </h1>
              <p className="text-xs mt-1"
                style={{ color: "var(--text-muted)" }}>
                Explore how machine learning algorithms work.
              </p>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_290px] gap-5">

              {/* Main visualization */}
              <div
                className="rounded-xl border p-5 min-w-0"
                style={surface}
              >
                <div className="flex justify-between items-center mb-5">
                  <div>
                    <h2 className="text-sm font-medium">Dataset Visualization</h2>
                    <p className="text-xs mt-1"
                      style={{ color: "var(--text-muted)" }}>
                      {datasets.find(d => d.name === dataset)?.description}
                    </p>
                  </div>

                  <button
                    onClick={regenerate}
                    className="flex items-center gap-2 text-xs"
                    style={{ color: "var(--accent)" }}
                  >
                    <Shuffle size={14} />
                    Regenerate
                  </button>
                </div>

                <ScatterPlot
                  points={data.points}
                  showLabels={showLabels}
                  title={dataset.toUpperCase()}
                />

                <div className="mt-4 text-[11px]"
                  style={{ color: "var(--text-muted)" }}>
                  Seed: {seed} · {samples} samples · Noise: {noise.toFixed(2)}
                </div>
              </div>

              {/* Dataset controls */}
              <aside
                className="rounded-xl border p-5"
                style={surface}
              >
                <h2 className="text-sm font-medium mb-5">
                  Dataset Generator
                </h2>

                <div className="grid grid-cols-2 gap-2 mb-6">
                  {datasets.map(d => (
                    <button
                      key={d.name}
                      onClick={() => selectDataset(d.name)}
                      className="text-left rounded-lg p-3 border transition-all"
                      style={{
                        backgroundColor: dataset === d.name
                          ? "var(--accent-soft)"
                          : "var(--panel-secondary)",
                        borderColor: dataset === d.name
                          ? "var(--accent)"
                          : "var(--border)",
                      }}
                    >
                      <div className="text-xs font-medium"
                        style={{
                          color: dataset === d.name
                            ? "var(--accent)"
                            : "var(--text-primary)",
                        }}>
                        {d.label}
                      </div>
                      <div className="text-[10px] mt-1"
                        style={{ color: "var(--text-muted)" }}>
                        {d.description}
                      </div>
                    </button>
                  ))}
                </div>

                <div className="border-t pt-5 space-y-6"
                  style={{ borderColor: "var(--border)" }}>

                  <div>
                    <label className="flex justify-between text-xs mb-3">
                      Number of samples
                      <span style={{ color: "var(--accent)" }}>{samples}</span>
                    </label>
                    <input
                      type="range"
                      min="50"
                      max="1000"
                      step="50"
                      value={samples}
                      onChange={e => setSamples(Number(e.target.value))}
                      className="w-full accent-cyan-500"
                    />
                  </div>

                  <div>
                    <label className="flex justify-between text-xs mb-3">
                      Noise
                      <span style={{ color: "var(--accent)" }}>
                        {noise.toFixed(2)}
                      </span>
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="0.5"
                      step="0.01"
                      value={noise}
                      onChange={e => setNoise(Number(e.target.value))}
                      className="w-full accent-cyan-500"
                    />
                  </div>

                  {algorithm === "K-Means" && (
                    <div>
                      <label className="flex justify-between text-xs mb-3">
                        Number of clusters
                        <span style={{ color: "var(--accent)" }}>{clusters}</span>
                      </label>
                      <input
                        type="range"
                        min="2"
                        max="8"
                        value={clusters}
                        onChange={e => {
                          setClusters(Number(e.target.value));
                          resetExecution();
                        }}
                        className="w-full accent-cyan-500"
                      />
                    </div>
                  )}

                  <div className="border-t pt-5"
                    style={{ borderColor: "var(--border)" }}>
                    <div className="text-xs font-medium mb-3">
                      Dataset information
                    </div>

                    {[
                      ["Type", dataset],
                      ["Samples", samples],
                      ["Features", 2],
                      ["Random seed", seed],
                      ["Algorithm", algorithm],
                    ].map(([label, value]) => (
                      <div
                        key={label}
                        className="flex justify-between py-2 text-xs border-b"
                        style={{ borderColor: "var(--border)" }}
                      >
                        <span style={{ color: "var(--text-muted)" }}>{label}</span>
                        <span>{value}</span>
                      </div>
                    ))}
                  </div>

                </div>
              </aside>
            </div>

            {/* Execution controls */}
            <div
              className="rounded-xl border p-5 mt-5"
              style={surface}
            >
              <div className="flex justify-between items-center mb-5">
                <div>
                  <h2 className="text-sm font-medium">
                    Algorithm Execution
                  </h2>
                  <p className="text-xs mt-1"
                    style={{ color: "var(--text-muted)" }}>
                    Step {step} / {executionSteps.length}
                  </p>
                </div>

                <button
                  onClick={resetExecution}
                  className="flex items-center gap-2 text-xs"
                  style={{ color: "var(--text-secondary)" }}
                >
                  <RotateCcw size={14} />
                  Reset
                </button>
              </div>

              <div className="flex gap-1 mb-5">
                {executionSteps.map((s, i) => (
                  <button
                    key={s}
                    title={s}
                    onClick={() => {
                      setStep(i + 1);
                      setPlaying(false);
                    }}
                    className="h-1 flex-1 rounded-full"
                    style={{
                      backgroundColor: i < step
                        ? "var(--accent)"
                        : "var(--border)",
                    }}
                  />
                ))}
              </div>

              <div className="flex justify-center items-center gap-4">
                <button
                  onClick={() => {
                    setStep(0);
                    setPlaying(false);
                  }}
                  style={{ color: "var(--text-secondary)" }}
                >
                  <SkipBack size={19} />
                </button>

                <button
                  onClick={() => {
                    setStep(s => Math.max(0, s - 1));
                    setPlaying(false);
                  }}
                  style={{ color: "var(--text-secondary)" }}
                >
                  <RotateCcw size={17} />
                </button>

                <button
                  onClick={() => setPlaying(p => !p)}
                  className="w-11 h-11 rounded-full flex items-center justify-center"
                  style={{
                    backgroundColor: "var(--accent)",
                    color: "#07111c",
                  }}
                >
                  {playing
                    ? <Pause size={18} />
                    : <Play size={18} fill="currentColor" />
                  }
                </button>

                <button
                  onClick={() => {
                    setStep(s => Math.min(executionSteps.length, s + 1));
                    setPlaying(false);
                  }}
                  style={{ color: "var(--text-secondary)" }}
                >
                  <SkipForward size={19} />
                </button>
              </div>

              <div className="text-center text-xs mt-4"
                style={{ color: "var(--text-muted)" }}>
                {executionSteps[currentStep]}
              </div>

              <p className="text-center text-[10px] mt-2"
                style={{ color: "var(--text-muted)" }}>
                Execution engine integration is the next development stage.
              </p>
            </div>

          </div>
        </div>
      </section>
    </main>
  );
}
