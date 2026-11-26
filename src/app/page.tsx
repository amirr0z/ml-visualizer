
"use client";

import { useEffect, useMemo, useState } from "react";
import { useTheme } from "next-themes";

import {
  BrainCircuit,
  Layers3,
  Sun,
  Moon,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw,
  PanelLeftClose,
  PanelLeft,
  Settings2,
  CircleHelp,
  Shuffle,
  SlidersHorizontal,
  Database,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";

import ScatterPlot from "@/components/visualizations/ScatterPlot";

import {
  generateDataset,
  type DatasetName,
} from "@/lib/datasets/generateDataset";

import {
  runAlgorithm,
} from "@/lib/algorithms/engine";

import {
  DEFAULT_PARAMETERS,
  type AlgorithmName,
  type AlgorithmParameters,
} from "@/lib/algorithms/types";

// =============================================
// CONFIGURATION
// =============================================

const groups = [
  {
    name: "Classification",
    algorithms: [
      "CART",
      "C4.5",
      "KNN",
      "SVM",
      "Naive Bayes",
    ],
  },
  {
    name: "Clustering",
    algorithms: ["K-Means"],
  },
  {
    name: "Regression",
    algorithms: ["Linear Regression"],
  },
  {
    name: "Dimensionality Reduction",
    algorithms: ["PCA"],
  },
];

const datasets = [
  { name: "blobs", label: "Blobs", description: "Gaussian clusters" },
  { name: "moons", label: "Moons", description: "Nonlinear separation" },
  { name: "circles", label: "Circles", description: "Concentric rings" },
  { name: "spiral", label: "Spiral", description: "Interlocking spirals" },
  { name: "linear", label: "Linear", description: "Linear separation" },
  { name: "xor", label: "XOR", description: "Exclusive OR" },
] as const;

const classificationAlgorithms: AlgorithmName[] = [
  "CART",
  "C4.5",
  "KNN",
  "SVM",
  "Naive Bayes",
];

// =============================================
// COMPONENT
// =============================================

export default function Home() {
  const { theme, setTheme } = useTheme();

  const [mounted, setMounted] = useState(false);

  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [algorithm, setAlgorithm] =
    useState<AlgorithmName>("K-Means");

  const [dataset, setDataset] =
    useState<DatasetName>("blobs");

  const [samples, setSamples] = useState(200);
  const [noise, setNoise] = useState(0.05);
  const [seed, setSeed] = useState(42);

  const [parameters, setParameters] =
    useState<AlgorithmParameters>(DEFAULT_PARAMETERS);

  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // =========================================
  // DATASET
  // =========================================

  const data = useMemo(() => {
    return generateDataset({
      type: dataset,
      samples,
      noise,
      seed,
    });
  }, [dataset, samples, noise, seed]);

  // =========================================
  // ALGORITHM EXECUTION
  // =========================================

  const result = useMemo(() => {
    return runAlgorithm({
      algorithm,
      points: data.points,
      parameters,
      seed,
    });
  }, [algorithm, data, parameters, seed]);

  const totalSteps = result.snapshots.length;

  const currentSnapshot =
    result.snapshots[
    Math.min(currentStep, totalSteps - 1)
    ];

  // =========================================
  // PLAYBACK
  // =========================================

  useEffect(() => {
    if (!playing) return;

    if (currentStep >= totalSteps - 1) {
      setPlaying(false);
      return;
    }

    const timer = window.setTimeout(() => {
      setCurrentStep(step => Math.min(step + 1, totalSteps - 1));
    }, 1000);

    return () => window.clearTimeout(timer);
  }, [playing, currentStep, totalSteps]);

  // =========================================
  // HELPERS
  // =========================================

  function updateParameter<K extends keyof AlgorithmParameters>(
    key: K,
    value: AlgorithmParameters[K]
  ) {
    setParameters(previous => ({
      ...previous,
      [key]: value,
    }));

    setCurrentStep(0);
    setPlaying(false);
  }

  function resetExecution() {
    setCurrentStep(0);
    setPlaying(false);
  }

  function regenerateDataset() {
    setSeed(value => value + 1);
    resetExecution();
  }

  function selectAlgorithm(value: string) {
    setAlgorithm(value as AlgorithmName);
    resetExecution();
  }

  function nextStep() {
    setCurrentStep(value => Math.min(value + 1, totalSteps - 1));
  }

  function previousStep() {
    setCurrentStep(value => Math.max(value - 1, 0));
  }

  function togglePlay() {
    if (currentStep >= totalSteps - 1) {
      setCurrentStep(0);
    }

    setPlaying(value => !value);
  }

  const showLabels =
    classificationAlgorithms.includes(algorithm);

  const selectedDataset = datasets.find(
    d => d.name === dataset
  );

  // =========================================
  // RENDER
  // =========================================

  return (
    <main className="app-shell">

      {/* MOBILE BACKDROP */}

      {sidebarOpen && (
        <button
          className="sidebar-backdrop"
          aria-label="Close sidebar"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* SIDEBAR */}

      <aside className={`app-sidebar ${sidebarOpen ? "" : "collapsed"
        }`}>

        <div
          className="h-[66px] px-5 flex items-center justify-between border-b shrink-0"
          style={{ borderColor: "var(--border)" }}
        >

          <div className="flex items-center gap-3">

            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center"
              style={{
                background: "var(--accent-soft)",
                color: "var(--accent)",
              }}
            >
              <BrainCircuit size={21} />
            </div>

            <div>
              <div className="text-sm font-semibold">
                ML Laboratory
              </div>

              <div
                className="text-[9px] tracking-[1.6px] uppercase mt-0.5"
                style={{ color: "var(--text-muted)" }}
              >
                Interactive Learning
              </div>
            </div>

          </div>

          <button
            className="icon-button md:hidden"
            onClick={() => setSidebarOpen(false)}
          >
            <PanelLeftClose size={17} />
          </button>

        </div>

        <div className="flex-1 overflow-y-auto">

          <div className="sidebar-section">

            <div className="sidebar-label">
              Workspace
            </div>

            <button className="algorithm-item active">
              <Layers3 size={16} className="mr-3" />
              Algorithm Explorer
            </button>

          </div>

          <div className="sidebar-section">

            <div className="sidebar-label">
              Algorithms
            </div>

            {groups.map(group => (

              <div key={group.name} className="algorithm-group">

                <div className="algorithm-group-title">
                  {group.name}
                </div>

                {group.algorithms.map(name => (

                  <button
                    key={name}
                    onClick={() => selectAlgorithm(name)}
                    className={`algorithm-item ${algorithm === name ? "active" : ""
                      }`}
                  >
                    {name}
                  </button>

                ))}

              </div>

            ))}

          </div>

        </div>

        <div
          className="px-5 py-4 border-t flex items-center gap-2 text-[11px] shrink-0"
          style={{
            borderColor: "var(--border)",
            color: "var(--text-muted)",
          }}
        >
          <span className="w-2 h-2 rounded-full bg-emerald-400" />
          Local computation
        </div>

      </aside>

      {/* MAIN */}

      <section className="app-main">

        {/* HEADER */}

        <header className="app-header">

          <div className="flex items-center gap-4 min-w-0">

            <button
              className="icon-button"
              onClick={() => setSidebarOpen(v => !v)}
            >
              {sidebarOpen
                ? <PanelLeftClose size={19} />
                : <PanelLeft size={19} />
              }
            </button>

            <div className="min-w-0">

              <div className="text-sm font-medium truncate">
                {algorithm}
              </div>

              <div
                className="text-[10px] mt-0.5"
                style={{ color: "var(--text-muted)" }}
              >
                Interactive algorithm workspace
              </div>

            </div>

          </div>

          <div className="flex items-center gap-2">

            <div
              className="hidden sm:flex items-center gap-2 text-[11px] mr-2"
              style={{ color: "var(--text-muted)" }}
            >
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              Browser environment
            </div>

            <button
              className="icon-button"
              onClick={() => setTheme(
                theme === "dark" ? "light" : "dark"
              )}
              aria-label="Toggle theme"
            >
              {mounted
                ? theme === "dark"
                  ? <Sun size={19} />
                  : <Moon size={19} />
                : <Sun size={19} />
              }
            </button>

            <button className="icon-button" title="Help">
              <CircleHelp size={18} />
            </button>

            <button className="icon-button" title="Settings">
              <Settings2 size={18} />
            </button>

          </div>

        </header>

        {/* WORKSPACE */}

        <div className="app-content">

          <div className="workspace">

            <div className="workspace-main">

              {/* VISUALIZATION */}

              <section className="panel visualization-panel">

                <div className="panel-header">

                  <div>
                    <div className="panel-title">
                      Visualization
                    </div>

                    <div className="panel-description">
                      {selectedDataset?.label} · {algorithm}
                    </div>
                  </div>

                  <button
                    className="text-button"
                    onClick={regenerateDataset}
                  >
                    <Shuffle size={14} />
                    Regenerate
                  </button>

                </div>

                <div className="visualization-body">

                  <ScatterPlot
                    points={data.points}
                    showLabels={showLabels}
                    assignments={currentSnapshot?.assignments}
                    centroids={currentSnapshot?.centroids}
                    regressionLine={currentSnapshot?.regressionLine}
                    decisionSplits={currentSnapshot?.decisionSplits}
                    highlightedPointIds={
                      currentSnapshot?.highlightedPointIds
                    }
                    title={selectedDataset?.label}
                  />

                  <div
                    className="flex justify-between items-center flex-wrap gap-2 mt-3 text-[10px]"
                    style={{ color: "var(--text-muted)" }}
                  >

                    <div className="flex gap-3 flex-wrap">
                      <span>{samples} samples</span>
                      <span>2 features</span>
                      <span>Seed: {seed}</span>
                    </div>

                    <span>
                      {currentSnapshot?.phase ?? "Ready"}
                    </span>

                  </div>

                </div>

              </section>

              {/* PARAMETERS */}

              <aside className="panel control-panel">

                <div className="flex items-center justify-between mb-4">

                  <div className="panel-title">
                    Dataset Generator
                  </div>

                  <Database
                    size={16}
                    style={{ color: "var(--accent)" }}
                  />

                </div>

                <div className="grid grid-cols-2 gap-2">

                  {datasets.map(item => (

                    <button
                      key={item.name}
                      onClick={() => {
                        setDataset(item.name);
                        resetExecution();
                      }}
                      className={`dataset-option ${dataset === item.name ? "active" : ""
                        }`}
                    >

                      <div className="dataset-option-name">
                        {item.label}
                      </div>

                      <div className="dataset-option-description">
                        {item.description}
                      </div>

                    </button>

                  ))}

                </div>

                {/* DATASET PARAMETERS */}

                <div className="control-section">

                  <div className="flex items-center gap-2 mb-4">
                    <SlidersHorizontal
                      size={14}
                      style={{ color: "var(--accent)" }}
                    />
                    <div className="text-xs font-semibold">
                      Dataset Parameters
                    </div>
                  </div>

                  <div className="mb-6">

                    <label className="control-label">
                      <span>Number of samples</span>
                      <span className="control-value">
                        {samples}
                      </span>
                    </label>

                    <input
                      type="range"
                      min="50"
                      max="1000"
                      step="50"
                      value={samples}
                      onChange={e => {
                        setSamples(Number(e.target.value));
                        resetExecution();
                      }}
                    />

                  </div>

                  <div>

                    <label className="control-label">
                      <span>Noise</span>
                      <span className="control-value">
                        {noise.toFixed(2)}
                      </span>
                    </label>

                    <input
                      type="range"
                      min="0"
                      max="0.5"
                      step="0.01"
                      value={noise}
                      onChange={e => {
                        setNoise(Number(e.target.value));
                        resetExecution();
                      }}
                    />

                  </div>

                </div>

                {/* ALGORITHM PARAMETERS */}

                <div className="control-section">

                  <div className="text-xs font-semibold mb-4">
                    {algorithm} Parameters
                  </div>

                  {algorithm === "K-Means" && (

                    <>
                      <label className="control-label">
                        <span>Number of clusters (K)</span>
                        <span className="control-value">
                          {parameters.clusters}
                        </span>
                      </label>

                      <input
                        type="range"
                        min="2"
                        max="8"
                        value={parameters.clusters}
                        onChange={e => updateParameter(
                          "clusters",
                          Number(e.target.value)
                        )}
                      />

                      <div className="mt-4">

                        <label className="control-label">
                          Initialization
                        </label>

                        <select
                          className="w-full rounded-lg p-2 text-xs border"
                          style={{
                            background: "var(--panel-secondary)",
                            borderColor: "var(--border)",
                            color: "var(--text-primary)",
                          }}
                          value={parameters.initialization}
                          onChange={e => updateParameter(
                            "initialization",
                            e.target.value as "random" | "kmeans++"
                          )}
                        >
                          <option value="random">Random</option>
                          <option value="kmeans++">K-Means++</option>
                        </select>

                      </div>
                    </>

                  )}

                  {algorithm === "KNN" && (
                    <>
                      <label className="control-label">
                        <span>Neighbors (K)</span>
                        <span className="control-value">
                          {parameters.neighbors}
                        </span>
                      </label>
                      <input
                        type="range"
                        min="1"
                        max="20"
                        value={parameters.neighbors}
                        onChange={e => updateParameter(
                          "neighbors",
                          Number(e.target.value)
                        )}
                      />
                    </>
                  )}

                  {(algorithm === "CART" || algorithm === "C4.5") && (
                    <>
                      <label className="control-label">
                        <span>Maximum tree depth</span>
                        <span className="control-value">
                          {parameters.maxDepth}
                        </span>
                      </label>
                      <input
                        type="range"
                        min="1"
                        max="8"
                        value={parameters.maxDepth}
                        onChange={e => updateParameter(
                          "maxDepth",
                          Number(e.target.value)
                        )}
                      />
                    </>
                  )}

                  {(algorithm === "SVM" ||
                    algorithm === "Linear Regression") && (
                      <>
                        <label className="control-label">
                          <span>Learning rate</span>
                          <span className="control-value">
                            {parameters.learningRate.toFixed(3)}
                          </span>
                        </label>
                        <input
                          type="range"
                          min="0.001"
                          max="0.1"
                          step="0.001"
                          value={parameters.learningRate}
                          onChange={e => updateParameter(
                            "learningRate",
                            Number(e.target.value)
                          )}
                        />

                        <div className="mt-4">
                          <label className="control-label">
                            <span>Training epochs</span>
                            <span className="control-value">
                              {parameters.epochs}
                            </span>
                          </label>
                          <input
                            type="range"
                            min="10"
                            max="200"
                            step="10"
                            value={parameters.epochs}
                            onChange={e => updateParameter(
                              "epochs",
                              Number(e.target.value)
                            )}
                          />
                        </div>
                      </>
                    )}

                </div>

                {/* DATASET INFORMATION */}

                <div className="control-section">

                  <div className="text-xs font-semibold mb-3">
                    Dataset Information
                  </div>

                  {[
                    ["Dataset", selectedDataset?.label ?? dataset],
                    ["Samples", samples],
                    ["Features", 2],
                    ["Noise", noise.toFixed(2)],
                    ["Seed", seed],
                  ].map(([label, value]) => (

                    <div
                      key={label}
                      className="flex justify-between py-2 border-b text-[11px]"
                      style={{ borderColor: "var(--border)" }}
                    >
                      <span style={{ color: "var(--text-muted)" }}>
                        {label}
                      </span>
                      <span>{value}</span>
                    </div>

                  ))}

                </div>

              </aside>

            </div>

            {/* EXECUTION PANEL */}

            <section className="panel execution-panel">

              <div className="flex items-center justify-between gap-3">

                <div>
                  <div className="panel-title">
                    Algorithm Execution
                  </div>

                  <div className="panel-description">
                    Step {currentStep + 1} / {totalSteps}
                  </div>
                </div>

                <button
                  className="text-button"
                  onClick={resetExecution}
                >
                  <RotateCcw size={14} />
                  Reset
                </button>

              </div>

              {/* PROGRESS */}

              <input
                type="range"
                min="0"
                max={Math.max(0, totalSteps - 1)}
                value={currentStep}
                onChange={e => {
                  setCurrentStep(Number(e.target.value));
                  setPlaying(false);
                }}
                className="w-full mt-4"
                aria-label="Execution timeline"
              />

              <div className="flex items-center justify-between text-[10px] mt-1"
                style={{ color: "var(--text-muted)" }}
              >
                <span>{result.snapshots[0]?.title}</span>
                <span>{result.snapshots[totalSteps - 1]?.title}</span>
              </div>

              {/* PLAYBACK */}

              <div className="flex items-center justify-center gap-3 mt-3">

                <button
                  className="execution-button"
                  onClick={() => {
                    setCurrentStep(0);
                    setPlaying(false);
                  }}
                  aria-label="First step"
                >
                  <SkipBack size={18} />
                </button>

                <button
                  className="execution-button"
                  onClick={previousStep}
                  disabled={currentStep === 0}
                  aria-label="Previous step"
                >
                  <ChevronLeft size={18} />
                </button>

                <button
                  className="execution-button primary"
                  onClick={togglePlay}
                  aria-label={playing ? "Pause" : "Play"}
                >
                  {playing
                    ? <Pause size={18} />
                    : <Play size={18} fill="currentColor" />
                  }
                </button>

                <button
                  className="execution-button"
                  onClick={nextStep}
                  disabled={currentStep >= totalSteps - 1}
                  aria-label="Next step"
                >
                  <ChevronRight size={18} />
                </button>

                <button
                  className="execution-button"
                  onClick={() => {
                    setCurrentStep(totalSteps - 1);
                    setPlaying(false);
                  }}
                  aria-label="Last step"
                >
                  <SkipForward size={18} />
                </button>

              </div>

              {/* CURRENT STEP EXPLANATION */}

              <div
                className="mt-4 pt-3 border-t"
                style={{ borderColor: "var(--border)" }}
              >

                <div className="flex justify-between items-start gap-3">

                  <div>
                    <div className="text-xs font-semibold">
                      {currentSnapshot?.title}
                    </div>

                    <p
                      className="text-[11px] leading-relaxed mt-1"
                      style={{ color: "var(--text-secondary)" }}
                    >
                      {currentSnapshot?.description}
                    </p>
                  </div>

                  <span
                    className="text-[10px] shrink-0 uppercase"
                    style={{ color: "var(--accent)" }}
                  >
                    {currentSnapshot?.phase}
                  </span>

                </div>

                {currentSnapshot?.equation && (
                  <div
                    className="mt-3 p-3 rounded-lg text-center font-mono text-xs"
                    style={{
                      background: "var(--panel-secondary)",
                      color: "var(--accent)",
                    }}
                  >
                    {currentSnapshot.equation}
                  </div>
                )}

                {/* METRICS */}

                <div className="flex flex-wrap gap-3 mt-3">

                  {Object.entries(currentSnapshot?.metrics ?? {}).map(
                    ([key, value]) => (

                      <div
                        key={key}
                        className="rounded-lg px-3 py-2"
                        style={{ background: "var(--panel-secondary)" }}
                      >

                        <div
                          className="text-[9px] uppercase"
                          style={{ color: "var(--text-muted)" }}
                        >
                          {key.replace(/([A-Z])/g, " $1")}
                        </div>

                        <div className="text-xs font-semibold mt-1 tabular-nums">
                          {Number.isFinite(value)
                            ? Number(value).toFixed(4)
                            : value}
                        </div>

                      </div>

                    )
                  )}

                </div>

                {/* DETAILS */}

                {currentSnapshot?.details?.length ? (

                  <div className="grid sm:grid-cols-2 gap-2 mt-3">

                    {currentSnapshot.details.map((detail, index) => (

                      <div
                        key={index}
                        className="flex justify-between gap-2 text-[10px] p-2 rounded"
                        style={{ background: "var(--panel-secondary)" }}
                      >
                        <span style={{ color: "var(--text-muted)" }}>
                          {detail.name}
                        </span>

                        <span className="text-right">
                          {detail.value}
                        </span>
                      </div>

                    ))}

                  </div>

                ) : null}

              </div>

            </section>

          </div>

        </div>

      </section>

    </main>
  );
}
