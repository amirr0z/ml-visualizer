
"use client";

import { useMemo, useState } from "react";
import { useTheme } from "next-themes";
import { useEffect } from "react";

import {
  BrainCircuit,
  Layers3,
  Activity,
  Sun,
  Moon,
  Play,
  SkipBack,
  SkipForward,
  RotateCcw,
  PanelLeftClose,
  PanelLeft,
  Settings2,
  CircleHelp,
  Shuffle,
  ChevronDown,
  SlidersHorizontal,
  Database,
} from "lucide-react";

import ScatterPlot from "@/components/visualizations/ScatterPlot";

import {
  generateDataset,
  type DatasetName,
} from "@/lib/datasets/generateDataset";

// =============================================
// TYPES AND CONFIGURATION
// =============================================

const algorithms = [
  { name: "CART", group: "Classification" },
  { name: "C4.5", group: "Classification" },
  { name: "KNN", group: "Classification" },
  { name: "SVM", group: "Classification" },
  { name: "Naive Bayes", group: "Classification" },

  { name: "K-Means", group: "Clustering" },

  { name: "Linear Regression", group: "Regression" },

  {
    name: "PCA",
    group: "Dimensionality Reduction",
  },
];

const algorithmGroups = [
  "Classification",
  "Clustering",
  "Regression",
  "Dimensionality Reduction",
];

const datasets: {
  name: DatasetName;
  label: string;
  description: string;
}[] = [
    {
      name: "blobs",
      label: "Blobs",
      description: "Gaussian clusters",
    },
    {
      name: "moons",
      label: "Moons",
      description: "Nonlinear separation",
    },
    {
      name: "circles",
      label: "Circles",
      description: "Concentric rings",
    },
    {
      name: "spiral",
      label: "Spiral",
      description: "Interlocking spirals",
    },
    {
      name: "linear",
      label: "Linear",
      description: "Linear separation",
    },
    {
      name: "xor",
      label: "XOR",
      description: "Exclusive OR",
    },
  ];

const executionSteps = [
  "Initialize algorithm",
  "Process dataset",
  "Calculate model",
  "Generate predictions",
  "Evaluate results",
];

// =============================================
// MAIN COMPONENT
// =============================================

export default function Home() {
  // -------------------------------------------
  // THEME
  // -------------------------------------------

  const { theme, setTheme } = useTheme();

  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // -------------------------------------------
  // APPLICATION STATE
  // -------------------------------------------

  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [algorithm, setAlgorithm] = useState("K-Means");

  // -------------------------------------------
  // DATASET STATE
  // -------------------------------------------

  const [dataset, setDataset] =
    useState<DatasetName>("blobs");

  const [samples, setSamples] = useState(200);

  const [noise, setNoise] = useState(0.05);

  const [seed, setSeed] = useState(42);

  // -------------------------------------------
  // ALGORITHM PARAMETERS
  // -------------------------------------------

  const [clusters, setClusters] = useState(3);

  // -------------------------------------------
  // EXECUTION STATE
  // -------------------------------------------

  const [step, setStep] = useState(0);

  // -------------------------------------------
  // DATA GENERATION
  // -------------------------------------------

  const data = useMemo(() => {
    return generateDataset({
      type: dataset,
      samples,
      noise,
      seed,
    });
  }, [dataset, samples, noise, seed]);

  // -------------------------------------------
  // VISUALIZATION SETTINGS
  // -------------------------------------------

  const showLabels = [
    "CART",
    "C4.5",
    "KNN",
    "SVM",
    "Naive Bayes",
  ].includes(algorithm);

  // -------------------------------------------
  // HELPERS
  // -------------------------------------------

  const resetExecution = () => {
    setStep(0);
  };

  const selectAlgorithm = (name: string) => {
    setAlgorithm(name);
    resetExecution();
  };

  const selectDataset = (name: DatasetName) => {
    setDataset(name);
    resetExecution();
  };

  const regenerateDataset = () => {
    setSeed(previous => previous + 1);
    resetExecution();
  };

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  // -------------------------------------------
  // CURRENT DATASET INFORMATION
  // -------------------------------------------

  const currentDataset = datasets.find(
    item => item.name === dataset
  );

  // =============================================
  // RENDER
  // =============================================

  return (
    <main className="app-shell">

      {/* =====================================
          MOBILE SIDEBAR BACKDROP
      ====================================== */}

      {sidebarOpen && (
        <button
          className="sidebar-backdrop"
          aria-label="Close navigation"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* =====================================
          SIDEBAR
      ====================================== */}

      <aside
        className={`app-sidebar ${sidebarOpen ? "" : "collapsed"
          }`}
      >

        {/* Logo */}

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

        {/* Navigation */}

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

            {algorithmGroups.map(group => (

              <div key={group} className="algorithm-group">

                <div className="algorithm-group-title">
                  {group}
                </div>

                {algorithms
                  .filter(item => item.group === group)
                  .map(item => (

                    <button
                      key={item.name}
                      onClick={() => selectAlgorithm(item.name)}
                      className={`algorithm-item ${algorithm === item.name
                          ? "active"
                          : ""
                        }`}
                    >
                      {item.name}
                    </button>

                  ))}

              </div>

            ))}

          </div>

        </div>

        {/* Footer */}

        <div
          className="px-5 py-4 border-t flex items-center gap-2 text-[11px] shrink-0"
          style={{
            borderColor: "var(--border)",
            color: "var(--text-muted)",
          }}
        >
          <span className="w-2 h-2 rounded-full bg-emerald-400" />

          All computations run locally

        </div>

      </aside>

      {/* =====================================
          MAIN CONTENT
      ====================================== */}

      <section className="app-main">

        {/* =================================
            HEADER
        ================================== */}

        <header className="app-header">

          <div className="flex items-center gap-4 min-w-0">

            <button
              className="icon-button"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-label="Toggle sidebar"
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
                className="text-[10px] mt-0.5 truncate"
                style={{ color: "var(--text-muted)" }}
              >
                Interactive algorithm visualization
              </div>

            </div>

          </div>

          <div className="flex items-center gap-2 sm:gap-3 shrink-0">

            <div
              className="hidden sm:flex items-center gap-2 text-[11px]"
              style={{ color: "var(--text-muted)" }}
            >
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              Local environment
            </div>

            {/* Theme toggle */}

            <button
              className="icon-button"
              onClick={toggleTheme}
              title="Toggle theme"
              aria-label="Toggle theme"
            >
              {mounted ? (
                theme === "dark"
                  ? <Sun size={19} />
                  : <Moon size={19} />
              ) : (
                <Sun size={19} />
              )}
            </button>

            <button
              className="icon-button"
              title="Help"
              aria-label="Help"
            >
              <CircleHelp size={18} />
            </button>

            <button
              className="icon-button"
              title="Settings"
              aria-label="Settings"
            >
              <Settings2 size={18} />
            </button>

          </div>

        </header>

        {/* =================================
            WORKSPACE
        ================================== */}

        <div className="app-content">

          <div className="workspace">

            {/* =================================
                MAIN VISUALIZATION + CONTROLS
            ================================== */}

            <div className="workspace-main">

              {/* ---------------------------------
                  VISUALIZATION PANEL
              ---------------------------------- */}

              <section className="panel visualization-panel">

                <div className="panel-header">

                  <div>
                    <div className="panel-title">
                      Visualization
                    </div>

                    <div className="panel-description">
                      Dataset: {currentDataset?.label}
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
                    title={currentDataset?.label ?? "Dataset"}
                  />

                  <div
                    className="flex justify-between items-center flex-wrap gap-2 mt-3 text-[10px]"
                    style={{ color: "var(--text-muted)" }}
                  >

                    <div className="flex gap-3 flex-wrap">

                      <span>
                        {samples} samples
                      </span>

                      <span>
                        2 features
                      </span>

                      <span>
                        Seed: {seed}
                      </span>

                    </div>

                    <span>
                      {showLabels
                        ? "Ground truth labels"
                        : algorithm === "PCA"
                          ? "Unlabeled projection view"
                          : "Unlabeled dataset view"
                      }
                    </span>

                  </div>

                </div>

              </section>

              {/* ---------------------------------
                  CONTROL PANEL
              ---------------------------------- */}

              <aside className="panel control-panel">

                {/* Dataset generator */}

                <div className="flex items-center justify-between mb-4">

                  <div className="panel-title">
                    Dataset Generator
                  </div>

                  <Database
                    size={16}
                    style={{ color: "var(--accent)" }}
                  />

                </div>

                {/* Dataset options */}

                <div className="grid grid-cols-2 gap-2">

                  {datasets.map(item => (

                    <button
                      key={item.name}
                      onClick={() => selectDataset(item.name)}
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

                {/* Dataset parameters */}

                <div className="control-section">

                  <div className="flex items-center gap-2 mb-4">

                    <SlidersHorizontal
                      size={14}
                      style={{ color: "var(--accent)" }}
                    />

                    <div className="text-xs font-semibold">
                      Parameters
                    </div>

                  </div>

                  {/* Samples */}

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
                      onChange={event => {
                        setSamples(Number(event.target.value));
                        resetExecution();
                      }}
                    />

                  </div>

                  {/* Noise */}

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
                      onChange={event => {
                        setNoise(Number(event.target.value));
                        resetExecution();
                      }}
                    />

                  </div>

                </div>

                {/* Algorithm parameters */}

                {algorithm === "K-Means" && (

                  <div className="control-section">

                    <div className="text-xs font-semibold mb-4">
                      K-Means Parameters
                    </div>

                    <label className="control-label">

                      <span>Number of clusters (K)</span>

                      <span className="control-value">
                        {clusters}
                      </span>

                    </label>

                    <input
                      type="range"
                      min="2"
                      max="8"
                      step="1"
                      value={clusters}
                      onChange={event => {
                        setClusters(Number(event.target.value));
                        resetExecution();
                      }}
                    />

                  </div>

                )}

                {/* Dataset information */}

                <div className="control-section">

                  <div className="text-xs font-semibold mb-3">
                    Dataset Information
                  </div>

                  {[
                    ["Dataset", currentDataset?.label ?? dataset],
                    ["Samples", samples],
                    ["Features", 2],
                    ["Random seed", seed],
                    ["Noise", noise.toFixed(2)],
                  ].map(([label, value]) => (

                    <div
                      key={label}
                      className="flex justify-between py-2 border-b text-[11px]"
                      style={{ borderColor: "var(--border)" }}
                    >

                      <span style={{ color: "var(--text-muted)" }}>
                        {label}
                      </span>

                      <span className="font-medium">
                        {value}
                      </span>

                    </div>

                  ))}

                </div>

                {/* Status */}

                <div className="pt-4">

                  <div className="text-xs font-semibold mb-2">
                    Execution Status
                  </div>

                  <div
                    className="text-[11px]"
                    style={{ color: "var(--text-muted)" }}
                  >
                    Ready to start
                  </div>

                  <div className="text-[10px] mt-2">
                    Algorithm engine not connected yet.
                  </div>

                </div>

              </aside>

            </div>

            {/* =================================
                EXECUTION PANEL
            ================================== */}

            <section className="panel execution-panel">

              <div className="flex items-center justify-between">

                <div>

                  <div className="panel-title">
                    Algorithm Execution
                  </div>

                  <div className="panel-description">
                    Step {step} / {executionSteps.length}
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

              {/* Progress */}

              <div className="execution-progress">

                {executionSteps.map((item, index) => (

                  <div
                    key={item}
                    className={`execution-progress-item ${index < step ? "completed" : ""
                      }`}
                    title={item}
                  />

                ))}

              </div>

              {/* Playback */}

              <div className="flex items-center justify-center gap-3">

                <button
                  className="execution-button"
                  disabled
                  aria-label="First step"
                >
                  <SkipBack size={18} />
                </button>

                <button
                  className="execution-button"
                  disabled
                  aria-label="Previous step"
                >
                  <RotateCcw size={17} />
                </button>

                <button
                  className="execution-button primary"
                  disabled
                  aria-label="Play execution"
                >
                  <Play size={18} fill="currentColor" />
                </button>

                <button
                  className="execution-button"
                  disabled
                  aria-label="Next step"
                >
                  <SkipForward size={18} />
                </button>

              </div>

              <div
                className="text-center text-[10px] mt-2"
                style={{ color: "var(--text-muted)" }}
              >
                Execution history will be available after
                algorithm integration.

              </div>

            </section>

          </div>

        </div>

      </section>

    </main>
  );
}
