"use client";

import { useEffect, useMemo, useState } from "react";
import { useTheme } from "next-themes";
import {
  BrainCircuit, Layers3, Sun, Moon, Play, Pause, SkipBack, SkipForward,
  RotateCcw, PanelLeftClose, PanelLeft, Settings2, CircleHelp, Shuffle,
  SlidersHorizontal, Database, ChevronLeft, ChevronRight, Activity,
  BookOpen, Calculator, GitBranch, BarChart3, Info,
} from "lucide-react";
import ScatterPlot from "@/components/visualizations/ScatterPlot";
import NeuralNetworkPlayground from "@/components/visualizations/NeuralNetworkPlayground";
import { generateDataset, type DatasetName } from "@/lib/datasets/generateDataset";
import { runAlgorithm } from "@/lib/algorithms/engine";
import { DEFAULT_PARAMETERS, type AlgorithmName, type AlgorithmParameters } from "@/lib/algorithms/types";

const groups = [
  { name: "Classification", algorithms: ["CART", "C4.5", "KNN", "SVM", "Naive Bayes", "Neural Network"] },
  { name: "Clustering", algorithms: ["K-Means"] },
  { name: "Regression", algorithms: ["Linear Regression"] },
  { name: "Dimensionality Reduction", algorithms: ["PCA"] },
];
const datasets = [
  { name: "blobs", label: "Blobs", description: "Gaussian clusters" },
  { name: "moons", label: "Moons", description: "Nonlinear separation" },
  { name: "circles", label: "Circles", description: "Concentric rings" },
  { name: "spiral", label: "Spiral", description: "Interlocking spirals" },
  { name: "linear", label: "Linear", description: "Linear separation" },
  { name: "xor", label: "XOR", description: "Exclusive OR" },
  { name: "linear-regression", label: "Regression", description: "Continuous linear target" },
] as const;
const classificationAlgorithms: AlgorithmName[] = ["CART", "C4.5", "KNN", "SVM", "Naive Bayes"];
const algorithmGuides: Record<AlgorithmName, { idea: string; mechanics: string[]; assumptions: string[]; complexity: string }> = {
  "CART": { idea: "Builds a binary decision tree by repeatedly selecting the feature threshold that produces the purest child nodes. This implementation uses Gini impurity.", mechanics: ["Evaluate candidate thresholds for each feature.", "Calculate the weighted Gini impurity after each possible split.", "Choose the split with the largest impurity reduction.", "Repeat recursively until the depth limit or a stopping condition is reached.", "Assign each leaf the majority class of its training observations."], assumptions: ["Produces axis-aligned decision boundaries.", "Deep trees can overfit; limiting depth controls model complexity."], complexity: "At each node, candidate thresholds are evaluated over the training observations; deeper trees create more nodes." },
  "C4.5": { idea: "A decision-tree learner that chooses splits using information gain ratio, reducing the bias of information gain toward attributes with many possible values.", mechanics: ["Calculate class entropy before splitting.", "Evaluate candidate feature thresholds and their information gain.", "Calculate split information for each candidate.", "Divide information gain by split information to obtain gain ratio.", "Choose the best split and continue recursively."], assumptions: ["Uses entropy and gain ratio as split criteria.", "This visualizer uses a simplified continuous-feature, binary-split version."], complexity: "Threshold search is repeated at each internal node; tree depth controls the number of recursive splits." },
  "KNN": { idea: "A lazy, instance-based method. It predicts a query point from the labels of its nearest training observations.", mechanics: ["Keep the training observations as the reference set.", "Calculate the distance from the query point to every training point.", "Sort observations by distance.", "Select the K nearest neighbors.", "Use majority voting to determine the predicted class."], assumptions: ["Distance is Euclidean in the current two-feature space.", "Feature scaling matters because distance is scale-sensitive.", "A small K is sensitive to noise; a large K creates smoother boundaries."], complexity: "A direct query requires distances to all training observations: O(n) distance calculations, followed by sorting." },
  "SVM": { idea: "Learns a separating hyperplane that aims to maximize the margin between two classes while penalizing margin violations.", mechanics: ["Encode the two classes as -1 and +1.", "Initialize the weight vector and bias.", "Calculate each observation's signed margin.", "Apply hinge-loss updates and L2 regularization.", "Evaluate the learned boundary on held-out observations."], assumptions: ["This implementation is a binary linear SVM.", "The separating boundary is linear; nonlinear kernels are not included.", "Regularization controls the trade-off between margin size and violations."], complexity: "Training performs repeated passes over the training set for the configured number of epochs." },
  "Naive Bayes": { idea: "Uses Bayes' theorem with a conditional-independence assumption to calculate the posterior probability of each class.", mechanics: ["Estimate prior probability for each class.", "Estimate each feature's mean and variance within each class.", "Evaluate the Gaussian likelihood of the query features.", "Combine likelihoods with class priors.", "Choose the class with the largest posterior score."], assumptions: ["Features are treated as conditionally independent given the class.", "Continuous features are modeled with Gaussian distributions.", "The independence assumption is a simplification, not a requirement of real data."], complexity: "Parameter estimation and prediction are linear in the number of observations and features." },
  "K-Means": { idea: "Partitions observations into K clusters by alternating nearest-centroid assignment and centroid recomputation.", mechanics: ["Choose K initial centroids.", "Assign every observation to its nearest centroid.", "Recalculate each centroid as the mean of its assigned observations.", "Repeat assignment and update until convergence or the iteration limit.", "Inspect cluster membership and within-cluster compactness."], assumptions: ["Uses Euclidean distance and mean-based centroids.", "The objective favors compact, roughly spherical clusters.", "The result can depend on initialization and the selected K."], complexity: "Each iteration compares every observation with every centroid: O(n × K × iterations)." },
  "Linear Regression": { idea: "Fits a linear relationship by minimizing the mean squared difference between predicted and observed target values.", mechanics: ["Initialize slope and intercept.", "Calculate predictions for all observations.", "Calculate residuals and mean squared error.", "Compute gradients for slope and intercept.", "Update parameters with gradient descent and inspect the final fit."], assumptions: ["The model is a straight line in the current one-predictor view.", "MSE penalizes large residuals more strongly.", "Gradient descent behavior depends on learning rate and feature scale."], complexity: "Each epoch evaluates predictions and gradients over the training observations: O(n) per epoch." },
  "Neural Network": { idea: "A multi-layer perceptron learns nonlinear decision boundaries by composing weighted sums and activation functions. It is trained using backpropagation and gradient descent.", mechanics: ["Pass the input features through each layer using weights and biases.", "Apply the selected activation function to each hidden layer.", "Calculate binary cross-entropy from the output probability and true class.", "Propagate gradients backward through the network using the chain rule.", "Update all weights and biases using gradient descent."], assumptions: ["This playground performs binary classification on two input features.", "All hidden layers are fully connected and share the selected activation function.", "The training loss is measured on the full supplied dataset; there is no held-out test split in this compact implementation."], complexity: "Each epoch performs a forward and backward pass for every sample; computation grows with the number of connections and observations." },
  "PCA": { idea: "Finds orthogonal directions of maximum variance and projects the centered data onto principal components.", mechanics: ["Calculate the mean of each feature.", "Center the observations by subtracting feature means.", "Calculate the covariance matrix.", "Find eigenvalues and eigenvectors of the covariance matrix.", "Project observations onto the first principal component and calculate explained variance."], assumptions: ["PCA is unsupervised and does not use class labels to choose components.", "The first component captures the largest possible variance among linear directions.", "Variance is not necessarily the same as class-separation quality."], complexity: "For two features, covariance and eigendecomposition are inexpensive; projection is linear in the sample count." },
};

export default function Home() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [algorithm, setAlgorithm] = useState<AlgorithmName>("K-Means");
  const [dataset, setDataset] = useState<DatasetName>("blobs");
  const [samples, setSamples] = useState(200);
  const [noise, setNoise] = useState(0.05);
  const [seed, setSeed] = useState(42);
  const [parameters, setParameters] = useState<AlgorithmParameters>(DEFAULT_PARAMETERS);
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => setMounted(true), []);
  const data = useMemo(() => generateDataset({ type: dataset, samples, noise, seed }), [dataset, samples, noise, seed]);
  const result = useMemo(() => runAlgorithm({ algorithm, points: data.points, parameters, seed }), [algorithm, data, parameters, seed]);
  const totalSteps = Math.max(1, result.snapshots.length);
  const currentSnapshot = result.snapshots[Math.min(currentStep, totalSteps - 1)];
  const selectedDataset = datasets.find(d => d.name === dataset);
  const guide = algorithmGuides[algorithm];
  const showLabels = classificationAlgorithms.includes(algorithm) || algorithm === "Neural Network";

  useEffect(() => {
    if (!playing) return;
    if (currentStep >= totalSteps - 1) { setPlaying(false); return; }
    const timer = window.setTimeout(() => setCurrentStep(step => Math.min(step + 1, totalSteps - 1)), 1000);
    return () => window.clearTimeout(timer);
  }, [playing, currentStep, totalSteps]);

  function resetExecution() { setCurrentStep(0); setPlaying(false); }
  function updateParameter<K extends keyof AlgorithmParameters>(key: K, value: AlgorithmParameters[K]) {
    setParameters(previous => ({ ...previous, [key]: value }));
    resetExecution();
  }
  function selectAlgorithm(value: string) { const next = value as AlgorithmName; setAlgorithm(next); if (next === "Linear Regression") setDataset("linear-regression"); else if (next === "Neural Network") setDataset("xor"); else if (dataset === "linear-regression") setDataset("blobs"); resetExecution(); }
  function togglePlay() {
    if (currentStep >= totalSteps - 1) setCurrentStep(0);
    setPlaying(value => !value);
  }
  const formatMetric = (value: number) => Number.isFinite(value) ? Number(value).toFixed(5) : String(value);

  return (
    <main className="app-shell">
      {sidebarOpen && <button className="sidebar-backdrop" aria-label="Close sidebar" onClick={() => setSidebarOpen(false)} />}
      <aside className={`app-sidebar ${sidebarOpen ? "" : "collapsed"}`}>
        <div className="h-[66px] px-5 flex items-center justify-between border-b shrink-0" style={{ borderColor: "var(--border)" }}>
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: "var(--accent-soft)", color: "var(--accent)" }}><BrainCircuit size={21} /></div>
            <div><div className="text-sm font-semibold">ML Laboratory</div><div className="text-[9px] tracking-[1.6px] uppercase mt-0.5" style={{ color: "var(--text-muted)" }}>Interactive Learning</div></div>
          </div>
          <button className="icon-button md:hidden" onClick={() => setSidebarOpen(false)}><PanelLeftClose size={17} /></button>
        </div>
        <div className="flex-1 overflow-y-auto">
          <div className="sidebar-section"><div className="sidebar-label">Workspace</div><button className="algorithm-item active"><Layers3 size={16} className="mr-3" />Algorithm Explorer</button></div>
          <div className="sidebar-section"><div className="sidebar-label">Algorithms</div>
            {groups.map(group => <div key={group.name} className="algorithm-group"><div className="algorithm-group-title">{group.name}</div>
              {group.algorithms.map(name => <button key={name} onClick={() => selectAlgorithm(name)} className={`algorithm-item ${algorithm === name ? "active" : ""}`}>{name}</button>)}
            </div>)}
          </div>
        </div>
        <div className="px-5 py-4 border-t flex items-center gap-2 text-[11px] shrink-0" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}><span className="w-2 h-2 rounded-full bg-emerald-400" />Local computation</div>
      </aside>

      <section className="app-main">
        <header className="app-header">
          <div className="flex items-center gap-4 min-w-0">
            <button className="icon-button" onClick={() => setSidebarOpen(v => !v)}>{sidebarOpen ? <PanelLeftClose size={19} /> : <PanelLeft size={19} />}</button>
            <div className="min-w-0"><div className="text-sm font-medium truncate">{algorithm}</div><div className="text-[10px] mt-0.5" style={{ color: "var(--text-muted)" }}>Interactive algorithm workspace</div></div>
          </div>
          <div className="flex items-center gap-2">
            <div className="hidden sm:flex items-center gap-2 text-[11px] mr-2" style={{ color: "var(--text-muted)" }}><span className="w-2 h-2 rounded-full bg-emerald-400" />Browser environment</div>
            <button className="icon-button" onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label="Toggle theme">{mounted ? theme === "dark" ? <Sun size={19} /> : <Moon size={19} /> : <Sun size={19} />}</button>
            <button className="icon-button" title="Help"><CircleHelp size={18} /></button>
            <button className="icon-button" title="Settings"><Settings2 size={18} /></button>
          </div>
        </header>

        <div className="app-content">
          <div className="workspace workspace-upgraded">
            <aside className="workspace-controls">
              <section className="panel control-panel">
                <div className="flex items-center justify-between mb-4"><div><div className="panel-title">Dataset Generator</div><div className="panel-description">Synthetic 2D data</div></div><Database size={16} style={{ color: "var(--accent)" }} /></div>
                <div className="grid grid-cols-2 gap-2">
                  {datasets.filter(item => item.name !== "linear-regression" || algorithm === "Linear Regression").map(item => <button key={item.name} onClick={() => { setDataset(item.name); resetExecution(); }} className={`dataset-option ${dataset === item.name ? "active" : ""}`}><div className="dataset-option-name">{item.label}</div><div className="dataset-option-description">{item.description}</div></button>)}
                </div>
                <div className="control-section">
                  <div className="flex items-center gap-2 mb-4"><SlidersHorizontal size={14} style={{ color: "var(--accent)" }} /><div className="text-xs font-semibold">Dataset Parameters</div></div>
                  <div className="mb-5"><label className="control-label"><span>Number of samples</span><span className="control-value">{samples}</span></label><input type="range" min="50" max="1000" step="50" value={samples} onChange={e => { setSamples(Number(e.target.value)); resetExecution(); }} /></div>
                  <div><label className="control-label"><span>Noise</span><span className="control-value">{noise.toFixed(2)}</span></label><input type="range" min="0" max="0.5" step="0.01" value={noise} onChange={e => { setNoise(Number(e.target.value)); resetExecution(); }} /></div>
                  <button className="text-button mt-4" onClick={() => { setSeed(v => v + 1); resetExecution(); }}><Shuffle size={14} />Regenerate samples</button>
                </div>
                <div className="control-section">
                  <div className="text-xs font-semibold mb-4">{algorithm} Parameters</div>
                  {algorithm === "K-Means" && <><label className="control-label"><span>Number of clusters (K)</span><span className="control-value">{parameters.clusters}</span></label><input type="range" min="2" max="8" value={parameters.clusters} onChange={e => updateParameter("clusters", Number(e.target.value))} /><div className="mt-4"><label className="control-label">Initialization</label><select className="control-select" value={parameters.initialization} onChange={e => updateParameter("initialization", e.target.value as "random" | "kmeans++")}><option value="random">Random</option><option value="kmeans++">K-Means++</option></select></div></>}
                  {algorithm === "KNN" && <><label className="control-label"><span>Neighbors (K)</span><span className="control-value">{parameters.neighbors}</span></label><input type="range" min="1" max="20" value={parameters.neighbors} onChange={e => updateParameter("neighbors", Number(e.target.value))} /></>}
                  {(algorithm === "CART" || algorithm === "C4.5") && <><label className="control-label"><span>Maximum tree depth</span><span className="control-value">{parameters.maxDepth}</span></label><input type="range" min="1" max="8" value={parameters.maxDepth} onChange={e => updateParameter("maxDepth", Number(e.target.value))} /></>}
                  {(algorithm === "SVM" || algorithm === "Linear Regression") && <><label className="control-label"><span>Learning rate</span><span className="control-value">{parameters.learningRate.toFixed(3)}</span></label><input type="range" min="0.001" max="0.1" step="0.001" value={parameters.learningRate} onChange={e => updateParameter("learningRate", Number(e.target.value))} /><div className="mt-4"><label className="control-label"><span>Training epochs</span><span className="control-value">{parameters.epochs}</span></label><input type="range" min="10" max="200" step="10" value={parameters.epochs} onChange={e => updateParameter("epochs", Number(e.target.value))} /></div></>}
                  {algorithm === "SVM" && <div className="mt-4"><label className="control-label"><span>Regularization (λ)</span><span className="control-value">{parameters.regularization.toFixed(3)}</span></label><input type="range" min="0.001" max="0.1" step="0.001" value={parameters.regularization} onChange={e => updateParameter("regularization", Number(e.target.value))} /></div>}
                </div>
                <div className="control-section">
                  <div className="text-xs font-semibold mb-3">Dataset Information</div>
                  {([["Dataset", selectedDataset?.label ?? dataset], ["Samples", samples], ["Features", 2], ["Noise", noise.toFixed(2)], ["Seed", seed]] as const).map(([label, value]) => <div key={label} className="flex justify-between py-2 border-b text-[11px]" style={{ borderColor: "var(--border)" }}><span style={{ color: "var(--text-muted)" }}>{label}</span><span>{value}</span></div>)}
                </div>
              </section>

              <section className="panel execution-panel execution-fixed">
                <div className="flex items-center justify-between gap-2"><div><div className="panel-title">Execution</div><div className="panel-description">Step {currentStep + 1} of {totalSteps} · {currentSnapshot?.phase ?? "Ready"}</div></div><button className="text-button" onClick={resetExecution}><RotateCcw size={13} />Reset</button></div>
                <input type="range" min="0" max={Math.max(0, totalSteps - 1)} value={currentStep} onChange={e => { setCurrentStep(Number(e.target.value)); setPlaying(false); }} className="w-full mt-3" aria-label="Execution timeline" />
                <div className="execution-compact-controls">
                  <button className="execution-button" onClick={() => { setCurrentStep(0); setPlaying(false); }} aria-label="First step"><SkipBack size={16} /></button>
                  <button className="execution-button" onClick={() => setCurrentStep(v => Math.max(0, v - 1))} disabled={currentStep === 0} aria-label="Previous step"><ChevronLeft size={17} /></button>
                  <button className="execution-button primary" onClick={togglePlay} aria-label={playing ? "Pause" : "Play"}>{playing ? <Pause size={17} /> : <Play size={17} fill="currentColor" />}</button>
                  <button className="execution-button" onClick={() => setCurrentStep(v => Math.min(totalSteps - 1, v + 1))} disabled={currentStep >= totalSteps - 1} aria-label="Next step"><ChevronRight size={17} /></button>
                  <button className="execution-button" onClick={() => { setCurrentStep(totalSteps - 1); setPlaying(false); }} aria-label="Last step"><SkipForward size={16} /></button>
                </div>
                <div className="execution-step-title">{currentSnapshot?.title}</div>
              </section>
            </aside>

            <section className="panel visualization-panel visualization-scroll-area">
              <div className="panel-header">
                <div><div className="panel-title">Algorithm Workspace</div><div className="panel-description">{selectedDataset?.label} · {algorithm} · {samples} observations</div></div>
                <span className="workspace-status"><Activity size={13} />{currentSnapshot?.phase ?? "Ready"}</span>
              </div>
              <div className="visualization-scroll-content">
                <section className="viz-section">
                  <div className="viz-section-heading"><div><div className="viz-section-title">Data visualization</div><div className="panel-description">Observe how the selected algorithm changes the data representation.</div></div><button className="text-button" onClick={() => { setSeed(v => v + 1); resetExecution(); }}><Shuffle size={13} />Regenerate</button></div>
                  <div className="viz-plot-wrap">
                    {algorithm === "Neural Network" ? <NeuralNetworkPlayground points={data.points} /> : <ScatterPlot points={data.points} showLabels={showLabels} assignments={currentSnapshot?.assignments} centroids={currentSnapshot?.centroids} regressionLine={currentSnapshot?.regressionLine} decisionSplits={currentSnapshot?.decisionSplits} highlightedPointIds={currentSnapshot?.highlightedPointIds} title={selectedDataset?.label} />}
                  </div>
                  <div className="plot-meta"><span>{samples} samples</span><span>2 features</span><span>Seed {seed}</span><span>{showLabels ? "Supervised labels visible" : "Unsupervised view"}</span></div>
                </section>

                <section className="viz-section">
                  <div className="viz-section-heading"><div><div className="viz-section-title">Current step analysis</div><div className="panel-description">Detailed explanation synchronized with the execution controls.</div></div><span className="phase-pill">{currentSnapshot?.phase ?? "Ready"}</span></div>
                  <div className="analysis-card">
                    <div className="analysis-card-title">{currentSnapshot?.title ?? "Ready to execute"}</div>
                    <p className="analysis-description">{currentSnapshot?.description ?? result.summary}</p>
                    {currentSnapshot?.equation && <div className="equation-block"><span>Mathematical expression</span><code>{currentSnapshot.equation}</code></div>}
                    <div className="metric-grid">
                      {Object.entries(currentSnapshot?.metrics ?? {}).map(([key, value]) => <div className="metric-card" key={key}><div className="metric-label">{key.replace(/([A-Z])/g, " $1").replace(/^./, c => c.toUpperCase())}</div><div className="metric-value">{formatMetric(value)}</div></div>)}
                    </div>
                    {currentSnapshot?.details?.length ? <div className="detail-table">{currentSnapshot.details.map((detail, index) => <div className="detail-row" key={index}><span>{detail.name}</span><strong>{detail.value}</strong></div>)}</div> : <p className="empty-note">No additional numeric details are emitted for this step.</p>}
                  </div>
                </section>

                <section className="viz-section">
                  <div className="viz-section-heading"><div><div className="viz-section-title">How {algorithm} works</div><div className="panel-description">Algorithm overview and the reasoning behind each stage.</div></div><BookOpen size={17} style={{ color: "var(--accent)" }} /></div>
                  <div className="guide-grid">
                    <div className="guide-card guide-wide"><div className="guide-card-heading"><Info size={15} />Core idea</div><p>{guide.idea}</p></div>
                    <div className="guide-card"><div className="guide-card-heading"><GitBranch size={15} />Algorithm workflow</div><ol className="guide-list">{guide.mechanics.map((item, i) => <li key={i}><span>{i + 1}</span>{item}</li>)}</ol></div>
                    <div className="guide-card"><div className="guide-card-heading"><Calculator size={15} />Assumptions & interpretation</div><ul className="guide-bullets">{guide.assumptions.map(item => <li key={item}>{item}</li>)}</ul><div className="complexity-note"><strong>Computational note</strong><p>{guide.complexity}</p></div></div>
                  </div>
                </section>

                <section className="viz-section">
                  <div className="viz-section-heading"><div><div className="viz-section-title">Execution history</div><div className="panel-description">Select any recorded stage to inspect its state and calculations.</div></div><BarChart3 size={17} style={{ color: "var(--accent)" }} /></div>
                  <div className="history-list">
                    {result.snapshots.map((step, index) => <button key={step.index} className={`history-item ${index === currentStep ? "selected" : ""}`} onClick={() => { setCurrentStep(index); setPlaying(false); }}><span className="history-index">{String(index + 1).padStart(2, "0")}</span><span className="history-copy"><strong>{step.title}</strong><small>{step.description}</small></span><span className="history-phase">{step.phase}</span></button>)}
                  </div>
                </section>

                <section className="viz-section">
                  <div className="viz-section-heading"><div><div className="viz-section-title">Model summary</div><div className="panel-description">Overall interpretation of this algorithm run.</div></div></div>
                  <div className="summary-card"><p>{result.summary}</p><div className="summary-stats"><div><span>Algorithm</span><strong>{algorithm}</strong></div><div><span>Execution steps</span><strong>{totalSteps}</strong></div><div><span>Training data</span><strong>{samples} points</strong></div><div><span>Current stage</span><strong>{currentStep + 1} / {totalSteps}</strong></div></div></div>
                </section>
              </div>
            </section>
          </div>
        </div>
      </section>
    </main>
  );
}
