import {
  ArrowLeft,
  ArrowRight,
  ChevronRight,
  FileText,
  Folder,
  FolderPlus,
  Loader2,
  UploadCloud,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ProjectsPanel } from "./components/ProjectsPanel.jsx";
import { LandingScreen } from "./components/LandingScreen.jsx";
import { ToolsPanel } from "./components/ToolsPanel.jsx";
import { TrackMotif } from "./components/TrackMotif.jsx";
import { ResultsView } from "./ResultsView.jsx";
import { API_BASE, apiFetch, readJsonResponse } from "./lib/api.js";

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isFastaName(name) {
  return /\.(fna|fa|fasta)$/i.test(name);
}

function formatNumber(value) {
  if (typeof value !== "number") return value;
  return new Intl.NumberFormat("en-US").format(value);
}

function formatElapsed(ms) {
  if (!ms || ms < 1000) return "<1s";
  const seconds = Math.round(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${seconds % 60}s`;
}

function TopBar({ elevated = false, onHome, projectName = "", variant = "default" }) {
  return (
    <header className={`topbar ${variant === "home" ? "topbar--home" : ""} ${variant === "landing" ? "topbar--landing" : ""} ${elevated ? "topbar--elevated" : ""}`.trim()}>
      <button className="topbar-brand" onClick={onHome} type="button">
        Annotator
      </button>
      <div className="topbar-meta">
        {projectName ? (
          <span className="topbar-project">{projectName}</span>
        ) : (
          <span className="topbar-tag">Local · HMM + CNN</span>
        )}
      </div>
      <a
        className="topbar-link"
        href="https://github.com/sansen405/genome-annotation-hidden-markov-model"
        rel="noopener noreferrer"
        target="_blank"
      >
        Docs
      </a>
    </header>
  );
}

function App() {
  const [projects, setProjects] = useState([]);
  const [activeProjectId, setActiveProjectId] = useState("");
  const [projectMeta, setProjectMeta] = useState(null);
  const [tree, setTree] = useState([]);
  const [selectedPath, setSelectedPath] = useState("inputs");
  const [directory, setDirectory] = useState(null);
  const [selectedRun, setSelectedRun] = useState(null);
  const [selectedPredictionId, setSelectedPredictionId] = useState("");
  const [loading, setLoading] = useState(true);
  const [creatingProject, setCreatingProject] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newFolderName, setNewFolderName] = useState("");
  const [runStatus, setRunStatus] = useState("idle");
  const [runningInputName, setRunningInputName] = useState("");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [errorMessage, setErrorMessage] = useState("");
  const [showLanding, setShowLanding] = useState(
    () => sessionStorage.getItem("annotator-entered") !== "1"
  );
  const fileInputRef = useRef(null);

  const refreshProjects = useCallback(async () => {
    const payload = await apiFetch("/api/projects");
    setProjects(payload);
  }, []);

  const refreshProject = useCallback(async (projectId) => {
    const payload = await apiFetch(`/api/projects/${projectId}`);
    setProjectMeta(payload);
    setTree(payload.tree ?? []);
  }, []);

  const refreshDirectory = useCallback(async (projectId, folderPath) => {
    const payload = await apiFetch(
      `/api/projects/${projectId}/list?path=${encodeURIComponent(folderPath)}`
    );
    setDirectory(payload);
  }, []);

  useEffect(() => {
    refreshProjects()
      .catch((error) => setErrorMessage(error.message))
      .finally(() => setLoading(false));
  }, [refreshProjects]);

  useEffect(() => {
    if (!activeProjectId) return;
    refreshProject(activeProjectId).catch((error) => setErrorMessage(error.message));
  }, [activeProjectId, refreshProject]);

  useEffect(() => {
    if (!activeProjectId || !selectedPath || selectedPath.startsWith("runs/")) return;
    const folderPath = selectedPath === "tools" ? "inputs" : selectedPath;
    refreshDirectory(activeProjectId, folderPath).catch((error) =>
      setErrorMessage(error.message)
    );
  }, [activeProjectId, refreshDirectory, selectedPath]);

  const selectedRunId = selectedPath.startsWith("runs/") ? selectedPath.split("/")[1] : "";

  useEffect(() => {
    if (!activeProjectId || !selectedRunId) {
      setSelectedRun(null);
      return;
    }
    apiFetch(`/api/projects/${activeProjectId}/runs/${selectedRunId}`)
      .then((run) => {
        setSelectedRun(run);
        setSelectedPredictionId(run.predictions?.[0]?.id ?? "");
      })
      .catch((error) => setErrorMessage(error.message));
  }, [activeProjectId, selectedRunId]);

  const predictions = selectedRun?.predictions ?? [];
  const selectedPrediction =
    predictions.find((prediction) => prediction.id === selectedPredictionId) ??
    predictions[0] ??
    null;

  const breadcrumbs = useMemo(() => {
    if (!projectMeta) return [];
    if (selectedPath === "tools") {
      return [
        { label: projectMeta.name, path: "inputs" },
        { label: "Tools", path: "tools" },
      ];
    }
    const parts = selectedPath.split("/").filter(Boolean);
    if (parts.length === 0) {
      return [{ label: projectMeta.name, path: "inputs" }];
    }

    const crumbs = [];
    let current = "";
    for (let index = 0; index < parts.length; index += 1) {
      const part = parts[index];
      current = current ? `${current}/${part}` : part;
      const label = index === 0 && part === "inputs" ? projectMeta.name : part;
      crumbs.push({ label, path: current });
    }
    return crumbs;
  }, [projectMeta, selectedPath]);

  async function handleCreateProject(event) {
    event.preventDefault();
    const name = newProjectName.trim();
    if (!name) return;
    try {
      const project = await apiFetch("/api/projects", {
        body: JSON.stringify({ name }),
        headers: { "Content-Type": "application/json" },
        method: "POST",
      });
      setProjects((existing) => [project, ...existing]);
      setNewProjectName("");
      setCreatingProject(false);
      setActiveProjectId(project.id);
      setSelectedPath("inputs");
      setSelectedRun(null);
      setErrorMessage("");
    } catch (error) {
      setErrorMessage(error.message);
    }
  }

  function openProject(projectId) {
    setActiveProjectId(projectId);
    setSelectedPath("inputs");
    setSelectedRun(null);
    setRunStatus("idle");
    setErrorMessage("");
  }

  function enterApp() {
    sessionStorage.setItem("annotator-entered", "1");
    setShowLanding(false);
  }

  function handleBrandHome() {
    if (activeProjectId) {
      leaveProject();
      return;
    }
    if (!showLanding) {
      setShowLanding(true);
    }
  }

  function leaveProject() {
    setActiveProjectId("");
    setProjectMeta(null);
    setDirectory(null);
    setSelectedRun(null);
    setSelectedPath("inputs");
    setRunStatus("idle");
    setRunningInputName("");
    setErrorMessage("");
    refreshProjects();
  }

  async function handleUploadFiles(files, folderPath = selectedPath) {
    if (!activeProjectId || files.length === 0) return;
    for (const file of files) {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch(
        `${API_BASE}/api/projects/${activeProjectId}/upload?path=${encodeURIComponent(folderPath)}`,
        { body, method: "POST" }
      );
      const payload = await readJsonResponse(response);
      if (!response.ok) {
        throw new Error(payload.detail || payload.error || `Upload failed for ${file.name}.`);
      }
    }
    await refreshProject(activeProjectId);
    await refreshDirectory(activeProjectId, folderPath);
    setSelectedPath(folderPath);
  }

  async function handleCreateFolder(event) {
    event.preventDefault();
    const name = newFolderName.trim();
    if (!activeProjectId || !name) return;
    await apiFetch(`/api/projects/${activeProjectId}/folders`, {
      body: JSON.stringify({ name, parentPath: selectedPath === "." ? "" : selectedPath }),
      headers: { "Content-Type": "application/json" },
      method: "POST",
    });
    setNewFolderName("");
    await refreshProject(activeProjectId);
    await refreshDirectory(activeProjectId, selectedPath);
  }

  async function handleRunPrediction(inputPath) {
    if (!activeProjectId || !inputPath) return;
    const started = Date.now();
    const inputName = inputPath.split("/").pop() || inputPath;
    setRunStatus("running");
    setRunningInputName(inputName);
    setElapsedSeconds(0);
    setErrorMessage("");
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - started) / 1000));
    }, 1000);

    try {
      const run = await apiFetch(`/api/projects/${activeProjectId}/runs`, {
        body: JSON.stringify({ inputPath }),
        headers: { "Content-Type": "application/json" },
        method: "POST",
      });
      await refreshProject(activeProjectId);
      setSelectedPath(`runs/${run.id}`);
      setRunStatus("done");
      setElapsedSeconds(Math.max(1, Math.round((Date.now() - started) / 1000)));
      window.setTimeout(() => {
        setRunStatus("idle");
        setRunningInputName("");
      }, 2400);
    } catch (error) {
      setRunStatus("failed");
      setErrorMessage(
        error.message === "Failed to fetch"
          ? "Cannot connect to Annotator on this machine. Restart the app to try again."
          : error.message
      );
    } finally {
      window.clearInterval(timer);
    }
  }

  if (loading) {
    return (
      <div className="app-root">
        <div className="loading-shell">
          <Loader2 className="spin page-loader" size={28} />
        </div>
      </div>
    );
  }

  if (!activeProjectId && showLanding) {
    return (
      <div className="app-root app-root--landing">
        <TopBar onHome={handleBrandHome} variant="landing" />

        <LandingScreen onEnter={enterApp} projectCount={projects.length} />
      </div>
    );
  }

  if (!activeProjectId) {
    return (
      <div className="app-root app-root--projects">
        <TopBar onHome={handleBrandHome} />

        <ProjectsPanel
          creatingProject={creatingProject}
          errorMessage={errorMessage}
          newProjectName={newProjectName}
          onCreateProject={handleCreateProject}
          onOpenCreate={() => setCreatingProject(true)}
          onOpenProject={openProject}
          onSetCreatingProject={setCreatingProject}
          onSetNewProjectName={setNewProjectName}
          projects={projects}
        />
      </div>
    );
  }

  return (
    <div className="app-root workspace-root">
      <TopBar onHome={handleBrandHome} projectName={projectMeta?.name} />

      <div className="app-shell">
        <aside className="sidebar project-sidebar">
          <ProjectSidebar
            onBack={leaveProject}
            onSelect={setSelectedPath}
            projectName={projectMeta?.name}
            selectedPath={selectedPath}
            tree={tree}
          />
        </aside>

        <main className="content project-content">
          {errorMessage && <p className="page-error">{errorMessage}</p>}

          {(runStatus === "running" || runStatus === "failed" || runStatus === "done") && (
            <RunProgressStrip
              elapsedSeconds={elapsedSeconds}
              inputName={runningInputName}
              runStatus={runStatus}
            />
          )}

          {selectedRun ? (
            <ResultsView
              breadcrumbs={breadcrumbs}
              exportBaseUrl={`${API_BASE}/api/projects/${activeProjectId}/runs/${selectedRun.id}/export`}
              onNavigate={setSelectedPath}
              onStartAnalysis={() => setSelectedPath("tools")}
              predictions={predictions}
              selectedPrediction={selectedPrediction}
              selectedPredictionId={selectedPredictionId}
              selectedRun={selectedRun}
              setSelectedPredictionId={setSelectedPredictionId}
            />
          ) : selectedPath === "tools" ? (
            <ToolsPanel
              fastaFiles={(directory?.entries ?? []).filter(
                (entry) => entry.type === "file" && isFastaName(entry.name)
              )}
              onNavigate={setSelectedPath}
              onRunPrediction={handleRunPrediction}
              runStatus={runStatus}
            />
          ) : (
            <FolderPanel
              breadcrumbs={breadcrumbs}
              directory={directory}
              fileInputRef={fileInputRef}
              newFolderName={newFolderName}
              onCreateFolder={handleCreateFolder}
              onNavigate={setSelectedPath}
              onUpload={(files) =>
                handleUploadFiles(files, selectedPath).catch((error) =>
                  setErrorMessage(error.message)
                )
              }
              selectedPath={selectedPath}
              setNewFolderName={setNewFolderName}
            />
          )}
        </main>
      </div>
    </div>
  );
}

const WORKSPACE_SECTIONS = [
  { id: "inputs", label: "Inputs", hint: "Upload files" },
  { id: "tools", label: "Tools", hint: "Pick analysis" },
  { id: "runs", label: "Runs", hint: "Results" },
];

function shortRunLabel(label, maxLength = 32) {
  if (!label || label.length <= maxLength) return label;
  return `${label.slice(0, maxLength - 1)}…`;
}

function ProjectSidebar({ onBack, onSelect, projectName, selectedPath, tree }) {
  const activeSection = selectedPath.split("/")[0] || "inputs";
  const runsNode = tree.find((node) => node.name === "runs");
  const runItems = runsNode?.children?.filter((node) => node.type === "run") ?? [];
  const inputsNode = tree.find((node) => node.name === "inputs");
  const inputFolders =
    inputsNode?.children?.filter((node) => node.type === "folder") ?? [];
  const inInputs = activeSection === "inputs";
  const inTools = activeSection === "tools";
  const inRuns = activeSection === "runs" || selectedPath.startsWith("runs/");

  return (
    <>
      <button className="sidebar-back" onClick={onBack} type="button">
        <ArrowLeft size={15} />
        All projects
      </button>

      <div className="sidebar-project">
        <TrackMotif className="sidebar-motif" />
        <p className="sidebar-kicker">Project</p>
        <h2 className="sidebar-title">{projectName}</h2>
      </div>

      <div className="sidebar-body">
        <nav className="sidebar-sections" aria-label="Workspace sections">
          {WORKSPACE_SECTIONS.map((section) => (
            <button
              className={
                activeSection === section.id ? "sidebar-section active" : "sidebar-section"
              }
              key={section.id}
              onClick={() => onSelect(section.id)}
              type="button"
            >
              <span className="sidebar-section-name">{section.label}</span>
              <span className="sidebar-section-hint">{section.hint}</span>
            </button>
          ))}
        </nav>

        {inInputs && inputFolders.length > 0 && (
          <div className="sidebar-subnav">
            <p className="nav-section-label">Subfolders</p>
            <ul className="sidebar-subnav-list">
              {inputFolders.map((folder) => (
                <li key={folder.path}>
                  <button
                    className={
                      selectedPath === folder.path
                        ? "sidebar-subnav-item active"
                        : "sidebar-subnav-item"
                    }
                    onClick={() => onSelect(folder.path)}
                    type="button"
                  >
                    <Folder size={14} />
                    <span className="sidebar-item-label">{folder.name}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {inRuns && runItems.length > 0 && (
          <div className="sidebar-subnav sidebar-subnav--runs">
            <p className="nav-section-label">Completed runs</p>
            <ul className="sidebar-subnav-list">
              {runItems.map((run) => {
                const runLabel = run.label || run.name;
                return (
                  <li key={run.path}>
                    <button
                      className={
                        selectedPath === run.path
                          ? "sidebar-subnav-item sidebar-run-item active"
                          : "sidebar-subnav-item sidebar-run-item"
                      }
                      onClick={() => onSelect(run.path)}
                      title={runLabel}
                      type="button"
                    >
                      <FileText size={14} />
                      <span className="sidebar-run-copy">
                        <span className="sidebar-run-name">{shortRunLabel(runLabel, 36)}</span>
                        <span className={`status-pill ${run.status}`}>
                          {run.status === "done" ? "Done" : "Fail"}
                        </span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </div>

      <div className="sidebar-footer">
        <a
          href="https://github.com/sansen405/genome-annotation-hidden-markov-model"
          rel="noopener noreferrer"
          target="_blank"
        >
          Documentation
        </a>
      </div>
    </>
  );
}

function RunProgressStrip({ elapsedSeconds, inputName, runStatus }) {
  const isRunning = runStatus === "running";
  const isFailed = runStatus === "failed";
  const isDone = runStatus === "done";

  return (
    <div
      className={`run-progress-strip ${isRunning ? "running" : ""} ${isFailed ? "failed" : ""} ${isDone ? "done" : ""}`.trim()}
      role="status"
    >
      <div className="run-progress-copy">
        <span className={`status-dot ${isRunning ? "running" : isFailed ? "failed" : "done"}`} />
        <div>
          <strong>
            {isRunning
              ? "Running prediction"
              : isFailed
                ? "Prediction failed"
                : "Prediction complete"}
          </strong>
          <p>
            {isRunning
              ? `Scoring splice sites and decoding ${inputName || "your FASTA"} with the HMM…`
              : isFailed
                ? "Check the error message below and try again from Tools."
                : `${inputName || "FASTA"} finished in ${elapsedSeconds}s. Opening results…`}
          </p>
        </div>
      </div>
      <div className="run-progress-meta">
        {isRunning && (
          <div aria-hidden="true" className="run-progress-bar">
            <span className="run-progress-bar-fill" />
          </div>
        )}
        <span className="run-progress-elapsed">{elapsedSeconds}s</span>
      </div>
    </div>
  );
}

function FolderPanel({
  breadcrumbs,
  directory,
  fileInputRef,
  newFolderName,
  onCreateFolder,
  onNavigate,
  onUpload,
  selectedPath,
  setNewFolderName,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const entries = directory?.entries ?? [];
  const folders = entries.filter((entry) => entry.type === "folder");
  const files = entries.filter((entry) => entry.type === "file");
  const runEntries = entries
    .filter((entry) => entry.type === "run")
    .sort((left, right) => right.name.localeCompare(left.name));
  const fastaFiles = files.filter((entry) => isFastaName(entry.name));
  const showUpload = selectedPath === "inputs" || selectedPath.startsWith("inputs/");
  const showRuns = selectedPath === "runs";

  const folderLabel = showRuns
    ? "Runs"
    : selectedPath.split("/").pop() || "Project";
  const sectionLabel = showUpload
    ? "Sequence files"
    : showRuns
      ? "Analysis runs"
      : "Folder";

  const sectionHint = showUpload
    ? "Upload FASTA sequence files for your project."
    : showRuns
      ? "Open a run to explore genes on the interactive genome track."
      : "Browse subfolders or return to Inputs to add sequence data.";

  const stats = showRuns
    ? [
        { label: "Total runs", value: runEntries.length },
        {
          label: "Completed",
          value: runEntries.filter((entry) => entry.status === "done").length,
        },
        {
          label: "Failed",
          value: runEntries.filter((entry) => entry.status === "failed").length,
        },
      ]
    : [
        { label: "Folders", value: folders.length },
        { label: "Files", value: files.length },
        { label: "FASTA ready", value: fastaFiles.length },
      ];

  return (
    <section className="workspace-folder">
      <header className="workspace-masthead">
        <TrackMotif className="workspace-motif" />
        <nav className="breadcrumbs workspace-breadcrumbs" aria-label="Breadcrumb">
          {breadcrumbs.map((crumb, index) => (
            <span className="breadcrumb-item" key={crumb.path}>
              {index > 0 && <ChevronRight size={14} />}
              <button onClick={() => onNavigate(crumb.path)} type="button">
                {crumb.label}
              </button>
            </span>
          ))}
        </nav>
        <div className="workspace-masthead-row">
          <div className="workspace-masthead-copy">
            <p className="home-kicker">{sectionLabel}</p>
            <h1 className="workspace-title">{folderLabel}</h1>
            <p className="workspace-lead">{sectionHint}</p>
          </div>
          {showUpload && (
            <form className="workspace-toolbar" onSubmit={onCreateFolder}>
              <input
                onChange={(event) => setNewFolderName(event.target.value)}
                placeholder="New folder name"
                value={newFolderName}
              />
              <button className="ghost-action" type="submit">
                <FolderPlus size={15} />
                Add folder
              </button>
            </form>
          )}
        </div>
        <dl className="workspace-stats">
          {stats.map((stat) => (
            <div key={stat.label}>
              <dt>{stat.label}</dt>
              <dd>{stat.value}</dd>
            </div>
          ))}
        </dl>
      </header>

      {showUpload && (
        <div
          className={isDragging ? "upload-rail dragging" : "upload-rail"}
          onClick={() => fileInputRef.current?.click()}
          onDragEnter={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            if (!event.currentTarget.contains(event.relatedTarget)) setIsDragging(false);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            setIsDragging(false);
            if (event.dataTransfer.files?.length) {
              onUpload(Array.from(event.dataTransfer.files));
            }
          }}
          role="button"
          tabIndex={0}
        >
          <UploadCloud size={20} strokeWidth={1.5} />
          <div className="upload-rail-copy">
            <strong>Drop FASTA files here</strong>
            <span>.fna · .fa · .fasta — or choose files from your machine</span>
          </div>
          <button className="ghost-action" type="button">
            Choose files
          </button>
          <input
            accept=".fna,.fa,.fasta"
            multiple
            onChange={(event) => onUpload(Array.from(event.target.files ?? []))}
            ref={fileInputRef}
            type="file"
          />
        </div>
      )}

      <div className="workspace-assets">
        {showRuns ? (
          runEntries.length === 0 ? (
            <div className="workspace-empty">
              <p>No runs yet</p>
              <span>Upload a FASTA file in Inputs, then run Gene annotation from Tools.</span>
              <button className="auremin-button" onClick={() => onNavigate("tools")} type="button">
                Go to Tools
                <ArrowRight size={16} />
              </button>
            </div>
          ) : (
            <ul className="run-card-list">
              {runEntries.map((entry) => (
                <li key={entry.path}>
                  <button
                    className="run-card-row"
                    onClick={() => onNavigate(entry.path)}
                    type="button"
                  >
                    <TrackMotif className="run-card-motif" />
                    <div className="run-card-body">
                      <div className="run-card-main">
                        <strong>{entry.fileName || entry.label}</strong>
                        <span>
                          {entry.date || "Completed run"}
                          {entry.genes ? ` · ${formatNumber(entry.genes)} genes` : ""}
                          {entry.totalBases ? ` · ${formatNumber(entry.totalBases)} bp` : ""}
                          {entry.elapsedMs ? ` · ${formatElapsed(entry.elapsedMs)}` : ""}
                        </span>
                      </div>
                      <span className={`status-pill ${entry.status === "done" ? "done" : "failed"}`}>
                        {entry.status === "done" ? "Done" : "Failed"}
                      </span>
                    </div>
                    <ArrowRight size={18} />
                  </button>
                </li>
              ))}
            </ul>
          )
        ) : entries.length === 0 ? (
          <div className="workspace-empty">
            <p>This folder is empty.</p>
            {showUpload && (
              <span>Use the upload strip above to add your first FASTA file.</span>
            )}
          </div>
        ) : (
          <>
            {folders.length > 0 && (
              <ul className="folder-entry-list">
                {folders.map((entry) => (
                  <li key={entry.path}>
                    <button
                      className="folder-entry"
                      onClick={() => onNavigate(entry.path)}
                      type="button"
                    >
                      <Folder size={18} />
                      <div className="folder-entry-copy">
                        <strong>{entry.name}</strong>
                        <span>Folder</span>
                      </div>
                      <ArrowRight size={16} />
                    </button>
                  </li>
                ))}
              </ul>
            )}

            {files.length > 0 && (
              <ul className="asset-list">
                {files.map((entry) => (
                  <li key={entry.path}>
                    <div className="asset-card asset-card--readonly">
                      <TrackMotif className="asset-card-motif" />
                      <div className="asset-card-body">
                        <div className="asset-card-main">
                          <strong>{entry.name}</strong>
                          <span>
                            {formatBytes(entry.size ?? 0)}
                            {isFastaName(entry.name) ? " · FASTA" : ""}
                          </span>
                        </div>
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </div>

      {fastaFiles.length > 0 && selectedPath === "inputs" && (
        <p className="workspace-footnote">
          {fastaFiles.length} FASTA file{fastaFiles.length === 1 ? "" : "s"} ready — go to Tools to
          run gene annotation.
        </p>
      )}
    </section>
  );
}

export default App;
