import {
  ArrowRight,
  Download,
  FileText,
  Folder,
  FolderPlus,
  Loader2,
  UploadCloud,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ExplorerContextMenu } from "./components/ExplorerContextMenu.jsx";
import { ProjectSidebar } from "./components/ProjectSidebar.jsx";
import { ProjectsPanel } from "./components/ProjectsPanel.jsx";
import { LandingScreen } from "./components/LandingScreen.jsx";
import { ToolsPanel } from "./components/ToolsPanel.jsx";
import { ResultsView } from "./ResultsView.jsx";
import { API_BASE, apiFetch, readJsonResponse } from "./lib/api.js";
import {
  isExplorerContextTarget,
  isExplorerDrag,
  isExplorerDraggable,
  isFolderDescendantPath,
  isInputFolderDropTarget,
  isMutableInputFile,
  isMutableInputFolder,
  readDraggedExplorerPath,
  setDraggedExplorerPath,
} from "./lib/explorer.js";

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isFastaName(name) {
  return /\.(fna|fa|fasta)$/i.test(name);
}

function isInputFilePath(path) {
  if (!path.startsWith("inputs/")) return false;
  const name = path.split("/").pop() || "";
  return name.includes(".") && !name.endsWith(".");
}

function getDirectoryListPath(selectedPath) {
  if (!selectedPath || selectedPath.startsWith("runs/")) return null;
  if (selectedPath.startsWith("tools")) return "inputs";
  if (selectedPath === "runs") return "runs";
  if (isInputFilePath(selectedPath)) {
    const parts = selectedPath.split("/");
    parts.pop();
    return parts.join("/") || "inputs";
  }
  return selectedPath;
}

function getUploadFolderPath(selectedPath) {
  if (isInputFilePath(selectedPath)) {
    return getDirectoryListPath(selectedPath);
  }
  return selectedPath.startsWith("inputs") ? selectedPath : "inputs";
}

function getFolderDisplayLabel(folderPath) {
  if (!folderPath || folderPath === "inputs") return "Inputs";
  return folderPath.split("/").pop() || "Inputs";
}

function getParentFolderPath(folderPath) {
  if (!folderPath || folderPath === "inputs") return null;
  const parts = folderPath.split("/").filter(Boolean);
  parts.pop();
  return parts.length ? parts.join("/") : "inputs";
}

function collectInputFastaFiles(nodes) {
  const files = [];
  for (const node of nodes ?? []) {
    if (node.type === "file" && isFastaName(node.name)) {
      files.push(node);
    } else if (node.type === "folder" && node.children?.length) {
      files.push(...collectInputFastaFiles(node.children));
    }
  }
  return files;
}

function getInputFileLocationLabel(filePath) {
  const parts = filePath.split("/").filter(Boolean);
  if (parts[0] === "inputs") parts.shift();
  if (parts.length <= 1) return "Inputs";
  parts.pop();
  return parts.join(" / ");
}

function getInputFileDisplayPath(filePath) {
  const name = filePath.split("/").pop() || filePath;
  const folder = getInputFileLocationLabel(filePath);
  if (!folder || folder === "Inputs") return `Inputs / ${name}`;
  return `Inputs / ${folder} / ${name}`;
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
        ) : variant === "landing" || variant === "projects" ? null : (
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
  const [runningInputPath, setRunningInputPath] = useState("");
  const [runningModelLabel, setRunningModelLabel] = useState("");
  const [runProgress, setRunProgress] = useState({ current: 0, total: 0 });
  const [pretrainedModels, setPretrainedModels] = useState([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [errorMessage, setErrorMessage] = useState("");
  const runAbortRef = useRef(null);
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
    apiFetch("/api/models")
      .then((models) => {
        setPretrainedModels(models);
        setSelectedModelId((current) => {
          if (current && models.some((model) => model.id === current && model.available)) {
            return current;
          }
          const preferred =
            models.find((model) => model.available && model.recommended) ??
            models.find((model) => model.available) ??
            models[0];
          return preferred?.id ?? "";
        });
      })
      .catch((error) => setErrorMessage(error.message));
  }, []);

  useEffect(() => {
    if (!activeProjectId) return;
    refreshProject(activeProjectId).catch((error) => setErrorMessage(error.message));
  }, [activeProjectId, refreshProject]);

  useEffect(() => {
    const folderPath = getDirectoryListPath(selectedPath);
    if (!activeProjectId || !folderPath) return;
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

  const inputFastaFiles = useMemo(() => {
    const inputsNode = tree.find((node) => node.name === "inputs");
    return collectInputFastaFiles(inputsNode?.children ?? []).sort((left, right) =>
      left.name.localeCompare(right.name)
    );
  }, [tree]);

  const breadcrumbs = useMemo(() => {
    if (!projectMeta) return [];
    if (selectedPath === "tools") {
      return [
        { label: "Inputs", path: "inputs" },
        { label: "Tools", path: "tools" },
      ];
    }
    const parts = selectedPath.split("/").filter(Boolean);
    if (parts.length === 0) {
      return [{ label: "Inputs", path: "inputs" }];
    }

    const sectionLabel = (part) => {
      if (part === "inputs") return "Inputs";
      if (part === "tools") return "Tools";
      if (part === "runs") return "Analyses";
      return part;
    };

    const crumbs = [];
    let current = "";
    for (let index = 0; index < parts.length; index += 1) {
      const part = parts[index];
      current = current ? `${current}/${part}` : part;
      crumbs.push({ label: sectionLabel(part), path: current });
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
    const parentPath = getUploadFolderPath(selectedPath);
    if (!activeProjectId || !name) return;
    await apiFetch(`/api/projects/${activeProjectId}/folders`, {
      body: JSON.stringify({ name, parentPath: parentPath === "." ? "" : parentPath }),
      headers: { "Content-Type": "application/json" },
      method: "POST",
    });
    setNewFolderName("");
    await refreshProject(activeProjectId);
    await refreshDirectory(activeProjectId, parentPath);
  }

  async function handleRenameFile(filePath, newName) {
    if (!activeProjectId || !filePath || !newName) return;

    const payload = await apiFetch(`/api/projects/${activeProjectId}/files`, {
      body: JSON.stringify({ name: newName, path: filePath }),
      headers: { "Content-Type": "application/json" },
      method: "PATCH",
    });

    const folderPath = getDirectoryListPath(selectedPath) ?? "inputs";
    if (selectedPath === filePath) {
      setSelectedPath(payload.path);
    }

    await refreshProject(activeProjectId);
    await refreshDirectory(activeProjectId, folderPath);
  }

  async function handleDeleteFile(filePath) {
    if (!activeProjectId || !filePath) return;

    await apiFetch(`/api/projects/${activeProjectId}/files`, {
      body: JSON.stringify({ path: filePath }),
      headers: { "Content-Type": "application/json" },
      method: "DELETE",
    });

    const folderPath = getDirectoryListPath(selectedPath) ?? "inputs";
    if (selectedPath === filePath) {
      setSelectedPath(folderPath);
    }

    await refreshProject(activeProjectId);
    await refreshDirectory(activeProjectId, folderPath);
  }

  async function handleMoveFile(filePath, destinationFolder) {
    if (!activeProjectId || !filePath || !destinationFolder) return;

    const currentFolder = filePath.split("/").slice(0, -1).join("/");
    if (currentFolder === destinationFolder) return;

    const payload = await apiFetch(`/api/projects/${activeProjectId}/files`, {
      body: JSON.stringify({ folder: destinationFolder, path: filePath }),
      headers: { "Content-Type": "application/json" },
      method: "PATCH",
    });

    const viewFolder = getDirectoryListPath(selectedPath) ?? "inputs";
    if (selectedPath === filePath) {
      setSelectedPath(payload.path);
    }

    await refreshProject(activeProjectId);
    await refreshDirectory(activeProjectId, viewFolder);
    if (destinationFolder !== viewFolder) {
      await refreshDirectory(activeProjectId, destinationFolder);
    }
  }

  async function handleRenameFolder(folderPath, newName) {
    if (!activeProjectId || !folderPath || !newName) return;

    const payload = await apiFetch(`/api/projects/${activeProjectId}/folders`, {
      body: JSON.stringify({ name: newName, path: folderPath }),
      headers: { "Content-Type": "application/json" },
      method: "PATCH",
    });

    const viewFolder = getDirectoryListPath(selectedPath) ?? "inputs";
    if (selectedPath === folderPath || selectedPath.startsWith(`${folderPath}/`)) {
      const suffix = selectedPath === folderPath ? "" : selectedPath.slice(folderPath.length);
      setSelectedPath(`${payload.path}${suffix}`);
    }

    await refreshProject(activeProjectId);
    await refreshDirectory(activeProjectId, viewFolder);
  }

  async function handleDeleteFolder(folderPath) {
    if (!activeProjectId || !folderPath) return;

    await apiFetch(`/api/projects/${activeProjectId}/folders`, {
      body: JSON.stringify({ path: folderPath }),
      headers: { "Content-Type": "application/json" },
      method: "DELETE",
    });

    const parentPath = folderPath.split("/").slice(0, -1).join("/") || "inputs";
    if (selectedPath === folderPath || selectedPath.startsWith(`${folderPath}/`)) {
      setSelectedPath(parentPath);
    }

    const viewFolder = getDirectoryListPath(selectedPath) ?? "inputs";
    await refreshProject(activeProjectId);
    await refreshDirectory(activeProjectId, viewFolder === folderPath ? parentPath : viewFolder);
  }

  async function handleMoveFolder(folderPath, destinationFolder) {
    if (!activeProjectId || !folderPath || !destinationFolder) return;
    if (isFolderDescendantPath(folderPath, destinationFolder)) return;

    const currentParent = folderPath.split("/").slice(0, -1).join("/");
    if (currentParent === destinationFolder) return;

    const payload = await apiFetch(`/api/projects/${activeProjectId}/folders`, {
      body: JSON.stringify({ folder: destinationFolder, path: folderPath }),
      headers: { "Content-Type": "application/json" },
      method: "PATCH",
    });

    const viewFolder = getDirectoryListPath(selectedPath) ?? "inputs";
    if (selectedPath === folderPath || selectedPath.startsWith(`${folderPath}/`)) {
      const suffix = selectedPath === folderPath ? "" : selectedPath.slice(folderPath.length);
      setSelectedPath(`${payload.path}${suffix}`);
    }

    await refreshProject(activeProjectId);
    await refreshDirectory(activeProjectId, viewFolder);
    if (destinationFolder !== viewFolder) {
      await refreshDirectory(activeProjectId, destinationFolder);
    }
  }

  async function handleRenameRun(runPath, newName) {
    if (!activeProjectId || !runPath || !newName) return;

    const payload = await apiFetch(`/api/projects/${activeProjectId}/runs`, {
      body: JSON.stringify({ name: newName, path: runPath }),
      headers: { "Content-Type": "application/json" },
      method: "PATCH",
    });

    if (selectedPath === runPath) {
      setSelectedPath(payload.path);
    }

    await refreshProject(activeProjectId);
    const viewFolder = getDirectoryListPath(selectedPath);
    if (viewFolder) {
      await refreshDirectory(activeProjectId, viewFolder);
    }
  }

  async function handleDeleteRun(runPath) {
    if (!activeProjectId || !runPath) return;

    await apiFetch(`/api/projects/${activeProjectId}/runs`, {
      body: JSON.stringify({ path: runPath }),
      headers: { "Content-Type": "application/json" },
      method: "DELETE",
    });

    if (selectedPath === runPath) {
      setSelectedPath("runs");
    }

    await refreshProject(activeProjectId);
    if (selectedPath === "runs") {
      await refreshDirectory(activeProjectId, "runs");
    }
  }

  async function handleRunPredictions(inputPaths, modelId) {
    const paths = [...new Set(inputPaths)].filter(Boolean);
    if (!activeProjectId || paths.length === 0 || !modelId) return;

    const model = pretrainedModels.find((entry) => entry.id === modelId);
    const started = Date.now();
    const controller = new AbortController();
    runAbortRef.current = controller;

    setRunStatus("running");
    setRunProgress({ current: 0, total: paths.length });
    setRunningInputPath(paths[0]);
    setRunningModelLabel(model?.label ?? "");
    setElapsedSeconds(0);
    setErrorMessage("");

    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - started) / 1000));
    }, 1000);

    let lastRun = null;

    try {
      for (let index = 0; index < paths.length; index += 1) {
        if (controller.signal.aborted) break;

        const inputPath = paths[index];
        setRunProgress({ current: index + 1, total: paths.length });
        setRunningInputPath(inputPath);

        try {
          lastRun = await apiFetch(`/api/projects/${activeProjectId}/runs`, {
            body: JSON.stringify({ inputPath, modelId }),
            headers: { "Content-Type": "application/json" },
            method: "POST",
            signal: controller.signal,
          });
        } catch (error) {
          if (error.name === "AbortError") break;
          const fileLabel = inputPath.split("/").pop() || inputPath;
          throw new Error(
            paths.length > 1
              ? `${error.message} (failed on ${fileLabel})`
              : error.message
          );
        }
      }

      if (controller.signal.aborted) {
        setRunStatus("idle");
        setRunningInputPath("");
        setRunningModelLabel("");
        setRunProgress({ current: 0, total: 0 });
        return;
      }

      await refreshProject(activeProjectId);
      if (lastRun) {
        setSelectedPath(`runs/${lastRun.id}`);
      }
      setRunStatus("done");
      setRunProgress({ current: paths.length, total: paths.length });
      setElapsedSeconds(Math.max(1, Math.round((Date.now() - started) / 1000)));
      window.setTimeout(() => {
        setRunStatus("idle");
        setRunningInputPath("");
        setRunningModelLabel("");
        setRunProgress({ current: 0, total: 0 });
      }, 2400);
    } catch (error) {
      if (error.name === "AbortError") {
        setRunStatus("idle");
        setRunningInputPath("");
        setRunningModelLabel("");
        setRunProgress({ current: 0, total: 0 });
        return;
      }
      setRunStatus("failed");
      setErrorMessage(
        error.message === "Failed to fetch"
          ? "Cannot connect to Annotator on this machine. Restart the app to try again."
          : error.message
      );
    } finally {
      window.clearInterval(timer);
      runAbortRef.current = null;
    }
  }

  function handleStopPredictions() {
    runAbortRef.current?.abort();
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
        <TopBar onHome={handleBrandHome} variant="projects" />

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
      <div className="app-shell">
        <aside className="sidebar project-sidebar">
          <ProjectSidebar
            onBack={leaveProject}
            onDeleteFile={(filePath) =>
              handleDeleteFile(filePath).catch((error) => setErrorMessage(error.message))
            }
            onDeleteFolder={(folderPath) =>
              handleDeleteFolder(folderPath).catch((error) => setErrorMessage(error.message))
            }
            onDeleteRun={(runPath) =>
              handleDeleteRun(runPath).catch((error) => setErrorMessage(error.message))
            }
            onMoveFile={(filePath, destinationFolder) =>
              handleMoveFile(filePath, destinationFolder).catch((error) =>
                setErrorMessage(error.message)
              )
            }
            onMoveFolder={(folderPath, destinationFolder) =>
              handleMoveFolder(folderPath, destinationFolder).catch((error) =>
                setErrorMessage(error.message)
              )
            }
            onRenameFile={(filePath, newName) =>
              handleRenameFile(filePath, newName).catch((error) => setErrorMessage(error.message))
            }
            onRenameFolder={(folderPath, newName) =>
              handleRenameFolder(folderPath, newName).catch((error) => setErrorMessage(error.message))
            }
            onRenameRun={(runPath, newName) =>
              handleRenameRun(runPath, newName).catch((error) => setErrorMessage(error.message))
            }
            onSelect={setSelectedPath}
            projectId={activeProjectId}
            projectName={projectMeta?.name}
            selectedPath={selectedPath}
            tree={tree}
          />
        </aside>

        <main className="content project-content">
          {errorMessage && <p className="page-error">{errorMessage}</p>}

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
          ) : selectedPath.startsWith("tools") ? (
            <ToolsPanel
              elapsedSeconds={elapsedSeconds}
              fastaFiles={inputFastaFiles}
              getLocationLabel={getInputFileLocationLabel}
              onNavigate={setSelectedPath}
              onRunPredictions={handleRunPredictions}
              onStopPredictions={handleStopPredictions}
              runProgress={runProgress}
              runErrorMessage={runStatus === "failed" ? errorMessage : ""}
              onSelectModel={setSelectedModelId}
              onSelectTool={(toolId) => setSelectedPath(toolId ? `tools/${toolId}` : "tools")}
              pretrainedModels={pretrainedModels}
              runStatus={runStatus}
              runningInputPath={runningInputPath}
              runningModelLabel={runningModelLabel}
              selectedModelId={selectedModelId}
              toolId={selectedPath === "tools" ? "" : selectedPath.split("/")[1] ?? ""}
            />
          ) : selectedPath === "runs" ? (
            <AnalysesPanel
              directory={directory}
              onDeleteRun={(runPath) =>
                handleDeleteRun(runPath).catch((error) => setErrorMessage(error.message))
              }
              onNavigate={setSelectedPath}
              onRenameRun={(runPath, newName) =>
                handleRenameRun(runPath, newName).catch((error) => setErrorMessage(error.message))
              }
            />
          ) : selectedPath.startsWith("inputs") || isInputFilePath(selectedPath) ? (
            <InputsPanel
              directory={directory}
              fileInputRef={fileInputRef}
              newFolderName={newFolderName}
              onCreateFolder={handleCreateFolder}
              onDeleteFile={(filePath) =>
                handleDeleteFile(filePath).catch((error) => setErrorMessage(error.message))
              }
              onDeleteFolder={(folderPath) =>
                handleDeleteFolder(folderPath).catch((error) => setErrorMessage(error.message))
              }
              onMoveFile={(filePath, destinationFolder) =>
                handleMoveFile(filePath, destinationFolder).catch((error) =>
                  setErrorMessage(error.message)
                )
              }
              onMoveFolder={(folderPath, destinationFolder) =>
                handleMoveFolder(folderPath, destinationFolder).catch((error) =>
                  setErrorMessage(error.message)
                )
              }
              onNavigate={setSelectedPath}
              onRenameFile={(filePath, newName) =>
                handleRenameFile(filePath, newName).catch((error) => setErrorMessage(error.message))
              }
              onRenameFolder={(folderPath, newName) =>
                handleRenameFolder(folderPath, newName).catch((error) => setErrorMessage(error.message))
              }
              onUpload={(files) =>
                handleUploadFiles(files, getUploadFolderPath(selectedPath)).catch((error) =>
                  setErrorMessage(error.message)
                )
              }
              projectId={activeProjectId}
              selectedPath={selectedPath}
              setNewFolderName={setNewFolderName}
            />
          ) : null}
        </main>
      </div>
    </div>
  );
}

function formatFastaPreview(lines) {
  const output = [];
  let sequenceParts = [];

  const flushSequence = () => {
    if (sequenceParts.length === 0) return;
    output.push(sequenceParts.join(""));
    sequenceParts = [];
  };

  for (const line of lines) {
    if (line.startsWith(">")) {
      flushSequence();
      output.push(line);
    } else if (line.trim()) {
      sequenceParts.push(line.trim());
    } else {
      flushSequence();
      output.push(line);
    }
  }

  flushSequence();
  return output.join("\n");
}

function formatPreviewText(lines, fileName) {
  if (!lines?.length) return "";
  if (isFastaName(fileName)) return formatFastaPreview(lines);
  return lines.join("\n");
}

function inputFileDownloadUrl(projectId, filePath) {
  return `${API_BASE}/api/projects/${projectId}/files/download?path=${encodeURIComponent(filePath)}`;
}

function InputsPanel({
  directory,
  fileInputRef,
  newFolderName,
  onCreateFolder,
  onDeleteFile,
  onDeleteFolder,
  onMoveFile,
  onMoveFolder,
  onNavigate,
  onRenameFile,
  onRenameFolder,
  onUpload,
  projectId,
  selectedPath,
  setNewFolderName,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [filePreview, setFilePreview] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState("");
  const [previewLineBudget, setPreviewLineBudget] = useState(0);
  const [previewClipHeight, setPreviewClipHeight] = useState(null);
  const [renamingPath, setRenamingPath] = useState("");
  const [renameValue, setRenameValue] = useState("");
  const [contextMenu, setContextMenu] = useState(null);
  const [dropTarget, setDropTarget] = useState("");
  const renameCommittingRef = useRef(false);
  const draggingPathRef = useRef("");
  const previewScrollRef = useRef(null);
  const entries = directory?.entries ?? [];
  const folderPath = getUploadFolderPath(selectedPath);
  const selectedFile = isInputFilePath(selectedPath)
    ? entries.find((entry) => entry.path === selectedPath) ??
      { name: selectedPath.split("/").pop(), path: selectedPath, type: "file" }
    : null;
  const folders = entries.filter((entry) => entry.type === "folder");
  const files = entries.filter((entry) => entry.type === "file");
  const folderLabel = getFolderDisplayLabel(folderPath);
  const parentFolderPath = getParentFolderPath(folderPath);
  const parentFolderLabel = parentFolderPath ? getFolderDisplayLabel(parentFolderPath) : "";
  const fastaCount = files.filter((entry) => isFastaName(entry.name)).length;

  function beginRename(entry) {
    setRenamingPath(entry.path);
    setRenameValue(entry.name);
  }

  function cancelRename() {
    setRenamingPath("");
    setRenameValue("");
  }

  async function commitRename(entry) {
    if (renameCommittingRef.current) return;

    const trimmed = renameValue.trim();
    const previousPath = entry.path;
    if (!trimmed || trimmed === entry.name) {
      cancelRename();
      return;
    }

    renameCommittingRef.current = true;
    cancelRename();
    try {
      if (entry.type === "folder") {
        await onRenameFolder(previousPath, trimmed);
      } else {
        await onRenameFile(previousPath, trimmed);
      }
    } finally {
      renameCommittingRef.current = false;
    }
  }

  async function handleDelete(entry) {
    const message =
      entry.type === "folder"
        ? `Delete folder "${entry.name}" and everything inside it? This cannot be undone.`
        : `Delete "${entry.name}"? This cannot be undone.`;
    if (!window.confirm(message)) return;
    cancelRename();
    if (entry.type === "folder") {
      await onDeleteFolder(entry.path);
    } else {
      await onDeleteFile(entry.path);
    }
  }

  function openContextMenu(event, entry) {
    if (!isExplorerContextTarget(entry)) return;
    event.preventDefault();
    setContextMenu({ entry, x: event.clientX, y: event.clientY });
  }

  function handleItemDragStart(event, entry) {
    if (!isExplorerDraggable(entry)) return;
    draggingPathRef.current = entry.path;
    setDraggedExplorerPath(event.dataTransfer, entry.path);
  }

  function handleItemDragEnd() {
    draggingPathRef.current = "";
    setDropTarget("");
  }

  function handleFolderDragOver(event, targetFolderPath) {
    if (!isExplorerDrag(event.dataTransfer)) return;
    if (!isInputFolderDropTarget(targetFolderPath)) return;

    const draggedPath = draggingPathRef.current;
    if (draggedPath) {
      if (isMutableInputFolder(draggedPath) && isFolderDescendantPath(draggedPath, targetFolderPath)) {
        return;
      }
      const currentParent = draggedPath.split("/").slice(0, -1).join("/");
      if (currentParent === targetFolderPath) return;
    }

    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = "move";
    setDropTarget(targetFolderPath);
  }

  function handleFolderDrop(event, targetFolderPath) {
    const draggedPath = readDraggedExplorerPath(event.dataTransfer) || draggingPathRef.current;
    draggingPathRef.current = "";
    setDropTarget("");
    if (!draggedPath) return;

    event.preventDefault();
    event.stopPropagation();
    if (isMutableInputFolder(draggedPath)) {
      onMoveFolder(draggedPath, targetFolderPath);
    } else {
      onMoveFile(draggedPath, targetFolderPath);
    }
  }

  function handleDownload(entry) {
    const link = document.createElement("a");
    link.href = inputFileDownloadUrl(projectId, entry.path);
    link.download = entry.name;
    link.click();
  }

  useEffect(() => {
    if (renamingPath && !entries.some((entry) => entry.path === renamingPath)) {
      cancelRename();
    }
  }, [entries, renamingPath]);

  useEffect(() => {
    if (!selectedFile?.path) {
      setPreviewLineBudget(0);
      setPreviewClipHeight(null);
      return;
    }

    const node = previewScrollRef.current;
    if (!node) return;

    const measure = () => {
      const lineHeight = 11 * 1.55;
      const paddingY = 28;
      const charsPerLine = Math.max(24, Math.floor((node.clientWidth - 32) / 6.6));
      const visibleLines = Math.max(8, Math.floor((node.clientHeight - paddingY) / lineHeight));
      const clipHeight = visibleLines * lineHeight + paddingY;
      const isFasta = isFastaName(selectedFile.name);
      const requestLines = isFasta
        ? Math.min(800, Math.max(12, Math.ceil((visibleLines * charsPerLine) / 70) + 2))
        : Math.min(800, visibleLines + 2);

      setPreviewClipHeight(clipHeight);
      setPreviewLineBudget((current) => (current === requestLines ? current : requestLines));
    };

    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, [selectedFile?.name, selectedFile?.path]);

  useEffect(() => {
    if (!projectId || !selectedFile?.path || previewLineBudget < 1) {
      if (!selectedFile?.path) {
        setFilePreview(null);
        setPreviewError("");
      }
      return;
    }

    setPreviewLoading(true);
    setPreviewError("");
    apiFetch(
      `/api/projects/${projectId}/files/preview?path=${encodeURIComponent(selectedFile.path)}&lines=${previewLineBudget}`
    )
      .then((payload) => setFilePreview(payload))
      .catch((error) => {
        setFilePreview(null);
        setPreviewError(error.message);
      })
      .finally(() => setPreviewLoading(false));
  }, [previewLineBudget, projectId, selectedFile?.path]);

  if (selectedFile) {
    const downloadUrl = inputFileDownloadUrl(projectId, selectedFile.path);
    const isRenamingSelected = renamingPath === selectedFile.path;

    return (
      <section
        className="workspace-folder workspace-folder--file-detail"
        onContextMenu={(event) => openContextMenu(event, selectedFile)}
      >
        <header className="workspace-masthead workspace-masthead--file-detail">
          <div className="workspace-masthead-row">
            <div className="workspace-masthead-copy">
              {isRenamingSelected ? (
                <form
                  className="file-detail-rename"
                  onSubmit={(event) => {
                    event.preventDefault();
                    commitRename(selectedFile);
                  }}
                >
                  <input
                    autoFocus
                    className="file-detail-rename-input workspace-title-input"
                    onBlur={() => commitRename(selectedFile)}
                    onChange={(event) => setRenameValue(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Escape") cancelRename();
                    }}
                    value={renameValue}
                  />
                </form>
              ) : (
                <h1 className="workspace-title workspace-title--literal">{selectedFile.name}</h1>
              )}
              <p className="workspace-lead">Preview and download this sequence file.</p>
            </div>
          </div>
          <div className="file-detail-meta">
            <dl className="workspace-stats workspace-stats--duo workspace-stats--inline">
              <div>
                <dt>Size</dt>
                <dd>{formatBytes(selectedFile.size ?? 0)}</dd>
              </div>
              <div>
                <dt>Location</dt>
                <dd>{folderLabel}</dd>
              </div>
            </dl>
            <div className="file-preview-actions">
              <a className="ghost-action file-detail-download" download href={downloadUrl}>
                <Download size={15} />
                Download
              </a>
            </div>
          </div>
        </header>

        <div className="workspace-assets workspace-assets--file-detail">
          <div className="file-preview-panel">
            <p className="tool-run-label">Preview</p>
            <div className="file-preview-scroll" ref={previewScrollRef}>
              {previewLoading ? (
                <div className="file-preview-state">
                  <Loader2 className="spin" size={18} />
                  <span>Loading preview…</span>
                </div>
              ) : previewError ? (
                <div className="file-preview-state">
                  <span>{previewError}</span>
                </div>
              ) : (
                <div
                  className="file-preview-code"
                  style={previewClipHeight ? { maxHeight: previewClipHeight } : undefined}
                >
                  {formatPreviewText(filePreview?.lines ?? [], selectedFile.name)}
                </div>
              )}
            </div>
          </div>

          <div className="file-detail-footer">
            {!previewLoading && !previewError && filePreview?.truncated ? (
              <p className="file-preview-footnote">
                Showing first {filePreview.lines.length.toLocaleString()} lines of{" "}
                {filePreview.lineCount.toLocaleString()}.
              </p>
            ) : null}
            <button className="tool-run-back" onClick={() => onNavigate(folderPath)} type="button">
              Back to {folderLabel}
            </button>
          </div>
        </div>
        <ExplorerContextMenu
          menu={contextMenu}
          onClose={() => setContextMenu(null)}
          onDelete={handleDelete}
          onDownload={handleDownload}
          onRename={beginRename}
        />
      </section>
    );
  }

  return (
    <section
      className={
        parentFolderPath
          ? "workspace-folder workspace-folder--inputs workspace-folder--with-back"
          : "workspace-folder workspace-folder--inputs"
      }
      onDragOver={(event) => {
        if (isExplorerDrag(event.dataTransfer)) {
          handleFolderDragOver(event, folderPath);
        }
      }}
      onDrop={(event) => {
        if (isExplorerDrag(event.dataTransfer)) {
          handleFolderDrop(event, folderPath);
        }
      }}
    >
      <header className="workspace-masthead workspace-masthead--file-detail">
        <div className="workspace-masthead-row">
          <div className="workspace-masthead-copy">
            <h1 className="workspace-title">{folderLabel}</h1>
            <p className="workspace-lead">
              Upload FASTA sequence files and organize them into folders for analysis.
            </p>
          </div>
          <form className="workspace-toolbar" onSubmit={onCreateFolder}>
            <input
              className="workspace-toolbar-input"
              onChange={(event) => setNewFolderName(event.target.value)}
              placeholder="New folder name"
              value={newFolderName}
            />
            <button className="ghost-action" type="submit">
              <FolderPlus size={15} />
              Add folder
            </button>
          </form>
        </div>
        <dl className="workspace-stats workspace-stats--duo">
          <div>
            <dt>Items</dt>
            <dd>{entries.length}</dd>
          </div>
          <div>
            <dt>FASTA ready</dt>
            <dd>{fastaCount}</dd>
          </div>
        </dl>
      </header>

      <div className="workspace-assets workspace-assets--file-detail">
        <p className="tool-run-label">Upload</p>
        <div
          className={isDragging ? "inputs-upload-card dragging" : "inputs-upload-card"}
          onClick={() => fileInputRef.current?.click()}
          onDragEnter={(event) => {
            event.preventDefault();
            if (!isExplorerDrag(event.dataTransfer)) {
              setIsDragging(true);
            }
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            if (!event.currentTarget.contains(event.relatedTarget)) {
              setIsDragging(false);
              setDropTarget("");
            }
          }}
          onDragOver={(event) => {
            if (isExplorerDrag(event.dataTransfer)) {
              handleFolderDragOver(event, folderPath);
              return;
            }
            event.preventDefault();
          }}
          onDrop={(event) => {
            if (isExplorerDrag(event.dataTransfer)) {
              setIsDragging(false);
              handleFolderDrop(event, folderPath);
              return;
            }
            event.preventDefault();
            setIsDragging(false);
            if (event.dataTransfer.files?.length) {
              onUpload(Array.from(event.dataTransfer.files));
            }
          }}
          role="button"
          tabIndex={0}
        >
          <div className="inputs-upload-body">
            <span className="tool-card-icon">
              <UploadCloud size={20} strokeWidth={1.5} />
            </span>
            <div className="inputs-upload-copy">
              <strong>Drop FASTA files here</strong>
              <p className="tool-card-copy">
                .fna · .fa · .fasta — uploads directly to {folderLabel}
              </p>
            </div>
          </div>
          <input
            accept=".fna,.fa,.fasta"
            multiple
            onChange={(event) => onUpload(Array.from(event.target.files ?? []))}
            ref={fileInputRef}
            type="file"
          />
        </div>

        {entries.length === 0 ? (
          <div className="workspace-empty">
            <p>This folder is empty.</p>
            <span>Use the upload card above to add your first FASTA file.</span>
          </div>
        ) : (
          <>
            <p className="tool-run-label">Contents</p>
            <div className="file-explorer">
              <div className="file-explorer-head">
                <span>Name</span>
                <span>Size</span>
              </div>
              <ul className="explorer-list">
                {folders.map((entry) => {
                  const isRenaming = renamingPath === entry.path;

                  return (
                  <li key={entry.path}>
                    <div
                      className={
                        dropTarget === entry.path
                          ? selectedPath === entry.path
                            ? "explorer-row explorer-row--folder active is-drop-target"
                            : "explorer-row explorer-row--folder is-drop-target"
                          : selectedPath === entry.path
                            ? "explorer-row explorer-row--folder active"
                            : "explorer-row explorer-row--folder"
                      }
                      draggable={!isRenaming && isExplorerDraggable(entry)}
                      onContextMenu={(event) => openContextMenu(event, entry)}
                      onDragEnd={handleItemDragEnd}
                      onDragLeave={() => setDropTarget("")}
                      onDragOver={(event) => handleFolderDragOver(event, entry.path)}
                      onDragStart={(event) => handleItemDragStart(event, entry)}
                      onDrop={(event) => handleFolderDrop(event, entry.path)}
                    >
                      <Folder size={15} strokeWidth={1.75} />
                      {isRenaming ? (
                        <form
                          className="explorer-row-main explorer-row-rename"
                          onSubmit={(event) => {
                            event.preventDefault();
                            commitRename(entry);
                          }}
                        >
                          <input
                            autoFocus
                            className="explorer-row-rename-input"
                            onBlur={() => commitRename(entry)}
                            onChange={(event) => setRenameValue(event.target.value)}
                            onKeyDown={(event) => {
                              if (event.key === "Escape") cancelRename();
                            }}
                            value={renameValue}
                          />
                        </form>
                      ) : (
                        <button
                          className="explorer-row-main explorer-row-main--button"
                          onClick={() => onNavigate(entry.path)}
                          type="button"
                        >
                          <strong>{entry.name}</strong>
                          <span>—</span>
                        </button>
                      )}
                    </div>
                  </li>
                  );
                })}
                {files.map((entry) => {
                  const displayPath = getInputFileDisplayPath(entry.path);
                  const isRenaming = renamingPath === entry.path;

                  return (
                  <li key={entry.path}>
                    <div
                      className={
                        selectedPath === entry.path
                          ? "explorer-row explorer-row--file active"
                          : "explorer-row explorer-row--file"
                      }
                      draggable={!isRenaming && isExplorerDraggable(entry)}
                      onContextMenu={(event) => openContextMenu(event, entry)}
                      onDragEnd={handleItemDragEnd}
                      onDragStart={(event) => handleItemDragStart(event, entry)}
                    >
                      {isRenaming ? (
                        <form
                          className="explorer-row-main explorer-row-rename"
                          onSubmit={(event) => {
                            event.preventDefault();
                            commitRename(entry);
                          }}
                        >
                          <input
                            autoFocus
                            className="explorer-row-rename-input"
                            onBlur={() => commitRename(entry)}
                            onChange={(event) => setRenameValue(event.target.value)}
                            onKeyDown={(event) => {
                              if (event.key === "Escape") cancelRename();
                            }}
                            value={renameValue}
                          />
                          <span className="explorer-row-path">{displayPath}</span>
                        </form>
                      ) : (
                        <button
                          className="explorer-row-main explorer-row-main--button"
                          onClick={() => onNavigate(entry.path)}
                          type="button"
                        >
                          <strong>{entry.name}</strong>
                          <span className="explorer-row-path">{displayPath}</span>
                        </button>
                      )}
                      <span className="explorer-row-size">{formatBytes(entry.size ?? 0)}</span>
                    </div>
                  </li>
                  );
                })}
              </ul>
            </div>
          </>
        )}

        {parentFolderPath ? (
          <div className="file-detail-footer">
            <button className="tool-run-back" onClick={() => onNavigate(parentFolderPath)} type="button">
              Back to {parentFolderLabel}
            </button>
          </div>
        ) : null}
      </div>
      <ExplorerContextMenu
        menu={contextMenu}
        onClose={() => setContextMenu(null)}
        onDelete={handleDelete}
        onDownload={handleDownload}
        onRename={beginRename}
      />
    </section>
  );
}

function AnalysesPanel({ directory, onDeleteRun, onNavigate, onRenameRun }) {
  const [contextMenu, setContextMenu] = useState(null);
  const [renamingPath, setRenamingPath] = useState("");
  const [renameValue, setRenameValue] = useState("");
  const renameCommittingRef = useRef(false);
  const runEntries = (directory?.entries ?? [])
    .filter((entry) => entry.type === "run")
    .sort((left, right) => right.name.localeCompare(left.name));

  function beginRename(entry) {
    setRenamingPath(entry.path);
    setRenameValue(entry.label || entry.name);
  }

  function cancelRename() {
    setRenamingPath("");
    setRenameValue("");
  }

  async function commitRename(entry) {
    if (renameCommittingRef.current) return;

    const trimmed = renameValue.trim();
    if (!trimmed || trimmed === (entry.label || entry.name)) {
      cancelRename();
      return;
    }

    renameCommittingRef.current = true;
    cancelRename();
    try {
      await onRenameRun(entry.path, trimmed);
    } finally {
      renameCommittingRef.current = false;
    }
  }

  async function handleDelete(entry) {
    if (!window.confirm(`Delete analysis "${entry.label || entry.name}"? This cannot be undone.`)) {
      return;
    }
    cancelRename();
    await onDeleteRun(entry.path);
  }

  function openContextMenu(event, entry) {
    if (!isExplorerContextTarget(entry)) return;
    event.preventDefault();
    setContextMenu({ entry, x: event.clientX, y: event.clientY });
  }

  return (
    <section className="workspace-folder">
      <header className="workspace-masthead workspace-masthead--file-detail">
        <div className="workspace-masthead-copy">
          <h1 className="workspace-title">Analyses</h1>
          <p className="workspace-lead">
            Open a run to explore genes on the interactive genome track.
          </p>
        </div>
      </header>

      <div className="workspace-assets workspace-assets--file-detail">
        {runEntries.length === 0 ? (
          <div className="workspace-empty workspace-empty--compact">
            <p>No runs yet</p>
            <span>Upload a FASTA file in Inputs, then run Gene annotation from Tools.</span>
            <button className="auremin-button" onClick={() => onNavigate("tools")} type="button">
              Go to Tools
              <ArrowRight size={16} />
            </button>
          </div>
        ) : (
          <ul className="explorer-list">
            {runEntries.map((entry) => {
              const isRenaming = renamingPath === entry.path;

              return (
              <li key={entry.path}>
                <div
                  className="explorer-row explorer-row--analysis"
                  onContextMenu={(event) => openContextMenu(event, entry)}
                >
                  <FileText size={16} strokeWidth={1.75} />
                  {isRenaming ? (
                    <form
                      className="explorer-row-main explorer-row-rename"
                      onSubmit={(event) => {
                        event.preventDefault();
                        commitRename(entry);
                      }}
                    >
                      <input
                        autoFocus
                        className="explorer-row-rename-input"
                        onBlur={() => commitRename(entry)}
                        onChange={(event) => setRenameValue(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Escape") cancelRename();
                        }}
                        value={renameValue}
                      />
                    </form>
                  ) : (
                    <button
                      className="explorer-row-main explorer-row-main--button"
                      onClick={() => onNavigate(entry.path)}
                      type="button"
                    >
                      <strong>{entry.fileName || entry.label}</strong>
                      <span>
                        {entry.date || "Run"}
                        {entry.genes ? ` · ${formatNumber(entry.genes)} genes` : ""}
                      </span>
                    </button>
                  )}
                  <span className={`status-pill ${entry.status === "done" ? "done" : "failed"}`}>
                    {entry.status === "done" ? "Done" : "Fail"}
                  </span>
                </div>
              </li>
              );
            })}
          </ul>
        )}
      </div>
      <ExplorerContextMenu
        menu={contextMenu}
        onClose={() => setContextMenu(null)}
        onDelete={handleDelete}
        onRename={beginRename}
      />
    </section>
  );
}

export default App;
