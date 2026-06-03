import { ArrowRight, Dna, Play, Search, Square } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

export const WORKSPACE_TOOLS = [
  {
    id: "gene-annotation",
    name: "Gene annotation",
    description:
      "Decode genes with a 21-state HMM and CNN splice-site scores. Outputs an interactive genome track plus GFF3, CSV, and BED exports.",
    tag: "HMM + CNN",
  },
];

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getFileDisplayPath(path, getLocationLabel) {
  const name = path.split("/").pop() || path;
  const folder = getLocationLabel?.(path);
  if (!folder || folder === "Inputs") return `Inputs / ${name}`;
  return `Inputs / ${folder} / ${name}`;
}

export function ToolsPanel({
  elapsedSeconds = 0,
  fastaFiles,
  getLocationLabel,
  onNavigate,
  onRunPredictions,
  onSelectModel,
  onSelectTool,
  onStopPredictions,
  pretrainedModels = [],
  runErrorMessage = "",
  runProgress = { current: 0, total: 0 },
  runStatus,
  runningInputPath = "",
  runningModelLabel = "",
  selectedModelId = "",
  toolId = "",
}) {
  const selectedTool = WORKSPACE_TOOLS.find((tool) => tool.id === toolId) ?? null;
  const [fileSearch, setFileSearch] = useState("");
  const [checkedFilePaths, setCheckedFilePaths] = useState(() => new Set());

  const filteredFastaFiles = useMemo(() => {
    const query = fileSearch.trim().toLowerCase();
    if (!query) return fastaFiles;
    return fastaFiles.filter((entry) => {
      const displayPath = getFileDisplayPath(entry.path, getLocationLabel);
      const haystack = `${entry.name} ${entry.path} ${displayPath}`.toLowerCase();
      return haystack.includes(query);
    });
  }, [fastaFiles, fileSearch, getLocationLabel]);

  const checkedPaths = useMemo(
    () => [...checkedFilePaths].filter((path) => fastaFiles.some((entry) => entry.path === path)),
    [checkedFilePaths, fastaFiles]
  );

  const selectedModel =
    pretrainedModels.find((model) => model.id === selectedModelId) ?? null;
  const selectedModelReady = Boolean(selectedModel?.available);
  const isRunning = runStatus === "running";
  const allFilteredChecked =
    filteredFastaFiles.length > 0 &&
    filteredFastaFiles.every((entry) => checkedFilePaths.has(entry.path));

  function toggleFilePath(path) {
    setCheckedFilePaths((current) => {
      const next = new Set(current);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }

  function toggleAllFiltered() {
    setCheckedFilePaths((current) => {
      const next = new Set(current);
      if (allFilteredChecked) {
        for (const entry of filteredFastaFiles) next.delete(entry.path);
      } else {
        for (const entry of filteredFastaFiles) next.add(entry.path);
      }
      return next;
    });
  }

  useEffect(() => {
    if (toolId && !selectedTool) onSelectTool("");
  }, [onSelectTool, selectedTool, toolId]);

  useEffect(() => {
    if (!selectedTool) {
      setFileSearch("");
      setCheckedFilePaths(new Set());
    }
  }, [selectedTool]);

  if (!selectedTool) {
    return (
      <section className="workspace-folder workspace-folder--tools">
        <header className="workspace-masthead workspace-masthead--file-detail">
          <div className="workspace-masthead-row">
            <div className="workspace-masthead-copy">
              <h1 className="workspace-title">Tools</h1>
              <p className="workspace-lead">
                Pick a tool, choose an uploaded FASTA file, and run the pipeline.
              </p>
            </div>
          </div>
        </header>

        <div className="workspace-assets workspace-assets--file-detail">
          <p className="tool-run-label">Available</p>
          <ul className="tool-card-list">
            {WORKSPACE_TOOLS.map((tool) => (
              <li key={tool.id}>
                <button
                  className="tool-card"
                  onClick={() => onSelectTool(tool.id)}
                  type="button"
                >
                  <div className="tool-card-body">
                    <div className="tool-card-main">
                      <span className="tool-card-icon">
                        <Dna size={18} />
                      </span>
                      <div>
                        <strong>{tool.name}</strong>
                        <span className="tool-card-tag">{tool.tag}</span>
                      </div>
                    </div>
                    <p className="tool-card-copy">{tool.description}</p>
                  </div>
                  <ArrowRight size={18} />
                </button>
              </li>
            ))}
          </ul>
        </div>
      </section>
    );
  }

  const runningFileName = runningInputPath.split("/").pop() || "your FASTA";
  const showRunFooter = isRunning || runStatus === "failed" || runStatus === "done";
  const progressPercent =
    runProgress.total > 0
      ? Math.min(
          100,
          Math.round(
            ((runStatus === "done" ? runProgress.total : runProgress.current) / runProgress.total) *
              100
          )
        )
      : 0;
  const progressIndeterminate = isRunning && runProgress.total <= 1;
  const statusTitle =
    runStatus === "failed"
      ? "Prediction failed"
      : runStatus === "done"
        ? "Prediction complete"
        : runProgress.total > 1
          ? `Running ${runProgress.current} of ${runProgress.total}`
          : "Running prediction";
  const statusDetail =
    runStatus === "failed"
      ? runErrorMessage || `Could not finish decoding ${runningFileName}.`
      : runStatus === "done"
        ? `Finished ${runningFileName}${runningModelLabel ? ` with ${runningModelLabel}` : ""}.`
        : `Scoring and decoding ${runningFileName}${runningModelLabel ? ` with ${runningModelLabel}` : ""}…`;
  const annotateFileCount = isRunning ? runProgress.total : checkedPaths.length;
  const annotateButtonLabel = `Annotate ${annotateFileCount} file${annotateFileCount === 1 ? "" : "s"}`;

  return (
    <section className="workspace-folder workspace-folder--tool-run">
      <div className="tool-run-panel">
        <div className="tool-run-hero">
          <div className="tool-run-header">
            <span className="tool-card-icon tool-card-icon--large">
              <Dna size={22} />
            </span>
            <div>
              <h1 className="tool-run-title">{selectedTool.name}</h1>
              <p>{selectedTool.description}</p>
            </div>
          </div>

          {pretrainedModels.length > 0 ? (
            <div className="tool-model-picker">
              <p className="tool-run-label">Pretrained model</p>
              <ul className="tool-model-picker-list">
                {pretrainedModels.map((model) => {
                  const isSelected = selectedModelId === model.id;
                  return (
                    <li key={model.id}>
                      <button
                        aria-pressed={isSelected}
                        className={
                          !model.available
                            ? "tool-model-option tool-model-option--disabled"
                            : isSelected
                              ? "tool-model-option is-selected"
                              : "tool-model-option"
                        }
                        disabled={!model.available || isRunning}
                        onClick={() => onSelectModel(model.id)}
                        type="button"
                      >
                        <span className="tool-model-option-head">
                          <strong>{model.label}</strong>
                          <span className="tool-model-option-tags">
                            {model.tag ? (
                              <span className="tool-model-option-tag">{model.tag}</span>
                            ) : null}
                            <span className="tool-model-option-version">{model.version}</span>
                          </span>
                        </span>
                        <span className="tool-model-option-copy">{model.description}</span>
                        <span className="tool-model-option-status">
                          {!model.available
                            ? `Not installed${model.missing?.length ? ` · missing ${model.missing.join(", ")}` : ""}`
                            : "\u00a0"}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          ) : null}
        </div>

        <div className="tool-run-files">
          {fastaFiles.length === 0 ? (
            <div className="workspace-empty">
              <p>No FASTA files yet</p>
              <span>Upload sequence files in Inputs before running this tool.</span>
              <button className="auremin-button" onClick={() => onNavigate("inputs")} type="button">
                Go to Inputs
                <ArrowRight size={16} />
              </button>
            </div>
          ) : (
            <div className="tool-file-picker">
              <p className="tool-run-label">Find input file</p>
              <div className="tool-file-picker-toolbar">
                <label className="tool-file-picker-search genes-filter genes-filter--search">
                  <span className="genes-filter-control">
                    <Search size={15} />
                    <input
                      aria-label="Search input files"
                      disabled={isRunning}
                      onChange={(event) => setFileSearch(event.target.value)}
                      placeholder="Search by name or path…"
                      type="search"
                      value={fileSearch}
                    />
                  </span>
                </label>
                {filteredFastaFiles.length > 0 ? (
                  <button
                    className="tool-file-picker-select-all"
                    disabled={isRunning}
                    onClick={toggleAllFiltered}
                    type="button"
                  >
                    {allFilteredChecked ? "Clear shown" : "Select all shown"}
                  </button>
                ) : null}
              </div>

              <p className="tool-file-picker-count">
                {checkedPaths.length > 0
                  ? `${checkedPaths.length} selected`
                  : filteredFastaFiles.length === fastaFiles.length
                    ? `${fastaFiles.length} file${fastaFiles.length === 1 ? "" : "s"} in Inputs`
                    : `${filteredFastaFiles.length} of ${fastaFiles.length} files`}
              </p>

              {filteredFastaFiles.length === 0 ? (
                <div className="tool-file-picker-empty">
                  <p>No files match your search</p>
                  <span>Try a different name or folder path.</span>
                </div>
              ) : (
                <>
                  <p className="tool-run-label">Contents</p>
                  <div className="file-explorer tool-file-explorer">
                    <div className="file-explorer-head tool-file-explorer-head">
                      <span className="tool-file-explorer-head-pick" aria-hidden="true" />
                      <span>Name</span>
                      <span>Size</span>
                    </div>
                    <ul className="explorer-list tool-file-explorer-list">
                      {filteredFastaFiles.map((entry) => {
                        const isChecked = checkedFilePaths.has(entry.path);
                        const isRunningThis = isRunning && runningInputPath === entry.path;
                        const displayPath = getFileDisplayPath(entry.path, getLocationLabel);
                        const checkboxId = `tool-file-${entry.path.replace(/[^\w-]+/g, "-")}`;

                        return (
                          <li key={entry.path}>
                            <label
                              className={
                                isRunningThis
                                  ? "tool-file-explorer-row is-running"
                                  : isChecked
                                    ? "tool-file-explorer-row is-checked"
                                    : "tool-file-explorer-row"
                              }
                              htmlFor={checkboxId}
                            >
                              <input
                                checked={isChecked}
                                className="tool-file-explorer-checkbox"
                                disabled={isRunning}
                                id={checkboxId}
                                onChange={() => toggleFilePath(entry.path)}
                                type="checkbox"
                              />
                              <span className="tool-file-explorer-main">
                                <strong>{entry.name}</strong>
                                <span className="tool-file-explorer-path">{displayPath}</span>
                              </span>
                              <span className="tool-file-explorer-size">
                                {formatBytes(entry.size ?? 0)}
                                {isRunningThis ? " · running" : ""}
                              </span>
                            </label>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                </>
              )}

              <div className="tool-run-footer">
                {showRunFooter && runningInputPath ? (
                  <div
                    className={
                      runStatus === "failed"
                        ? "tool-run-footer-progress tool-run-footer-progress--failed"
                        : "tool-run-footer-progress"
                    }
                    role="status"
                  >
                    <div className="tool-run-footer-copy">
                      <strong>{statusTitle}</strong>
                      <p className="tool-run-footer-detail">
                        <span>{statusDetail}</span>
                        {isRunning ? (
                          <span className="tool-run-footer-elapsed">{elapsedSeconds}s</span>
                        ) : null}
                      </p>
                    </div>
                    <div
                      aria-hidden={runStatus === "failed"}
                      aria-valuemax={100}
                      aria-valuemin={0}
                      aria-valuenow={progressIndeterminate ? undefined : progressPercent}
                      className="tool-run-progress-track"
                      role="progressbar"
                    >
                      <span
                        className={
                          progressIndeterminate
                            ? "tool-run-progress-bar tool-run-progress-bar--indeterminate"
                            : "tool-run-progress-bar"
                        }
                        style={
                          progressIndeterminate ? undefined : { width: `${progressPercent}%` }
                        }
                      />
                    </div>
                  </div>
                ) : (
                  <div className="tool-run-footer-spacer" aria-hidden="true" />
                )}

                <div className="tool-run-footer-actions">
                  {isRunning ? (
                    <button
                      className="tool-file-picker-stop"
                      onClick={onStopPredictions}
                      type="button"
                    >
                      <Square size={15} fill="currentColor" />
                      Stop
                    </button>
                  ) : null}
                  <button
                    className="tool-file-picker-annotate"
                    disabled={
                      isRunning ||
                      checkedPaths.length === 0 ||
                      !selectedModelId ||
                      !selectedModelReady
                    }
                    onClick={() => onRunPredictions(checkedPaths, selectedModelId)}
                    type="button"
                  >
                    <Play size={16} />
                    {annotateButtonLabel}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
