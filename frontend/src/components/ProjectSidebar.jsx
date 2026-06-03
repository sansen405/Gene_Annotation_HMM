import {
  ArrowLeft,
  ChevronDown,
  ChevronRight,
  Folder,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE } from "../lib/api.js";
import {
  isExplorerContextTarget,
  isExplorerDrag,
  isExplorerDraggable,
  isFolderDescendantPath,
  isInputFolderDropTarget,
  isMutableInputFolder,
  readDraggedExplorerPath,
  setDraggedExplorerPath,
} from "../lib/explorer.js";
import { ExplorerContextMenu } from "./ExplorerContextMenu.jsx";
import { WORKSPACE_TOOLS } from "./ToolsPanel.jsx";

const ROOT_SECTIONS = ["inputs", "tools", "runs"];

function inputFileDownloadUrl(projectId, filePath) {
  return `${API_BASE}/api/projects/${projectId}/files/download?path=${encodeURIComponent(filePath)}`;
}

function shortRunLabel(label, maxLength = 34) {
  if (!label || label.length <= maxLength) return label;
  return `${label.slice(0, maxLength - 1)}…`;
}

function collectAncestorPaths(path) {
  if (!path) return [];
  const parts = path.split("/").filter(Boolean);
  const ancestors = [];
  let current = "";
  for (const part of parts) {
    current = current ? `${current}/${part}` : part;
    ancestors.push(current);
  }
  return ancestors;
}

function getRenameLabel(entry) {
  if (entry.type === "run") return entry.label || entry.name;
  return entry.name;
}

function SidebarChevron({ expanded, hasChildren, onToggle }) {
  if (!hasChildren) {
    return <span aria-hidden="true" className="sidebar-tree-chevron sidebar-tree-chevron--spacer" />;
  }

  const Icon = expanded ? ChevronDown : ChevronRight;
  return (
    <button
      aria-expanded={expanded}
      aria-label={expanded ? "Collapse" : "Expand"}
      className="sidebar-tree-chevron"
      onClick={(event) => {
        event.stopPropagation();
        onToggle();
      }}
      type="button"
    >
      <Icon size={14} strokeWidth={2} />
    </button>
  );
}

function SidebarTreeRow({
  active = false,
  draggable = false,
  dropTarget = false,
  expanded = false,
  hasChildren = false,
  icon: Icon,
  label,
  onContextMenu,
  onDragEnd,
  onDragLeave,
  onDragOver,
  onDragStart,
  onDrop,
  onSelect,
  onToggle,
  variant = "section",
}) {
  return (
    <div
      className={`sidebar-tree-row-wrap ${variant === "leaf" ? "sidebar-tree-row-wrap--leaf" : ""}`.trim()}
      draggable={draggable}
      onContextMenu={onContextMenu}
      onDragEnd={onDragEnd}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDragStart={onDragStart}
      onDrop={onDrop}
    >
      <SidebarChevron expanded={expanded} hasChildren={hasChildren} onToggle={onToggle} />
      <button
        className={
          dropTarget
            ? `sidebar-tree-row sidebar-tree-row--${variant} is-drop-target`.trim()
            : active
              ? `sidebar-tree-row sidebar-tree-row--${variant} active`.trim()
              : `sidebar-tree-row sidebar-tree-row--${variant}`.trim()
        }
        onClick={onSelect}
        title={label}
        type="button"
      >
        {Icon ? <Icon size={15} strokeWidth={1.75} /> : null}
        <span className="sidebar-tree-label">{label}</span>
      </button>
    </div>
  );
}

function InputsTreeNodes({
  dropTargetPath,
  expandedPaths,
  nodes,
  onContextMenu,
  onDragEnd,
  onDragStart,
  onFolderDragLeave,
  onFolderDragOver,
  onFolderDrop,
  onRenameCommit,
  onSelect,
  onToggle,
  renamingPath,
  renameValue,
  setRenameValue,
  selectedPath,
}) {
  if (!nodes?.length) return null;

  return nodes.map((node) => {
    const isRenaming = renamingPath === node.path;

    if (node.type === "file" || node.type === "folder") {
      if (isRenaming) {
        return (
          <li key={node.path}>
            <form
              className="sidebar-tree-rename"
              onSubmit={(event) => {
                event.preventDefault();
                onRenameCommit(node);
              }}
            >
              <input
                autoFocus
                className="sidebar-tree-rename-input"
                onBlur={() => onRenameCommit(node)}
                onChange={(event) => setRenameValue(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Escape") onRenameCommit(node, true);
                }}
                value={renameValue}
              />
            </form>
          </li>
        );
      }
    }

    if (node.type === "file") {
      return (
        <li key={node.path}>
          <SidebarTreeRow
            active={selectedPath === node.path}
            draggable={isExplorerDraggable(node)}
            hasChildren={false}
            label={node.name}
            onContextMenu={(event) => onContextMenu(event, node)}
            onDragEnd={onDragEnd}
            onDragStart={(event) => onDragStart(event, node)}
            onSelect={() => onSelect(node.path)}
            onToggle={() => {}}
            variant="leaf"
          />
        </li>
      );
    }

    const children = node.children ?? [];
    const hasChildren = children.length > 0;
    const isExpanded = expandedPaths.has(node.path);

    return (
      <li className="sidebar-tree-branch" key={node.path}>
        <SidebarTreeRow
          active={selectedPath === node.path}
          draggable={isExplorerDraggable(node)}
          dropTarget={dropTargetPath === node.path}
          expanded={isExpanded}
          hasChildren={hasChildren}
          icon={Folder}
          label={node.name}
          onContextMenu={(event) => onContextMenu(event, node)}
          onDragEnd={onDragEnd}
          onDragLeave={onFolderDragLeave}
          onDragOver={(event) => onFolderDragOver(event, node.path)}
          onDragStart={(event) => onDragStart(event, node)}
          onDrop={(event) => onFolderDrop(event, node.path)}
          onSelect={() => onSelect(node.path)}
          onToggle={() => onToggle(node.path)}
          variant="leaf"
        />
        {hasChildren && isExpanded ? (
          <ul className="sidebar-tree-children">
            <InputsTreeNodes
              dropTargetPath={dropTargetPath}
              expandedPaths={expandedPaths}
              nodes={children}
              onContextMenu={onContextMenu}
              onDragEnd={onDragEnd}
              onDragStart={onDragStart}
              onFolderDragLeave={onFolderDragLeave}
              onFolderDragOver={onFolderDragOver}
              onFolderDrop={onFolderDrop}
              onRenameCommit={onRenameCommit}
              onSelect={onSelect}
              onToggle={onToggle}
              renamingPath={renamingPath}
              renameValue={renameValue}
              setRenameValue={setRenameValue}
              selectedPath={selectedPath}
            />
          </ul>
        ) : null}
      </li>
    );
  });
}

export function ProjectSidebar({
  onBack,
  onDeleteFile,
  onDeleteFolder,
  onDeleteRun,
  onMoveFile,
  onMoveFolder,
  onRenameFile,
  onRenameFolder,
  onRenameRun,
  onSelect,
  projectId,
  projectName,
  selectedPath,
  tree,
}) {
  const inputsNode = tree.find((node) => node.name === "inputs");
  const inputItems = inputsNode?.children ?? [];
  const runsNode = tree.find((node) => node.name === "runs");
  const runItems = runsNode?.children?.filter((node) => node.type === "run") ?? [];

  const [expandedPaths, setExpandedPaths] = useState(
    () => new Set(["inputs", "tools", "runs"])
  );
  const [contextMenu, setContextMenu] = useState(null);
  const [dropTargetPath, setDropTargetPath] = useState("");
  const [renamingPath, setRenamingPath] = useState("");
  const [renameValue, setRenameValue] = useState("");
  const renameCommittingRef = useRef(false);
  const draggingPathRef = useRef("");

  const togglePath = useCallback((path) => {
    setExpandedPaths((current) => {
      const next = new Set(current);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }, []);

  useEffect(() => {
    const pathsToOpen = new Set(ROOT_SECTIONS);
    if (selectedPath.startsWith("inputs")) {
      collectAncestorPaths(selectedPath).forEach((path) => pathsToOpen.add(path));
    }
    if (selectedPath.startsWith("tools")) pathsToOpen.add("tools");
    if (selectedPath === "runs" || selectedPath.startsWith("runs/")) pathsToOpen.add("runs");

    setExpandedPaths((current) => new Set([...current, ...pathsToOpen]));
  }, [selectedPath]);

  function beginRename(entry) {
    setRenamingPath(entry.path);
    setRenameValue(getRenameLabel(entry));
  }

  function cancelRename() {
    setRenamingPath("");
    setRenameValue("");
  }

  async function commitRename(entry, cancelled = false) {
    if (renameCommittingRef.current) return;

    const trimmed = renameValue.trim();
    const previousPath = entry.path;
    const currentLabel = getRenameLabel(entry);
    if (cancelled || !trimmed || trimmed === currentLabel) {
      cancelRename();
      return;
    }

    renameCommittingRef.current = true;
    cancelRename();
    try {
      if (entry.type === "folder") {
        await onRenameFolder(previousPath, trimmed);
      } else if (entry.type === "run") {
        await onRenameRun(previousPath, trimmed);
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
        : entry.type === "run"
          ? `Delete analysis "${getRenameLabel(entry)}"? This cannot be undone.`
          : `Delete "${entry.name}"? This cannot be undone.`;
    if (!window.confirm(message)) return;
    cancelRename();
    if (entry.type === "folder") {
      await onDeleteFolder(entry.path);
    } else if (entry.type === "run") {
      await onDeleteRun(entry.path);
    } else {
      await onDeleteFile(entry.path);
    }
  }

  function openContextMenu(event, entry) {
    if (!isExplorerContextTarget(entry)) return;
    event.preventDefault();
    event.stopPropagation();
    setContextMenu({ entry, x: event.clientX, y: event.clientY });
  }

  function handleItemDragStart(event, entry) {
    if (!isExplorerDraggable(entry)) return;
    draggingPathRef.current = entry.path;
    setDraggedExplorerPath(event.dataTransfer, entry.path);
  }

  function handleItemDragEnd() {
    draggingPathRef.current = "";
    setDropTargetPath("");
  }

  function handleFolderDragOver(event, folderPath) {
    if (!isExplorerDrag(event.dataTransfer)) return;
    if (!isInputFolderDropTarget(folderPath)) return;

    const draggedPath = draggingPathRef.current;
    if (draggedPath) {
      if (isMutableInputFolder(draggedPath) && isFolderDescendantPath(draggedPath, folderPath)) {
        return;
      }
      const currentParent = draggedPath.split("/").slice(0, -1).join("/");
      if (currentParent === folderPath) return;
    }

    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = "move";
    setDropTargetPath(folderPath);
  }

  function handleFolderDrop(event, folderPath) {
    const draggedPath = readDraggedExplorerPath(event.dataTransfer) || draggingPathRef.current;
    draggingPathRef.current = "";
    setDropTargetPath("");
    if (!draggedPath) return;

    event.preventDefault();
    event.stopPropagation();
    if (isMutableInputFolder(draggedPath)) {
      onMoveFolder(draggedPath, folderPath);
    } else {
      onMoveFile(draggedPath, folderPath);
    }
  }

  function handleDownload(entry) {
    const link = document.createElement("a");
    link.href = inputFileDownloadUrl(projectId, entry.path);
    link.download = entry.name;
    link.click();
  }

  const inputsExpanded = expandedPaths.has("inputs");
  const toolsExpanded = expandedPaths.has("tools");
  const analysesExpanded = expandedPaths.has("runs");
  const inputsActive = selectedPath.startsWith("inputs");
  const toolsActive = selectedPath.startsWith("tools");
  const analysesActive = selectedPath === "runs" || selectedPath.startsWith("runs/");

  return (
    <>
      <div className="sidebar-head">
        <h2 className="sidebar-title">{projectName}</h2>
      </div>

      <div className="sidebar-body">
        <nav className="sidebar-tree" aria-label="Project workspace">
          <ul className="sidebar-tree-list">
            <li className="sidebar-tree-branch sidebar-tree-branch--section sidebar-tree-branch--inputs">
              <SidebarTreeRow
                active={inputsActive && selectedPath === "inputs"}
                dropTarget={dropTargetPath === "inputs"}
                expanded={inputsExpanded}
                hasChildren
                icon={Folder}
                label="Inputs"
                onDragLeave={() => setDropTargetPath("")}
                onDragOver={(event) => handleFolderDragOver(event, "inputs")}
                onDrop={(event) => handleFolderDrop(event, "inputs")}
                onSelect={() => onSelect("inputs")}
                onToggle={() => togglePath("inputs")}
                variant="section"
              />
              {inputsExpanded && inputItems.length > 0 ? (
                <ul className="sidebar-tree-children sidebar-tree-children--inputs">
                  <InputsTreeNodes
                    dropTargetPath={dropTargetPath}
                    expandedPaths={expandedPaths}
                    nodes={inputItems}
                    onContextMenu={openContextMenu}
                    onDragEnd={handleItemDragEnd}
                    onDragStart={handleItemDragStart}
                    onFolderDragLeave={() => setDropTargetPath("")}
                    onFolderDragOver={handleFolderDragOver}
                    onFolderDrop={handleFolderDrop}
                    onRenameCommit={commitRename}
                    onSelect={onSelect}
                    onToggle={togglePath}
                    renamingPath={renamingPath}
                    renameValue={renameValue}
                    setRenameValue={setRenameValue}
                    selectedPath={selectedPath}
                  />
                </ul>
              ) : null}
            </li>

            <li className="sidebar-tree-branch sidebar-tree-branch--section">
              <SidebarTreeRow
                active={toolsActive && selectedPath === "tools"}
                expanded={toolsExpanded}
                hasChildren
                icon={Folder}
                label="Tools"
                onSelect={() => onSelect("tools")}
                onToggle={() => togglePath("tools")}
                variant="section"
              />
              {toolsExpanded ? (
                <ul className="sidebar-tree-children sidebar-tree-children--flat">
                  {WORKSPACE_TOOLS.map((tool) => (
                    <li key={tool.id}>
                      <SidebarTreeRow
                        active={selectedPath === `tools/${tool.id}`}
                        hasChildren={false}
                        label={tool.name}
                        onSelect={() => onSelect(`tools/${tool.id}`)}
                        onToggle={() => {}}
                        variant="leaf"
                      />
                    </li>
                  ))}
                </ul>
              ) : null}
            </li>

            <li className="sidebar-tree-branch sidebar-tree-branch--section">
              <SidebarTreeRow
                active={analysesActive && selectedPath === "runs"}
                expanded={analysesExpanded}
                hasChildren
                icon={Folder}
                label="Analyses"
                onSelect={() => onSelect("runs")}
                onToggle={() => togglePath("runs")}
                variant="section"
              />
              {analysesExpanded && runItems.length > 0 ? (
                <ul className="sidebar-tree-children sidebar-tree-children--flat">
                  {runItems.map((run) => {
                    const runLabel = run.label || run.name;
                    const isRenaming = renamingPath === run.path;

                    if (isRenaming) {
                      return (
                        <li key={run.path}>
                          <form
                            className="sidebar-tree-rename"
                            onSubmit={(event) => {
                              event.preventDefault();
                              commitRename(run);
                            }}
                          >
                            <input
                              autoFocus
                              className="sidebar-tree-rename-input"
                              onBlur={() => commitRename(run)}
                              onChange={(event) => setRenameValue(event.target.value)}
                              onKeyDown={(event) => {
                                if (event.key === "Escape") commitRename(run, true);
                              }}
                              value={renameValue}
                            />
                          </form>
                        </li>
                      );
                    }

                    return (
                      <li key={run.path}>
                        <SidebarTreeRow
                          active={selectedPath === run.path}
                          hasChildren={false}
                          label={shortRunLabel(runLabel)}
                          onContextMenu={(event) => openContextMenu(event, run)}
                          onSelect={() => onSelect(run.path)}
                          onToggle={() => {}}
                          variant="leaf"
                        />
                      </li>
                    );
                  })}
                </ul>
              ) : null}
            </li>
          </ul>
        </nav>
      </div>

      <div className="sidebar-footer">
        <button className="sidebar-back" onClick={onBack} type="button">
          <ArrowLeft size={15} />
          All projects
        </button>
      </div>

      <ExplorerContextMenu
        menu={contextMenu}
        onClose={() => setContextMenu(null)}
        onDelete={handleDelete}
        onDownload={handleDownload}
        onRename={beginRename}
      />
    </>
  );
}
