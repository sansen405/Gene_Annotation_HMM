export const EXPLORER_DRAG_TYPE = "application/x-gene-hmm-explorer-path";
export const INPUT_FILE_DRAG_TYPE = EXPLORER_DRAG_TYPE;

export const LOCKED_ROOT_PATHS = new Set(["inputs", "tools", "runs"]);

export function isExplorerDrag(dataTransfer) {
  return dataTransfer?.types?.includes(EXPLORER_DRAG_TYPE) ?? false;
}

export function isInputFileDrag(dataTransfer) {
  return isExplorerDrag(dataTransfer);
}

export function isLockedRootPath(folderPath) {
  return LOCKED_ROOT_PATHS.has(folderPath);
}

export function isMutableInputFile(filePath) {
  if (!filePath?.startsWith("inputs/")) return false;
  const parts = filePath.split("/").filter(Boolean);
  return parts.length >= 2 && parts[0] === "inputs" && parts[parts.length - 1].includes(".");
}

export function isMutableInputFolder(folderPath) {
  if (!folderPath?.startsWith("inputs/")) return false;
  const parts = folderPath.split("/").filter(Boolean);
  return parts.length >= 2 && parts[0] === "inputs";
}

export function isMutableRun(runPath) {
  if (!runPath?.startsWith("runs/")) return false;
  const parts = runPath.split("/").filter(Boolean);
  return parts.length === 2 && parts[0] === "runs";
}

export function isExplorerContextTarget(entry) {
  if (!entry?.path) return false;
  if (entry.type === "file") return isMutableInputFile(entry.path);
  if (entry.type === "folder") return isMutableInputFolder(entry.path);
  if (entry.type === "run") return isMutableRun(entry.path);
  return false;
}

export function isExplorerDraggable(entry) {
  if (!entry?.path) return false;
  if (entry.type === "file") return isMutableInputFile(entry.path);
  if (entry.type === "folder") return isMutableInputFolder(entry.path);
  return false;
}

export function isInputFolderDropTarget(folderPath) {
  return folderPath === "inputs" || folderPath?.startsWith("inputs/");
}

export function isFolderDescendantPath(ancestorPath, descendantPath) {
  return descendantPath === ancestorPath || descendantPath.startsWith(`${ancestorPath}/`);
}

export function readDraggedExplorerPath(dataTransfer) {
  return dataTransfer.getData(EXPLORER_DRAG_TYPE) || "";
}

export function setDraggedExplorerPath(dataTransfer, itemPath) {
  dataTransfer.setData(EXPLORER_DRAG_TYPE, itemPath);
  dataTransfer.effectAllowed = "move";
}

export function setDraggedInputPath(dataTransfer, filePath) {
  setDraggedExplorerPath(dataTransfer, filePath);
}

export function readDraggedInputPath(dataTransfer) {
  return readDraggedExplorerPath(dataTransfer);
}
