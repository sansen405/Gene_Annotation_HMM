import cors from "cors";
import Database from "better-sqlite3";
import express from "express";
import { execFile } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import multer from "multer";

const execFileAsync = promisify(execFile);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..");
const localDataDir = path.join(__dirname, "local_data");
const uploadDir = path.join(localDataDir, "uploads");
const scoresDir = path.join(localDataDir, "splice_scores");
const binDir = path.join(localDataDir, "bin");
const predictorBin = path.join(binDir, "hmm_predict_fna");
const DEFAULT_MODEL_ID = "fission_yeasts";
const PRETRAINED_MODEL_CATALOG = [
  {
    id: "fission_yeasts",
    label: "Fission yeasts",
    tag: "Recommended",
    version: "v4.2",
    recommended: true,
    profilePath: "src/genome_profiles/fission_yeasts/fission_yeasts.json",
  },
  {
    id: "fungi_diverse",
    label: "Diverse fungi",
    tag: "Pan-fungal",
    version: "24 genomes",
    profilePath: "src/genome_profiles/fungi_diverse/fungi_diverse.json",
  },
];
const scoreSpliceScript = path.join(repoRoot, "src/model/cnn/score_fasta.py");
const scoreStartScript = path.join(repoRoot, "src/model/cnn/score_start_fasta.py");
const trainCachedModelScript = path.join(
  repoRoot,
  "src/model/training_pipeline/train_cached_model.py"
);
const projectsDir = path.join(localDataDir, "projects");
const DEFAULT_PROJECT_FOLDERS = ["inputs", "runs"];
const venvPython = path.join(repoRoot, ".venv", "bin", "python3");
const pythonBin = process.env.PYTHON || (fs.existsSync(venvPython) ? venvPython : "python3");

fs.mkdirSync(uploadDir, { recursive: true });
fs.mkdirSync(scoresDir, { recursive: true });
fs.mkdirSync(binDir, { recursive: true });
fs.mkdirSync(projectsDir, { recursive: true });

// ---------------------------------------------------------------------------
// SQLite run index
// Stores lightweight metadata for all runs so project-tree and list endpoints
// don't have to read individual run.json files on every request.
// ---------------------------------------------------------------------------
const db = new Database(path.join(localDataDir, "runs.db"));
db.exec(`
  CREATE TABLE IF NOT EXISTS runs (
    project_id      TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    name            TEXT NOT NULL,
    file_name       TEXT,
    model_id        TEXT,
    model_label     TEXT,
    date            TEXT,
    status          TEXT NOT NULL DEFAULT 'done',
    genes           INTEGER NOT NULL DEFAULT 0,
    total_bases     INTEGER NOT NULL DEFAULT 0,
    scaffolds_count INTEGER NOT NULL DEFAULT 0,
    elapsed_ms      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (project_id, run_id)
  )
`);

function upsertRun(projectId, meta) {
  db.prepare(`
    INSERT OR REPLACE INTO runs
      (project_id, run_id, name, file_name, model_id, model_label, date,
       status, genes, total_bases, scaffolds_count, elapsed_ms)
    VALUES
      (@project_id, @run_id, @name, @file_name, @model_id, @model_label, @date,
       @status, @genes, @total_bases, @scaffolds_count, @elapsed_ms)
  `).run({
    project_id: projectId,
    run_id: meta.id,
    name: meta.name,
    file_name: meta.fileName ?? null,
    model_id: meta.modelId ?? null,
    model_label: meta.modelLabel ?? null,
    date: meta.date ?? null,
    status: meta.status ?? "done",
    genes: meta.summary?.genes ?? 0,
    total_bases: meta.summary?.totalBases ?? 0,
    scaffolds_count: meta.summary?.scaffolds ?? 0,
    elapsed_ms: meta.elapsedMs ?? 0,
  });
}

function dbGetRun(projectId, runId) {
  return db
    .prepare("SELECT * FROM runs WHERE project_id = ? AND run_id = ?")
    .get(projectId, runId);
}

function dbDeleteRun(projectId, runId) {
  db.prepare("DELETE FROM runs WHERE project_id = ? AND run_id = ?").run(projectId, runId);
}

function dbRenameRun(projectId, oldRunId, newRunId, newName) {
  db.prepare(
    "UPDATE runs SET run_id = ?, name = ? WHERE project_id = ? AND run_id = ?"
  ).run(newRunId, newName, projectId, oldRunId);
}

function dbRunCountsByProject() {
  const rows = db
    .prepare("SELECT project_id, COUNT(*) AS count FROM runs GROUP BY project_id")
    .all();
  return new Map(rows.map((r) => [r.project_id, r.count]));
}

// On startup, backfill any existing runs that are not yet in the index.
const _migrateInsert = db.prepare(`
  INSERT OR IGNORE INTO runs
    (project_id, run_id, name, file_name, model_id, model_label, date,
     status, genes, total_bases, scaffolds_count, elapsed_ms)
  VALUES
    (@project_id, @run_id, @name, @file_name, @model_id, @model_label, @date,
     @status, @genes, @total_bases, @scaffolds_count, @elapsed_ms)
`);

db.transaction(() => {
  if (!fs.existsSync(projectsDir)) return;
  for (const projectEntry of fs.readdirSync(projectsDir, { withFileTypes: true })) {
    if (!projectEntry.isDirectory()) continue;
    const projectId = projectEntry.name;
    const runsPath = path.join(projectsDir, projectId, "runs");
    if (!fs.existsSync(runsPath)) continue;
    for (const runEntry of fs.readdirSync(runsPath, { withFileTypes: true })) {
      if (!runEntry.isDirectory()) continue;
      const runId = runEntry.name;
      const runJsonPath = path.join(runsPath, runId, "run.json");
      if (!fs.existsSync(runJsonPath)) continue;
      try {
        const run = JSON.parse(fs.readFileSync(runJsonPath, "utf8"));
        _migrateInsert.run({
          project_id: projectId,
          run_id: runId,
          name: run.name ?? runId,
          file_name: run.fileName ?? null,
          model_id: run.modelId ?? null,
          model_label: run.modelLabel ?? null,
          date: run.date ?? null,
          status: run.status ?? "done",
          genes: run.summary?.genes ?? 0,
          total_bases: run.summary?.totalBases ?? 0,
          scaffolds_count: run.summary?.scaffolds ?? 0,
          elapsed_ms: run.elapsedMs ?? 0,
        });
      } catch {
        // skip corrupt run files
      }
    }
  }
})();

// ---------------------------------------------------------------------------

const app = express();
const upload = multer({ dest: uploadDir });

app.use(cors());
app.use(express.json());

function execFilePromise(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    execFile(command, args, { maxBuffer: 1024 * 1024 * 256, ...options }, (error, stdout, stderr) => {
      if (error) {
        error.stderr = stderr;
        reject(error);
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

function jsonIncludeCandidates() {
  const candidates = [
    process.env.NLOHMANN_JSON_INCLUDE,
    "/opt/homebrew/include",
    "/usr/local/include",
    path.join(repoRoot, "third_party"),
  ].filter(Boolean);

  return [...new Set(candidates.filter((dir) => fs.existsSync(path.join(dir, "nlohmann", "json.hpp"))))];
}

async function resolveJsonIncludeFlags() {
  const fromDisk = jsonIncludeCandidates();
  if (fromDisk.length > 0) {
    return fromDisk.flatMap((dir) => ["-I", dir]);
  }

  try {
    const { stdout } = await execFileAsync("brew", ["--prefix", "nlohmann-json"]);
    const prefix = stdout.trim();
    const includeDir = path.join(prefix, "include");
    if (fs.existsSync(path.join(includeDir, "nlohmann", "json.hpp"))) {
      return ["-I", includeDir];
    }
  } catch {
    // brew not available or package not installed
  }

  throw new Error(
    "Missing nlohmann/json.hpp. Install it with: brew install nlohmann-json"
  );
}

async function ensurePredictorBuilt() {
  const sources = [
    "src/tools/predict_fna.cpp",
    "src/decoding/Viterbi.cpp",
    "src/decoding/Forward_Backward.cpp",
    "src/model/transition/Transition_Model.cpp",
    "src/genome_profiles/Genome_Profile.cpp",
    "src/parsers/FNA_Parser.cpp",
    "src/parsers/GFF_Parser.cpp",
    "src/model/emission/Emission_Model.cpp",
    "src/model/cnn/splice/Splice_CNN_Scores.cpp",
    "src/model/cnn/start/Start_CNN_Scores.cpp",
  ].map((source) => path.join(repoRoot, source));

  if (fs.existsSync(predictorBin)) {
    const binaryMtime = fs.statSync(predictorBin).mtimeMs;
    const newestSourceMtime = Math.max(...sources.map((source) => fs.statSync(source).mtimeMs));
    if (binaryMtime >= newestSourceMtime) return;
  }

  const jsonIncludes = await resolveJsonIncludeFlags();

  await execFilePromise(
    "clang++",
    [
      "-std=c++17",
      "-Isrc",
      ...jsonIncludes,
      "src/tools/predict_fna.cpp",
      "src/decoding/Viterbi.cpp",
      "src/decoding/Forward_Backward.cpp",
      "src/model/transition/Transition_Model.cpp",
      "src/genome_profiles/Genome_Profile.cpp",
      "src/parsers/FNA_Parser.cpp",
      "src/parsers/GFF_Parser.cpp",
      "src/model/emission/Emission_Model.cpp",
      "src/model/cnn/splice/Splice_CNN_Scores.cpp",
      "src/model/cnn/start/Start_CNN_Scores.cpp",
      "-o",
      predictorBin,
    ],
    { cwd: repoRoot }
  );
}

function resolveProfilePath(modelId = DEFAULT_MODEL_ID) {
  const entry = PRETRAINED_MODEL_CATALOG.find((model) => model.id === modelId);
  if (!entry) {
    throw new Error(`Unknown pretrained model "${modelId}".`);
  }
  return path.join(repoRoot, entry.profilePath);
}

function resolveModelEntry(modelId = DEFAULT_MODEL_ID) {
  const entry = PRETRAINED_MODEL_CATALOG.find((model) => model.id === modelId);
  if (!entry) {
    throw new Error(`Unknown pretrained model "${modelId}".`);
  }
  return entry;
}

function readProfileAt(profilePath) {
  return JSON.parse(fs.readFileSync(profilePath, "utf8"));
}

function modelCheckpointStatus(profile) {
  const splicePath = path.join(repoRoot, profile.splice_cnn.model);
  const startPath = path.join(repoRoot, profile.start_cnn.model);
  const missing = [];
  if (!fs.existsSync(splicePath)) missing.push("splice CNN");
  if (!fs.existsSync(startPath)) missing.push("start CNN");
  return {
    available: missing.length === 0,
    missing,
    splicePath,
    startPath,
  };
}

function listPretrainedModels() {
  return PRETRAINED_MODEL_CATALOG.map((entry) => {
    const profilePath = path.join(repoRoot, entry.profilePath);
    const profile = readProfileAt(profilePath);
    const checkpoints = modelCheckpointStatus(profile);
    return {
      id: entry.id,
      label: entry.label,
      tag: entry.tag,
      version: entry.version,
      description: profile.description ?? "",
      available: checkpoints.available,
      missing: checkpoints.missing,
      recommended: Boolean(entry.recommended),
    };
  });
}

function spliceModelPath(profilePath) {
  return path.join(repoRoot, readProfileAt(profilePath).splice_cnn.model);
}

function startModelPath(profilePath) {
  return path.join(repoRoot, readProfileAt(profilePath).start_cnn.model);
}

async function ensureCnnModels(profilePath, modelEntry) {
  const profile = readProfileAt(profilePath);
  const { available, missing, splicePath, startPath } = modelCheckpointStatus(profile);
  if (available) {
    return;
  }

  if (modelEntry.id === DEFAULT_MODEL_ID) {
    console.log(
      `CNN checkpoints missing for ${modelEntry.label}; refreshing cached models (first run only)...`
    );
    await execFilePromise(
      pythonBin,
      [trainCachedModelScript, "--profile", profilePath, "--skip-compile"],
      { cwd: repoRoot, maxBuffer: 1024 * 1024 * 64 }
    );
    if (fs.existsSync(splicePath) && fs.existsSync(startPath)) {
      return;
    }
  }

  throw new Error(
    `Missing ${missing.join(" and ")} for ${modelEntry.label}. Install the pretrained checkpoints before running this model.`
  );
}

async function ensureSpliceScores(fastaPath, scoresPath, profilePath, modelEntry) {
  await ensureCnnModels(profilePath, modelEntry);
  await execFilePromise(
    pythonBin,
    [
      scoreSpliceScript,
      "--fasta",
      fastaPath,
      "--model",
      spliceModelPath(profilePath),
      "--scores-out",
      scoresPath,
    ],
    { cwd: repoRoot, maxBuffer: 1024 * 1024 * 64 }
  );
}

async function ensureStartScores(fastaPath, scoresPath, profilePath, modelEntry) {
  await ensureCnnModels(profilePath, modelEntry);
  await execFilePromise(
    pythonBin,
    [
      scoreStartScript,
      "--fasta",
      fastaPath,
      "--model",
      startModelPath(profilePath),
      "--scores-out",
      scoresPath,
    ],
    { cwd: repoRoot, maxBuffer: 1024 * 1024 * 64 }
  );
}

function makeRunName(fileName) {
  const stem = path.basename(fileName).replace(/\.(fna|fa|fasta)$/i, "");
  const stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15);
  return `${stem}_${stamp}`;
}

// Returns the lightweight metadata object saved to run.json.
function makeRunMeta(fileName, inputPath, predictionResult, elapsedMs, runId, modelEntry) {
  return {
    id: runId,
    name: runId,
    fileName,
    inputPath,
    modelId: modelEntry.id,
    modelLabel: modelEntry.label,
    date: new Date().toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    }),
    status: "done",
    elapsedMs,
    summary: predictionResult.summary,
    scaffolds: predictionResult.scaffolds,
  };
}

// Returns the large data object saved to run_data.json.
function makeRunData(predictionResult) {
  return {
    predictions: predictionResult.predictions,
    confidenceByScaffold: predictionResult.confidenceByScaffold,
  };
}

function slugifyProjectName(name) {
  const slug = String(name)
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
  return slug || "project";
}

function projectRoot(projectId) {
  return path.join(projectsDir, projectId);
}

function projectMetaPath(projectId) {
  return path.join(projectRoot(projectId), "project.json");
}

function readProjectMeta(projectId) {
  return JSON.parse(fs.readFileSync(projectMetaPath(projectId), "utf8"));
}

function writeProjectMeta(projectId, meta) {
  fs.writeFileSync(projectMetaPath(projectId), JSON.stringify(meta, null, 2));
}

function touchProject(projectId) {
  const meta = readProjectMeta(projectId);
  meta.updatedAt = new Date().toISOString();
  writeProjectMeta(projectId, meta);
  return meta;
}

function resolveProjectPath(projectId, relPath = "") {
  const root = path.resolve(projectRoot(projectId));
  const resolved = path.resolve(root, relPath);
  if (resolved !== root && !resolved.startsWith(`${root}${path.sep}`)) {
    throw new Error("Invalid project path.");
  }
  return resolved;
}

function normalizeInputFilePath(relPath) {
  const normalized = String(relPath).trim().replace(/\\/g, "/");
  if (!normalized.startsWith("inputs/")) {
    throw new Error("Only files under Inputs can be modified.");
  }
  const parts = normalized.split("/").filter(Boolean);
  if (parts.length < 2 || parts[0] !== "inputs") {
    throw new Error("Invalid file path.");
  }
  const fileName = parts[parts.length - 1];
  if (!fileName || fileName === "." || fileName === "..") {
    throw new Error("Invalid file path.");
  }
  return normalized;
}

function sanitizeInputFileName(name) {
  const trimmed = String(name).trim();
  if (!trimmed) {
    throw new Error("File name is required.");
  }
  if (trimmed.includes("/") || trimmed.includes("\\") || trimmed === "." || trimmed === "..") {
    throw new Error("File name cannot contain path separators.");
  }
  return trimmed;
}

function normalizeInputFolderPath(relPath) {
  const normalized = String(relPath).trim().replace(/\\/g, "/");
  if (normalized !== "inputs" && !normalized.startsWith("inputs/")) {
    throw new Error("Destination must be under Inputs.");
  }
  return normalized;
}

function normalizeMutableInputFolderPath(relPath) {
  const normalized = normalizeInputFolderPath(relPath);
  if (normalized === "inputs") {
    throw new Error("The Inputs folder cannot be modified.");
  }
  const parts = normalized.split("/").filter(Boolean);
  if (parts.length < 2 || parts[0] !== "inputs") {
    throw new Error("Invalid folder path.");
  }
  return normalized;
}

function normalizeRunPath(relPath) {
  const normalized = String(relPath).trim().replace(/\\/g, "/");
  if (!normalized.startsWith("runs/")) {
    throw new Error("Invalid run path.");
  }
  const parts = normalized.split("/").filter(Boolean);
  if (parts.length !== 2 || parts[0] !== "runs") {
    throw new Error("Invalid run path.");
  }
  return normalized;
}

function sanitizeFolderName(name) {
  return sanitizeInputFileName(name);
}

function isFolderDescendantPath(ancestorPath, descendantPath) {
  return descendantPath === ancestorPath || descendantPath.startsWith(`${ancestorPath}/`);
}

function deleteDirectoryRecursive(dirPath) {
  fs.rmSync(dirPath, { recursive: true, force: true });
}

function listProjects() {
  const runCounts = dbRunCountsByProject();
  return fs
    .readdirSync(projectsDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      try {
        const meta = readProjectMeta(entry.name);
        return { ...meta, runCount: runCounts.get(entry.name) ?? 0 };
      } catch {
        return null;
      }
    })
    .filter(Boolean)
    .sort((left, right) => new Date(right.updatedAt) - new Date(left.updatedAt));
}

function createProject(name, description = "") {
  const base = slugifyProjectName(name);
  let projectId = base;
  let suffix = 1;
  while (fs.existsSync(projectRoot(projectId))) {
    projectId = `${base}-${suffix}`;
    suffix += 1;
  }

  fs.mkdirSync(projectRoot(projectId), { recursive: true });
  for (const folder of DEFAULT_PROJECT_FOLDERS) {
    fs.mkdirSync(path.join(projectRoot(projectId), folder), { recursive: true });
  }

  const now = new Date().toISOString();
  const meta = {
    id: projectId,
    name: String(name).trim(),
    description: String(description).trim(),
    createdAt: now,
    updatedAt: now,
  };
  writeProjectMeta(projectId, meta);
  return meta;
}

// Resolve run metadata for a directory entry using the SQLite index,
// falling back to reading run.json for runs not yet in the index.
function resolveRunEntry(projectId, runId, entryPath, relPath) {
  const row = dbGetRun(projectId, runId);
  if (row) {
    return {
      name: runId,
      path: relPath,
      type: "run",
      status: row.status,
      label: row.name,
      fileName: row.file_name,
      date: row.date,
      genes: row.genes,
      totalBases: row.total_bases,
      elapsedMs: row.elapsed_ms,
    };
  }
  // Fallback: read file (handles runs created before the index existed)
  const runJsonPath = path.join(entryPath, "run.json");
  const run = JSON.parse(fs.readFileSync(runJsonPath, "utf8"));
  return {
    name: runId,
    path: relPath,
    type: "run",
    status: run.status,
    label: run.name,
    fileName: run.fileName,
    date: run.date,
    genes: run.summary?.genes ?? 0,
    totalBases: run.summary?.totalBases ?? 0,
    elapsedMs: run.elapsedMs ?? 0,
  };
}

function listDirectory(projectId, relPath = "") {
  const dirPath = resolveProjectPath(projectId, relPath);
  if (!fs.existsSync(dirPath) || !fs.statSync(dirPath).isDirectory()) {
    throw new Error("Directory not found.");
  }

  const entries = fs
    .readdirSync(dirPath, { withFileTypes: true })
    .map((entry) => {
      const entryPath = path.join(dirPath, entry.name);
      const childRelPath = relPath ? `${relPath}/${entry.name}` : entry.name;

      if (entry.isDirectory()) {
        const runJsonPath = path.join(entryPath, "run.json");
        if (fs.existsSync(runJsonPath)) {
          return resolveRunEntry(projectId, entry.name, entryPath, childRelPath);
        }
        return {
          name: entry.name,
          path: childRelPath,
          type: "folder",
        };
      }
      return {
        name: entry.name,
        path: childRelPath,
        type: "file",
        size: fs.statSync(entryPath).size,
      };
    })
    .sort((left, right) => {
      if (left.type !== right.type) {
        if (left.type === "folder" || left.type === "run") return -1;
        if (right.type === "folder" || right.type === "run") return 1;
      }
      return left.name.localeCompare(right.name);
    });

  return {
    path: relPath.replace(/\\/g, "/") || ".",
    entries,
  };
}

function buildProjectTree(projectId, relPath = "") {
  const dirPath = resolveProjectPath(projectId, relPath);
  if (!fs.existsSync(dirPath) || !fs.statSync(dirPath).isDirectory()) {
    return [];
  }

  const includeFiles = relPath === "inputs" || relPath.startsWith("inputs/");

  return fs
    .readdirSync(dirPath, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() || (includeFiles && entry.isFile()))
    .map((entry) => {
      const childRel = relPath ? `${relPath}/${entry.name}` : entry.name;
      const entryPath = path.join(dirPath, entry.name);

      if (entry.isFile()) {
        return {
          name: entry.name,
          path: childRel,
          type: "file",
          size: fs.statSync(entryPath).size,
        };
      }

      const runJsonPath = path.join(entryPath, "run.json");

      if (fs.existsSync(runJsonPath)) {
        return resolveRunEntry(projectId, entry.name, entryPath, childRel);
      }

      return {
        name: entry.name,
        path: childRel,
        type: "folder",
        children: buildProjectTree(projectId, childRel),
      };
    })
    .sort((left, right) => {
      const leftIsFolder = left.type === "folder" || left.type === "run";
      const rightIsFolder = right.type === "folder" || right.type === "run";
      if (leftIsFolder !== rightIsFolder) {
        return leftIsFolder ? -1 : 1;
      }
      return left.name.localeCompare(right.name);
    });
}

// Read a run, merging metadata (run.json) with prediction data (run_data.json).
// Supports both the new split format and the legacy single-file format.
function readRun(projectId, runId) {
  const runDir = resolveProjectPath(projectId, path.join("runs", runId));
  const runJsonPath = path.join(runDir, "run.json");
  if (!fs.existsSync(runJsonPath)) {
    throw new Error("Run not found.");
  }
  const meta = JSON.parse(fs.readFileSync(runJsonPath, "utf8"));
  const runDataPath = path.join(runDir, "run_data.json");
  if (fs.existsSync(runDataPath)) {
    const data = JSON.parse(fs.readFileSync(runDataPath, "utf8"));
    return { ...meta, ...data };
  }
  // Legacy format: all data was stored in run.json
  return meta;
}

async function executeProjectRun(projectId, inputRelPath, modelId = DEFAULT_MODEL_ID) {
  const inputPath = resolveProjectPath(projectId, inputRelPath);
  if (!fs.existsSync(inputPath) || !fs.statSync(inputPath).isFile()) {
    throw new Error("Input FASTA file not found.");
  }

  const modelEntry = resolveModelEntry(modelId);
  const profilePath = resolveProfilePath(modelId);
  const checkpoints = modelCheckpointStatus(readProfileAt(profilePath));
  if (!checkpoints.available) {
    throw new Error(
      `Model "${modelEntry.label}" is not fully installed (${checkpoints.missing.join(", ")} missing).`
    );
  }

  const runId = makeRunName(path.basename(inputRelPath));
  const runDir = resolveProjectPath(projectId, path.join("runs", runId));
  fs.mkdirSync(runDir, { recursive: true });

  const spliceScoresPath = path.join(runDir, "splice_scores.tsv");
  const startScoresPath = path.join(runDir, "start_scores.tsv");
  const statusPath = path.join(runDir, "status.json");
  const started = Date.now();

  fs.writeFileSync(
    statusPath,
    JSON.stringify({ status: "running", startedAt: started }, null, 2)
  );

  try {
    await ensurePredictorBuilt();
    await ensureSpliceScores(inputPath, spliceScoresPath, profilePath, modelEntry);
    await ensureStartScores(inputPath, startScoresPath, profilePath, modelEntry);
    const { stdout } = await execFilePromise(
      predictorBin,
      [
        "--fna",
        inputPath,
        "--profile",
        profilePath,
        "--splice-cnn-scores",
        spliceScoresPath,
        "--start-cnn-scores",
        startScoresPath,
      ],
      { cwd: repoRoot }
    );
    const predictionResult = JSON.parse(stdout);
    const meta = makeRunMeta(
      path.basename(inputRelPath),
      inputRelPath.replace(/\\/g, "/"),
      predictionResult,
      Date.now() - started,
      runId,
      modelEntry
    );
    const data = makeRunData(predictionResult);
    fs.writeFileSync(path.join(runDir, "run.json"), JSON.stringify(meta, null, 2));
    fs.writeFileSync(path.join(runDir, "run_data.json"), JSON.stringify(data, null, 2));
    upsertRun(projectId, meta);
    fs.writeFileSync(statusPath, JSON.stringify({ status: "done" }, null, 2));
    touchProject(projectId);
    return { ...meta, ...data };
  } catch (error) {
    fs.writeFileSync(
      statusPath,
      JSON.stringify(
        {
          status: "failed",
          detail: error.stderr || error.message,
        },
        null,
        2
      )
    );
    throw error;
  }
}

function escapeAttribute(value) {
  return String(value).replace(/[;\t\n\r]/g, "_");
}

function runToGff3(run) {
  const lines = ["##gff-version 3"];
  for (const prediction of run.predictions) {
    const strand = prediction.strand === "-" ? "-" : "+";
    lines.push(
      [
        prediction.scaffold,
        "HMMGenePredictor",
        "gene",
        prediction.start,
        prediction.end,
        ".",
        strand,
        ".",
        `ID=${escapeAttribute(prediction.id)}`,
      ].join("\t")
    );
    prediction.exons.forEach((exon, index) => {
      lines.push(
        [
          prediction.scaffold,
          "HMMGenePredictor",
          "CDS",
          exon.start,
          exon.end,
          ".",
          strand,
          ".",
          `ID=${escapeAttribute(prediction.id)}.cds${index + 1};Parent=${escapeAttribute(prediction.id)}`,
        ].join("\t")
      );
    });
  }
  return `${lines.join("\n")}\n`;
}

function runToCsv(run) {
  const rows = ["id,scaffold,strand,start,end,length,exon_count,intron_count"];
  for (const prediction of run.predictions) {
    rows.push(
      [
        prediction.id,
        prediction.scaffold,
        prediction.strand ?? "+",
        prediction.start,
        prediction.end,
        prediction.end - prediction.start + 1,
        prediction.exons.length,
        prediction.introns.length,
      ].join(",")
    );
  }
  return `${rows.join("\n")}\n`;
}

function runToBed(run) {
  const lines = [];
  for (const prediction of run.predictions) {
    lines.push(
      [
        prediction.scaffold,
        prediction.start - 1,
        prediction.end,
        prediction.id,
        0,
        prediction.strand ?? "+",
      ].join("\t")
    );
  }
  return `${lines.join("\n")}\n`;
}

function runToFasta(run) {
  const lines = [];
  for (const prediction of run.predictions) {
    lines.push(`>${prediction.id} ${prediction.scaffold}:${prediction.start}-${prediction.end}`);
    lines.push(prediction.sequence || "");
  }
  return `${lines.join("\n")}\n`;
}

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, mode: "local" });
});

app.get("/api/models", (_req, res) => {
  res.json(listPretrainedModels());
});

app.get("/api/projects", (_req, res) => {
  res.json(listProjects());
});

app.post("/api/projects", (req, res) => {
  const name = String(req.body?.name ?? "").trim();
  if (!name) {
    res.status(400).json({ error: "Project name is required." });
    return;
  }
  const project = createProject(name, req.body?.description ?? "");
  res.status(201).json(project);
});

app.get("/api/projects/:projectId", (req, res) => {
  try {
    const meta = readProjectMeta(req.params.projectId);
    res.json({ ...meta, tree: buildProjectTree(req.params.projectId) });
  } catch {
    res.status(404).json({ error: "Project not found." });
  }
});

app.get("/api/projects/:projectId/list", (req, res) => {
  try {
    const relPath = String(req.query.path ?? "");
    res.json(listDirectory(req.params.projectId, relPath));
  } catch (error) {
    res.status(404).json({ error: error.message });
  }
});

app.get("/api/projects/:projectId/files/download", (req, res) => {
  try {
    const relPath = String(req.query.path ?? "").trim();
    if (!relPath) {
      res.status(400).json({ error: "File path is required." });
      return;
    }
    const filePath = resolveProjectPath(req.params.projectId, relPath);
    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      res.status(404).json({ error: "File not found." });
      return;
    }
    res.download(filePath, path.basename(filePath));
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.get("/api/projects/:projectId/files/preview", (req, res) => {
  try {
    const relPath = String(req.query.path ?? "").trim();
    const maxLines = Math.min(Math.max(Number(req.query.lines) || 48, 1), 800);
    if (!relPath) {
      res.status(400).json({ error: "File path is required." });
      return;
    }
    const filePath = resolveProjectPath(req.params.projectId, relPath);
    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      res.status(404).json({ error: "File not found." });
      return;
    }
    const content = fs.readFileSync(filePath, "utf8");
    const lines = content.split(/\r?\n/);
    const previewLines = lines.slice(0, maxLines);
    res.json({
      lines: previewLines,
      lineCount: lines.length,
      truncated: lines.length > maxLines,
    });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.patch("/api/projects/:projectId/files", (req, res) => {
  try {
    const relPath = normalizeInputFilePath(req.body?.path ?? "");
    const projectId = req.params.projectId;
    const filePath = resolveProjectPath(projectId, relPath);
    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      res.status(404).json({ error: "File not found." });
      return;
    }

    const currentName = path.posix.basename(relPath);
    const newName =
      req.body?.name != null && String(req.body.name).trim()
        ? sanitizeInputFileName(req.body.name)
        : currentName;
    const destinationFolder =
      req.body?.folder != null
        ? normalizeInputFolderPath(req.body.folder)
        : path.posix.dirname(relPath);
    const destinationDir = resolveProjectPath(projectId, destinationFolder);
    if (!fs.existsSync(destinationDir) || !fs.statSync(destinationDir).isDirectory()) {
      res.status(404).json({ error: "Destination folder not found." });
      return;
    }

    const nextRelPath = `${destinationFolder}/${newName}`;
    if (nextRelPath === relPath) {
      res.json({
        folder: destinationFolder,
        name: currentName,
        path: relPath,
        size: fs.statSync(filePath).size,
      });
      return;
    }

    const nextPath = resolveProjectPath(projectId, nextRelPath);
    if (fs.existsSync(nextPath)) {
      res.status(409).json({ error: "A file with that name already exists in the destination folder." });
      return;
    }

    fs.renameSync(filePath, nextPath);
    touchProject(projectId);
    res.json({
      folder: destinationFolder,
      name: newName,
      path: nextRelPath,
      size: fs.statSync(nextPath).size,
    });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.delete("/api/projects/:projectId/files", (req, res) => {
  try {
    const relPath = normalizeInputFilePath(req.body?.path ?? req.query?.path ?? "");
    const projectId = req.params.projectId;
    const filePath = resolveProjectPath(projectId, relPath);
    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      res.status(404).json({ error: "File not found." });
      return;
    }

    fs.unlinkSync(filePath);
    touchProject(projectId);
    res.json({ ok: true, path: relPath });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.post("/api/projects/:projectId/folders", (req, res) => {
  try {
    const name = String(req.body?.name ?? "").trim();
    const parentPath = String(req.body?.parentPath ?? "").trim();
    if (!name) {
      res.status(400).json({ error: "Folder name is required." });
      return;
    }
    const folderPath = resolveProjectPath(
      req.params.projectId,
      parentPath ? path.join(parentPath, name) : name
    );
    fs.mkdirSync(folderPath, { recursive: true });
    touchProject(req.params.projectId);
    res.status(201).json({ path: parentPath ? `${parentPath}/${name}` : name });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.patch("/api/projects/:projectId/folders", (req, res) => {
  try {
    const relPath = normalizeMutableInputFolderPath(req.body?.path ?? "");
    const projectId = req.params.projectId;
    const folderPath = resolveProjectPath(projectId, relPath);
    if (!fs.existsSync(folderPath) || !fs.statSync(folderPath).isDirectory()) {
      res.status(404).json({ error: "Folder not found." });
      return;
    }
    if (fs.existsSync(path.join(folderPath, "run.json"))) {
      res.status(400).json({ error: "Cannot modify an analysis run as a folder." });
      return;
    }

    const currentName = path.posix.basename(relPath);
    const newName =
      req.body?.name != null && String(req.body.name).trim()
        ? sanitizeFolderName(req.body.name)
        : currentName;
    const destinationParent =
      req.body?.folder != null
        ? normalizeInputFolderPath(req.body.folder)
        : path.posix.dirname(relPath);
    const nextRelPath = `${destinationParent}/${newName}`;

    if (nextRelPath === relPath) {
      res.json({ name: currentName, path: relPath });
      return;
    }

    if (isFolderDescendantPath(relPath, destinationParent)) {
      res.status(400).json({ error: "Cannot move a folder into itself." });
      return;
    }

    const destinationDir = resolveProjectPath(projectId, destinationParent);
    if (!fs.existsSync(destinationDir) || !fs.statSync(destinationDir).isDirectory()) {
      res.status(404).json({ error: "Destination folder not found." });
      return;
    }

    const nextPath = resolveProjectPath(projectId, nextRelPath);
    if (fs.existsSync(nextPath)) {
      res.status(409).json({ error: "A folder with that name already exists in the destination." });
      return;
    }

    fs.renameSync(folderPath, nextPath);
    touchProject(projectId);
    res.json({ name: newName, path: nextRelPath });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.delete("/api/projects/:projectId/folders", (req, res) => {
  try {
    const relPath = normalizeMutableInputFolderPath(req.body?.path ?? req.query?.path ?? "");
    const projectId = req.params.projectId;
    const folderPath = resolveProjectPath(projectId, relPath);
    if (!fs.existsSync(folderPath) || !fs.statSync(folderPath).isDirectory()) {
      res.status(404).json({ error: "Folder not found." });
      return;
    }

    deleteDirectoryRecursive(folderPath);
    touchProject(projectId);
    res.json({ ok: true, path: relPath });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.post("/api/projects/:projectId/upload", upload.single("file"), (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: "FASTA file is required." });
    return;
  }

  try {
    const folderPath = String(req.query.path ?? "inputs");
    const destPath = resolveProjectPath(
      req.params.projectId,
      path.join(folderPath, req.file.originalname)
    );
    fs.mkdirSync(path.dirname(destPath), { recursive: true });
    fs.renameSync(req.file.path, destPath);
    touchProject(req.params.projectId);
    res.status(201).json({
      name: req.file.originalname,
      path: path.join(folderPath, req.file.originalname).replace(/\\/g, "/"),
      size: req.file.size,
    });
  } catch (error) {
    fs.unlink(req.file.path, () => {});
    res.status(400).json({ error: error.message });
  }
});

app.post("/api/projects/:projectId/runs", async (req, res) => {
  const inputPath = String(req.body?.inputPath ?? "").trim();
  const modelId = String(req.body?.modelId ?? DEFAULT_MODEL_ID).trim();
  if (!inputPath) {
    res.status(400).json({ error: "Input path is required." });
    return;
  }

  try {
    const run = await executeProjectRun(req.params.projectId, inputPath, modelId);
    res.json(run);
  } catch (error) {
    res.status(500).json({
      error: "Prediction failed.",
      detail: error.stderr || error.message,
    });
  }
});

app.get("/api/projects/:projectId/runs/:runId", (req, res) => {
  try {
    res.json(readRun(req.params.projectId, req.params.runId));
  } catch {
    res.status(404).json({ error: "Run not found." });
  }
});

app.patch("/api/projects/:projectId/runs", (req, res) => {
  try {
    const relPath = normalizeRunPath(req.body?.path ?? "");
    const projectId = req.params.projectId;
    const oldRunId = relPath.split("/")[1];
    const runDir = resolveProjectPath(projectId, relPath);
    const runJsonPath = path.join(runDir, "run.json");
    if (!fs.existsSync(runDir) || !fs.existsSync(runJsonPath)) {
      res.status(404).json({ error: "Run not found." });
      return;
    }

    const newName = sanitizeFolderName(req.body?.name ?? "");
    const nextRelPath = `runs/${newName}`;
    const nextRunDir = resolveProjectPath(projectId, nextRelPath);

    if (nextRelPath !== relPath && fs.existsSync(nextRunDir)) {
      res.status(409).json({ error: "An analysis with that name already exists." });
      return;
    }

    if (nextRelPath !== relPath) {
      fs.renameSync(runDir, nextRunDir);
    }

    const run = JSON.parse(fs.readFileSync(path.join(nextRunDir, "run.json"), "utf8"));
    run.id = newName;
    run.name = newName;
    fs.writeFileSync(path.join(nextRunDir, "run.json"), JSON.stringify(run, null, 2));
    dbRenameRun(projectId, oldRunId, newName, newName);
    touchProject(projectId);
    res.json({ label: run.name, name: newName, path: nextRelPath });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.delete("/api/projects/:projectId/runs", (req, res) => {
  try {
    const relPath = normalizeRunPath(req.body?.path ?? req.query?.path ?? "");
    const projectId = req.params.projectId;
    const runId = relPath.split("/")[1];
    const runDir = resolveProjectPath(projectId, relPath);
    if (!fs.existsSync(runDir) || !fs.statSync(runDir).isDirectory()) {
      res.status(404).json({ error: "Run not found." });
      return;
    }

    deleteDirectoryRecursive(runDir);
    dbDeleteRun(projectId, runId);
    touchProject(projectId);
    res.json({ ok: true, path: relPath });
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.get("/api/projects/:projectId/runs/:runId/export/:format", (req, res) => {
  try {
    const run = readRun(req.params.projectId, req.params.runId);
    const format = req.params.format;
    const exporters = {
      bed: ["text/plain", runToBed],
      csv: ["text/csv", runToCsv],
      fasta: ["text/plain", runToFasta],
      gff3: ["text/plain", runToGff3],
    };
    const exporter = exporters[format];
    if (!exporter) {
      res.status(400).json({ error: "Unsupported export format." });
      return;
    }

    const [contentType, render] = exporter;
    res.setHeader("Content-Type", contentType);
    res.setHeader("Content-Disposition", `attachment; filename="${run.name}.${format}"`);
    res.send(render(run));
  } catch {
    res.status(404).json({ error: "Run not found." });
  }
});

app.get("/", (_req, res) => {
  res.redirect("http://localhost:5173/");
});

const port = Number(process.env.PORT || 5174);
const httpServer = app.listen(port, () => {
  console.log(`Local HMM API listening on http://localhost:${port}`);
});

httpServer.on("error", (error) => {
  if (error.code === "EADDRINUSE") {
    console.error(`Port ${port} is already in use. Stop the other process or set PORT.`);
  } else {
    console.error(error);
  }
  process.exit(1);
});
