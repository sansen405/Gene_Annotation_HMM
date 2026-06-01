import cors from "cors";
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
const profilePath = path.join(repoRoot, "src/genome_profiles/fission_yeasts.json");
const scoreFastaScript = path.join(repoRoot, "src/model/cnn/score_fasta.py");
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
    "src/model/cnn/Splice_CNN_Scores.cpp",
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
      "src/model/cnn/Splice_CNN_Scores.cpp",
      "-o",
      predictorBin,
    ],
    { cwd: repoRoot }
  );
}

function cnnModelPath() {
  const profile = JSON.parse(fs.readFileSync(profilePath, "utf8"));
  return path.join(repoRoot, profile.splice_cnn.model);
}

async function ensureCnnModel() {
  const modelPath = cnnModelPath();
  if (fs.existsSync(modelPath)) {
    return modelPath;
  }

  console.log("CNN checkpoint missing; training cached fission-yeast splice model (first run only)...");
  await execFilePromise(
    pythonBin,
    [trainCachedModelScript, "--profile", profilePath, "--skip-compile"],
    { cwd: repoRoot, maxBuffer: 1024 * 1024 * 64 }
  );
  if (!fs.existsSync(modelPath)) {
    throw new Error(`CNN checkpoint was not created at ${modelPath}`);
  }
  return modelPath;
}

async function ensureSpliceScores(fastaPath, scoresPath) {
  await ensureCnnModel();
  await execFilePromise(
    pythonBin,
    [
      scoreFastaScript,
      "--fasta",
      fastaPath,
      "--model",
      cnnModelPath(),
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

function makeRunRecord(fileName, inputPath, predictionResult, elapsedMs, runId) {
  return {
    id: runId,
    name: runId,
    fileName,
    inputPath,
    date: new Date().toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    }),
    status: "done",
    elapsedMs,
    summary: predictionResult.summary,
    scaffolds: predictionResult.scaffolds,
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

function listProjects() {
  return fs
    .readdirSync(projectsDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      try {
        const meta = readProjectMeta(entry.name);
        const runsPath = path.join(projectRoot(entry.name), "runs");
        const runCount = fs.existsSync(runsPath)
          ? fs
              .readdirSync(runsPath, { withFileTypes: true })
              .filter((runEntry) => runEntry.isDirectory()).length
          : 0;
        return { ...meta, runCount };
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

function listDirectory(projectId, relPath = "") {
  const dirPath = resolveProjectPath(projectId, relPath);
  if (!fs.existsSync(dirPath) || !fs.statSync(dirPath).isDirectory()) {
    throw new Error("Directory not found.");
  }

  const entries = fs
    .readdirSync(dirPath, { withFileTypes: true })
    .map((entry) => {
      const entryPath = path.join(dirPath, entry.name);
      if (entry.isDirectory()) {
        const runJsonPath = path.join(entryPath, "run.json");
        if (fs.existsSync(runJsonPath)) {
          const run = JSON.parse(fs.readFileSync(runJsonPath, "utf8"));
          return {
            name: entry.name,
            path: relPath ? `${relPath}/${entry.name}` : entry.name,
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
        return {
          name: entry.name,
          path: relPath ? `${relPath}/${entry.name}` : entry.name,
          type: "folder",
        };
      }
      return {
        name: entry.name,
        path: relPath ? `${relPath}/${entry.name}` : entry.name,
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

  return fs
    .readdirSync(dirPath, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      const childRel = relPath ? `${relPath}/${entry.name}` : entry.name;
      const entryPath = path.join(dirPath, entry.name);
      const runJsonPath = path.join(entryPath, "run.json");

      if (fs.existsSync(runJsonPath)) {
        const run = JSON.parse(fs.readFileSync(runJsonPath, "utf8"));
        return {
          name: entry.name,
          path: childRel,
          type: "run",
          status: run.status,
          label: run.name,
          fileName: run.fileName,
          date: run.date,
          genes: run.summary?.genes ?? 0,
          totalBases: run.summary?.totalBases ?? 0,
        };
      }

      return {
        name: entry.name,
        path: childRel,
        type: "folder",
        children: buildProjectTree(projectId, childRel),
      };
    });
}

function readRun(projectId, runId) {
  const runJsonPath = resolveProjectPath(projectId, path.join("runs", runId, "run.json"));
  if (!fs.existsSync(runJsonPath)) {
    throw new Error("Run not found.");
  }
  return JSON.parse(fs.readFileSync(runJsonPath, "utf8"));
}

async function executeProjectRun(projectId, inputRelPath) {
  const inputPath = resolveProjectPath(projectId, inputRelPath);
  if (!fs.existsSync(inputPath) || !fs.statSync(inputPath).isFile()) {
    throw new Error("Input FASTA file not found.");
  }

  const runId = makeRunName(path.basename(inputRelPath));
  const runDir = resolveProjectPath(projectId, path.join("runs", runId));
  fs.mkdirSync(runDir, { recursive: true });

  const scoresPath = path.join(runDir, "splice_scores.tsv");
  const statusPath = path.join(runDir, "status.json");
  const started = Date.now();

  fs.writeFileSync(
    statusPath,
    JSON.stringify({ status: "running", startedAt: started }, null, 2)
  );

  try {
    await ensurePredictorBuilt();
    await ensureSpliceScores(inputPath, scoresPath);
    const { stdout } = await execFilePromise(
      predictorBin,
      ["--fna", inputPath, "--profile", profilePath, "--splice-cnn-scores", scoresPath],
      { cwd: repoRoot }
    );
    const predictionResult = JSON.parse(stdout);
    const run = makeRunRecord(
      path.basename(inputRelPath),
      inputRelPath.replace(/\\/g, "/"),
      predictionResult,
      Date.now() - started,
      runId
    );
    fs.writeFileSync(path.join(runDir, "run.json"), JSON.stringify(run, null, 2));
    fs.writeFileSync(statusPath, JSON.stringify({ status: "done" }, null, 2));
    touchProject(projectId);
    return run;
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
    lines.push(
      [
        prediction.scaffold,
        "HMMGenePredictor",
        "gene",
        prediction.start,
        prediction.end,
        ".",
        ".",
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
          ".",
          ".",
          `ID=${escapeAttribute(prediction.id)}.cds${index + 1};Parent=${escapeAttribute(prediction.id)}`,
        ].join("\t")
      );
    });
  }
  return `${lines.join("\n")}\n`;
}

function runToCsv(run) {
  const rows = ["id,scaffold,start,end,length,exon_count,intron_count"];
  for (const prediction of run.predictions) {
    rows.push(
      [
        prediction.id,
        prediction.scaffold,
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
  if (!inputPath) {
    res.status(400).json({ error: "Input path is required." });
    return;
  }

  try {
    const run = await executeProjectRun(req.params.projectId, inputPath);
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
