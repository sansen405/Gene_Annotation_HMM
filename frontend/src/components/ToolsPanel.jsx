import { ArrowRight, Dna, Loader2, Play } from "lucide-react";
import { useState } from "react";
import { TrackMotif } from "./TrackMotif.jsx";

const TOOLS = [
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

export function ToolsPanel({
  fastaFiles,
  onNavigate,
  onRunPrediction,
  runStatus,
}) {
  const [selectedToolId, setSelectedToolId] = useState("");
  const selectedTool = TOOLS.find((tool) => tool.id === selectedToolId) ?? null;

  return (
    <section className="workspace-folder">
      <header className="workspace-masthead">
        <TrackMotif className="workspace-motif" />
        <div className="workspace-masthead-row">
          <div className="workspace-masthead-copy">
            <p className="home-kicker">Analysis</p>
            <h1 className="workspace-title">Tools</h1>
            <p className="workspace-lead">
              Pick a tool, choose an uploaded FASTA file, and run the pipeline.
            </p>
          </div>
        </div>
        <dl className="workspace-stats">
          <div>
            <dt>Available</dt>
            <dd>{TOOLS.length}</dd>
          </div>
          <div>
            <dt>FASTA ready</dt>
            <dd>{fastaFiles.length}</dd>
          </div>
        </dl>
      </header>

      <div className="workspace-assets">
        {!selectedTool ? (
          <ul className="tool-card-list">
            {TOOLS.map((tool) => (
              <li key={tool.id}>
                <button
                  className="tool-card"
                  onClick={() => setSelectedToolId(tool.id)}
                  type="button"
                >
                  <TrackMotif className="tool-card-motif" />
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
        ) : (
          <div className="tool-run-panel">
            <button className="tool-run-back" onClick={() => setSelectedToolId("")} type="button">
              All tools
            </button>

            <div className="tool-run-header">
              <span className="tool-card-icon tool-card-icon--large">
                <Dna size={22} />
              </span>
              <div>
                <h2>{selectedTool.name}</h2>
                <p>{selectedTool.description}</p>
              </div>
            </div>

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
              <>
                <p className="tool-run-label">Choose a file to analyze</p>
                <ul className="asset-list">
                  {fastaFiles.map((entry) => (
                    <li key={entry.path}>
                      <div className="asset-card">
                        <TrackMotif className="asset-card-motif" />
                        <div className="asset-card-body">
                          <div className="asset-card-main">
                            <strong>{entry.name}</strong>
                            <span>
                              {formatBytes(entry.size ?? 0)} · FASTA
                            </span>
                          </div>
                          <button
                            className="primary-action asset-run"
                            disabled={runStatus === "running"}
                            onClick={() => onRunPrediction(entry.path)}
                            type="button"
                          >
                            {runStatus === "running" ? (
                              <Loader2 className="spin" size={16} />
                            ) : (
                              <Play size={16} />
                            )}
                            Run
                          </button>
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
