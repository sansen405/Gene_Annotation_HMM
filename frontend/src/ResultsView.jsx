import {
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  FileArchive,
  FileCode2,
  // FlaskConical, // unused — only used in dead !selectedRun branch
  // Plus,         // unused — only used in dead !selectedRun branch
  Search,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

function intronCount(prediction) {
  return prediction?.intronCount ?? prediction?.intron_count ?? prediction?.introns?.length ?? 0;
}
const GENES_PER_PAGE = 10;
const MAX_CONFIDENCE_BP = 500;
const MAX_REGIONAL_CONFIDENCE_BP = 5000;
const DEFAULT_VIEW_BP = 100000;
const MIN_VIEW_BP = 5;
const MAX_VIEW_BP = 500000;

export function ResultsView({
  breadcrumbs = [],
  exportBaseUrl,
  onNavigate,
  onStartAnalysis,
  predictions,
  selectedPrediction,
  selectedRun,
  selectedPredictionId,
  setSelectedPredictionId,
}) {
  const firstScaffold = selectedRun?.scaffolds?.[0]?.name ?? "";
  const [scaffold, setScaffold] = useState(firstScaffold);
  const [start, setStart] = useState(1);
  const [end, setEnd] = useState(DEFAULT_VIEW_BP);
  const [frameStartInput, setFrameStartInput] = useState("1");
  const [frameEndInput, setFrameEndInput] = useState(String(DEFAULT_VIEW_BP));
  const [showConfidence, setShowConfidence] = useState(true);
  const [genePage, setGenePage] = useState(1);
  const [geneSearch, setGeneSearch] = useState("");
  const [geneRangeStart, setGeneRangeStart] = useState("");
  const [geneRangeEnd, setGeneRangeEnd] = useState("");
  const [rangeMessage, setRangeMessage] = useState("");

  const activeScaffold = scaffold || firstScaffold;
  function scaffoldLengthFor(scaffoldName) {
    const match = selectedRun?.scaffolds?.find((item) => item.name === scaffoldName);
    return match?.length ?? DEFAULT_VIEW_BP;
  }

  const scaffoldLength = scaffoldLengthFor(activeScaffold);

  useEffect(() => {
    const defaultEnd = Math.min(DEFAULT_VIEW_BP, scaffoldLengthFor(firstScaffold));
    setScaffold(firstScaffold);
    setStart(1);
    setEnd(defaultEnd);
    setFrameStartInput("1");
    setFrameEndInput(String(defaultEnd));
    setGenePage(1);
    setGeneSearch("");
    setGeneRangeStart("");
    setGeneRangeEnd("");
    setRangeMessage("");
  }, [firstScaffold, selectedRun?.id]);

  function clampRangeForLength(nextStart, nextEnd, length) {
    const boundedLength = Math.max(1, length);
    const requestedStart = Number(nextStart) || 1;
    const requestedEnd = Number(nextEnd) || requestedStart;
    const minFrame = Math.min(MIN_VIEW_BP, boundedLength);
    const maxFrame = Math.min(MAX_VIEW_BP, boundedLength);
    const messages = [];

    let normalizedStart = Math.max(1, Math.min(requestedStart, boundedLength));
    let normalizedEnd = Math.max(1, Math.min(requestedEnd, boundedLength));

    if (requestedStart < 1 || requestedEnd < 1 || requestedStart > boundedLength || requestedEnd > boundedLength) {
      messages.push(`Coordinates must stay within this scaffold (1-${formatNumber(boundedLength)} bp).`);
    }

    if (normalizedEnd < normalizedStart) {
      normalizedEnd = normalizedStart;
      messages.push("Start must be before end; adjusted the frame.");
    }

    let frameLength = normalizedEnd - normalizedStart + 1;
    if (frameLength < minFrame) {
      normalizedEnd = Math.min(boundedLength, normalizedStart + minFrame - 1);
      normalizedStart = Math.max(1, normalizedEnd - minFrame + 1);
      frameLength = normalizedEnd - normalizedStart + 1;
      messages.push(`Visualization frame must be at least ${MIN_VIEW_BP} bp.`);
    }

    if (frameLength > maxFrame) {
      normalizedEnd = Math.min(boundedLength, normalizedStart + maxFrame - 1);
      normalizedStart = Math.max(1, normalizedEnd - maxFrame + 1);
      messages.push(`Visualization frame cannot exceed ${formatNumber(MAX_VIEW_BP)} bp.`);
    }

    return { message: messages[0] ?? "", start: normalizedStart, end: normalizedEnd };
  }

  function clampRange(nextStart, nextEnd) {
    return clampRangeForLength(nextStart, nextEnd, scaffoldLength);
  }

  function validateFrameRange(nextStart, nextEnd, length) {
    const trimmedStart = String(nextStart).trim();
    const trimmedEnd = String(nextEnd).trim();
    const numericStart = Number(trimmedStart);
    const numericEnd = Number(trimmedEnd);
    const boundedLength = Math.max(1, length);

    if (!trimmedStart || !trimmedEnd || !Number.isFinite(numericStart) || !Number.isFinite(numericEnd)) {
      return { message: "Enter numeric start and end positions.", valid: false };
    }
    if (numericStart < 1 || numericEnd > boundedLength || numericEnd < 1 || numericStart > boundedLength) {
      return {
        message: `Coordinates must stay within this scaffold (1-${formatNumber(boundedLength)} bp).`,
        valid: false,
      };
    }
    if (numericEnd < numericStart) {
      return { message: "Start must be before end.", valid: false };
    }

    const frameLength = numericEnd - numericStart + 1;
    if (frameLength < MIN_VIEW_BP) {
      return { message: `Visualization frame must be at least ${MIN_VIEW_BP} bp.`, valid: false };
    }
    if (frameLength > MAX_VIEW_BP) {
      return { message: `Visualization frame cannot exceed ${formatNumber(MAX_VIEW_BP)} bp.`, valid: false };
    }

    return { end: numericEnd, message: "", start: numericStart, valid: true };
  }

  function setCommittedFrame(nextStart, nextEnd, message = "") {
    setStart(nextStart);
    setEnd(nextEnd);
    setFrameStartInput(String(nextStart));
    setFrameEndInput(String(nextEnd));
    setRangeMessage(message);
  }

  function handleFrameInputChange(field, value) {
    const nextStartInput = field === "start" ? value : frameStartInput;
    const nextEndInput = field === "end" ? value : frameEndInput;
    if (field === "start") {
      setFrameStartInput(value);
    } else {
      setFrameEndInput(value);
    }

    const validation = validateFrameRange(nextStartInput, nextEndInput, scaffoldLength);
    setRangeMessage(validation.message);
    if (validation.valid) {
      setStart(validation.start);
      setEnd(validation.end);
    }
  }

  function handleTrackRangeSelect(nextStart, nextEnd) {
    const next = clampRange(nextStart, nextEnd);
    const validation = validateFrameRange(next.start, next.end, scaffoldLength);
    if (!validation.valid) {
      setRangeMessage(validation.message);
      return;
    }
    setCommittedFrame(next.start, next.end, next.message);
  }

  function handlePredictionSelect(prediction) {
    const nextScaffoldLength = scaffoldLengthFor(prediction.scaffold);
    const geneLength = Math.max(prediction.end - prediction.start + 1, 1);
    const margin = Math.ceil(geneLength * 0.25);
    const next = clampRangeForLength(
      prediction.start - margin,
      prediction.end + margin,
      nextScaffoldLength
    );

    setSelectedPredictionId(prediction.id);
    setScaffold(prediction.scaffold);
    setCommittedFrame(next.start, next.end, next.message);
  }

  const filteredPredictions = useMemo(() => {
    const query = geneSearch.trim().toLowerCase();
    const rangeStart = Number(geneRangeStart);
    const rangeEnd = Number(geneRangeEnd);
    const hasRangeStart = geneRangeStart !== "" && Number.isFinite(rangeStart);
    const hasRangeEnd = geneRangeEnd !== "" && Number.isFinite(rangeEnd);
    const minBp = hasRangeStart && hasRangeEnd ? Math.min(rangeStart, rangeEnd) : rangeStart;
    const maxBp = hasRangeStart && hasRangeEnd ? Math.max(rangeStart, rangeEnd) : rangeEnd;

    return predictions.filter(
      (prediction) => {
        const matchesQuery =
          !query ||
        prediction.id.toLowerCase().includes(query) ||
          prediction.scaffold.toLowerCase().includes(query);
        const matchesRange =
          (!hasRangeStart || prediction.end >= minBp) &&
          (!hasRangeEnd || prediction.start <= maxBp);
        return matchesQuery && matchesRange;
      }
    );
  }, [geneRangeEnd, geneRangeStart, geneSearch, predictions]);

  const totalGenePages = Math.max(1, Math.ceil(filteredPredictions.length / GENES_PER_PAGE));

  useEffect(() => {
    setGenePage((page) => Math.min(page, totalGenePages));
  }, [totalGenePages]);

  const pagedPredictions = filteredPredictions.slice(
    (genePage - 1) * GENES_PER_PAGE,
    genePage * GENES_PER_PAGE
  );
  const confidenceData = useMemo(() => {
    const allConfidence = selectedRun?.confidenceByScaffold?.[activeScaffold] ?? [];
    return allConfidence.filter(
      (base) => base.position >= start && base.position <= end
    );
  }, [activeScaffold, end, selectedRun, start]);

  // ResultsView is only mounted when selectedRun is truthy (App.jsx guards the render),
  // so this branch is unreachable. Kept for reference.
  // if (!selectedRun) {
  //   return (
  //     <section className="page results-page">
  //       <PageHeader
  //         section="Results"
  //         title="Prediction results"
  //         subtitle="Run a local prediction to view genes, tracks, and confidence."
  //       />
  //       <div className="card empty-results">
  //         <div className="empty-results-icon">
  //           <FlaskConical size={28} strokeWidth={1.75} />
  //         </div>
  //         <h2>No results yet</h2>
  //         <p>Upload a FASTA file and run the HMM predictor to see genes, tracks, and confidence here.</p>
  //         <button className="primary-action" onClick={onStartAnalysis} type="button">
  //           <Plus size={17} />
  //           Start new analysis
  //         </button>
  //       </div>
  //     </section>
  //   );
  // }

  return (
    <>
      <header className="workspace-masthead workspace-masthead--file-detail workspace-masthead--results">
        <div className="results-masthead-row">
          <div className="results-masthead-copy">
            <p className="home-kicker">Results</p>
            <h1 className="workspace-title">{selectedRun.fileName || selectedRun.name}</h1>
            <p className="workspace-lead">
              {selectedRun.date}
              {" · "}
              {formatNumber(selectedRun.summary.genes)} genes
              {" · "}
              {formatNumber(selectedRun.summary.totalBases)} bp
              {" · "}
              {selectedRun.summary.scaffolds} scaffold
              {selectedRun.summary.scaffolds === 1 ? "" : "s"}
            </p>
          </div>
          <div className="results-masthead-actions">
            <a className="ghost-action" href={`${exportBaseUrl}/gff3`}>
              <FileCode2 size={15} />
              GFF3
            </a>
            <a className="ghost-action" href={`${exportBaseUrl}/csv`}>
              <Download size={15} />
              CSV
            </a>
            <a className="primary-action inline" href={`${exportBaseUrl}/gff3`}>
              <Download size={15} />
              Export
            </a>
          </div>
        </div>
      </header>

      <section className="page results-page results-page--viewer">
      <div className="card genome-viewer genome-viewer--hero">
        <div className="viewer-controls">
          <label>
            Scaffold
            <select
              onChange={(event) => {
                const nextScaffold = event.target.value;
                const nextLength =
                  selectedRun.scaffolds.find((item) => item.name === nextScaffold)?.length ??
                  DEFAULT_VIEW_BP;
                const nextEnd = Math.min(DEFAULT_VIEW_BP, nextLength);
                setScaffold(nextScaffold);
                setCommittedFrame(1, nextEnd);
              }}
              value={activeScaffold}
            >
              {selectedRun.scaffolds.map((item) => (
                <option key={item.name} value={item.name}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Start
            <input
              onBlur={() => setFrameStartInput(String(start))}
              onChange={(event) => handleFrameInputChange("start", event.target.value)}
              type="number"
              value={frameStartInput}
            />
          </label>
          <label>
            End
            <input
              onBlur={() => setFrameEndInput(String(end))}
              onChange={(event) => handleFrameInputChange("end", event.target.value)}
              type="number"
              value={frameEndInput}
            />
          </label>
        </div>
        {rangeMessage ? (
          <div className="track-wrap">
            <div className="frame-error">{rangeMessage}</div>
          </div>
        ) : (
          <>
            <TrackLegend />
            <GenomeTrack
              onRangeSelect={handleTrackRangeSelect}
              predictions={predictions.filter((item) => item.scaffold === activeScaffold)}
              start={start}
              end={end}
            />
          </>
        )}

        <ConfidencePanel
          confidenceData={confidenceData}
          end={end}
          rangeMessage={rangeMessage}
          showConfidence={showConfidence}
          start={start}
          toggleConfidence={() => setShowConfidence((visible) => !visible)}
        />
      </div>

      <section className="genes-workspace">
        <div className="genes-workspace-head">
          <div className="genes-workspace-copy">
            <p className="home-kicker">Gene list</p>
            <h2 className="genes-workspace-title">Predicted genes</h2>
            <p className="genes-workspace-count">
              {filteredPredictions.length === 0
                ? "No genes in this view"
                : `${formatNumber(filteredPredictions.length)} genes · page ${genePage} of ${totalGenePages}`}
            </p>
          </div>
          <div className="genes-workspace-filters">
            <label className="genes-filter genes-filter--search">
              <span className="genes-filter-label">Search</span>
              <span className="genes-filter-control">
                <Search size={15} />
                <input
                  onChange={(event) => {
                    setGeneSearch(event.target.value);
                    setGenePage(1);
                  }}
                  placeholder="ID or scaffold"
                  value={geneSearch}
                />
              </span>
            </label>
            <label className="genes-filter genes-filter--range">
              <span className="genes-filter-label">BP range</span>
              <span className="genes-filter-range">
                <input
                  min={1}
                  onChange={(event) => {
                    setGeneRangeStart(event.target.value);
                    setGenePage(1);
                  }}
                  placeholder="Start"
                  type="number"
                  value={geneRangeStart}
                />
                <span aria-hidden="true">–</span>
                <input
                  min={1}
                  onChange={(event) => {
                    setGeneRangeEnd(event.target.value);
                    setGenePage(1);
                  }}
                  placeholder="End"
                  type="number"
                  value={geneRangeEnd}
                />
              </span>
            </label>
          </div>
        </div>

        <div className="results-grid">
          <div className="card table-card">
            <div className="table-scroll">
              <table className="genes-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Scaffold</th>
                    <th className="num">Start</th>
                    <th className="num">End</th>
                    <th className="num">Len</th>
                    <th className="num">UTR</th>
                    <th className="num">Introns</th>
                  </tr>
                </thead>
                <tbody>
                  {pagedPredictions.length === 0 ? (
                    <tr>
                      <td className="genes-table-empty" colSpan={7}>
                        No genes match your filters.
                      </td>
                    </tr>
                  ) : (
                    pagedPredictions.map((prediction) => (
                      <tr
                        className={
                          prediction.id === selectedPredictionId ? "selected" : ""
                        }
                        key={prediction.id}
                        onClick={() => handlePredictionSelect(prediction)}
                      >
                        <td className="gene-id">{prediction.id}</td>
                        <td>{prediction.scaffold}</td>
                        <td className="num">{formatNumber(prediction.start)}</td>
                        <td className="num">{formatNumber(prediction.end)}</td>
                        <td className="num">
                          {formatNumber(prediction.end - prediction.start + 1)}
                        </td>
                        <td className="num">{prediction.exons.length}</td>
                        <td className="num">{intronCount(prediction)}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
            <div className="table-pagination">
              <span>
                {filteredPredictions.length === 0
                  ? "No genes"
                  : `${(genePage - 1) * GENES_PER_PAGE + 1}-${Math.min(
                      genePage * GENES_PER_PAGE,
                      filteredPredictions.length
                    )} of ${filteredPredictions.length}`}
              </span>
              <div className="pagination-controls">
                <button
                  aria-label="Previous page"
                  disabled={genePage <= 1}
                  onClick={() => setGenePage((page) => Math.max(1, page - 1))}
                  type="button"
                >
                  <ChevronLeft size={16} />
                </button>
                <span>
                  Page {genePage} / {totalGenePages}
                </span>
                <button
                  aria-label="Next page"
                  disabled={genePage >= totalGenePages}
                  onClick={() =>
                    setGenePage((page) => Math.min(totalGenePages, page + 1))
                  }
                  type="button"
                >
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>
          </div>

          <SelectedPrediction prediction={selectedPrediction} />
        </div>

        <ExportCard exportBaseUrl={exportBaseUrl} />
      </section>
      </section>
    </>
  );
}

function TrackLegend() {
  return (
    <ul aria-label="Track legend" className="track-legend">
      <li>
        <span className="track-legend-swatch track-legend-swatch--exon" />
        Exon
      </li>
      <li>
        <span className="track-legend-swatch track-legend-swatch--intron" />
        Intron
      </li>
      <li>
        <span className="track-legend-swatch track-legend-swatch--ig" />
        Intergenic
      </li>
    </ul>
  );
}

function ConfidencePanel({
  confidenceData,
  end,
  rangeMessage,
  showConfidence,
  start,
  toggleConfidence,
}) {
  const rangeLength = Math.max(end - start + 1, 0);
  const canShowPerBase = rangeLength > 0 && rangeLength <= MAX_CONFIDENCE_BP;
  const canShowRegional = rangeLength > 0 && rangeLength <= MAX_REGIONAL_CONFIDENCE_BP;

  return (
    <div className="confidence-panel">
      <div className="confidence-header">
        <div>
          <h3>Prediction Confidence</h3>
          <p>
            {showConfidence
              ? "Per-base Forward-Backward posterior for the selected range"
              : "Average confidence by predicted region in the selected range"}
          </p>
        </div>
        <label className="toggle-row">
          <span>Per-base detail</span>
          <button
            aria-pressed={showConfidence}
            className={showConfidence ? "toggle on" : "toggle"}
            onClick={toggleConfidence}
            type="button"
          >
            <span />
          </button>
        </label>
      </div>

      {rangeMessage ? (
        <div className="confidence-empty error">{rangeMessage}</div>
      ) : confidenceData.length === 0 ? (
        <div className="confidence-empty">
          No confidence values are available for this range.
        </div>
      ) : showConfidence && !canShowPerBase ? (
        <div className="confidence-empty">
          {canShowRegional
            ? "Per-base detail is limited to 500 bp. Turn it off to view regional averages for this window."
            : "Per-base detail is limited to 500 bp. Regional averages are limited to 5,000 bp."}
        </div>
      ) : !showConfidence && !canShowRegional ? (
        <div className="confidence-empty">
          Regional averages are limited to 5,000 bp. Select a smaller window to view confidence.
        </div>
      ) : !showConfidence ? (
        <>
          <div className="confidence-chart grouped" aria-label="Average confidence by predicted group">
            <div className="confidence-y-axis">
              <span>1.0</span>
              <span>0.5</span>
              <span>0.0</span>
            </div>
            <div className="confidence-bars grouped-bars">
              {groupConfidenceData(confidenceData).map((group) => (
                <div className="confidence-group" key={`${group.state}-${group.start}-${group.end}`}>
                  <div className="confidence-group-bar-area">
                    <div
                      className={`confidence-bar ${stateClass(group.state)}`}
                      style={{ height: `${Math.max(group.average * 100, 4)}%` }}
                      title={`Range: ${group.start}-${group.end}\nState: ${group.state}\nAverage confidence: ${group.average.toFixed(2)}`}
                    />
                  </div>
                  <span>{group.state}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="confidence-axis">
            <div className="confidence-axis-labels">
              <span className="axis-edge">{start}</span>
              <span className="axis-center">Regional averages</span>
              <span className="axis-edge">{end}</span>
            </div>
          </div>
        </>
      ) : (
        <>
          <div className="confidence-chart per-base" aria-label="Per-base confidence histogram">
            <div className="confidence-y-axis">
              <span>1.0</span>
              <span>0.5</span>
              <span>0.0</span>
            </div>
            <div className="confidence-bars">
              {confidenceData.map((base) => (
                <div
                  className={`confidence-bar ${stateClass(base.state)}`}
                  key={base.position}
                  style={{ height: `${Math.max(base.confidence * 100, 4)}%` }}
                  title={`Position: ${base.position}\nState: ${base.state}\nConfidence: ${base.confidence.toFixed(2)}`}
                />
              ))}
            </div>
          </div>
          <div className="confidence-axis">
            <div className="confidence-axis-labels">
              <span className="axis-edge">{start}</span>
              <span className="axis-center">Genomic position (bp)</span>
              <span className="axis-edge">{end}</span>
            </div>
          </div>
        </>
      )}

      <p className="confidence-note">
        Confidence is the model posterior for the predicted state at each base;
        it is not biological accuracy unless compared against a reference annotation.
      </p>
    </div>
  );
}

function GenomeTrack({ predictions, start, end, onRangeSelect }) {
  const trackRef = useRef(null);
  const [selection, setSelection] = useState(null);
  const visiblePredictions = predictions.filter(
    (prediction) => prediction.end >= start && prediction.start <= end
  );
  const span = Math.max(end - start, 0);
  const rangeLength = Math.max(end - start + 1, 1);

  function positionRatio(position) {
    if (span === 0) return 0.5;
    return (position - start) / span;
  }

  function xFor(position) {
    return 2 + positionRatio(position) * 96;
  }

  function boundaryFor(position, edge = "start") {
    const offset = edge === "end" ? 1 : 0;
    return 2 + ((position - start + offset) / rangeLength) * 96;
  }

  function intervalRect(intervalStart, intervalEnd, minWidth = 0.35) {
    if (intervalEnd < start || intervalStart > end) return null;
    const clippedStart = Math.max(intervalStart, start);
    const clippedEnd = Math.min(intervalEnd, end);
    const x = boundaryFor(clippedStart, "start");
    return {
      width: Math.max(boundaryFor(clippedEnd, "end") - x, minWidth),
      x,
    };
  }

  function bpFromClientX(clientX) {
    const rect = trackRef.current?.getBoundingClientRect();
    if (!rect?.width) return start;
    const ratio = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
    if (span === 0) return start;
    if (ratio <= 0) return start;
    if (ratio >= 1) return end;
    return Math.min(end, Math.max(start, Math.round(start + ratio * span)));
  }

  function handlePointerDown(event) {
    if (event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const anchorBp = bpFromClientX(event.clientX);
    setSelection({ anchorBp, currentBp: anchorBp });
  }

  function handlePointerMove(event) {
    if (!selection) return;
    setSelection((current) =>
      current ? { ...current, currentBp: bpFromClientX(event.clientX) } : current
    );
  }

  function finishSelection(event) {
    if (!selection) return;
    event.currentTarget.releasePointerCapture(event.pointerId);
    const nextStart = Math.min(selection.anchorBp, selection.currentBp);
    const nextEnd = Math.max(selection.anchorBp, selection.currentBp);
    setSelection(null);
    if (nextEnd >= nextStart) {
      onRangeSelect(nextStart, nextEnd);
    }
  }

  const selectionStyle = useMemo(() => {
    if (!selection) return null;
    const leftBp = Math.min(selection.anchorBp, selection.currentBp);
    const rightBp = Math.max(selection.anchorBp, selection.currentBp);
    if (span === 0) {
      return { left: "0%", width: "100%" };
    }
    const leftPct = ((leftBp - start) / span) * 100;
    const widthPct = ((rightBp - leftBp) / span) * 100;
    return {
      left: `${Math.max(0, leftPct)}%`,
      width: `${Math.min(100 - leftPct, Math.max(0.4, widthPct))}%`,
    };
  }, [end, selection, span, start]);

  return (
    <div className="track-wrap">
      <div className="axis-labels">
        <span>{start}</span>
        <span>{span === 0 ? start : Math.round(start + span * 0.33)}</span>
        <span>{span === 0 ? start : Math.round(start + span * 0.66)}</span>
        <span>{end}</span>
      </div>
      <div className="track-interactive" ref={trackRef}>
        <svg
          className="gene-track"
          onPointerCancel={finishSelection}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={finishSelection}
          viewBox="0 0 100 22"
          preserveAspectRatio="none"
        >
        <line className="intergenic-line" x1="2" x2="98" y1="11" y2="11" />
        {visiblePredictions.map((prediction) => {
          const showStartMarker = prediction.start >= start && prediction.start <= end;
          const showStopMarker = prediction.end >= start && prediction.end <= end;
          const geneStart = showStartMarker ? boundaryFor(prediction.start, "start") : null;
          const geneEnd = showStopMarker ? boundaryFor(prediction.end, "end") : null;
          return (
            <g key={prediction.id}>
              {prediction.introns.map((intron) => {
                const intronRect = intervalRect(intron.start, intron.end);
                if (!intronRect) return null;
                return (
                  <rect
                    className="intron-block"
                    height="2"
                    key={`${prediction.id}-intron-${intron.start}`}
                    width={intronRect.width}
                    x={intronRect.x}
                    y="10"
                  />
                );
              })}
              {prediction.exons.map((exon) => {
                const exonRect = intervalRect(exon.start, exon.end);
                if (!exonRect) return null;
                return (
                  <rect
                    className="exon-block"
                    height="5"
                    key={`${prediction.id}-${exon.start}`}
                    width={exonRect.width}
                    x={exonRect.x}
                    y="8.5"
                  />
                );
              })}
              {showStartMarker && (
                <line
                  className="start-marker"
                  x1={geneStart}
                  x2={geneStart}
                  y1="5"
                  y2="17"
                />
              )}
              {showStopMarker && (
                <line
                  className="stop-marker"
                  x1={geneEnd}
                  x2={geneEnd}
                  y1="5"
                  y2="17"
                />
              )}
            </g>
          );
        })}
        </svg>
        {selectionStyle && (
          <div className="range-selection" style={selectionStyle} />
        )}
      </div>
      <p className="track-hint">Drag on the track to select a sub-range</p>
      {/* Duplicate legend removed — TrackLegend component renders above the track.
      <div className="track-legend">
        <span><i className="legend-gray" /> Intergenic</span>
        <span><i className="legend-blue" /> UTR</span>
        <span><i className="legend-purple" /> Intron</span>
        <span><i className="legend-green" /> Start</span>
        <span><i className="legend-red" /> Stop</span>
      </div> */}
    </div>
  );
}

function SelectedPrediction({ prediction }) {
  if (!prediction) {
    return (
      <div className="card selected-card">
        <p className="home-kicker">Selected gene</p>
        <h2 className="selected-gene-title">No selection</h2>
        <p className="selected-empty-copy">Choose a row in the table to inspect locus details and sequence.</p>
      </div>
    );
  }

  const geneLength = prediction.end - prediction.start + 1;

  return (
    <div className="card selected-card">
      <div className="selected-head">
        <div>
          <p className="home-kicker">Selected gene</p>
          <h2 className="selected-gene-title">{prediction.id}</h2>
          <p className="selected-gene-locus">
            {prediction.scaffold}:{formatNumber(prediction.start)}–{formatNumber(prediction.end)}
          </p>
        </div>
        <button
          aria-label="Copy sequence"
          className="selected-copy"
          onClick={() => navigator.clipboard?.writeText(prediction.sequence || "")}
          type="button"
        >
          <Copy size={15} />
        </button>
      </div>

      <dl className="selected-metrics">
        <div>
          <dt>Length</dt>
          <dd>{formatNumber(geneLength)} bp</dd>
        </div>
        <div>
          <dt>UTR segments</dt>
          <dd>{prediction.exons.length}</dd>
        </div>
        <div>
          <dt>Introns</dt>
          <dd>{intronCount(prediction)}</dd>
        </div>
      </dl>

      <div className="selected-details">
        <div className="selected-detail-block">
          <h3>UTR intervals</h3>
          <p>{formatIntervals(prediction.exons)}</p>
        </div>
        <div className="selected-detail-block">
          <h3>Codon positions</h3>
          <p>
            Start {formatNumber(prediction.start)} · Stop {formatNumber(prediction.end - 2)}
          </p>
        </div>
      </div>

      <div className="selected-sequence">
        <h3>Sequence preview</h3>
        <pre>{highlightSequence(prediction.sequence || "")}</pre>
      </div>
    </div>
  );
}

function ExportCard({ exportBaseUrl }) {
  const formats = [
    { ext: "gff3", label: "GFF3", hint: "Annotation", icon: FileCode2 },
    { ext: "csv", label: "CSV", hint: "Coordinates", icon: Download },
    { ext: "bed", label: "BED", hint: "Intervals", icon: Download },
    { ext: "fasta", label: "FASTA", hint: "Nucleotides", icon: FileArchive },
  ];

  return (
    <div className="card export-strip">
      <div className="export-strip-head">
        <div>
          <p className="home-kicker">Download</p>
          <h2 className="export-strip-title">Export predictions</h2>
        </div>
        <p className="export-strip-lead">Standard formats for downstream analysis or genome browsers.</p>
      </div>
      <div className="export-strip-formats">
        {formats.map(({ ext, hint, icon: Icon, label }) => (
          <a className="export-format-link" href={`${exportBaseUrl}/${ext}`} key={ext}>
            <Icon size={16} strokeWidth={1.75} />
            <span className="export-format-copy">
              <strong>{label}</strong>
              <small>{hint}</small>
            </span>
          </a>
        ))}
      </div>
    </div>
  );
}

// Unused — only referenced from the dead !selectedRun branch above.
// function PageHeader({ section, title, subtitle }) {
//   return (
//     <header className="page-header">
//       {section && <p className="page-section">{section}</p>}
//       <h1>{title}</h1>
//       {subtitle ? <p className="page-subtitle">{subtitle}</p> : null}
//     </header>
//   );
// }

// Unused — defined but never called.
// function SectionTitle({ icon, title }) {
//   return (
//     <div className="section-title">
//       <div className="section-icon">{icon}</div>
//       <h2>{title}</h2>
//     </div>
//   );
// }

// Unused — defined but never called.
// function Metric({ label, value }) {
//   return (
//     <div className="metric">
//       <span>{label}</span>
//       <strong>{value}</strong>
//     </div>
//   );
// }

// Unused in this file — defined but never called here.
// function formatBytes(bytes) {
//   if (bytes < 1024) return `${bytes} B`;
//   if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
//   return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
// }

function formatNumber(value) {
  if (typeof value !== "number") return value;
  return new Intl.NumberFormat("en-US").format(value);
}

function formatIntervals(intervals) {
  return intervals.map((item) => `${item.start}-${item.end}`).join(", ");
}

function highlightSequence(sequence) {
  if (sequence.length <= 6) return sequence;
  return (
    <>
      <span className="start-codon">{sequence.slice(0, 3)}</span>
      {" "}
      {sequence.slice(3, -3)}
      {" "}
      <span className="stop-codon">{sequence.slice(-3)}</span>
    </>
  );
}

function stateClass(state) {
  return state.toLowerCase();
}

function groupConfidenceData(confidenceData) {
  const groups = [];

  for (const base of confidenceData) {
    const previous = groups[groups.length - 1];
    if (previous && previous.state === base.state && previous.end + 1 === base.position) {
      previous.end = base.position;
      previous.total += base.confidence;
      previous.count++;
      previous.average = previous.total / previous.count;
      continue;
    }

    groups.push({
      average: base.confidence,
      count: 1,
      end: base.position,
      start: base.position,
      state: base.state,
      total: base.confidence,
    });
  }

  return groups;
}

