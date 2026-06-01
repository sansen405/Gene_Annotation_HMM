const STEPS = 22;

function helixPoint(y, amplitude, phase) {
  const x = 80 + Math.cos(phase) * amplitude;
  return { x, y };
}

function buildStrandPath(amplitude, phaseOffset) {
  const points = [];
  for (let index = 0; index <= STEPS; index += 1) {
    const y = (index / STEPS) * 480;
    const phase = phaseOffset + (index / STEPS) * Math.PI * 3.5;
    points.push(helixPoint(y, amplitude, phase));
  }
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`)
    .join(" ");
}

function buildRungs(amplitude) {
  const rungs = [];
  for (let index = 0; index <= STEPS; index += 1) {
    const y = (index / STEPS) * 480;
    const phase = (index / STEPS) * Math.PI * 3.5;
    const x1 = 80 + Math.cos(phase) * amplitude;
    const x2 = 80 - Math.cos(phase) * amplitude;
    rungs.push({ x1, x2, y, key: index });
  }
  return rungs;
}

export function DnaStrand({ className = "" }) {
  const rungs = buildRungs(30);

  return (
    <svg
      aria-hidden="true"
      className={`dna-strand ${className}`.trim()}
      fill="none"
      viewBox="0 0 160 480"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path className="dna-strand-path dna-strand-path--a" d={buildStrandPath(30, 0)} />
      <path className="dna-strand-path dna-strand-path--b" d={buildStrandPath(30, Math.PI)} />
      {rungs.map((rung) => (
        <line
          className="dna-strand-rung"
          key={rung.key}
          x1={rung.x1}
          x2={rung.x2}
          y1={rung.y}
          y2={rung.y}
        />
      ))}
    </svg>
  );
}
