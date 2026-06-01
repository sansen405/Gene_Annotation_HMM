const SEGMENTS = [
  { w: 18, c: "utr" },
  { w: 8, c: "intron" },
  { w: 24, c: "exon" },
  { w: 6, c: "intron" },
  { w: 14, c: "exon" },
  { w: 5, c: "intron" },
  { w: 20, c: "exon" },
  { w: 10, c: "ig" },
  { w: 16, c: "exon" },
  { w: 7, c: "intron" },
  { w: 22, c: "exon" },
];

export function TrackMotif({ className = "" }) {
  return (
    <div aria-hidden="true" className={`track-motif ${className}`.trim()}>
      {SEGMENTS.map((seg, index) => (
        <span className={`track-seg track-seg--${seg.c}`} key={index} style={{ flex: seg.w }} />
      ))}
    </div>
  );
}
