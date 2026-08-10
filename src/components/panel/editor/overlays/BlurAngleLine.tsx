interface BlurAngleLineProps {
  angle: number;
}

export default function BlurAngleLine({ angle }: BlurAngleLineProps) {
  return (
    <div aria-hidden="true" className="absolute inset-0 z-40 pointer-events-none">
      <svg
        width={1}
        height={1}
        style={{ position: 'absolute', left: '50%', top: '50%', overflow: 'visible', opacity: 0.9 }}
      >
        <g transform={`rotate(${angle})`}>
          <line
            x1={-4000}
            y1={0}
            x2={4000}
            y2={0}
            stroke="rgba(255, 255, 255, 0.85)"
            strokeWidth={1.5}
            strokeDasharray="6 4"
          />
        </g>
        <circle cx={0} cy={0} r={4} fill="none" stroke="rgba(255, 255, 255, 0.85)" strokeWidth={1.5} />
      </svg>
    </div>
  );
}
