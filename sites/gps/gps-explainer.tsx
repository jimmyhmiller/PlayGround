import { useState, useEffect, useRef, useCallback } from "react";

const W = 800, H = 580;
const GROUND_Y = 470;
const C = 0.5;
const CIRCLE_STROKES = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#f7dc6f"];

const gpsSats = [
  { id: 0, x: 100, y: 55, label: "SV1" },
  { id: 1, x: 300, y: 35, label: "SV2" },
  { id: 2, x: 500, y: 45, label: "SV3" },
  { id: 3, x: 680, y: 65, label: "SV4" },
];

const geoSat = { x: 710, y: 130, label: "GEO" };
const groundStation = { x: 620, y: GROUND_Y - 8 };
const uplink = { x: 660, y: GROUND_Y - 50 };

function dist(a, b) { return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2); }

function circleIntersections(c1, r1, c2, r2) {
  const dx = c2.x - c1.x, dy = c2.y - c1.y;
  const d = Math.sqrt(dx * dx + dy * dy);
  if (d > r1 + r2 || d < Math.abs(r1 - r2) || d === 0) return [];
  const a = (r1 * r1 - r2 * r2 + d * d) / (2 * d);
  const hSq = r1 * r1 - a * a;
  if (hSq < 0) return [];
  const h = Math.sqrt(hSq);
  const mx = c1.x + a * dx / d, my = c1.y + a * dy / d;
  return [
    { x: mx + h * dy / d, y: my - h * dx / d },
    { x: mx - h * dy / d, y: my + h * dx / d },
  ];
}

function solveBias(sats, measDists) {
  let bestB = 0, bestErr = Infinity;
  for (let b = -200; b <= 200; b += 0.5) {
    const corr = measDists.map(d => d - b);
    if (corr.some(d => d < 5)) continue;
    const pts = circleIntersections(sats[0], corr[0], sats[1], corr[1]);
    if (!pts.length) continue;
    for (const p of pts) {
      let err = 0;
      for (let i = 2; i < sats.length; i++) err += Math.abs(dist(p, sats[i]) - corr[i]);
      if (err < bestErr) { bestErr = err; bestB = b; }
    }
  }
  return bestB;
}

function findFix(sats, dists) {
  if (sats.length < 3) return null;
  const pts01 = circleIntersections(sats[0], dists[0], sats[1], dists[1]);
  const pts02 = circleIntersections(sats[0], dists[0], sats[2], dists[2]);
  let best = null, bestD = Infinity;
  for (const p1 of pts01) for (const p2 of pts02) {
    const d = dist(p1, p2);
    if (d < bestD) { bestD = d; best = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 }; }
  }
  return bestD < 50 ? best : null;
}

function Satellite({ x, y, label, color, active, geo }) {
  if (geo) return (
    <g opacity={active ? 1 : 0.25}>
      <polygon points={`${x},${y - 16} ${x + 14},${y + 8} ${x - 14},${y + 8}`} fill="#226" stroke="#88f" strokeWidth={1.5} />
      <line x1={x - 20} y1={y} x2={x - 14} y2={y} stroke="#88f" strokeWidth={2} />
      <line x1={x + 14} y1={y} x2={x + 20} y2={y} stroke="#88f" strokeWidth={2} />
      <rect x={x - 22} y={y - 5} width={6} height={10} rx={1} fill="#88f" opacity={0.7} />
      <rect x={x + 16} y={y - 5} width={6} height={10} rx={1} fill="#88f" opacity={0.7} />
      <text x={x} y={y + 26} textAnchor="middle" fill="#88f" fontSize={10} fontWeight="bold">{label}</text>
    </g>
  );
  return (
    <g opacity={active ? 1 : 0.2}>
      <rect x={x - 18} y={y - 10} width={36} height={20} rx={3} fill="#334" stroke={active ? color : "#555"} strokeWidth={1.5} />
      <line x1={x - 28} y1={y} x2={x - 18} y2={y} stroke={active ? color : "#555"} strokeWidth={2} />
      <line x1={x + 18} y1={y} x2={x + 28} y2={y} stroke={active ? color : "#555"} strokeWidth={2} />
      <rect x={x - 30} y={y - 6} width={6} height={12} rx={1} fill={active ? color : "#555"} opacity={0.7} />
      <rect x={x + 24} y={y - 6} width={6} height={12} rx={1} fill={active ? color : "#555"} opacity={0.7} />
      <text x={x} y={y + 28} textAnchor="middle" fill={active ? color : "#555"} fontSize={11} fontWeight="bold">{label}</text>
    </g>
  );
}

function Plane({ x, y }) {
  return (
    <g>
      <ellipse cx={x} cy={y} rx={28} ry={8} fill="#e0e0e0" />
      <polygon points={`${x - 8},${y} ${x - 22},${y - 18} ${x + 2},${y - 2}`} fill="#e0e0e0" opacity={0.9} />
      <polygon points={`${x + 15},${y} ${x + 28},${y - 8} ${x + 20},${y - 1}`} fill="#e0e0e0" opacity={0.7} />
      <circle cx={x + 22} cy={y + 1} r={2} fill="#333" />
      <text x={x} y={y + 24} textAnchor="middle" fill="#fff" fontSize={11} fontWeight="bold">✈ You</text>
    </g>
  );
}

function Pulse({ from, to, color, delay, dur = 2.5 }) {
  const [t, setT] = useState(0);
  useEffect(() => {
    let s = null, raf;
    const anim = ts => { if (!s) s = ts; const e = ((ts - s) / 1000 - delay) % dur; setT(e < 0 ? 0 : e / dur); raf = requestAnimationFrame(anim); };
    raf = requestAnimationFrame(anim);
    return () => cancelAnimationFrame(raf);
  }, [delay, dur]);
  if (t <= 0) return null;
  const x = from.x + (to.x - from.x) * t, y = from.y + (to.y - from.y) * t;
  const o = t < 0.8 ? 0.8 : (1 - t) * 4;
  return <circle cx={x} cy={y} r={4} fill={color} opacity={Math.max(0, o)} />;
}

function GroundStation({ x, y, active }) {
  return (
    <g opacity={active ? 1 : 0.2}>
      {/* building */}
      <rect x={x - 14} y={y - 20} width={28} height={20} rx={2} fill="#555" stroke="#888" strokeWidth={1} />
      {/* antenna dish */}
      <line x1={x} y1={y - 20} x2={x} y2={y - 34} stroke="#aaa" strokeWidth={2} />
      <path d={`M${x - 8},${y - 38} Q${x},${y - 30} ${x + 8},${y - 38}`} fill="none" stroke="#aaa" strokeWidth={2} />
      <circle cx={x} cy={y - 34} r={2} fill="#fff" />
      {/* label */}
      <text x={x} y={y + 14} textAnchor="middle" fill={active ? "#88f" : "#666"} fontSize={9} fontWeight="bold">GROUND</text>
      <text x={x} y={y + 24} textAnchor="middle" fill={active ? "#88f" : "#666"} fontSize={9} fontWeight="bold">STATION</text>
      {/* known position marker */}
      {active && <circle cx={x} cy={y - 10} r={3} fill="#0f0">
        <animate attributeName="opacity" values="1;0.3;1" dur="2s" repeatCount="indefinite" />
      </circle>}
    </g>
  );
}

const STEPS = [
  { n: 0, title: "No Satellites", desc: "Without any satellite signals, the plane has no idea where it is." },
  { n: 1, title: "1 Satellite — A Ring of Uncertainty", desc: "One satellite gives a distance — but that's a circle of possibilities." },
  { n: 2, title: "2 Satellites — Two Candidates", desc: "Two range circles intersect at two points." },
  { n: 3, title: "3 Satellites — Fix (Perfect Clock Only)", desc: "Three circles meet at one point. But this only works if the clock is perfect. Add clock error below to see the problem!" },
  { n: 4, title: "4 Satellites — Solves Clock Error", desc: "The 4th satellite over-determines the system, letting the receiver solve for its own clock bias. This gives an accurate fix even with a cheap quartz crystal." },
];

export default function GPSExplainer() {
  const [step, setStep] = useState(4);
  const [planePos, setPlanePos] = useState({ x: 360, y: 280 });
  const [dragging, setDragging] = useState(false);
  const [clockErr, setClockErr] = useState(0);
  const [ionoErr, setIonoErr] = useState(0);
  const [orbitErr, setOrbitErr] = useState(0);
  const [waasOn, setWaasOn] = useState(false);
  const [showMath, setShowMath] = useState(false);
  const svgRef = useRef(null);

  const { n: activeSats, title, desc } = STEPS[step];
  const satList = gpsSats.slice(0, activeSats);

  // Per-satellite errors: iono and orbit affect each sat differently but consistently (same for ground station and plane)
  const satErrors = gpsSats.map((_, i) => {
    const ionoFactor = [1.0, 0.7, 1.3, 0.9][i];
    const orbitFactor = [0.8, 1.2, 0.6, 1.1][i];
    return (ionoErr * ionoFactor + orbitErr * orbitFactor);
  });

  const trueDists = satList.map(s => dist(s, planePos));
  const trueTimes = trueDists.map(d => d / C);

  // Ground station true distances and errors
  const gsTrueDists = gpsSats.map(s => dist(s, groundStation));
  const gsTrueTimes = gsTrueDists.map(d => d / C);

  // Ground station measured (with same atmospheric/orbit errors)
  const gsMeasuredTimes = gsTrueTimes.map((t, i) => t + clockErr + satErrors[i]);
  const gsMeasuredDists = gsMeasuredTimes.map(t => t * C);
  // Ground station KNOWS its position, so it knows the true distance
  // Correction per sat = true dist - measured dist (what needs to be added to fix)
  const corrections = gpsSats.map((s, i) => gsTrueDists[i] - gsMeasuredDists[i]);

  // Plane measured times include clock + atmo + orbit errors
  const measuredTimes = trueTimes.map((t, i) => t + clockErr + satErrors[i]);
  const measuredDists = measuredTimes.map(t => t * C);

  // With WAAS: apply corrections (removes atmo + orbit errors, clock error remains)
  // correction[i] = -(clockErr + satErrors[i]) * C ... so applying it removes satErrors but also removes clockErr
  // Actually the ground station has its OWN clock synced to GPS, so the correction isolates the atmospheric/orbital part.
  // Let's model it properly: ground station has a perfect clock, so:
  // gs_measured_time[i] = gs_true_time[i] + satErrors[i] (no clock error for ground station)
  // correction[i] = gs_true_dist[i] - gs_measured_dist[i] = -satErrors[i] * C
  // plane corrected dist[i] = plane_measured_dist[i] + correction[i] 
  //   = (trueDist[i] + (clockErr + satErrors[i]) * C) + (-satErrors[i] * C)
  //   = trueDist[i] + clockErr * C  <-- only clock error remains!
  
  const gsCorrections = gpsSats.map((_, i) => -satErrors[i] * C);
  const waasCorrectedDists = measuredDists.map((d, i) => d + gsCorrections[i]);
  // After WAAS correction, only clock error remains → 4th sat solves it

  const hasAtmoErrors = ionoErr !== 0 || orbitErr !== 0;
  const hasClockErr = clockErr !== 0;
  const hasAnyError = hasClockErr || hasAtmoErrors;

  // Determine which distances to use for fix
  let fixDists = measuredDists;
  let solvedBias = 0;
  let correctionApplied = false;

  if (waasOn && hasAtmoErrors) {
    fixDists = waasCorrectedDists;
    correctionApplied = true;
  }

  // 4-sat clock correction
  if (activeSats === 4) {
    const remainingClockLike = fixDists.map((d, i) => d - trueDists[i]);
    // Check if there's residual bias to solve
    const avgBias = remainingClockLike.reduce((a, b) => a + b, 0) / remainingClockLike.length;
    if (Math.abs(avgBias) > 1) {
      solvedBias = solveBias(satList, fixDists);
      fixDists = fixDists.map(d => d - solvedBias);
    }
  }

  let intersectionPts = [];
  if (activeSats === 2) intersectionPts = circleIntersections(satList[0], fixDists[0], satList[1], fixDists[1]);

  let fixPoint = null, fixError = 0;
  if (activeSats >= 3) {
    fixPoint = findFix(satList, fixDists);
    if (fixPoint) fixError = dist(fixPoint, planePos);
    else if (!hasAnyError) { fixPoint = planePos; fixError = 0; }
  }

  const fixIsGood = fixPoint && fixError < 10;

  const handlePointerDown = useCallback(e => {
    const svg = svgRef.current; if (!svg) return;
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
    if (Math.abs(loc.x - planePos.x) < 40 && Math.abs(loc.y - planePos.y) < 30) { setDragging(true); e.preventDefault(); }
  }, [planePos]);
  const handlePointerMove = useCallback(e => {
    if (!dragging) return;
    const svg = svgRef.current; if (!svg) return;
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
    setPlanePos({ x: Math.max(40, Math.min(W - 40, loc.x)), y: Math.max(120, Math.min(GROUND_Y - 30, loc.y)) });
  }, [dragging]);
  const handlePointerUp = useCallback(() => setDragging(false), []);

  const fixColor = fixIsGood ? "#0f0" : "#ff8800";

  return (
    <div style={{ background: "#111", minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", padding: 16, fontFamily: "system-ui, sans-serif", color: "#eee" }}>
      <h1 style={{ margin: "0 0 2px", fontSize: 22, color: "#ffcc00" }}>📡 GPS {waasOn ? "+ WAAS" : ""} Explainer</h1>
      <p style={{ margin: "0 0 8px", fontSize: 12, color: "#aaa" }}>Drag the plane · Step through satellites · Add errors · Toggle WAAS</p>

      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", maxWidth: 800, borderRadius: 8, cursor: dragging ? "grabbing" : "default", touchAction: "none" }}
        onPointerDown={handlePointerDown} onPointerMove={handlePointerMove} onPointerUp={handlePointerUp} onPointerLeave={handlePointerUp}>
        <defs>
          <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#020810" /><stop offset="100%" stopColor="#0f2744" /></linearGradient>
          <linearGradient id="gnd" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#3a6b1e" /><stop offset="100%" stopColor="#1a3a08" /></linearGradient>
        </defs>
        <rect width={W} height={H} fill="url(#sky)" />
        {[...Array(40)].map((_, i) => <circle key={i} cx={(i * 137 + 50) % W} cy={(i * 89 + 20) % (GROUND_Y - 100)} r={0.8} fill="#fff" opacity={0.3 + (i % 3) * 0.2} />)}
        <rect x={0} y={GROUND_Y} width={W} height={H - GROUND_Y} fill="url(#gnd)" />
        <line x1={0} y1={GROUND_Y} x2={W} y2={GROUND_Y} stroke="#5a5" strokeWidth={2} />

        {/* True range circles when errors present */}
        {hasAnyError && activeSats > 0 && satList.map((s, i) => (
          <circle key={`true-${i}`} cx={s.x} cy={s.y} r={trueDists[i]} fill="none" stroke={CIRCLE_STROKES[i]} strokeWidth={1} strokeDasharray="2 4" opacity={0.2} />
        ))}

        {/* Active range circles (using fixDists) */}
        {satList.map((s, i) => (
          <circle key={`fix-${i}`} cx={s.x} cy={s.y} r={Math.max(0, fixDists[i])} fill="none" stroke={CIRCLE_STROKES[i]} strokeWidth={2} strokeDasharray="6 4" opacity={0.6} />
        ))}

        {/* Uncorrected circles shown faintly when WAAS is correcting */}
        {waasOn && hasAtmoErrors && satList.map((s, i) => (
          <circle key={`uncorr-${i}`} cx={s.x} cy={s.y} r={Math.max(0, measuredDists[i])} fill="none" stroke="#ff880044" strokeWidth={1.5} strokeDasharray="3 5" opacity={0.35} />
        ))}

        {activeSats === 1 && <circle cx={satList[0].x} cy={satList[0].y} r={Math.max(0, fixDists[0])} fill={CIRCLE_STROKES[0] + "22"} />}

        {/* Signal pulses from GPS sats to plane */}
        {satList.map((s, i) => <Pulse key={`sp-${i}`} from={s} to={planePos} color={CIRCLE_STROKES[i]} delay={i * 0.5} />)}

        {/* Distance lines */}
        {satList.map((s, i) => <line key={`dl-${i}`} x1={s.x} y1={s.y} x2={planePos.x} y2={planePos.y} stroke={CIRCLE_STROKES[i]} strokeWidth={1} strokeDasharray="4 3" opacity={0.2} />)}

        {/* WAAS infrastructure */}
        {waasOn && (
          <g>
            {/* GPS signals to ground station */}
            {satList.map((s, i) => <Pulse key={`gs-${i}`} from={s} to={groundStation} color={CIRCLE_STROKES[i]} delay={i * 0.5 + 0.2} dur={3} />)}
            {satList.map((s, i) => <line key={`gsl-${i}`} x1={s.x} y1={s.y} x2={groundStation.x} y2={groundStation.y - 20} stroke={CIRCLE_STROKES[i]} strokeWidth={0.5} strokeDasharray="2 4" opacity={0.15} />)}

            {/* Ground to uplink */}
            <line x1={groundStation.x} y1={groundStation.y - 34} x2={uplink.x} y2={uplink.y} stroke="#88f" strokeWidth={1} strokeDasharray="3 3" opacity={0.3} />
            <Pulse from={{ x: groundStation.x, y: groundStation.y - 34 }} to={uplink} color="#88f" delay={1.2} dur={3} />

            {/* Uplink to GEO */}
            <line x1={uplink.x} y1={uplink.y} x2={geoSat.x} y2={geoSat.y} stroke="#88f" strokeWidth={1} strokeDasharray="3 3" opacity={0.3} />
            <Pulse from={uplink} to={geoSat} color="#88f" delay={1.6} dur={3} />

            {/* GEO to plane */}
            <line x1={geoSat.x} y1={geoSat.y} x2={planePos.x} y2={planePos.y} stroke="#88f" strokeWidth={1} strokeDasharray="3 3" opacity={0.3} />
            <Pulse from={geoSat} to={planePos} color="#88f" delay={2.0} dur={3} />

            {/* Uplink station */}
            <g>
              <rect x={uplink.x - 8} y={uplink.y - 6} width={16} height={12} rx={2} fill="#336" stroke="#88f" strokeWidth={1} />
              <line x1={uplink.x} y1={uplink.y - 6} x2={uplink.x} y2={uplink.y - 16} stroke="#88f" strokeWidth={1.5} />
              <circle cx={uplink.x} cy={uplink.y - 16} r={3} fill="#88f" opacity={0.6} />
              <text x={uplink.x} y={uplink.y + 18} textAnchor="middle" fill="#88f" fontSize={8} fontWeight="bold">UPLINK</text>
            </g>

            <GroundStation x={groundStation.x} y={groundStation.y} active={true} />
            <Satellite x={geoSat.x} y={geoSat.y} label="WAAS GEO" color="#88f" active={true} geo={true} />
          </g>
        )}

        {/* 2-sat intersections */}
        {intersectionPts.map((p, i) => (
          <g key={`int-${i}`}>
            <circle cx={p.x} cy={p.y} r={8} fill="none" stroke="#fff" strokeWidth={2} />
            <circle cx={p.x} cy={p.y} r={3} fill="#fff" />
            <text x={p.x + 12} y={p.y + 4} fill="#fff" fontSize={10}>candidate {i + 1}</text>
          </g>
        ))}

        {/* Fix point */}
        {fixPoint && activeSats >= 3 && (
          <g>
            <circle cx={fixPoint.x} cy={fixPoint.y} r={12} fill="none" stroke={fixColor} strokeWidth={2.5}>
              <animate attributeName="r" values="10;16;10" dur="1.5s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="1;0.4;1" dur="1.5s" repeatCount="indefinite" />
            </circle>
            <circle cx={fixPoint.x} cy={fixPoint.y} r={4} fill={fixColor} />
            <text x={fixPoint.x + 16} y={fixPoint.y - 2} fill={fixColor} fontSize={11} fontWeight="bold">
              {fixIsGood ? "POSITION FIX ✓" : `FIX (err: ${fixError.toFixed(0)}px)`}
            </text>
          </g>
        )}

        {/* GPS Satellites */}
        {gpsSats.map((s, i) => <Satellite key={s.id} x={s.x} y={s.y} label={s.label} color={CIRCLE_STROKES[i]} active={i < activeSats} />)}

        {/* WAAS inactive indicators */}
        {!waasOn && (
          <g opacity={0.15}>
            <GroundStation x={groundStation.x} y={groundStation.y} active={false} />
            <Satellite x={geoSat.x} y={geoSat.y} label="GEO" color="#88f" active={false} geo={true} />
          </g>
        )}

        <Plane x={planePos.x} y={planePos.y} />
      </svg>

      {/* Step buttons */}
      <div style={{ display: "flex", gap: 6, marginTop: 10, flexWrap: "wrap", justifyContent: "center" }}>
        {STEPS.map((s, i) => (
          <button key={i} onClick={() => setStep(i)} style={{
            padding: "7px 12px", borderRadius: 6, fontSize: 12, cursor: "pointer",
            border: step === i ? "2px solid #ffcc00" : "2px solid #444",
            background: step === i ? "#ffcc0022" : "#222",
            color: step === i ? "#ffcc00" : "#aaa",
            fontWeight: step === i ? "bold" : "normal",
          }}>{s.n === 0 ? "No Sats" : `${s.n} Sat${s.n > 1 ? "s" : ""}`}</button>
        ))}

        <div style={{ width: 1, background: "#444", margin: "0 4px" }} />

        <button onClick={() => setWaasOn(v => !v)} style={{
          padding: "7px 16px", borderRadius: 6, fontSize: 12, cursor: "pointer", fontWeight: "bold",
          border: waasOn ? "2px solid #88f" : "2px solid #444",
          background: waasOn ? "#88f22" : "#222",
          color: waasOn ? "#88f" : "#666",
        }}>
          {waasOn ? "WAAS ON" : "WAAS OFF"}
        </button>
      </div>

      {/* Error sliders */}
      <div style={{ marginTop: 10, padding: "10px 16px", background: "#1a1a2e", borderRadius: 8, maxWidth: 700, width: "100%", display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{ fontSize: 13, color: "#aaa", fontWeight: "bold", marginBottom: 2 }}>Error Sources</div>

        {[
          { label: "⏱ Clock Error", val: clockErr, set: setClockErr, min: -120, max: 120, unit: "ns", desc: "Receiver quartz clock drift" },
          { label: "🌫 Ionospheric Delay", val: ionoErr, set: setIonoErr, min: 0, max: 100, unit: "ns", desc: "Signal slowed by charged particles" },
          { label: "🛰 Orbit Error", val: orbitErr, set: setOrbitErr, min: 0, max: 80, unit: "ns", desc: "Satellite position uncertainty" },
        ].map(({ label, val, set, min, max, unit, desc }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <div style={{ minWidth: 160, fontSize: 12 }}>
              <span>{label}</span>
              <span style={{ color: "#666", marginLeft: 6, fontSize: 10 }}>{desc}</span>
            </div>
            <input type="range" min={min} max={max} value={val} onChange={e => set(Number(e.target.value))} style={{ flex: 1, minWidth: 100 }} />
            <span style={{ fontFamily: "monospace", fontSize: 13, minWidth: 60, textAlign: "right", color: val === 0 ? "#0f0" : "#ff8800" }}>
              {val >= 0 ? "+" : ""}{val} {unit}
            </span>
          </div>
        ))}

        <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
          <button onClick={() => { setClockErr(0); setIonoErr(0); setOrbitErr(0); }} style={{
            padding: "4px 12px", borderRadius: 4, border: "1px solid #555", background: "#333", color: "#ccc", cursor: "pointer", fontSize: 11
          }}>Reset All Errors</button>
          <button onClick={() => { setClockErr(50); setIonoErr(40); setOrbitErr(30); }} style={{
            padding: "4px 12px", borderRadius: 4, border: "1px solid #555", background: "#333", color: "#ccc", cursor: "pointer", fontSize: 11
          }}>Add Realistic Errors</button>
        </div>
      </div>

      {/* Info panel */}
      <div style={{
        marginTop: 8, padding: "12px 16px", background: "#1a1a2e", borderRadius: 8,
        maxWidth: 700, width: "100%", borderLeft: `3px solid ${waasOn && hasAtmoErrors ? "#88f" : (hasAnyError && activeSats > 0 ? "#ff8800" : "#ffcc00")}`
      }}>
        <h3 style={{ margin: "0 0 4px", fontSize: 15, color: waasOn && hasAtmoErrors ? "#88f" : (hasAnyError && activeSats > 0 ? "#ff8800" : "#ffcc00") }}>{title}</h3>
        <p style={{ margin: 0, fontSize: 13, lineHeight: 1.5, color: "#ccc" }}>{desc}</p>

        {hasAtmoErrors && activeSats > 0 && !waasOn && (
          <p style={{ margin: "6px 0 0", fontSize: 12, color: "#ff8800", lineHeight: 1.4 }}>
            ⚠ Ionospheric delay and orbit errors distort each satellite's range differently. Even with 4 satellites, the receiver can only solve for clock bias — not these per-satellite errors. Try enabling WAAS!
          </p>
        )}

        {waasOn && activeSats > 0 && (
          <p style={{ margin: "6px 0 0", fontSize: 12, color: "#88f", lineHeight: 1.4 }}>
            📡 <strong>WAAS active:</strong> The ground station receives the same GPS signals but <em>knows its exact position</em>. It computes the error for each satellite (ionospheric delay + orbit error) and sends corrections via the GEO satellite. The plane applies these corrections to clean up its ranges.
            {hasClockErr && activeSats >= 4 && " The 4th satellite then solves for the remaining clock bias."}
            {hasAtmoErrors && <span style={{ color: "#aaa" }}> (Faint orange circles = uncorrected ranges)</span>}
          </p>
        )}

        {waasOn && !hasAtmoErrors && activeSats > 0 && (
          <p style={{ margin: "6px 0 0", fontSize: 12, color: "#88f", lineHeight: 1.4 }}>
            WAAS is active but there are no atmospheric/orbit errors to correct. Try adding ionospheric delay or orbit error above to see WAAS in action!
          </p>
        )}
      </div>

      {/* Show Math toggle */}
      <button onClick={() => setShowMath(v => !v)} style={{
        marginTop: 8, padding: "7px 18px", borderRadius: 6, cursor: "pointer", fontSize: 12,
        border: "1px solid #555", background: showMath ? "#2a2a4e" : "#222", color: "#aaa"
      }}>{showMath ? "Hide Math ▲" : "Show Math ▼"}</button>

      {showMath && (
        <div style={{
          marginTop: 6, padding: "14px 16px", background: "#0d0d1a", borderRadius: 8,
          maxWidth: 700, width: "100%", border: "1px solid #333", fontFamily: "monospace", fontSize: 12, lineHeight: 1.7
        }}>
          <div style={{ color: "#ffcc00", fontWeight: "bold", marginBottom: 6, fontFamily: "system-ui", fontSize: 13 }}>Per-Satellite Breakdown</div>
          <div style={{ color: "#888", fontSize: 11, marginBottom: 8 }}>
            measured_dist = c × (true_time + clock_err + iono_delay + orbit_err)
          </div>

          {activeSats === 0 && <div style={{ color: "#666" }}>Add satellites to see calculations...</div>}

          {satList.map((s, i) => {
            const td = trueDists[i], md = measuredDists[i], fd = fixDists[i];
            const satE = satErrors[i];
            return (
              <div key={s.id} style={{ marginBottom: 6, paddingBottom: 6, borderBottom: "1px solid #1a1a2a" }}>
                <div style={{ color: CIRCLE_STROKES[i], fontWeight: "bold" }}>{s.label}</div>
                <div style={{ color: "#aaa" }}>True: {td.toFixed(1)} px</div>
                <div style={{ color: "#ff8800" }}>Measured: {td.toFixed(1)} + ({clockErr} + {satE.toFixed(1)})×{C} = {md.toFixed(1)} px</div>
                {waasOn && hasAtmoErrors && (
                  <div style={{ color: "#88f" }}>WAAS correction: {gsCorrections[i].toFixed(1)} px → {(md + gsCorrections[i]).toFixed(1)} px</div>
                )}
                {activeSats === 4 && Math.abs(solvedBias) > 0.5 && (
                  <div style={{ color: "#0f0" }}>Clock correction: -{solvedBias.toFixed(1)} px → {fd.toFixed(1)} px</div>
                )}
              </div>
            );
          })}

          {waasOn && hasAtmoErrors && (
            <div style={{ marginTop: 6, padding: "6px 10px", background: "#111133", borderRadius: 4, fontFamily: "system-ui", fontSize: 11, color: "#88f" }}>
              Ground station knows its position → computes per-satellite atmospheric & orbit errors → broadcasts corrections via GEO → plane subtracts errors from its measurements
            </div>
          )}
        </div>
      )}
    </div>
  );
}
