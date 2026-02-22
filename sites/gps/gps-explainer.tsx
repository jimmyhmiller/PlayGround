import { useState, useEffect, useRef, useCallback } from "react";

const W = 820, H = 580, GY = 470;
const COL = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#f7dc6f"];
const FOCAL = 800;

// Scale: 1 px = 62.5 ft. Display area ≈ 51,000 ft wide (~8.4 NM)
const FT_PER_PX = 62.5;
const C_FT_S = 983571056; // speed of light ft/s
const C_FT_US = C_FT_S / 1e6; // ft per microsecond ≈ 983.57

function fmtFt(px) { return (px * FT_PER_PX).toFixed(0); }
function fmtUs(px) { return (px * FT_PER_PX / C_FT_US).toFixed(2); }

const gpsSats3D = [
  { id: 0, x: 100, y: 108, z: -40, label: "SV1" },
  { id: 1, x: 300, y: 92, z: 60, label: "SV2" },
  { id: 2, x: 500, y: 98, z: -30, label: "SV3" },
  { id: 3, x: 680, y: 112, z: 50, label: "SV4" },
];
const geoP = { x: 600, y: 38 };
const wrs3D = { x: 150, y: GY - 8, z: 20 };
const wulP = { x: 215, y: GY - 42 };

function l3(a, b) { return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + ((a.z || 0) - (b.z || 0)) ** 2); }

function proj(p) {
  const z = p.z || 0, s = FOCAL / (FOCAL + z), cx = W / 2, cy = H / 2;
  return { x: cx + (p.x - cx) * s, y: cy + (p.y - cy) * s, s };
}

function lsSolve(centers, prs) {
  const n = centers.length;
  if (n < 3) return null;
  const solveBias = n >= 4;
  const nU = solveBias ? 4 : 3;
  let x = 400, y = 300, z = 0, b = 0;
  for (let it = 0; it < 120; it++) {
    const Hr = [], rs = [];
    for (let i = 0; i < n; i++) {
      const dx = x - centers[i].x, dy = y - centers[i].y, dz = z - (centers[i].z || 0);
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
      if (solveBias) Hr.push([dx / r, dy / r, dz / r, 1]);
      else Hr.push([dx / r, dy / r, dz / r]);
      rs.push(prs[i] - (r + b));
    }
    const m = nU;
    const HtH = Array.from({ length: m }, () => new Float64Array(m));
    const Htr = new Float64Array(m);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      Htr[j] += Hr[i][j] * rs[i];
      for (let k = 0; k < m; k++) HtH[j][k] += Hr[i][j] * Hr[i][k];
    }
    for (let j = 0; j < m; j++) HtH[j][j] += 1e-8;
    const A = HtH.map((r, i) => [...r, Htr[i]]);
    for (let c = 0; c < m; c++) {
      let best = c;
      for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[best][c])) best = r;
      [A[c], A[best]] = [A[best], A[c]];
      if (Math.abs(A[c][c]) < 1e-14) continue;
      for (let r = 0; r < m; r++) { if (r === c) continue; const f = A[r][c] / A[c][c]; for (let k = c; k <= m; k++) A[r][k] -= f * A[c][k]; }
    }
    const d = []; for (let i = 0; i < m; i++) d.push(A[i][m] / (A[i][i] || 1e-12));
    x += d[0] || 0; y += d[1] || 0; z += d[2] || 0;
    if (solveBias) b += d[3] || 0;
    if (d.reduce((s, v) => s + v * v, 0) < 1e-8) break;
  }
  return { x, y, z, bias: b };
}

function Sat({ px, py, sc, label, color, active, isGeo }) {
  const g = isGeo;
  return (
    <g opacity={active ? 1 : 0.18} transform={`translate(${px},${py}) scale(${sc || 1})`}>
      {g ? <polygon points="0,-16 14,8 -14,8" fill="#226" stroke="#88f" strokeWidth={1.5} /> :
        <rect x={-18} y={-10} width={36} height={20} rx={3} fill="#334" stroke={active ? color : "#555"} strokeWidth={1.5} />}
      {[[-28, -18], [18, 28]].map(([a, b], j) => <line key={j} x1={g ? a + 8 : a} y1={0} x2={g ? b - 8 : b} y2={0} stroke={g ? "#88f" : (active ? color : "#555")} strokeWidth={2} />)}
      {!g && <><rect x={-30} y={-6} width={6} height={12} rx={1} fill={active ? color : "#555"} opacity={0.7} /><rect x={24} y={-6} width={6} height={12} rx={1} fill={active ? color : "#555"} opacity={0.7} /></>}
      {g && <><rect x={-22} y={-5} width={6} height={10} rx={1} fill="#88f" opacity={0.7} /><rect x={16} y={-5} width={6} height={10} rx={1} fill="#88f" opacity={0.7} /></>}
      <text x={0} y={28} textAnchor="middle" fill={g ? "#88f" : (active ? color : "#555")} fontSize={g ? 9 : 11} fontWeight="bold">{label}</text>
    </g>
  );
}

function PlaneGfx({ px, py, sc }) {
  return (
    <g transform={`translate(${px},${py}) scale(${sc || 1})`}>
      <ellipse cx={0} cy={0} rx={28} ry={8} fill="#e0e0e0" />
      <polygon points="-8,0 -22,-18 2,-2" fill="#e0e0e0" opacity={0.9} />
      <polygon points="15,0 28,-8 20,-1" fill="#e0e0e0" opacity={0.7} />
      <circle cx={22} cy={1} r={2} fill="#333" />
      <text x={0} y={24} textAnchor="middle" fill="#fff" fontSize={11} fontWeight="bold">✈ Aircraft</text>
    </g>
  );
}

function Pulse({ from, to, color, delay, dur = 2.5 }) {
  const [t, setT] = useState(0);
  useEffect(() => {
    let s = null, raf;
    const fn = ts => { if (!s) s = ts; const e = ((ts - s) / 1000 - delay) % dur; setT(e < 0 ? 0 : e / dur); raf = requestAnimationFrame(fn); };
    raf = requestAnimationFrame(fn); return () => cancelAnimationFrame(raf);
  }, [delay, dur]);
  if (t <= 0) return null;
  const px = from.x + (to.x - from.x) * t, py = from.y + (to.y - from.y) * t;
  const o = t < 0.8 ? 0.8 : (1 - t) * 4;
  return <circle cx={px} cy={py} r={4} fill={color} opacity={Math.max(0, o)} />;
}

function GndStn({ x, y, active }) {
  return (
    <g opacity={active ? 1 : 0.15}>
      <rect x={x - 14} y={y - 20} width={28} height={20} rx={2} fill="#555" stroke={active ? "#88f" : "#888"} strokeWidth={1} />
      <line x1={x} y1={y - 20} x2={x} y2={y - 34} stroke={active ? "#88f" : "#aaa"} strokeWidth={2} />
      <path d={`M${x - 8},${y - 38} Q${x},${y - 30} ${x + 8},${y - 38}`} fill="none" stroke={active ? "#88f" : "#aaa"} strokeWidth={2} />
      {active && <circle cx={x} cy={y - 10} r={3} fill="#0f0"><animate attributeName="opacity" values="1;0.3;1" dur="2s" repeatCount="indefinite" /></circle>}
      <text x={x} y={y + 14} textAnchor="middle" fill={active ? "#88f" : "#666"} fontSize={8} fontWeight="bold">WRS</text>
    </g>
  );
}

const STEPS = [
  { n: 0, title: "No Satellites", desc: "No signals — no position." },
  { n: 1, title: "1 SV — Pseudorange Sphere", desc: "One pseudorange defines a sphere of possible positions in 3D space." },
  { n: 2, title: "2 SVs — A Circle in 3D", desc: "Two spheres intersect in a circle." },
  { n: 3, title: "3 SVs — 3D Fix (perfect clock only)", desc: "Three pseudoranges → three equations → three unknowns (x, y, z). Gives a 3D fix, but only if the receiver clock is perfectly synced to GPS time. Any clock bias corrupts the solution." },
  { n: 4, title: "4 SVs — Full 3D Fix + Clock", desc: "Four pseudoranges → four equations → four unknowns (x, y, z, Δt). Over-determined with redundancy. Solved via iterative least squares — the same method real GPS receivers use." },
];

export default function App() {
  const [step, setStep] = useState(4);
  const [ac, setAc] = useState({ x: 380, y: 310, z: 0 });
  const [drag, setDrag] = useState(false);
  // Real-unit sliders
  const [clkUs, setClkUs] = useState(0);       // microseconds
  const [ionoFt, setIonoFt] = useState(0);     // feet (base, scaled per SV)
  const [ephFt, setEphFt] = useState(0);        // feet
  const [waas, setWaas] = useState(false);
  const [showMath, setShowMath] = useState(false);
  const svgRef = useRef(null);

  const { n: nSV, title, desc } = STEPS[step];
  const sats = gpsSats3D.slice(0, nSV);

  // Convert real units to px for internal math
  const clkB = clkUs * C_FT_US / FT_PER_PX;         // μs → ft → px
  const ionoBasePx = ionoFt / FT_PER_PX;              // ft → px
  const ephPx = ephFt / FT_PER_PX;                    // ft → px

  const ionoF = [1.0, 0.72, 1.28, 0.88];
  const ephD = [[0.9, 0.4, 0.3], [-0.6, 0.8, -0.5], [0.7, -0.7, 0.6], [-0.4, 1.0, -0.8]];
  const ionoPx = gpsSats3D.map((_, i) => ionoBasePx * ionoF[i]);
  const bcast = gpsSats3D.map((s, i) => ({ x: s.x + ephPx * ephD[i][0], y: s.y + ephPx * ephD[i][1], z: s.z + ephPx * ephD[i][2] }));

  const trueR = sats.map(s => l3(s, ac));
  const acPR = trueR.map((r, i) => r + ionoPx[i] + clkB);

  const wrsTR = gpsSats3D.map(s => l3(s, wrs3D));
  const wrsPR = wrsTR.map((r, i) => r + ionoPx[i]);
  const wrsExpR = gpsSats3D.map((_, i) => l3(bcast[i], wrs3D));
  const wrsCorr = gpsSats3D.map((_, i) => wrsExpR[i] - wrsPR[i]);
  const corrPR = acPR.map((pr, i) => pr + wrsCorr[i]);

  const hasIE = ionoFt !== 0 || ephFt !== 0;
  const hasClk = clkUs !== 0;
  const hasErr = hasClk || hasIE;

  const sPR = waas ? corrPR : acPR;
  const sC = waas ? sats.map(s => ({ x: s.x, y: s.y, z: s.z })) : bcast.slice(0, nSV);

  const sol = nSV >= 3 ? lsSolve(sC, sPR) : null;
  const fix = sol ? { x: sol.x, y: sol.y, z: sol.z } : null;
  const fixErr = fix ? l3(fix, ac) : 0;
  const fixErrFt = fixErr * FT_PER_PX;
  const fixGood = fix && fixErrFt < 50;

  const acP = proj(ac);
  const satPs = gpsSats3D.map(proj);
  const fixP = fix ? proj(fix) : null;
  const wrsP = proj(wrs3D);

  const refP = fix ? fixP : acP;
  const dispCenters = sats.map((_, i) => waas ? satPs[i] : proj(bcast[i]));
  const dispR = sats.map((_, i) => {
    const c = dispCenters[i];
    return Math.sqrt((c.x - refP.x) ** 2 + (c.y - refP.y) ** 2);
  });
  const trueDispR = sats.map((_, i) => Math.sqrt((satPs[i].x - acP.x) ** 2 + (satPs[i].y - acP.y) ** 2));

  let intPts = [];
  if (nSV === 2 && dispR[0] > 0 && dispR[1] > 0) {
    const c0 = dispCenters[0], c1 = dispCenters[1];
    const dx = c1.x - c0.x, dy = c1.y - c0.y, dd = Math.sqrt(dx * dx + dy * dy);
    if (dd > 0 && dd < dispR[0] + dispR[1] && dd > Math.abs(dispR[0] - dispR[1])) {
      const a = (dispR[0] * dispR[0] - dispR[1] * dispR[1] + dd * dd) / (2 * dd);
      const hSq = dispR[0] * dispR[0] - a * a;
      if (hSq >= 0) {
        const h = Math.sqrt(hSq), mx = c0.x + a * dx / dd, my = c0.y + a * dy / dd;
        intPts = [{ x: mx + h * dy / dd, y: my - h * dx / dd }, { x: mx - h * dy / dd, y: my + h * dx / dd }];
      }
    }
  }

  const toSVG = useCallback(e => {
    const svg = svgRef.current; if (!svg) return null;
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }, []);
  const pD = useCallback(e => {
    const l = toSVG(e); if (!l) return;
    if (Math.abs(l.x - acP.x) < 50 && Math.abs(l.y - acP.y) < 40) { setDrag(true); e.preventDefault(); }
  }, [acP, toSVG]);
  const pM = useCallback(e => {
    if (!drag) return;
    const l = toSVG(e); if (!l) return;
    const s = FOCAL / (FOCAL + ac.z), cx = W / 2, cy = H / 2;
    const wx = cx + (l.x - cx) / s, wy = cy + (l.y - cy) / s;
    setAc(p => ({ ...p, x: Math.max(40, Math.min(W - 40, wx)), y: Math.max(135, Math.min(GY - 30, wy)) }));
  }, [drag, ac.z, toSVG]);
  const pU = useCallback(() => setDrag(false), []);

  const fc = fixGood ? "#0f0" : "#ff8800";

  // Format fix error nicely
  const fmtFixErr = fixErrFt < 5280 ? `${fixErrFt.toFixed(0)} ft` : `${(fixErrFt / 6076).toFixed(2)} NM`;

  return (
    <div style={{ background: "#111", minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", padding: 16, fontFamily: "system-ui, sans-serif", color: "#eee" }}>
      <h1 style={{ margin: "0 0 2px", fontSize: 22, color: "#ffcc00" }}>📡 GPS{waas ? " + WAAS" : ""} Position Fix</h1>
      <p style={{ margin: "0 0 8px", fontSize: 12, color: "#aaa" }}>Drag aircraft · Depth slider for Z axis · Add errors · Toggle WAAS</p>

      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`}
        style={{ width: "100%", maxWidth: 820, borderRadius: 8, cursor: drag ? "grabbing" : "default", touchAction: "none" }}
        onPointerDown={pD} onPointerMove={pM} onPointerUp={pU} onPointerLeave={pU}>
        <defs>
          <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#020810" /><stop offset="100%" stopColor="#0f2744" /></linearGradient>
          <linearGradient id="gnd" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#3a6b1e" /><stop offset="100%" stopColor="#1a3a08" /></linearGradient>
        </defs>
        <rect width={W} height={H} fill="url(#sky)" />
        {[...Array(50)].map((_, i) => <circle key={i} cx={(i * 137 + 50) % W} cy={(i * 89 + 10) % (GY - 120)} r={0.7} fill="#fff" opacity={0.2 + (i % 4) * 0.12} />)}
        <rect x={0} y={GY} width={W} height={H - GY} fill="url(#gnd)" />
        <line x1={0} y1={GY} x2={W} y2={GY} stroke="#5a5" strokeWidth={2} />
        <text x={W - 10} y={86} textAnchor="end" fill="#333" fontSize={9}>← GPS orbit (~10,900 NM)</text>
        <text x={W - 10} y={34} textAnchor="end" fill="#222" fontSize={9}>← GEO orbit (~19,300 NM)</text>

        <ellipse cx={acP.x} cy={GY + 5} rx={18 * acP.s} ry={4 * acP.s} fill="#000" opacity={0.15 * acP.s} />

        {hasErr && sats.map((_, i) => (
          <circle key={`tr-${i}`} cx={satPs[i].x} cy={satPs[i].y} r={trueDispR[i]} fill="none" stroke={COL[i]} strokeWidth={0.7} strokeDasharray="2 5" opacity={0.12} />
        ))}

        {sats.map((_, i) => dispR[i] > 0 && (
          <circle key={`dr-${i}`} cx={dispCenters[i].x} cy={dispCenters[i].y} r={dispR[i]} fill="none" stroke={COL[i]} strokeWidth={2} strokeDasharray="6 4" opacity={0.45} />
        ))}

        {ephFt > 0 && !waas && sats.map((_, i) => {
          const bp = proj(bcast[i]);
          return (
            <g key={`ep-${i}`} opacity={0.4}>
              <line x1={satPs[i].x} y1={satPs[i].y} x2={bp.x} y2={bp.y} stroke={COL[i]} strokeWidth={1} strokeDasharray="2 2" />
              <circle cx={bp.x} cy={bp.y} r={3} fill={COL[i]} opacity={0.5} />
            </g>
          );
        })}

        {nSV === 1 && dispR[0] > 0 && <circle cx={dispCenters[0].x} cy={dispCenters[0].y} r={dispR[0]} fill={COL[0] + "12"} />}

        {sats.map((_, i) => <Pulse key={`p-${i}`} from={satPs[i]} to={acP} color={COL[i]} delay={i * 0.4} />)}
        {sats.map((_, i) => <line key={`l-${i}`} x1={satPs[i].x} y1={satPs[i].y} x2={acP.x} y2={acP.y} stroke={COL[i]} strokeWidth={0.6} strokeDasharray="3 4" opacity={0.1} />)}

        {waas && (
          <g>
            {sats.map((_, i) => <Pulse key={`wp-${i}`} from={satPs[i]} to={wrsP} color={COL[i]} delay={i * 0.4 + 0.15} dur={3.2} />)}
            <line x1={wrs3D.x} y1={wrs3D.y - 34} x2={wulP.x} y2={wulP.y} stroke="#88f" strokeWidth={0.8} strokeDasharray="3 3" opacity={0.3} />
            <Pulse from={{ x: wrs3D.x, y: wrs3D.y - 34 }} to={wulP} color="#88f" delay={1.0} dur={3.2} />
            <line x1={wulP.x} y1={wulP.y} x2={geoP.x} y2={geoP.y} stroke="#88f" strokeWidth={0.8} strokeDasharray="3 3" opacity={0.3} />
            <Pulse from={wulP} to={geoP} color="#88f" delay={1.5} dur={3.2} />
            <line x1={geoP.x} y1={geoP.y} x2={acP.x} y2={acP.y} stroke="#88f" strokeWidth={0.8} strokeDasharray="3 3" opacity={0.3} />
            <Pulse from={geoP} to={acP} color="#88f" delay={2.0} dur={3.2} />
            <rect x={wulP.x - 7} y={wulP.y - 5} width={14} height={10} rx={2} fill="#336" stroke="#88f" strokeWidth={0.8} />
            <line x1={wulP.x} y1={wulP.y - 5} x2={wulP.x} y2={wulP.y - 14} stroke="#88f" strokeWidth={1.2} />
            <circle cx={wulP.x} cy={wulP.y - 14} r={2.5} fill="#88f" opacity={0.6} />
            <text x={wulP.x} y={wulP.y + 16} textAnchor="middle" fill="#88f" fontSize={7} fontWeight="bold">UPLINK</text>
          </g>
        )}
        <GndStn x={wrs3D.x} y={wrs3D.y} active={waas} />
        <Sat px={geoP.x} py={geoP.y} sc={1} label="WAAS GEO" color="#88f" active={waas} isGeo={true} />

        {intPts.map((p, i) => (
          <g key={`i-${i}`}><circle cx={p.x} cy={p.y} r={7} fill="none" stroke="#fff" strokeWidth={1.5} /><circle cx={p.x} cy={p.y} r={2.5} fill="#fff" /></g>
        ))}

        {fixP && nSV >= 3 && (
          <g>
            <circle cx={fixP.x} cy={fixP.y} r={12 * fixP.s} fill="none" stroke={fc} strokeWidth={2.5}>
              <animate attributeName="r" values={`${10 * fixP.s};${16 * fixP.s};${10 * fixP.s}`} dur="1.5s" repeatCount="indefinite" />
              <animate attributeName="opacity" values="1;0.3;1" dur="1.5s" repeatCount="indefinite" />
            </circle>
            <circle cx={fixP.x} cy={fixP.y} r={4 * fixP.s} fill={fc} />
            <text x={fixP.x + 16} y={fixP.y + 4} fill={fc} fontSize={11} fontWeight="bold">
              {fixGood ? "FIX ✓" : `FIX (err: ${fmtFixErr})`}
            </text>
          </g>
        )}

        {gpsSats3D.map((s, i) => <Sat key={i} px={satPs[i].x} py={satPs[i].y} sc={satPs[i].s} label={s.label} color={COL[i]} active={i < nSV} />)}
        <PlaneGfx px={acP.x} py={acP.y} sc={acP.s} />
      </svg>

      {/* Depth control */}
      <div style={{ marginTop: 6, padding: "8px 14px", background: "#1a1a2e", borderRadius: 8, maxWidth: 720, width: "100%", display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <span style={{ fontSize: 12, color: "#888", fontWeight: "bold" }}>✈ Depth (Z)</span>
        <input type="range" min={-150} max={150} value={ac.z} onChange={e => setAc(p => ({ ...p, z: Number(e.target.value) }))} style={{ flex: 1, minWidth: 140 }} />
        <span style={{ fontFamily: "monospace", fontSize: 11, color: ac.z === 0 ? "#666" : "#4ecdc4", minWidth: 80 }}>
          {ac.z < 0 ? "◀ near" : ac.z > 0 ? "far ▶" : "center"} ({(ac.z * FT_PER_PX).toFixed(0)} ft)
        </span>
      </div>

      {/* Step buttons */}
      <div style={{ display: "flex", gap: 5, marginTop: 6, flexWrap: "wrap", justifyContent: "center" }}>
        {STEPS.map((s, i) => (
          <button key={i} onClick={() => setStep(i)} style={{
            padding: "6px 11px", borderRadius: 5, fontSize: 12, cursor: "pointer",
            border: step === i ? "2px solid #ffcc00" : "2px solid #444",
            background: step === i ? "#ffcc0020" : "#222", color: step === i ? "#ffcc00" : "#aaa",
            fontWeight: step === i ? "bold" : "normal",
          }}>{s.n} SV{s.n !== 1 ? "s" : ""}</button>
        ))}
        <div style={{ width: 1, background: "#444", margin: "0 2px" }} />
        <button onClick={() => setWaas(v => !v)} style={{
          padding: "6px 14px", borderRadius: 5, fontSize: 12, cursor: "pointer", fontWeight: "bold",
          border: waas ? "2px solid #88f" : "2px solid #444",
          background: waas ? "#222244" : "#222", color: waas ? "#88f" : "#666",
        }}>{waas ? "WAAS ON" : "WAAS OFF"}</button>
      </div>

      {/* Error sliders — real aviation units */}
      <div style={{ marginTop: 6, padding: "10px 14px", background: "#1a1a2e", borderRadius: 8, maxWidth: 720, width: "100%", display: "flex", flexDirection: "column", gap: 6 }}>
        <div style={{ fontSize: 12, color: "#666", fontWeight: "bold" }}>Error Sources</div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ minWidth: 150, fontSize: 11 }}><div>⏱ Clock Bias</div><div style={{ color: "#444", fontSize: 9 }}>Receiver quartz oscillator drift</div></div>
          <input type="range" min={-5} max={5} step={0.1} value={clkUs} onChange={e => setClkUs(Number(e.target.value))} style={{ flex: 1, minWidth: 80 }} />
          <span style={{ fontFamily: "monospace", fontSize: 12, minWidth: 70, textAlign: "right", color: clkUs === 0 ? "#0f0" : "#ff8800" }}>
            {clkUs >= 0 ? "+" : ""}{clkUs.toFixed(1)} μs
          </span>
          <span style={{ fontFamily: "monospace", fontSize: 10, color: "#555", minWidth: 60 }}>
            ({(clkUs * C_FT_US).toFixed(0)} ft)
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ minWidth: 150, fontSize: 11 }}><div>🌫 Iono Delay</div><div style={{ color: "#444", fontSize: 9 }}>Ionospheric signal delay (varies per SV)</div></div>
          <input type="range" min={0} max={150} step={1} value={ionoFt} onChange={e => setIonoFt(Number(e.target.value))} style={{ flex: 1, minWidth: 80 }} />
          <span style={{ fontFamily: "monospace", fontSize: 12, minWidth: 70, textAlign: "right", color: ionoFt === 0 ? "#0f0" : "#ff8800" }}>
            {ionoFt} ft
          </span>
          <span style={{ fontFamily: "monospace", fontSize: 10, color: "#555", minWidth: 60 }}>
            ({(ionoFt / C_FT_US).toFixed(3)} μs)
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ minWidth: 150, fontSize: 11 }}><div>📍 Ephemeris Error</div><div style={{ color: "#444", fontSize: 9 }}>SV broadcast position error</div></div>
          <input type="range" min={0} max={50} step={1} value={ephFt} onChange={e => setEphFt(Number(e.target.value))} style={{ flex: 1, minWidth: 80 }} />
          <span style={{ fontFamily: "monospace", fontSize: 12, minWidth: 70, textAlign: "right", color: ephFt === 0 ? "#0f0" : "#ff8800" }}>
            {ephFt} ft
          </span>
        </div>

        <div style={{ display: "flex", gap: 6, marginTop: 2 }}>
          <button onClick={() => { setClkUs(0); setIonoFt(0); setEphFt(0); }} style={{ padding: "3px 10px", borderRadius: 4, border: "1px solid #444", background: "#282828", color: "#888", cursor: "pointer", fontSize: 10 }}>Clear All</button>
          <button onClick={() => { setClkUs(1.0); setIonoFt(50); setEphFt(8); }} style={{ padding: "3px 10px", borderRadius: 4, border: "1px solid #444", background: "#282828", color: "#888", cursor: "pointer", fontSize: 10 }}>Typical Errors</button>
          <button onClick={() => { setClkUs(3.0); setIonoFt(130); setEphFt(30); }} style={{ padding: "3px 10px", borderRadius: 4, border: "1px solid #444", background: "#282828", color: "#888", cursor: "pointer", fontSize: 10 }}>Worst Case</button>
        </div>
      </div>

      {/* Info */}
      <div style={{ marginTop: 6, padding: "10px 14px", background: "#1a1a2e", borderRadius: 8, maxWidth: 720, width: "100%", borderLeft: `3px solid ${waas && hasIE ? "#88f" : hasErr && nSV > 0 ? "#ff8800" : "#ffcc00"}` }}>
        <h3 style={{ margin: "0 0 3px", fontSize: 14, color: waas && hasIE ? "#88f" : hasErr && nSV > 0 ? "#ff8800" : "#ffcc00" }}>{title}</h3>
        <p style={{ margin: 0, fontSize: 12, lineHeight: 1.5, color: "#bbb" }}>{desc}</p>
        {hasIE && nSV > 0 && !waas && (
          <p style={{ margin: "5px 0 0", fontSize: 11, color: "#ff8800" }}>⚠ Iono and ephemeris errors vary per SV — the solver absorbs them into the position estimate, corrupting the fix. Enable WAAS!</p>
        )}
        {waas && nSV > 0 && hasIE && (
          <p style={{ margin: "5px 0 0", fontSize: 11, color: "#88f" }}>📡 WAAS: WRS computes per-SV corrections. Iono delay cancels. Ephemeris residual is small (aircraft and WRS see SVs at nearly the same angle). Only clock bias remains → 4th SV solves it.</p>
        )}
      </div>

      {/* Math toggle */}
      <button onClick={() => setShowMath(v => !v)} style={{ marginTop: 6, padding: "6px 16px", borderRadius: 5, cursor: "pointer", fontSize: 12, border: "1px solid #444", background: showMath ? "#2a2a4e" : "#222", color: "#888" }}>
        {showMath ? "Hide Math ▲" : "Show Math ▼"}
      </button>

      {showMath && (
        <div style={{ marginTop: 5, padding: "14px", background: "#0a0a18", borderRadius: 8, maxWidth: 720, width: "100%", border: "1px solid #282828", fontSize: 12, lineHeight: 1.7, overflowX: "auto" }}>
          <div style={{ color: "#ffcc00", fontWeight: "bold", fontSize: 14, marginBottom: 8 }}>Aircraft: System of Equations</div>
          <div style={{ background: "#0e0e22", padding: "8px 10px", borderRadius: 4, marginBottom: 8, fontFamily: "monospace", fontSize: 11 }}>
            <div style={{ color: "#ddd", marginBottom: 4 }}>
              ρ<sub>i</sub> = √[(x−x<sub>i</sub>)² + (y−y<sub>i</sub>)² + (z−z<sub>i</sub>)²] + c·Δt
            </div>
            <div style={{ color: "#666", fontSize: 10 }}>
              Unknowns: <span style={{ color: "#4ecdc4" }}>x, y, z</span> (position in ft), <span style={{ color: "#4ecdc4" }}>Δt</span> (clock bias in μs)
            </div>
            <div style={{ color: "#666", fontSize: 10 }}>c = 983,571,056 ft/s &nbsp;|&nbsp; 1 μs of clock error ≈ 984 ft of range error</div>
          </div>

          {nSV > 0 && (
            <div style={{ background: "#0e0e22", padding: "8px 10px", borderRadius: 4, marginBottom: 8, fontFamily: "monospace", fontSize: 11 }}>
              <div style={{ color: "#888", marginBottom: 6 }}>{nSV} equation{nSV > 1 ? "s" : ""}{nSV >= 4 ? " → iterative least squares:" : ":"}</div>
              {sats.map((s, i) => {
                const c = sC[i];
                const prFt = sPR[i] * FT_PER_PX;
                return (
                  <div key={i} style={{ marginBottom: 4 }}>
                    <span style={{ color: COL[i] }}>{s.label}:</span>{" "}
                    <span style={{ color: "#fff" }}>{prFt.toFixed(0)} ft</span>
                    <span style={{ color: "#777" }}> = √[(<span style={{ color: "#4ecdc4" }}>x</span>−{(c.x * FT_PER_PX).toFixed(0)})² + (<span style={{ color: "#4ecdc4" }}>y</span>−{(c.y * FT_PER_PX).toFixed(0)})² + (<span style={{ color: "#4ecdc4" }}>z</span>−{((c.z || 0) * FT_PER_PX).toFixed(0)})²] + c·<span style={{ color: "#4ecdc4" }}>Δt</span></span>
                  </div>
                );
              })}
            </div>
          )}

          {sol && nSV >= 3 && (
            <div style={{ background: fixGood ? "#0a1a0a" : "#1a1208", padding: "8px 10px", borderRadius: 4, marginBottom: 8, fontFamily: "monospace", fontSize: 11 }}>
              <div style={{ color: fixGood ? "#0f0" : "#ff8800", fontWeight: "bold", marginBottom: 4 }}>Solution:</div>
              <div style={{ color: "#ccc" }}>
                <span style={{ color: "#4ecdc4" }}>x</span> = {(sol.x * FT_PER_PX).toFixed(0)} ft &nbsp;
                <span style={{ color: "#4ecdc4" }}>y</span> = {(sol.y * FT_PER_PX).toFixed(0)} ft &nbsp;
                <span style={{ color: "#4ecdc4" }}>z</span> = {(sol.z * FT_PER_PX).toFixed(0)} ft
              </div>
              <div style={{ color: "#ccc", marginTop: 2 }}>
                <span style={{ color: "#4ecdc4" }}>Δt</span> = {(sol.bias * FT_PER_PX / C_FT_US).toFixed(3)} μs
                <span style={{ color: "#555" }}> ({(sol.bias * FT_PER_PX).toFixed(0)} ft range equivalent)</span>
              </div>
              <div style={{ color: "#555", marginTop: 4, fontSize: 10 }}>
                True pos: ({(ac.x * FT_PER_PX).toFixed(0)}, {(ac.y * FT_PER_PX).toFixed(0)}, {(ac.z * FT_PER_PX).toFixed(0)}) ft | True Δt: {clkUs.toFixed(1)} μs | Position error: <span style={{ color: fixGood ? "#0f0" : "#ff8800" }}>{fmtFixErr}</span>
              </div>
            </div>
          )}

          {waas && nSV > 0 && (
            <>
              <div style={{ color: "#88f", fontWeight: "bold", fontSize: 14, marginTop: 12, marginBottom: 8, borderTop: "1px solid #282828", paddingTop: 10 }}>WRS: Correction Computation</div>
              <div style={{ background: "#0e0e28", padding: "8px 10px", borderRadius: 4, marginBottom: 8, fontFamily: "monospace", fontSize: 11 }}>
                <div style={{ color: "#aaa", marginBottom: 4 }}>WRS: surveyed position, atomic clock (Δt ≈ 0)</div>
                <div style={{ color: "#ddd" }}>correction<sub>i</sub> = dist(WRS, SV<sub>broadcast</sub>) − [dist(WRS, SV<sub>true</sub>) + I<sub>i</sub>]</div>
                <div style={{ color: "#666", fontSize: 10, marginTop: 2 }}>Captures iono delay + ephemeris geometric error in one correction per SV</div>
              </div>
              {sats.map((s, i) => (
                <div key={`wm-${i}`} style={{ marginBottom: 4, fontFamily: "monospace", fontSize: 11 }}>
                  <span style={{ color: COL[i], fontWeight: "bold" }}>{s.label}: </span>
                  <span style={{ color: "#aaa" }}>
                    R<sub>exp</sub>={fmtFt(wrsExpR[i])} ft &nbsp;
                    ρ<sub>wrs</sub>={fmtFt(wrsPR[i])} ft &nbsp;
                  </span>
                  <span style={{ color: "#88f", fontWeight: "bold" }}>
                    corr = {wrsCorr[i] >= 0 ? "+" : ""}{fmtFt(wrsCorr[i])} ft
                  </span>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  );
}
