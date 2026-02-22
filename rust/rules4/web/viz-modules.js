// viz-modules.js — Pluggable layout modules for Learnable Programming
//
// Each module: { name, label, analyze(r4, maxStep, tracedFns) }
// Modules read trace events and write dynamic rules for the view layer.

// ── Shared helpers ──

export function assertNum(r4, name, val) {
  r4.assertRule(r4.sym(name), r4.num(val));
}

export function assertCallNum(r4, name, argVal, rhs) {
  r4.assertRule(r4.call(r4.sym(name), [r4.num(argVal)]), r4.num(rhs));
}

export function assertCall2Num(r4, name, arg1, arg2, rhs) {
  r4.assertRule(r4.call(r4.sym(name), [r4.num(arg1), r4.num(arg2)]), r4.num(rhs));
}

export function assertCallTerm(r4, name, argVal, rhs) {
  r4.assertRule(r4.call(r4.sym(name), [r4.num(argVal)]), rhs);
}

export function assertCall2Term(r4, name, arg1, arg2, rhs) {
  r4.assertRule(r4.call(r4.sym(name), [r4.num(arg1), r4.num(arg2)]), rhs);
}

export function assertCallStr(r4, name, argVal, str) {
  r4.assertRule(r4.call(r4.sym(name), [r4.num(argVal)]), r4.sym(str));
}

export function assertCall2Str(r4, name, arg1, arg2, str) {
  r4.assertRule(r4.call(r4.sym(name), [r4.num(arg1), r4.num(arg2)]), r4.sym(str));
}

/**
 * Read a trace event and return a structured object.
 * Returns null if event is not a recognized trace entry.
 */
export function readTraceEvent(r4, step) {
  const event = r4.eval(r4.call(r4.sym("trace"), [r4.num(step)]));
  const eventHead = r4.termCallHead(event);
  if (r4.termTag(eventHead) !== 1) return null;
  const eventType = r4.termSymName(eventHead);

  if (eventType === 'reduced') {
    const oldTerm = r4.termCallArg(event, 0);
    const newTerm = r4.termCallArg(event, 1);
    const kind = r4.termCallArg(event, 2);
    const fnName = r4.termSymName(r4.termCallArg(kind, 0));
    return { type: 'reduced', step, oldTerm, newTerm, kind, fnName, event };
  } else if (eventType === 'completed') {
    const callTerm = r4.termCallArg(event, 0);  // quote(call)
    const valueTerm = r4.termCallArg(event, 1);
    const callInner = r4.termCallArg(callTerm, 0);
    const callHead = r4.termCallHead(callInner);
    let fnName = null;
    if (r4.termTag(callHead) === 1) {
      fnName = r4.termSymName(callHead);
    }
    return { type: 'completed', step, callTerm, callInner, valueTerm, fnName, event };
  }

  return null;
}

// ── Call Tree Module ──

export const callTreeModule = {
  name: 'call_tree',
  label: 'Call Tree',

  analyze(r4, maxStep, tracedFns) {
    const stack = [];
    const allNodes = [];
    const allEdges = [];

    for (let i = 0; i <= maxStep; i++) {
      const ev = readTraceEvent(r4, i);
      if (!ev) continue;

      if (ev.type === 'reduced') {
        tracedFns.add(ev.fnName);

        const node = {
          id: allNodes.length,
          callTermId: ev.oldTerm,
          entryStep: i,
          completionStep: i,
          children: [],
          isRoot: stack.length === 0,
        };

        if (stack.length > 0) {
          const parent = stack[stack.length - 1];
          parent.children.push(node);
          allEdges.push({ from: parent.id, to: node.id });
        }

        allNodes.push(node);
        stack.push(node);

      } else if (ev.type === 'completed') {
        if (ev.fnName && tracedFns.has(ev.fnName) && stack.length > 0) {
          const done = stack.pop();
          done.completionStep = i;
        }
      }
    }

    // Layout tree
    if (allNodes.length > 0) {
      const roots = allNodes.filter(n => n.isRoot);
      let leafIdx = 0;
      function layoutX(node, depth) {
        node.depth = depth;
        if (!node.children.length) { node.x = leafIdx++; return; }
        for (const child of node.children) layoutX(child, depth + 1);
        node.x = (node.children[0].x + node.children[node.children.length - 1].x) / 2;
      }
      for (let ri = 0; ri < roots.length; ri++) {
        if (ri > 0) leafIdx += 1;
        layoutX(roots[ri], 0);
      }

      const HSPC = 54, R = 16, PAD = 36, VSPC = 52;
      const maxDepth = allNodes.reduce((m, nd) => Math.max(m, nd.depth), 0);
      const svgW = Math.max(leafIdx * HSPC + PAD * 2, 120);
      const svgH = (maxDepth + 1) * VSPC + PAD * 2;

      const nodeR = allNodes.length <= 6 ? 10 : allNodes.length <= 12 ? 14 : 16;

      assertNum(r4, "node_count", allNodes.length);
      assertNum(r4, "edge_count", allEdges.length);
      assertNum(r4, "svg_width", svgW);
      assertNum(r4, "svg_height", svgH);
      assertNum(r4, "node_r", nodeR);

      for (const nd of allNodes) {
        const cx = Math.round(PAD + nd.x * HSPC + HSPC / 2);
        const cy = Math.round(PAD + nd.depth * VSPC + R + 4);
        assertCallNum(r4, "node_cx", nd.id, cx);
        assertCallNum(r4, "node_cy", nd.id, cy);
        assertCallNum(r4, "node_entry", nd.id, nd.entryStep);
        assertCallNum(r4, "node_completion", nd.id, nd.completionStep);
        assertCallTerm(r4, "node_call", nd.id, nd.callTermId);
      }

      for (let i = 0; i < allEdges.length; i++) {
        assertCallNum(r4, "edge_from", i, allEdges[i].from);
        assertCallNum(r4, "edge_to", i, allEdges[i].to);
      }
    } else {
      assertNum(r4, "node_count", 0);
      assertNum(r4, "edge_count", 0);
      assertNum(r4, "svg_width", 100);
      assertNum(r4, "svg_height", 60);
      assertNum(r4, "node_r", 10);
    }
  }
};

// ── Step Card Module (view-only, no analyze phase needed) ──

export const stepCardModule = {
  name: 'step_card',
  label: 'Step Card',
  analyze(r4, maxStep, tracedFns) {
    // Step card is purely reactive view code — no layout rules needed.
  }
};

// ── Value Timeline Module ──

export const valueTimelineModule = {
  name: 'value_timeline',
  label: 'Value Timeline',

  analyze(r4, maxStep, tracedFns) {
    // Collect completed events grouped by function name
    const seriesMap = new Map(); // fnName -> [{step, value, valueTerm, callInner}]

    for (let i = 0; i <= maxStep; i++) {
      const ev = readTraceEvent(r4, i);
      if (!ev || ev.type !== 'completed') continue;
      if (!ev.fnName || !tracedFns.has(ev.fnName)) continue;

      if (!seriesMap.has(ev.fnName)) seriesMap.set(ev.fnName, []);

      // Try to get numeric value
      let numVal = null;
      if (r4.termTag(ev.valueTerm) === 0) {
        numVal = r4.termNum(ev.valueTerm);
      } else if (r4.termTag(ev.valueTerm) === 3) {
        numVal = r4.termFloat(ev.valueTerm);
      }

      seriesMap.get(ev.fnName).push({
        step: i,
        numVal,
        label: r4.display(ev.valueTerm),
        callLabel: r4.display(ev.callInner),
      });
    }

    const seriesNames = [...seriesMap.keys()];
    const seriesCount = seriesNames.length;

    if (seriesCount === 0) {
      assertNum(r4, "vt_series_count", 0);
      assertNum(r4, "vt_svg_width", 100);
      assertNum(r4, "vt_svg_height", 60);
      return;
    }

    // Compute layout
    const MARGIN_L = 40, MARGIN_R = 20, MARGIN_T = 30, MARGIN_B = 20;
    const CHART_W = 500, ROW_H = 60;
    const svgW = MARGIN_L + CHART_W + MARGIN_R;
    const svgH = MARGIN_T + seriesCount * ROW_H + MARGIN_B;

    assertNum(r4, "vt_series_count", seriesCount);
    assertNum(r4, "vt_svg_width", svgW);
    assertNum(r4, "vt_svg_height", svgH);

    for (let s = 0; s < seriesCount; s++) {
      const name = seriesNames[s];
      const points = seriesMap.get(name);
      const baseY = MARGIN_T + s * ROW_H + ROW_H / 2;

      assertCallStr(r4, "vt_series_name", s, name);
      assertCallNum(r4, "vt_series_y", s, Math.round(baseY));
      assertCallNum(r4, "vt_point_count", s, points.length);

      // Find value range for y positioning within the row
      const numVals = points.filter(p => p.numVal !== null).map(p => p.numVal);
      const hasNumeric = numVals.length > 0;
      const minVal = hasNumeric ? Math.min(...numVals) : 0;
      const maxVal = hasNumeric ? Math.max(...numVals) : 0;
      const valRange = maxVal - minVal || 1;

      for (let j = 0; j < points.length; j++) {
        const pt = points[j];
        // x = proportional to step
        const x = Math.round(MARGIN_L + (maxStep > 0 ? (pt.step / maxStep) * CHART_W : CHART_W / 2));
        // y = within the row band (±15px from center)
        let y;
        if (hasNumeric && pt.numVal !== null) {
          y = Math.round(baseY - ((pt.numVal - minVal) / valRange - 0.5) * 24);
        } else {
          y = Math.round(baseY);
        }

        assertCall2Num(r4, "vt_point_x", s, j, x);
        assertCall2Num(r4, "vt_point_y", s, j, y);
        assertCall2Num(r4, "vt_point_step", s, j, pt.step);
        assertCall2Str(r4, "vt_point_label", s, j, pt.label);
        assertCall2Str(r4, "vt_point_call", s, j, pt.callLabel);
      }
    }
  }
};

// ── Iteration Table Module ──

export const iterationTableModule = {
  name: 'iteration_table',
  label: 'Iteration Table',

  analyze(r4, maxStep, tracedFns) {
    // Find completed events for "leaf" functions (not the recursive wrapper).
    // Heuristic: a function is a "leaf" if it never appears as a recursive caller
    // in the trace. E.g., for map(double, [1,2,3]), "double" completions are leaves.
    //
    // Alternative simpler heuristic: collect all completed events, identify functions
    // that have completions where the call has arity 1 (single-arg functions applied
    // to each element). Group them as table rows.

    const completions = []; // [{step, fnName, callInner, valueTerm, args}]

    for (let i = 0; i <= maxStep; i++) {
      const ev = readTraceEvent(r4, i);
      if (!ev || ev.type !== 'completed') continue;
      if (!ev.fnName || !tracedFns.has(ev.fnName)) continue;

      const arity = r4.termCallArity(ev.callInner);
      const args = [];
      for (let a = 0; a < arity; a++) {
        args.push(r4.display(r4.termCallArg(ev.callInner, a)));
      }

      completions.push({
        step: i,
        fnName: ev.fnName,
        callInner: ev.callInner,
        valueTerm: ev.valueTerm,
        valueLabel: r4.display(ev.valueTerm),
        callLabel: r4.display(ev.callInner),
        args,
        arity,
      });
    }

    if (completions.length === 0) {
      assertNum(r4, "it_row_count", 0);
      assertNum(r4, "it_col_count", 0);
      return;
    }

    // Find "element" functions: functions with the most completions
    // that aren't the recursive wrapper. Use heuristic: shortest arity.
    const fnCounts = new Map();
    for (const c of completions) {
      fnCounts.set(c.fnName, (fnCounts.get(c.fnName) || 0) + 1);
    }

    // Pick the function with lowest arity that has multiple completions,
    // or the one with highest count among low-arity ones.
    let bestFn = null;
    let bestScore = -1;
    for (const [fn, count] of fnCounts) {
      if (count >= 2) {
        const sample = completions.find(c => c.fnName === fn);
        // Prefer arity-1 functions, then by count
        const score = (sample.arity <= 1 ? 10000 : 0) + count;
        if (score > bestScore) {
          bestScore = score;
          bestFn = fn;
        }
      }
    }

    if (!bestFn) {
      // Fallback: use all completions as rows
      bestFn = completions[0].fnName;
    }

    const rows = completions.filter(c => c.fnName === bestFn);

    // Columns: Input(s), Output
    const sampleArity = rows[0].arity;
    const headers = [];
    for (let a = 0; a < sampleArity; a++) {
      headers.push(sampleArity === 1 ? "Input" : `Arg ${a + 1}`);
    }
    headers.push("Output");

    const colCount = headers.length;
    assertNum(r4, "it_row_count", rows.length);
    assertNum(r4, "it_col_count", colCount);
    assertCallStr(r4, "it_fn_name", 0, bestFn);

    for (let c = 0; c < colCount; c++) {
      assertCallStr(r4, "it_header", c, headers[c]);
    }

    for (let r = 0; r < rows.length; r++) {
      const row = rows[r];
      for (let a = 0; a < sampleArity; a++) {
        assertCall2Str(r4, "it_cell", r, a, row.args[a]);
      }
      assertCall2Str(r4, "it_cell", r, sampleArity, row.valueLabel);
      assertCallNum(r4, "it_row_step", r, row.step);
    }
  }
};

// ── All modules ──

export const allModules = [
  callTreeModule,
  stepCardModule,
  valueTimelineModule,
  iterationTableModule,
];
