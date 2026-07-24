/* bat runs viewer — plain DOM, no build step. */

const $runs = document.getElementById("runs");
const $main = document.getElementById("main");
let current = null;

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

function when(iso) {
  const d = new Date(iso.replace(/-(\d{2})-(\d{3})Z$/, ":$1.$2Z"));
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleTimeString();
}

async function loadRuns() {
  const groups = await (await fetch("/api/runs")).json();
  $runs.replaceChildren(
    ...groups.map((g) =>
      el(
        "div",
        { class: "flow-group" },
        el("h2", {}, g.runs[0]?.flowName ?? g.slug),
        g.runs.map((r) =>
          el(
            "button",
            {
              class: "run-item",
              onclick: (e) => {
                document.querySelectorAll(".run-item.active").forEach((n) => n.classList.remove("active"));
                e.currentTarget.classList.add("active");
                openRun(g.slug, r.id);
              },
            },
            el("span", { class: `dot ${r.status}` }),
            el("span", {}, r.failedStep ? `failed @ step ${r.failedStep}` : "pass"),
            el("span", { class: "when" }, when(r.startedAt)),
          ),
        ),
      ),
    ),
  );
}

async function openRun(slug, id) {
  const { trace, report, reruns } = await (await fetch(`/api/trace?flow=${slug}&run=${id}`)).json();
  current = { slug, id, trace, reruns: reruns ?? [] };

  const header = el(
    "div",
    { class: "run-header" },
    el("h1", {}, trace.flow, el("span", { class: `badge ${trace.status}` }, trace.status.toUpperCase())),
    el(
      "div",
      { class: "meta" },
      [
        trace.file,
        trace.worldFingerprint ? `world ${trace.worldFingerprint}` : null,
        trace.worldVerification ? `L${trace.worldVerification.level}` : null,
        trace.conditions ? `conditions: seed ${trace.conditions.seed}` : null,
      ]
        .filter(Boolean)
        .join("  ·  "),
    ),
  );

  const steps = trace.steps.map((s) => renderStep(s, trace));

  const explanation = trace.explanation
    ? el(
        "div",
        { class: "explanation" },
        el("h3", {}, "why this failed"),
        el(
          "pre",
          {},
          [
            ...trace.explanation.failed,
            ...trace.explanation.whatHappened,
            ...trace.explanation.reproducibility.map((l, i) => (i === 0 ? `reproducibility: ${l}` : l)),
            ...(trace.explanation.meaning.length ? ["what this means:", ...trace.explanation.meaning.map((m) => `  ${m}`)] : []),
          ].join("\n"),
        ),
      )
    : null;

  const rawReport = el(
    "details",
    { class: "sub" },
    el("summary", {}, "full text report"),
    el("pre", { class: "reportpre" }, report),
  );

  const diff = trace.status === "fail" && current.reruns.length ? renderDiffPicker(trace) : null;

  $main.replaceChildren(header, explanation ?? "", diff ?? "", ...steps, rawReport);
}

/* ------- run-vs-rerun diff (the reruns bat took while explaining) ------- */

function renderDiffPicker(trace) {
  const wrap = el("div", { class: "explanation diff-wrap" });
  const passing = current.reruns.filter((r) => r.status === "pass");
  const failing = current.reruns.filter((r) => r.status !== "pass");
  const label = el(
    "h3",
    {},
    `compare with a rerun `,
    el("span", { class: "dim" }, `(bat reran this flow ${current.reruns.length}× while explaining — ${passing.length} reached the expected state)`),
  );
  const out = el("div");
  const buttons = current.reruns.map((r, i) =>
    el(
      "button",
      {
        class: "replay",
        onclick: async () => {
          const rerun = await (await fetch(`/api/rerun?flow=${current.slug}&run=${current.id}&name=${r.name}`)).json();
          out.replaceChildren(renderDiff(trace, rerun, r));
        },
      },
      el("span", { class: `dot ${r.status}` }),
      ` rerun ${i + 1} (${r.status})`,
    ),
  );
  wrap.append(label, el("div", { class: "diff-buttons" }, ...buttons), out);
  return wrap;
}

function stepSummary(s) {
  if (!s || s.status === "not-run") return { order: "—", effects: [], duration: null };
  const order = (s.requests ?? [])
    .filter((r) => r.finishSeq != null && r.resourceType !== "document")
    .sort((a, b) => a.finishSeq - b.finishSeq)
    .map((r) => `${r.method} ${new URL(r.url).pathname}`)
    .join(" → ");
  return { order: order || "(no api traffic)", effects: s.effects ?? [], duration: s.durationMs };
}

function renderDiff(trace, rerun, meta) {
  const failIdx = trace.steps.findIndex((s) => s.status === "fail");
  const a = stepSummary(trace.steps[failIdx]);
  const b = stepSummary(rerun.steps[failIdx]);
  const orderDiffers = a.order !== b.order;

  const col = (title, status, sum) =>
    el(
      "div",
      { class: "diff-col" },
      el("h4", {}, title, el("span", { class: `badge ${status}` }, status.toUpperCase())),
      el("div", { class: "section-label" }, "response completion order"),
      el("div", { class: `order ${orderDiffers ? "differs" : ""}` }, sum.order),
      el("div", { class: "section-label" }, "effects"),
      ...sum.effects.map((e) =>
        el("div", { class: `effect ${e.pass ? "pass" : "fail"}` }, e.rendered, e.observed && !e.pass ? el("span", { class: "observed" }, `observed: ${e.observed}`) : null),
      ),
      sum.duration != null ? el("div", { class: "section-label" }, `step duration: ${sum.duration}ms`) : null,
    );

  return el(
    "div",
    {},
    el(
      "div",
      { class: "section-label" },
      `step ${failIdx + 1} — ${trace.steps[failIdx]?.source ?? ""}` + (orderDiffers ? "  ·  completion order DIFFERS between the runs" : "  ·  completion order identical — the difference is elsewhere"),
    ),
    el("div", { class: "diff-grid" }, col("this run", trace.status, a), col(`rerun (${meta.status})`, rerun.status, b)),
  );
}

function renderStep(s, trace) {
  const mark = s.status === "pass" ? "✓" : s.status === "fail" ? "✗" : "·";
  const body = [];

  if (s.status !== "not-run") {
    body.push(
      ...s.effects.map((e) =>
        el("div", { class: `effect ${e.pass ? "pass" : "fail"}` }, e.rendered, e.observed && !e.pass ? el("span", { class: "observed" }, `observed: ${e.observed}`) : null),
      ),
    );

    if (s.requests?.length) {
      body.push(el("div", { class: "section-label" }, "network (start order)"));
      body.push(
        el(
          "table",
          { class: "net" },
          el("tr", {}, ...["method", "path", "status", "finished #", "notes"].map((h) => el("th", {}, h))),
          ...s.requests.map((r) =>
            el(
              "tr",
              {},
              el("td", {}, r.method),
              el("td", {}, new URL(r.url).pathname),
              el("td", {}, r.failure ? `FAILED (${r.failure})` : (r.status ?? "pending")),
              el("td", {}, r.finishSeq ?? "—"),
              el("td", {}, [r.injected, r.streaming ? "streaming" : null].filter(Boolean).join(", ")),
            ),
          ),
        ),
      );
    }

    if (s.consoleErrors?.length) {
      body.push(el("div", { class: "section-label" }, "page errors"));
      body.push(...s.consoleErrors.map((c) => el("div", { class: "effect fail" }, `[${c.kind}] ${c.text.slice(0, 300)}`)));
    }

    if (s.screenshot) {
      body.push(el("div", { class: "section-label" }, "screenshot"));
      body.push(el("img", { class: "shot", src: `/api/artifact?flow=${current.slug}&run=${current.id}&name=${s.screenshot}`, alt: `step ${s.index + 1} screenshot` }));
    }

    if (s.ariaSnapshot) {
      body.push(
        el("details", { class: "sub" }, el("summary", {}, "semantic tree at failure"), el("pre", { class: "snapshot" }, s.ariaSnapshot)),
      );
    }

    if (s.status === "fail") {
      const btn = el("button", { class: "replay" }, `replay this step (headed)`);
      const out = el("div", { class: "replay-out" });
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        btn.textContent = "replaying…";
        try {
          const res = await (
            await fetch("/api/replay", {
              method: "POST",
              headers: { "content-type": "application/json" },
              body: JSON.stringify({ file: trace.file, step: s.index + 1, headed: true }),
            })
          ).json();
          out.replaceChildren(
            el("div", { class: "section-label" }, `replay: ${res.status ?? "error"} (${res.tier ?? ""})`),
            el("pre", { class: "reportpre" }, res.report ?? res.error ?? ""),
          );
        } finally {
          btn.disabled = false;
          btn.textContent = "replay this step (headed)";
        }
      });
      body.push(el("div", { class: "section-label" }, "replay"), btn, out);
    }
  }

  const details = el(
    "details",
    { class: `step ${s.status}`, ...(s.status === "fail" ? { open: "" } : {}) },
    el(
      "summary",
      {},
      el("span", { class: "mark" }, mark),
      el("span", {}, `step ${s.index + 1}`),
      el("span", {}, s.source),
      el("span", { class: "duration" }, s.durationMs ? `${s.durationMs}ms` : ""),
    ),
    el("div", { class: "step-body" }, ...body),
  );
  return details;
}

loadRuns();
