// Differential comparison of two observation records (reference vs diffpack).
//
// Every difference is reported. Nothing is truncated away: `findings` carries
// the full list, and callers write it to disk verbatim; only the console
// rendering shows a head plus an explicit "+N more" count.

const NUMERIC = /^-?\d+(\.\d+)?px$/;
const TOLERANCE_PX = 0.75;

const numeric = (value) => (NUMERIC.test(value) ? Number.parseFloat(value) : null);

/** Style values are equal when identical, or numerically within a sub-pixel tolerance. */
export const styleValuesMatch = (a, b) => {
  if (a === b) return true;
  const na = numeric(a);
  const nb = numeric(b);
  if (na !== null && nb !== null) return Math.abs(na - nb) <= TOLERANCE_PX;
  return false;
};

const finding = (channel, severity, summary, detail) => ({ channel, severity, summary, detail });

/**
 * Whether a finding counts against the app.
 *
 * Three severities exist. `fail` is an observed difference. `info` is a
 * difference in diffpack's FAVOUR (an error only the reference produces) and is
 * recorded, not charged. `error` means the comparison could not be made at all —
 * today only "probe missing", i.e. one side never produced a record for the
 * route.
 *
 * `error` used to be charged to nobody: the scoreboard counted `fail` alone, so
 * an app whose routes diffpack could not serve scored a clean `pass` with the
 * unmeasurable routes filed as a footnote. An unmeasurable route is the LOUDEST
 * thing the suite can see, not the quietest — it is scored as a failure.
 */
export const isFailure = (f) => f.severity === "fail" || f.severity === "error";

/**
 * Masks values an app legitimately renders differently on every request.
 *
 * The determinism shim (probe.mjs) seeds `Math.random` and freezes `Date` in the
 * BROWSER, which cannot reach anything the server rendered — an app that prints
 * the server's clock differs between two deployments no matter what the bundlers
 * do. Such patterns must be declared per app in `corpus.json` as `volatile`,
 * each with a `volatileNote` saying why: an undeclared difference is always a
 * finding, and a declared one is auditable.
 */
const maskVolatile = (text, patterns) => {
  let out = text ?? "";
  for (const pattern of patterns ?? []) out = out.replace(new RegExp(pattern, "g"), "█");
  return out;
};

export const compareRecords = (ref, dp, { label, volatile: volatilePatterns = [] }) => {
  const findings = [];
  const add = (...args) => findings.push(finding(...args));

  if (!ref || !dp) {
    add("probe", "error", `${label}: probe missing on ${!ref ? "reference" : "diffpack"}`, { ref: Boolean(ref), dp: Boolean(dp) });
    return findings;
  }

  if (ref.title !== dp.title) {
    add("text", "fail", `${label}: <title> differs`, { reference: ref.title, diffpack: dp.title });
  }
  if (ref.lang !== dp.lang) {
    add("text", "fail", `${label}: <html lang> differs`, { reference: ref.lang, diffpack: dp.lang });
  }
  const refBody = maskVolatile(ref.bodyText, volatilePatterns);
  const dpBody = maskVolatile(dp.bodyText, volatilePatterns);
  if (refBody !== dpBody) {
    add("text", "fail", `${label}: rendered body text differs`, {
      reference: ref.bodyText,
      diffpack: dp.bodyText,
      firstDivergenceAt: firstDivergence(refBody, dpBody),
    });
  }
  if (ref.headings.join("\n") !== dp.headings.join("\n")) {
    add("text", "fail", `${label}: heading outline differs`, { reference: ref.headings, diffpack: dp.headings });
  }

  const sameStructure = ref.tagOutline === dp.tagOutline;
  if (!sameStructure) {
    add("structure", "fail", `${label}: DOM structure differs (${ref.elementCount} vs ${dp.elementCount} elements)`, {
      referenceOutline: ref.tagOutline,
      diffpackOutline: dp.tagOutline,
      firstDivergenceAt: firstDivergence(ref.tagOutline, dp.tagOutline),
    });
  }

  if (sameStructure) {
    const styleDiffs = [];
    const attrDiffs = [];
    const boxDiffs = [];
    for (let i = 0; i < ref.elements.length; i++) {
      const a = ref.elements[i];
      const b = dp.elements[i];
      if (!b) break;
      if (maskVolatile(a.text, volatilePatterns) !== maskVolatile(b.text, volatilePatterns)) {
        attrDiffs.push({ index: i, tag: a.tag, property: "text", reference: a.text, diffpack: b.text });
      }
      for (const [key, value] of Object.entries(a.attrs)) {
        if (b.attrs[key] !== value) {
          attrDiffs.push({ index: i, tag: a.tag, property: `@${key}`, reference: value, diffpack: b.attrs[key] ?? null });
        }
      }
      for (const key of Object.keys(b.attrs)) {
        if (!(key in a.attrs)) {
          attrDiffs.push({ index: i, tag: a.tag, property: `@${key}`, reference: null, diffpack: b.attrs[key] });
        }
      }
      for (const [prop, value] of Object.entries(a.styles)) {
        if (!styleValuesMatch(value, b.styles[prop])) {
          styleDiffs.push({ index: i, tag: a.tag, property: prop, reference: value, diffpack: b.styles[prop] });
        }
      }
      if (Math.abs(a.box.w - b.box.w) > 1 || Math.abs(a.box.h - b.box.h) > 1) {
        boxDiffs.push({ index: i, tag: a.tag, reference: a.box, diffpack: b.box });
      }
      if (a.image && b.image && a.image.loaded !== b.image.loaded) {
        // Direction matters. diffpack failing to load an image the reference
        // loads is a defect; the reverse means the reference's own pipeline
        // could not serve it here (Next's runtime /_next/image optimizer needs
        // a live server and 404s in some setups) and says nothing about
        // diffpack.
        const diffpackWorse = a.image.loaded && !b.image.loaded;
        add(
          "assets",
          diffpackWorse ? "fail" : "info",
          `${label}: <img alt="${a.image.alt}"> loaded=${a.image.loaded} on reference, ${b.image.loaded} on diffpack`,
          { index: i, reference: a, diffpack: b }
        );
      }
      if (a.field && b.field && JSON.stringify(a.field) !== JSON.stringify(b.field)) {
        attrDiffs.push({ index: i, tag: a.tag, property: "field", reference: a.field, diffpack: b.field });
      }
    }
    if (attrDiffs.length) {
      add("attributes", "fail", `${label}: ${attrDiffs.length} element attribute/text difference(s)`, attrDiffs);
    }
    if (styleDiffs.length) {
      add("styles", "fail", `${label}: ${styleDiffs.length} computed-style difference(s) across ${new Set(styleDiffs.map((d) => d.index)).size} element(s)`, styleDiffs);
    }
    if (boxDiffs.length) {
      add("layout", "fail", `${label}: ${boxDiffs.length} element(s) laid out at a different size`, boxDiffs);
    }
  }

  const refLinks = ref.links.filter((l) => l.internal).map((l) => l.path).sort();
  const dpLinks = dp.links.filter((l) => l.internal).map((l) => l.path).sort();
  if (refLinks.join("|") !== dpLinks.join("|")) {
    add("links", "fail", `${label}: internal link targets differ`, { reference: refLinks, diffpack: dpLinks });
  }

  if (ref.stylesheetCount > 0 && dp.stylesheetCount === 0) {
    add("assets", "fail", `${label}: reference ships ${ref.stylesheetCount} stylesheet(s), diffpack ships none`, {
      reference: ref.stylesheetCount,
      diffpack: dp.stylesheetCount,
    });
  }

  if (ref.clickable.length !== dp.clickable.length) {
    add("interaction", "fail", `${label}: ${ref.clickable.length} interactive element(s) on reference, ${dp.clickable.length} on diffpack`, {
      reference: ref.clickable,
      diffpack: dp.clickable,
    });
  }

  if (ref.hydrationHints.hasReactFiber && !dp.hydrationHints.hasReactFiber) {
    add("hydration", "fail", `${label}: React hydrated on the reference build but not on the diffpack build`, {
      reference: ref.hydrationHints,
      diffpack: dp.hydrationHints,
    });
  }

  return findings;
};

export const firstDivergence = (a = "", b = "") => {
  const limit = Math.min(a.length, b.length);
  for (let i = 0; i < limit; i++) {
    if (a[i] !== b[i]) {
      return { index: i, reference: a.slice(Math.max(0, i - 40), i + 60), diffpack: b.slice(Math.max(0, i - 40), i + 60) };
    }
  }
  if (a.length !== b.length) {
    return { index: limit, reference: a.slice(limit, limit + 80), diffpack: b.slice(limit, limit + 80) };
  }
  return null;
};

const NOISE = [
  /Download the React DevTools/i,
  /React DevTools/i,
  /\[Fast Refresh\]/i,
  /favicon\.ico/i,
];

const isNoise = (text) => NOISE.some((re) => re.test(text ?? ""));

/**
 * Errors are compared as sets: an error class present on the diffpack side and
 * absent on the reference side is a diffpack defect. The converse is recorded
 * as informational (the reference misbehaving is not diffpack's problem).
 */
export const compareErrors = (refObs, dpObs, { label }) => {
  const findings = [];
  const classify = (obs) => ({
    console: (obs.console ?? []).filter((m) => m.type === "error" && !isNoise(m.text)).map((m) => m.text),
    pageErrors: (obs.pageErrors ?? []).filter((e) => !isNoise(e.text)).map((e) => (e.text ?? "").split("\n")[0]),
    failedRequests: (obs.network ?? [])
      .filter((r) => (r.failed || (r.status && r.status >= 400)) && !isNoise(r.url))
      .map((r) => `${r.status ?? "FAILED"} ${r.url}`),
  });
  const a = classify(refObs);
  const b = classify(dpObs);
  for (const key of ["console", "pageErrors", "failedRequests"]) {
    const refSet = new Set(a[key].map(normalizeErrorText));
    const extra = b[key].filter((text) => !refSet.has(normalizeErrorText(text)));
    if (extra.length) {
      findings.push(
        finding("errors", "fail", `${label}: ${extra.length} ${key} present on diffpack but not on the reference`, {
          diffpackOnly: extra,
          reference: a[key],
          diffpack: b[key],
        })
      );
    }
    const missing = a[key].filter((text) => !new Set(b[key].map(normalizeErrorText)).has(normalizeErrorText(text)));
    if (missing.length) {
      findings.push(
        finding("errors", "info", `${label}: ${missing.length} ${key} on the reference that diffpack does not produce`, {
          referenceOnly: missing,
        })
      );
    }
  }
  return findings;
};

const normalizeErrorText = (text) =>
  (text ?? "")
    .replace(/https?:\/\/127\.0\.0\.1:\d+/g, "ORIGIN")
    .replace(/https?:\/\/localhost:\d+/g, "ORIGIN")
    .replace(/:\d+:\d+/g, ":L:C")
    .replace(/[0-9a-f]{8,}/gi, "HASH")
    .trim();
export const maskVolatileText = (text, patterns) => {
  let out = text ?? "";
  for (const pattern of patterns ?? []) out = out.replace(new RegExp(pattern, "g"), "█");
  return out;
};
