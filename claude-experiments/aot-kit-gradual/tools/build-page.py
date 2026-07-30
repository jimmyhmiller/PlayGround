#!/usr/bin/env python3
"""Assemble out/ir-gallery.html from the rendered Graphviz SVGs.

Run tools/render-dot.sh first. Inlining the SVGs (rather than linking them) is required:
the artifact host blocks every external request, so a linked asset would silently vanish.

Design plan this file implements:
  Color   ground #EEF0F3 / #14161A, surface #FFFFFF / #1C1F25, ink #16181D / #E8EAEE,
          muted #5B6270 / #969DAA, accent #B26B00 / #E0A44A, rule #D8DCE2 / #2A2E36.
          The accent is lifted straight out of the diagrams: it is the stroke colour of a
          control node. Diagram wells stay light in BOTH themes, because Graphviz draws in
          dark ink and a white-on-dark inversion would wreck legibility; that is a decision,
          not an oversight, and the page says so.
  Type    Display and labels in mono, heavy and tightened, because every identifier in the
          subject is already mono and a compiler tool has no business in a fashion sans.
          Body in the system sans for reading. Eyebrows uppercase, letterspaced.
  Layout  One column of specimen panels. Before/after pairs are a two-column grid because
          they genuinely are a pair; it collapses on narrow screens. A legend up front
          teaches the visual grammar, which is real information design, not decoration.
"""
import re, pathlib, html

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOT = ROOT / "out" / "dot"
OUT = ROOT / "out" / "ir-gallery.html"


def svg(name):
    """Inline one SVG, stripped of its prolog and its fixed pixel size."""
    raw = (DOT / f"{name}.svg").read_text()
    raw = raw[raw.index("<svg"):]
    raw = re.sub(r'<svg width="[^"]*" height="[^"]*"', "<svg", raw, count=1)
    return raw.strip()


def stats(name):
    """live / created / leaked, as reported by the compiler itself."""
    first = (DOT / f"{name}.dot").read_text().splitlines()[0]
    m = re.match(r"// stats live=(\d+) created=(\d+) leaked=(\d+)", first)
    if not m:
        raise SystemExit(f"{name}.dot has no stats header; re-run tools/render-dot.sh")
    return {k: int(v) for k, v in zip(("live", "created", "leaked"), m.groups())}


def chips(name):
    s = stats(name)
    return (
        f'<span class="chip"><b>{s["live"]}</b> live</span>'
        f'<span class="chip"><b>{s["created"]}</b> created</span>'
        f'<span class="chip{" chip-ok" if s["leaked"] == 0 else " chip-bad"}">'
        f'<b>{s["leaked"]}</b> leaked</span>'
    )


def pair(before, after, title, source, lead, look):
    return f"""
<section class="specimen">
  <header class="spec-head">
    <h2>{title}</h2>
    <pre class="source"><code>{html.escape(source)}</code></pre>
  </header>
  <p class="lead">{lead}</p>
  <div class="compare">
    <figure>
      <figcaption><span class="eyebrow">before</span> {chips(before)}</figcaption>
      <div class="well">{svg(before)}</div>
    </figure>
    <figure>
      <figcaption><span class="eyebrow">after</span> {chips(after)}</figcaption>
      <div class="well">{svg(after)}</div>
    </figure>
  </div>
  <p class="note"><span class="note-key">Look for</span>{look}</p>
</section>"""


def single(name, title, source, lead, look):
    return f"""
<section class="specimen">
  <header class="spec-head">
    <h2>{title}</h2>
    <pre class="source"><code>{html.escape(source)}</code></pre>
  </header>
  <p class="lead">{lead}</p>
  <div class="compare compare-one">
    <figure>
      <figcaption><span class="eyebrow">analysed and optimised</span> {chips(name)}</figcaption>
      <div class="well">{svg(name)}</div>
    </figure>
  </div>
  <p class="note"><span class="note-key">Look for</span>{look}</p>
</section>"""


CSS = """
<style>
:root {
  --ground:#EEF0F3; --surface:#FFFFFF; --ink:#16181D; --muted:#5B6270;
  --accent:#B26B00; --accent-soft:#FFF3DF; --rule:#D8DCE2; --well:#FDFDFE;
  --ok:#1F7A4D; --bad:#B3261E;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",Arial,sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ground:#14161A; --surface:#1C1F25; --ink:#E8EAEE; --muted:#969DAA;
    --accent:#E0A44A; --accent-soft:#2C2416; --rule:#2A2E36; --well:#E9EBEF;
    --ok:#5FCB92; --bad:#F2837B;
  }
}
:root[data-theme="dark"] {
  --ground:#14161A; --surface:#1C1F25; --ink:#E8EAEE; --muted:#969DAA;
  --accent:#E0A44A; --accent-soft:#2C2416; --rule:#2A2E36; --well:#E9EBEF;
  --ok:#5FCB92; --bad:#F2837B;
}
:root[data-theme="light"] {
  --ground:#EEF0F3; --surface:#FFFFFF; --ink:#16181D; --muted:#5B6270;
  --accent:#B26B00; --accent-soft:#FFF3DF; --rule:#D8DCE2; --well:#FDFDFE;
  --ok:#1F7A4D; --bad:#B3261E;
}

body { background:var(--ground); color:var(--ink); font-family:var(--sans);
       line-height:1.6; margin:0; padding:clamp(1.25rem,4vw,3.5rem) 1.25rem 5rem; }
.page { max-width:60rem; margin:0 auto; display:flex; flex-direction:column; gap:2.5rem; }

.eyebrow { font-family:var(--mono); font-size:.68rem; font-weight:700;
           letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }

header.masthead { display:flex; flex-direction:column; gap:.75rem;
                  border-bottom:2px solid var(--ink); padding-bottom:1.5rem; }
h1 { font-family:var(--mono); font-size:clamp(1.6rem,4.5vw,2.4rem); font-weight:700;
     letter-spacing:-.03em; line-height:1.1; margin:0; text-wrap:balance; }
.standfirst { max-width:62ch; margin:0; color:var(--muted); font-size:1.02rem; }
.standfirst strong { color:var(--ink); font-weight:600; }

.panel { background:var(--surface); border:1px solid var(--rule); border-radius:4px;
         padding:1.5rem; }

.legend { display:grid; grid-template-columns:repeat(auto-fit,minmax(15rem,1fr)); gap:1.25rem 2rem; }
.legend h2 { grid-column:1/-1; margin:0; font-family:var(--mono); font-size:.9rem;
             font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
.legend dl { margin:0; display:flex; flex-direction:column; gap:.6rem; }
.legend div { display:grid; grid-template-columns:5.5rem 1fr; gap:.75rem; align-items:baseline; }
.legend dt { font-family:var(--mono); font-size:.78rem; color:var(--accent); font-weight:700; }
.legend dd { margin:0; font-size:.9rem; color:var(--muted); }
.legend dd b { color:var(--ink); font-weight:600; }

.specimen { display:flex; flex-direction:column; gap:1rem; }
.spec-head { display:flex; flex-wrap:wrap; align-items:baseline; gap:.5rem 1rem;
             border-bottom:1px solid var(--rule); padding-bottom:.6rem; }
.specimen h2 { margin:0; font-size:1.15rem; font-weight:650; letter-spacing:-.01em; }
.source { margin:0; font-family:var(--mono); font-size:.82rem; color:var(--accent);
          background:var(--accent-soft); border-radius:3px; padding:.15rem .5rem;
          overflow-x:auto; max-width:100%; }
.lead { margin:0; max-width:70ch; color:var(--muted); font-size:.95rem; }

.compare { display:grid; grid-template-columns:1fr 1fr; gap:1rem; align-items:start; }
.compare-one { grid-template-columns:1fr; }
@media (max-width:46rem) { .compare { grid-template-columns:1fr; } }
.compare figure { margin:0; display:flex; flex-direction:column; gap:.5rem; }
figcaption { display:flex; flex-wrap:wrap; align-items:center; gap:.4rem; }

.chip { font-family:var(--mono); font-size:.68rem; color:var(--muted);
        border:1px solid var(--rule); border-radius:99px; padding:.05rem .45rem;
        font-variant-numeric:tabular-nums; }
.chip b { color:var(--ink); font-weight:700; }
.chip-ok b { color:var(--ok); } .chip-bad b { color:var(--bad); }

/* The wells stay light in both themes on purpose: Graphviz draws in dark ink, and
   inverting the diagrams would cost more legibility than the theme match is worth. */
.well { background:var(--well); border:1px solid var(--rule); border-radius:4px;
        padding:1rem; overflow-x:auto; display:flex; justify-content:center; }
.well svg { max-width:100%; height:auto; }

.note { margin:0; font-size:.92rem; max-width:74ch;
        border-left:3px solid var(--accent); padding:.1rem 0 .1rem .85rem; }
.note-key { font-family:var(--mono); font-size:.68rem; font-weight:700; letter-spacing:.12em;
            text-transform:uppercase; color:var(--accent); display:block; }

.finding { display:flex; flex-direction:column; gap:.75rem; }
.finding h2 { margin:0; font-family:var(--mono); font-size:.9rem; font-weight:700;
              letter-spacing:.06em; text-transform:uppercase; }
.finding p { margin:0; max-width:72ch; font-size:.95rem; color:var(--muted); }
.finding p strong, .finding code { color:var(--ink); }
code { font-family:var(--mono); font-size:.88em; }
footer { border-top:1px solid var(--rule); padding-top:1rem; color:var(--muted);
         font-family:var(--mono); font-size:.72rem; }
</style>
"""

BODY = f"""
<title>Reading the graph: aot-kit-gradual M1</title>
{CSS}
<div class="page">

<header class="masthead">
  <span class="eyebrow">aot-kit-gradual &middot; milestone M1 &middot; node engine</span>
  <h1>Reading the graph</h1>
  <p class="standfirst">The IR is a sea of nodes, so there is no line-by-line listing to read.
  These are the compiler's own Graphviz dumps, before and after peepholes, from
  <code>src/dot.coil</code>. Every node is labelled with <strong>what the lattice concluded
  about it</strong>, because that is the question this compiler is mostly about.</p>
</header>

<div class="panel legend">
  <h2>Visual grammar</h2>
  <dl>
    <div><dt>box</dt><dd>a <b>control</b> node: Start, Return, Stop. These are the spine.</dd></div>
    <div><dt>ellipse</dt><dd>a <b>data</b> node. Blue is a constant, grey an opaque argument.</dd></div>
    <div><dt>second line</dt><dd>the node's <b>computed type</b>, e.g. <code>int=[2..8]</code>.</dd></div>
  </dl>
  <dl>
    <div><dt>red edge</dt><dd><b>control</b> flow. Blue dashed will be memory, from M4.</dd></div>
    <div><dt>dotted</dt><dd>an <b>anchor</b>: a literal is rooted at Start, but control does
      not flow into it. Drawn faint so it stops burying the real spine.</dd></div>
    <div><dt>edge label</dt><dd>the <b>input index</b>, so operand order is visible.
      <code>a-b</code> is not <code>b-a</code>.</dd></div>
  </dl>
  <dl>
    <div><dt>direction</dt><dd>edges run use &rarr; def, so <b>definitions sit above their
      uses</b>. Read a graph from the bottom.</dd></div>
    <div><dt>live / created</dt><dd>nodes surviving, and nodes ever allocated. The gap is what
      the optimiser threw away.</dd></div>
  </dl>
</div>

{pair("01-fold-before", "02-fold-after",
      "Everything folds", "return 1 + 2 * 3;",
      "The whole expression is constant, so nothing should survive but a literal. Before is "
      "the graph as built, with types computed but no rewriting applied.",
      "The <code>Mul</code> already carries <code>int=6</code> and the <code>Add</code> "
      "<code>int=7</code> before any rewriting happens. Constant folding is not a special "
      "case here: it is the lattice reporting a single-value type, and the peephole simply "
      "believing it.")}

{pair("03-gvn-before", "04-gvn-after",
      "Two adds are one add", "return (arg + 2) - (2 + arg);",
      "Three separate mechanisms have to fire in sequence for this to reach zero, and if any "
      "one of them is missing the expression survives intact.",
      "Before, there are two <code>Add</code> nodes and two <code>int=2</code> constants. "
      "Operand order is canonicalised first, which makes the two adds structurally identical; "
      "value numbering then collapses them to one node; only then does <code>x - x</code> "
      "apply. Note the surviving graph has no arithmetic at all.")}

{pair("05-big-before", "06-big-after",
      "Nine expressions, one multiply", "-(-((arg₁+0)*1 + 3 - (3 + (arg₁+0)*1) + arg₂*2))",
      "The M1 test fixture, built to make several rewrites interact: identity elimination, "
      "canonicalisation, value numbering, self-subtraction, and double negation.",
      "The result is <code>arg₂ * 2</code> with the correct range <code>int=[2..8]</code>, "
      "and <code>arg₁</code> is gone entirely. 20 nodes were allocated and 6 remain, with "
      "nothing leaked: every discarded node is genuinely dead, not merely unreachable.")}

{single("07-ranges",
        "What the lattice knows", "return (a * b + 5) + (dyn - 1);",
        "No constants to fold and no identities to apply, so this specimen is about the "
        "analysis rather than the rewriting. <code>a</code> is <code>int=[0..10]</code>, "
        "<code>b</code> is <code>int=[2..3]</code>, and the third argument is fully dynamic.",
        "Ranges compose: <code>[0..10] * [2..3]</code> gives <code>[0..30]</code>, then "
        "<code>+5</code> gives <code>[5..35]</code>. The dynamic side is where the JavaScript "
        "semantics show: <code>dyn - 1</code> is <code>num</code>, because subtraction coerces "
        "and NaN is a float, and once <code>num</code> meets the integer range the result is "
        "<code>num</code> too. Had that been <code>dyn + 1</code> it would be <code>dyn</code>, "
        "since <code>+</code> can concatenate strings.")}

<div class="panel finding">
  <h2>What the pictures showed that the tests did not</h2>
  <p>In <em>Nine expressions</em>, look at the before graph: the <code>Sub</code> of two
  <code>int=[3..103]</code> operands is typed <code>int=[-100..100]</code>, not zero. Interval
  arithmetic cannot see that both operands are <strong>the same value</strong>, only that they
  occupy the same range. That is why <code>x - x</code> has to exist as a rewrite rather than
  falling out of the analysis, and it is a useful reminder that the lattice and the peepholes
  are answering different questions.</p>
  <p>It also shows why <code>x - x</code> is withheld from floats. The rule is sound for
  integers and wrong for numbers in general, because <code>NaN - NaN</code> is
  <code>NaN</code>. Same for <code>x == x</code> and <code>x * 0</code>. There is a test
  asserting all three stay withheld, and the graphs are how you check the test is testing what
  you think.</p>
</div>

<footer>
  Generated by <code>tools/dot-dump.coil</code> &rarr; <code>tools/render-dot.sh</code> &rarr;
  <code>tools/build-page.py</code>. Node and edge counts are reported by the compiler, not
  counted from the output.
</footer>

</div>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(BODY)
print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} bytes)")
