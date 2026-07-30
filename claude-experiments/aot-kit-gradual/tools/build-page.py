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
  <span class="eyebrow">aot-kit-gradual &middot; milestones M1&ndash;M2 &middot; node engine and control flow</span>
  <h1>Reading the graph</h1>
  <p class="standfirst">The IR is a sea of nodes, so there is no line-by-line listing to read.
  These are the compiler's own Graphviz dumps, before and after peepholes, from
  <code>src/dot.coil</code>. Every node is labelled with <strong>what the lattice concluded
  about it</strong>, because that is the question this compiler is mostly about. The last
  specimen is the one the whole design is aimed at: a guard, with an unboxed fast path and a
  generic fallback, both real code in the same binary.</p>
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
    <div><dt>dashed red</dt><dd>a loop <b>back edge</b>, drawn without a ranking constraint so
      it does not turn the layout inside out.</dd></div>
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

{pair("09-dead-branch-before", "10-dead-branch-after",
      "A branch that is not a branch", "if (0) return 1; else return 2;",
      "Reachability is not a separate pass here. An <code>If</code>'s type is a tuple of its two "
      "control outputs, so an untaken branch is simply <code>~ctrl</code> in one slot and "
      "everything downstream follows from ordinary type propagation.",
      "In the before graph the <code>If</code> is already typed <code>[~ctrl ctrl]</code> and the "
      "true arm is <code>~ctrl</code>: the analysis has decided, but nothing has been rewritten. "
      "After, the <code>If</code>, both projections and the unreachable <code>Return</code> are "
      "all gone, and <code>Stop</code> has one input instead of two. Note that JavaScript "
      "truthiness is doing the deciding, not \"is it a non-zero integer\": "
      "<code>undefined</code>, <code>NaN</code> and <code>-0.0</code> all take the false arm, and "
      "every object takes the true one.")}

{single("15-nested-dead-branch",
        "A dead branch inside a live one",
        "if (p) return 1; else { if (0) return 2; else return 3; }",
        "The same <code>if (0)</code> as above, moved one level down so that its control input is "
        "an ordinary projection instead of <code>Start</code>. That difference is the whole "
        "specimen: it looks like a redundant variation and it is the only version that catches "
        "the bug.",
        "The outer branch survives because <code>p</code> is opaque; the inner one is gone "
        "entirely, and the arm that returns 3 is wired straight to the outer false projection. "
        "Getting there needs an <code>If</code> to stay alive while <em>both</em> of its "
        "projections are built: folding the first arm to <code>~ctrl</code> drops the "
        "<code>If</code>'s last use, so the kill cascade takes the <code>If</code> with it and "
        "the second arm is then rewritten against a corpse whose inputs have been cleared. In the "
        "flat version above that read landed on <code>Start</code>, which is pinned, so the wrong "
        "answer happened to be the right node and every test stayed green. Here it landed on a "
        "node that had just died, and the surviving <code>Return</code> read dead control: the "
        "verifier says <code>VERR-DEAD-INPUT</code> and running it with <code>p = 0</code> gets "
        "stuck instead of returning 3.")}

{single("11-diamond",
        "Merging with a phi", "if (arg) x = 1; else x = a + 2; return x;",
        "When a branch does produce a value, the merge needs a <code>Phi</code>: one input per "
        "path, positionally matched to the <code>Region</code>'s control inputs.",
        "The phi's type is the meet over the live arms, so it is narrower than the declared "
        "<code>dyn</code> it was built with. That positional match between region paths and phi "
        "inputs is a single invariant rather than two, which is why removing a dead path removes "
        "the matching phi input in the same operation. Letting those diverge for even one "
        "peephole gives a phi that reads the wrong arm, which is a miscompile that typechecks.")}

{single("12-loop",
        "A loop, and why it terminates", "i = 0; while (true) i = i + 1;",
        "The dashed edge is the back edge. The interesting thing is not the shape but the "
        "<code>Phi</code>'s type, because that is where the analysis could fail to terminate.",
        "The phi settles at <code>int</code>. It passes through <code>int=0</code>, then "
        "<code>int=[0..1]</code>, then wider ranges, and a widening counter in the type forces it "
        "to the axis bottom after a few steps instead of letting it climb for ever. Without that "
        "counter this graph does not converge at all. The type is checked to be identical under "
        "twelve worklist seeds, because inferred types that depend on visit order are a latent "
        "miscompile.")}

{single("14-nested-loops",
        "Nesting, and why it is recorded", "while (a) { while (b) { } }",
        "Dominators and loop nesting have no consumer yet. Global code motion will need them to "
        "place a node in the shallowest block where it is still legal, and to weigh how often a "
        "block runs. They are built now because they are small, and because a dominator table "
        "computed against a stale graph is a scheduling bug that only surfaces under "
        "optimisation.",
        "The <code>loop N</code> annotations are nesting depth. The inner header, its test and "
        "its body read <code>loop 2</code>; the inner <em>exit</em> reads <code>loop 1</code>, "
        "because leaving the inner loop does not leave the outer one; past the outer exit there "
        "is no annotation at all. Note also that a loop header's immediate dominator is its "
        "entry and never its back edge. Changing any control edge invalidates both tables, and "
        "reading a stale one is a hard error rather than a quietly wrong answer.")}

{single("13-guard",
        "A guard, in full", "if (x < 100) { fast: (int)x + 1 } else { slow: x + 1 }",
        "This is the shape DECISIONS.md D4 describes and the reason for the whole design. There "
        "is no deoptimisation machinery anywhere in it: a guard is an ordinary branch, and the "
        "fallback is ordinary code compiled into the same binary.",
        "Follow the two arms. On the true arm a <code>Cast to int</code> carries the narrowed "
        "type in, and the add above it is <code>Add int</code>: an unboxed integer add. On the "
        "false arm the same source expression is <code>Add dyn</code>, the fully generic path. "
        "Both are real, both are reachable, and the <code>Phi</code> merges them. Every "
        "mechanism in that picture already existed for other reasons, which is exactly why "
        "guards are not a node kind.")}

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
  <p><em>A dead branch inside a live one</em> is in the gallery for a reason that is worth
  stating on its own: <strong>a fixture can be correct by accident.</strong> The flat
  <code>if (0)</code> two specimens up exercised the same rewrite and passed, because the one
  node the broken code reached for happened to be a pinned root. Nothing about the flat picture
  says so; you only find out by moving the same construct somewhere its neighbours are ordinary
  nodes. The general form: when a rule reaches for a node it did not create, ask what happens
  when that node is not special.</p>
  <p>The loop specimen exposed something sharper. Building it in the wrong order, closing the
  loop before peepholing its body, deletes the entire loop and reports no error. The phi
  momentarily reads <code>int=0</code> because its back-edge value has no type yet; that is a
  fine <em>optimistic</em> answer and it would fall. But <code>i + 1</code> then computes
  <code>int=1</code>, which is a constant, and folding is irreversible. Every individual step is
  locally justified and the result is a deleted loop. The rule that falls out of it is worth
  stating plainly: <strong>an analysis may act on a provisional type, a transformation may
  not.</strong> That single mistake was sitting in three separate places.</p>
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
