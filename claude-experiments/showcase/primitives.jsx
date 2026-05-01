// Visual primitives: code blocks, terminal mocks, browser/phone frames,
// reveal-on-scroll wrapper.

const Reveal = ({ children, delay = 0, as: Tag = "div", className, style, ...rest }) => {
  const ref = React.useRef(null);
  const [shown, setShown] = React.useState(false);
  React.useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            setShown(true);
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -8% 0px" }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <Tag
      ref={ref}
      className={(className || "") + " reveal" + (shown ? " reveal-in" : "")}
      style={{ ...style, transitionDelay: `${delay}ms` }}
      {...rest}
    >
      {children}
    </Tag>
  );
};

// ─── Code block ────────────────────────────────────────────────────────────
// Tokenized via simple regex per language. Good enough for placeholder code.
const TOKEN_COLORS = {
  kw: "var(--code-kw)",
  str: "var(--code-str)",
  fn: "var(--code-fn)",
  num: "var(--code-num)",
  cmt: "var(--code-cmt)",
  pun: "var(--code-pun)",
  type: "var(--code-type)",
};

const KEYWORDS = {
  ts: /\b(import|from|const|let|var|function|return|if|else|for|while|class|new|export|default|async|await|=>|true|false|null|undefined)\b/g,
  go: /\b(func|return|if|else|for|range|var|const|import|package|type|struct|interface|map|chan|go|defer|select|switch|case|nil|true|false|int|string|bool|byte|rune)\b/g,
  rust: /\b(fn|let|mut|const|use|pub|struct|enum|impl|trait|match|if|else|for|while|loop|return|break|continue|in|as|self|Self|None|Some|Ok|Err|true|false|vec|move|ref)\b/g,
  simd: /\b(fn|stream|over|carry|return|let|if|else|for|in|ptr|u8|u16|u32|u64|i8|i16|i32|i64|f32|f64)\b/g,
  beagle: /\b(fn|let|use|as|loop|if|else|break|return|null|true|false|match|enum|struct|extend|with|handle|perform|resume)\b/g,
  pseudo: /\b(for|in|if|else|return)\b/g,
  tensor: /\b(fn|let|return|for|in|if|else|axis)\b/g,
  datalog: /\b(define|enum|assert|find|where|as_of|as_of_time|string|i64|f64|bool|required|unique|indexed)\b/g,
};

function tokenize(line, lang) {
  // Very small highlighter. Replace strings, comments, numbers, then keywords.
  const out = [];
  let i = 0;
  let buf = "";
  const flush = () => { if (buf) { out.push({ t: "txt", v: buf }); buf = ""; } };
  while (i < line.length) {
    const rest = line.slice(i);
    // line comment
    if (rest.startsWith("//")) { flush(); out.push({ t: "cmt", v: rest }); i = line.length; break; }
    // string
    const sm = rest.match(/^("[^"]*"|'[^']*'|`[^`]*`)/);
    if (sm) { flush(); out.push({ t: "str", v: sm[0] }); i += sm[0].length; continue; }
    // number
    const nm = rest.match(/^\b\d+(\.\d+)?\b/);
    if (nm) { flush(); out.push({ t: "num", v: nm[0] }); i += nm[0].length; continue; }
    buf += line[i];
    i++;
  }
  flush();
  // Keyword pass on plain text tokens
  const kwRe = KEYWORDS[lang];
  if (!kwRe) return out;
  const final = [];
  out.forEach((tok) => {
    if (tok.t !== "txt") { final.push(tok); return; }
    let last = 0;
    let m;
    kwRe.lastIndex = 0;
    while ((m = kwRe.exec(tok.v)) !== null) {
      if (m.index > last) final.push({ t: "txt", v: tok.v.slice(last, m.index) });
      final.push({ t: "kw", v: m[0] });
      last = m.index + m[0].length;
    }
    if (last < tok.v.length) final.push({ t: "txt", v: tok.v.slice(last) });
  });
  return final;
}

const CodeBlock = ({ code, accent }) => {
  const lines = code.lines;
  const lang = code.lang;
  return (
    <div className="codeblock" style={{ "--accent": accent }}>
      <div className="codeblock-bar">
        <span className="codeblock-lang">{lang}</span>
        {code.filename && <span className="codeblock-file">{code.filename}</span>}
      </div>
      <pre className="codeblock-pre">
        {lines.map((segs, idx) => {
          // segs is an array of strings making one line; we tokenize the joined line
          const text = Array.isArray(segs) ? segs.join("") : String(segs);
          const toks = tokenize(text, lang);
          return (
            <div className="codeblock-line" key={idx}>
              <span className="codeblock-ln">{String(idx + 1).padStart(2, " ")}</span>
              <span className="codeblock-code">
                {toks.length === 0 ? (
                  <span>&nbsp;</span>
                ) : (
                  toks.map((t, i) => (
                    <span key={i} style={{ color: TOKEN_COLORS[t.t] || "inherit" }}>{t.v}</span>
                  ))
                )}
              </span>
            </div>
          );
        })}
      </pre>
    </div>
  );
};

// ─── Terminal mock ─────────────────────────────────────────────────────────
const TerminalBlock = ({ lines, accent, label = "~/projects" }) => (
  <div className="terminal" style={{ "--accent": accent }}>
    <div className="terminal-bar">
      <span className="terminal-dots">
        <i /><i /><i />
      </span>
      <span className="terminal-title">{label}</span>
      <span style={{ width: 36 }} />
    </div>
    <div className="terminal-body">
      {lines.map((l, i) => {
        if (l.cmd) {
          return (
            <div className="terminal-line" key={i}>
              <span className="terminal-prompt">{l.prompt || "$"}</span>
              <span>{l.cmd}</span>
            </div>
          );
        }
        return (
          <div
            className={"terminal-line terminal-out" + (l.muted ? " terminal-muted" : "")}
            key={i}
          >
            {l.out || "\u00a0"}
          </div>
        );
      })}
      <div className="terminal-line">
        <span className="terminal-prompt">$</span>
        <span className="terminal-caret" />
      </div>
    </div>
  </div>
);

// ─── Browser frame mock ────────────────────────────────────────────────────
const BrowserFrame = ({ url, accent, children }) => (
  <div className="browser" style={{ "--accent": accent }}>
    <div className="browser-bar">
      <span className="browser-dots"><i /><i /><i /></span>
      <span className="browser-url">{url}</span>
      <span style={{ width: 36 }} />
    </div>
    <div className="browser-body">
      {children}
    </div>
  </div>
);

// ─── Phone frame mock ──────────────────────────────────────────────────────
const PhoneFrame = ({ accent, children }) => (
  <div className="phone" style={{ "--accent": accent }}>
    <div className="phone-screen">
      <div className="phone-notch" />
      {children}
    </div>
  </div>
);

// Synthetic visual content for the visual projects (placeholder UI snapshots).
const HalftoneVisual = ({ accent }) => (
  <div className="visual-halftone" style={{ "--accent": accent }}>
    <div className="vh-rule" />
    <div className="vh-headline">
      <span className="vh-num">№ 04</span>
      <span className="vh-meta">Composition · 2026</span>
    </div>
    <div className="vh-display">Aa</div>
    <div className="vh-grid">
      <div><span>Tracking</span><b>−12</b></div>
      <div><span>Optical</span><b>96pt</b></div>
      <div><span>Weight</span><b>720</b></div>
      <div><span>Slant</span><b>−2°</b></div>
    </div>
    <div className="vh-strip">
      {Array.from({ length: 24 }).map((_, i) => (
        <i key={i} style={{ height: 6 + ((i * 13) % 20) }} />
      ))}
    </div>
  </div>
);

const AtriumVisual = ({ accent }) => (
  <div className="visual-atrium" style={{ "--accent": accent }}>
    <div className="va-nav">
      <span>Atrium</span>
      <span className="va-nav-mid">Now showing — <em>Field of Light</em></span>
      <span>Rooms</span>
    </div>
    <div className="va-stage">
      <div className="va-piece" />
      <div className="va-caption">
        <span className="va-caption-num">01 / 06</span>
        <span className="va-caption-title">Field of Light</span>
        <span className="va-caption-artist">M. Okafor · 2026</span>
      </div>
    </div>
    <div className="va-thumbs">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className={"va-thumb" + (i === 0 ? " active" : "")} />
      ))}
    </div>
  </div>
);

const FieldnotesVisual = ({ accent }) => (
  <div className="visual-fieldnotes" style={{ "--accent": accent }}>
    <div className="vf-hd">
      <span>Tuesday</span>
      <span className="vf-time">11:42</span>
    </div>
    <div className="vf-feed">
      <div className="vf-note">
        <span className="vf-note-time">9:14</span>
        <p>The light through the kitchen window at exactly this hour. Note for the studio.</p>
      </div>
      <div className="vf-note">
        <span className="vf-note-time">10:02</span>
        <p>What if the index isn't a tree but a tide?</p>
      </div>
      <div className="vf-note">
        <span className="vf-note-time">10:48</span>
        <p>Re: yesterday — the difference between <em>private</em> and <em>quiet</em>.</p>
      </div>
      <div className="vf-note vf-note-active">
        <span className="vf-note-time">11:42</span>
        <p>Coffee. The good kind. Bring the small notebook.</p>
        <span className="vf-caret" />
      </div>
    </div>
    <div className="vf-input">
      <span>A new note</span>
      <span className="vf-plus">+</span>
    </div>
  </div>
);

const MeridianVisual = ({ accent }) => (
  <div className="visual-meridian" style={{ "--accent": accent }}>
    <div className="vm-hd">
      <span>Pinnacles</span>
      <span className="vm-loc">36.49°N · 121.18°W</span>
    </div>
    <div className="vm-temp">
      <span className="vm-num">58°</span>
      <span className="vm-cond">Clearing · wind dropping</span>
    </div>
    <div className="vm-chart">
      <svg viewBox="0 0 200 60" preserveAspectRatio="none">
        <path d="M0 38 L20 32 L40 28 L60 22 L80 18 L100 16 L120 18 L140 24 L160 30 L180 36 L200 42"
              fill="none" stroke="var(--accent)" strokeWidth="1.2" />
        <path d="M0 38 L20 32 L40 28 L60 22 L80 18 L100 16 L120 18 L140 24 L160 30 L180 36 L200 42 L200 60 L0 60 Z"
              fill="var(--accent)" opacity="0.08" />
      </svg>
      <div className="vm-axis">
        <span>now</span><span>+2h</span><span>+4h</span><span>+6h</span><span>+8h</span>
      </div>
    </div>
    <div className="vm-grid">
      <div><span>Wind</span><b>8 → 3</b></div>
      <div><span>Golden</span><b>18:42</b></div>
      <div><span>Low</span><b>49°</b></div>
    </div>
  </div>
);

const VISUAL_MAP = {
  halftone: HalftoneVisual,
  atrium: AtriumVisual,
  fieldnotes: FieldnotesVisual,
  meridian: MeridianVisual,
};

// ─── Perf table ────────────────────────────────────────────────────────────
const PerfTable = ({ table, accent }) => (
  <div className="perftable" style={{ "--accent": accent }}>
    <div className="perftable-head">
      {table.headers.map((h, i) => <span key={i}>{h}</span>)}
    </div>
    {table.rows.map((row, i) => {
      const highlight = row[row.length - 1] === "highlight";
      const cells = highlight ? row.slice(0, -1) : row;
      return (
        <div className={"perftable-row" + (highlight ? " is-highlight" : "")} key={i}>
          {cells.map((c, j) => <span key={j}>{c}</span>)}
        </div>
      );
    })}
    {table.caption && <div className="perftable-caption">{table.caption}</div>}
  </div>
);

// ─── Architecture list ────────────────────────────────────────────────────
const ArchList = ({ arch, accent, projectId }) => (
  <div className="archlist" style={{ "--accent": accent }}>
    {arch.title && (
      <div
        className="archlist-title"
        data-edit-key={projectId ? `${projectId}.arch.title` : undefined}
      >
        {arch.title}
      </div>
    )}
    <div className="archlist-items">
      {arch.items.map((it, i) => (
        <div className="archlist-item" key={i}>
          <span className="archlist-num">{String(i + 1).padStart(2, "0")}</span>
          <div className="archlist-body">
            <b data-edit-key={projectId ? `${projectId}.arch.${i}.label` : undefined}>
              {it.label}
            </b>
            <p data-edit-key={projectId ? `${projectId}.arch.${i}.body` : undefined}>
              {it.body}
            </p>
          </div>
        </div>
      ))}
    </div>
  </div>
);

Object.assign(window, {
  Reveal,
  CodeBlock,
  TerminalBlock,
  BrowserFrame,
  PhoneFrame,
  VISUAL_MAP,
  PerfTable,
  ArchList,
});
