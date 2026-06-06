//! JavaScript tokenizer — Stage 2 (scalar).
//!
//! Stage 1 (`examples/js_stage1.simd`) is a SIMD pass that emits a stream of
//! *token-start positions*, context-blind: it flags word-starts and every
//! punctuator/quote/operator byte, but knows nothing about strings, comments,
//! regex-vs-divide, template `${}` nesting, or multi-char operators.
//!
//! Stage 2 consumes that position stream as its driver and owns exactly the
//! irreducibly-sequential parts SIMD can't do:
//!   * string / comment / template / regex interiors (it scans them and skips
//!     the spurious stage-1 starts that fall inside),
//!   * regex-vs-division disambiguation (via the previous significant token),
//!   * template `${ ... }` nesting (a brace/template stack),
//!   * longest-match for multi-char operators (`===`, `>>>=`, `=>`, ...).
//!
//! The oracle is a round-trip: the tokens (plus the whitespace gaps between
//! them) must reconstruct the source byte-for-byte, which proves every byte is
//! covered exactly once.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokKind {
    Ident,
    Keyword,
    Number,
    String,
    /// `` `...` `` with no `${}` — a complete template literal.
    TemplateNoSub,
    /// `` `...${ `` — opens a substitution.
    TemplateHead,
    /// `` }...${ `` — closes one substitution, opens the next.
    TemplateMiddle,
    /// `` }...` `` — closes the final substitution and the literal.
    TemplateTail,
    Regex,
    Punct,
    LineComment,
    BlockComment,
}

impl TokKind {
    #[inline]
    fn is_comment(self) -> bool {
        matches!(self, TokKind::LineComment | TokKind::BlockComment)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Token {
    pub kind: TokKind,
    pub start: usize,
    pub end: usize,
}

impl Token {
    pub fn text<'a>(&self, src: &'a [u8]) -> &'a str {
        std::str::from_utf8(&src[self.start..self.end]).unwrap_or("<non-utf8>")
    }
}

/// For each length 0..=10, a bitmask (bit `c-'a'`) of which lowercase first
/// letters begin *some* keyword of that length. Lets `is_keyword` reject most
/// identifiers from their (length, first-byte) alone — both already in hand from
/// `word_end` / `classify` — without reading the identifier body at all.
const COULD_BE_KW: [u32; 11] = {
    let kws: [&[u8]; 44] = [
        b"await", b"break", b"case", b"catch", b"class", b"const", b"continue",
        b"debugger", b"default", b"delete", b"do", b"else", b"enum", b"export",
        b"extends", b"false", b"finally", b"for", b"function", b"if", b"import",
        b"in", b"instanceof", b"let", b"new", b"null", b"of", b"return", b"super",
        b"switch", b"this", b"throw", b"true", b"try", b"typeof", b"var", b"void",
        b"while", b"with", b"yield", b"async", b"static", b"get", b"set",
    ];
    let mut t = [0u32; 11];
    let mut i = 0;
    while i < kws.len() {
        let w = kws[i];
        if w.len() <= 10 {
            t[w.len()] |= 1u32 << (w[0] - b'a');
        }
        i += 1;
    }
    t
};

/// Keyword test on raw bytes (no UTF-8 validation needed — keywords are ASCII).
#[inline]
fn is_keyword(w: &[u8]) -> bool {
    let len = w.len();
    if len < 2 || len > 10 {
        return false;
    }
    let first = w[0];
    // Reject from (length, first letter) without touching the body: keywords are
    // all-lowercase, and most identifiers share no (len, first) with any keyword.
    if !first.is_ascii_lowercase() || (COULD_BE_KW[len] >> (first - b'a')) & 1 == 0 {
        return false;
    }
    matches!(
        w,
        b"await" | b"break" | b"case" | b"catch" | b"class" | b"const" | b"continue"
            | b"debugger" | b"default" | b"delete" | b"do" | b"else" | b"enum"
            | b"export" | b"extends" | b"false" | b"finally" | b"for" | b"function"
            | b"if" | b"import" | b"in" | b"instanceof" | b"let" | b"new" | b"null"
            | b"of" | b"return" | b"super" | b"switch" | b"this" | b"throw" | b"true"
            | b"try" | b"typeof" | b"var" | b"void" | b"while" | b"with" | b"yield"
            | b"async" | b"static" | b"get" | b"set"
    )
}

/// End (exclusive) of the word run starting at `p`, read from stage-1's `is_word`
/// bitmap (one u64 per 64-byte chunk) by a bit-scan — no byte re-read. `p` must
/// be a word byte. Returns the first index >= p whose word bit is 0, or `len`.
#[inline]
fn word_end(word_masks: &[u64], p: usize, len: usize) -> usize {
    let mut chunk = p / 64;
    let mut bit = p % 64;
    loop {
        if chunk >= word_masks.len() {
            return len;
        }
        // Force the bits below `bit` to 1 so they aren't mistaken for the run end,
        // then the first 0 bit is at index `trailing_ones`.
        let masked = word_masks[chunk] | ((1u64 << bit) - 1);
        if masked != u64::MAX {
            return (chunk * 64 + masked.trailing_ones() as usize).min(len);
        }
        chunk += 1;
        bit = 0;
    }
}

#[inline]
fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$'
}

#[inline]
fn is_digit(b: u8) -> bool {
    b.is_ascii_digit()
}

#[inline]
fn is_js_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

// Lead-byte class, so `classify` is one table load + jump instead of a chain of
// ~7 failed comparisons (punctuators — ~half of minified tokens — used to fall
// through all of them).
const C_PUNCT1: u8 = 0; // always-single-char punctuator: ( ) [ ] { } ; , : ~ @ # …
const C_WORD: u8 = 1; // [A-Za-z_$] — identifier/keyword start
const C_DIGIT: u8 = 2; // [0-9] — number start
const C_DOT: u8 = 3; // `.` — number (if followed by digit) or operator
const C_SLASH: u8 = 4; // `/` — comment / regex / divide
const C_DQUOTE: u8 = 5;
const C_SQUOTE: u8 = 6;
const C_BACKTICK: u8 = 7;
const C_OP: u8 = 8; // operator starter: + - * = < > & | ! ^ % ? — needs longest-match

const CLASS: [u8; 256] = {
    let mut t = [C_PUNCT1; 256]; // default: single-char punctuator
    let mut i = 0;
    while i < 256 {
        let b = i as u8;
        if b.is_ascii_alphabetic() || b == b'_' || b == b'$' {
            t[i] = C_WORD;
        } else if b.is_ascii_digit() {
            t[i] = C_DIGIT;
        }
        i += 1;
    }
    // operator starters (multi-char possible)
    let ops = b"+-*=<>&|!^%?";
    let mut j = 0;
    while j < ops.len() {
        t[ops[j] as usize] = C_OP;
        j += 1;
    }
    t[b'.' as usize] = C_DOT;
    t[b'/' as usize] = C_SLASH;
    t[b'"' as usize] = C_DQUOTE;
    t[b'\'' as usize] = C_SQUOTE;
    t[b'`' as usize] = C_BACKTICK;
    t
};

/// After which previous significant token is a leading `/` a regex literal
/// rather than a division operator? This is the lexer/parser feedback that
/// makes JS impossible to lex context-free.
fn regex_allowed(prev: Option<(TokKind, &str)>) -> bool {
    match prev {
        None => true,
        // value-ending tokens → `/` is division. A complete template
        // (`TemplateNoSub`) or a `TemplateTail` ends a value; `TemplateHead`/
        // `TemplateMiddle` end with `${`, opening a fresh expression → regex ok.
        Some((TokKind::Ident, _)) | Some((TokKind::Number, _))
        | Some((TokKind::String, _)) | Some((TokKind::TemplateNoSub, _))
        | Some((TokKind::TemplateTail, _)) | Some((TokKind::Regex, _)) => false,
        Some((TokKind::Keyword, kw)) => {
            // value-producing keywords are followed by division; the rest
            // (return, typeof, delete, void, in, of, instanceof, new, ...)
            // introduce an expression where `/` starts a regex.
            !matches!(kw, "this" | "super" | "true" | "false" | "null")
        }
        Some((TokKind::Punct, p)) => {
            // `)`, `]`, `}` close a value → division. Everything else → regex.
            !matches!(p, ")" | "]" | "}")
        }
        // `${`-opening template parts, and comments (filtered out before we get
        // here anyway) → regex allowed.
        Some((TokKind::TemplateHead, _)) | Some((TokKind::TemplateMiddle, _))
        | Some((TokKind::LineComment, _)) | Some((TokKind::BlockComment, _)) => true,
    }
}

fn scan_string(src: &[u8], start: usize) -> usize {
    // Scalar loop: JS string literals are short, so memchr's SIMD setup cost
    // doesn't pay off here (it does for the long comment scans above).
    let quote = src[start];
    let mut i = start + 1;
    while i < src.len() {
        match src[i] {
            b'\\' => i += 2, // skip escaped char
            c if c == quote => return i + 1,
            _ => i += 1,
        }
    }
    src.len()
}

/// Scan template literal text from byte `i` (just past an opening `` ` `` or a
/// `}`) to the next `${` or closing backtick. Returns `TemplateHead` if it
/// stopped at `${` (end just past it), else `TemplateNoSub` at the closing
/// backtick. `template_tail` maps Head→Middle and NoSub→Tail for the `}`-led
/// case. The `${ }` *nesting* is resolved by the driver/parser, not here — this
/// only scans one flat run of template text, exactly like oxc's
/// `read_template_literal`.
fn template_scan(src: &[u8], mut i: usize) -> (TokKind, usize) {
    while i < src.len() {
        match src[i] {
            b'\\' => i += 2,
            b'`' => return (TokKind::TemplateNoSub, i + 1),
            b'$' if i + 1 < src.len() && src[i + 1] == b'{' => {
                return (TokKind::TemplateHead, i + 2);
            }
            _ => i += 1,
        }
    }
    (TokKind::TemplateNoSub, src.len())
}

/// From the opening backtick at `start`.
fn template_head(src: &[u8], start: usize) -> (TokKind, usize) {
    template_scan(src, start + 1)
}

/// From a `}` at `start` that closes a `${ }` substitution.
fn template_tail(src: &[u8], start: usize) -> (TokKind, usize) {
    let (k, end) = template_scan(src, start + 1);
    let k = match k {
        TokKind::TemplateHead => TokKind::TemplateMiddle,
        _ => TokKind::TemplateTail,
    };
    (k, end)
}

fn scan_regex(src: &[u8], start: usize) -> usize {
    let mut i = start + 1;
    let mut in_class = false;
    while i < src.len() {
        match src[i] {
            b'\\' => i += 2,
            b'[' => {
                in_class = true;
                i += 1;
            }
            b']' => {
                in_class = false;
                i += 1;
            }
            b'/' if !in_class => {
                i += 1;
                // flags
                while i < src.len() && is_word(src[i]) {
                    i += 1;
                }
                return i;
            }
            b'\n' => return i, // unterminated — bail at line end
            _ => i += 1,
        }
    }
    src.len()
}

fn scan_number(src: &[u8], start: usize) -> usize {
    let mut i = start;
    let n = src.len();
    // radix prefixes
    if src[i] == b'0' && i + 1 < n {
        match src[i + 1] {
            b'x' | b'X' | b'b' | b'B' | b'o' | b'O' => {
                i += 2;
                while i < n && (src[i].is_ascii_alphanumeric() || src[i] == b'_') {
                    i += 1;
                }
                if i < n && src[i] == b'n' {
                    i += 1;
                }
                return i;
            }
            _ => {}
        }
    }
    while i < n && (is_digit(src[i]) || src[i] == b'_') {
        i += 1;
    }
    if i < n && src[i] == b'.' {
        i += 1;
        while i < n && (is_digit(src[i]) || src[i] == b'_') {
            i += 1;
        }
    }
    if i < n && matches!(src[i], b'e' | b'E') {
        i += 1;
        if i < n && matches!(src[i], b'+' | b'-') {
            i += 1;
        }
        while i < n && (is_digit(src[i]) || src[i] == b'_') {
            i += 1;
        }
    }
    if i < n && src[i] == b'n' {
        i += 1; // bigint
    }
    i
}

/// Longest-match JS punctuator, by direct byte dispatch (no string scan).
/// `/`-led operators (`/`, `/=`) reach here only when stage-2 has already ruled
/// out comment/regex.
#[inline]
fn match_operator(src: &[u8], start: usize) -> usize {
    let b0 = src[start];
    // Fast path: bytes that can never begin a multi-char operator are single
    // punctuators — return immediately without any look-ahead reads. This is the
    // common case in real code ( `(` `)` `[` `]` `{` `}` `;` `,` `:` `~` ).
    if !matches!(
        b0,
        b'>' | b'<' | b'=' | b'!' | b'*' | b'&' | b'|' | b'?' | b'+' | b'-' | b'%' | b'^' | b'/' | b'.'
    ) {
        return start + 1;
    }
    let n = src.len();
    let b = |o: usize| if start + o < n { src[start + o] } else { 0 };
    let b1 = b(1);
    let b2 = b(2);
    let b3 = b(3);
    let len = match b0 {
        b'>' => match (b1, b2, b3) {
            (b'>', b'>', b'=') => 4, // >>>=
            (b'>', b'>', _) => 3,    // >>>
            (b'>', b'=', _) => 3,    // >>=
            (b'>', _, _) => 2,       // >>
            (b'=', _, _) => 2,       // >=
            _ => 1,
        },
        b'<' => match (b1, b2) {
            (b'<', b'=') => 3, // <<=
            (b'<', _) => 2,    // <<
            (b'=', _) => 2,    // <=
            _ => 1,
        },
        b'=' => match (b1, b2) {
            (b'=', b'=') => 3, // ===
            (b'=', _) => 2,    // ==
            (b'>', _) => 2,    // =>
            _ => 1,
        },
        b'!' => match (b1, b2) {
            (b'=', b'=') => 3, // !==
            (b'=', _) => 2,    // !=
            _ => 1,
        },
        b'*' => match (b1, b2) {
            (b'*', b'=') => 3, // **=
            (b'*', _) => 2,    // **
            (b'=', _) => 2,    // *=
            _ => 1,
        },
        b'&' => match (b1, b2) {
            (b'&', b'=') => 3, // &&=
            (b'&', _) => 2,    // &&
            (b'=', _) => 2,    // &=
            _ => 1,
        },
        b'|' => match (b1, b2) {
            (b'|', b'=') => 3, // ||=
            (b'|', _) => 2,    // ||
            (b'=', _) => 2,    // |=
            _ => 1,
        },
        b'?' => match (b1, b2) {
            (b'?', b'=') => 3, // ??=
            (b'?', _) => 2,    // ??
            (b'.', _) => 2,    // ?.
            _ => 1,
        },
        b'+' => if b1 == b'+' || b1 == b'=' { 2 } else { 1 }, // ++ +=
        b'-' => if b1 == b'-' || b1 == b'=' { 2 } else { 1 }, // -- -=
        b'%' => if b1 == b'=' { 2 } else { 1 },               // %=
        b'^' => if b1 == b'=' { 2 } else { 1 },               // ^=
        b'/' => if b1 == b'=' { 2 } else { 1 },               // /=
        b'.' => if b1 == b'.' && b2 == b'.' { 3 } else { 1 }, // ...
        _ => 1, // { } ( ) [ ] ; , ~ : and any unknown byte
    };
    start + len
}

/// Pull-based lexer over the stage-1 `positions` stream + `word_masks` bitmap.
///
/// A parser drives it exactly like oxc drives its own lexer: pull tokens with
/// [`Lexer::next_token`] / [`Lexer::peek`], and at the two context-sensitive
/// sites call back — [`Lexer::re_lex_regex`] when a `/` is a regex in expression
/// position, and [`Lexer::re_lex_template_tail`] when a `}` closes a `${ }`
/// substitution. The lexer itself is context-free; the policy for *when* to call
/// the hooks lives in the driver/parser (see [`tokenize`] for the standalone
/// driver — the stand-in for a real parser).
///
/// Identifier ends come from `word_masks` via a bit-scan; identifier bytes are
/// read at most once (for keyword classification), never re-scanned to find the
/// end. The lexer never re-reads bytes it has already classified.
pub struct Lexer<'a> {
    src: &'a [u8],
    start_masks: &'a [u64], // bit i of word c ⇔ byte c*64+i starts a token
    word_masks: &'a [u64],  // bit i of word c ⇔ byte c*64+i is a word char
    cursor: usize,          // bytes before this are consumed
}

/// Classify the token starting at byte `p`, returning (kind, end). `/` is always
/// a punctuator (the driver re-lexes regexes); a backtick yields a
/// `TemplateHead`/`TemplateNoSub`, never a whole nested template. Free function
/// so both the pull `Lexer` and the fast `count_tokens` loop share it.
#[inline]
fn classify(src: &[u8], word_masks: &[u64], p: usize) -> (TokKind, usize) {
    match CLASS[src[p] as usize] {
        C_WORD => {
            let e = word_end(word_masks, p, src.len());
            let kind = if is_keyword(&src[p..e]) { TokKind::Keyword } else { TokKind::Ident };
            (kind, e)
        }
        C_DIGIT => (TokKind::Number, scan_number(src, p)),
        C_PUNCT1 => (TokKind::Punct, p + 1), // single-char punctuator
        C_OP => (TokKind::Punct, match_operator(src, p)),
        C_DOT => {
            if p + 1 < src.len() && is_digit(src[p + 1]) {
                (TokKind::Number, scan_number(src, p))
            } else {
                (TokKind::Punct, match_operator(src, p))
            }
        }
        C_SLASH => {
            if p + 1 < src.len() && src[p + 1] == b'/' {
                let e = memchr::memchr(b'\n', &src[p + 2..]).map_or(src.len(), |o| p + 2 + o);
                (TokKind::LineComment, e)
            } else if p + 1 < src.len() && src[p + 1] == b'*' {
                let e = memchr::memmem::find(&src[p + 2..], b"*/").map_or(src.len(), |o| p + 2 + o + 2);
                (TokKind::BlockComment, e)
            } else {
                (TokKind::Punct, match_operator(src, p))
            }
        }
        C_BACKTICK => template_head(src, p),
        _ => (TokKind::String, scan_string(src, p)), // C_DQUOTE | C_SQUOTE
    }
}

/// First token-start at or after byte `from`, by ctz over `start_masks`.
#[inline]
fn next_set_bit(masks: &[u64], from: usize) -> Option<usize> {
    let mut chunk = from / 64;
    if chunk >= masks.len() {
        return None;
    }
    // mask off bits below `from` in the first word
    let mut m = masks[chunk] & (!0u64 << (from % 64));
    loop {
        if m != 0 {
            return Some(chunk * 64 + m.trailing_zeros() as usize);
        }
        chunk += 1;
        if chunk >= masks.len() {
            return None;
        }
        m = masks[chunk];
    }
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a [u8], start_masks: &'a [u64], word_masks: &'a [u64]) -> Self {
        Lexer { src, start_masks, word_masks, cursor: 0 }
    }

    /// Classify the token starting at byte `p`. `/` is always returned as a
    /// Find and classify the next token starting at or after `cursor`, without
    /// mutating self.
    #[inline]
    fn lex_from(&self, cursor: usize) -> Option<Token> {
        let p = next_set_bit(self.start_masks, cursor)?;
        if p >= self.src.len() {
            return None;
        }
        let (kind, end) = classify(self.src, self.word_masks, p);
        Some(Token { kind, start: p, end })
    }

    pub fn next_token(&mut self) -> Option<Token> {
        let tok = self.lex_from(self.cursor)?;
        self.cursor = tok.end;
        Some(tok)
    }

    /// Look at the next token without consuming it.
    pub fn peek(&self) -> Option<Token> {
        self.lex_from(self.cursor)
    }

    /// Re-lex `at` (the `/`/`/=` token just returned by `next_token`) as a regex.
    pub fn re_lex_regex(&mut self, at: Token) -> Token {
        let end = scan_regex(self.src, at.start);
        self.cursor = end;
        Token { kind: TokKind::Regex, start: at.start, end }
    }

    /// Re-lex `at` (the `}` token just returned) as a `TemplateMiddle`/`Tail`.
    pub fn re_lex_template_tail(&mut self, at: Token) -> Token {
        let (kind, end) = template_tail(self.src, at.start);
        self.cursor = end;
        Token { kind, start: at.start, end }
    }
}

/// Standalone driver: drive [`Lexer`] to completion applying the standard
/// regex/template feedback policy, calling `emit` for each token. This is the
/// stand-in for a real parser — it replicates exactly what the parser feeds back
/// (the prev-significant-token regex rule + the `${ }` brace-depth stack), the
/// same way `bench::oxc_tokens_standalone` drives oxc's lexer. A real parser
/// would replace `emit` + the policy with its grammar-aware consumption.
#[inline]
pub fn drive<F: FnMut(Token)>(src: &[u8], start_masks: &[u64], word_masks: &[u64], mut emit: F) {
    let mut lex = Lexer::new(src, start_masks, word_masks);
    let mut tmpl: Vec<u32> = Vec::new(); // `{`-depth within each active `${ }` substitution
    // Whether the previous significant token ends an expression (so a following
    // `/` is division, not regex). A single bool replaces the old
    // `Option<(kind,start,end)>` + per-slash `from_utf8` + `regex_allowed` match
    // — the per-token bookkeeping was the dominant cost on dense code.
    let mut prev_ends_expr = false;

    while let Some(mut tok) = lex.next_token() {
        // Punctuator feedback: regex re-lex, template-tail re-lex, brace depth.
        if tok.kind == TokKind::Punct {
            match src[tok.start] {
                b'/' if !prev_ends_expr => tok = lex.re_lex_regex(tok),
                b'}' if tmpl.last() == Some(&0) => {
                    // closes a `${ }` substitution → TemplateMiddle/Tail
                    tok = lex.re_lex_template_tail(tok);
                }
                b'{' if !tmpl.is_empty() => *tmpl.last_mut().unwrap() += 1,
                b'}' if !tmpl.is_empty() => {
                    let d = tmpl.last_mut().unwrap();
                    *d = d.saturating_sub(1);
                }
                _ => {}
            }
        }

        // Template-part transitions (only fire for the rare template kinds).
        match tok.kind {
            TokKind::TemplateHead => tmpl.push(0),
            TokKind::TemplateMiddle => *tmpl.last_mut().unwrap() = 0,
            TokKind::TemplateTail => {
                tmpl.pop();
            }
            _ => {}
        }

        // Update prev-ends-expression (regex context), cheaply.
        prev_ends_expr = match tok.kind {
            TokKind::Ident
            | TokKind::Number
            | TokKind::String
            | TokKind::Regex
            | TokKind::TemplateNoSub
            | TokKind::TemplateTail => true,
            TokKind::Punct => matches!(src[tok.start], b')' | b']' | b'}'),
            TokKind::Keyword => {
                matches!(&src[tok.start..tok.end], b"this" | b"super" | b"true" | b"false" | b"null")
            }
            // TemplateHead/Middle end with `${` (expression follows); comments
            // are transparent — keep the prior value.
            TokKind::TemplateHead | TokKind::TemplateMiddle => false,
            _ => prev_ends_expr,
        };

        emit(tok);
    }
}

/// Materializing convenience driver (used by tests / round-trip / token dumps).
pub fn tokenize(src: &[u8], start_masks: &[u64], word_masks: &[u64]) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    drive(src, start_masks, word_masks, |t| tokens.push(t));
    tokens
}

/// Non-materializing driver: produce the full token stream but only count it —
/// the fair, parser-relevant throughput measurement (a real parser consumes
/// tokens on the fly; oxc's lexer loop likewise stores nothing).
///
/// Iterates the start bitmap **word-at-a-time** (simdjson-style): the current
/// 64-bit word lives in a register and bits are cleared as tokens are consumed,
/// instead of re-loading + re-masking the word on every token (`next_set_bit`).
/// Shares `classify` with the pull `Lexer` so the token stream is identical.
pub fn count_tokens(src: &[u8], start_masks: &[u64], word_masks: &[u64]) -> usize {
    let n_words = start_masks.len();
    if n_words == 0 {
        return 0;
    }
    let src_len = src.len();
    let mut n = 0usize;
    let mut prev_ends_expr = false;
    let mut tmpl: Vec<u32> = Vec::new();

    let mut wi = 0usize;
    let mut word = start_masks[0];

    loop {
        while word == 0 {
            wi += 1;
            if wi >= n_words {
                return n;
            }
            word = start_masks[wi];
        }
        let p = wi * 64 + word.trailing_zeros() as usize;
        if p >= src_len {
            return n;
        }

        let (mut kind, mut end) = classify(src, word_masks, p);

        // Regex / template feedback (same policy as the pull Lexer driver).
        if tmpl.is_empty() {
            if kind == TokKind::Punct {
                let lead = src[p];
                if lead == b'/' && !prev_ends_expr {
                    end = scan_regex(src, p);
                    kind = TokKind::Regex;
                    prev_ends_expr = true;
                } else {
                    prev_ends_expr = matches!(lead, b')' | b']' | b'}');
                }
            } else {
                prev_ends_expr = ends_expr_k(src, kind, p, end, prev_ends_expr);
            }
            if kind == TokKind::TemplateHead {
                tmpl.push(0);
            }
        } else {
            let r = template_step_free(src, &mut tmpl, kind, p, end, &mut prev_ends_expr);
            kind = r.0;
            end = r.1;
        }
        let _ = kind;
        n += 1;

        // Advance to `end`: clear bits below it (staying in-register if possible).
        let ewi = end / 64;
        let ebit = end % 64;
        if ewi == wi {
            word &= !0u64 << ebit; // ebit > p%64 here, never a 64-shift
        } else {
            if ewi >= n_words {
                return n;
            }
            wi = ewi;
            word = start_masks[wi] & (!0u64 << ebit);
        }
    }
}

/// regex-context flag for non-punctuator tokens (comments are transparent).
#[inline]
fn ends_expr_k(src: &[u8], kind: TokKind, p: usize, end: usize, prev: bool) -> bool {
    match kind {
        TokKind::Ident
        | TokKind::Number
        | TokKind::String
        | TokKind::Regex
        | TokKind::TemplateNoSub
        | TokKind::TemplateTail => true,
        TokKind::Keyword => {
            matches!(&src[p..end], b"this" | b"super" | b"true" | b"false" | b"null")
        }
        TokKind::TemplateHead | TokKind::TemplateMiddle => false,
        _ => prev, // comments transparent
    }
}

/// Cold slow-path: a `${ }` substitution is open — brace-depth tracking +
/// template-tail / regex re-lex. Returns the (possibly re-lexed) (kind, end).
#[cold]
fn template_step_free(
    src: &[u8],
    tmpl: &mut Vec<u32>,
    mut kind: TokKind,
    p: usize,
    mut end: usize,
    prev_ends_expr: &mut bool,
) -> (TokKind, usize) {
    if kind == TokKind::Punct {
        match src[p] {
            b'/' if !*prev_ends_expr => {
                end = scan_regex(src, p);
                kind = TokKind::Regex;
            }
            b'}' if tmpl.last() == Some(&0) => {
                let (k, e) = template_tail(src, p);
                kind = k;
                end = e;
            }
            b'{' => *tmpl.last_mut().unwrap() += 1,
            b'}' => {
                let d = tmpl.last_mut().unwrap();
                *d = d.saturating_sub(1);
            }
            _ => {}
        }
    }
    match kind {
        TokKind::TemplateHead => tmpl.push(0),
        TokKind::TemplateMiddle => *tmpl.last_mut().unwrap() = 0,
        TokKind::TemplateTail => {
            tmpl.pop();
        }
        _ => {}
    }
    *prev_ends_expr = if kind == TokKind::Punct {
        matches!(src[p], b')' | b']' | b'}')
    } else {
        ends_expr_k(src, kind, p, end, *prev_ends_expr)
    };
    (kind, end)
}

/// Diagnostic: raw lex (bitmap-iterate + classify only, NO regex/template
/// feedback). Measures the floor of classification cost vs the driver overhead.
pub fn count_raw(src: &[u8], start_masks: &[u64], word_masks: &[u64]) -> usize {
    let mut lex = Lexer::new(src, start_masks, word_masks);
    let mut n = 0usize;
    while lex.next_token().is_some() {
        n += 1;
    }
    n
}

/// Diagnostic: pure word-at-a-time bitmap-iteration floor — visit every set bit
/// with no classify and no end-skipping. The hard lower bound on stage-2.
pub fn count_floor(start_masks: &[u64]) -> usize {
    let mut n = 0usize;
    for &w in start_masks {
        let mut word = w;
        while word != 0 {
            n += 1;
            word &= word - 1; // clear lowest set bit
        }
    }
    n
}

/// Reconstruct the source from the token list, filling inter-token gaps from
/// the original bytes. Returns Err if any gap byte is non-whitespace (which
/// would mean a byte fell through tokenization). On success the returned String
/// equals the original source byte-for-byte.
pub fn reconstruct(src: &[u8], tokens: &[Token]) -> Result<String, String> {
    let mut out: Vec<u8> = Vec::with_capacity(src.len());
    let mut cursor = 0usize;
    for t in tokens {
        if t.start < cursor {
            return Err(format!(
                "token overlap at {}..{} (cursor {})",
                t.start, t.end, cursor
            ));
        }
        // gap before this token must be pure whitespace
        for (off, &b) in src[cursor..t.start].iter().enumerate() {
            if !is_js_ws(b) {
                return Err(format!(
                    "non-whitespace gap byte {:?} at offset {}",
                    b as char,
                    cursor + off
                ));
            }
        }
        out.extend_from_slice(&src[cursor..t.end]);
        cursor = t.end;
    }
    // trailing gap
    for (off, &b) in src[cursor..].iter().enumerate() {
        if !is_js_ws(b) {
            return Err(format!(
                "trailing non-whitespace byte {:?} at offset {}",
                b as char,
                cursor + off
            ));
        }
    }
    out.extend_from_slice(&src[cursor..]);
    String::from_utf8(out).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{codegen, parser};
    use std::collections::HashMap;

    /// JIT-compile examples/js_stage1.simd and run it. Returns (start_masks,
    /// word_masks) bitmaps. `input` must be padded to a multiple of 64.
    fn run_stage1(input: &[u8]) -> (Vec<u64>, Vec<u64>) {
        let source = std::fs::read_to_string("examples/js_stage1.simd")
            .expect("cannot read examples/js_stage1.simd");
        let ctx = codegen::create_context();
        let items = parser::parse(&source);
        let mut module = codegen::compile_module(&ctx, &items, &HashMap::new(), 8);
        codegen::lower_to_llvm(&ctx, &mut module).expect("lowering failed");
        let engine = melior::ExecutionEngine::new(&module, 2, &[], false);
        let fptr = engine.lookup("js_stage1");
        assert!(!fptr.is_null(), "js_stage1 symbol not found");
        let nchunks = input.len() / 64 + 1;
        let mut start_masks = vec![0u64; nchunks];
        let mut word_masks = vec![0u64; nchunks];
        unsafe {
            // js_stage1(input: memref<?xi8>, start_masks: memref<?xi64>,
            //           word_masks: memref<?xi64>) -> vector<1xi32>
            let f: extern "C" fn(
                *const u8, *const u8, i64, i64, i64,
                *mut u64, *mut u64, i64, i64, i64,
                *mut u64, *mut u64, i64, i64, i64,
            ) -> f32 = std::mem::transmute(fptr);
            f(
                input.as_ptr(), input.as_ptr(), 0, input.len() as i64, 1,
                start_masks.as_mut_ptr(), start_masks.as_mut_ptr(), 0, start_masks.len() as i64, 1,
                word_masks.as_mut_ptr(), word_masks.as_mut_ptr(), 0, word_masks.len() as i64, 1,
            );
        }
        (start_masks, word_masks)
    }

    /// Expand a token-start bitmap to the list of start positions < `len`.
    fn starts_of(start_masks: &[u64], len: usize) -> Vec<i32> {
        let mut v = Vec::new();
        let mut c = 0usize;
        while let Some(p) = next_set_bit(start_masks, c) {
            if p >= len {
                break;
            }
            v.push(p as i32);
            c = p + 1;
        }
        v
    }

    /// Pad to a multiple of 64 with NUL so the SIMD stream sees whole chunks.
    fn padded(src: &str) -> Vec<u8> {
        let mut v = src.as_bytes().to_vec();
        while v.len() % 64 != 0 {
            v.push(0);
        }
        v
    }

    #[test]
    fn stage1_finds_token_starts() {
        let src = "let x = foo + 42;";
        //         0123456789...
        let buf = padded(src);
        let (start_masks, _) = run_stage1(&buf);
        let got = starts_of(&start_masks, buf.len());
        // token starts: l(0) x(4) =(6) f(8) +(12) 4(14) ;(16)
        let expected = vec![0, 4, 6, 8, 12, 14, 16];
        assert_eq!(got, expected, "stage-1 token-start positions");
    }

    #[test]
    fn stage1_crosses_chunk_boundary() {
        // Put an identifier straddling the 64-byte boundary; the carried
        // "prev was word char" bit must prevent a false token-start at lane 0.
        let prefix = "a".repeat(60); // bytes 0..60, one long ident
        let src = format!("{}bcdefg = 1", prefix); // ident continues past byte 64
        let buf = padded(&src);
        let (start_masks, _) = run_stage1(&buf);
        let got = starts_of(&start_masks, buf.len());
        // exactly one word-start (byte 0), then '=' and '1'.
        let eq = src.find('=').unwrap() as i32;
        let one = src.find('1').unwrap() as i32;
        assert_eq!(got, vec![0, eq, one], "no false start at chunk boundary");
    }

    fn tokenize_str(src: &str) -> (Vec<u8>, Vec<Token>) {
        let buf = padded(src);
        let (start_masks, word_masks) = run_stage1(&buf);
        // restrict to real source length
        let real = &buf[..src.len()];
        let toks = tokenize(real, &start_masks, &word_masks);
        (real.to_vec(), toks)
    }

    fn assert_roundtrip(src: &str) -> Vec<Token> {
        let (real, toks) = tokenize_str(src);
        let recon = reconstruct(&real, &toks).expect("reconstruct failed");
        assert_eq!(recon.as_bytes(), &real[..], "round-trip mismatch for: {:?}", src);
        toks
    }

    #[test]
    fn roundtrip_basic() {
        assert_roundtrip("const y = (a + b) * 2;");
    }

    #[test]
    fn count_matches_tokenize() {
        // The fast word-at-a-time count_tokens must agree with the Lexer-based
        // tokenize on every shape (regex, templates, comments, operators).
        for src in [
            "const re = /ab+c/gi; let z = a / b / c;",
            "let q = `outer ${ `inner ${x + 1} done` } end`;",
            "function f() { return 0xFF + 1_000 + 3.14e2; } // c\n/* b */ x>>>=2;",
            "const g = (x) => x === 2 ? x : 0;",
        ] {
            let buf = padded(src);
            let (sm, wm) = run_stage1(&buf);
            let real = &buf[..src.len()];
            let toks = tokenize(real, &sm, &wm);
            let n = count_tokens(real, &sm, &wm);
            assert_eq!(n, toks.len(), "count_tokens vs tokenize mismatch for {src:?}");
        }
    }

    #[test]
    fn roundtrip_strings_and_comments() {
        let src = r#"
            // a line comment with { not structural }
            const s = "he said \"hi\" /* not a comment */";
            const t = 'a string with // not a comment';
            /* block
               comment */ const u = 3;
        "#;
        let toks = assert_roundtrip(src);
        let kinds: Vec<TokKind> = toks.iter().map(|t| t.kind).collect();
        assert!(kinds.contains(&TokKind::LineComment));
        assert!(kinds.contains(&TokKind::BlockComment));
        assert!(kinds.iter().filter(|k| **k == TokKind::String).count() == 2);
    }

    #[test]
    fn roundtrip_template_nesting() {
        let src = "let q = `outer ${ `inner ${x + 1} done` } end`;";
        let toks = assert_roundtrip(src);
        let buf = padded(src);
        // Two nested templates → 2 heads + 2 tails, and the interpolation
        // expressions (`x`, `+`, `1`) are now real tokens (head/middle/tail
        // splitting via the re_lex_template_tail hook).
        let heads = toks.iter().filter(|t| t.kind == TokKind::TemplateHead).count();
        let tails = toks.iter().filter(|t| t.kind == TokKind::TemplateTail).count();
        assert_eq!((heads, tails), (2, 2), "nested template head/tail count");
        let texts: Vec<&str> = toks.iter().map(|t| t.text(&buf[..src.len()])).collect();
        assert!(texts.contains(&"x") && texts.contains(&"+") && texts.contains(&"1"),
            "interpolation expression is tokenized: {texts:?}");
    }

    #[test]
    fn regex_vs_division() {
        // first `/.../ ` is a regex (after `=`); the `/` in `a / b` is division.
        let src = "const re = /ab+c/gi; const z = a / b / c;";
        let toks = assert_roundtrip(src);
        let regexes: Vec<&Token> = toks.iter().filter(|t| t.kind == TokKind::Regex).collect();
        assert_eq!(regexes.len(), 1, "exactly one regex literal");
        assert_eq!(regexes[0].text(&padded(src)[..src.len()]), "/ab+c/gi");
        let divides = toks
            .iter()
            .filter(|t| t.kind == TokKind::Punct && t.text(&padded(src)[..src.len()]) == "/")
            .count();
        assert_eq!(divides, 2, "two division operators");
    }

    #[test]
    fn classifies_numbers_keywords_idents() {
        let src = "function f() { return 0xFF + 1_000 + 3.14e2 + 9n; }";
        let toks = assert_roundtrip(src);
        let buf = padded(src);
        let texts: Vec<&str> = toks.iter().map(|t| t.text(&buf[..src.len()])).collect();
        assert!(texts.contains(&"function"));
        assert!(texts.contains(&"return"));
        assert!(texts.contains(&"0xFF"));
        assert!(texts.contains(&"1_000"));
        assert!(texts.contains(&"3.14e2"));
        assert!(texts.contains(&"9n"));
        // multi-char operator longest-match
        let arrow_src = "const g = (x) => x === 2 ? x : 0;";
        let at = assert_roundtrip(arrow_src);
        let abuf = padded(arrow_src);
        let atexts: Vec<&str> = at.iter().map(|t| t.text(&abuf[..arrow_src.len()])).collect();
        assert!(atexts.contains(&"=>"), "arrow operator");
        assert!(atexts.contains(&"==="), "strict-equals longest match");
    }
}
