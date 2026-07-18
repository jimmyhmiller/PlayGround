//! A Clojure reader: text -> data. Distinguishes the collection literals the
//! core toolkit reader cannot — `[]` vectors, `{}` maps, `#{}` sets — plus
//! `:keywords`, `"strings"`, `\chars`, and `'quote`. Everything is built as a
//! toolkit heap value (list / `Obj::Vector` / tagged `Obj::Record`), so the GC
//! traces it and `analyze` self-evaluates the literal ones.
//!
//! Collection literals are DATA, exactly as in Clojure: `[a b]` reads to the
//! real runtime persistent vector holding the raw element forms (see `data`),
//! so macros receive actual collections. The frontend's `expand` turns one in
//! expression position into a constructor call (evaluating elements) and
//! leaves it literal under `quote`. Keywords/strings/chars/numbers are
//! self-evaluating (non-cons refs → `Ir::Const`).

use crate::data;
use microlang::runtime::ObjView;
use microlang::value::Sym;
use microlang::{Obj, Repr, Runtime, Val, ValueModel};

/// The type tag for keywords, which the frontend builds on `Obj::Record` (a
/// record holding the interned name symbol). `type-of` reports this tag, so
/// `clojure.core` predicates dispatch on it. Keywords have always had ONE
/// representation shared by reader and runtime; vectors/maps/sets get the same
/// treatment via `data` (the reader constructs the runtime collection itself).
pub const KEYWORD: &str = "Keyword";
/// The unresolved `::foo` marker (field 0 = the `foo` / `alias/foo` name sym);
/// resolved to a plain `Keyword` per the current namespace at eval time.
pub const KEYWORD_AUTO_NS: &str = "KeywordAutoNs";

pub fn read_all<M: ValueModel>(rt: &mut Runtime<M>, src: &str) -> Vec<u64> {
    let toks = tokenize(src);
    let mut p = Parser { toks, pos: 0 };
    let mut forms = Vec::new();
    while p.pos < p.toks.len() {
        // A TOP-LEVEL `#_` discards the next form and yields nothing — it does not
        // stand for the form after it. `Parser::form` cannot express "no value", so
        // the skip has to happen here, exactly as the collection reader does it.
        // Handling it there meant a trailing `#_(…)` at EOF read off the end of the
        // token stream and panicked (core.async's mutex.clj ends with one).
        if p.toks[p.pos] == Tok::Discard {
            p.pos += 1;
            p.form(rt); // read & drop
            continue;
        }
        forms.push(p.form(rt));
    }
    forms
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Open(char),  // ( [ {
    Close(char), // ) ] }
    HashBrace,   // #{
    HashParen,   // #(  anonymous-fn literal
    ReaderCond,  // #?  reader conditional (followed by a `(...)`)
    ReaderCondSplice, // #?@  splicing reader conditional (splices a collection)
    Discard,     // #_  discard the next form
    Regex(String), // #"pat"  regex literal -> (re-pattern "pat")
    UuidLit,     // #uuid "…"  -> (uuid "…")
    SymbolicVal(f64), // ##Inf / ##-Inf / ##NaN
    VarQuote,    // #'  -> (var x)
    Quote,
    Backtick,      // `  syntax-quote
    Unquote,       // ~
    UnquoteSplice, // ~@
    Deref,         // @  -> (deref x)
    Caret,         // ^  metadata (parsed, then discarded for now)
    Str(String),
    Char(u32),
    Atom(String), // symbol / keyword / number / nil / true / false
}

fn tokenize(src: &str) -> Vec<Tok> {
    let cs: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < cs.len() {
        let c = cs[i];
        match c {
            c if c.is_whitespace() || c == ',' => i += 1, // commas are whitespace in Clojure
            ';' => {
                while i < cs.len() && cs[i] != '\n' {
                    i += 1;
                }
            }
            '(' | '[' | '{' => {
                out.push(Tok::Open(c));
                i += 1;
            }
            ')' | ']' | '}' => {
                out.push(Tok::Close(c));
                i += 1;
            }
            '#' if i + 1 < cs.len() && cs[i + 1] == '{' => {
                out.push(Tok::HashBrace);
                i += 2;
            }
            // `#_` — discard the next form entirely.
            '#' if i + 1 < cs.len() && cs[i + 1] == '_' => {
                out.push(Tok::Discard);
                i += 2;
            }
            // `#uuid "…"` — a UUID tagged literal -> (uuid "…").
            '#' if cs[i + 1..].starts_with(&['u', 'u', 'i', 'd']) => {
                out.push(Tok::UuidLit);
                i += 5;
            }
            // `#"pat"` — a regex literal; read the pattern RAW (backslashes kept).
            '#' if i + 1 < cs.len() && cs[i + 1] == '"' => {
                i += 2;
                let mut s = String::new();
                while i < cs.len() && cs[i] != '"' {
                    if cs[i] == '\\' && i + 1 < cs.len() {
                        s.push(cs[i]);
                        s.push(cs[i + 1]);
                        i += 2;
                    } else {
                        s.push(cs[i]);
                        i += 1;
                    }
                }
                i += 1; // closing quote
                out.push(Tok::Regex(s));
            }
            '#' if i + 1 < cs.len() && cs[i + 1] == '(' => {
                // `#(...)` anonymous-fn literal: the `(` is consumed here so the
                // body reads as a normal parenthesized form under `HashParen`.
                out.push(Tok::HashParen);
                out.push(Tok::Open('('));
                i += 2;
            }
            // `#^{...}` / `#^Foo` — the old metadata reader macro, identical to `^`.
            '#' if i + 1 < cs.len() && cs[i + 1] == '^' => {
                out.push(Tok::Caret);
                i += 2;
            }
            // `#'x` — var-quote, sugar for `(var x)`.
            '#' if i + 1 < cs.len() && cs[i + 1] == '\'' => {
                out.push(Tok::VarQuote);
                i += 2;
            }
            // `##Inf` / `##-Inf` / `##NaN` — the symbolic-value literals. Real
            // libraries use them: meander compares an overflowing `Math/pow`
            // against `##Inf` to decide whether a search space is finite.
            '#' if i + 1 < cs.len() && cs[i + 1] == '#' => {
                i += 2;
                let start = i;
                while i < cs.len()
                    && !cs[i].is_whitespace()
                    && !matches!(cs[i], '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';')
                {
                    i += 1;
                }
                let name: String = cs[start..i].iter().collect();
                match name.as_str() {
                    "Inf" => out.push(Tok::SymbolicVal(f64::INFINITY)),
                    "-Inf" => out.push(Tok::SymbolicVal(f64::NEG_INFINITY)),
                    "NaN" => out.push(Tok::SymbolicVal(f64::NAN)),
                    // Clojure rejects any other `##name` outright; so do we,
                    // rather than read it as some other value.
                    other => panic!("reader: unknown symbolic value ##{other}"),
                }
            }
            // `#?@(...)` splicing reader conditional — splice the chosen collection.
            '#' if i + 2 < cs.len() && cs[i + 1] == '?' && cs[i + 2] == '@' => {
                out.push(Tok::ReaderCondSplice);
                i += 3; // leave the `(` to tokenize as a normal Open
            }
            // `#?(...)` reader conditional.
            '#' if i + 1 < cs.len() && cs[i + 1] == '?' => {
                out.push(Tok::ReaderCond);
                i += 2; // leave the `(` to tokenize as a normal Open
            }
            '\'' => {
                out.push(Tok::Quote);
                i += 1;
            }
            '^' => {
                out.push(Tok::Caret);
                i += 1;
            }
            '`' => {
                out.push(Tok::Backtick);
                i += 1;
            }
            '@' => {
                out.push(Tok::Deref);
                i += 1;
            }
            '~' => {
                if i + 1 < cs.len() && cs[i + 1] == '@' {
                    out.push(Tok::UnquoteSplice);
                    i += 2;
                } else {
                    out.push(Tok::Unquote);
                    i += 1;
                }
            }
            '"' => {
                i += 1;
                let mut s = String::new();
                while i < cs.len() && cs[i] != '"' {
                    if cs[i] == '\\' && i + 1 < cs.len() {
                        i += 1;
                        s.push(match cs[i] {
                            'n' => '\n',
                            't' => '\t',
                            'r' => '\r',
                            other => other,
                        });
                    } else {
                        s.push(cs[i]);
                    }
                    i += 1;
                }
                i += 1; // closing quote
                out.push(Tok::Str(s));
            }
            '\\' => {
                // a character literal: \a, \newline, \space, \tab, and the
                // delimiter chars \( \) \[ \] \{ \} (Clojure allows these).
                i += 1;
                if i < cs.len() && matches!(cs[i], '(' | ')' | '[' | ']' | '{' | '}') {
                    out.push(Tok::Char(cs[i] as u32));
                    i += 1;
                } else {
                    let start = i;
                    while i < cs.len() && !cs[i].is_whitespace() && !matches!(cs[i], '(' | ')' | '[' | ']' | '{' | '}') {
                        i += 1;
                    }
                    let name: String = cs[start..i].iter().collect();
                    let ch: u32 = match name.as_str() {
                        "newline" => '\n' as u32,
                        "space" => ' ' as u32,
                        "tab" => '\t' as u32,
                        "return" => '\r' as u32,
                        "backspace" => 0x0008,
                        "formfeed" => 0x000c,
                        // \uXXXX (4 hex) and \oNNN (up to 3 octal) numeric escapes,
                        // exactly as Clojure's reader. `\uD83D` — a lone
                        // surrogate half — is a LEGAL char literal (chars are
                        // UTF-16 units, as on the JVM). A single leading
                        // `u`/`o` is still the literal char (`\u`, `\o`).
                        _ if name.len() > 1
                            && name.starts_with('u')
                            && name[1..].chars().all(|c| c.is_ascii_hexdigit()) =>
                        {
                            u32::from_str_radix(&name[1..], 16).unwrap_or(' ' as u32)
                        }
                        _ if name.len() > 1
                            && name.starts_with('o')
                            && name[1..].chars().all(|c| ('0'..='7').contains(&c)) =>
                        {
                            u32::from_str_radix(&name[1..], 8).unwrap_or(' ' as u32)
                        }
                        _ => name.chars().next().map(|c| c as u32).unwrap_or(' ' as u32),
                    };
                    out.push(Tok::Char(ch));
                }
            }
            _ => {
                let start = i;
                while i < cs.len()
                    && !cs[i].is_whitespace()
                    // `'` is allowed INSIDE a symbol (`+'`, `inc'`) — a LEADING quote
                    // is handled by the `'` token case before this branch is reached.
                    && !matches!(cs[i], '(' | ')' | '[' | ']' | '{' | '}' | '"' | ';' | ',' | '`' | '~' | '^')
                {
                    i += 1;
                }
                out.push(Tok::Atom(cs[start..i].iter().collect()));
            }
        }
    }
    out
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn form<M: ValueModel>(&mut self, rt: &mut Runtime<M>) -> u64 {
        let tok = self.toks[self.pos].clone();
        self.pos += 1;
        match tok {
            Tok::Open('(') => {
                let items = self.until(rt, ')');
                if items.is_empty() {
                    // `()` is the empty list — a distinct value from nil.
                    rt.enc_empty_list()
                } else {
                    rt.vec_to_list(&items)
                }
            }
            Tok::Open('[') => {
                let items = self.until(rt, ']');
                data::make_vector(rt, &items)
            }
            Tok::Open('{') => {
                let items = self.until(rt, '}');
                data::make_map(rt, &items)
            }
            Tok::HashBrace => {
                let items = self.until(rt, '}');
                data::make_set(rt, &items)
            }
            Tok::HashParen => {
                // `#(...)` -> `(fn [%1 %2 … & %&] (...))`. Read the body (the
                // following `(...)` form), scan it for the implicit params `%`,
                // `%1`…`%N`, `%&`, then synthesize the param vector. Bare `%` is
                // an alias for `%1`, so it is rewritten before emission.
                let mut body = self.form(rt);
                let mut max_n = 0usize;
                let mut has_rest = false;
                let mut has_bare = false;
                scan_pct(rt, body, &mut max_n, &mut has_rest, &mut has_bare);
                if has_bare {
                    if max_n < 1 {
                        max_n = 1;
                    }
                    let pct1 = sym(rt, "%1");
                    body = rewrite_bare_pct(rt, body, pct1);
                }
                let mut params = Vec::new();
                for i in 1..=max_n {
                    params.push(sym(rt, &format!("%{i}")));
                }
                if has_rest {
                    params.push(sym(rt, "&"));
                    params.push(sym(rt, "%&"));
                }
                let pvec = data::make_vector(rt, &params);
                let fnsym = sym(rt, "fn");
                rt.vec_to_list(&[fnsym, pvec, body])
            }
            Tok::Quote => self.wrap(rt, "quote"),
            Tok::VarQuote => self.wrap(rt, "var"),
            Tok::Caret => {
                // ^meta target : we discard metadata for now, EXCEPT `:macro`
                // (as `^{:macro true}` or `^:macro`), which real core.clj uses to
                // define macros — wrap it so `def` can register the macro.
                let meta = self.form(rt);
                let target = self.form(rt);
                if is_macro_meta(rt, meta) {
                    let m = sym(rt, "-macro-meta");
                    rt.vec_to_list(&[m, target])
                } else if is_dynamic_meta(rt, meta) {
                    // `^:dynamic name` -> `(-dynamic-meta name)`; the compiler's
                    // `def` marks the var dynamic so refs read the binding stack.
                    let m = sym(rt, "-dynamic-meta");
                    rt.vec_to_list(&[m, target])
                } else if meta_has_key(rt, meta, "private") {
                    // `^:private name` -> `(-private-meta name)`; the compiler marks
                    // the var private (cross-namespace access errors).
                    let m = sym(rt, "-private-meta");
                    rt.vec_to_list(&[m, target])
                } else if meta_has_key(rt, meta, "unsynchronized-mutable")
                    || meta_has_key(rt, meta, "volatile-mutable")
                    || meta_has_key(rt, meta, "mutable")
                {
                    // `^:unsynchronized-mutable field` (a `deftype` mutable field) ->
                    // `(-mutable-field field)`, so `deftype` compiles `set!` on it to
                    // a real record-slot write (a stateful deftype — data.json's
                    // StringPBR reader — depends on the mutation persisting).
                    let m = sym(rt, "-mutable-field");
                    rt.vec_to_list(&[m, target])
                } else {
                    target
                }
            }
            Tok::ReaderCond => {
                // `#?(:platform form :platform form …)`. This dialect models the JVM, so it
                // selects the `:clj` branch — see `select_reader_cond`. (It used to prefer
                // `:cljs`, from before the JVM layer existed.) All branches are READ
                // (harmless); only the selected one is kept.
                
                match self.toks.get(self.pos) {
                    Some(Tok::Open('(')) => self.pos += 1,
                    _ => panic!("reader: #? must be followed by `(`"),
                }
                let items = self.until(rt, ')');
                self.select_reader_cond(rt, &items)
            }
            Tok::Backtick => self.wrap(rt, "syntax-quote"),
            Tok::Deref => self.wrap(rt, "deref"),
            Tok::Unquote => self.wrap(rt, "unquote"),
            Tok::UnquoteSplice => self.wrap(rt, "unquote-splice"),
            Tok::Str(s) => alloc(rt, Obj::Str(s)),
            // `#"pat"` -> `(re-pattern "pat")`, built at runtime into a Regex value.
            Tok::Regex(p) => {
                let rp = sym(rt, "re-pattern");
                let pat = alloc(rt, Obj::Str(p));
                rt.vec_to_list(&[rp, pat])
            }
            // `#uuid "…"` -> `(uuid "…")`.
            Tok::UuidLit => self.wrap(rt, "uuid"),
            Tok::Char(c) => alloc(rt, Obj::Char(c)),
            Tok::SymbolicVal(f) => rt.encode(Val::Float(f)),
            Tok::Atom(a) => self.atom(rt, &a),
            // `#?@` outside a collection is unusual; return the selected collection
            // as a value (it can only be meaningfully spliced inside `until`).
            Tok::ReaderCondSplice => {
                let elems = self.read_cond_splice(rt);
                rt.vec_to_list(&elems)
            }
            // `#_` in a VALUE position: the discarded form is dropped and the value
            // is the form after it (`(f #_a b)` passes b). Top level and collection
            // elements handle their own skipping — there a discard yields nothing at
            // all, and there may be no following form.
            Tok::Discard => {
                self.form(rt); // read & drop the discarded form
                self.form(rt)
            }
            Tok::Open(c) | Tok::Close(c) => panic!("reader: unexpected {c}"),
        }
    }

    /// Read a `#?@(:platform coll …)` and return the SPLICED elements of the chosen
    /// collection (its members), for the caller to inline into the enclosing seq.
    fn read_cond_splice<M: ValueModel>(&mut self, rt: &mut Runtime<M>) -> Vec<u64> {
        match self.toks.get(self.pos) {
            Some(Tok::Open('(')) => self.pos += 1,
            _ => panic!("reader: #?@ must be followed by `(`"),
        }
        let branches = self.until(rt, ')');
        let selected = self.select_reader_cond(rt, &branches);
        splice_elements(rt, selected)
    }

    /// Pick a reader-conditional branch from `[kw form kw form …]`. Priority:
    /// `:cljs` (JVM-free, like us), then `:default`, then `:clj`; if no platform
    /// matches, the form vanishes (reads as `nil`).
    /// `:clj` first, then `:default` — the platform order JVM Clojure itself
    /// uses.
    ///
    /// This preferred `:cljs` back when the dialect had no JVM layer. It has
    /// one now (`java.lang.System`, `clojure.lang.*`, `defclass`), it resolves
    /// `deps.edn` Maven coordinates, and the goal is a drop-in for JVM Clojure
    /// — so a `.cljc` library's `:cljs` branch is the wrong half: it targets a
    /// host we are not (`js/…`, `goog`, `cljs.pprint`). meander/epsilon fails
    /// to load for exactly that reason: its `:cljs` branch requires
    /// `cljs.pprint`.
    ///
    /// `:cljs` is NOT in the list at all: selecting it would mean claiming to be
    /// a JavaScript host, and a library that offers only a `:cljs` branch has
    /// nothing to run here.
    fn select_reader_cond<M: ValueModel>(&mut self, rt: &mut Runtime<M>, items: &[u64]) -> u64 {
        for pref in ["clj", "default"] {
            let mut i = 0;
            while i + 1 < items.len() {
                if kw_name(rt, items[i]).as_deref() == Some(pref) {
                    return items[i + 1];
                }
                i += 2;
            }
        }
        rt.encode(Val::Nil)
    }

    /// `(head <next-form>)` — for reader macros `'`, `` ` ``, `~`, `~@`.
    fn wrap<M: ValueModel>(&mut self, rt: &mut Runtime<M>, head: &str) -> u64 {
        let inner = self.form(rt);
        let h = sym(rt, head);
        rt.vec_to_list(&[h, inner])
    }

    fn until<M: ValueModel>(&mut self, rt: &mut Runtime<M>, close: char) -> Vec<u64> {
        let mut items = Vec::new();
        loop {
            match self.toks.get(self.pos) {
                Some(Tok::Close(c)) if *c == close => {
                    self.pos += 1;
                    return items;
                }
                Some(Tok::ReaderCondSplice) => {
                    self.pos += 1;
                    let elems = self.read_cond_splice(rt);
                    items.extend(elems);
                }
                // `#_ form` inside a collection: read and drop the next form.
                Some(Tok::Discard) => {
                    self.pos += 1;
                    self.form(rt);
                }
                Some(_) => items.push(self.form(rt)),
                None => panic!("reader: unbalanced, expected {close}"),
            }
        }
    }

    fn atom<M: ValueModel>(&mut self, rt: &mut Runtime<M>, a: &str) -> u64 {
        if a == "nil" {
            return rt.encode(Val::Nil);
        }
        if a == "true" {
            return rt.encode(Val::Bool(true));
        }
        if a == "false" {
            return rt.encode(Val::Bool(false));
        }
        // Radix forms FIRST: `0123` is octal 83 on the JVM (and in tools.reader),
        // and a plain decimal parse would silently read it as 123.
        if let Some(v) = radix_int(rt, a) {
            return v;
        }
        if let Ok(i) = a.parse::<i128>() {
            return rt.encode(Val::Int(i));
        }
        // `123N` — a (big)integer literal. This tower auto-promotes, so it is just
        // the integer; a value beyond i128 becomes a boxed arbitrary-precision int.
        if let Some(digits) = a.strip_suffix('N') {
            if let Ok(i) = digits.parse::<i128>() {
                return rt.encode(Val::Int(i));
            }
            if let Some(b) = microlang::bigint::BigInt::from_str(digits) {
                return rt.alloc_bigint(b);
            }
        }
        // `n/d` — a ratio literal (both parts numeric; `foo/bar` stays a symbol).
        if let Some((np, dp)) = a.split_once('/') {
            if let (Ok(n), Ok(d)) = (np.parse::<i128>(), dp.parse::<i128>()) {
                if d != 0 {
                    return rt.make_ratio(n, d);
                }
            }
        }
        // Only tokens that LOOK numeric ([+-]?digit…) may parse as floats:
        // Rust's f64 parser accepts "NaN"/"inf"/"Infinity", which are ordinary
        // SYMBOLS in Clojure (test.check defs a private named NaN; reading it
        // as a float made the def unresolvable — and NaN ≠ NaN, so it broke
        // every lookup it touched). ##NaN/##Inf are separate reader literals.
        let numericish = {
            let b = a.as_bytes();
            let d = if b.first().is_some_and(|c| matches!(c, b'+' | b'-')) { 1 } else { 0 };
            b.get(d).is_some_and(|c| c.is_ascii_digit())
        };
        if numericish {
            if let Ok(f) = a.parse::<f64>() {
                return rt.encode(Val::Float(f));
            }
        }
        // `::foo` / `::alias/foo` — an AUTO-NAMESPACED keyword. The reader can't
        // know the current namespace (forms are read before the preceding `ns`
        // forms evaluate), so it emits a marker record that `eval_form` resolves
        // against the compiler's namespace state just before evaluation.
        if let Some(kw) = a.strip_prefix("::") {
            let name = rt.intern(kw);
            let name_v = rt.encode(Val::Sym(name));
            return record(rt, KEYWORD_AUTO_NS, vec![name_v]);
        }
        if let Some(kw) = a.strip_prefix(':') {
            // `:foo` -> THE canonical keyword object for that name. Interning
            // here (rather than building a record per token) is what makes
            // `(identical? :a :a)` true, as in Clojure and ClojureScript: two
            // occurrences of `:foo` are the same object, not two records that
            // merely compare equal.
            let name = rt.intern(kw);
            return rt.intern_keyword(name);
        }
        sym(rt, a)
    }
}

fn sym<M: ValueModel>(rt: &mut Runtime<M>, n: &str) -> u64 {
    let s = rt.intern(n);
    rt.encode(Val::Sym(s))
}

/// The JVM reader's integer-literal grammar beyond plain decimal: `0x…` hex,
/// leading-`0` octal, `NrDIGITS` arbitrary radix 2-36 (`2r1010`, `36rZZ`),
/// each with an optional sign and optional `N` suffix. Returns None when `a`
/// is not such a literal (it then falls through to decimal/float/symbol).
/// A value past i128 accumulates into a big integer, matching the JVM, which
/// promotes any out-of-long-range literal (`0xbf58476d1ce4e5b9` reads as
/// 13791443137822795193N — test.check's `longify` depends on exactly that).
fn radix_int<M: ValueModel>(rt: &mut Runtime<M>, a: &str) -> Option<u64> {
    let (neg, body) = match a.as_bytes().first()? {
        b'-' => (true, &a[1..]),
        b'+' => (false, &a[1..]),
        _ => (false, a),
    };
    let body = body.strip_suffix('N').unwrap_or(body);
    let (radix, digits) = if let Some(h) =
        body.strip_prefix("0x").or_else(|| body.strip_prefix("0X"))
    {
        (16u32, h)
    } else if body.len() > 1 && body.starts_with('0') && body[1..].bytes().all(|b| b.is_ascii_digit())
    {
        (8u32, &body[1..])
    } else if let Some((rp, dp)) = body.split_once(['r', 'R']) {
        let r: u32 = rp.parse().ok()?;
        if !(2..=36).contains(&r) {
            return None;
        }
        (r, dp)
    } else {
        return None;
    };
    if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
        return None;
    }
    if let Ok(v) = i128::from_str_radix(digits, radix) {
        return Some(rt.encode(Val::Int(if neg { -v } else { v })));
    }
    let mut b = microlang::bigint::BigInt::from_i128(0);
    let rad = microlang::bigint::BigInt::from_i128(radix as i128);
    for c in digits.chars() {
        let d = c.to_digit(radix).expect("digit validated above") as i128;
        b = b.mul(&rad).add(&microlang::bigint::BigInt::from_i128(d));
    }
    if neg {
        b = b.neg();
    }
    Some(rt.alloc_bigint(b))
}

fn field0<M: ValueModel>(rt: &Runtime<M>, v: u64, tag: &str) -> Option<u64> {
    if let Val::Ref(id) = rt.decode(v) {
        if let ObjView::Record { type_id, fields } = rt.view_gc(id) {
            if rt.sym_name(type_id) == tag {
                return fields.first().copied();
            }
        }
    }
    None
}

/// Is `meta` a macro marker — `^:macro` / `^{:macro true}`?
fn is_macro_meta<M: ValueModel>(rt: &Runtime<M>, meta: u64) -> bool {
    meta_has_key(rt, meta, "macro")
}

/// Is `meta` a `^:dynamic` marker — `^:dynamic` / `^{:dynamic true}`?
fn is_dynamic_meta<M: ValueModel>(rt: &Runtime<M>, meta: u64) -> bool {
    meta_has_key(rt, meta, "dynamic")
}

/// Does the reader metadata form carry keyword `key` — either as the bare
/// keyword `^:key` or as a `^{:key true}` map entry?
fn meta_has_key<M: ValueModel>(rt: &Runtime<M>, meta: u64, key: &str) -> bool {
    if let Some(name) = field0(rt, meta, KEYWORD) {
        return matches!(rt.decode(name), Val::Sym(s) if rt.sym_name(s) == key);
    }
    if let Some(kvs) = data::map_entries(rt, meta) {
        let mut i = 0;
        while i + 1 < kvs.len() {
            if let Some(kname) = field0(rt, kvs[i], KEYWORD) {
                if matches!(rt.decode(kname), Val::Sym(s) if rt.sym_name(s) == key) {
                    return true;
                }
            }
            i += 2;
        }
    }
    false
}

fn alloc<M: ValueModel>(rt: &mut Runtime<M>, o: Obj) -> u64 {
    let id = rt.alloc(o);
    <M::R as Repr>::enc_ref(id)
}

fn record<M: ValueModel>(rt: &mut Runtime<M>, ty: &str, fields: Vec<u64>) -> u64 {
    let type_id: Sym = rt.intern(ty);
    let id = rt.alloc_record(type_id, &fields);
    <M::R as Repr>::enc_ref(id)
}

/// The members of a `#?@`-selected collection (a `[..]` vector or a `(..)`
/// list), to splice into the enclosing seq. `nil` (no branch matched) -> empty.
fn splice_elements<M: ValueModel>(rt: &Runtime<M>, form: u64) -> Vec<u64> {
    if let Some(items) = data::vector_items(rt, form) {
        return items;
    }
    // a cons list (or nil): list_to_vec handles both (nil -> empty).
    rt.list_to_vec(form)
}

/// The name of a `:keyword` value (a `Keyword` record), or `None` for anything else.
fn kw_name<M: ValueModel>(rt: &Runtime<M>, v: u64) -> Option<String> {
    if let Val::Ref(id) = rt.decode(v) {
        if let ObjView::Record { type_id, fields } = rt.view_gc(id) {
            if rt.sym_name(type_id) == KEYWORD {
                if let Some(&f0) = fields.first() {
                    if let Val::Sym(s) = rt.decode(f0) {
                        return Some(rt.sym_name(s).to_string());
                    }
                }
            }
        }
    }
    None
}

/// Walk a `#(...)` body form (cons lists, record contents, and raw arrays —
/// collection tries hold their element forms in `Obj::Vector` leaves)
/// collecting the implicit anonymous-fn params it mentions: the largest `%N`,
/// whether `%&` (rest) appears, and whether the bare `%` appears. Purely
/// inspects — shared borrows only.
fn scan_pct<M: ValueModel>(
    rt: &Runtime<M>,
    form: u64,
    max_n: &mut usize,
    has_rest: &mut bool,
    has_bare: &mut bool,
) {
    match rt.decode(form) {
        Val::Sym(s) => {
            let name = rt.sym_name(s);
            if name == "%&" {
                *has_rest = true;
            } else if name == "%" {
                *has_bare = true;
            } else if let Some(rest) = name.strip_prefix('%') {
                if let Ok(n) = rest.parse::<usize>() {
                    if n > *max_n {
                        *max_n = n;
                    }
                }
            }
        }
        Val::Ref(id) => {
            let (head, tail, fields) = match rt.view_gc(id) {
                ObjView::Cons { head, tail } => (Some(head), Some(tail), Vec::new()),
                ObjView::Record { fields, .. } => (None, None, fields.to_vec()),
                ObjView::Vector { elems, .. } => (None, None, elems.to_vec()),
                _ => (None, None, Vec::new()),
            };
            if let Some(h) = head {
                scan_pct(rt, h, max_n, has_rest, has_bare);
            }
            if let Some(t) = tail {
                scan_pct(rt, t, max_n, has_rest, has_bare);
            }
            for f in fields {
                scan_pct(rt, f, max_n, has_rest, has_bare);
            }
        }
        _ => {}
    }
}

/// Rebuild `form`, replacing every bare `%` symbol with `pct1` (the `%1` symbol).
/// A node whose children are all UNCHANGED is returned as itself (structural
/// sharing) — copying unconditionally minted duplicate objects for every datum
/// in the body, including 'Keyword records, which broke keyword canonicality
/// (`identical?`). Keyword records are never descended into at all: their one
/// field is a name SYM, not code — a keyword literally named `:%` must survive.
fn rewrite_bare_pct<M: ValueModel>(rt: &mut Runtime<M>, form: u64, pct1: u64) -> u64 {
    match rt.decode(form) {
        Val::Sym(s) if rt.sym_name(s) == "%" => pct1,
        Val::Ref(id) => {
            // Copy the shape out of the view first: the recursive calls below
            // need `&mut rt`, which a live borrow from `view_gc` would block.
            enum Shape {
                Cons(u64, u64),
                Record(Sym, Vec<u64>),
                Vector(Vec<u64>),
                Other,
            }
            let shape = match rt.view_gc(id) {
                ObjView::Cons { head, tail } => Shape::Cons(head, tail),
                ObjView::Record { type_id, fields } => Shape::Record(type_id, fields.to_vec()),
                ObjView::Vector { elems, .. } => Shape::Vector(elems.to_vec()),
                _ => Shape::Other,
            };
            match shape {
                Shape::Cons(head, tail) => {
                    let h = rewrite_bare_pct(rt, head, pct1);
                    let t = rewrite_bare_pct(rt, tail, pct1);
                    if h == head && t == tail {
                        form
                    } else {
                        alloc(rt, Obj::Cons { head: h, tail: t })
                    }
                }
                Shape::Record(type_id, fields) => {
                    if rt.sym_name(type_id) == KEYWORD {
                        return form; // a keyword's field is its NAME, not code
                    }
                    let nf: Vec<u64> =
                        fields.iter().map(|&f| rewrite_bare_pct(rt, f, pct1)).collect();
                    if nf == fields {
                        return form;
                    }
                    let rid = rt.alloc_record(type_id, &nf);
                    <M::R as Repr>::enc_ref(rid)
                }
                Shape::Vector(elems) => {
                    let ne: Vec<u64> =
                        elems.iter().map(|&e| rewrite_bare_pct(rt, e, pct1)).collect();
                    if ne == elems {
                        return form;
                    }
                    let rid = rt.alloc_vector(&ne);
                    <M::R as Repr>::enc_ref(rid)
                }
                Shape::Other => form,
            }
        }
        _ => form,
    }
}
