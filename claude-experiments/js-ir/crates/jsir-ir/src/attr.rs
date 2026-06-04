//! IR attributes and their byte-exact textual rendering.

use std::any::Any;
use std::sync::Arc;

/// A dialect-defined attribute payload — the generic escape hatch by which any
/// dialect attaches arbitrary structured state to an op without the core IR
/// knowing its schema. jsir-ir only ever stores it, clones it (an `Arc` bump),
/// prints it (`render`), and compares it (`dyn_eq`); it never interprets the
/// contents. A dialect recovers its concrete type via `as_any().downcast_ref`.
///
/// This is how MLIR lets dialects hang their own data on operations. All
/// React/JSLIR analysis state (mutable ranges, aliasing effects, reactive scopes,
/// identifier/place metadata) will ride here as concrete dialect types; the core
/// stays dialect-agnostic.
pub trait DialectPayload: std::fmt::Debug + Send + Sync {
    /// Textual form for the printer (conventionally `#dialect<...>`).
    fn render(&self) -> String;
    /// Downcast hook so the owning dialect can recover its concrete type.
    fn as_any(&self) -> &dyn Any;
    /// Structural equality against another payload (typically: downcast `other`
    /// to `Self` and compare). Lets [`Attr`] keep a derived `PartialEq`.
    fn dyn_eq(&self, other: &dyn DialectPayload) -> bool;
}

/// Newtype wrapper around an [`DialectPayload`] so [`Attr`] keeps its derived
/// `Clone`/`Debug`/`PartialEq` (a bare `Arc<dyn _>` is neither `Debug`-derivable
/// in the enum nor `PartialEq`). Cheap to clone (shared `Arc`).
#[derive(Clone)]
pub struct OpaqueAttr(pub Arc<dyn DialectPayload>);

impl OpaqueAttr {
    pub fn new<T: DialectPayload + 'static>(payload: T) -> Self {
        OpaqueAttr(Arc::new(payload))
    }
    /// Recover the concrete dialect type, if this payload is a `T`.
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.0.as_any().downcast_ref::<T>()
    }
}

impl std::fmt::Debug for OpaqueAttr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl PartialEq for OpaqueAttr {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || self.0.dyn_eq(other.0.as_ref())
    }
}

/// An MLIR/JSIR attribute value.
#[derive(Debug, Clone, PartialEq)]
pub enum Attr {
    /// A dialect-defined opaque payload (see [`DialectPayload`]). The escape hatch
    /// for CFG/JSLIR dialect state; the core never interprets it.
    Opaque(OpaqueAttr),
    /// A string attribute, e.g. `name = "a"` or `operator_ = "+"`.
    Str(String),
    Bool(bool),
    /// A 64-bit float attribute, printed `... : f64` (e.g. `1.000000e+00 : f64`).
    F64(f64),
    /// A signed integer attribute (printed plain).
    I64(i64),
    /// An array attribute, e.g. `comments = []` or `[#jsir<...>, ...]`.
    Array(Vec<Attr>),
    /// A dense `i32` array attribute, e.g. `operandSegmentSizes = array<i32: 3, 2>`.
    I32Array(Vec<i32>),
    /// `#jsir<numeric_literal_extra "<raw>", <value> : f64>`.
    NumericLiteralExtra { raw: String, value: f64 },
    /// `#jsir<string_literal_extra "<raw>", "<raw_value>">`.
    StringLiteralExtra { raw: String, raw_value: String },
    /// `#jsir<big_int_literal_extra "<raw>", "<raw_value>">`.
    BigIntLiteralExtra { raw: String, raw_value: String },
    /// `#jsir<reg_exp_literal_extra "<raw>">`.
    RegExpLiteralExtra { raw: String },
    /// `#jsir<identifier   <L sl C sc>, <L el C ec>, "<ident_name>", <si>, <ei>, <scope>, "<name>">`
    /// (the loc-present, no-comment/symbol form used for identifiers carried as
    /// op attributes: member properties, function/class ids, object keys, ...).
    Identifier(Box<IdentifierAttr>),
    /// A `JsirStringLiteralAttr` (full, with loc) used as an object/class key:
    /// `#jsir<string_literal   <L..>, <L..>, si, ei, scope, "value",  "raw", "rawValue">`.
    StringLiteralKey(Box<StringLiteralKeyAttr>),
    /// A `JsirNumericLiteralAttr` used as an object/class key:
    /// `#jsir<numeric_literal   <L..>, <L..>, si, ei, scope, v : f64,  "raw", rv : f64>`.
    NumericLiteralKey(Box<NumericLiteralKeyAttr>),
    /// A comment, `#jsir<comment_line  <L..>, <L..>, si, ei, "value">` (or
    /// `comment_block`). `block` selects the mnemonic.
    Comment(Box<CommentAttr>),
    /// `#jsir<directive_literal_extra "<raw>", "<raw_value>">`.
    DirectiveLiteralExtra { raw: String, raw_value: String },
    /// `#jsir<interpreter_directive   <L..>, <L..>, si, ei, "value">`.
    InterpreterDirective(Box<InterpreterDirectiveAttr>),
    /// `#jsir<for_in_of_declaration   <decl 5loc>,   <declarator 5loc>,  "sym", scope, "kind">`
    /// (the `for (let x in/of y)` declaration metadata).
    ForInOfDeclaration(Box<ForInOfDeclarationAttr>),
    /// `#jsir<export_specifier   <L..>, <L..>, si, ei, scope, <exported attr>, <local attr>>`
    /// where exported/local are nested identifier/string-literal attrs.
    ExportSpecifier(Box<ExportSpecifierAttr>),
    /// `#jsir<private_name   <L..>, <L..>, si, ei, scope,    <flattened identifier>>`.
    PrivateName(Box<PrivateNameAttr>),
    /// An import specifier: `import_specifier` / `import_default_specifier` /
    /// `import_namespace_specifier`. Format:
    /// `#jsir<<mnemonic>   <L..>, <L..>, si, ei, scope,  "sym", symscope,[ <imported #jsir>,]    <flattened local>>`.
    ImportSpecifier(Box<ImportSpecifierAttr>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImportSpecKind {
    Named,
    Default,
    Namespace,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportSpecifierAttr {
    pub kind: ImportSpecKind,
    pub start_line: i64,
    pub start_col: i64,
    pub end_line: i64,
    pub end_col: i64,
    pub start_index: i64,
    pub end_index: i64,
    pub scope_uid: i64,
    pub sym_name: String,
    pub sym_scope: i64,
    /// Present only for `Named`; rendered as a full nested `#jsir<...>` attr.
    pub imported: Option<Attr>,
    /// The local binding, rendered flattened.
    pub local: IdentifierAttr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrivateNameAttr {
    pub start_line: i64,
    pub start_col: i64,
    pub end_line: i64,
    pub end_col: i64,
    pub start_index: i64,
    pub end_index: i64,
    pub scope_uid: i64,
    pub id: IdentifierAttr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExportSpecifierAttr {
    pub start_line: i64,
    pub start_col: i64,
    pub end_line: i64,
    pub end_col: i64,
    pub start_index: i64,
    pub end_index: i64,
    pub scope_uid: i64,
    pub exported: Attr,
    pub local: Attr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForInOfDeclarationAttr {
    // declaration location (the `let x` span)
    pub d_start_line: i64,
    pub d_start_col: i64,
    pub d_end_line: i64,
    pub d_end_col: i64,
    pub d_start_index: i64,
    pub d_end_index: i64,
    pub d_scope: i64,
    // declarator location (the `x` span)
    pub r_start_line: i64,
    pub r_start_col: i64,
    pub r_end_line: i64,
    pub r_end_col: i64,
    pub r_start_index: i64,
    pub r_end_index: i64,
    pub r_scope: i64,
    // the declarator's defined symbols (`(name, defScopeUid)`); a destructuring
    // pattern like `let [a, b]` binds more than one.
    pub symbols: Vec<(String, i64)>,
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InterpreterDirectiveAttr {
    pub start_line: i64,
    pub start_col: i64,
    pub end_line: i64,
    pub end_col: i64,
    pub start_index: i64,
    pub end_index: i64,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CommentAttr {
    pub block: bool,
    pub start_line: i64,
    pub start_col: i64,
    pub end_line: i64,
    pub end_col: i64,
    pub start_index: i64,
    pub end_index: i64,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StringLiteralKeyAttr {
    pub start_line: i64,
    pub start_col: i64,
    pub end_line: i64,
    pub end_col: i64,
    pub start_index: i64,
    pub end_index: i64,
    pub scope_uid: i64,
    pub value: String,
    pub raw: String,
    pub raw_value: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NumericLiteralKeyAttr {
    pub start_line: i64,
    pub start_col: i64,
    pub end_line: i64,
    pub end_col: i64,
    pub start_index: i64,
    pub end_index: i64,
    pub scope_uid: i64,
    pub value: f64,
    pub raw: String,
    pub raw_value: f64,
}

/// The fields of a [`Attr::Identifier`] (a `JsirIdentifierAttr` with full loc).
#[derive(Debug, Clone, PartialEq)]
pub struct IdentifierAttr {
    pub start_line: i64,
    pub start_col: i64,
    pub end_line: i64,
    pub end_col: i64,
    pub identifier_name: String,
    pub start_index: i64,
    pub end_index: i64,
    pub scope_uid: i64,
    pub name: String,
}

impl IdentifierAttr {
    fn render(&self) -> String {
        format!("#jsir<identifier   {}>", self.render_inner())
    }

    /// The identifier's parameters without the `#jsir<identifier ...>` wrapper,
    /// as embedded (flattened) inside e.g. a `private_name` attribute.
    pub fn render_inner(&self) -> String {
        format!(
            "<L {} C {}>, <L {} C {}>, {}, {}, {}, {}, {}",
            self.start_line,
            self.start_col,
            self.end_line,
            self.end_col,
            quote_mlir_string(&self.identifier_name),
            self.start_index,
            self.end_index,
            self.scope_uid,
            quote_mlir_string(&self.name),
        )
    }
}

impl Attr {
    /// Render the attribute value (the right-hand side of `key = ...`).
    pub fn render(&self) -> String {
        match self {
            Attr::Opaque(o) => o.0.render(),
            Attr::Str(s) => quote_mlir_string(s),
            Attr::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            Attr::F64(f) => format!("{} : f64", format_mlir_f64(*f)),
            Attr::I64(i) => i.to_string(),
            Attr::Array(items) => {
                let inner: Vec<String> = items.iter().map(|a| a.render()).collect();
                format!("[{}]", inner.join(", "))
            }
            Attr::I32Array(items) => {
                let inner: Vec<String> = items.iter().map(|i| i.to_string()).collect();
                format!("array<i32: {}>", inner.join(", "))
            }
            Attr::NumericLiteralExtra { raw, value } => format!(
                "#jsir<numeric_literal_extra {}, {} : f64>",
                quote_mlir_string(raw),
                format_mlir_f64(*value)
            ),
            Attr::StringLiteralExtra { raw, raw_value } => format!(
                "#jsir<string_literal_extra {}, {}>",
                quote_mlir_string(raw),
                quote_mlir_string(raw_value)
            ),
            Attr::BigIntLiteralExtra { raw, raw_value } => format!(
                "#jsir<big_int_literal_extra {}, {}>",
                quote_mlir_string(raw),
                quote_mlir_string(raw_value)
            ),
            Attr::RegExpLiteralExtra { raw } => {
                format!("#jsir<reg_exp_literal_extra {}>", quote_mlir_string(raw))
            }
            Attr::Identifier(id) => id.render(),
            Attr::StringLiteralKey(k) => format!(
                "#jsir<string_literal   <L {} C {}>, <L {} C {}>, {}, {}, {}, {},  {}, {}>",
                k.start_line,
                k.start_col,
                k.end_line,
                k.end_col,
                k.start_index,
                k.end_index,
                k.scope_uid,
                quote_mlir_string(&k.value),
                quote_mlir_string(&k.raw),
                quote_mlir_string(&k.raw_value),
            ),
            Attr::NumericLiteralKey(k) => format!(
                "#jsir<numeric_literal   <L {} C {}>, <L {} C {}>, {}, {}, {}, {} : f64,  {}, {} : f64>",
                k.start_line,
                k.start_col,
                k.end_line,
                k.end_col,
                k.start_index,
                k.end_index,
                k.scope_uid,
                format_mlir_f64(k.value),
                quote_mlir_string(&k.raw),
                format_mlir_f64(k.raw_value),
            ),
            Attr::Comment(c) => format!(
                "#jsir<{}  <L {} C {}>, <L {} C {}>, {}, {}, {}>",
                if c.block { "comment_block" } else { "comment_line" },
                c.start_line,
                c.start_col,
                c.end_line,
                c.end_col,
                c.start_index,
                c.end_index,
                quote_mlir_string(&c.value),
            ),
            Attr::DirectiveLiteralExtra { raw, raw_value } => format!(
                "#jsir<directive_literal_extra {}, {}>",
                quote_mlir_string(raw),
                quote_mlir_string(raw_value)
            ),
            Attr::InterpreterDirective(d) => format!(
                "#jsir<interpreter_directive   <L {} C {}>, <L {} C {}>, {}, {}, {}>",
                d.start_line,
                d.start_col,
                d.end_line,
                d.end_col,
                d.start_index,
                d.end_index,
                quote_mlir_string(&d.value),
            ),
            Attr::ForInOfDeclaration(f) => {
                // Each symbol renders as `,  "name", scope`; the kind follows as
                // `, "kind">`. Single and multi-binding share this shape.
                let mut syms = String::new();
                for (name, scope) in &f.symbols {
                    syms.push_str(&format!(",  {}, {}", quote_mlir_string(name), scope));
                }
                format!(
                    "#jsir<for_in_of_declaration   <L {} C {}>, <L {} C {}>, {}, {}, {},   <L {} C {}>, <L {} C {}>, {}, {}, {}{}, {}>",
                    f.d_start_line, f.d_start_col, f.d_end_line, f.d_end_col,
                    f.d_start_index, f.d_end_index, f.d_scope,
                    f.r_start_line, f.r_start_col, f.r_end_line, f.r_end_col,
                    f.r_start_index, f.r_end_index, f.r_scope,
                    syms,
                    quote_mlir_string(&f.kind),
                )
            }
            Attr::ExportSpecifier(s) => format!(
                "#jsir<export_specifier   <L {} C {}>, <L {} C {}>, {}, {}, {}, {}, {}>",
                s.start_line, s.start_col, s.end_line, s.end_col,
                s.start_index, s.end_index, s.scope_uid,
                s.exported.render(), s.local.render(),
            ),
            Attr::PrivateName(p) => format!(
                "#jsir<private_name   <L {} C {}>, <L {} C {}>, {}, {}, {},    {}>",
                p.start_line, p.start_col, p.end_line, p.end_col,
                p.start_index, p.end_index, p.scope_uid,
                p.id.render_inner(),
            ),
            Attr::ImportSpecifier(s) => {
                let mnemonic = match s.kind {
                    ImportSpecKind::Named => "import_specifier",
                    ImportSpecKind::Default => "import_default_specifier",
                    ImportSpecKind::Namespace => "import_namespace_specifier",
                };
                let imported = match &s.imported {
                    // Named specifiers print the nested `imported` attr before local.
                    Some(a) => format!(", {}", a.render()),
                    None => String::new(),
                };
                format!(
                    "#jsir<{}   <L {} C {}>, <L {} C {}>, {}, {}, {},  {}, {}{},    {}>",
                    mnemonic,
                    s.start_line, s.start_col, s.end_line, s.end_col,
                    s.start_index, s.end_index, s.scope_uid,
                    quote_mlir_string(&s.sym_name), s.sym_scope,
                    imported,
                    s.local.render_inner(),
                )
            }
        }
    }
}

/// Quote a string the way MLIR's AsmPrinter does: wrap in `"`, escape `\` as
/// `\\` (doubled), and every other non-printable byte (including `"`) as `\HH`
/// (uppercase hex). So `"` -> `\22` but `\` -> `\\`.
pub fn quote_mlir_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for b in s.bytes() {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'"' => out.push_str("\\22"),
            0x20..=0x7e => out.push(b as char),
            _ => out.push_str(&format!("\\{:02X}", b)),
        }
    }
    out.push('"');
    out
}

/// Format an f64 exactly the way MLIR's `printFloatValue` (AsmPrinter.cpp) does,
/// a faithful port of LLVM's `IEEEFloat::toString` / `toStringImpl`:
///
/// 1. Try `toString(precision=6, maxPadding=0, truncateZero=false)` (always
///    scientific, 6 significant digits zero-filled, lowercase `e`). If it
///    round-trips bit-for-bit, use it (`1.000000e+00`).
/// 2. Otherwise the default `toString()` (precision=0 -> 17 sig digits,
///    maxPadding=3, truncateZero=true): fixed or uppercase-`E` scientific
///    depending on the exponent. Use it only if it contains a `.`.
/// 3. Otherwise the hex bitcast (`0x` + 16 uppercase hex digits).
pub fn format_mlir_f64(f: f64) -> String {
    if !f.is_finite() {
        return format!("0x{:016X}", f.to_bits());
    }
    let s1 = apfloat_to_string(f, 6, 0, false);
    if s1.parse::<f64>().map(|p| p.to_bits() == f.to_bits()).unwrap_or(false) {
        return s1;
    }
    let s3 = apfloat_to_string(f, 0, 3, true);
    if s3.contains('.') {
        return s3;
    }
    format!("0x{:016X}", f.to_bits())
}

/// Port of `IEEEFloat::toStringImpl` for f64 (binary64). `significand`/`exp`
/// decomposition follows `value = significand * 2^exp` with a 53-bit
/// significand; the result is `value` rendered per the given format knobs.
fn apfloat_to_string(f: f64, format_precision: u32, format_max_padding: u32, truncate_zero: bool) -> String {
    use num_bigint::BigUint;
    use num_traits::Zero;

    let bits = f.to_bits();
    let neg = (bits >> 63) & 1 == 1;
    let raw_exp = ((bits >> 52) & 0x7ff) as i64;
    let frac = bits & 0x000f_ffff_ffff_ffff;
    let sem_precision: i64 = 53;

    let mut out = String::new();
    if neg {
        out.push('-');
    }

    // fcZero
    if raw_exp == 0 && frac == 0 {
        if format_max_padding == 0 {
            if truncate_zero {
                out.push_str("0.0E+0");
            } else {
                out.push_str("0.0");
                if format_precision > 1 {
                    for _ in 0..format_precision - 1 {
                        out.push('0');
                    }
                }
                out.push_str("e+00");
            }
        } else {
            out.push('0');
        }
        return out;
    }

    // Decompose: significand (integer) and binary exponent.
    let (mut significand, mut exp): (BigUint, i64) = if raw_exp == 0 {
        // subnormal: value = frac * 2^(1 - 1075)
        (BigUint::from(frac), 1 - (1023 + (sem_precision - 1)))
    } else {
        // normal: value = (2^52 + frac) * 2^(raw_exp - 1075)
        (
            BigUint::from(frac | 0x0010_0000_0000_0000),
            raw_exp - (1023 + (sem_precision - 1)),
        )
    };

    // Resolve precision-if-zero exactly as LLVM (2 + bits*59/196 -> 17 for f64).
    let format_precision: u32 = if format_precision == 0 {
        2 + (sem_precision as u32) * 59 / 196
    } else {
        format_precision
    };

    // Drop trailing binary zeros.
    let tz = significand.trailing_zeros().unwrap_or(0);
    if tz > 0 {
        exp += tz as i64;
        significand >>= tz;
    }

    // Convert 2^exp to a base-10 scale: value = significand * 10^exp.
    if exp > 0 {
        significand <<= exp as usize;
        exp = 0;
    } else if exp < 0 {
        let texp = (-exp) as u32;
        // multiply by 5^texp (value stays significand * 10^exp)
        significand *= pow5(texp);
    }

    // APFloat::AdjustToPrecision (on the integer): truncate (round-down) the
    // low decimal digits beyond what `format_precision` needs, BEFORE generating
    // the digit buffer. This guard-digit truncation is what makes the final
    // round-half-up match LLVM exactly at near-half boundaries.
    {
        let bits = significand.bits();
        let bits_required = (format_precision as u64 * 196 + 58) / 59;
        if bits > bits_required {
            let tens_removable = ((bits - bits_required) * 59 / 196) as u32;
            if tens_removable > 0 {
                exp += tens_removable as i64;
                significand /= pow10(tens_removable);
            }
        }
    }

    // Generate decimal digits, least-significant first, dropping trailing zeros.
    let mut buffer: Vec<u8> = Vec::new();
    let ten = BigUint::from(10u32);
    let mut in_trail = true;
    while !significand.is_zero() {
        let d = (&significand % &ten).to_u32_digits().first().copied().unwrap_or(0) as u8;
        significand /= &ten;
        if in_trail && d == 0 {
            exp += 1;
        } else {
            buffer.push(b'0' + d);
            in_trail = false;
        }
    }
    debug_assert!(!buffer.is_empty());

    adjust_to_precision(&mut buffer, &mut exp, format_precision);
    let ndigits = buffer.len();

    // Scientific vs fixed.
    let format_scientific = if format_max_padding == 0 {
        true
    } else if exp >= 0 {
        exp as u32 > format_max_padding || ndigits as u32 + exp as u32 > format_precision
    } else {
        let msd = exp + (ndigits as i64 - 1);
        if msd >= 0 {
            false
        } else {
            (-msd) as u32 > format_max_padding
        }
    };

    if format_scientific {
        let mut e = exp + (ndigits as i64 - 1);
        out.push(buffer[ndigits - 1] as char);
        out.push('.');
        if ndigits == 1 && truncate_zero {
            out.push('0');
        } else {
            for i in 1..ndigits {
                out.push(buffer[ndigits - 1 - i] as char);
            }
        }
        if !truncate_zero && format_precision as usize > ndigits - 1 {
            for _ in 0..(format_precision as usize - (ndigits - 1)) {
                out.push('0');
            }
        }
        out.push(if truncate_zero { 'E' } else { 'e' });
        out.push(if e >= 0 { '+' } else { '-' });
        if e < 0 {
            e = -e;
        }
        let mut expbuf: Vec<u8> = Vec::new();
        loop {
            expbuf.push(b'0' + (e % 10) as u8);
            e /= 10;
            if e == 0 {
                break;
            }
        }
        if !truncate_zero && expbuf.len() < 2 {
            expbuf.push(b'0');
        }
        for &c in expbuf.iter().rev() {
            out.push(c as char);
        }
        return out;
    }

    // Non-scientific, positive exponent.
    if exp >= 0 {
        for i in 0..ndigits {
            out.push(buffer[ndigits - 1 - i] as char);
        }
        for _ in 0..exp {
            out.push('0');
        }
        return out;
    }

    // Non-scientific, negative exponent.
    let n_whole = exp + ndigits as i64;
    let mut i = 0usize;
    if n_whole > 0 {
        while i != n_whole as usize {
            out.push(buffer[ndigits - i - 1] as char);
            i += 1;
        }
        out.push('.');
    } else {
        let n_zeros = 1 + (-n_whole) as usize;
        out.push('0');
        out.push('.');
        for _ in 1..n_zeros {
            out.push('0');
        }
    }
    while i != ndigits {
        out.push(buffer[ndigits - i - 1] as char);
        i += 1;
    }
    out
}

/// 5^n as a BigUint (binary exponentiation).
fn pow5(n: u32) -> num_bigint::BigUint {
    powb(5, n)
}

/// 10^n as a BigUint.
fn pow10(n: u32) -> num_bigint::BigUint {
    powb(10, n)
}

fn powb(b: u32, mut n: u32) -> num_bigint::BigUint {
    use num_bigint::BigUint;
    let mut result = BigUint::from(1u32);
    let mut base = BigUint::from(b);
    while n > 0 {
        if n & 1 == 1 {
            result *= &base;
        }
        n >>= 1;
        if n > 0 {
            base = &base * &base;
        }
    }
    result
}

/// Port of `AdjustToPrecision(SmallVectorImpl<char>&, int&, unsigned)`:
/// round the LSB-first digit `buffer` to `format_precision` significant digits
/// using round-half-up, adjusting `exp`.
fn adjust_to_precision(buffer: &mut Vec<u8>, exp: &mut i64, format_precision: u32) {
    let n = buffer.len();
    let fp = format_precision as usize;
    if n <= fp {
        return;
    }
    // Most significant digits are at the end of the buffer.
    let mut first_significant = n - fp;

    if buffer[first_significant - 1] < b'5' {
        // Round down: truncate, then drop newly-trailing zeros.
        while first_significant < n && buffer[first_significant] == b'0' {
            first_significant += 1;
        }
        *exp += first_significant as i64;
        buffer.drain(0..first_significant);
        return;
    }

    // Round up: decimal add-with-carry over the kept (high) digits.
    let mut carried_through = true;
    for i in first_significant..n {
        if buffer[i] == b'9' {
            first_significant += 1;
        } else {
            buffer[i] += 1;
            carried_through = false;
            break;
        }
    }

    if carried_through && first_significant == n {
        *exp += first_significant as i64;
        buffer.clear();
        buffer.push(b'1');
        return;
    }

    *exp += first_significant as i64;
    buffer.drain(0..first_significant);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_format() {
        assert_eq!(format_mlir_f64(1.0), "1.000000e+00");
        assert_eq!(format_mlir_f64(2.0), "2.000000e+00");
        assert_eq!(format_mlir_f64(0.0), "0.000000e+00");
        assert_eq!(format_mlir_f64(-2.5), "-2.500000e+00");
    }

    #[test]
    fn numeric_literal_extra_render() {
        let a = Attr::NumericLiteralExtra {
            raw: "1".into(),
            value: 1.0,
        };
        assert_eq!(a.render(), "#jsir<numeric_literal_extra \"1\", 1.000000e+00 : f64>");
    }
}
