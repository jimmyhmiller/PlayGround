//! Glaze AST. A `Program` is a parsed `.glz` source: a set of token definitions
//! and style definitions. Expressions are deliberately the intersection of
//! "CSS math" and "shader math" so the same grammar serves both stages.

#[derive(Debug, Clone)]
pub struct Program {
    pub tokens: Vec<TokenDef>,
    pub fns: Vec<FnDef>,
    pub styles: Vec<StyleDef>,
}

#[derive(Debug, Clone)]
pub struct TokenDef {
    pub name: String,
    pub value: Expr,
}

/// A pure user function: `fn space(n) = n * 4px`. Callable in any expression
/// (style props, token defs, `when` conditions, shader bodies). Non-recursive.
#[derive(Debug, Clone)]
pub struct FnDef {
    pub name: String,
    pub params: Vec<String>,
    pub body: Expr,
}

/// Substitute `args` for `params` in `body` (function inlining). Used by both
/// the CPU evaluator and the shader lowerer, so a `fn` works on either side of
/// the staging boundary.
pub fn subst_params(body: &Expr, params: &[String], args: &[Expr]) -> Expr {
    fn go(e: &Expr, map: &[(String, Expr)]) -> Expr {
        match e {
            Expr::Ident(n) => map
                .iter()
                .find(|(p, _)| p == n)
                .map(|(_, a)| a.clone())
                .unwrap_or_else(|| e.clone()),
            Expr::Num(..) | Expr::Hex(_) | Expr::Color { .. } => e.clone(),
            Expr::Unary(op, x) => Expr::Unary(*op, Box::new(go(x, map))),
            Expr::Bin(op, l, r) => {
                Expr::Bin(op.clone(), Box::new(go(l, map)), Box::new(go(r, map)))
            }
            Expr::Call(n, a) => Expr::Call(n.clone(), a.iter().map(|x| go(x, map)).collect()),
            Expr::Tern(c, a, b) => Expr::Tern(
                Box::new(go(c, map)),
                Box::new(go(a, map)),
                Box::new(go(b, map)),
            ),
        }
    }
    let map: Vec<(String, Expr)> = params.iter().cloned().zip(args.iter().cloned()).collect();
    go(body, &map)
}

#[derive(Debug, Clone)]
pub struct StyleDef {
    pub name: String,
    /// static variant parameters, e.g. `button(intent, size)`
    pub params: Vec<String>,
    pub body: Vec<Item>,
}

#[derive(Debug, Clone)]
pub enum Item {
    /// A property / paint-or-box layer: `fill <expr>`, `pad 8px 12px`, `radius radius.md`.
    Prop { name: String, args: Vec<Expr> },
    /// A local variable binding within a style body: `let p = space(4)`. In
    /// scope for the properties that follow it.
    Let { name: String, value: Expr },
    /// A discrete-state overlay plan: `:hover { … }`.
    State { state: String, body: Vec<Item> },
    /// A responsive (media-query) overlay: `when vw < 560 { … }`. The condition
    /// is evaluated over the viewport (`vw`/`vh`) at resolve time.
    When { cond: Expr, body: Vec<Item> },
    /// A shader layer: `shader { … }` or `overlay shader { … }`.
    Shader { overlay: bool, body: ShaderBody },
}

/// The body of a `shader {}` block: a sequence of `let` bindings and a single
/// `emit <expr>` producing the layer's output color (an rgba `vec4`).
#[derive(Debug, Clone)]
pub struct ShaderBody {
    pub lets: Vec<(String, Expr)>,
    pub emit: Expr,
}

#[derive(Debug, Clone)]
pub enum Expr {
    /// numeric literal with optional unit (`8px`, `1.25`, `50%`)
    Num(f64, Option<String>),
    /// `#rrggbb`
    Hex(String),
    /// `oklch(L C H [/ a])` / `oklab(L a b [/ alpha])`
    Color { space: String, nums: Vec<f64> },
    /// a bare identifier: a token ref, a variant param, or a symbol/keyword
    Ident(String),
    Unary(char, Box<Expr>),
    Bin(String, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
    /// `cond ? a : b`
    Tern(Box<Expr>, Box<Expr>, Box<Expr>),
}
