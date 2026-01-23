use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum BinOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    // Comparison
    Lt,
    Gt,
    Lte,
    Gte,
    Eq,
    NotEq,
    // Logical
    And,
    Or,
    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    UShr,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::Mod => write!(f, "%"),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
            BinOp::Lte => write!(f, "<="),
            BinOp::Gte => write!(f, ">="),
            BinOp::Eq => write!(f, "=="),
            BinOp::NotEq => write!(f, "!="),
            BinOp::And => write!(f, "&&"),
            BinOp::Or => write!(f, "||"),
            BinOp::BitAnd => write!(f, "&"),
            BinOp::BitOr => write!(f, "|"),
            BinOp::BitXor => write!(f, "^"),
            BinOp::Shl => write!(f, "<<"),
            BinOp::Shr => write!(f, ">>"),
            BinOp::UShr => write!(f, ">>>"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    // Literals
    Int(i64),
    Bool(bool),
    String(String),
    Undefined,
    Null,

    // Variables
    Var(String),

    // Binary and unary operations
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    BitNot(Box<Expr>),   // ~x
    LogNot(Box<Expr>),   // !x

    // Control flow
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Switch {
        discriminant: Box<Expr>,
        cases: Vec<(Expr, Vec<Expr>)>,  // (match_value, body_statements)
        default: Option<Vec<Expr>>,
    },

    // Bindings
    Let(String, Box<Expr>, Box<Expr>),

    // Functions
    Fn(Vec<String>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),

    // Arrays
    Array(Vec<Expr>),
    Index(Box<Expr>, Box<Expr>),
    Len(Box<Expr>),

    // Objects
    Object(Vec<(String, Expr)>),           // {key: value, ...}
    PropAccess(Box<Expr>, String),          // obj.prop
    PropSet(Box<Expr>, String, Box<Expr>),  // (prop-set! obj "prop" value)
    ComputedAccess(Box<Expr>, Box<Expr>),   // obj[expr] (when not array)
    ComputedSet(Box<Expr>, Box<Expr>, Box<Expr>), // obj[expr] = value

    // Loops
    While(Box<Expr>, Box<Expr>),
    For {
        init: Option<Box<Expr>>,
        cond: Option<Box<Expr>>,
        update: Option<Box<Expr>>,
        body: Box<Expr>,
    },
    Break,
    Continue,

    // Mutation
    Set(String, Box<Expr>),

    // Sequencing
    Begin(Vec<Expr>),

    // Error handling
    Throw(Box<Expr>),
    TryCatch {
        try_block: Box<Expr>,
        catch_param: Option<String>,
        catch_block: Box<Expr>,
        finally_block: Option<Box<Expr>>,
    },

    // Opaque expressions (evaluate to themselves)
    New(Box<Expr>, Vec<Expr>),  // new Constructor(args)
    Opaque(String),              // Opaque value with a label
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Int(n) => write!(f, "{}", n),
            Expr::Bool(b) => write!(f, "{}", b),
            Expr::String(s) => write!(f, "\"{}\"", s.escape_default()),
            Expr::Undefined => write!(f, "undefined"),
            Expr::Null => write!(f, "null"),
            Expr::Var(name) => write!(f, "{}", name),
            Expr::BinOp(op, left, right) => {
                write!(f, "({} {} {})", op, left, right)
            }
            Expr::BitNot(expr) => write!(f, "(~ {})", expr),
            Expr::LogNot(expr) => write!(f, "(! {})", expr),
            Expr::If(cond, then_branch, else_branch) => {
                write!(f, "(if {} {} {})", cond, then_branch, else_branch)
            }
            Expr::Switch { discriminant, cases, default } => {
                write!(f, "(switch {}", discriminant)?;
                for (val, body) in cases {
                    write!(f, " (case {}", val)?;
                    for stmt in body {
                        write!(f, " {}", stmt)?;
                    }
                    write!(f, ")")?;
                }
                if let Some(def) = default {
                    write!(f, " (default")?;
                    for stmt in def {
                        write!(f, " {}", stmt)?;
                    }
                    write!(f, ")")?;
                }
                write!(f, ")")
            }
            Expr::Let(name, value, body) => {
                write!(f, "(let {} {} {})", name, value, body)
            }
            Expr::Fn(params, body) => {
                write!(f, "(fn ({}) {})", params.join(" "), body)
            }
            Expr::Call(func, args) => {
                write!(f, "(call {}", func)?;
                for arg in args {
                    write!(f, " {}", arg)?;
                }
                write!(f, ")")
            }
            Expr::Array(elements) => {
                write!(f, "(array")?;
                for elem in elements {
                    write!(f, " {}", elem)?;
                }
                write!(f, ")")
            }
            Expr::Index(arr, idx) => {
                write!(f, "(index {} {})", arr, idx)
            }
            Expr::Len(arr) => {
                write!(f, "(len {})", arr)
            }
            Expr::Object(props) => {
                write!(f, "(object")?;
                for (key, val) in props {
                    write!(f, " ({} {})", key, val)?;
                }
                write!(f, ")")
            }
            Expr::PropAccess(obj, prop) => {
                write!(f, "(prop {} \"{}\")", obj, prop)
            }
            Expr::PropSet(obj, prop, val) => {
                write!(f, "(prop-set! {} \"{}\" {})", obj, prop, val)
            }
            Expr::ComputedAccess(obj, key) => {
                write!(f, "(computed-prop {} {})", obj, key)
            }
            Expr::ComputedSet(obj, key, val) => {
                write!(f, "(computed-set! {} {} {})", obj, key, val)
            }
            Expr::While(cond, body) => {
                write!(f, "(while {} {})", cond, body)
            }
            Expr::For { init, cond, update, body } => {
                write!(f, "(for")?;
                match init {
                    Some(i) => write!(f, " {}", i)?,
                    None => write!(f, " #f")?,
                }
                match cond {
                    Some(c) => write!(f, " {}", c)?,
                    None => write!(f, " #t")?,
                }
                match update {
                    Some(u) => write!(f, " {}", u)?,
                    None => write!(f, " #f")?,
                }
                write!(f, " {})", body)
            }
            Expr::Break => write!(f, "(break)"),
            Expr::Continue => write!(f, "(continue)"),
            Expr::Set(name, value) => {
                write!(f, "(set! {} {})", name, value)
            }
            Expr::Begin(exprs) => {
                write!(f, "(begin")?;
                for e in exprs {
                    write!(f, " {}", e)?;
                }
                write!(f, ")")
            }
            Expr::Throw(expr) => {
                write!(f, "(throw {})", expr)
            }
            Expr::TryCatch { try_block, catch_param, catch_block, finally_block } => {
                write!(f, "(try {} ", try_block)?;
                if let Some(param) = catch_param {
                    write!(f, "(catch {} {})", param, catch_block)?;
                } else {
                    write!(f, "(catch {})", catch_block)?;
                }
                if let Some(finally) = finally_block {
                    write!(f, " (finally {})", finally)?;
                }
                write!(f, ")")
            }
            Expr::New(constructor, args) => {
                write!(f, "(new {}", constructor)?;
                for arg in args {
                    write!(f, " {}", arg)?;
                }
                write!(f, ")")
            }
            Expr::Opaque(label) => write!(f, "(opaque \"{}\")", label),
        }
    }
}

/// Pretty print an expression with indentation
pub fn pretty_print(expr: &Expr) -> String {
    pretty_print_indent(expr, 0)
}

fn pretty_print_indent(expr: &Expr, indent: usize) -> String {
    let _ind = "  ".repeat(indent);
    let ind1 = "  ".repeat(indent + 1);

    match expr {
        // Simple atoms - no newlines needed
        Expr::Int(n) => format!("{}", n),
        Expr::Bool(b) => format!("{}", b),
        Expr::String(s) => format!("\"{}\"", s.escape_default()),
        Expr::Undefined => "undefined".to_string(),
        Expr::Null => "null".to_string(),
        Expr::Var(name) => name.clone(),
        Expr::Break => "(break)".to_string(),
        Expr::Continue => "(continue)".to_string(),
        Expr::Opaque(label) => format!("(opaque \"{}\")", label),

        // Binary operations - inline if simple
        Expr::BinOp(op, left, right) => {
            let l = pretty_print_indent(left, indent);
            let r = pretty_print_indent(right, indent);
            if l.len() + r.len() < 60 && !l.contains('\n') && !r.contains('\n') {
                format!("({} {} {})", op, l, r)
            } else {
                format!("({} {} {})", op, l, r)
            }
        }

        Expr::BitNot(e) => format!("(~ {})", pretty_print_indent(e, indent)),
        Expr::LogNot(e) => format!("(! {})", pretty_print_indent(e, indent)),

        // Let bindings - always multiline
        Expr::Let(name, value, body) => {
            let val = pretty_print_indent(value, indent + 1);
            let bod = pretty_print_indent(body, indent + 1);
            if val.len() < 40 && !val.contains('\n') {
                format!("(let {} {}\n{}{})", name, val, ind1, bod)
            } else {
                format!("(let {} \n{}{}\n{}{})", name, ind1, val, ind1, bod)
            }
        }

        // Function definitions
        Expr::Fn(params, body) => {
            let bod = pretty_print_indent(body, indent + 1);
            if bod.len() < 60 && !bod.contains('\n') {
                format!("(fn ({}) {})", params.join(" "), bod)
            } else {
                format!("(fn ({})\n{}{})", params.join(" "), ind1, bod)
            }
        }

        // Function calls
        Expr::Call(func, args) => {
            let f = pretty_print_indent(func, indent);
            let arg_strs: Vec<String> = args.iter().map(|a| pretty_print_indent(a, indent + 1)).collect();
            let total_len: usize = arg_strs.iter().map(|s| s.len()).sum();

            if total_len < 50 && arg_strs.iter().all(|s| !s.contains('\n')) {
                format!("(call {} {})", f, arg_strs.join(" "))
            } else {
                let args_formatted = arg_strs.join(&format!("\n{}", ind1));
                format!("(call {}\n{}{})", f, ind1, args_formatted)
            }
        }

        // Begin blocks - each expression on its own line
        Expr::Begin(exprs) => {
            if exprs.is_empty() {
                "(begin)".to_string()
            } else if exprs.len() == 1 {
                format!("(begin {})", pretty_print_indent(&exprs[0], indent))
            } else {
                let parts: Vec<String> = exprs.iter()
                    .map(|e| format!("{}{}", ind1, pretty_print_indent(e, indent + 1)))
                    .collect();
                format!("(begin\n{})", parts.join("\n"))
            }
        }

        // While loops
        Expr::While(cond, body) => {
            let c = pretty_print_indent(cond, indent);
            let b = pretty_print_indent(body, indent + 1);
            format!("(while {}\n{}{})", c, ind1, b)
        }

        // For loops
        Expr::For { init, cond, update, body } => {
            let i = init.as_ref().map(|e| pretty_print_indent(e, indent)).unwrap_or_else(|| "#f".to_string());
            let c = cond.as_ref().map(|e| pretty_print_indent(e, indent)).unwrap_or_else(|| "#t".to_string());
            let u = update.as_ref().map(|e| pretty_print_indent(e, indent)).unwrap_or_else(|| "#f".to_string());
            let b = pretty_print_indent(body, indent + 1);
            format!("(for {} {} {}\n{}{})", i, c, u, ind1, b)
        }

        // Switch statements
        Expr::Switch { discriminant, cases, default } => {
            let d = pretty_print_indent(discriminant, indent);
            let mut result = format!("(switch {}", d);

            for (val, body) in cases {
                let v = pretty_print_indent(val, indent + 1);
                result.push_str(&format!("\n{}(case {}", ind1, v));
                for stmt in body {
                    result.push_str(&format!("\n{}  {}", ind1, pretty_print_indent(stmt, indent + 2)));
                }
                result.push(')');
            }

            if let Some(def) = default {
                result.push_str(&format!("\n{}(default", ind1));
                for stmt in def {
                    result.push_str(&format!("\n{}  {}", ind1, pretty_print_indent(stmt, indent + 2)));
                }
                result.push(')');
            }

            result.push(')');
            result
        }

        // If expressions
        Expr::If(cond, then_branch, else_branch) => {
            let c = pretty_print_indent(cond, indent);
            let t = pretty_print_indent(then_branch, indent + 1);
            let e = pretty_print_indent(else_branch, indent + 1);

            if t.len() + e.len() < 50 && !t.contains('\n') && !e.contains('\n') {
                format!("(if {} {} {})", c, t, e)
            } else {
                format!("(if {}\n{}{}\n{}{})", c, ind1, t, ind1, e)
            }
        }

        // Arrays
        Expr::Array(elements) => {
            if elements.is_empty() {
                "(array)".to_string()
            } else {
                let parts: Vec<String> = elements.iter()
                    .map(|e| pretty_print_indent(e, indent))
                    .collect();
                let total_len: usize = parts.iter().map(|s| s.len()).sum();

                if total_len < 60 && parts.iter().all(|s| !s.contains('\n')) {
                    format!("(array {})", parts.join(" "))
                } else {
                    let formatted = parts.iter()
                        .map(|p| format!("{}{}", ind1, p))
                        .collect::<Vec<_>>()
                        .join("\n");
                    format!("(array\n{})", formatted)
                }
            }
        }

        // Objects
        Expr::Object(props) => {
            if props.is_empty() {
                "(object)".to_string()
            } else {
                let parts: Vec<String> = props.iter()
                    .map(|(k, v)| format!("({} {})", k, pretty_print_indent(v, indent + 1)))
                    .collect();
                let total_len: usize = parts.iter().map(|s| s.len()).sum();

                if total_len < 60 && parts.iter().all(|s| !s.contains('\n')) {
                    format!("(object {})", parts.join(" "))
                } else {
                    let formatted = parts.iter()
                        .map(|p| format!("{}{}", ind1, p))
                        .collect::<Vec<_>>()
                        .join("\n");
                    format!("(object\n{})", formatted)
                }
            }
        }

        // Property access/set
        Expr::Index(arr, idx) => format!("(index {} {})", pretty_print_indent(arr, indent), pretty_print_indent(idx, indent)),
        Expr::Len(arr) => format!("(len {})", pretty_print_indent(arr, indent)),
        Expr::PropAccess(obj, prop) => format!("(prop {} \"{}\")", pretty_print_indent(obj, indent), prop),
        Expr::PropSet(obj, prop, val) => format!("(prop-set! {} \"{}\" {})", pretty_print_indent(obj, indent), prop, pretty_print_indent(val, indent)),
        Expr::ComputedAccess(obj, key) => format!("(computed-prop {} {})", pretty_print_indent(obj, indent), pretty_print_indent(key, indent)),
        Expr::ComputedSet(obj, key, val) => format!("(computed-set! {} {} {})", pretty_print_indent(obj, indent), pretty_print_indent(key, indent), pretty_print_indent(val, indent)),

        // Set
        Expr::Set(name, value) => format!("(set! {} {})", name, pretty_print_indent(value, indent)),

        // Throw
        Expr::Throw(e) => format!("(throw {})", pretty_print_indent(e, indent)),

        // Try-catch
        Expr::TryCatch { try_block, catch_param, catch_block, finally_block } => {
            let t = pretty_print_indent(try_block, indent + 1);
            let c = pretty_print_indent(catch_block, indent + 1);

            let mut result = format!("(try\n{}{}\n{}", ind1, t, ind1);
            if let Some(param) = catch_param {
                result.push_str(&format!("(catch {} {})", param, c));
            } else {
                result.push_str(&format!("(catch {})", c));
            }

            if let Some(finally) = finally_block {
                let f = pretty_print_indent(finally, indent + 1);
                result.push_str(&format!("\n{}(finally {})", ind1, f));
            }

            result.push(')');
            result
        }

        // New
        Expr::New(ctor, args) => {
            let c = pretty_print_indent(ctor, indent);
            let arg_strs: Vec<String> = args.iter().map(|a| pretty_print_indent(a, indent)).collect();
            format!("(new {} {})", c, arg_strs.join(" "))
        }
    }
}
