use super::Expr;

impl Into<Expr> for &i64 {
    fn into(self) -> Expr {
        Expr::Num(*self)
    }
}
impl Into<Expr> for i64 {
    fn into(self) -> Expr {
        Expr::Num(self)
    }
}
impl Into<Expr> for &str {
    fn into(self) -> Expr {
        self.to_string().into()
    }
}
impl Into<Expr> for String {
    fn into(self) -> Expr {
        if self.starts_with('?') {
            Expr::LogicVariable(self)
        } else {
            Expr::Symbol(self)
        }
    }
}

impl Into<Expr> for &String {
    fn into(self) -> Expr {
        Expr::Symbol(self.to_string())
    }
}

impl<T : Into<Expr>, S : Into<Expr>> Into<Expr> for (T, S) {
    fn into(self) -> Expr {
        let f = self.0.into();
        match f {
            Expr::Symbol(s) if s == "quote" => Expr::Exhausted(box self.1.into()),
            _ => Expr::Call(box f, vec![self.1.into()])
        }

    }
}

impl<T : Into<Expr>, S : Into<Expr>, R : Into<Expr>> Into<Expr> for (T, S, R) {
    fn into(self) -> Expr {
        Expr::Call(box self.0.into(), vec![self.1.into(), self.2.into()])
    }
}

impl<T : Into<Expr>, S : Into<Expr>> Into<Expr> for Vec<(T, S)> {
    fn into(self) -> Expr {
        let mut results = Vec::with_capacity(self.len());
        for (key, value) in self {
            results.push((key.into(), value.into()));
        }
        Expr::Map(results)
    }
}

