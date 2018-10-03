#[derive(Debug)]
enum Token {
    OpenParen,
    CloseParen,
    Atom(String),
}

fn to_token(token: &str) -> Token {
    match token {
        "(" => Token::OpenParen,
        ")" => Token::CloseParen,
        str => Token::Atom(str.to_string()),
    }
}

fn tokenize(text: String) -> Vec<Token> {
    text.replace("(", " ( ")
        .replace(")", " ) ")
        .split_whitespace()
        .map(to_token)
        .collect()
}

#[derive(Debug)]
enum Expr {
    SExpr(Vec<Expr>),
    Atom(String),
    Int(i64),
    Float(f64)
}



fn main() {
    println!("{:?}", tokenize("(+ 2 2)".to_string()))
}
