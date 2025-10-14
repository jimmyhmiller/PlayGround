use nom::{
    branch::alt,
    bytes::complete::take_while1,
    character::complete::{char, multispace1},
    combinator::{map as nom_map, opt, recognize},
    multi::many0,
    sequence::{delimited, preceded, tuple},
    IResult,
};

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Symbol(String),
    Keyword(String),
    String(String),
    Integer(i64),
    Float(f64),
    List(Vec<Value>),
    Vector(Vec<Value>),
    Map(Vec<(Value, Value)>),
}

// Parse whitespace and comments
fn ws(input: &str) -> IResult<&str, ()> {
    nom_map(
        many0(alt((
            nom_map(multispace1, |_| ()),
            nom_map(preceded(char(';'), take_while1(|c| c != '\n')), |_| ()),
        ))),
        |_| (),
    )(input)
}

// Parse a symbol
fn symbol(input: &str) -> IResult<&str, Value> {
    use nom::bytes::complete::take_while;

    let first_char = |c: char| {
        c.is_alphabetic()
            || c == '*'
            || c == '+'
            || c == '-'
            || c == '/'
            || c == '='
            || c == '<'
            || c == '>'
            || c == '!'
            || c == '?'
            || c == '_'
            || c == '%'
            || c == '@'
            || c == '^'
            || c == '$'
    };

    let rest_char = |c: char| {
        first_char(c) || c.is_numeric() || c == '.' || c == ':'
    };

    nom_map(
        recognize(tuple((
            take_while1(first_char),
            take_while(rest_char),
        ))),
        |s: &str| Value::Symbol(s.to_string()),
    )(input)
}

// Parse a keyword (starts with :)
fn keyword(input: &str) -> IResult<&str, Value> {
    nom_map(
        preceded(
            char(':'),
            take_while1(|c: char| c.is_alphanumeric() || c == '-' || c == '_'),
        ),
        |s: &str| Value::Keyword(s.to_string()),
    )(input)
}

// Parse a string
fn string(input: &str) -> IResult<&str, Value> {
    nom_map(
        delimited(
            char('"'),
            take_while1(|c| c != '"'),
            char('"'),
        ),
        |s: &str| Value::String(s.to_string()),
    )(input)
}

// Parse an integer
fn integer(input: &str) -> IResult<&str, Value> {
    nom_map(
        recognize(tuple((opt(char('-')), take_while1(|c: char| c.is_numeric())))),
        |s: &str| Value::Integer(s.parse().unwrap()),
    )(input)
}

// Parse a float
fn float(input: &str) -> IResult<&str, Value> {
    nom_map(
        recognize(tuple((
            opt(char('-')),
            take_while1(|c: char| c.is_numeric()),
            char('.'),
            take_while1(|c: char| c.is_numeric()),
        ))),
        |s: &str| Value::Float(s.parse().unwrap()),
    )(input)
}

// Forward declaration for recursive parsing
fn value(input: &str) -> IResult<&str, Value> {
    preceded(
        ws,
        alt((
            float,
            integer,
            keyword,
            string,
            list,
            vector,
            parse_map,
            symbol,
        )),
    )(input)
}

// Parse a list
fn list(input: &str) -> IResult<&str, Value> {
    nom_map(
        delimited(
            char('('),
            many0(preceded(ws, value)),
            preceded(ws, char(')')),
        ),
        Value::List,
    )(input)
}

// Parse a vector
fn vector(input: &str) -> IResult<&str, Value> {
    nom_map(
        delimited(
            char('['),
            many0(preceded(ws, value)),
            preceded(ws, char(']')),
        ),
        Value::Vector,
    )(input)
}

// Parse a map
fn parse_map(input: &str) -> IResult<&str, Value> {
    nom_map(
        delimited(
            char('{'),
            many0(tuple((preceded(ws, value), preceded(ws, value)))),
            preceded(ws, char('}')),
        ),
        Value::Map,
    )(input)
}

// Parse multiple forms
pub fn parse(input: &str) -> IResult<&str, Vec<Value>> {
    use nom::sequence::terminated;
    terminated(many0(value), ws)(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol() {
        assert_eq!(
            symbol("foo"),
            Ok(("", Value::Symbol("foo".to_string())))
        );
        assert_eq!(
            symbol("arith.addi"),
            Ok(("", Value::Symbol("arith.addi".to_string())))
        );
        assert_eq!(
            symbol("%result"),
            Ok(("", Value::Symbol("%result".to_string())))
        );
    }

    #[test]
    fn test_keyword() {
        assert_eq!(
            keyword(":value"),
            Ok(("", Value::Keyword("value".to_string())))
        );
        assert_eq!(
            keyword(":sym-name"),
            Ok(("", Value::Keyword("sym-name".to_string())))
        );
    }

    #[test]
    fn test_integer() {
        assert_eq!(integer("42"), Ok(("", Value::Integer(42))));
        assert_eq!(integer("-10"), Ok(("", Value::Integer(-10))));
    }

    #[test]
    fn test_list() {
        let result = parse("(+ 1 2)").unwrap();
        assert_eq!(
            result.1,
            vec![Value::List(vec![
                Value::Symbol("+".to_string()),
                Value::Integer(1),
                Value::Integer(2),
            ])]
        );
    }

    #[test]
    fn test_nested_list() {
        let result = parse("(defn foo [] (+ 1 2))").unwrap();
        assert_eq!(
            result.1,
            vec![Value::List(vec![
                Value::Symbol("defn".to_string()),
                Value::Symbol("foo".to_string()),
                Value::Vector(vec![]),
                Value::List(vec![
                    Value::Symbol("+".to_string()),
                    Value::Integer(1),
                    Value::Integer(2),
                ]),
            ])]
        );
    }

    #[test]
    fn test_map() {
        let result = parse("{:value 10 :type i32}").unwrap();
        assert_eq!(
            result.1,
            vec![Value::Map(vec![
                (Value::Keyword("value".to_string()), Value::Integer(10)),
                (Value::Keyword("type".to_string()), Value::Symbol("i32".to_string())),
            ])]
        );
    }
}
