// Lifted from beagle/src/ast.rs, stripped of the compile/codegen impls.
// Keeps only the AST type definitions + basic helpers used by the parser.

type TokenPosition = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenRange {
    pub start: usize,
    pub end: usize,
}

impl TokenRange {
    pub fn new(start: usize, end: usize) -> Self {
        TokenRange { start, end }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Condition {
    LessThanOrEqual,
    LessThan,
    Equal,
    NotEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StringInterpolationPart {
    Literal(String),
    Expression(Box<Ast>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Ast {
    Program {
        elements: Vec<Ast>,
        token_range: TokenRange,
    },
    Function {
        name: Option<String>,
        args: Vec<Pattern>,
        rest_param: Option<String>,
        body: Vec<Ast>,
        token_range: TokenRange,
        docstring: Option<String>,
    },
    Struct {
        name: String,
        fields: Vec<Ast>,
        token_range: TokenRange,
        docstring: Option<String>,
    },
    StructField {
        name: String,
        mutable: bool,
        default_value: Option<Box<Ast>>,
        token_range: TokenRange,
        docstring: Option<String>,
    },
    Enum {
        name: String,
        variants: Vec<Ast>,
        token_range: TokenRange,
        docstring: Option<String>,
    },
    EnumVariant {
        name: String,
        fields: Vec<Ast>,
        token_range: TokenRange,
    },
    EnumStaticVariant {
        name: String,
        token_range: TokenRange,
    },
    Protocol {
        name: String,
        type_params: Vec<String>,
        body: Vec<Ast>,
        token_range: TokenRange,
    },
    Extend {
        target_type: String,
        protocol: String,
        protocol_type_args: Vec<String>,
        body: Vec<Ast>,
        token_range: TokenRange,
    },
    FunctionStub {
        name: String,
        args: Vec<Pattern>,
        rest_param: Option<String>,
        token_range: TokenRange,
    },
    If {
        condition: Box<Ast>,
        then: Vec<Ast>,
        else_: Vec<Ast>,
        token_range: TokenRange,
    },
    Condition {
        operator: Condition,
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    Add {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    Sub {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    Mul {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    Div {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    Modulo {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    Recurse {
        args: Vec<Ast>,
        token_range: TokenRange,
    },
    TailRecurse {
        args: Vec<Ast>,
        token_range: TokenRange,
    },
    Call {
        name: String,
        args: Vec<Ast>,
        token_range: TokenRange,
    },
    CallExpr {
        callee: Box<Ast>,
        args: Vec<Ast>,
        token_range: TokenRange,
    },
    Let {
        pattern: Pattern,
        value: Box<Ast>,
        token_range: TokenRange,
    },
    LetMut {
        pattern: Pattern,
        value: Box<Ast>,
        token_range: TokenRange,
    },
    LetDynamic {
        name: String,
        value: Box<Ast>,
        token_range: TokenRange,
    },
    Binding {
        var_name: String,
        value_expr: Box<Ast>,
        body: Vec<Ast>,
        token_range: TokenRange,
    },
    IntegerLiteral(i64, TokenPosition),
    FloatLiteral(String, TokenPosition),
    Identifier(String, TokenPosition),
    String(String, TokenPosition),
    StringInterpolation {
        parts: Vec<StringInterpolationPart>,
        token_range: TokenRange,
    },
    Keyword(String, TokenPosition),
    True(TokenPosition),
    False(TokenPosition),
    StructCreation {
        name: String,
        fields: Vec<(String, Ast)>,
        spread: Option<Box<Ast>>,
        token_range: TokenRange,
    },
    PropertyAccess {
        object: Box<Ast>,
        property: Box<Ast>,
        token_range: TokenRange,
    },
    Null(TokenPosition),
    EnumCreation {
        name: String,
        variant: String,
        fields: Vec<(String, Ast)>,
        token_range: TokenRange,
    },
    Namespace {
        name: String,
        token_range: TokenRange,
    },
    Use {
        namespace_name: String,
        alias: Box<Ast>,
        token_range: TokenRange,
    },
    ShiftLeft {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    ShiftRight {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    ShiftRightZero {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    BitWiseAnd {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    BitWiseOr {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    BitWiseXor {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    And {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    Or {
        left: Box<Ast>,
        right: Box<Ast>,
        token_range: TokenRange,
    },
    Not {
        expr: Box<Ast>,
        token_range: TokenRange,
    },
    Array {
        array: Vec<Ast>,
        token_range: TokenRange,
    },
    MapLiteral {
        pairs: Vec<(Ast, Ast)>,
        token_range: TokenRange,
    },
    SetLiteral {
        elements: Vec<Ast>,
        token_range: TokenRange,
    },
    IndexOperator {
        array: Box<Ast>,
        index: Box<Ast>,
        token_range: TokenRange,
    },
    Loop {
        body: Vec<Ast>,
        token_range: TokenRange,
    },
    While {
        condition: Box<Ast>,
        body: Vec<Ast>,
        token_range: TokenRange,
    },
    Break {
        value: Box<Ast>,
        token_range: TokenRange,
    },
    Continue {
        token_range: TokenRange,
    },
    Return {
        value: Box<Ast>,
        token_range: TokenRange,
    },
    For {
        binding: String,
        collection: Box<Ast>,
        body: Vec<Ast>,
        token_range: TokenRange,
    },
    Assignment {
        name: Box<Ast>,
        value: Box<Ast>,
        token_range: TokenRange,
    },
    Try {
        body: Vec<Ast>,
        exception_binding: String,
        resume_binding: Option<String>,
        catch_body: Vec<Ast>,
        token_range: TokenRange,
    },
    Throw {
        value: Box<Ast>,
        token_range: TokenRange,
    },
    Match {
        value: Box<Ast>,
        arms: Vec<MatchArm>,
        token_range: TokenRange,
    },
    ProtocolDispatch {
        args: Vec<String>,
        cache_location: usize,
        dispatch_table_ptr: usize,
        default_fn_ptr: usize,
        num_args: usize,
        token_range: TokenRange,
    },
    MultiArityFunction {
        name: Option<String>,
        cases: Vec<ArityCase>,
        token_range: TokenRange,
        docstring: Option<String>,
    },
    Reset {
        body: Vec<Ast>,
        token_range: TokenRange,
    },
    Shift {
        continuation_param: String,
        body: Vec<Ast>,
        token_range: TokenRange,
    },
    Perform {
        value: Box<Ast>,
        token_range: TokenRange,
    },
    Handle {
        protocol: String,
        protocol_type_args: Vec<String>,
        handler_instance: Box<Ast>,
        body: Vec<Ast>,
        token_range: TokenRange,
    },
    Future {
        body: Box<Ast>,
        token_range: TokenRange,
    },
    Test {
        name: String,
        body: Vec<Ast>,
        token_range: TokenRange,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Box<Ast>>,
    pub body: Vec<Ast>,
    pub token_range: TokenRange,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Pattern {
    Identifier {
        name: String,
        token_range: TokenRange,
    },
    EnumVariant {
        enum_name: String,
        variant_name: String,
        fields: Vec<FieldPattern>,
        token_range: TokenRange,
    },
    Struct {
        name: String,
        fields: Vec<FieldPattern>,
        token_range: TokenRange,
    },
    Array {
        elements: Vec<Pattern>,
        rest: Option<Box<Pattern>>,
        token_range: TokenRange,
    },
    Map {
        fields: Vec<MapFieldPattern>,
        token_range: TokenRange,
    },
    Literal {
        value: Box<Ast>,
        token_range: TokenRange,
    },
    Wildcard {
        token_range: TokenRange,
    },
}

impl Pattern {
    pub fn as_identifier(&self) -> Option<&str> {
        match self {
            Pattern::Identifier { name, .. } => Some(name),
            _ => None,
        }
    }

    pub fn is_identifier(&self) -> bool {
        matches!(self, Pattern::Identifier { .. })
    }

    pub fn token_range(&self) -> TokenRange {
        match self {
            Pattern::Identifier { token_range, .. }
            | Pattern::EnumVariant { token_range, .. }
            | Pattern::Struct { token_range, .. }
            | Pattern::Array { token_range, .. }
            | Pattern::Map { token_range, .. }
            | Pattern::Literal { token_range, .. }
            | Pattern::Wildcard { token_range, .. } => *token_range,
        }
    }

    pub fn binding_names(&self) -> Vec<String> {
        match self {
            Pattern::Identifier { name, .. } => vec![name.clone()],
            Pattern::EnumVariant { fields, .. } | Pattern::Struct { fields, .. } => fields
                .iter()
                .map(|f| {
                    f.binding_name
                        .clone()
                        .unwrap_or_else(|| f.field_name.clone())
                })
                .collect(),
            Pattern::Array { elements, rest, .. } => {
                let mut names: Vec<String> =
                    elements.iter().flat_map(|p| p.binding_names()).collect();
                if let Some(rest_pattern) = rest {
                    names.extend(rest_pattern.binding_names());
                }
                names
            }
            Pattern::Map { fields, .. } => fields.iter().map(|f| f.binding_name.clone()).collect(),
            Pattern::Literal { .. } | Pattern::Wildcard { .. } => vec![],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FieldPattern {
    pub field_name: String,
    pub binding_name: Option<String>,
    pub token_range: TokenRange,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MapKey {
    Keyword(String),
    String(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MapFieldPattern {
    pub key: MapKey,
    pub binding_name: String,
    pub token_range: TokenRange,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArityCase {
    pub args: Vec<Pattern>,
    pub rest_param: Option<String>,
    pub body: Vec<Ast>,
    pub token_range: TokenRange,
}

impl Ast {
    pub fn token_range(&self) -> TokenRange {
        match self {
            Ast::Program { token_range, .. }
            | Ast::Function { token_range, .. }
            | Ast::Struct { token_range, .. }
            | Ast::Enum { token_range, .. }
            | Ast::EnumVariant { token_range, .. }
            | Ast::EnumStaticVariant { token_range, .. }
            | Ast::Protocol { token_range, .. }
            | Ast::Extend { token_range, .. }
            | Ast::FunctionStub { token_range, .. }
            | Ast::If { token_range, .. }
            | Ast::Condition { token_range, .. }
            | Ast::Add { token_range, .. }
            | Ast::Sub { token_range, .. }
            | Ast::Mul { token_range, .. }
            | Ast::Div { token_range, .. }
            | Ast::Modulo { token_range, .. }
            | Ast::Recurse { token_range, .. }
            | Ast::TailRecurse { token_range, .. }
            | Ast::Call { token_range, .. }
            | Ast::CallExpr { token_range, .. }
            | Ast::Let { token_range, .. }
            | Ast::LetMut { token_range, .. }
            | Ast::LetDynamic { token_range, .. }
            | Ast::Binding { token_range, .. }
            | Ast::Assignment { token_range, .. }
            | Ast::Namespace { token_range, .. }
            | Ast::Use { token_range, .. }
            | Ast::ShiftLeft { token_range, .. }
            | Ast::ShiftRight { token_range, .. }
            | Ast::ShiftRightZero { token_range, .. }
            | Ast::BitWiseAnd { token_range, .. }
            | Ast::BitWiseOr { token_range, .. }
            | Ast::BitWiseXor { token_range, .. }
            | Ast::And { token_range, .. }
            | Ast::Or { token_range, .. }
            | Ast::Not { token_range, .. }
            | Ast::Array { token_range, .. }
            | Ast::MapLiteral { token_range, .. }
            | Ast::SetLiteral { token_range, .. }
            | Ast::IndexOperator { token_range, .. }
            | Ast::Loop { token_range, .. }
            | Ast::While { token_range, .. }
            | Ast::Break { token_range, .. }
            | Ast::Continue { token_range, .. }
            | Ast::Return { token_range, .. }
            | Ast::For { token_range, .. }
            | Ast::StructCreation { token_range, .. }
            | Ast::PropertyAccess { token_range, .. }
            | Ast::EnumCreation { token_range, .. }
            | Ast::Try { token_range, .. }
            | Ast::Throw { token_range, .. }
            | Ast::Match { token_range, .. }
            | Ast::ProtocolDispatch { token_range, .. }
            | Ast::StructField { token_range, .. }
            | Ast::MultiArityFunction { token_range, .. }
            | Ast::StringInterpolation { token_range, .. }
            | Ast::Reset { token_range, .. }
            | Ast::Shift { token_range, .. }
            | Ast::Perform { token_range, .. }
            | Ast::Handle { token_range, .. }
            | Ast::Future { token_range, .. }
            | Ast::Test { token_range, .. } => *token_range,
            Ast::IntegerLiteral(_, position)
            | Ast::FloatLiteral(_, position)
            | Ast::Identifier(_, position)
            | Ast::String(_, position)
            | Ast::Keyword(_, position)
            | Ast::True(position)
            | Ast::False(position)
            | Ast::Null(position) => TokenRange::new(*position, *position),
        }
    }

    pub fn has_top_level(&self) -> bool {
        self.nodes().iter().any(|node| {
            matches!(node, Ast::Function { .. } | Ast::MultiArityFunction { .. })
                || !matches!(
                    node,
                    Ast::Struct { .. } | Ast::Namespace { .. } | Ast::Test { .. }
                )
        })
    }

    pub fn nodes(&self) -> &Vec<Ast> {
        match self {
            Ast::Program { elements, .. } => elements,
            _ => panic!("Only works on program"),
        }
    }

    pub fn name(&self) -> Option<String> {
        match self {
            Ast::Function { name, .. } => name.clone(),
            _ => panic!("Only works on function"),
        }
    }

    pub fn get_string(&self) -> String {
        match self {
            Ast::String(str, _) => str.replace("\"", ""),
            Ast::Keyword(str, _) => str.to_string(),
            Ast::Identifier(str, _) => str.to_string(),
            _ => panic!("Expected string"),
        }
    }

    pub fn uses(&self) -> Vec<Ast> {
        self.nodes()
            .iter()
            .filter(|node| matches!(node, Ast::Use { .. }))
            .cloned()
            .collect()
    }
}
