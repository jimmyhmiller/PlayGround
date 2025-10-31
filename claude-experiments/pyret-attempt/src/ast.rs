// Pyret AST in Rust
// Mirrors the structure from src/arr/trove/ast.arr

/// Source location information
#[derive(Debug, Clone, PartialEq)]
pub struct Loc {
    pub source: String,
    pub start_line: usize,
    pub start_column: usize,
    pub start_char: usize,
    pub end_line: usize,
    pub end_column: usize,
    pub end_char: usize,
}

impl Loc {
    pub fn builtin(name: &str) -> Self {
        Loc {
            source: format!("builtin:{}", name),
            start_line: 0,
            start_column: 0,
            start_char: 0,
            end_line: 0,
            end_column: 0,
            end_char: 0,
        }
    }

    pub fn dummy() -> Self {
        Self::builtin("dummy location")
    }
}

/// Name variants in Pyret
#[derive(Debug, Clone, PartialEq)]
pub enum Name {
    /// Underscore placeholder
    Underscore(Loc),
    /// Regular name
    Name { loc: Loc, name: String },
    /// Global name
    Global(String),
    /// Module-scoped global
    ModuleGlobal(String),
    /// Type-scoped global
    TypeGlobal(String),
    /// Generated atom with serial number
    Atom { base: String, serial: u64 },
}

impl Name {
    pub fn loc(&self) -> Option<&Loc> {
        match self {
            Name::Underscore(loc) | Name::Name { loc, .. } => Some(loc),
            _ => None,
        }
    }

    pub fn to_string(&self) -> String {
        match self {
            Name::Underscore(_) => "_".to_string(),
            Name::Name { name, .. } => name.clone(),
            Name::Global(s) | Name::ModuleGlobal(s) | Name::TypeGlobal(s) => s.clone(),
            Name::Atom { base, serial } => format!("{}{}", base, serial),
        }
    }
}

/// Program root node
#[derive(Debug, Clone)]
pub struct Program {
    pub loc: Loc,
    pub use_stmt: Option<Use>,
    pub provide: Provide,
    pub provided_types: ProvideTypes,
    pub provide_blocks: Vec<ProvideBlock>,
    pub imports: Vec<Import>,
    pub block: Expr,
}

/// Use statement
#[derive(Debug, Clone)]
pub struct Use {
    pub loc: Loc,
    pub name: Name,
    pub module: ImportType,
}

/// Provide statement
#[derive(Debug, Clone)]
pub enum Provide {
    Provide(Loc, Box<Expr>),
    ProvideStar(Loc),
    ProvideNone(Loc),
}

/// Provide types statement
#[derive(Debug, Clone)]
pub enum ProvideTypes {
    ProvideTypes(Loc, Ann),
    ProvideTypesStar(Loc),
    ProvideTypesNone(Loc),
}

/// Provide block
#[derive(Debug, Clone)]
pub struct ProvideBlock {
    pub loc: Loc,
    pub from: Option<Vec<Name>>,
    pub specs: Vec<ProvideSpec>,
}

/// Provide specification
#[derive(Debug, Clone)]
pub enum ProvideSpec {
    ProvideName(NameSpec),
    ProvideType(NameSpec),
    ProvideData(DataNameSpec, Option<HidingSpec>),
    ProvideModule(NameSpec),
}

/// Name specification for imports/provides
#[derive(Debug, Clone)]
pub enum NameSpec {
    All(Option<HidingSpec>),
    Name(Vec<Name>),
    NameAs(Vec<Name>, Name),
}

/// Data name specification
#[derive(Debug, Clone)]
pub enum DataNameSpec {
    All,
    Name(Vec<Name>),
}

/// Hiding specification
#[derive(Debug, Clone)]
pub struct HidingSpec {
    pub names: Vec<Name>,
}

/// Import statement
#[derive(Debug, Clone)]
pub enum Import {
    Include {
        loc: Loc,
        module: ImportType,
    },
    IncludeFrom {
        loc: Loc,
        module: Vec<Name>,
        specs: Vec<IncludeSpec>,
    },
    Import {
        loc: Loc,
        file: ImportType,
        name: Name,
    },
    ImportTypes {
        loc: Loc,
        file: ImportType,
        name: Name,
        types: Name,
    },
    ImportFields {
        loc: Loc,
        fields: Vec<Name>,
        file: ImportType,
    },
}

/// Include specification
#[derive(Debug, Clone)]
pub enum IncludeSpec {
    IncludeName(NameSpec),
    IncludeType(NameSpec),
    IncludeData(DataNameSpec, Option<HidingSpec>),
    IncludeModule(NameSpec),
}

/// Import type (source)
#[derive(Debug, Clone)]
pub enum ImportType {
    File(String),
    Special(Name, Vec<String>),
    Name(Name),
}

/// Binding (variable or tuple destructuring)
#[derive(Debug, Clone)]
pub enum Binding {
    Name {
        loc: Loc,
        shadow: bool,
        name: Name,
        ann: Option<Ann>,
    },
    Tuple {
        loc: Loc,
        fields: Vec<Binding>,
        as_name: Option<Box<Binding>>,
    },
}

/// Type parameter binding
#[derive(Debug, Clone)]
pub struct TypeBind {
    pub name: Name,
    pub params: Vec<Name>,
    pub ann: Ann,
}

/// Newtype binding
#[derive(Debug, Clone)]
pub struct NewtypeBind {
    pub name: Name,
    pub as_name: Name,
}

/// Type-let binding
#[derive(Debug, Clone)]
pub enum TypeLetBind {
    Type(TypeBind),
    Newtype(NewtypeBind),
}

/// Let binding
#[derive(Debug, Clone)]
pub enum LetBind {
    Let {
        binding: Binding,
        value: Box<Expr>,
    },
    Var {
        binding: Binding,
        value: Box<Expr>,
    },
}

/// Letrec binding (for recursive definitions)
#[derive(Debug, Clone)]
pub struct LetrecBind {
    pub binding: Binding,
    pub value: Box<Expr>,
}

/// Expression - the main AST node type
#[derive(Debug, Clone)]
pub enum Expr {
    /// Module definition
    Module {
        loc: Loc,
        answer: Box<Expr>,
        defined_modules: Vec<DefinedModule>,
        defined_values: Vec<DefinedValue>,
        defined_types: Vec<DefinedType>,
        checks: Box<Expr>,
    },

    /// Template placeholder
    Template(Loc),

    /// Type-let expression
    TypeLet {
        loc: Loc,
        binds: Vec<TypeLetBind>,
        body: Box<Expr>,
        blocky: bool,
    },

    /// Let expression
    Let {
        loc: Loc,
        binds: Vec<LetBind>,
        body: Box<Expr>,
        blocky: bool,
    },

    /// Letrec expression
    Letrec {
        loc: Loc,
        binds: Vec<LetrecBind>,
        body: Box<Expr>,
        blocky: bool,
    },

    /// Type instantiation
    Instantiate {
        loc: Loc,
        expr: Box<Expr>,
        params: Vec<Ann>,
    },

    /// Block of statements
    Block {
        loc: Loc,
        stmts: Vec<Expr>,
    },

    /// User-written block
    UserBlock {
        loc: Loc,
        body: Box<Expr>,
    },

    /// Function definition
    Fun {
        loc: Loc,
        name: String,
        params: Vec<Name>,
        args: Vec<Binding>,
        ann: Ann,
        doc: String,
        body: Box<Expr>,
        check_loc: Option<Loc>,
        check: Option<Box<Expr>>,
        blocky: bool,
    },

    /// Type alias definition
    Type {
        loc: Loc,
        name: Name,
        params: Vec<Name>,
        ann: Ann,
    },

    /// Newtype definition
    Newtype {
        loc: Loc,
        name: Name,
        as_name: Name,
    },

    /// Variable definition
    Var {
        loc: Loc,
        name: Binding,
        value: Box<Expr>,
    },

    /// Recursive definition
    Rec {
        loc: Loc,
        name: Binding,
        value: Box<Expr>,
    },

    /// Let binding (within a block)
    LetStmt {
        loc: Loc,
        name: Binding,
        value: Box<Expr>,
        keyword_val: bool,
    },

    /// Reference type
    Ref {
        loc: Loc,
        ann: Option<Ann>,
    },

    /// Contract statement
    Contract {
        loc: Loc,
        name: Name,
        params: Vec<Name>,
        ann: Ann,
    },

    /// When expression (conditional side effect)
    When {
        loc: Loc,
        test: Box<Expr>,
        block: Box<Expr>,
        blocky: bool,
    },

    /// Assignment
    Assign {
        loc: Loc,
        id: Name,
        value: Box<Expr>,
    },

    /// If-pipe expression
    IfPipe {
        loc: Loc,
        branches: Vec<IfPipeBranch>,
        blocky: bool,
    },

    /// If-pipe with else
    IfPipeElse {
        loc: Loc,
        branches: Vec<IfPipeBranch>,
        else_branch: Box<Expr>,
        blocky: bool,
    },

    /// If expression
    If {
        loc: Loc,
        branches: Vec<IfBranch>,
        blocky: bool,
    },

    /// If with else
    IfElse {
        loc: Loc,
        branches: Vec<IfBranch>,
        else_branch: Box<Expr>,
        blocky: bool,
    },

    /// Cases expression (pattern matching)
    Cases {
        loc: Loc,
        typ: Ann,
        val: Box<Expr>,
        branches: Vec<CasesBranch>,
        blocky: bool,
    },

    /// Cases with else
    CasesElse {
        loc: Loc,
        typ: Ann,
        val: Box<Expr>,
        branches: Vec<CasesBranch>,
        else_branch: Box<Expr>,
        blocky: bool,
    },

    /// Binary operation
    Op {
        loc: Loc,
        op_loc: Loc,
        op: String,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Check test
    CheckTest {
        loc: Loc,
        op: CheckOp,
        refinement: Option<Box<Expr>>,
        left: Box<Expr>,
        right: Option<Box<Expr>>,
        cause: Option<Box<Expr>>,
    },

    /// Lambda expression
    Lambda {
        loc: Loc,
        params: Vec<Name>,
        args: Vec<Binding>,
        ann: Ann,
        doc: String,
        body: Box<Expr>,
        check_loc: Option<Loc>,
        check: Option<Box<Expr>>,
        blocky: bool,
    },

    /// Method definition
    Method {
        loc: Loc,
        params: Vec<Name>,
        args: Vec<Binding>,
        ann: Ann,
        doc: String,
        body: Box<Expr>,
        check_loc: Option<Loc>,
        check: Option<Box<Expr>>,
        blocky: bool,
    },

    /// Extend expression
    Extend {
        loc: Loc,
        obj: Box<Expr>,
        fields: Vec<Member>,
    },

    /// Update expression
    Update {
        loc: Loc,
        obj: Box<Expr>,
        fields: Vec<Member>,
    },

    /// Object literal
    Obj {
        loc: Loc,
        fields: Vec<Member>,
    },

    /// Tuple literal
    Tuple {
        loc: Loc,
        fields: Vec<Expr>,
    },

    /// Tuple access
    TupleGet {
        loc: Loc,
        tuple: Box<Expr>,
        index: usize,
    },

    /// Array/list constructor
    Construct {
        loc: Loc,
        modifier: ConstructModifier,
        constructor: Box<Expr>,
        values: Vec<Expr>,
    },

    /// Function application
    App {
        loc: Loc,
        func: Box<Expr>,
        args: Vec<Expr>,
    },

    /// Primitive application
    PrimApp {
        loc: Loc,
        func: String,
        args: Vec<Expr>,
    },

    /// Qualified variable access with dot
    Dot {
        loc: Loc,
        obj: Box<Expr>,
        field: String,
    },

    /// Indexing with brackets
    Bracket {
        loc: Loc,
        obj: Box<Expr>,
        index: Box<Expr>,
    },

    /// Mutable field access
    GetBang {
        loc: Loc,
        obj: Box<Expr>,
        field: String,
    },

    /// For expression
    For {
        loc: Loc,
        iterator: Box<Expr>,
        bindings: Vec<ForBind>,
        ann: Ann,
        body: Box<Expr>,
        blocky: bool,
    },

    /// Check expression
    Check {
        loc: Loc,
        name: Option<String>,
        body: Box<Expr>,
    },

    /// Spy statement (debugging)
    Spy {
        loc: Loc,
        expr: Option<Box<Expr>>,
        fields: Vec<SpyField>,
    },

    /// Data definition
    Data {
        loc: Loc,
        name: Name,
        params: Vec<Name>,
        variants: Vec<Variant>,
        shared_members: Vec<Member>,
        check: Option<Box<Expr>>,
    },

    /// Table expression
    Table {
        loc: Loc,
        headers: Vec<TableHeader>,
        rows: Vec<Vec<Expr>>,
    },

    /// Table operations
    TableSelect {
        loc: Loc,
        columns: Vec<Name>,
        table: Box<Expr>,
    },
    TableFilter {
        loc: Loc,
        table: Box<Expr>,
        bindings: Vec<Binding>,
        predicate: Box<Expr>,
    },
    TableOrder {
        loc: Loc,
        table: Box<Expr>,
        columns: Vec<ColumnOrder>,
    },
    TableExtend {
        loc: Loc,
        table: Box<Expr>,
        bindings: Vec<Binding>,
        fields: Vec<TableExtendField>,
    },
    TableExtract {
        loc: Loc,
        column: Name,
        table: Box<Expr>,
    },
    TableUpdate {
        loc: Loc,
        table: Box<Expr>,
        bindings: Vec<Binding>,
        fields: Vec<Member>,
    },
    LoadTable {
        loc: Loc,
        headers: Vec<TableHeader>,
        specs: Vec<LoadTableSpec>,
    },

    /// Reactor expression
    Reactor {
        loc: Loc,
        fields: Vec<Member>,
    },

    /// Identifier
    Id {
        loc: Loc,
        name: Name,
    },

    /// Number literal
    Num {
        loc: Loc,
        value: f64,
    },

    /// Rational number literal
    Frac {
        loc: Loc,
        numerator: i64,
        denominator: i64,
    },

    /// Rough rational (approximate)
    RoughFrac {
        loc: Loc,
        numerator: i64,
        denominator: i64,
    },

    /// Boolean literal
    Bool {
        loc: Loc,
        value: bool,
    },

    /// String literal
    Str {
        loc: Loc,
        value: String,
    },
}

impl Expr {
    pub fn loc(&self) -> &Loc {
        match self {
            Expr::Module { loc, .. } | Expr::Template(loc) | Expr::TypeLet { loc, .. } |
            Expr::Let { loc, .. } | Expr::Letrec { loc, .. } | Expr::Instantiate { loc, .. } |
            Expr::Block { loc, .. } | Expr::UserBlock { loc, .. } | Expr::Fun { loc, .. } |
            Expr::Type { loc, .. } | Expr::Newtype { loc, .. } | Expr::Var { loc, .. } |
            Expr::Rec { loc, .. } | Expr::LetStmt { loc, .. } | Expr::Ref { loc, .. } |
            Expr::Contract { loc, .. } | Expr::When { loc, .. } | Expr::Assign { loc, .. } |
            Expr::IfPipe { loc, .. } | Expr::IfPipeElse { loc, .. } | Expr::If { loc, .. } |
            Expr::IfElse { loc, .. } | Expr::Cases { loc, .. } | Expr::CasesElse { loc, .. } |
            Expr::Op { loc, .. } | Expr::CheckTest { loc, .. } | Expr::Lambda { loc, .. } |
            Expr::Method { loc, .. } | Expr::Extend { loc, .. } | Expr::Update { loc, .. } |
            Expr::Obj { loc, .. } | Expr::Tuple { loc, .. } | Expr::TupleGet { loc, .. } |
            Expr::Construct { loc, .. } | Expr::App { loc, .. } | Expr::PrimApp { loc, .. } |
            Expr::Dot { loc, .. } | Expr::Bracket { loc, .. } | Expr::GetBang { loc, .. } |
            Expr::For { loc, .. } | Expr::Check { loc, .. } | Expr::Spy { loc, .. } |
            Expr::Data { loc, .. } | Expr::Table { loc, .. } | Expr::TableSelect { loc, .. } |
            Expr::TableFilter { loc, .. } | Expr::TableOrder { loc, .. } |
            Expr::TableExtend { loc, .. } | Expr::TableExtract { loc, .. } |
            Expr::TableUpdate { loc, .. } | Expr::LoadTable { loc, .. } |
            Expr::Reactor { loc, .. } | Expr::Id { loc, .. } | Expr::Num { loc, .. } |
            Expr::Frac { loc, .. } | Expr::RoughFrac { loc, .. } | Expr::Bool { loc, .. } |
            Expr::Str { loc, .. } => loc,
        }
    }
}

/// If branch
#[derive(Debug, Clone)]
pub struct IfBranch {
    pub loc: Loc,
    pub test: Box<Expr>,
    pub body: Box<Expr>,
}

/// If-pipe branch
#[derive(Debug, Clone)]
pub struct IfPipeBranch {
    pub loc: Loc,
    pub test: Box<Expr>,
    pub body: Box<Expr>,
}

/// Cases branch
#[derive(Debug, Clone)]
pub struct CasesBranch {
    pub loc: Loc,
    pub name: Name,
    pub args: Vec<Binding>,
    pub body: Box<Expr>,
}

/// For binding
#[derive(Debug, Clone)]
pub struct ForBind {
    pub binding: Binding,
    pub value: Box<Expr>,
}

/// Spy field
#[derive(Debug, Clone)]
pub enum SpyField {
    Field(Name, Box<Expr>),
    Expr(Box<Expr>),
}

/// Check operation
#[derive(Debug, Clone)]
pub enum CheckOp {
    Is,
    IsEqual,
    IsEqualTilde,
    IsSpaceship,
    IsRoughly,
    IsNotRoughly,
    IsNot,
    IsNotEqual,
    IsNotEqualTilde,
    IsNotSpaceship,
    Raises,
    RaisesOther,
    RaisesNot,
    Satisfies,
    SatisfiesNot,
    RaisesSatisfies,
    RaisesViolates,
}

/// Data variant
#[derive(Debug, Clone)]
pub struct Variant {
    pub loc: Loc,
    pub name: Name,
    pub members: Vec<VariantMember>,
    pub with_members: Vec<Member>,
}

/// Variant member
#[derive(Debug, Clone)]
pub struct VariantMember {
    pub is_ref: bool,
    pub binding: Binding,
}

/// Object/record member
#[derive(Debug, Clone)]
pub enum Member {
    Field {
        loc: Loc,
        name: String,
        value: Box<Expr>,
    },
    MutableField {
        loc: Loc,
        name: String,
        ann: Option<Ann>,
        value: Box<Expr>,
    },
    Method {
        loc: Loc,
        name: String,
        params: Vec<Name>,
        args: Vec<Binding>,
        ann: Ann,
        doc: String,
        body: Box<Expr>,
        check_loc: Option<Loc>,
        check: Option<Box<Expr>>,
        blocky: bool,
    },
}

/// Construct modifier (for list/array construction)
#[derive(Debug, Clone, Copy)]
pub enum ConstructModifier {
    Eager,
    Lazy,
}

/// Table header
#[derive(Debug, Clone)]
pub struct TableHeader {
    pub name: Name,
    pub ann: Option<Ann>,
}

/// Column ordering
#[derive(Debug, Clone)]
pub struct ColumnOrder {
    pub column: Name,
    pub direction: OrderDirection,
}

#[derive(Debug, Clone, Copy)]
pub enum OrderDirection {
    Ascending,
    Descending,
}

/// Table extend field
#[derive(Debug, Clone)]
pub enum TableExtendField {
    Field {
        name: String,
        ann: Option<Ann>,
        value: Box<Expr>,
    },
    Computed {
        name: String,
        ann: Option<Ann>,
        expr: Box<Expr>,
        column: Name,
    },
}

/// Load table specification
#[derive(Debug, Clone)]
pub enum LoadTableSpec {
    Source(Box<Expr>),
    Sanitize { column: Name, sanitizer: Box<Expr> },
}

/// Defined module (for module exports)
#[derive(Debug, Clone)]
pub struct DefinedModule {
    pub name: String,
    pub module_ref: Name,
}

/// Defined value (for module exports)
#[derive(Debug, Clone)]
pub struct DefinedValue {
    pub name: String,
    pub value_ref: Name,
}

/// Defined type (for module exports)
#[derive(Debug, Clone)]
pub struct DefinedType {
    pub name: String,
    pub typ: Ann,
}

/// Type annotation
#[derive(Debug, Clone)]
pub enum Ann {
    /// Named type
    Name {
        loc: Loc,
        name: Name,
    },

    /// Type application (parameterized type)
    App {
        loc: Loc,
        base: Box<Ann>,
        args: Vec<Ann>,
    },

    /// Arrow type (function type)
    Arrow {
        loc: Loc,
        args: Vec<Ann>,
        ret: Box<Ann>,
    },

    /// Record type
    Record {
        loc: Loc,
        fields: Vec<AnnField>,
    },

    /// Tuple type
    Tuple {
        loc: Loc,
        fields: Vec<Ann>,
    },

    /// Predicate type (refinement type)
    Pred {
        loc: Loc,
        base: Box<Ann>,
        predicate: Name,
    },

    /// Qualified type name
    Dot {
        loc: Loc,
        obj: Name,
        field: String,
    },

    /// Blank annotation (Any)
    Blank(Loc),
}

impl Ann {
    pub fn loc(&self) -> &Loc {
        match self {
            Ann::Name { loc, .. } | Ann::App { loc, .. } | Ann::Arrow { loc, .. } |
            Ann::Record { loc, .. } | Ann::Tuple { loc, .. } | Ann::Pred { loc, .. } |
            Ann::Dot { loc, .. } | Ann::Blank(loc) => loc,
        }
    }
}

/// Annotation field
#[derive(Debug, Clone)]
pub struct AnnField {
    pub name: String,
    pub ann: Ann,
}
