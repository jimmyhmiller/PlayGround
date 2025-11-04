//! Pyret Abstract Syntax Tree
//!
//! Complete AST node definitions matching the reference implementation.
//! All nodes are serializable to JSON with exact format matching.
//!
//! Reference: /pyret-lang/src/arr/trove/ast.arr

use serde::Serialize;

// ============================================================================
// SECTION 1: Source Locations
// ============================================================================

/// Source location information for AST nodes
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Loc {
    pub source: String,
    #[serde(rename = "start-line")]
    pub start_line: usize,
    #[serde(rename = "start-column")]
    pub start_column: usize,
    #[serde(rename = "start-char")]
    pub start_char: usize,
    #[serde(rename = "end-line")]
    pub end_line: usize,
    #[serde(rename = "end-column")]
    pub end_column: usize,
    #[serde(rename = "end-char")]
    pub end_char: usize,
}

impl Loc {
    pub fn new(
        source: String,
        start_line: usize,
        start_column: usize,
        start_char: usize,
        end_line: usize,
        end_column: usize,
        end_char: usize,
    ) -> Self {
        Loc {
            source,
            start_line,
            start_column,
            start_char,
            end_line,
            end_column,
            end_char,
        }
    }
}

// ============================================================================
// SECTION 2: Names (6 variants)
// ============================================================================

/// Name types in Pyret
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Name {
    /// Underscore wildcard pattern
    #[serde(rename = "s-underscore")]
    SUnderscore { l: Loc },

    /// Regular identifier name
    #[serde(rename = "s-name")]
    SName { l: Loc, s: String },

    /// Global name reference
    #[serde(rename = "s-global")]
    SGlobal { s: String },

    /// Module-level global reference
    #[serde(rename = "s-module-global")]
    SModuleGlobal { s: String },

    /// Type-level global reference
    #[serde(rename = "s-type-global")]
    STypeGlobal { s: String },

    /// Generated/atom name with serial number
    #[serde(rename = "s-atom")]
    SAtom { base: String, serial: u32 },
}

// ============================================================================
// SECTION 3: Type Annotations (12 variants)
// ============================================================================

/// Type annotation nodes
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Ann {
    /// No annotation (blank)
    #[serde(rename = "a-blank")]
    ABlank,

    /// Any type
    #[serde(rename = "a-any")]
    AAny { l: Loc },

    /// Named type
    #[serde(rename = "a-name")]
    AName { l: Loc, id: Name },

    /// Type variable
    #[serde(rename = "a-type-var")]
    ATypeVar { l: Loc, id: Name },

    /// Arrow/function type
    #[serde(rename = "a-arrow")]
    AArrow {
        l: Loc,
        args: Vec<Ann>,
        ret: Box<Ann>,
        #[serde(rename = "use-parens")]
        use_parens: bool,
    },

    /// Arrow type with argument names
    #[serde(rename = "a-arrow-argnames")]
    AArrowArgnames {
        l: Loc,
        args: Vec<AField>,
        ret: Box<Ann>,
        #[serde(rename = "use-parens")]
        use_parens: bool,
    },

    /// Method type
    #[serde(rename = "a-method")]
    AMethod {
        l: Loc,
        args: Vec<Ann>,
        ret: Box<Ann>,
    },

    /// Record type
    #[serde(rename = "a-record")]
    ARecord { l: Loc, fields: Vec<AField> },

    /// Tuple type
    #[serde(rename = "a-tuple")]
    ATuple { l: Loc, fields: Vec<Ann> },

    /// Type application (generic instantiation)
    #[serde(rename = "a-app")]
    AApp {
        l: Loc,
        ann: Box<Ann>,
        args: Vec<Ann>,
    },

    /// Predicate/refinement type
    #[serde(rename = "a-pred")]
    APred {
        l: Loc,
        ann: Box<Ann>,
        exp: Box<Expr>,
    },

    /// Qualified/dot type
    #[serde(rename = "a-dot")]
    ADot { l: Loc, obj: Name, field: String },
}

/// Annotation field (for record types, etc.)
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AField {
    #[serde(rename = "type")]
    pub node_type: String, // "a-field"
    pub l: Loc,
    pub name: String,
    pub ann: Ann,
}

// ============================================================================
// SECTION 4: Bindings
// ============================================================================

/// Binding forms
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Bind {
    /// Simple name binding
    #[serde(rename = "s-bind")]
    SBind {
        l: Loc,
        shadows: bool,
        id: Name,
        ann: Ann,
    },

    /// Tuple destructuring binding
    #[serde(rename = "s-tuple-bind")]
    STupleBind {
        l: Loc,
        fields: Vec<Bind>,
        #[serde(rename = "as-name")]
        as_name: Option<Box<Bind>>,
    },
}

/// Let binding (for let-expr)
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum LetBind {
    #[serde(rename = "s-let-bind")]
    SLetBind { l: Loc, b: Bind, value: Box<Expr> },

    #[serde(rename = "s-var-bind")]
    SVarBind { l: Loc, b: Bind, value: Box<Expr> },
}

/// Letrec binding
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LetrecBind {
    #[serde(rename = "type")]
    pub node_type: String, // "s-letrec-bind"
    pub l: Loc,
    pub b: Bind,
    pub value: Box<Expr>,
}

/// Type-let binding
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum TypeLetBind {
    #[serde(rename = "s-type-bind")]
    STypeBind {
        l: Loc,
        name: Name,
        params: Vec<Name>,
        ann: Ann,
    },

    #[serde(rename = "s-newtype-bind")]
    SNewtypeBind { l: Loc, name: Name, namet: Name },
}

/// For-loop binding
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ForBind {
    #[serde(rename = "type")]
    pub node_type: String, // "s-for-bind"
    pub l: Loc,
    pub bind: Bind,
    pub value: Box<Expr>,
}

// ============================================================================
// SECTION 5: Expressions (60+ variants)
// ============================================================================

/// Expression nodes
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Expr {
    // ========== Control Flow ==========

    /// If expression without else
    #[serde(rename = "s-if")]
    SIf {
        l: Loc,
        branches: Vec<IfBranch>,
        blocky: bool,
    },

    /// If expression with else
    #[serde(rename = "s-if-else")]
    SIfElse {
        l: Loc,
        branches: Vec<IfBranch>,
        #[serde(rename = "_else")]
        _else: Box<Expr>,
        blocky: bool,
    },

    /// If-pipe expression without else
    #[serde(rename = "s-if-pipe")]
    SIfPipe {
        l: Loc,
        branches: Vec<IfPipeBranch>,
        blocky: bool,
    },

    /// If-pipe expression with else
    #[serde(rename = "s-if-pipe-else")]
    SIfPipeElse {
        l: Loc,
        branches: Vec<IfPipeBranch>,
        #[serde(rename = "_else")]
        _else: Box<Expr>,
        blocky: bool,
    },

    /// Cases expression without else
    #[serde(rename = "s-cases")]
    SCases {
        l: Loc,
        typ: Ann,
        val: Box<Expr>,
        branches: Vec<CasesBranch>,
        blocky: bool,
    },

    /// Cases expression with else
    #[serde(rename = "s-cases-else")]
    SCasesElse {
        l: Loc,
        typ: Ann,
        val: Box<Expr>,
        branches: Vec<CasesBranch>,
        #[serde(rename = "_else")]
        _else: Box<Expr>,
        blocky: bool,
    },

    /// When expression (conditional execution)
    #[serde(rename = "s-when")]
    SWhen {
        l: Loc,
        test: Box<Expr>,
        block: Box<Expr>,
        blocky: bool,
    },

    // ========== Functions & Lambdas ==========

    /// Function definition
    #[serde(rename = "s-fun")]
    SFun {
        l: Loc,
        name: String,
        params: Vec<Name>,      // Type parameters
        args: Vec<Bind>,        // Value parameters
        ann: Ann,               // Return type
        doc: String,
        body: Box<Expr>,
        #[serde(rename = "_check-loc")]
        check_loc: Option<Loc>,
        #[serde(rename = "_check")]
        check: Option<Box<Expr>>,
        blocky: bool,
    },

    /// Lambda expression
    #[serde(rename = "s-lam")]
    SLam {
        l: Loc,
        name: String,
        params: Vec<Name>,
        args: Vec<Bind>,
        ann: Ann,
        doc: String,
        body: Box<Expr>,
        #[serde(rename = "_check-loc")]
        check_loc: Option<Loc>,
        #[serde(rename = "_check")]
        check: Option<Box<Expr>>,
        blocky: bool,
    },

    /// Method definition
    #[serde(rename = "s-method")]
    SMethod {
        l: Loc,
        name: String,
        params: Vec<Name>,
        args: Vec<Bind>,
        ann: Ann,
        doc: String,
        body: Box<Expr>,
        #[serde(rename = "_check-loc")]
        check_loc: Option<Loc>,
        #[serde(rename = "_check")]
        check: Option<Box<Expr>>,
        blocky: bool,
    },

    // ========== Bindings ==========

    /// Let binding statement
    #[serde(rename = "s-let")]
    SLet {
        l: Loc,
        name: Bind,
        value: Box<Expr>,
        #[serde(rename = "keyword-val")]
        keyword_val: bool,
    },

    /// Var binding statement
    #[serde(rename = "s-var")]
    SVar {
        l: Loc,
        name: Bind,
        value: Box<Expr>,
    },

    /// Rec binding statement
    #[serde(rename = "s-rec")]
    SRec {
        l: Loc,
        name: Bind,
        value: Box<Expr>,
    },

    /// Let expression with multiple bindings
    #[serde(rename = "s-let-expr")]
    SLetExpr {
        l: Loc,
        binds: Vec<LetBind>,
        body: Box<Expr>,
        blocky: bool,
    },

    /// Letrec expression
    #[serde(rename = "s-letrec")]
    SLetrec {
        l: Loc,
        binds: Vec<LetrecBind>,
        body: Box<Expr>,
        blocky: bool,
    },

    // ========== Type System ==========

    /// Type alias definition
    #[serde(rename = "s-type")]
    SType {
        l: Loc,
        name: Name,
        params: Vec<Name>,
        ann: Ann,
    },

    /// Newtype definition
    #[serde(rename = "s-newtype")]
    SNewtype {
        l: Loc,
        name: Name,
        namet: Name,
    },

    /// Type-let expression
    #[serde(rename = "s-type-let-expr")]
    STypeLetExpr {
        l: Loc,
        binds: Vec<TypeLetBind>,
        body: Box<Expr>,
        blocky: bool,
    },

    /// Contract statement
    #[serde(rename = "s-contract")]
    SContract {
        l: Loc,
        name: Name,
        params: Vec<Name>,
        ann: Ann,
    },

    // ========== Data Definitions ==========

    /// Data type definition
    #[serde(rename = "s-data")]
    SData {
        l: Loc,
        name: String,
        params: Vec<Name>,
        mixins: Vec<Box<Expr>>,
        variants: Vec<Variant>,
        #[serde(rename = "shared-members")]
        shared_members: Vec<Member>,
        #[serde(rename = "_check-loc")]
        check_loc: Option<Loc>,
        #[serde(rename = "_check")]
        check: Option<Box<Expr>>,
    },

    /// Data expression (data as value)
    #[serde(rename = "s-data-expr")]
    SDataExpr {
        l: Loc,
        name: String,
        params: Vec<Name>,
        mixins: Vec<Box<Expr>>,
        variants: Vec<Variant>,
        #[serde(rename = "shared-members")]
        shared_members: Vec<Member>,
        #[serde(rename = "_check-loc")]
        check_loc: Option<Loc>,
        #[serde(rename = "_check")]
        check: Option<Box<Expr>>,
    },

    // ========== Operators ==========

    /// Binary operation
    #[serde(rename = "s-op")]
    SOp {
        l: Loc,
        #[serde(rename = "op-l")]
        op_l: Loc,
        op: String,
        left: Box<Expr>,
        right: Box<Expr>,
    },

    /// Unary operation
    #[serde(rename = "s-unary-op")]
    SUnaryOp {
        l: Loc,
        #[serde(rename = "op-l")]
        op_l: Loc,
        op: String,
        arg: Box<Expr>,
    },

    // ========== Primitives ==========

    /// Number literal
    /// Stores both the parsed float value and original string to preserve precision
    #[serde(rename = "s-num")]
    SNum { l: Loc, n: f64, #[serde(skip)] original: Option<String> },

    /// Fraction literal (rational)
    #[serde(rename = "s-frac")]
    SFrac { l: Loc, num: i64, den: i64 },

    /// Rough fraction literal
    #[serde(rename = "s-rfrac")]
    SRfrac { l: Loc, num: i64, den: i64 },

    /// Boolean literal
    #[serde(rename = "s-bool")]
    SBool { l: Loc, b: bool },

    /// String literal
    #[serde(rename = "s-str")]
    SStr { l: Loc, s: String },

    // ========== Identifiers ==========

    /// Identifier reference
    #[serde(rename = "s-id")]
    SId { l: Loc, id: Name },

    /// Var identifier (mutable variable)
    #[serde(rename = "s-id-var")]
    SIdVar { l: Loc, id: Name },

    /// Letrec identifier
    #[serde(rename = "s-id-letrec")]
    SIdLetrec { l: Loc, id: Name, safe: bool },

    // ========== Collections ==========

    /// Object literal
    #[serde(rename = "s-obj")]
    SObj { l: Loc, fields: Vec<Member> },

    /// Array literal
    #[serde(rename = "s-array")]
    SArray { l: Loc, values: Vec<Box<Expr>> },

    /// Tuple literal
    #[serde(rename = "s-tuple")]
    STuple { l: Loc, fields: Vec<Box<Expr>> },

    /// Tuple element access
    #[serde(rename = "s-tuple-get")]
    STupleGet {
        l: Loc,
        tup: Box<Expr>,
        index: usize,
        #[serde(rename = "index-loc")]
        index_loc: Loc,
    },

    /// Constructor application
    #[serde(rename = "s-construct")]
    SConstruct {
        l: Loc,
        modifier: ConstructModifier,
        constructor: Box<Expr>,
        values: Vec<Box<Expr>>,
    },

    // ========== Access & Update ==========

    /// Dot field access
    #[serde(rename = "s-dot")]
    SDot {
        l: Loc,
        obj: Box<Expr>,
        field: String,
    },

    /// Bracket access
    #[serde(rename = "s-bracket")]
    SBracket {
        l: Loc,
        obj: Box<Expr>,
        field: Box<Expr>,
    },

    /// Get-bang (mutable field access)
    #[serde(rename = "s-get-bang")]
    SGetBang {
        l: Loc,
        obj: Box<Expr>,
        field: String,
    },

    /// Object extension
    #[serde(rename = "s-extend")]
    SExtend {
        l: Loc,
        supe: Box<Expr>,
        fields: Vec<Member>,
    },

    /// Object update
    #[serde(rename = "s-update")]
    SUpdate {
        l: Loc,
        supe: Box<Expr>,
        fields: Vec<Member>,
    },

    /// Assignment
    #[serde(rename = "s-assign")]
    SAssign {
        l: Loc,
        id: Name,
        value: Box<Expr>,
    },

    // ========== Application ==========

    /// Function application
    #[serde(rename = "s-app")]
    SApp {
        l: Loc,
        #[serde(rename = "_fun")]
        _fun: Box<Expr>,
        args: Vec<Box<Expr>>,
    },

    /// Prim application (internal)
    #[serde(rename = "s-prim-app")]
    SPrimApp {
        l: Loc,
        #[serde(rename = "_fun")]
        _fun: String,
        args: Vec<Box<Expr>>,
    },

    /// Prim value (internal)
    #[serde(rename = "s-prim-val")]
    SPrimVal { l: Loc, name: String },

    /// Type instantiation
    #[serde(rename = "s-instantiate")]
    SInstantiate {
        l: Loc,
        expr: Box<Expr>,
        params: Vec<Ann>,
    },

    // ========== Blocks ==========

    /// Block expression
    #[serde(rename = "s-block")]
    SBlock { l: Loc, stmts: Vec<Box<Expr>> },

    /// User block (block:)
    #[serde(rename = "s-user-block")]
    SUserBlock { l: Loc, body: Box<Expr> },

    // ========== Tables ==========

    /// Table literal
    #[serde(rename = "s-table")]
    STable {
        l: Loc,
        headers: Vec<FieldName>,
        rows: Vec<TableRow>,
    },

    /// Load table from source
    #[serde(rename = "s-load-table")]
    SLoadTable {
        l: Loc,
        headers: Vec<FieldName>,
        spec: Vec<LoadTableSpec>,
    },

    /// Table select operation
    #[serde(rename = "s-table-select")]
    STableSelect {
        l: Loc,
        columns: Vec<Name>,
        table: Box<Expr>,
    },

    /// Table filter operation
    #[serde(rename = "s-table-filter")]
    STableFilter {
        l: Loc,
        #[serde(rename = "column-binds")]
        column_binds: ColumnBinds,
        predicate: Box<Expr>,
    },

    /// Table order operation
    #[serde(rename = "s-table-order")]
    STableOrder {
        l: Loc,
        table: Box<Expr>,
        ordering: Vec<ColumnSort>,
    },

    /// Table extract operation
    #[serde(rename = "s-table-extract")]
    STableExtract {
        l: Loc,
        column: Name,
        table: Box<Expr>,
    },

    /// Table update operation
    #[serde(rename = "s-table-update")]
    STableUpdate {
        l: Loc,
        #[serde(rename = "column-binds")]
        column_binds: ColumnBinds,
        updates: Vec<Member>,
    },

    /// Table extend operation
    #[serde(rename = "s-table-extend")]
    STableExtend {
        l: Loc,
        #[serde(rename = "column-binds")]
        column_binds: ColumnBinds,
        extensions: Vec<TableExtendField>,
    },

    // ========== Iteration ==========

    /// For expression
    #[serde(rename = "s-for")]
    SFor {
        l: Loc,
        iterator: Box<Expr>,
        bindings: Vec<ForBind>,
        ann: Ann,
        body: Box<Expr>,
        blocky: bool,
    },

    // ========== Testing ==========

    /// Check block
    #[serde(rename = "s-check")]
    SCheck {
        l: Loc,
        name: Option<String>,
        body: Box<Expr>,
        #[serde(rename = "keyword-check")]
        keyword_check: bool,
    },

    /// Check test assertion
    #[serde(rename = "s-check-test")]
    SCheckTest {
        l: Loc,
        op: CheckOp,
        refinement: Option<Box<Expr>>,
        left: Box<Expr>,
        right: Option<Box<Expr>>,
        cause: Option<Box<Expr>>,
    },

    /// Check expression (as value)
    #[serde(rename = "s-check-expr")]
    SCheckExpr {
        l: Loc,
        expr: Box<Expr>,
        #[serde(rename = "op-l")]
        op_l: Loc,
        op: CheckOp,
        refinement: Option<Box<Expr>>,
        right: Option<Box<Expr>>,
    },

    // ========== Reactors ==========

    /// Reactor definition
    #[serde(rename = "s-reactor")]
    SReactor { l: Loc, fields: Vec<Member> },

    // ========== Spy ==========

    /// Spy block
    #[serde(rename = "s-spy-block")]
    SSpyBlock {
        l: Loc,
        message: Option<Box<Expr>>,
        contents: Vec<SpyField>,
    },

    // ========== Misc ==========

    /// Parenthesized expression
    #[serde(rename = "s-paren")]
    SParen { l: Loc, expr: Box<Expr> },

    /// Template expression
    #[serde(rename = "s-template")]
    STemplate { l: Loc },

    /// Undefined value
    #[serde(rename = "s-undefined")]
    SUndefined { l: Loc },

    /// Source location literal
    #[serde(rename = "s-srcloc")]
    SSrcloc { l: Loc, loc: Loc },
}

// ============================================================================
// SECTION 6: Members & Fields
// ============================================================================

/// Object/data member
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Member {
    /// Immutable data field
    #[serde(rename = "s-data-field")]
    SDataField {
        l: Loc,
        name: String,
        value: Box<Expr>,
    },

    /// Mutable field
    #[serde(rename = "s-mutable-field")]
    SMutableField {
        l: Loc,
        name: String,
        ann: Ann,
        value: Box<Expr>,
    },

    /// Method field
    #[serde(rename = "s-method-field")]
    SMethodField {
        l: Loc,
        name: String,
        params: Vec<Name>,
        args: Vec<Bind>,
        ann: Ann,
        doc: String,
        body: Box<Expr>,
        #[serde(rename = "_check-loc")]
        check_loc: Option<Loc>,
        #[serde(rename = "_check")]
        check: Option<Box<Expr>>,
        blocky: bool,
    },
}

// ============================================================================
// SECTION 7: Variants & Data
// ============================================================================

/// Data type variant
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Variant {
    /// Variant with constructor arguments
    #[serde(rename = "s-variant")]
    SVariant {
        l: Loc,
        #[serde(rename = "constr-loc")]
        constr_loc: Loc,
        name: String,
        members: Vec<VariantMember>,
        #[serde(rename = "with-members")]
        with_members: Vec<Member>,
    },

    /// Singleton variant (no arguments)
    #[serde(rename = "s-singleton-variant")]
    SSingletonVariant {
        l: Loc,
        name: String,
        #[serde(rename = "with-members")]
        with_members: Vec<Member>,
    },
}

/// Variant member (constructor argument)
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VariantMember {
    #[serde(rename = "type")]
    pub node_type: String, // "s-variant-member"
    pub l: Loc,
    #[serde(rename = "member-type")]
    pub member_type: VariantMemberType,
    pub bind: Bind,
}

/// Variant member type
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum VariantMemberType {
    #[serde(rename = "s-normal")]
    SNormal,

    #[serde(rename = "s-mutable")]
    SMutable,
}

// ============================================================================
// SECTION 8: Branches (if/cases)
// ============================================================================

/// If branch
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IfBranch {
    #[serde(rename = "type")]
    pub node_type: String, // "s-if-branch"
    pub l: Loc,
    pub test: Box<Expr>,
    pub body: Box<Expr>,
}

/// If-pipe branch
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IfPipeBranch {
    #[serde(rename = "type")]
    pub node_type: String, // "s-if-pipe-branch"
    pub l: Loc,
    pub test: Box<Expr>,
    pub body: Box<Expr>,
}

/// Cases branch
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum CasesBranch {
    /// Regular cases branch with bindings
    #[serde(rename = "s-cases-branch")]
    SCasesBranch {
        l: Loc,
        #[serde(rename = "pattern-loc")]
        pattern_loc: Loc,
        name: String,
        args: Vec<CasesBind>,
        body: Box<Expr>,
    },

    /// Singleton cases branch (no bindings)
    #[serde(rename = "s-singleton-cases-branch")]
    SSingletonCasesBranch {
        l: Loc,
        #[serde(rename = "pattern-loc")]
        pattern_loc: Loc,
        name: String,
        body: Box<Expr>,
    },
}

/// Cases binding
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CasesBind {
    #[serde(rename = "type")]
    pub node_type: String, // "s-cases-bind"
    pub l: Loc,
    #[serde(rename = "field-type")]
    pub field_type: CasesBindType,
    pub bind: Bind,
}

/// Cases bind type
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum CasesBindType {
    #[serde(rename = "s-normal")]
    SNormal,

    #[serde(rename = "s-mutable")]
    SMutable,
}

// ============================================================================
// SECTION 9: Imports/Exports
// ============================================================================

/// Import statement
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Import {
    /// Include module
    #[serde(rename = "s-include")]
    SInclude { l: Loc, import: ImportType },

    /// Include from module with specs
    #[serde(rename = "s-include-from")]
    SIncludeFrom {
        l: Loc,
        import: ImportType,
        names: Vec<IncludeSpec>,
    },

    /// Import module as name
    #[serde(rename = "s-import")]
    SImport {
        l: Loc,
        import: ImportType,
        name: Name,
    },

    /// Import specific fields from module
    #[serde(rename = "s-import-fields")]
    SImportFields {
        l: Loc,
        fields: Vec<Name>,
        import: ImportType,
    },

    /// Import types from module
    #[serde(rename = "s-import-types")]
    SImportTypes {
        l: Loc,
        import: ImportType,
        types: Vec<Name>,
        name: Name,
    },
}

/// Import type/source
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum ImportType {
    #[serde(rename = "s-const-import")]
    SConstImport { l: Loc, module: String },

    #[serde(rename = "s-special-import")]
    SSpecialImport { l: Loc, kind: String, args: Vec<String> },
}

/// Include specification
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum IncludeSpec {
    #[serde(rename = "s-include-name")]
    SIncludeName { l: Loc, name: NameSpec },

    #[serde(rename = "s-include-type")]
    SIncludeType { l: Loc, name: Name },

    #[serde(rename = "s-include-data")]
    SIncludeData { l: Loc, name: Name },

    #[serde(rename = "s-include-module")]
    SIncludeModule { l: Loc, name: Name },
}

/// Name specification
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum NameSpec {
    #[serde(rename = "s-star")]
    SStar { l: Loc },

    #[serde(rename = "s-module-ref")]
    SModuleRef { l: Loc, name: Name },

    #[serde(rename = "s-remote-ref")]
    SRemoteRef { l: Loc, uri: String, name: Name },

    #[serde(rename = "s-local-ref")]
    SLocalRef { l: Loc, name: Name },
}

/// Provide statement
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Provide {
    #[serde(rename = "s-provide")]
    SProvide { l: Loc, block: Box<Expr> },

    #[serde(rename = "s-provide-all")]
    SProvideAll { l: Loc },

    #[serde(rename = "s-provide-none")]
    SProvideNone { l: Loc },
}

/// Provide-types statement
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum ProvideTypes {
    #[serde(rename = "s-provide-types")]
    SProvideTypes { l: Loc, anns: Vec<Ann> },

    #[serde(rename = "s-provide-types-all")]
    SProvideTypesAll { l: Loc },

    #[serde(rename = "s-provide-types-none")]
    SProvideTypesNone { l: Loc },
}

/// Use statement
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Use {
    #[serde(rename = "type")]
    pub node_type: String, // "s-use"
    pub l: Loc,
    pub name: Name,
    pub module: ImportType,
}

/// Provide block
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProvideBlock {
    #[serde(rename = "type")]
    pub node_type: String, // "s-provide-block"
    pub l: Loc,
    pub path: Vec<String>,
    pub specs: Vec<ProvideSpec>,
}

/// Provide specification
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum ProvideSpec {
    #[serde(rename = "s-provide-name")]
    SProvideName { l: Loc, name: NameSpec },

    #[serde(rename = "s-provide-type")]
    SProvideType { l: Loc, name: Ann },

    #[serde(rename = "s-provide-data")]
    SProvideData { l: Loc, name: Ann },

    #[serde(rename = "s-provide-module")]
    SProvideModule { l: Loc, name: Name },
}

// ============================================================================
// SECTION 10: Table Operations
// ============================================================================

/// Table row
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TableRow {
    #[serde(rename = "type")]
    pub node_type: String, // "s-table-row"
    pub l: Loc,
    pub elems: Vec<Box<Expr>>,
}

/// Field name (table header)
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FieldName {
    #[serde(rename = "type")]
    pub node_type: String, // "s-field-name"
    pub l: Loc,
    pub name: String,
    pub ann: Ann,
}

/// Load table specification
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum LoadTableSpec {
    #[serde(rename = "s-sanitize")]
    SSanitize {
        l: Loc,
        name: Name,
        sanitizer: Box<Expr>,
    },

    #[serde(rename = "s-table-src")]
    STableSrc { l: Loc, src: Box<Expr> },
}

/// Table extend field
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum TableExtendField {
    #[serde(rename = "s-table-extend-field")]
    STableExtendField {
        l: Loc,
        name: String,
        value: Box<Expr>,
        ann: Ann,
    },

    #[serde(rename = "s-table-extend-reducer")]
    STableExtendReducer {
        l: Loc,
        name: String,
        reducer: Box<Expr>,
        col: Name,
        ann: Ann,
    },
}

/// Column bindings
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ColumnBinds {
    #[serde(rename = "type")]
    pub node_type: String, // "s-column-binds"
    pub l: Loc,
    pub binds: Vec<Bind>,
    pub table: Box<Expr>,
}

/// Column sort specification
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ColumnSort {
    #[serde(rename = "type")]
    pub node_type: String, // "s-column-sort"
    pub l: Loc,
    pub column: Name,
    pub direction: ColumnSortOrder,
}

/// Column sort order
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ColumnSortOrder {
    Ascending,
    Descending,
}

// ============================================================================
// SECTION 11: Check Operations
// ============================================================================

/// Check test operators
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum CheckOp {
    #[serde(rename = "s-op-is")]
    SOpIs { l: Loc },

    #[serde(rename = "s-op-is-roughly")]
    SOpIsRoughly { l: Loc },

    #[serde(rename = "s-op-is-not-roughly")]
    SOpIsNotRoughly { l: Loc },

    #[serde(rename = "s-op-is-op")]
    SOpIsOp { l: Loc, op: String },

    #[serde(rename = "s-op-is-not")]
    SOpIsNot { l: Loc },

    #[serde(rename = "s-op-is-not-op")]
    SOpIsNotOp { l: Loc, op: String },

    #[serde(rename = "s-op-satisfies")]
    SOpSatisfies { l: Loc },

    #[serde(rename = "s-op-satisfies-not")]
    SOpSatisfiesNot { l: Loc },

    #[serde(rename = "s-op-raises")]
    SOpRaises { l: Loc },

    #[serde(rename = "s-op-raises-other")]
    SOpRaisesOther { l: Loc },

    #[serde(rename = "s-op-raises-not")]
    SOpRaisesNot { l: Loc },

    #[serde(rename = "s-op-raises-satisfies")]
    SOpRaisesSatisfies { l: Loc },

    #[serde(rename = "s-op-raises-violates")]
    SOpRaisesViolates { l: Loc },
}

/// Spy field
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpyField {
    #[serde(rename = "type")]
    pub node_type: String, // "s-spy-expr"
    pub l: Loc,
    pub name: Option<String>,
    pub value: Box<Expr>,
    #[serde(rename = "implicit-label")]
    pub implicit_label: bool,
}

// ============================================================================
// SECTION 12: Top-level Program
// ============================================================================

/// Construct modifier
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum ConstructModifier {
    #[serde(rename = "s-construct-normal")]
    SConstructNormal,

    #[serde(rename = "s-construct-lazy")]
    SConstructLazy,
}

/// Top-level program
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Program {
    #[serde(rename = "type")]
    pub node_type: String, // "s-program"
    pub l: Loc,
    #[serde(rename = "_use")]
    pub _use: Option<Use>,
    #[serde(rename = "_provide")]
    pub _provide: Provide,
    #[serde(rename = "provided-types")]
    pub provided_types: ProvideTypes,
    pub provides: Vec<ProvideBlock>,
    pub imports: Vec<Import>,
    pub body: Box<Expr>,
}

impl Program {
    pub fn new(
        l: Loc,
        _use: Option<Use>,
        _provide: Provide,
        provided_types: ProvideTypes,
        provides: Vec<ProvideBlock>,
        imports: Vec<Import>,
        body: Expr,
    ) -> Self {
        Program {
            node_type: "s-program".to_string(),
            l,
            _use,
            _provide,
            provided_types,
            provides,
            imports,
            body: Box::new(body),
        }
    }
}
