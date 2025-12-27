use std::collections::HashMap;

use crate::value::Value;

/// Type representation
#[derive(Debug, Clone, PartialEq)]
pub struct Type {
    pub name: String,
}

impl Type {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// Function type (-> [args] [returns])
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionType {
    pub arg_types: Vec<Type>,
    pub return_types: Vec<Type>,
}

impl FunctionType {
    pub fn new() -> Self {
        Self {
            arg_types: Vec::new(),
            return_types: Vec::new(),
        }
    }
}

impl Default for FunctionType {
    fn default() -> Self {
        Self::new()
    }
}

/// Typed number (number with explicit type)
#[derive(Debug, Clone, PartialEq)]
pub struct TypedNumber {
    pub value: f64,
    pub typ: Type,
}

/// Typed MLIR literal (e.g., dense<[1,2,3]> with type tensor<3xi32>)
#[derive(Debug, Clone, PartialEq)]
pub struct TypedMLIRLiteral {
    pub literal: String,
    pub typ: Type,
}

/// Attribute value (for operation attributes)
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<AttributeValue>),
    Type(Type),
    FunctionType(FunctionType),
    TypedNumber(TypedNumber),
    /// Typed MLIR attribute literal: (: dense<[1,2,3]> tensor<3xi32>)
    TypedMLIRLiteral(TypedMLIRLiteral),
    /// MLIR attribute literal syntax like array<i32: 0, 1, 1>, dense<...>, etc.
    /// Stored with spaces already converted to commas for MLIR parsing.
    MLIRLiteral(String),
}

/// Block argument
#[derive(Debug, Clone, PartialEq)]
pub struct BlockArgument {
    pub name: String,
    pub typ: Option<Type>,
}

impl BlockArgument {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            typ: None,
        }
    }

    pub fn with_type(name: impl Into<String>, typ: Type) -> Self {
        Self {
            name: name.into(),
            typ: Some(typ),
        }
    }
}

/// Block (labeled block with arguments)
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub label: Option<String>,
    pub arguments: Vec<BlockArgument>,
    pub operations: Vec<Node>,
}

impl Block {
    pub fn new() -> Self {
        Self {
            label: None,
            arguments: Vec::new(),
            operations: Vec::new(),
        }
    }

    pub fn with_label(label: impl Into<String>) -> Self {
        Self {
            label: Some(label.into()),
            arguments: Vec::new(),
            operations: Vec::new(),
        }
    }
}

impl Default for Block {
    fn default() -> Self {
        Self::new()
    }
}

/// Region (contains blocks)
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    pub blocks: Vec<Block>,
}

impl Region {
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }
}

impl Default for Region {
    fn default() -> Self {
        Self::new()
    }
}

/// Operation (MLIR operation)
#[derive(Debug, Clone, PartialEq)]
pub struct Operation {
    pub name: String,
    pub namespace: Option<String>,
    pub attributes: HashMap<String, AttributeValue>,
    pub operands: Vec<Node>,
    pub regions: Vec<Region>,
    pub result_types: Vec<Type>,
}

impl Operation {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            namespace: None,
            attributes: HashMap::new(),
            operands: Vec::new(),
            regions: Vec::new(),
            result_types: Vec::new(),
        }
    }

    pub fn with_namespace(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            namespace: Some(namespace.into()),
            attributes: HashMap::new(),
            operands: Vec::new(),
            regions: Vec::new(),
            result_types: Vec::new(),
        }
    }

    pub fn qualified_name(&self) -> String {
        if let Some(ref ns) = self.namespace {
            format!("{}.{}", ns, self.name)
        } else {
            self.name.clone()
        }
    }
}

/// Module (top-level container)
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub body: Vec<Node>,
}

/// Require specification for loading another file
#[derive(Debug, Clone, PartialEq)]
pub struct Require {
    pub path: String,
    pub alias: String,
    pub is_project_relative: bool,
}

/// A compilation pass with optional attributes
#[derive(Debug, Clone, PartialEq)]
pub struct Pass {
    pub name: String,
    pub attributes: HashMap<String, String>,
}

impl Pass {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            attributes: HashMap::new(),
        }
    }

    pub fn with_attributes(name: impl Into<String>, attributes: HashMap<String, String>) -> Self {
        Self {
            name: name.into(),
            attributes,
        }
    }

    /// Convert to mlir-opt pass argument format
    pub fn to_pass_arg(&self, runtime_attrs: &HashMap<String, String>) -> String {
        let mut attrs = self.attributes.clone();
        // Merge runtime attributes (runtime takes precedence)
        for (k, v) in runtime_attrs {
            attrs.insert(k.clone(), v.clone());
        }

        if attrs.is_empty() {
            format!("--{}", self.name)
        } else {
            let attr_str = attrs
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(",");
            format!("--{}='{}'", self.name, attr_str)
        }
    }

    /// Convert to pipeline string format for parse_pass_pipeline
    /// Format: "pass-name" or "pass-name{attr1=value1 attr2=value2}"
    pub fn to_pipeline_string(&self, runtime_attrs: &HashMap<String, String>) -> String {
        let mut attrs = self.attributes.clone();
        // Merge runtime attributes (runtime takes precedence)
        for (k, v) in runtime_attrs {
            attrs.insert(k.clone(), v.clone());
        }

        if attrs.is_empty() {
            self.name.clone()
        } else {
            let attr_str = attrs
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(" ");
            format!("{}{{{}}}", self.name, attr_str)
        }
    }
}

/// A compilation target (e.g., rocm, cuda, cpu)
#[derive(Debug, Clone, PartialEq)]
pub struct Target {
    pub backend: String,
    pub passes: Vec<Pass>,
}

impl Target {
    pub fn new(backend: impl Into<String>) -> Self {
        Self {
            backend: backend.into(),
            passes: Vec::new(),
        }
    }
}

/// Compilation specification with multiple targets
#[derive(Debug, Clone, PartialEq)]
pub struct Compilation {
    pub targets: Vec<Target>,
}

impl Compilation {
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
        }
    }

    pub fn get_target(&self, backend: &str) -> Option<&Target> {
        self.targets.iter().find(|t| t.backend == backend)
    }
}

impl Default for Compilation {
    fn default() -> Self {
        Self::new()
    }
}

/// External FFI declaration
#[derive(Debug, Clone, PartialEq)]
pub struct Extern {
    /// The FFI library to load (e.g., "value-ffi" for Value manipulation)
    pub library: String,
}

impl Extern {
    pub fn new(library: impl Into<String>) -> Self {
        Self {
            library: library.into(),
        }
    }
}

/// Link a shared library for external symbols
#[derive(Debug, Clone, PartialEq)]
pub struct LinkLibrary {
    /// The library to link
    /// Can be a keyword like :c for libc, or a path string
    pub library: String,
}

impl LinkLibrary {
    pub fn new(library: impl Into<String>) -> Self {
        Self {
            library: library.into(),
        }
    }

    /// Get the platform-specific library path
    pub fn resolve_path(&self) -> String {
        match self.library.as_str() {
            ":c" | ":libc" => {
                // Platform-specific libc - use absolute paths
                #[cfg(target_os = "macos")]
                { "/usr/lib/libSystem.B.dylib".to_string() }
                #[cfg(target_os = "linux")]
                { "/lib/x86_64-linux-gnu/libc.so.6".to_string() }
                #[cfg(not(any(target_os = "macos", target_os = "linux")))]
                { "c".to_string() }
            }
            ":m" | ":libm" => {
                #[cfg(target_os = "macos")]
                { "/usr/lib/libSystem.B.dylib".to_string() }
                #[cfg(target_os = "linux")]
                { "/lib/x86_64-linux-gnu/libm.so.6".to_string() }
                #[cfg(not(any(target_os = "macos", target_os = "linux")))]
                { "m".to_string() }
            }
            // Otherwise treat as a path or library name
            _ => self.library.clone(),
        }
    }
}

/// Macro registration declaration
/// Registers a compiled function as a macro for use during expansion
#[derive(Debug, Clone, PartialEq)]
pub struct Defmacro {
    /// The name of the function to register as a macro
    pub name: String,
}

impl Defmacro {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// Require macros declaration
/// Loads and compiles a macro module, registering its macros for use in this module
#[derive(Debug, Clone, PartialEq)]
pub struct RequireMacros {
    /// The path to the macro module
    pub path: String,
    /// Whether the path is project-relative (starts with @)
    pub is_project_relative: bool,
}

impl RequireMacros {
    pub fn new(path: impl Into<String>) -> Self {
        let path = path.into();
        let is_project_relative = path.starts_with('@');
        Self {
            path,
            is_project_relative,
        }
    }
}

impl Module {
    pub fn new() -> Self {
        Self { body: Vec::new() }
    }
}

impl Default for Module {
    fn default() -> Self {
        Self::new()
    }
}

/// Binding (def or let)
#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub names: Vec<String>,
    pub value: Box<Node>,
}

impl Binding {
    pub fn new(names: Vec<String>, value: Node) -> Self {
        Self {
            names,
            value: Box::new(value),
        }
    }

    pub fn single(name: impl Into<String>, value: Node) -> Self {
        Self {
            names: vec![name.into()],
            value: Box::new(value),
        }
    }
}

/// Let expression: bindings vector and body expressions
#[derive(Debug, Clone, PartialEq)]
pub struct LetExpr {
    pub bindings: Vec<Binding>,
    pub body: Vec<Node>,
}

impl LetExpr {
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
            body: Vec::new(),
        }
    }
}

impl Default for LetExpr {
    fn default() -> Self {
        Self::new()
    }
}

/// Type annotation
#[derive(Debug, Clone, PartialEq)]
pub struct TypeAnnotation {
    pub value: Box<Node>,
    pub typ: Type,
}

impl TypeAnnotation {
    pub fn new(value: Node, typ: Type) -> Self {
        Self {
            value: Box::new(value),
            typ,
        }
    }
}

/// AST Node
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Module(Module),
    Operation(Operation),
    Region(Region),
    Block(Block),
    Def(Binding),
    Let(LetExpr),
    TypeAnnotation(TypeAnnotation),
    FunctionType(FunctionType),
    Literal(Value),
    Require(Require),
    RequireMacros(RequireMacros),
    Compilation(Compilation),
    Extern(Extern),
    Defmacro(Defmacro),
    LinkLibrary(LinkLibrary),
}

impl Node {
    pub fn module(module: Module) -> Self {
        Node::Module(module)
    }

    pub fn operation(operation: Operation) -> Self {
        Node::Operation(operation)
    }

    pub fn region(region: Region) -> Self {
        Node::Region(region)
    }

    pub fn block(block: Block) -> Self {
        Node::Block(block)
    }

    pub fn def(binding: Binding) -> Self {
        Node::Def(binding)
    }

    pub fn let_expr(let_expr: LetExpr) -> Self {
        Node::Let(let_expr)
    }

    pub fn type_annotation(annotation: TypeAnnotation) -> Self {
        Node::TypeAnnotation(annotation)
    }

    pub fn function_type(ft: FunctionType) -> Self {
        Node::FunctionType(ft)
    }

    pub fn literal(value: Value) -> Self {
        Node::Literal(value)
    }

    pub fn require(require: Require) -> Self {
        Node::Require(require)
    }

    pub fn require_macros(require_macros: RequireMacros) -> Self {
        Node::RequireMacros(require_macros)
    }

    pub fn compilation(compilation: Compilation) -> Self {
        Node::Compilation(compilation)
    }

    pub fn extern_decl(extern_decl: Extern) -> Self {
        Node::Extern(extern_decl)
    }

    pub fn defmacro(defmacro: Defmacro) -> Self {
        Node::Defmacro(defmacro)
    }

    pub fn is_module(&self) -> bool {
        matches!(self, Node::Module(_))
    }

    pub fn is_operation(&self) -> bool {
        matches!(self, Node::Operation(_))
    }

    pub fn is_region(&self) -> bool {
        matches!(self, Node::Region(_))
    }

    pub fn is_block(&self) -> bool {
        matches!(self, Node::Block(_))
    }

    pub fn is_def(&self) -> bool {
        matches!(self, Node::Def(_))
    }

    pub fn is_let(&self) -> bool {
        matches!(self, Node::Let(_))
    }

    pub fn is_type_annotation(&self) -> bool {
        matches!(self, Node::TypeAnnotation(_))
    }

    pub fn is_function_type(&self) -> bool {
        matches!(self, Node::FunctionType(_))
    }

    pub fn is_literal(&self) -> bool {
        matches!(self, Node::Literal(_))
    }

    pub fn is_require(&self) -> bool {
        matches!(self, Node::Require(_))
    }

    pub fn is_require_macros(&self) -> bool {
        matches!(self, Node::RequireMacros(_))
    }

    pub fn is_compilation(&self) -> bool {
        matches!(self, Node::Compilation(_))
    }

    pub fn is_extern(&self) -> bool {
        matches!(self, Node::Extern(_))
    }

    pub fn is_defmacro(&self) -> bool {
        matches!(self, Node::Defmacro(_))
    }

    pub fn as_module(&self) -> &Module {
        match self {
            Node::Module(m) => m,
            _ => panic!("expected Module"),
        }
    }

    pub fn as_operation(&self) -> &Operation {
        match self {
            Node::Operation(o) => o,
            _ => panic!("expected Operation"),
        }
    }

    pub fn as_region(&self) -> &Region {
        match self {
            Node::Region(r) => r,
            _ => panic!("expected Region"),
        }
    }

    pub fn as_block(&self) -> &Block {
        match self {
            Node::Block(b) => b,
            _ => panic!("expected Block"),
        }
    }

    pub fn as_def(&self) -> &Binding {
        match self {
            Node::Def(b) => b,
            _ => panic!("expected Def"),
        }
    }

    pub fn as_let(&self) -> &LetExpr {
        match self {
            Node::Let(l) => l,
            _ => panic!("expected Let"),
        }
    }

    pub fn as_type_annotation(&self) -> &TypeAnnotation {
        match self {
            Node::TypeAnnotation(t) => t,
            _ => panic!("expected TypeAnnotation"),
        }
    }

    pub fn as_function_type(&self) -> &FunctionType {
        match self {
            Node::FunctionType(ft) => ft,
            _ => panic!("expected FunctionType"),
        }
    }

    pub fn as_literal(&self) -> &Value {
        match self {
            Node::Literal(v) => v,
            _ => panic!("expected Literal"),
        }
    }

    pub fn as_require(&self) -> &Require {
        match self {
            Node::Require(r) => r,
            _ => panic!("expected Require"),
        }
    }

    pub fn as_require_macros(&self) -> &RequireMacros {
        match self {
            Node::RequireMacros(r) => r,
            _ => panic!("expected RequireMacros"),
        }
    }

    pub fn as_compilation(&self) -> &Compilation {
        match self {
            Node::Compilation(c) => c,
            _ => panic!("expected Compilation"),
        }
    }

    pub fn as_extern(&self) -> &Extern {
        match self {
            Node::Extern(e) => e,
            _ => panic!("expected Extern"),
        }
    }

    pub fn as_defmacro(&self) -> &Defmacro {
        match self {
            Node::Defmacro(d) => d,
            _ => panic!("expected Defmacro"),
        }
    }
}
