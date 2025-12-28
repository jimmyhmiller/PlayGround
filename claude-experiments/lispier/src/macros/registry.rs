//! Macro registry for looking up macros by name

use std::collections::HashMap;

use super::builtins::{
    AddFMacro, AddIMacro, AndMacro, CallBangMacro, CallMacro, CondMacro, DefnMacro, DivFMacro,
    DivSIMacro, DivUIMacro, EqIMacro, ExternMacro, GeIMacro, GtIMacro, IfMacro, LeIMacro,
    LoopMacro, LtIMacro, MulFMacro, MulIMacro, NeIMacro, NullCheckMacro, NullPtrMacro, OrMacro,
    PrintI64Macro, PrintMacro, PrintlnMacro, PtrAtMacro, PtrLoadMacro, PtrOffsetMacro, PtrStoreMacro,
    QuasiquoteMacro, SubFMacro, SubIMacro, VarargCallMacro, WhenMacro,
};
use super::Macro;

/// Registry of available macros
pub struct MacroRegistry {
    macros: HashMap<String, Box<dyn Macro>>,
}

impl MacroRegistry {
    /// Create a new registry with built-in macros registered
    pub fn new() -> Self {
        let mut registry = Self {
            macros: HashMap::new(),
        };
        registry.register_builtins();
        registry
    }

    /// Create an empty registry (no built-in macros)
    pub fn empty() -> Self {
        Self {
            macros: HashMap::new(),
        }
    }

    /// Register the built-in macros
    fn register_builtins(&mut self) {
        // Core macros
        self.register(Box::new(DefnMacro));
        self.register(Box::new(WhenMacro));
        self.register(Box::new(CondMacro));
        self.register(Box::new(IfMacro));
        self.register(Box::new(AndMacro));
        self.register(Box::new(OrMacro));
        self.register(Box::new(QuasiquoteMacro));
        self.register(Box::new(LoopMacro));

        // Integer arithmetic: +i, -i, *i, /i, /ui
        self.register(Box::new(AddIMacro));
        self.register(Box::new(SubIMacro));
        self.register(Box::new(MulIMacro));
        self.register(Box::new(DivSIMacro));
        self.register(Box::new(DivUIMacro));

        // Float arithmetic: +f, -f, *f, /f
        self.register(Box::new(AddFMacro));
        self.register(Box::new(SubFMacro));
        self.register(Box::new(MulFMacro));
        self.register(Box::new(DivFMacro));

        // Integer comparisons: <=i, <i, >=i, >i, =i, !=i
        self.register(Box::new(LeIMacro));
        self.register(Box::new(LtIMacro));
        self.register(Box::new(GeIMacro));
        self.register(Box::new(GtIMacro));
        self.register(Box::new(EqIMacro));
        self.register(Box::new(NeIMacro));

        // Function calls: call, call!, vararg-call
        self.register(Box::new(CallMacro));
        self.register(Box::new(CallBangMacro));
        self.register(Box::new(VarargCallMacro));

        // Pointer operations: null-ptr, null?, ptr-load, ptr-store!, ptr-offset, ptr-at
        self.register(Box::new(NullPtrMacro));
        self.register(Box::new(NullCheckMacro));
        self.register(Box::new(PtrLoadMacro));
        self.register(Box::new(PtrStoreMacro));
        self.register(Box::new(PtrOffsetMacro));
        self.register(Box::new(PtrAtMacro));

        // External declarations: extern
        self.register(Box::new(ExternMacro));

        // Print macros: print, println, print-i64
        self.register(Box::new(PrintMacro));
        self.register(Box::new(PrintlnMacro));
        self.register(Box::new(PrintI64Macro));
    }

    /// Register a macro
    pub fn register(&mut self, macro_impl: Box<dyn Macro>) {
        self.macros
            .insert(macro_impl.name().to_string(), macro_impl);
    }

    /// Look up a macro by name
    pub fn get(&self, name: &str) -> Option<&dyn Macro> {
        self.macros.get(name).map(|m| m.as_ref())
    }

    /// Check if a macro with the given name exists
    pub fn contains(&self, name: &str) -> bool {
        self.macros.contains_key(name)
    }

    /// Get the names of all registered macros
    pub fn macro_names(&self) -> Vec<&str> {
        self.macros.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for MacroRegistry {
    fn default() -> Self {
        Self::new()
    }
}
