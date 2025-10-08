use crate::parser::Value;
use std::collections::HashMap;

/// Macro definition
pub struct Macro {
    pub params: Vec<String>,
    pub body: Value,
}

/// Macro expander maintains a table of macros and expands them
pub struct MacroExpander {
    macros: HashMap<String, Macro>,
    gensym_counter: std::cell::RefCell<usize>,
}

impl MacroExpander {
    pub fn new() -> Self {
        let mut expander = Self {
            macros: HashMap::new(),
            gensym_counter: std::cell::RefCell::new(0),
        };
        // Register built-in macros
        expander.register_builtins();
        expander
    }

    /// Generate a unique symbol (for unique block/value names)
    fn gensym(&self, prefix: &str) -> String {
        let mut counter = self.gensym_counter.borrow_mut();
        let id = *counter;
        *counter += 1;
        format!("{}_{}", prefix, id)
    }

    /// Register built-in macros
    fn register_builtins(&mut self) {
        // Register 'if' macro
        // Syntax: (if condition then-expr else-expr)
        // Expands to:
        // (block entry []
        //   condition-evaluation
        //   (op cf.cond_br :operands [%cond] :true then_block :false else_block))
        // (block then_block []
        //   then-expr
        //   (op cf.br :dest exit_block :args [%then_result]))
        // (block else_block []
        //   else-expr
        //   (op cf.br :dest exit_block :args [%else_result]))
        // (block exit_block [result_type]
        //   ...)

        // Note: We'll implement these as macros that generate the block structure
        // For now, users can write their own macros using defmacro
    }

    /// Check if a form is an IRDL dialect definition
    fn is_irdl_form(&self, form: &Value) -> Option<&str> {
        if let Value::List(elements) = form {
            if let Some(Value::Symbol(name)) = elements.first() {
                match name.as_str() {
                    "defirdl-dialect" => return Some("defirdl-dialect"),
                    "defirdl-op" => return Some("defirdl-op"),
                    "import-dialect" => return Some("import-dialect"),
                    "export-dialect" => return Some("export-dialect"),
                    _ => {}
                }
            }
        }
        None
    }

    /// Expand defirdl-dialect form
    /// (defirdl-dialect name :namespace "ns" :description "desc" (defirdl-op ...) ...)
    /// Expands to IRDL operations that register the dialect
    fn expand_irdl_dialect(&self, elements: &[Value]) -> Result<Value, String> {
        if elements.len() < 2 {
            return Err("defirdl-dialect requires at least a name".into());
        }

        let dialect_name = match &elements[1] {
            Value::Symbol(s) => s.clone(),
            _ => return Err("defirdl-dialect name must be a symbol".into()),
        };

        let mut namespace = dialect_name.clone();
        let mut description = String::new();
        let mut ops = Vec::new();

        // Parse keyword arguments and operations
        let mut i = 2;
        while i < elements.len() {
            match &elements[i] {
                Value::Keyword(kw) if kw == "namespace" => {
                    if i + 1 < elements.len() {
                        if let Value::String(s) = &elements[i + 1] {
                            namespace = s.clone();
                        }
                        i += 2;
                    } else {
                        return Err(":namespace requires a value".into());
                    }
                }
                Value::Keyword(kw) if kw == "description" => {
                    if i + 1 < elements.len() {
                        if let Value::String(s) = &elements[i + 1] {
                            description = s.clone();
                        }
                        i += 2;
                    } else {
                        return Err(":description requires a value".into());
                    }
                }
                Value::List(_) => {
                    // This is an operation definition - expand it
                    let expanded_op = self.expand_irdl_op(
                        if let Value::List(op_elements) = &elements[i] {
                            op_elements
                        } else {
                            &[]
                        }
                    )?;
                    ops.push(expanded_op);
                    i += 1;
                }
                _ => {
                    i += 1;
                }
            }
        }

        // Generate IRDL IR
        // We'll create a special form that the emitter recognizes
        Ok(Value::List(vec![
            Value::Symbol("irdl-dialect-definition".to_string()),
            Value::String(dialect_name),
            Value::String(namespace),
            Value::String(description),
            Value::Vector(ops),
        ]))
    }

    /// Expand defirdl-op form
    /// (defirdl-op name :operands [...] :results [...] :attributes [...] :traits [...])
    /// Returns metadata for the operation
    fn expand_irdl_op(&self, elements: &[Value]) -> Result<Value, String> {
        if elements.len() < 2 {
            return Err("defirdl-op requires at least a name".into());
        }

        let op_name = match &elements[1] {
            Value::Symbol(s) => s.clone(),
            _ => return Err("defirdl-op name must be a symbol".into()),
        };

        // Parse all keyword arguments
        let mut summary = String::new();
        let mut description = String::new();
        let mut operands = Vec::new();
        let mut results = Vec::new();
        let mut attributes = Vec::new();
        let mut traits = Vec::new();
        let mut constraints = Vec::new();

        let mut i = 2;
        while i < elements.len() {
            match &elements[i] {
                Value::Keyword(kw) => {
                    if i + 1 < elements.len() {
                        match kw.as_str() {
                            "summary" => {
                                if let Value::String(s) = &elements[i + 1] {
                                    summary = s.clone();
                                }
                            }
                            "description" => {
                                if let Value::String(s) = &elements[i + 1] {
                                    description = s.clone();
                                }
                            }
                            "operands" => {
                                if let Value::Vector(v) = &elements[i + 1] {
                                    operands = v.clone();
                                }
                            }
                            "results" => {
                                if let Value::Vector(v) = &elements[i + 1] {
                                    results = v.clone();
                                }
                            }
                            "attributes" => {
                                if let Value::Vector(v) = &elements[i + 1] {
                                    attributes = v.clone();
                                }
                            }
                            "traits" => {
                                if let Value::Vector(v) = &elements[i + 1] {
                                    traits = v.clone();
                                }
                            }
                            "constraints" => {
                                if let Value::Vector(v) = &elements[i + 1] {
                                    constraints = v.clone();
                                }
                            }
                            _ => {}
                        }
                        i += 2;
                    } else {
                        return Err(format!(":{} requires a value", kw));
                    }
                }
                _ => {
                    i += 1;
                }
            }
        }

        // Return operation metadata
        Ok(Value::List(vec![
            Value::Symbol("irdl-op-definition".to_string()),
            Value::String(op_name),
            Value::Map(vec![
                (Value::Keyword("summary".to_string()), Value::String(summary)),
                (Value::Keyword("description".to_string()), Value::String(description)),
                (Value::Keyword("operands".to_string()), Value::Vector(operands)),
                (Value::Keyword("results".to_string()), Value::Vector(results)),
                (Value::Keyword("attributes".to_string()), Value::Vector(attributes)),
                (Value::Keyword("traits".to_string()), Value::Vector(traits)),
                (Value::Keyword("constraints".to_string()), Value::Vector(constraints)),
            ]),
        ]))
    }

    /// Expand deftransform form
    /// (deftransform name :description "..." body)
    /// Expands to transform.sequence IR
    fn expand_deftransform(&self, elements: &[Value]) -> Result<Value, String> {
        if elements.len() < 3 {
            return Err("deftransform requires name and body".into());
        }

        let transform_name = match &elements[1] {
            Value::Symbol(s) => s.clone(),
            _ => return Err("deftransform name must be a symbol".into()),
        };

        let mut description = String::new();
        let mut body = None;

        let mut i = 2;
        while i < elements.len() {
            match &elements[i] {
                Value::Keyword(kw) if kw == "description" => {
                    if i + 1 < elements.len() {
                        if let Value::String(s) = &elements[i + 1] {
                            description = s.clone();
                        }
                        i += 2;
                    } else {
                        return Err(":description requires a value".into());
                    }
                }
                _ => {
                    // Assume this is the body
                    body = Some(elements[i].clone());
                    i += 1;
                }
            }
        }

        let body = body.ok_or("deftransform requires a body")?;

        Ok(Value::List(vec![
            Value::Symbol("transform-definition".to_string()),
            Value::String(transform_name),
            Value::String(description),
            body,
        ]))
    }

    /// Expand defpdl-pattern form
    /// (defpdl-pattern name :benefit N :match ... :rewrite ...)
    /// Expands to PDL pattern IR
    fn expand_pdl_pattern(&self, elements: &[Value]) -> Result<Value, String> {
        if elements.len() < 2 {
            return Err("defpdl-pattern requires at least a name".into());
        }

        let pattern_name = match &elements[1] {
            Value::Symbol(s) => s.clone(),
            _ => return Err("defpdl-pattern name must be a symbol".into()),
        };

        let mut benefit = 1;
        let mut description = String::new();
        let mut match_body = None;
        let mut rewrite_body = None;
        let mut constraint_body = Vec::new();

        let mut i = 2;
        while i < elements.len() {
            match &elements[i] {
                Value::Keyword(kw) => {
                    match kw.as_str() {
                        "benefit" => {
                            if i + 1 < elements.len() {
                                if let Value::Integer(n) = &elements[i + 1] {
                                    benefit = *n;
                                }
                                i += 2;
                            }
                        }
                        "description" => {
                            if i + 1 < elements.len() {
                                if let Value::String(s) = &elements[i + 1] {
                                    description = s.clone();
                                }
                                i += 2;
                            }
                        }
                        "match" => {
                            if i + 1 < elements.len() {
                                match_body = Some(elements[i + 1].clone());
                                i += 2;
                            }
                        }
                        "rewrite" => {
                            if i + 1 < elements.len() {
                                rewrite_body = Some(elements[i + 1].clone());
                                i += 2;
                            }
                        }
                        "constraint" => {
                            if i + 1 < elements.len() {
                                constraint_body.push(elements[i + 1].clone());
                                i += 2;
                            }
                        }
                        _ => {
                            i += 1;
                        }
                    }
                }
                _ => {
                    i += 1;
                }
            }
        }

        Ok(Value::List(vec![
            Value::Symbol("pdl-pattern-definition".to_string()),
            Value::String(pattern_name),
            Value::Map(vec![
                (Value::Keyword("benefit".to_string()), Value::Integer(benefit)),
                (Value::Keyword("description".to_string()), Value::String(description)),
                (Value::Keyword("match".to_string()), match_body.unwrap_or(Value::List(vec![]))),
                (Value::Keyword("rewrite".to_string()), rewrite_body.unwrap_or(Value::List(vec![]))),
                (Value::Keyword("constraints".to_string()), Value::Vector(constraint_body)),
            ]),
        ]))
    }

    /// Define a macro
    pub fn define_macro(&mut self, name: String, params: Vec<String>, body: Value) {
        self.macros.insert(name, Macro { params, body });
    }

    /// Check if a form is a macro call
    pub fn is_macro_call(&self, form: &Value) -> Option<String> {
        if let Value::List(elements) = form {
            if let Some(Value::Symbol(name)) = elements.first() {
                if self.macros.contains_key(name) {
                    return Some(name.clone());
                }
            }
        }
        None
    }

    /// Expand all macros in a form recursively
    pub fn expand(&self, form: &Value) -> Result<Value, String> {
        match form {
            Value::List(elements) if !elements.is_empty() => {
                // Check for special forms first
                if let Some(Value::Symbol(s)) = elements.first() {
                    match s.as_str() {
                        "quote" => {
                            // Don't expand inside quote
                            return Ok(form.clone());
                        }
                        "defmacro" => {
                            // Don't expand defmacro itself
                            return Ok(form.clone());
                        }
                        "quasiquote" => {
                            // Handle quasiquote
                            if elements.len() != 2 {
                                return Err("quasiquote requires exactly one argument".into());
                            }
                            return self.expand_quasiquote(&elements[1]);
                        }
                        "defirdl-dialect" => {
                            // Expand IRDL dialect definition
                            return self.expand_irdl_dialect(elements);
                        }
                        "defirdl-op" => {
                            // Expand IRDL operation definition
                            return self.expand_irdl_op(elements);
                        }
                        "deftransform" => {
                            // Expand Transform dialect transformation
                            return self.expand_deftransform(elements);
                        }
                        "defpdl-pattern" => {
                            // Expand PDL pattern definition
                            return self.expand_pdl_pattern(elements);
                        }
                        "import-dialect" | "export-dialect" | "import-transform" | "export-transform" | "export-pattern" => {
                            // Pass through for now - will be handled by runtime
                            return Ok(form.clone());
                        }
                        // "if" is now handled at the expression compiler level using scf.if
                        // No macro expansion needed
                        // "if" => {
                        //     // Expand if to block-based control flow
                        //     if elements.len() != 4 {
                        //         return Err("if requires: condition, then-expr, else-expr".into());
                        //     }
                        //     return self.expand_if(&elements[1], &elements[2], &elements[3]);
                        // }
                        _ => {}
                    }
                }

                // Check if this is a macro call
                if let Some(macro_name) = self.is_macro_call(form) {
                    // Separate macro args from :as clause
                    let args = &elements[1..];
                    let mut macro_args = Vec::new();
                    let mut as_clause = None;

                    let mut i = 0;
                    while i < args.len() {
                        if let Value::Keyword(kw) = &args[i] {
                            if kw == "as" && i + 1 < args.len() {
                                as_clause = Some(args[i + 1].clone());
                                i += 2;
                                continue;
                            }
                        }
                        macro_args.push(args[i].clone());
                        i += 1;
                    }

                    let mut expanded = self.expand_macro_call(&macro_name, &macro_args)?;

                    // Re-attach :as clause if present
                    if let (Some(as_name), Value::List(list)) = (as_clause, &mut expanded) {
                        list.push(Value::Keyword("as".to_string()));
                        list.push(as_name);
                    }

                    // Recursively expand the result
                    return self.expand(&expanded);
                }

                // Recursively expand all elements
                let mut expanded_elements = Vec::new();
                for element in elements {
                    expanded_elements.push(self.expand(element)?);
                }
                Ok(Value::List(expanded_elements))
            }
            Value::Vector(elements) => {
                let mut expanded_elements = Vec::new();
                for element in elements {
                    expanded_elements.push(self.expand(element)?);
                }
                Ok(Value::Vector(expanded_elements))
            }
            Value::Map(pairs) => {
                let mut expanded_pairs = Vec::new();
                for (k, v) in pairs {
                    expanded_pairs.push((self.expand(k)?, self.expand(v)?));
                }
                Ok(Value::Map(expanded_pairs))
            }
            _ => Ok(form.clone()),
        }
    }

    /// Expand a macro call
    fn expand_macro_call(&self, name: &str, args: &[Value]) -> Result<Value, String> {
        let macro_def = self.macros.get(name)
            .ok_or(format!("Macro not found: {}", name))?;

        if args.len() != macro_def.params.len() {
            return Err(format!(
                "Macro {} expects {} arguments, got {}",
                name,
                macro_def.params.len(),
                args.len()
            ));
        }

        // Create bindings
        let mut bindings = HashMap::new();
        for (param, arg) in macro_def.params.iter().zip(args.iter()) {
            bindings.insert(param.clone(), arg.clone());
        }

        // Substitute bindings in the macro body
        self.substitute(&macro_def.body, &bindings)
    }

    /// Substitute bindings in a template
    fn substitute(&self, template: &Value, bindings: &HashMap<String, Value>) -> Result<Value, String> {
        match template {
            Value::Symbol(s) => {
                // If this symbol is bound, substitute it
                if let Some(value) = bindings.get(s) {
                    Ok(value.clone())
                } else {
                    Ok(template.clone())
                }
            }
            Value::List(elements) => {
                let mut result = Vec::new();
                for element in elements {
                    result.push(self.substitute(element, bindings)?);
                }
                Ok(Value::List(result))
            }
            Value::Vector(elements) => {
                let mut result = Vec::new();
                for element in elements {
                    result.push(self.substitute(element, bindings)?);
                }
                Ok(Value::Vector(result))
            }
            Value::Map(pairs) => {
                let mut result = Vec::new();
                for (k, v) in pairs {
                    result.push((
                        self.substitute(k, bindings)?,
                        self.substitute(v, bindings)?,
                    ));
                }
                Ok(Value::Map(result))
            }
            _ => Ok(template.clone()),
        }
    }

    /// Expand quasiquote
    fn expand_quasiquote(&self, template: &Value) -> Result<Value, String> {
        match template {
            Value::List(elements) if !elements.is_empty() => {
                // Check for unquote
                if let Some(Value::Symbol(s)) = elements.first() {
                    if s == "unquote" {
                        if elements.len() != 2 {
                            return Err("unquote requires exactly one argument".into());
                        }
                        // Evaluate the unquoted expression
                        return self.expand(&elements[1]);
                    }
                }

                // Recursively expand quasiquote in list elements
                let mut result = Vec::new();
                for element in elements {
                    result.push(self.expand_quasiquote(element)?);
                }
                Ok(Value::List(result))
            }
            Value::Vector(elements) => {
                let mut result = Vec::new();
                for element in elements {
                    result.push(self.expand_quasiquote(element)?);
                }
                Ok(Value::Vector(result))
            }
            _ => Ok(template.clone()),
        }
    }

    /// Expand if to block-based control flow
    /// (if cond then else) =>
    /// (block entry [] cond-ops (cf.cond_br cond then else))
    /// (block then [] then-ops (cf.br exit [result]))
    /// (block else [] else-ops (cf.br exit [result]))
    /// (block exit [type] (return ^0))
    fn expand_if(&self, cond: &Value, then_expr: &Value, else_expr: &Value) -> Result<Value, String> {
        // Generate unique block names
        let entry = self.gensym("if_entry");
        let then_block = self.gensym("then");
        let else_block = self.gensym("else");
        let exit_block = self.gensym("exit");
        let cond_name = self.gensym("cond");
        let then_result = self.gensym("then_result");
        let else_result = self.gensym("else_result");

        // Expand the sub-expressions
        let cond_expanded = self.expand(cond)?;
        let then_expanded = self.expand(then_expr)?;
        let else_expanded = self.expand(else_expr)?;

        // Build the block structure
        // For now, return a special marker that tells the emitter to handle this specially
        // Actually, let's return the block structure that the emitter expects

        // Return a special list that marks this as an if-blocks form
        Ok(Value::List(vec![
            Value::Symbol("if-blocks".to_string()),
            Value::Symbol(entry.clone()),
            Value::Symbol(then_block.clone()),
            Value::Symbol(else_block.clone()),
            Value::Symbol(exit_block.clone()),
            Value::Symbol(cond_name.clone()),
            cond_expanded,
            then_expanded,
            else_expanded,
        ]))
    }

    /// Expand all top-level forms
    pub fn expand_all(&self, forms: &[Value]) -> Result<Vec<Value>, String> {
        let mut result = Vec::new();
        for form in forms {
            // Check if this is a defmacro
            if let Value::List(elements) = form {
                if let Some(Value::Symbol(s)) = elements.first() {
                    if s == "defmacro" {
                        // Don't include defmacro in output
                        continue;
                    }
                }
            }
            result.push(self.expand(form)?);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_macro() {
        let mut expander = MacroExpander::new();

        // Define a simple macro: (defmacro inc [x] (+ x 1))
        expander.define_macro(
            "inc".to_string(),
            vec!["x".to_string()],
            Value::List(vec![
                Value::Symbol("+".to_string()),
                Value::Symbol("x".to_string()),
                Value::Integer(1),
            ]),
        );

        // Expand (inc 5)
        let form = Value::List(vec![
            Value::Symbol("inc".to_string()),
            Value::Integer(5),
        ]);

        let expanded = expander.expand(&form).unwrap();

        // Should expand to (+ 5 1)
        assert_eq!(
            expanded,
            Value::List(vec![
                Value::Symbol("+".to_string()),
                Value::Integer(5),
                Value::Integer(1),
            ])
        );
    }
}
