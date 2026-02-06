//! State machine detection and extraction
//!
//! Identifies the while-switch pattern used in control flow flattening:
//! ```javascript
//! while (state >= 0) {
//!     switch (state & mask) {
//!         case 0: ...; state = next; break;
//!         case 1: ...; break;
//!     }
//! }
//! ```

use std::collections::HashMap;
use swc_ecma_ast::*;

use crate::parser::{expr_as_ident, expr_as_number};

/// A detected state machine
#[derive(Debug)]
pub struct StateMachine {
    /// The variable used to track state
    pub state_var: String,
    /// The mask applied to state (from `state & mask`)
    pub mask: Option<i64>,
    /// The cases in the switch, keyed by case value
    pub cases: HashMap<i64, SwitchCase>,
    /// The default case, if any
    pub default: Option<Vec<Stmt>>,
    /// Initial state value (from variable declaration)
    pub initial_state: Option<i64>,
}

/// A single case in the state machine
#[derive(Debug, Clone)]
pub struct SwitchCase {
    /// The statements in this case
    pub stmts: Vec<Stmt>,
    /// Detected state transition (if any)
    pub next_state: Option<StateTransition>,
}

/// How the state transitions
#[derive(Debug, Clone)]
pub enum StateTransition {
    /// Unconditional: state = N
    Goto(i64),
    /// Exit the state machine: state = -1
    Exit,
    /// Conditional: state = cond ? a : b
    Branch {
        condition: Box<Expr>,
        if_true: i64,
        if_false: i64,
    },
    /// Dynamic: state = expr (we don't know the value)
    Dynamic(Box<Expr>),
}

/// Detect if a statement is a while-switch state machine
pub fn detect_state_machine(stmt: &Stmt) -> Option<StateMachine> {
    // Look for: while (state >= 0) { switch (...) { ... } }
    let while_stmt = match stmt {
        Stmt::While(w) => w,
        _ => return None,
    };

    // Check condition: state >= 0
    let state_var = extract_state_var_from_condition(&while_stmt.test)?;

    // Check body: switch (state & mask) { ... }
    let switch_stmt = extract_switch_from_body(&while_stmt.body)?;

    // Extract the mask from switch discriminant
    let (switch_state_var, mask) = extract_switch_discriminant(&switch_stmt.discriminant)?;

    // Verify it's the same state variable
    if switch_state_var != state_var {
        return None;
    }

    // Extract cases
    let mut cases = HashMap::new();
    let mut default = None;

    for case in &switch_stmt.cases {
        if let Some(test) = &case.test {
            // Regular case
            let case_value = expr_as_number(test)? as i64;
            let (stmts, next_state) = extract_case_body(&case.cons, &state_var);
            cases.insert(
                case_value,
                SwitchCase {
                    stmts,
                    next_state,
                },
            );
        } else {
            // Default case
            let (stmts, _) = extract_case_body(&case.cons, &state_var);
            default = Some(stmts);
        }
    }

    Some(StateMachine {
        state_var,
        mask,
        cases,
        default,
        initial_state: None,
    })
}

/// Extract state variable from condition like `state >= 0`
fn extract_state_var_from_condition(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Bin(BinExpr {
            op: BinaryOp::GtEq,
            left,
            right,
            ..
        }) => {
            // state >= 0
            let var = expr_as_ident(left)?;
            let zero = expr_as_number(right)?;
            if zero == 0.0 {
                Some(var.to_string())
            } else {
                None
            }
        }
        Expr::Bin(BinExpr {
            op: BinaryOp::Gt,
            left,
            right,
            ..
        }) => {
            // state > -1 (equivalent)
            let var = expr_as_ident(left)?;
            let minus_one = expr_as_number(right)?;
            if minus_one == -1.0 {
                Some(var.to_string())
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Extract switch statement from while body
fn extract_switch_from_body(body: &Stmt) -> Option<&SwitchStmt> {
    match body {
        Stmt::Switch(s) => Some(s),
        Stmt::Block(b) => {
            // Might be wrapped in a block
            if b.stmts.len() == 1 {
                if let Stmt::Switch(s) = &b.stmts[0] {
                    return Some(s);
                }
            }
            None
        }
        _ => None,
    }
}

/// Extract state variable and mask from switch discriminant like `state & 7`
fn extract_switch_discriminant(expr: &Expr) -> Option<(String, Option<i64>)> {
    match expr {
        Expr::Ident(id) => {
            // Just `switch (state)`
            Some((id.sym.to_string(), None))
        }
        Expr::Bin(BinExpr {
            op: BinaryOp::BitAnd,
            left,
            right,
            ..
        }) => {
            // `switch (state & mask)`
            let var = expr_as_ident(left)?;
            let mask = expr_as_number(right)? as i64;
            Some((var.to_string(), Some(mask)))
        }
        _ => None,
    }
}

/// Extract statements from case body and detect state transition
fn extract_case_body(stmts: &[Stmt], state_var: &str) -> (Vec<Stmt>, Option<StateTransition>) {
    let mut result_stmts = Vec::new();
    let mut transition = None;

    for stmt in stmts {
        match stmt {
            // Skip break statements
            Stmt::Break(_) => {}

            // Check for state assignment
            Stmt::Expr(ExprStmt { expr, .. }) => {
                if let Some(t) = check_state_assignment(expr, state_var) {
                    transition = Some(t);
                } else {
                    result_stmts.push(stmt.clone());
                }
            }

            _ => {
                result_stmts.push(stmt.clone());
            }
        }
    }

    (result_stmts, transition)
}

/// Check if an expression is a state assignment and extract the transition
fn check_state_assignment(expr: &Expr, state_var: &str) -> Option<StateTransition> {
    match expr {
        Expr::Assign(AssignExpr {
            op: AssignOp::Assign,
            left,
            right,
            ..
        }) => {
            // Check if assigning to state variable
            let target = match left {
                AssignTarget::Simple(SimpleAssignTarget::Ident(id)) => &id.id.sym,
                _ => return None,
            };

            if target != state_var {
                return None;
            }

            // Check what value is being assigned
            if let Some(n) = expr_as_number(right) {
                let n = n as i64;
                if n < 0 {
                    Some(StateTransition::Exit)
                } else {
                    Some(StateTransition::Goto(n))
                }
            } else if let Expr::Cond(cond) = right.as_ref() {
                // state = cond ? a : b
                let if_true = expr_as_number(&cond.cons)? as i64;
                let if_false = expr_as_number(&cond.alt)? as i64;
                Some(StateTransition::Branch {
                    condition: cond.test.clone(),
                    if_true,
                    if_false,
                })
            } else {
                // Dynamic assignment
                Some(StateTransition::Dynamic(right.clone()))
            }
        }
        _ => None,
    }
}

/// Find all state machines in a module
pub fn find_state_machines(module: &Module) -> Vec<(usize, StateMachine)> {
    let mut result = Vec::new();

    for (i, item) in module.body.iter().enumerate() {
        if let ModuleItem::Stmt(stmt) = item {
            if let Some(sm) = detect_state_machine(stmt) {
                result.push((i, sm));
            }
        }
    }

    result
}

/// Find state machines in a function body
pub fn find_state_machines_in_stmts(stmts: &[Stmt]) -> Vec<(usize, StateMachine)> {
    let mut result = Vec::new();

    for (i, stmt) in stmts.iter().enumerate() {
        if let Some(sm) = detect_state_machine(stmt) {
            result.push((i, sm));
        }

        // Also check inside function declarations
        if let Stmt::Decl(Decl::Fn(fn_decl)) = stmt {
            if let Some(body) = &fn_decl.function.body {
                let nested = find_state_machines_in_stmts(&body.stmts);
                for (j, sm) in nested {
                    result.push((i * 1000 + j, sm)); // Encode nested position
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_js;

    #[test]
    fn test_detect_simple_state_machine() {
        let source = r#"
            var state = 200;
            while (state >= 0) {
                switch (state & 1) {
                    case 0:
                        console.log("case 0");
                        state = -1;
                        break;
                    case 1:
                        state = -1;
                        break;
                }
            }
        "#;

        let module = parse_js(source).unwrap();
        let machines = find_state_machines(&module);

        assert_eq!(machines.len(), 1);
        let (_, sm) = &machines[0];
        assert_eq!(sm.state_var, "state");
        assert_eq!(sm.mask, Some(1));
        assert_eq!(sm.cases.len(), 2);
    }

    #[test]
    fn test_detect_branch_transition() {
        let source = r#"
            while (state >= 0) {
                switch (state & 7) {
                    case 3:
                        state = condition ? 5 : 2;
                        break;
                }
            }
        "#;

        let module = parse_js(source).unwrap();
        let machines = find_state_machines(&module);

        assert_eq!(machines.len(), 1);
        let (_, sm) = &machines[0];

        let case3 = sm.cases.get(&3).unwrap();
        match &case3.next_state {
            Some(StateTransition::Branch {
                if_true, if_false, ..
            }) => {
                assert_eq!(*if_true, 5);
                assert_eq!(*if_false, 2);
            }
            _ => panic!("Expected branch transition"),
        }
    }
}
