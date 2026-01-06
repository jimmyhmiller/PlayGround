//! Use-def chain analysis.
//!
//! Tracks where each variable is defined and where it is used.
//! In SSA form, each variable has exactly one definition.

use std::collections::{HashMap, HashSet};

use crate::traits::InstructionFactory;
use crate::translator::SSATranslator;
use crate::types::{BlockId, PhiId, SsaVariable};

use super::super::traits::{OptimizableValue, OptimizableInstruction};

/// Location of a definition or use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Location {
    /// In an instruction: (block, instruction index)
    Instruction { block: BlockId, index: usize },
    /// In a phi node
    Phi(PhiId),
}

/// Information about a variable definition.
#[derive(Debug, Clone)]
pub struct DefInfo {
    /// Where the variable is defined
    pub location: Location,
    /// All locations where this variable is used
    pub uses: HashSet<Location>,
}

/// Use-def chain analysis result.
///
/// In SSA form, each variable has exactly one definition.
/// This analysis tracks:
/// - Where each variable is defined
/// - Where each variable is used
#[derive(Debug, Clone)]
pub struct UseDefChains<V: OptimizableValue> {
    /// Map from variable to its definition info
    pub defs: HashMap<SsaVariable, DefInfo>,
    /// Map from location to variables used at that location
    pub uses_at: HashMap<Location, HashSet<SsaVariable>>,
    /// Variables that are used but not defined (parameters, globals)
    pub undefined_uses: HashSet<SsaVariable>,
    _phantom: std::marker::PhantomData<V>,
}

impl<V: OptimizableValue> UseDefChains<V> {
    /// Compute use-def chains for the given SSA program.
    pub fn compute<I, F>(translator: &SSATranslator<V, I, F>) -> Self
    where
        I: OptimizableInstruction<Value = V>,
        F: InstructionFactory<Instr = I>,
    {
        let mut defs: HashMap<SsaVariable, DefInfo> = HashMap::new();
        let mut uses_at: HashMap<Location, HashSet<SsaVariable>> = HashMap::new();

        // Collect definitions from instructions
        for block in &translator.blocks {
            for (idx, instr) in block.instructions.iter().enumerate() {
                if let Some(dest) = instr.destination() {
                    let location = Location::Instruction {
                        block: block.id,
                        index: idx,
                    };
                    defs.insert(
                        dest.clone(),
                        DefInfo {
                            location,
                            uses: HashSet::new(),
                        },
                    );
                }
            }
        }

        // Collect definitions from phi nodes
        for (phi_id, phi) in &translator.phis {
            if let Some(dest) = &phi.dest {
                let location = Location::Phi(*phi_id);
                defs.insert(
                    dest.clone(),
                    DefInfo {
                        location,
                        uses: HashSet::new(),
                    },
                );
            }
        }

        // Collect uses from instructions
        for block in &translator.blocks {
            for (idx, instr) in block.instructions.iter().enumerate() {
                let location = Location::Instruction {
                    block: block.id,
                    index: idx,
                };

                let mut vars_used = HashSet::new();
                instr.visit_values(|value| {
                    if let Some(var) = value.as_var() {
                        vars_used.insert(var.clone());

                        // Record use in the def's use set
                        if let Some(def_info) = defs.get_mut(var) {
                            def_info.uses.insert(location);
                        }
                    }
                });

                if !vars_used.is_empty() {
                    uses_at.insert(location, vars_used);
                }
            }
        }

        // Collect uses from phi operands
        for (phi_id, phi) in &translator.phis {
            let location = Location::Phi(*phi_id);
            let mut vars_used = HashSet::new();

            for operand in &phi.operands {
                if let Some(var) = operand.as_var() {
                    vars_used.insert(var.clone());

                    // Record use in the def's use set
                    if let Some(def_info) = defs.get_mut(var) {
                        def_info.uses.insert(location);
                    }
                }
            }

            if !vars_used.is_empty() {
                uses_at.insert(location, vars_used);
            }
        }

        // Find undefined uses
        let mut undefined_uses = HashSet::new();
        for vars in uses_at.values() {
            for var in vars {
                if !defs.contains_key(var) {
                    undefined_uses.insert(var.clone());
                }
            }
        }

        UseDefChains {
            defs,
            uses_at,
            undefined_uses,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the definition location for a variable.
    pub fn get_def(&self, var: &SsaVariable) -> Option<Location> {
        self.defs.get(var).map(|info| info.location)
    }

    /// Get all uses of a variable.
    pub fn get_uses(&self, var: &SsaVariable) -> Option<&HashSet<Location>> {
        self.defs.get(var).map(|info| &info.uses)
    }

    /// Check if a variable has any uses.
    pub fn has_uses(&self, var: &SsaVariable) -> bool {
        self.defs
            .get(var)
            .map(|info| !info.uses.is_empty())
            .unwrap_or(false)
    }

    /// Get the number of uses of a variable.
    pub fn use_count(&self, var: &SsaVariable) -> usize {
        self.defs
            .get(var)
            .map(|info| info.uses.len())
            .unwrap_or(0)
    }

    /// Check if a variable is dead (defined but never used).
    pub fn is_dead(&self, var: &SsaVariable) -> bool {
        self.defs
            .get(var)
            .map(|info| info.uses.is_empty())
            .unwrap_or(false)
    }

    /// Get all dead variables (defined but never used).
    pub fn dead_variables(&self) -> Vec<SsaVariable> {
        self.defs
            .iter()
            .filter(|(_, info)| info.uses.is_empty())
            .map(|(var, _)| var.clone())
            .collect()
    }

    /// Get variables used at a location.
    pub fn vars_used_at(&self, location: Location) -> Option<&HashSet<SsaVariable>> {
        self.uses_at.get(&location)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_location_equality() {
        let loc1 = Location::Instruction {
            block: BlockId(0),
            index: 5,
        };
        let loc2 = Location::Instruction {
            block: BlockId(0),
            index: 5,
        };
        let loc3 = Location::Instruction {
            block: BlockId(1),
            index: 5,
        };

        assert_eq!(loc1, loc2);
        assert_ne!(loc1, loc3);

        let phi_loc = Location::Phi(PhiId(0));
        assert_ne!(loc1, phi_loc);
    }

    #[test]
    fn test_def_info() {
        let def = DefInfo {
            location: Location::Instruction {
                block: BlockId(0),
                index: 0,
            },
            uses: HashSet::new(),
        };

        assert!(def.uses.is_empty());
    }
}
