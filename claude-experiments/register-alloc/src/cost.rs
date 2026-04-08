//! Cost models for measuring the quality of a register allocation.

use crate::allocator::Allocation;
use crate::ir::Function;
use crate::target::Target;
use crate::types::*;

/// A breakdown of costs from an allocation.
#[derive(Debug, Clone, Default)]
pub struct AllocationCost {
    /// Total number of spill stores inserted.
    pub spill_stores: u64,
    /// Total number of spill loads (reloads) inserted.
    pub spill_loads: u64,
    /// Total number of register-to-register moves inserted.
    pub reg_moves: u64,
    /// Weighted total cost (using the cost model's weights).
    pub total_cost: f64,
    /// Per-block cost breakdown, if requested.
    pub block_costs: Vec<(BlockId, f64)>,
}

/// Trait for defining custom cost models.
///
/// Different users care about different things: embedded systems
/// might care most about code size (move count), HPC might care
/// about memory traffic (spills), and JIT compilers might want
/// a balance.
pub trait CostModel {
    /// Cost of a spill store (register -> stack).
    fn spill_store_cost(&self, class: RegClass) -> f64;

    /// Cost of a spill load / reload (stack -> register).
    fn spill_load_cost(&self, class: RegClass) -> f64;

    /// Cost of a register-to-register move.
    fn reg_move_cost(&self, class: RegClass) -> f64;

    /// Cost of a move on a block edge (for phi resolution).
    /// May differ from a normal move if it's on a cold path.
    fn edge_move_cost(&self, class: RegClass) -> f64 {
        self.reg_move_cost(class)
    }

    /// Optional: multiplier for costs in a block based on execution
    /// frequency. Default returns 1.0 for all blocks.
    /// Users with profile data can weight hot blocks higher.
    fn block_frequency(&self, block: BlockId) -> f64 {
        let _ = block;
        1.0
    }

    /// Evaluate the total cost of an allocation.
    fn evaluate<F: Function, T: Target>(
        &self,
        func: &F,
        target: &T,
        alloc: &Allocation,
    ) -> AllocationCost {
        let _ = (func, target);
        let mut cost = AllocationCost::default();

        for mv in &alloc.moves {
            use crate::allocator::{MoveOperand, MovePosition};
            let class = mv.class;
            let is_edge = matches!(mv.at, MovePosition::BlockEdge { .. });

            match (&mv.from, &mv.to) {
                (MoveOperand::Reg(_), MoveOperand::SpillSlot(_)) => {
                    cost.spill_stores += 1;
                    cost.total_cost += self.spill_store_cost(class);
                }
                (MoveOperand::SpillSlot(_), MoveOperand::Reg(_)) => {
                    cost.spill_loads += 1;
                    cost.total_cost += self.spill_load_cost(class);
                }
                (MoveOperand::Reg(_), MoveOperand::Reg(_)) => {
                    cost.reg_moves += 1;
                    if is_edge {
                        cost.total_cost += self.edge_move_cost(class);
                    } else {
                        cost.total_cost += self.reg_move_cost(class);
                    }
                }
                (MoveOperand::SpillSlot(_), MoveOperand::SpillSlot(_)) => {
                    // Memory-to-memory move = load + store
                    cost.spill_loads += 1;
                    cost.spill_stores += 1;
                    cost.total_cost +=
                        self.spill_load_cost(class) + self.spill_store_cost(class);
                }
                // Remat moves are just MOV imm — very cheap
                (MoveOperand::Remat(_), _) | (_, MoveOperand::Remat(_)) => {
                    cost.reg_moves += 1;
                    cost.total_cost += self.reg_move_cost(class);
                }
            }
        }

        cost
    }
}

/// A simple uniform cost model where all operations cost 1.0.
/// Good baseline for comparison.
pub struct UniformCostModel;

impl CostModel for UniformCostModel {
    fn spill_store_cost(&self, _class: RegClass) -> f64 {
        1.0
    }
    fn spill_load_cost(&self, _class: RegClass) -> f64 {
        1.0
    }
    fn reg_move_cost(&self, _class: RegClass) -> f64 {
        1.0
    }
}

/// A cost model that weights spills heavily (memory is expensive).
/// Typical for modern out-of-order processors.
pub struct MemoryExpensiveCostModel;

impl CostModel for MemoryExpensiveCostModel {
    fn spill_store_cost(&self, _class: RegClass) -> f64 {
        5.0
    }
    fn spill_load_cost(&self, _class: RegClass) -> f64 {
        4.0
    }
    fn reg_move_cost(&self, _class: RegClass) -> f64 {
        1.0
    }
}
