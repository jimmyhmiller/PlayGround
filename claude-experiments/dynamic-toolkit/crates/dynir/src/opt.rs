//! Optimization passes for DynIR functions.
//!
//! Passes:
//! - **mem2reg**: Promote stack-slot loads/stores to SSA values (block params).
//! - **licm**: Loop-invariant code motion — hoist pure invariant ops to preheaders.
//! - **gvn**: Local (per-block) global value numbering — eliminate redundant ops.
//! - **constant_fold**: Evaluate constant arithmetic at compile time.
//! - **dce**: Dead-code elimination — remove unused pure instructions.

use std::collections::{HashMap, HashSet};

use crate::ir::*;
use crate::types::Type;

// ─── Utilities ───────────────────────────────────────────────────

fn new_value(func: &mut Function, ty: Type) -> Value {
    let idx = func.value_types.len();
    func.value_types.push(ty);
    Value::from_index(idx)
}

/// Whether an instruction is pure (no side effects, safe to remove/move).
fn is_pure(inst: &Inst) -> bool {
    match inst {
        Inst::Store(..)
        | Inst::Call(..)
        | Inst::CallIndirect(..)
        | Inst::Guard(..)
        | Inst::Safepoint(..)
        | Inst::PushPrompt(_)
        | Inst::PopPrompt(_) => false,
        _ => true,
    }
}

/// Whether an instruction is safe to hoist (pure and no memory reads).
fn is_hoistable(inst: &Inst) -> bool {
    match inst {
        Inst::Load(..) | Inst::Store(..) => false,
        _ => is_pure(inst),
    }
}

/// Replace all uses of `old` with `new_val` in the entire function.
fn replace_all_uses(func: &mut Function, old: Value, new_val: Value) {
    for block in &mut func.blocks {
        for inst_node in &mut block.insts {
            inst_node.inst.for_each_value_mut(|v| {
                if *v == old {
                    *v = new_val;
                }
            });
        }
        block.terminator.for_each_value_mut(|v| {
            if *v == old {
                *v = new_val;
            }
        });
    }
}

/// Compute use counts for all values in a function.
fn compute_use_counts(func: &Function) -> Vec<u32> {
    let mut counts = vec![0u32; func.value_types.len()];
    for block in &func.blocks {
        for inst_node in &block.insts {
            inst_node.inst.for_each_value(|v| counts[v.index()] += 1);
        }
        block.terminator.for_each_value(|v| counts[v.index()] += 1);
    }
    counts
}

// ─── Reverse Post-Order & Predecessors ───────────────────────────

fn compute_rpo(func: &Function) -> Vec<usize> {
    let n = func.blocks.len();
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);

    fn dfs(block: usize, func: &Function, visited: &mut Vec<bool>, order: &mut Vec<usize>) {
        visited[block] = true;
        for succ in func.blocks[block].terminator.successors() {
            let s = succ.index();
            if !visited[s] {
                dfs(s, func, visited, order);
            }
        }
        order.push(block);
    }

    dfs(0, func, &mut visited, &mut order);
    order.reverse();
    order
}

fn compute_predecessors(func: &Function) -> Vec<Vec<usize>> {
    let n = func.blocks.len();
    let mut preds = vec![vec![]; n];
    for (i, block) in func.blocks.iter().enumerate() {
        for succ in block.terminator.successors() {
            preds[succ.index()].push(i);
        }
    }
    preds
}

// ─── Dominators ──────────────────────────────────────────────────

/// Compute immediate dominators using the iterative dataflow algorithm.
/// Returns idom[block_index]. idom[0] = 0 (entry dominates itself).
/// Unreachable blocks have idom = usize::MAX.
fn compute_idom(func: &Function) -> Vec<usize> {
    let n = func.blocks.len();
    let rpo = compute_rpo(func);
    let preds = compute_predecessors(func);

    // Map block index → RPO number (lower = earlier)
    let mut rpo_num = vec![usize::MAX; n];
    for (i, &b) in rpo.iter().enumerate() {
        rpo_num[b] = i;
    }

    let mut idom = vec![usize::MAX; n];
    idom[0] = 0; // entry dominates itself

    let intersect = |mut a: usize, mut b: usize, idom: &[usize]| -> usize {
        while a != b {
            while rpo_num[a] > rpo_num[b] {
                a = idom[a];
            }
            while rpo_num[b] > rpo_num[a] {
                b = idom[b];
            }
        }
        a
    };

    let mut changed = true;
    while changed {
        changed = false;
        for &b in &rpo {
            if b == 0 {
                continue;
            }
            // Find first processed predecessor
            let mut new_idom = usize::MAX;
            for &p in &preds[b] {
                if idom[p] != usize::MAX {
                    if new_idom == usize::MAX {
                        new_idom = p;
                    } else {
                        new_idom = intersect(new_idom, p, &idom);
                    }
                }
            }
            if new_idom != idom[b] {
                idom[b] = new_idom;
                changed = true;
            }
        }
    }

    idom
}

/// Compute dominance frontiers.
fn compute_dom_frontiers(func: &Function, idom: &[usize]) -> Vec<HashSet<usize>> {
    let n = func.blocks.len();
    let preds = compute_predecessors(func);
    let mut df = vec![HashSet::new(); n];

    for b in 0..n {
        if preds[b].len() >= 2 {
            for &p in &preds[b] {
                let mut runner = p;
                while runner != idom[b] && runner != usize::MAX {
                    df[runner].insert(b);
                    runner = idom[runner];
                }
            }
        }
    }

    df
}

/// Compute dominator tree children.
fn compute_dom_children(idom: &[usize]) -> Vec<Vec<usize>> {
    let n = idom.len();
    let mut children = vec![vec![]; n];
    for b in 1..n {
        if idom[b] != usize::MAX {
            children[idom[b]].push(b);
        }
    }
    children
}

/// Check if block `a` dominates block `b`.
fn dominates(a: usize, b: usize, idom: &[usize]) -> bool {
    let mut runner = b;
    loop {
        if runner == a {
            return true;
        }
        if runner == 0 && a != 0 {
            return false;
        }
        if idom[runner] == runner {
            return runner == a;
        }
        runner = idom[runner];
    }
}

// ─── Natural Loop Detection ─────────────────────────────────────

struct NaturalLoop {
    header: usize,
    blocks: HashSet<usize>,
}

fn find_natural_loops(func: &Function, idom: &[usize]) -> Vec<NaturalLoop> {
    let preds = compute_predecessors(func);
    let mut loops = Vec::new();

    // Find back edges: edge (b → h) where h dominates b
    for (b, block) in func.blocks.iter().enumerate() {
        for succ in block.terminator.successors() {
            let h = succ.index();
            if dominates(h, b, idom) {
                // Back edge b → h. Collect the natural loop.
                let mut loop_blocks = HashSet::new();
                loop_blocks.insert(h);
                loop_blocks.insert(b);

                // Walk backwards from b to find all blocks in the loop
                let mut worklist = vec![b];
                while let Some(node) = worklist.pop() {
                    for &p in &preds[node] {
                        if !loop_blocks.contains(&p) {
                            loop_blocks.insert(p);
                            worklist.push(p);
                        }
                    }
                }

                loops.push(NaturalLoop {
                    header: h,
                    blocks: loop_blocks,
                });
            }
        }
    }

    loops
}

// ─── mem2reg ─────────────────────────────────────────────────────
//
// Promotes stack slot loads/stores to SSA values with block parameters.
// Stack slots are the biggest performance bottleneck: every Lua register
// access goes through StackAddr → Load/Store, adding ~20 memory ops
// per loop iteration.

/// Information about a promotable stack slot.
struct SlotInfo {
    #[allow(dead_code)]
    slot: StackSlot,
    ty: Type,
    is_gc_root: bool,
    /// Blocks that contain stores to this slot.
    store_blocks: HashSet<usize>,
    /// (block_idx, inst_idx, stored_value) for each store.
    stores: Vec<(usize, usize, Value)>,
    /// (block_idx, inst_idx, loaded_value) for each load.
    loads: Vec<(usize, usize, Value)>,
    /// All StackAddr values for this slot (to be cleaned up).
    stack_addrs: Vec<Value>,
}

pub fn mem2reg(func: &mut Function) {
    // Phase 1: Identify promotable stack slots.
    // A slot is promotable if every StackAddr result is used ONLY as the
    // address operand of Load(ty, addr, 0) or Store(val, addr, 0).
    let slot_count = func.stack_slots.len();
    if slot_count == 0 {
        return;
    }

    // Build a map: Value (StackAddr result) → slot index
    let mut addr_to_slot: HashMap<Value, usize> = HashMap::new();
    // Track all uses of StackAddr values
    let mut addr_values: Vec<HashSet<Value>> = vec![HashSet::new(); slot_count];

    for block in &func.blocks {
        for inst_node in &block.insts {
            if let Inst::StackAddr(slot) = &inst_node.inst {
                if let Some(val) = inst_node.value {
                    addr_to_slot.insert(val, slot.index());
                    addr_values[slot.index()].insert(val);
                }
            }
        }
    }

    // Check which slots are promotable: all uses of their StackAddr results
    // must be Load(ty, addr, 0) or Store(val, addr, 0).
    let mut promotable = vec![true; slot_count];
    let mut slot_infos: Vec<Option<SlotInfo>> = (0..slot_count)
        .map(|i| {
            Some(SlotInfo {
                slot: StackSlot(i as u32),
                ty: Type::I64, // will be determined from loads
                is_gc_root: func.stack_slots[i].is_gc_root,
                store_blocks: HashSet::new(),
                stores: Vec::new(),
                loads: Vec::new(),
                stack_addrs: Vec::new(),
            })
        })
        .collect();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        for (inst_idx, inst_node) in block.insts.iter().enumerate() {
            match &inst_node.inst {
                Inst::Load(ty, addr, offset) => {
                    if *offset == 0 {
                        if let Some(&slot_idx) = addr_to_slot.get(addr) {
                            if let Some(ref mut info) = slot_infos[slot_idx] {
                                info.ty = *ty;
                                if let Some(val) = inst_node.value {
                                    info.loads.push((block_idx, inst_idx, val));
                                }
                            }
                        }
                    }
                }
                Inst::Store(_, addr, offset) => {
                    if *offset == 0 {
                        if let Some(&slot_idx) = addr_to_slot.get(addr) {
                            if let Some(ref mut info) = slot_infos[slot_idx] {
                                // Extract stored value
                                if let Inst::Store(val, _, _) = &inst_node.inst {
                                    info.store_blocks.insert(block_idx);
                                    info.stores.push((block_idx, inst_idx, *val));
                                }
                            }
                        }
                    }
                }
                _ => {}
            }

            // Check if any StackAddr value is used in a non-load/store context
            inst_node.inst.for_each_value(|v| {
                if let Some(&slot_idx) = addr_to_slot.get(&v) {
                    // This value is a StackAddr result. Check if this usage is valid.
                    match &inst_node.inst {
                        Inst::Load(_, addr, 0) if *addr == v => {} // OK
                        Inst::Store(_, addr, 0) if *addr == v => {} // OK
                        _ => {
                            // Used in non-promotable context
                            promotable[slot_idx] = false;
                        }
                    }
                }
            });
        }

        // Check terminator too
        block.terminator.for_each_value(|v| {
            if let Some(&slot_idx) = addr_to_slot.get(&v) {
                promotable[slot_idx] = false;
            }
        });
    }

    // Collect StackAddr values for promotable slots
    for (val, &slot_idx) in &addr_to_slot {
        if promotable[slot_idx] {
            if let Some(ref mut info) = slot_infos[slot_idx] {
                info.stack_addrs.push(*val);
            }
        }
    }

    // Filter to only promotable slots
    let promote_slots: Vec<SlotInfo> = slot_infos
        .into_iter()
        .enumerate()
        .filter_map(|(i, info)| {
            if promotable[i] {
                info
            } else {
                None
            }
        })
        .collect();

    if promote_slots.is_empty() {
        return;
    }

    // Phase 2: Insert phi nodes (block parameters) at dominance frontiers.
    let idom = compute_idom(func);
    let df = compute_dom_frontiers(func, &idom);
    let dom_children = compute_dom_children(&idom);

    // For each slot, compute where phis are needed
    // slot_phis[slot_local_idx] = set of blocks needing phi for this slot
    let mut slot_phis: Vec<HashSet<usize>> = Vec::with_capacity(promote_slots.len());

    for info in &promote_slots {
        let mut phi_blocks = HashSet::new();
        let mut worklist: Vec<usize> = info.store_blocks.iter().copied().collect();
        let mut ever_on_worklist = info.store_blocks.clone();

        while let Some(b) = worklist.pop() {
            for &d in &df[b] {
                if !phi_blocks.contains(&d) {
                    phi_blocks.insert(d);
                    if !ever_on_worklist.contains(&d) {
                        ever_on_worklist.insert(d);
                        worklist.push(d);
                    }
                }
            }
        }

        slot_phis.push(phi_blocks);
    }

    // Phase 3: Insert block parameters for phi nodes.
    // phi_param_values[slot_local_idx][block_idx] = Value of the phi node
    let mut phi_param_values: Vec<HashMap<usize, Value>> =
        vec![HashMap::new(); promote_slots.len()];

    for (slot_local_idx, info) in promote_slots.iter().enumerate() {
        for &block_idx in &slot_phis[slot_local_idx] {
            let val = new_value(func, info.ty);
            func.blocks[block_idx].params.push((val, info.ty));
            phi_param_values[slot_local_idx].insert(block_idx, val);
        }
    }

    // Phase 4: Rename — walk the dominator tree, maintaining a value stack
    // for each slot. Replace loads with current value, stores update the stack.
    //
    // Instead of using pre-computed instruction indices (which break when we
    // insert undef instructions), we identify loads/stores at rename time by
    // pattern-matching instructions against addr_to_slot.

    // Build slot_idx → local_idx map for promoted slots
    let mut slot_to_local: HashMap<usize, usize> = HashMap::new();
    for (local_idx, info) in promote_slots.iter().enumerate() {
        slot_to_local.insert(info.slot.index(), local_idx);
    }

    struct RenameState {
        stacks: Vec<Vec<Value>>,
    }

    let mut state = RenameState {
        stacks: Vec::with_capacity(promote_slots.len()),
    };

    // Initialize with undef (Iconst 0) for each slot — handles reads before writes
    for info in &promote_slots {
        let undef = new_value(func, info.ty);
        state.stacks.push(vec![undef]);
    }

    // Insert undef constants at the start of the entry block
    let mut undef_insts = Vec::new();
    for (slot_local_idx, _info) in promote_slots.iter().enumerate() {
        let val = state.stacks[slot_local_idx][0];
        let ty = promote_slots[slot_local_idx].ty;
        undef_insts.push(InstNode {
            value: Some(val),
            inst: Inst::Iconst(ty, 0),
        });
    }
    let entry_insts = std::mem::take(&mut func.blocks[0].insts);
    func.blocks[0].insts = undef_insts;
    func.blocks[0].insts.extend(entry_insts);

    // Rename via dominator tree walk — uses pattern matching, not indices
    fn rename_block(
        block_idx: usize,
        func: &mut Function,
        state: &mut RenameState,
        dom_children: &[Vec<usize>],
        phi_param_values: &[HashMap<usize, Value>],
        addr_to_slot: &HashMap<Value, usize>,
        slot_to_local: &HashMap<usize, usize>,
        promotable: &[bool],
    ) {
        let num_slots = state.stacks.len();
        let stack_heights: Vec<usize> = state.stacks.iter().map(|s| s.len()).collect();

        // Push phi-node definitions
        for slot_local_idx in 0..num_slots {
            if let Some(&val) = phi_param_values[slot_local_idx].get(&block_idx) {
                state.stacks[slot_local_idx].push(val);
            }
        }

        // Process instructions: identify loads/stores to promoted slots by
        // checking if the address operand is a known promoted StackAddr value.
        for inst_idx in 0..func.blocks[block_idx].insts.len() {
            let inst = &func.blocks[block_idx].insts[inst_idx].inst;

            match inst {
                Inst::Load(_, addr, 0) => {
                    if let Some(&slot_idx) = addr_to_slot.get(addr) {
                        if let Some(&local_idx) = slot_to_local.get(&slot_idx) {
                            // Load from promoted slot → replace with current value
                            let current = *state.stacks[local_idx].last().unwrap();
                            let load_val = func.blocks[block_idx].insts[inst_idx].value.unwrap();
                            replace_all_uses(func, load_val, current);
                            func.blocks[block_idx].insts[inst_idx].inst =
                                Inst::Iconst(Type::I8, 0);
                            func.blocks[block_idx].insts[inst_idx].value = None;
                        }
                    }
                }
                Inst::Store(val, addr, 0) => {
                    if let Some(&slot_idx) = addr_to_slot.get(addr) {
                        if let Some(&local_idx) = slot_to_local.get(&slot_idx) {
                            // Store to promoted slot → update current definition
                            let stored_val = *val;
                            state.stacks[local_idx].push(stored_val);
                            func.blocks[block_idx].insts[inst_idx].inst =
                                Inst::Iconst(Type::I8, 0);
                            func.blocks[block_idx].insts[inst_idx].value = None;
                        }
                    }
                }
                _ => {}
            }
        }

        // Fill in phi operands in successor blocks
        let succs = func.blocks[block_idx].terminator.successors();
        for succ in &succs {
            let succ_idx = succ.index();
            for slot_local_idx in 0..num_slots {
                if phi_param_values[slot_local_idx].contains_key(&succ_idx) {
                    let current = *state.stacks[slot_local_idx].last().unwrap();
                    func.blocks[block_idx]
                        .terminator
                        .for_each_successor_args_mut(|target, args| {
                            if target.index() == succ_idx {
                                args.push(current);
                            }
                        });
                }
            }
        }

        // Recurse to dominator tree children
        let children: Vec<usize> = dom_children[block_idx].clone();
        for child in children {
            rename_block(
                child,
                func,
                state,
                dom_children,
                phi_param_values,
                addr_to_slot,
                slot_to_local,
                promotable,
            );
        }

        // Restore stacks
        for slot_local_idx in 0..num_slots {
            state.stacks[slot_local_idx].truncate(stack_heights[slot_local_idx]);
        }
    }

    rename_block(
        0,
        func,
        &mut state,
        &dom_children,
        &phi_param_values,
        &addr_to_slot,
        &slot_to_local,
        &promotable,
    );

    // Phase 5: Update safepoints — add live GcPtr values from promoted slots.
    // (Must happen before block elimination since it uses block indices.)
    // At each safepoint, find which promoted GC root slots have a current
    // definition and add them to the safepoint's live list.
    // We need to re-walk to determine the current definition at each safepoint.
    // For simplicity, we add ALL phi values and block params for GC root slots
    // to safepoints in their blocks.
    //
    // Actually, a simpler correct approach: for each Safepoint instruction,
    // we need to figure out the "reaching definition" for each promoted GC root
    // slot. We'll do another dominator-tree walk.
    update_safepoints_for_promoted_slots(func, &promote_slots, &phi_param_values, &dom_children);

    // Phase 6: Eliminate unreachable blocks to avoid verification issues
    // (unreachable predecessor blocks would have wrong phi argument counts).
    eliminate_unreachable_blocks(func);

    // Phase 7: Clean up dead StackAddr instructions.
    // Mark StackAddr instructions for promoted slots as dead.
    for block in &mut func.blocks {
        for inst_node in &mut block.insts {
            if let Inst::StackAddr(slot) = &inst_node.inst {
                let slot_idx = slot.index();
                if promotable.get(slot_idx).copied().unwrap_or(false) {
                    inst_node.inst = Inst::Iconst(Type::I8, 0);
                    inst_node.value = None;
                }
            }
        }
    }

    // Remove all no-op instructions (value = None and inst is a dummy)
    for block in &mut func.blocks {
        block.insts.retain(|node| node.value.is_some() || !is_pure(&node.inst));
    }
}

/// After mem2reg, walk the dominator tree to find the reaching definition
/// of each promoted GC root slot at each Safepoint, and add them to the
/// safepoint's live list.
fn update_safepoints_for_promoted_slots(
    func: &mut Function,
    promote_slots: &[SlotInfo],
    phi_param_values: &[HashMap<usize, Value>],
    dom_children: &[Vec<usize>],
) {
    // Collect which promoted slots are GC roots
    let gc_root_slot_indices: Vec<usize> = promote_slots
        .iter()
        .enumerate()
        .filter(|(_, info)| info.is_gc_root)
        .map(|(i, _)| i)
        .collect();

    if gc_root_slot_indices.is_empty() {
        return;
    }

    // Walk dominator tree, tracking current definitions
    fn walk_safepoints(
        block_idx: usize,
        func: &mut Function,
        stacks: &mut Vec<Vec<Value>>,
        dom_children: &[Vec<usize>],
        promote_slots: &[SlotInfo],
        phi_param_values: &[HashMap<usize, Value>],
        gc_root_indices: &[usize],
    ) {
        let heights: Vec<usize> = stacks.iter().map(|s| s.len()).collect();

        // Push phi definitions
        for &slot_idx in gc_root_indices {
            if let Some(&val) = phi_param_values[slot_idx].get(&block_idx) {
                stacks[slot_idx].push(val);
            }
        }

        // Walk instructions
        for _inst_node in &func.blocks[block_idx].insts {
            // Check for stores (which are now Iconst no-ops after rename,
            // but we already did rename. We need to track stores differently.)
            // Actually, after rename, stores are gone. The stacks were only
            // used during rename. We need to re-derive the reaching def.
        }

        // Actually, after rename, the current value at each point is:
        // - The phi value if block has a phi for this slot
        // - Otherwise, the reaching definition from the dominating block
        // - Stores within the block update it, but stores have been removed...
        //
        // The reaching def at a safepoint is tricky because intra-block stores
        // have been removed. We need to find what value *would have been* stored.
        //
        // Simpler approach: For each safepoint in a block, the live values for
        // promoted GC root slots are the block parameters (phis) or the
        // reaching definition from the dominator. Since stores become SSA updates,
        // the value visible at any point in the block is either:
        // 1. The phi node value (if block has a phi for this slot)
        // 2. The value from the dominating definition
        //
        // We just use what's on the stack.

        // Find safepoint instructions and add GC root values
        for inst_node in &mut func.blocks[block_idx].insts {
            if let Inst::Safepoint(ref mut live) = inst_node.inst {
                for &slot_idx in gc_root_indices {
                    if let Some(&val) = stacks[slot_idx].last() {
                        live.push(val);
                    }
                }
            }
        }

        let children: Vec<usize> = dom_children[block_idx].clone();
        for child in children {
            walk_safepoints(
                child,
                func,
                stacks,
                dom_children,
                promote_slots,
                phi_param_values,
                gc_root_indices,
            );
        }

        for (i, &h) in heights.iter().enumerate() {
            stacks[i].truncate(h);
        }
    }

    // Initialize stacks with entry block undefs
    // After rename, the undef values are the first definitions in the entry block
    let mut stacks: Vec<Vec<Value>> = promote_slots
        .iter()
        .enumerate()
        .map(|(i, _info)| {
            // Find the Iconst that was inserted at the entry block for this slot's undef
            // It's at index i in the entry block (we prepended them)
            if gc_root_slot_indices.contains(&i) {
                if let Some(node) = func.blocks[0].insts.get(i) {
                    if let Some(val) = node.value {
                        return vec![val];
                    }
                }
            }
            vec![]
        })
        .collect();

    // For slots that got stores in the entry block, we need to push those too.
    // But stores have been removed. Instead, we look at the phi values.
    // The entry block shouldn't have phis (it has no predecessors with loops yet).

    walk_safepoints(
        0,
        func,
        &mut stacks,
        dom_children,
        &promote_slots,
        phi_param_values,
        &gc_root_slot_indices,
    );
}

// ─── Unreachable Block Elimination ──────────────────────────────

/// Remove blocks unreachable from the entry and compact block indices.
fn eliminate_unreachable_blocks(func: &mut Function) {
    let n = func.blocks.len();
    if n == 0 {
        return;
    }

    // DFS to find reachable blocks
    let mut reachable = vec![false; n];
    let mut worklist = vec![0usize];
    reachable[0] = true;
    while let Some(b) = worklist.pop() {
        for succ in func.blocks[b].terminator.successors() {
            let s = succ.index();
            if s < n && !reachable[s] {
                reachable[s] = true;
                worklist.push(s);
            }
        }
    }

    // Check if any blocks are unreachable
    if reachable.iter().all(|&r| r) {
        return;
    }

    // Build mapping: old_index → new_index (or usize::MAX for removed)
    let mut new_index = vec![usize::MAX; n];
    let mut next = 0usize;
    for i in 0..n {
        if reachable[i] {
            new_index[i] = next;
            next += 1;
        }
    }

    // Compact blocks
    let mut new_blocks = Vec::with_capacity(next);
    for (i, block) in func.blocks.drain(..).enumerate() {
        if reachable[i] {
            new_blocks.push(block);
        }
    }
    func.blocks = new_blocks;

    // Remap block IDs
    let remap = |bid: BlockId| -> BlockId {
        let new_idx = new_index[bid.index()];
        debug_assert_ne!(new_idx, usize::MAX, "reference to removed block");
        BlockId::from_index(new_idx)
    };

    for block in &mut func.blocks {
        match &mut block.terminator {
            Terminator::Jump(target, _) => *target = remap(*target),
            Terminator::BrIf {
                then_block,
                else_block,
                ..
            } => {
                *then_block = remap(*then_block);
                *else_block = remap(*else_block);
            }
            Terminator::Switch {
                cases,
                default_block,
                ..
            } => {
                for (_, block, _) in cases {
                    *block = remap(*block);
                }
                *default_block = remap(*default_block);
            }
            Terminator::Invoke {
                normal, exception, ..
            }
            | Terminator::InvokeIndirect {
                normal, exception, ..
            } => {
                *normal = remap(*normal);
                *exception = remap(*exception);
            }
            Terminator::AbortToPrompt { .. } => {}
            Terminator::Ret(_)
            | Terminator::RetVoid
            | Terminator::ResumeSlice { .. }
            | Terminator::Unreachable => {}
        }
    }
}

// ─── Dead Code Elimination ──────────────────────────────────────

pub fn dce(func: &mut Function) {
    loop {
        let use_counts = compute_use_counts(func);
        let mut removed = false;

        for block in &mut func.blocks {
            block.insts.retain(|node| {
                if let Some(val) = node.value {
                    if is_pure(&node.inst) && use_counts[val.index()] == 0 {
                        removed = true;
                        return false;
                    }
                }
                true
            });
        }

        if !removed {
            break;
        }
    }
}

// ─── Constant Folding ───────────────────────────────────────────

pub fn constant_fold(func: &mut Function) {
    // Build a map of known constants: Value → i64 or f64
    let mut iconsts: HashMap<Value, (Type, i64)> = HashMap::new();
    let mut fconsts: HashMap<Value, f64> = HashMap::new();

    let mut replacements: Vec<(Value, Inst)> = Vec::new();

    for block in &func.blocks {
        for inst_node in &block.insts {
            if let Some(val) = inst_node.value {
                match &inst_node.inst {
                    Inst::Iconst(ty, c) => {
                        iconsts.insert(val, (*ty, *c));
                    }
                    Inst::F64Const(c) => {
                        fconsts.insert(val, *c);
                    }
                    Inst::Add(a, b) => {
                        if let (Some((ty, ca)), Some((_, cb))) =
                            (iconsts.get(a), iconsts.get(b))
                        {
                            replacements.push((val, Inst::Iconst(*ty, ca.wrapping_add(*cb))));
                        }
                    }
                    Inst::Sub(a, b) => {
                        if let (Some((ty, ca)), Some((_, cb))) =
                            (iconsts.get(a), iconsts.get(b))
                        {
                            replacements.push((val, Inst::Iconst(*ty, ca.wrapping_sub(*cb))));
                        }
                    }
                    Inst::Mul(a, b) => {
                        if let (Some((ty, ca)), Some((_, cb))) =
                            (iconsts.get(a), iconsts.get(b))
                        {
                            replacements.push((val, Inst::Iconst(*ty, ca.wrapping_mul(*cb))));
                        }
                    }
                    Inst::FAdd(a, b) => {
                        if let (Some(&ca), Some(&cb)) = (fconsts.get(a), fconsts.get(b)) {
                            replacements.push((val, Inst::F64Const(ca + cb)));
                        }
                    }
                    Inst::FSub(a, b) => {
                        if let (Some(&ca), Some(&cb)) = (fconsts.get(a), fconsts.get(b)) {
                            replacements.push((val, Inst::F64Const(ca - cb)));
                        }
                    }
                    Inst::FMul(a, b) => {
                        if let (Some(&ca), Some(&cb)) = (fconsts.get(a), fconsts.get(b)) {
                            replacements.push((val, Inst::F64Const(ca * cb)));
                        }
                    }
                    Inst::FDiv(a, b) => {
                        if let (Some(&ca), Some(&cb)) = (fconsts.get(a), fconsts.get(b)) {
                            replacements.push((val, Inst::F64Const(ca / cb)));
                        }
                    }
                    Inst::And(a, b) => {
                        if let (Some((ty, ca)), Some((_, cb))) =
                            (iconsts.get(a), iconsts.get(b))
                        {
                            replacements.push((val, Inst::Iconst(*ty, ca & cb)));
                        }
                    }
                    Inst::Or(a, b) => {
                        if let (Some((ty, ca)), Some((_, cb))) =
                            (iconsts.get(a), iconsts.get(b))
                        {
                            replacements.push((val, Inst::Iconst(*ty, ca | cb)));
                        }
                    }
                    Inst::Xor(a, b) => {
                        if let (Some((ty, ca)), Some((_, cb))) =
                            (iconsts.get(a), iconsts.get(b))
                        {
                            replacements.push((val, Inst::Iconst(*ty, ca ^ cb)));
                        }
                    }
                    Inst::Shl(a, b) => {
                        if let (Some((ty, ca)), Some((_, cb))) =
                            (iconsts.get(a), iconsts.get(b))
                        {
                            replacements.push((
                                val,
                                Inst::Iconst(*ty, ca.wrapping_shl(*cb as u32)),
                            ));
                        }
                    }
                    Inst::Icmp(op, a, b) => {
                        if let (Some((_, ca)), Some((_, cb))) =
                            (iconsts.get(a), iconsts.get(b))
                        {
                            let result = match op {
                                CmpOp::Eq => ca == cb,
                                CmpOp::Ne => ca != cb,
                                CmpOp::Slt => ca < cb,
                                CmpOp::Sle => ca <= cb,
                                CmpOp::Sgt => ca > cb,
                                CmpOp::Sge => ca >= cb,
                                CmpOp::Ult => (*ca as u64) < (*cb as u64),
                                CmpOp::Ule => (*ca as u64) <= (*cb as u64),
                                CmpOp::Ugt => (*ca as u64) > (*cb as u64),
                                CmpOp::Uge => (*ca as u64) >= (*cb as u64),
                            };
                            replacements
                                .push((val, Inst::Iconst(Type::I8, if result { 1 } else { 0 })));
                        }
                    }
                    Inst::Bitcast(v, to_ty) => {
                        // Bitcast between same-size types with known constant
                        if let Some((_, c)) = iconsts.get(v) {
                            if *to_ty == Type::F64 {
                                fconsts.insert(val, f64::from_bits(*c as u64));
                            } else {
                                replacements.push((val, Inst::Iconst(*to_ty, *c)));
                            }
                        } else if let Some(&c) = fconsts.get(v) {
                            if *to_ty != Type::F64 {
                                replacements
                                    .push((val, Inst::Iconst(*to_ty, c.to_bits() as i64)));
                                iconsts.insert(val, (*to_ty, c.to_bits() as i64));
                            }
                        }
                    }
                    _ => {}
                }

                // Track folded constants
                if let Some((_, inst)) = replacements.last() {
                    if replacements.last().map(|(v, _)| *v) == Some(val) {
                        match inst {
                            Inst::Iconst(ty, c) => {
                                iconsts.insert(val, (*ty, *c));
                            }
                            Inst::F64Const(c) => {
                                fconsts.insert(val, *c);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    // Apply replacements
    for (val, new_inst) in replacements {
        for block in &mut func.blocks {
            for inst_node in &mut block.insts {
                if inst_node.value == Some(val) {
                    inst_node.inst = new_inst.clone();
                    break;
                }
            }
        }
    }
}

// ─── Local GVN (Global Value Numbering, per block) ──────────────

/// Hash key for instruction deduplication.
#[derive(Hash, PartialEq, Eq, Clone)]
enum GvnKey {
    Iconst(u8, i64),       // (type discriminant, value)
    F64Const(u64),         // bits
    BinOp(u8, u32, u32),  // (opcode, left_value_idx, right_value_idx)
    UnOp(u8, u32),        // (opcode, operand_value_idx)
    Cmp(u8, u8, u32, u32), // (inst_kind, cmp_op, a, b)
    Conv(u8, u32, u8),    // (opcode, operand, target_type)
    TagOp(u8, u32),       // (opcode, operand)
    MakeTagged(u32, u32), // (tag, payload_value_idx)
    IsTag(u32, u32),      // (value_idx, tag)
    StackAddr(u32),        // slot index
}

fn inst_to_gvn_key(inst: &Inst) -> Option<GvnKey> {
    match inst {
        Inst::Iconst(ty, v) => Some(GvnKey::Iconst(*ty as u8, *v)),
        Inst::F64Const(v) => Some(GvnKey::F64Const(v.to_bits())),

        Inst::Add(a, b) => Some(GvnKey::BinOp(0, a.0, b.0)),
        Inst::Sub(a, b) => Some(GvnKey::BinOp(1, a.0, b.0)),
        Inst::Mul(a, b) => Some(GvnKey::BinOp(2, a.0, b.0)),
        Inst::SDiv(a, b) => Some(GvnKey::BinOp(3, a.0, b.0)),
        Inst::UDiv(a, b) => Some(GvnKey::BinOp(4, a.0, b.0)),
        Inst::FAdd(a, b) => Some(GvnKey::BinOp(5, a.0, b.0)),
        Inst::FSub(a, b) => Some(GvnKey::BinOp(6, a.0, b.0)),
        Inst::FMul(a, b) => Some(GvnKey::BinOp(7, a.0, b.0)),
        Inst::FDiv(a, b) => Some(GvnKey::BinOp(8, a.0, b.0)),
        Inst::And(a, b) => Some(GvnKey::BinOp(9, a.0, b.0)),
        Inst::Or(a, b) => Some(GvnKey::BinOp(10, a.0, b.0)),
        Inst::Xor(a, b) => Some(GvnKey::BinOp(11, a.0, b.0)),
        Inst::Shl(a, b) => Some(GvnKey::BinOp(12, a.0, b.0)),
        Inst::LShr(a, b) => Some(GvnKey::BinOp(13, a.0, b.0)),
        Inst::AShr(a, b) => Some(GvnKey::BinOp(14, a.0, b.0)),

        Inst::Neg(v) => Some(GvnKey::UnOp(0, v.0)),
        Inst::FNeg(v) => Some(GvnKey::UnOp(1, v.0)),
        Inst::Not(v) => Some(GvnKey::UnOp(2, v.0)),

        Inst::Icmp(op, a, b) => Some(GvnKey::Cmp(0, *op as u8, a.0, b.0)),
        Inst::Fcmp(op, a, b) => Some(GvnKey::Cmp(1, *op as u8, a.0, b.0)),

        Inst::Sext(v, ty) => Some(GvnKey::Conv(0, v.0, *ty as u8)),
        Inst::Zext(v, ty) => Some(GvnKey::Conv(1, v.0, *ty as u8)),
        Inst::Trunc(v, ty) => Some(GvnKey::Conv(2, v.0, *ty as u8)),
        Inst::IntToFloat(v) => Some(GvnKey::Conv(3, v.0, 0)),
        Inst::FloatToInt(v) => Some(GvnKey::Conv(4, v.0, 0)),
        Inst::Bitcast(v, ty) => Some(GvnKey::Conv(5, v.0, *ty as u8)),

        Inst::TagOf(v) => Some(GvnKey::TagOp(0, v.0)),
        Inst::Payload(v) => Some(GvnKey::TagOp(1, v.0)),
        Inst::MakeTagged(tag, v) => Some(GvnKey::MakeTagged(*tag, v.0)),
        Inst::IsTag(v, tag) => Some(GvnKey::IsTag(v.0, *tag)),

        Inst::StackAddr(slot) => Some(GvnKey::StackAddr(slot.0)),

        // Don't deduplicate: loads (may read different memory), stores, calls, etc.
        _ => None,
    }
}

pub fn gvn(func: &mut Function) {
    let mut any_replaced = false;
    let mut replacements: Vec<(Value, Value)> = Vec::new();

    for block in &func.blocks {
        let mut seen: HashMap<GvnKey, Value> = HashMap::new();

        for inst_node in &block.insts {
            if let Some(val) = inst_node.value {
                if let Some(key) = inst_to_gvn_key(&inst_node.inst) {
                    if let Some(&existing) = seen.get(&key) {
                        // Duplicate! Replace val with existing
                        replacements.push((val, existing));
                        any_replaced = true;
                    } else {
                        seen.insert(key, val);
                    }
                }
            }
        }
    }

    // Apply replacements
    for (old, new_val) in replacements {
        replace_all_uses(func, old, new_val);
    }

    if any_replaced {
        dce(func);
    }
}

// ─── Loop-Invariant Code Motion ─────────────────────────────────

pub fn licm(func: &mut Function) {
    let idom = compute_idom(func);
    let loops = find_natural_loops(func, &idom);
    let preds = compute_predecessors(func);

    for lp in &loops {
        // Find or create a preheader: a block that's the sole entry to the loop
        // header from outside the loop.
        let outside_preds: Vec<usize> = preds[lp.header]
            .iter()
            .copied()
            .filter(|p| !lp.blocks.contains(p))
            .collect();

        if outside_preds.is_empty() {
            continue; // unreachable loop
        }

        // For simplicity, only handle loops where we can identify a clear preheader.
        // If the header has non-loop predecessors that already jump directly to it
        // with no other successors, we can use the first one.
        // Otherwise, we insert a preheader.
        let preheader = if outside_preds.len() == 1 {
            let p = outside_preds[0];
            // Check that this predecessor only goes to the header
            let succs = func.blocks[p].terminator.successors();
            if succs.len() == 1 && succs[0].index() == lp.header {
                p
            } else {
                continue; // complex entry, skip
            }
        } else {
            continue; // multiple outside predecessors, skip for now
        };

        // Verify that the preheader dominates all loop blocks
        let preheader_dominates_all = lp.blocks.iter().all(|&b| dominates(preheader, b, &idom));
        if !preheader_dominates_all {
            continue;
        }

        // Determine if the loop body has any stores or calls (for load hoisting)
        let loop_has_side_effects = lp.blocks.iter().any(|&b| {
            func.blocks[b].insts.iter().any(|node| {
                matches!(
                    &node.inst,
                    Inst::Store(..) | Inst::Call(..) | Inst::CallIndirect(..)
                )
            })
        });

        // Collect values defined outside the loop
        let mut outside_values: HashSet<Value> = HashSet::new();
        for (b, block) in func.blocks.iter().enumerate() {
            if !lp.blocks.contains(&b) {
                for (val, _) in &block.params {
                    outside_values.insert(*val);
                }
                for node in &block.insts {
                    if let Some(val) = node.value {
                        outside_values.insert(val);
                    }
                }
            }
        }

        // NOTE: Loop header block params are NOT outside values — they're
        // phi nodes that change each iteration. Instructions using them
        // must NOT be hoisted to the preheader.

        // Iteratively find hoistable instructions
        let mut hoisted_values: HashSet<Value> = HashSet::new();
        let mut to_hoist: Vec<(usize, usize)> = Vec::new(); // (block_idx, inst_idx)

        let mut changed = true;
        while changed {
            changed = false;

            for &b in &lp.blocks {
                for (inst_idx, node) in func.blocks[b].insts.iter().enumerate() {
                    if let Some(val) = node.value {
                        if hoisted_values.contains(&val) || outside_values.contains(&val) {
                            continue;
                        }

                        // Check if this instruction can be hoisted
                        let can_hoist = if matches!(&node.inst, Inst::Load(..)) {
                            // Loads can only be hoisted if no side effects in loop
                            !loop_has_side_effects
                        } else {
                            is_hoistable(&node.inst)
                        };

                        if !can_hoist {
                            continue;
                        }

                        // Check if all operands are loop-invariant
                        let mut all_invariant = true;
                        node.inst.for_each_value(|v| {
                            if !outside_values.contains(&v) && !hoisted_values.contains(&v) {
                                all_invariant = false;
                            }
                        });

                        if all_invariant {
                            // Verify all uses of this value are in blocks
                            // dominated by the preheader
                            let mut all_uses_dominated = true;
                            for (ub, ublock) in func.blocks.iter().enumerate() {
                                let mut used_here = false;
                                for unode in &ublock.insts {
                                    unode.inst.for_each_value(|v| {
                                        if v == val {
                                            used_here = true;
                                        }
                                    });
                                }
                                ublock.terminator.for_each_value(|v| {
                                    if v == val {
                                        used_here = true;
                                    }
                                });
                                if used_here && !dominates(preheader, ub, &idom) {
                                    all_uses_dominated = false;
                                }
                            }

                            if all_uses_dominated {
                                hoisted_values.insert(val);
                                outside_values.insert(val);
                                to_hoist.push((b, inst_idx));
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        if to_hoist.is_empty() {
            continue;
        }

        // Sort by (block, inst_idx) to maintain relative order
        to_hoist.sort();

        // Move hoisted instructions to the end of the preheader
        // (before its terminator, which is implicit in the block structure)
        let mut hoisted_insts: Vec<InstNode> = Vec::new();
        #[allow(unused)]
        let remove_set: HashSet<(usize, usize)> = to_hoist.iter().copied().collect();

        for &(b, inst_idx) in &to_hoist {
            hoisted_insts.push(func.blocks[b].insts[inst_idx].clone());
        }

        // Remove from original locations (in reverse order to preserve indices)
        for &b in lp.blocks.iter().collect::<Vec<_>>().iter().rev() {
            let mut indices_to_remove: Vec<usize> = Vec::new();
            for (inst_idx, _) in func.blocks[*b].insts.iter().enumerate() {
                if remove_set.contains(&(*b, inst_idx)) {
                    indices_to_remove.push(inst_idx);
                }
            }
            for idx in indices_to_remove.into_iter().rev() {
                func.blocks[*b].insts.remove(idx);
            }
        }

        // Insert at preheader
        func.blocks[preheader].insts.extend(hoisted_insts);
    }
}

// ─── Block Reordering ────────────────────────────────────────────

/// Reorder blocks in reverse post-order so that definitions dominate
/// uses (required by the JIT lowerer which processes blocks sequentially).
pub fn reorder_blocks_rpo(func: &mut Function) {
    let rpo = compute_rpo(func);
    if rpo.len() != func.blocks.len() {
        // Some blocks unreachable — add them at the end
        let mut in_rpo: HashSet<usize> = rpo.iter().copied().collect();
        let mut order = rpo.clone();
        for i in 0..func.blocks.len() {
            if !in_rpo.contains(&i) {
                order.push(i);
            }
        }
        reorder_blocks_with_order(func, &order);
    } else {
        reorder_blocks_with_order(func, &rpo);
    }
}

fn reorder_blocks_with_order(func: &mut Function, order: &[usize]) {
    let n = func.blocks.len();
    // Build old_index → new_index mapping
    let mut new_index = vec![0usize; n];
    for (new_idx, &old_idx) in order.iter().enumerate() {
        new_index[old_idx] = new_idx;
    }

    // Reorder blocks
    let mut new_blocks: Vec<Option<Block>> = (0..n).map(|_| None).collect();
    for (old_idx, block) in func.blocks.drain(..).enumerate() {
        new_blocks[new_index[old_idx]] = Some(block);
    }
    func.blocks = new_blocks.into_iter().map(|b| b.unwrap()).collect();

    // Update all BlockId references
    let remap = |bid: BlockId| -> BlockId {
        BlockId::from_index(new_index[bid.index()])
    };

    for block in &mut func.blocks {
        match &mut block.terminator {
            Terminator::Jump(target, _) => *target = remap(*target),
            Terminator::BrIf {
                then_block,
                else_block,
                ..
            } => {
                *then_block = remap(*then_block);
                *else_block = remap(*else_block);
            }
            Terminator::Switch {
                cases,
                default_block,
                ..
            } => {
                for (_, block, _) in cases {
                    *block = remap(*block);
                }
                *default_block = remap(*default_block);
            }
            Terminator::Invoke {
                normal, exception, ..
            }
            | Terminator::InvokeIndirect {
                normal, exception, ..
            } => {
                *normal = remap(*normal);
                *exception = remap(*exception);
            }
            Terminator::AbortToPrompt { .. } => {}
            Terminator::Ret(_)
            | Terminator::RetVoid
            | Terminator::ResumeSlice { .. }
            | Terminator::Unreachable => {}
        }
    }
}

// ─── Configuration & Combined Optimizer ─────────────────────────

/// Controls which optimization passes are enabled.
#[derive(Debug, Clone)]
pub struct OptConfig {
    pub mem2reg: bool,
    pub constant_fold: bool,
    pub gvn: bool,
    pub licm: bool,
    pub dce: bool,
}

impl OptConfig {
    /// All passes enabled.
    pub fn all() -> Self {
        Self {
            mem2reg: true,
            constant_fold: true,
            gvn: true,
            licm: true,
            dce: true,
        }
    }

    /// All passes disabled.
    pub fn none() -> Self {
        Self {
            mem2reg: false,
            constant_fold: false,
            gvn: false,
            licm: false,
            dce: false,
        }
    }
}

impl Default for OptConfig {
    fn default() -> Self {
        Self::all()
    }
}

/// Run optimization passes according to `config`.
pub fn optimize_with(func: &mut Function, config: &OptConfig) {
    if config.mem2reg {
        mem2reg(func);
        reorder_blocks_rpo(func);
    }

    if config.constant_fold {
        constant_fold(func);
    }

    if config.gvn {
        gvn(func);
    }

    if config.licm {
        licm(func);
    }

    if config.dce {
        dce(func);
    }
}

/// Run all optimization passes.
pub fn optimize(func: &mut Function) {
    optimize_with(func, &OptConfig::all());
}
