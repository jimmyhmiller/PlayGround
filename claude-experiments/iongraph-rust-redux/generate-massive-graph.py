#!/usr/bin/env python3
"""
Generate massive IonGraph JSON files for stress testing.

This script takes an existing ion-example as a template and scales it up
to create massive graphs with controllable size.

Usage:
    ./generate-massive-graph.py <input.json> <output.json> --blocks <num_blocks>
    ./generate-massive-graph.py <input.json> <output.json> --multiplier <factor>
"""

import argparse
import json
import random
import sys


def generate_unique_ptr():
    """Generate a unique pointer value (used as block/instruction IDs)."""
    return random.randint(100000000, 4294967295)


def create_instruction(instr_id, opcode, inputs=None, uses=None, instr_type="None"):
    """Create a basic instruction."""
    return {
        "ptr": generate_unique_ptr(),
        "id": instr_id,
        "opcode": opcode,
        "attributes": [],
        "inputs": inputs or [],
        "uses": uses or [],
        "memInputs": [],
        "type": instr_type
    }


def create_block(block_id, predecessors, successors, instructions, loop_depth=0, attributes=None):
    """Create a basic block."""
    return {
        "ptr": generate_unique_ptr(),
        "id": block_id,
        "loopDepth": loop_depth,
        "attributes": attributes or [],
        "predecessors": predecessors,
        "successors": successors,
        "instructions": instructions,
        "resumePoint": {"mode": "ResumeAt", "operands": []}
    }


def generate_chain_graph(num_blocks, instructions_per_block=5):
    """
    Generate a simple linear chain of blocks.
    Block 0 -> Block 1 -> Block 2 -> ... -> Block N-1
    """
    blocks = []

    for i in range(num_blocks):
        predecessors = [i - 1] if i > 0 else []
        successors = [i + 1] if i < num_blocks - 1 else []

        instructions = []
        base_instr_id = i * instructions_per_block

        for j in range(instructions_per_block):
            instr_id = base_instr_id + j
            if j == 0:
                # First instruction - could reference previous block's output
                if i > 0:
                    prev_instr_id = (i - 1) * instructions_per_block + (instructions_per_block - 1)
                    instr = create_instruction(
                        instr_id,
                        f"Add <- Instr#{prev_instr_id}, Constant [int32]",
                        inputs=[prev_instr_id],
                        uses=[instr_id + 1] if j < instructions_per_block - 1 else [],
                        instr_type="Int32"
                    )
                else:
                    instr = create_instruction(instr_id, "Start", instr_type="None")
            elif j == instructions_per_block - 1:
                # Last instruction - control flow
                if i < num_blocks - 1:
                    instr = create_instruction(
                        instr_id,
                        f"Goto -> block {i + 1}",
                        inputs=[instr_id - 1],
                        instr_type="None"
                    )
                else:
                    instr = create_instruction(
                        instr_id,
                        f"Return <- Instr#{instr_id - 1}",
                        inputs=[instr_id - 1],
                        instr_type="None"
                    )
            else:
                # Middle instructions - math operations
                instr = create_instruction(
                    instr_id,
                    f"Add <- Instr#{instr_id - 1}, Constant [int32]",
                    inputs=[instr_id - 1],
                    uses=[instr_id + 1],
                    instr_type="Int32"
                )
            instructions.append(instr)

        blocks.append(create_block(i, predecessors, successors, instructions))

    return blocks


def generate_diamond_graph(num_diamonds, instructions_per_block=5):
    r"""
    Generate a graph with diamond patterns (if-then-else).

    Each diamond:
           Entry
          /     \
        Then   Else
          \     /
           Merge

    Diamonds are chained together.
    """
    blocks = []
    block_id = 0

    for d in range(num_diamonds):
        # Entry block
        entry_id = block_id
        predecessors = [block_id - 1] if d > 0 else []  # Previous merge
        then_id = block_id + 1
        else_id = block_id + 2
        merge_id = block_id + 3

        # Entry instructions
        base_instr = block_id * instructions_per_block
        entry_instrs = []
        for j in range(instructions_per_block - 1):
            instr_id = base_instr + j
            if j == 0 and d == 0:
                entry_instrs.append(create_instruction(instr_id, "Start"))
            else:
                entry_instrs.append(create_instruction(
                    instr_id,
                    f"Add <- Instr#{instr_id - 1}, Constant [int32]",
                    inputs=[instr_id - 1] if instr_id > 0 else [],
                    uses=[instr_id + 1],
                    instr_type="Int32"
                ))
        # Branch instruction
        branch_instr_id = base_instr + instructions_per_block - 1
        entry_instrs.append(create_instruction(
            branch_instr_id,
            f"Test <- Instr#{branch_instr_id - 1} -> block {then_id}, block {else_id}",
            inputs=[branch_instr_id - 1],
            instr_type="Bool"
        ))
        blocks.append(create_block(entry_id, predecessors, [then_id, else_id], entry_instrs))
        block_id += 1

        # Then block
        then_base = block_id * instructions_per_block
        then_instrs = []
        for j in range(instructions_per_block):
            instr_id = then_base + j
            if j == instructions_per_block - 1:
                then_instrs.append(create_instruction(
                    instr_id,
                    f"Goto -> block {merge_id}",
                    inputs=[instr_id - 1] if j > 0 else [],
                    instr_type="None"
                ))
            else:
                then_instrs.append(create_instruction(
                    instr_id,
                    f"Multiply <- Instr#{instr_id - 1}, Constant [int32]",
                    inputs=[instr_id - 1] if instr_id > 0 else [],
                    uses=[instr_id + 1],
                    instr_type="Int32"
                ))
        blocks.append(create_block(block_id, [entry_id], [merge_id], then_instrs))
        block_id += 1

        # Else block
        else_base = block_id * instructions_per_block
        else_instrs = []
        for j in range(instructions_per_block):
            instr_id = else_base + j
            if j == instructions_per_block - 1:
                else_instrs.append(create_instruction(
                    instr_id,
                    f"Goto -> block {merge_id}",
                    inputs=[instr_id - 1] if j > 0 else [],
                    instr_type="None"
                ))
            else:
                else_instrs.append(create_instruction(
                    instr_id,
                    f"Subtract <- Instr#{instr_id - 1}, Constant [int32]",
                    inputs=[instr_id - 1] if instr_id > 0 else [],
                    uses=[instr_id + 1],
                    instr_type="Int32"
                ))
        blocks.append(create_block(block_id, [entry_id], [merge_id], else_instrs))
        block_id += 1

        # Merge block
        merge_base = block_id * instructions_per_block
        merge_instrs = []
        then_last = then_base + instructions_per_block - 2
        else_last = else_base + instructions_per_block - 2

        for j in range(instructions_per_block):
            instr_id = merge_base + j
            if j == 0:
                # Phi instruction
                merge_instrs.append(create_instruction(
                    instr_id,
                    f"Phi <- Instr#{then_last}, Instr#{else_last}",
                    inputs=[then_last, else_last],
                    uses=[instr_id + 1] if j < instructions_per_block - 1 else [],
                    instr_type="Int32"
                ))
            elif j == instructions_per_block - 1:
                if d < num_diamonds - 1:
                    merge_instrs.append(create_instruction(
                        instr_id,
                        f"Goto -> block {block_id + 1}",
                        inputs=[instr_id - 1],
                        instr_type="None"
                    ))
                else:
                    merge_instrs.append(create_instruction(
                        instr_id,
                        f"Return <- Instr#{instr_id - 1}",
                        inputs=[instr_id - 1],
                        instr_type="None"
                    ))
            else:
                merge_instrs.append(create_instruction(
                    instr_id,
                    f"Add <- Instr#{instr_id - 1}, Constant [int32]",
                    inputs=[instr_id - 1],
                    uses=[instr_id + 1],
                    instr_type="Int32"
                ))

        next_successors = [block_id + 1] if d < num_diamonds - 1 else []
        blocks.append(create_block(block_id, [then_id, else_id], next_successors, merge_instrs))
        block_id += 1

    return blocks


def generate_loop_graph(num_loops, loop_iterations=3, instructions_per_block=5):
    """
    Generate a graph with nested loops.

    Structure:
        Entry -> LoopHeader -> LoopBody -> LoopBack (backedge to LoopHeader)
                     |
                     v
                   Exit
    """
    blocks = []
    block_id = 0

    for loop_idx in range(num_loops):
        loop_depth = 1

        # Entry block (or continuation from previous loop)
        entry_id = block_id
        header_id = block_id + 1
        body_id = block_id + 2
        back_id = block_id + 3
        exit_id = block_id + 4

        predecessors = [block_id - 1] if loop_idx > 0 else []

        # Entry block
        entry_base = block_id * instructions_per_block
        entry_instrs = []
        for j in range(instructions_per_block):
            instr_id = entry_base + j
            if j == 0 and loop_idx == 0:
                entry_instrs.append(create_instruction(instr_id, "Start"))
            elif j == instructions_per_block - 1:
                entry_instrs.append(create_instruction(
                    instr_id,
                    f"Goto -> block {header_id}",
                    inputs=[instr_id - 1] if instr_id > 0 else [],
                    instr_type="None"
                ))
            else:
                entry_instrs.append(create_instruction(
                    instr_id,
                    f"Constant 0x{j:x}",
                    uses=[instr_id + 1],
                    instr_type="Int32"
                ))
        blocks.append(create_block(block_id, predecessors, [header_id], entry_instrs))
        block_id += 1

        # Loop header
        header_base = block_id * instructions_per_block
        header_instrs = []
        for j in range(instructions_per_block):
            instr_id = header_base + j
            if j == 0:
                # Phi for loop variable
                header_instrs.append(create_instruction(
                    instr_id,
                    f"Phi <- Entry, BackEdge",
                    uses=[instr_id + 1],
                    instr_type="Int32"
                ))
            elif j == instructions_per_block - 1:
                # Loop condition test
                header_instrs.append(create_instruction(
                    instr_id,
                    f"Test <- Compare#{instr_id - 1} -> block {body_id}, block {exit_id}",
                    inputs=[instr_id - 1],
                    instr_type="Bool"
                ))
            elif j == instructions_per_block - 2:
                # Compare instruction
                header_instrs.append(create_instruction(
                    instr_id,
                    f"Compare <- Phi#{header_base}, Constant Lt",
                    inputs=[header_base],
                    uses=[instr_id + 1],
                    instr_type="Bool"
                ))
            else:
                header_instrs.append(create_instruction(
                    instr_id,
                    "InterruptCheck",
                    instr_type="None"
                ))
        blocks.append(create_block(
            block_id,
            [entry_id, back_id],
            [body_id, exit_id],
            header_instrs,
            loop_depth=loop_depth,
            attributes=["loopheader"]
        ))
        block_id += 1

        # Loop body
        body_base = block_id * instructions_per_block
        body_instrs = []
        for j in range(instructions_per_block):
            instr_id = body_base + j
            if j == instructions_per_block - 1:
                body_instrs.append(create_instruction(
                    instr_id,
                    f"Goto -> block {back_id}",
                    inputs=[instr_id - 1],
                    instr_type="None"
                ))
            else:
                body_instrs.append(create_instruction(
                    instr_id,
                    f"Add <- Phi#{header_base}, Constant [int32]",
                    inputs=[header_base],
                    uses=[instr_id + 1] if j < instructions_per_block - 2 else [],
                    instr_type="Int32"
                ))
        blocks.append(create_block(block_id, [header_id], [back_id], body_instrs, loop_depth=loop_depth))
        block_id += 1

        # Backedge block
        back_base = block_id * instructions_per_block
        back_instrs = []
        for j in range(instructions_per_block):
            instr_id = back_base + j
            if j == instructions_per_block - 1:
                back_instrs.append(create_instruction(
                    instr_id,
                    f"Goto -> block {header_id}",
                    inputs=[instr_id - 1],
                    instr_type="None"
                ))
            elif j == 0:
                # Increment loop counter
                back_instrs.append(create_instruction(
                    instr_id,
                    f"Add <- Phi#{header_base}, Constant 0x1 [int32]",
                    inputs=[header_base],
                    uses=[instr_id + 1],
                    instr_type="Int32"
                ))
            else:
                back_instrs.append(create_instruction(
                    instr_id,
                    f"Nop",
                    uses=[instr_id + 1] if j < instructions_per_block - 2 else [],
                    instr_type="None"
                ))
        blocks.append(create_block(
            block_id,
            [body_id],
            [header_id],
            back_instrs,
            loop_depth=loop_depth,
            attributes=["backedge"]
        ))
        block_id += 1

        # Exit block
        exit_base = block_id * instructions_per_block
        exit_instrs = []
        for j in range(instructions_per_block):
            instr_id = exit_base + j
            if j == instructions_per_block - 1:
                if loop_idx < num_loops - 1:
                    exit_instrs.append(create_instruction(
                        instr_id,
                        f"Goto -> block {block_id + 1}",
                        inputs=[instr_id - 1],
                        instr_type="None"
                    ))
                else:
                    exit_instrs.append(create_instruction(
                        instr_id,
                        f"Return <- Instr#{instr_id - 1}",
                        inputs=[instr_id - 1],
                        instr_type="None"
                    ))
            else:
                exit_instrs.append(create_instruction(
                    instr_id,
                    f"Add <- Phi#{header_base}, Constant [int32]",
                    inputs=[header_base],
                    uses=[instr_id + 1],
                    instr_type="Int32"
                ))

        next_successors = [block_id + 1] if loop_idx < num_loops - 1 else []
        blocks.append(create_block(block_id, [header_id], next_successors, exit_instrs))
        block_id += 1

    return blocks


def generate_complex_cfg(num_blocks, branch_probability=0.3, loop_probability=0.1, instructions_per_block=5):
    """
    Generate a complex control flow graph with random branching and loops.
    """
    if num_blocks < 2:
        return generate_chain_graph(num_blocks, instructions_per_block)

    blocks = []
    block_id = 0

    # Keep track of blocks that need successors
    pending_blocks = []

    while block_id < num_blocks:
        predecessors = []

        # Connect to pending blocks or start fresh
        if pending_blocks and block_id > 0:
            # Take some pending blocks as predecessors
            num_preds = min(len(pending_blocks), random.randint(1, 3))
            for _ in range(num_preds):
                if pending_blocks:
                    pred_id = pending_blocks.pop(0)
                    predecessors.append(pred_id)
                    # Update predecessor's successors
                    for block in blocks:
                        if block["id"] == pred_id:
                            block["successors"].append(block_id)
                            break

        # Determine successors
        successors = []
        is_branch = random.random() < branch_probability and block_id < num_blocks - 2
        is_loop = random.random() < loop_probability and block_id > 2
        loop_depth = 0
        attributes = []

        if block_id == num_blocks - 1:
            # Last block - no successors (return)
            successors = []
        elif is_branch:
            # Branch to two blocks
            next_block = block_id + 1
            other_block = min(block_id + 2, num_blocks - 1)
            if next_block != other_block:
                successors = [next_block, other_block]
                pending_blocks.append(other_block)
            else:
                successors = [next_block]
        elif is_loop and predecessors:
            # Create a backedge
            loop_target = random.choice([p for p in range(max(0, block_id - 5), block_id) if p != block_id])
            successors = [loop_target]
            loop_depth = 1
            attributes = ["backedge"]
            # Mark target as loop header
            for block in blocks:
                if block["id"] == loop_target:
                    block["attributes"] = ["loopheader"]
                    block["loopDepth"] = 1
                    if block_id not in block["predecessors"]:
                        block["predecessors"].append(block_id)
                    break
        else:
            # Simple forward edge
            if block_id < num_blocks - 1:
                successors = [block_id + 1]

        # Generate instructions
        base_instr = block_id * instructions_per_block
        instructions = []

        for j in range(instructions_per_block):
            instr_id = base_instr + j

            if j == 0 and block_id == 0:
                instructions.append(create_instruction(instr_id, "Start"))
            elif j == instructions_per_block - 1:
                # Terminal instruction
                if not successors:
                    instructions.append(create_instruction(
                        instr_id,
                        f"Return <- Instr#{instr_id - 1}",
                        inputs=[instr_id - 1] if instr_id > 0 else [],
                        instr_type="None"
                    ))
                elif len(successors) == 2:
                    instructions.append(create_instruction(
                        instr_id,
                        f"Test <- Compare#{instr_id - 1} -> block {successors[0]}, block {successors[1]}",
                        inputs=[instr_id - 1],
                        instr_type="Bool"
                    ))
                else:
                    instructions.append(create_instruction(
                        instr_id,
                        f"Goto -> block {successors[0]}",
                        inputs=[instr_id - 1] if instr_id > 0 else [],
                        instr_type="None"
                    ))
            elif j == instructions_per_block - 2 and len(successors) == 2:
                # Compare for branch
                instructions.append(create_instruction(
                    instr_id,
                    f"Compare <- Instr#{instr_id - 1}, Constant Lt",
                    inputs=[instr_id - 1] if instr_id > 0 else [],
                    uses=[instr_id + 1],
                    instr_type="Bool"
                ))
            else:
                # Regular instruction
                op = random.choice(["Add", "Subtract", "Multiply", "BitAnd", "BitOr"])
                instructions.append(create_instruction(
                    instr_id,
                    f"{op} <- Instr#{instr_id - 1}, Constant [int32]",
                    inputs=[instr_id - 1] if instr_id > 0 else [],
                    uses=[instr_id + 1] if j < instructions_per_block - 1 else [],
                    instr_type="Int32"
                ))

        blocks.append(create_block(
            block_id,
            predecessors,
            successors,
            instructions,
            loop_depth=loop_depth,
            attributes=attributes
        ))
        block_id += 1

    return blocks


def create_massive_graph(num_blocks, pattern="mixed", instructions_per_block=5):
    """Create a massive graph with the specified number of blocks."""

    if pattern == "chain":
        blocks = generate_chain_graph(num_blocks, instructions_per_block)
    elif pattern == "diamond":
        num_diamonds = max(1, num_blocks // 4)
        blocks = generate_diamond_graph(num_diamonds, instructions_per_block)
    elif pattern == "loop":
        num_loops = max(1, num_blocks // 5)
        blocks = generate_loop_graph(num_loops, instructions_per_block=instructions_per_block)
    elif pattern == "complex":
        blocks = generate_complex_cfg(num_blocks, instructions_per_block=instructions_per_block)
    else:  # mixed
        # Combine different patterns
        blocks = []
        remaining = num_blocks

        # Start with some diamonds
        if remaining > 20:
            num_diamonds = remaining // 10
            diamond_blocks = generate_diamond_graph(num_diamonds, instructions_per_block)
            blocks.extend(diamond_blocks)
            remaining -= len(diamond_blocks)

        # Add some loops
        if remaining > 25:
            num_loops = remaining // 10
            loop_blocks = generate_loop_graph(num_loops, instructions_per_block=instructions_per_block)
            # Renumber blocks
            offset = len(blocks)
            for block in loop_blocks:
                block["id"] += offset
                block["predecessors"] = [p + offset for p in block["predecessors"]]
                block["successors"] = [s + offset for s in block["successors"]]

            # Connect to previous
            if blocks:
                last_block = blocks[-1]
                first_loop = loop_blocks[0]
                last_block["successors"] = [first_loop["id"]]
                first_loop["predecessors"] = [last_block["id"]]
                # Update last instruction
                last_instr = last_block["instructions"][-1]
                last_instr["opcode"] = f"Goto -> block {first_loop['id']}"

            blocks.extend(loop_blocks)
            remaining -= len(loop_blocks)

        # Fill rest with chain
        if remaining > 0:
            chain_blocks = generate_chain_graph(remaining, instructions_per_block)
            offset = len(blocks)
            for block in chain_blocks:
                block["id"] += offset
                block["predecessors"] = [p + offset for p in block["predecessors"]]
                block["successors"] = [s + offset for s in block["successors"]]

            # Connect to previous
            if blocks:
                last_block = blocks[-1]
                first_chain = chain_blocks[0]
                last_block["successors"] = [first_chain["id"]]
                first_chain["predecessors"] = [last_block["id"]]
                # Update last instruction
                last_instr = last_block["instructions"][-1]
                last_instr["opcode"] = f"Goto -> block {first_chain['id']}"

            blocks.extend(chain_blocks)

    return blocks


def scale_existing_graph(input_data, multiplier):
    """
    Scale an existing graph by duplicating its structure.
    Creates multiple copies of each function with expanded blocks.
    """
    output = {
        "version": input_data.get("version", 1),
        "functions": []
    }

    for func_idx, func in enumerate(input_data.get("functions", [])):
        for copy_idx in range(multiplier):
            new_func = {
                "name": f"{func['name']}_copy{copy_idx}",
                "passes": []
            }

            for pass_data in func.get("passes", []):
                new_pass = {
                    "name": pass_data["name"],
                    "mir": {"blocks": []},
                    "lir": {"blocks": []}
                }

                # Copy and expand MIR blocks
                mir_blocks = pass_data.get("mir", {}).get("blocks", [])
                block_offset = copy_idx * len(mir_blocks) * 10

                for block in mir_blocks:
                    new_block = {
                        "ptr": generate_unique_ptr(),
                        "id": block["id"] + block_offset,
                        "loopDepth": block.get("loopDepth", 0),
                        "attributes": block.get("attributes", []).copy(),
                        "predecessors": [p + block_offset for p in block.get("predecessors", [])],
                        "successors": [s + block_offset for s in block.get("successors", [])],
                        "instructions": [],
                        "resumePoint": block.get("resumePoint", {"mode": "ResumeAt", "operands": []})
                    }

                    # Copy instructions with new IDs
                    for instr in block.get("instructions", []):
                        new_instr = {
                            "ptr": generate_unique_ptr(),
                            "id": instr["id"] + block_offset * 100,
                            "opcode": instr["opcode"],
                            "attributes": instr.get("attributes", []).copy(),
                            "inputs": instr.get("inputs", []).copy(),
                            "uses": instr.get("uses", []).copy(),
                            "memInputs": instr.get("memInputs", []).copy(),
                            "type": instr.get("type", "None")
                        }
                        new_block["instructions"].append(new_instr)

                    new_pass["mir"]["blocks"].append(new_block)

                new_func["passes"].append(new_pass)

            output["functions"].append(new_func)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Generate massive IonGraph JSON files for stress testing"
    )
    parser.add_argument("input", help="Input ion-example JSON file (used as template)")
    parser.add_argument("output", help="Output JSON file")

    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument("--blocks", type=int, help="Target number of blocks per function")
    size_group.add_argument("--multiplier", type=int, help="Multiply existing graph size by this factor")

    parser.add_argument("--pattern", choices=["chain", "diamond", "loop", "complex", "mixed"],
                       default="mixed", help="Graph pattern to generate (default: mixed)")
    parser.add_argument("--instructions", type=int, default=8,
                       help="Instructions per block (default: 8)")
    parser.add_argument("--functions", type=int, default=1,
                       help="Number of functions to generate (default: 1)")
    parser.add_argument("--passes", type=int, default=5,
                       help="Number of passes per function (default: 5)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Read input file
    with open(args.input, 'r') as f:
        input_data = json.load(f)

    if args.multiplier:
        # Scale existing graph
        output_data = scale_existing_graph(input_data, args.multiplier)
        print(f"Scaled graph by {args.multiplier}x")
        print(f"  Functions: {len(output_data['functions'])}")
        total_blocks = sum(
            len(p.get("mir", {}).get("blocks", []))
            for f in output_data["functions"]
            for p in f.get("passes", [])
        )
        print(f"  Total blocks (across all passes): {total_blocks}")
    else:
        # Generate new graph with specified size
        output_data = {
            "version": 1,
            "functions": []
        }

        pass_names = [
            "BuildSSA", "Prune Unused Branches", "Fold Empty Blocks",
            "Eliminate phis", "Fold Tests", "Split Critical Edges",
            "Renumber Blocks", "Apply types", "Alias analysis",
            "GVN", "LICM", "Range Analysis", "De-Beta"
        ]

        for func_idx in range(args.functions):
            func = {
                "name": f"massive_func_{func_idx}",
                "passes": []
            }

            for pass_idx in range(args.passes):
                blocks = create_massive_graph(
                    args.blocks,
                    pattern=args.pattern,
                    instructions_per_block=args.instructions
                )

                func["passes"].append({
                    "name": pass_names[pass_idx % len(pass_names)],
                    "mir": {"blocks": blocks},
                    "lir": {"blocks": []}
                })

            output_data["functions"].append(func)

        print(f"Generated massive graph:")
        print(f"  Pattern: {args.pattern}")
        print(f"  Functions: {args.functions}")
        print(f"  Passes per function: {args.passes}")
        print(f"  Blocks per pass: {len(output_data['functions'][0]['passes'][0]['mir']['blocks'])}")
        print(f"  Instructions per block: {args.instructions}")

    # Write output
    with open(args.output, 'w') as f:
        json.dump(output_data, f)

    # Report file size
    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  Output file size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
