#!/bin/bash
# Debug script for stepping through JIT code

set -e

echo "Building debug example..."
cargo build --example debug_spill

echo ""
echo "Starting lldb..."
echo ""
echo "Quick reference:"
echo "  run        - Run until breakpoint"
echo "  si         - Step one instruction"
echo "  c          - Continue"
echo "  register read x0 x1 x29 x30 sp"
echo "  memory read -fx -c16 \$sp"
echo "  memory read -fx -c16 \$x29"
echo ""
echo "Or load custom commands:"
echo "  command source debug_commands.lldb"
echo ""

lldb target/debug/examples/debug_spill
