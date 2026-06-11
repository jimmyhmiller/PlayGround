// Binary trees, allocation stress — mirrors benchmarks/ail/binary_trees.ail.
// Pointer-per-node allocation through Go's GC (the closest analogue of
// ai-lang's per-node GC allocation).
package main

import (
	"fmt"
	"time"
)

type tree struct {
	left, right *tree // nil = leaf
}

func mk(d int64) *tree {
	if d == 0 {
		return nil
	}
	return &tree{mk(d - 1), mk(d - 1)}
}

func check(t *tree) int64 {
	if t == nil {
		return 1
	}
	return 1 + check(t.left) + check(t.right)
}

func main() {
	var depth, iters int64 = 16, 40
	t0 := time.Now()
	var acc int64 = 0
	for i := int64(0); i < iters; i++ {
		acc += check(mk(depth))
	}
	ms := time.Since(t0).Milliseconds()
	fmt.Printf("RESULT binary_trees %d ms checksum=%d\n", ms, acc)
}
