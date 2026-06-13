// Tight integer loop with xor/shift mix — mirrors benchmarks/ail/loop_mix.ail.
// The shift is logical (zero-filling) to match ai-lang's bit_shr.
package main

import (
	"fmt"
	"time"
)

func main() {
	var n int64 = 500_000_000
	t0 := time.Now()
	var acc int64 = 0
	for i := int64(0); i < n; i++ {
		acc = (acc + i) ^ int64(uint64(acc)>>13)
	}
	ms := time.Since(t0).Milliseconds()
	fmt.Printf("RESULT loop_mix %d ms checksum=%d\n", ms, acc)
}
