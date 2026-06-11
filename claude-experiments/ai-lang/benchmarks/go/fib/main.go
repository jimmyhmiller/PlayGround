// Naive recursive Fibonacci — mirrors benchmarks/ail/fib.ail.
package main

import (
	"fmt"
	"time"
)

func fib(n int64) int64 {
	if n < 2 {
		return n
	}
	return fib(n-1) + fib(n-2)
}

func main() {
	var n int64 = 32
	t0 := time.Now()
	r := fib(n)
	ms := time.Since(t0).Milliseconds()
	fmt.Printf("RESULT fib %d ms checksum=%d\n", ms, r)
}
