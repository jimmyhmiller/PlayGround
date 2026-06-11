// Mandelbrot escape-count sum — mirrors benchmarks/ail/mandelbrot.ail.
package main

import (
	"fmt"
	"time"
)

func escape(cRe, cIm float64, maxIter int64) int64 {
	zRe, zIm := 0.0, 0.0
	for i := int64(0); i < maxIter; i++ {
		zr2 := zRe * zRe
		zi2 := zIm * zIm
		if zr2+zi2 > 4.0 {
			return i
		}
		newIm := 2.0*zRe*zIm + cIm
		zRe = zr2 - zi2 + cRe
		zIm = newIm
	}
	return maxIter
}

func main() {
	var width, height, maxIter int64 = 1000, 1000, 100
	t0 := time.Now()
	var acc int64 = 0
	for py := int64(0); py < height; py++ {
		cIm := -1.25 + 2.5*float64(py)/float64(height)
		for px := int64(0); px < width; px++ {
			cRe := -2.0 + 3.0*float64(px)/float64(width)
			acc += escape(cRe, cIm, maxIter)
		}
	}
	ms := time.Since(t0).Milliseconds()
	fmt.Printf("RESULT mandelbrot %d ms checksum=%d\n", ms, acc)
}
