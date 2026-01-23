// Some static computations
let a = 5;
let b = 3;
let c = a + b;  // Will be folded to 8

// Dynamic computation using external input
let d = input * 2;
let e = c + input;  // c is static (8), but result is dynamic
