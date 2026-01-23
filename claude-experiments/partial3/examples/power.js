// Compute 2^10 by iterative multiplication
let base = 2;
let exp = 10;
let result = 1;
let i = 0;
while (i < exp) {
    result = result * base;
    i = i + 1;
}
result
