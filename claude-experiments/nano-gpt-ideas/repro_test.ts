// Minimal repro: allocate a large Float32Array at the end of a function
// that also uses many other buffers, and verify the return value is correct.

export const Float32Array_ID = idof<Float32Array>();

export function test_return(input: Float32Array): Float32Array {
    // Simulate a compute graph with many buffers
    const n: i32 = 1000;
    const buf0 = new Float32Array(n);
    const buf1 = new Float32Array(n);
    const buf2 = new Float32Array(n);
    const buf3 = new Float32Array(n);
    const buf4 = new Float32Array(n);

    // Do some work
    for (let i: i32 = 0; i < n; i++) {
        buf0[i] = input[i % input.length] + f32(1.0);
        buf1[i] = buf0[i] * f32(2.0);
        buf2[i] = buf1[i] - f32(0.5);
        buf3[i] = buf2[i] * buf0[i];
        buf4[i] = buf3[i] + buf1[i];
    }

    // Allocate large output and copy (like our multi-output code)
    const total: i32 = 100000;
    const __out = new Float32Array(total);
    let __off: i32 = 0;

    for (let __i: i32 = 0; __i < n; __i++) __out[__off + __i] = buf0[__i];
    __off += n;
    for (let __i: i32 = 0; __i < n; __i++) __out[__off + __i] = buf1[__i];
    __off += n;
    for (let __i: i32 = 0; __i < n; __i++) __out[__off + __i] = buf2[__i];
    __off += n;
    for (let __i: i32 = 0; __i < n; __i++) __out[__off + __i] = buf3[__i];
    __off += n;
    for (let __i: i32 = 0; __i < n; __i++) __out[__off + __i] = buf4[__i];
    __off += n;

    // Fill the rest with a known pattern
    for (let __i: i32 = __off; __i < total; __i++) __out[__i] = f32(42.0);

    return __out;
}
