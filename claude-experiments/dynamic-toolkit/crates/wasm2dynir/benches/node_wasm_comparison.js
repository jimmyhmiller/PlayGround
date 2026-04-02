// Node.js WebAssembly benchmark — same programs as the Rust JIT benchmarks
// Run: node benches/node_wasm_comparison.js

const fs = require('fs');

const wat_to_wasm = null; // We'll use pre-compiled wasm

// Inline WAT as wasm bytes using wabt if available, else use hand-crafted approach
// For simplicity, we'll define the functions in JS and compile them to wasm via WebAssembly API

async function main() {
    // Fibonacci (iterative)
    const fibWat = `(module
        (func (export "fib") (param $n i32) (result i32)
            (local $a i32) (local $b i32) (local $i i32) (local $tmp i32)
            i32.const 0  local.set $a
            i32.const 1  local.set $b
            i32.const 0  local.set $i
            block $exit loop $loop
                local.get $i  local.get $n  i32.ge_s  br_if $exit
                local.get $a  local.get $b  i32.add  local.set $tmp
                local.get $b  local.set $a
                local.get $tmp  local.set $b
                local.get $i  i32.const 1  i32.add  local.set $i
                br $loop
            end end
            local.get $a))`;

    const factWat = `(module
        (func (export "fact") (param $n i32) (result i32)
            (local $result i32)
            i32.const 1  local.set $result
            block $exit loop $loop
                local.get $n  i32.const 1  i32.le_s  br_if $exit
                local.get $result  local.get $n  i32.mul  local.set $result
                local.get $n  i32.const 1  i32.sub  local.set $n
                br $loop
            end end
            local.get $result))`;

    const sumWat = `(module
        (func (export "sum") (param $n i32) (result i32)
            (local $i i32) (local $acc i32)
            i32.const 0  local.set $i
            i32.const 0  local.set $acc
            block $exit loop $loop
                local.get $i  local.get $n  i32.ge_s  br_if $exit
                local.get $acc  local.get $i  i32.add  local.set $acc
                local.get $i  i32.const 1  i32.add  local.set $i
                br $loop
            end end
            local.get $acc))`;

    const nestedWat = `(module
        (func (export "nested") (param $n i32) (result i32)
            (local $i i32) (local $j i32) (local $acc i32)
            i32.const 0  local.set $acc
            i32.const 0  local.set $i
            block $ei loop $li
                local.get $i  local.get $n  i32.ge_s  br_if $ei
                i32.const 0  local.set $j
                block $ej loop $lj
                    local.get $j  local.get $n  i32.ge_s  br_if $ej
                    local.get $acc  local.get $i  local.get $j  i32.mul  i32.add  local.set $acc
                    local.get $j  i32.const 1  i32.add  local.set $j
                    br $lj
                end end
                local.get $i  i32.const 1  i32.add  local.set $i
                br $li
            end end
            local.get $acc))`;

    // Try to use wabt to compile WAT to WASM
    let wabt;
    try {
        wabt = require('wabt');
    } catch {
        console.log("wabt not available, trying wat2wasm CLI...");
    }

    async function compileWat(wat) {
        if (wabt) {
            const w = await wabt();
            const mod = w.parseWat('bench.wat', wat);
            const { buffer } = mod.toBinary({});
            return buffer;
        }
        // Fallback: use wat2wasm if installed
        const { execSync } = require('child_process');
        const tmp = '/tmp/bench_' + Math.random().toString(36).slice(2) + '.wat';
        fs.writeFileSync(tmp, wat);
        try {
            const result = execSync(`wat2wasm ${tmp} -o /dev/stdout`);
            return result;
        } finally {
            fs.unlinkSync(tmp);
        }
    }

    function bench(name, fn, warmup = 1000, iterations = 10000) {
        for (let i = 0; i < warmup; i++) fn();
        const start = process.hrtime.bigint();
        for (let i = 0; i < iterations; i++) fn();
        const elapsed = Number(process.hrtime.bigint() - start);
        const per_iter = elapsed / iterations;
        console.log(`${name}: ${per_iter.toFixed(1)} ns/iter`);
    }

    try {
        const fibWasm = await compileWat(fibWat);
        const factWasm = await compileWat(factWat);
        const sumWasm = await compileWat(sumWat);
        const nestedWasm = await compileWat(nestedWat);

        const fibMod = await WebAssembly.instantiate(fibWasm);
        const factMod = await WebAssembly.instantiate(factWasm);
        const sumMod = await WebAssembly.instantiate(sumWasm);
        const nestedMod = await WebAssembly.instantiate(nestedWasm);

        const fib = fibMod.instance.exports.fib;
        const fact = factMod.instance.exports.fact;
        const sum = sumMod.instance.exports.sum;
        const nested = nestedMod.instance.exports.nested;

        console.log("=== Node.js " + process.version + " WebAssembly ===");
        bench("wasm fib(30)", () => fib(30));
        bench("wasm factorial(20)", () => fact(20));
        bench("wasm sum(10000)", () => sum(10000), 100, 1000);
        bench("wasm nested_loop(100)", () => nested(100), 100, 1000);
    } catch (e) {
        console.error("Could not compile WAT:", e.message);
        console.error("Install wabt: npm install wabt");
    }
}

main();
