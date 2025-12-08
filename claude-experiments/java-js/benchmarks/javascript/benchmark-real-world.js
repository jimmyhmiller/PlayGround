#!/usr/bin/env node

/**
 * Real-World JavaScript Library Benchmarks
 *
 * Tests parser performance on actual production libraries:
 * - TypeScript compiler (8.6 MB)
 * - Three.js (1.3 MB)
 * - Lodash (531 KB)
 * - Vue 3 (130 KB)
 * - React DOM (129 KB)
 */

import * as acorn from 'acorn';
import { parse as babelParse } from '@babel/parser';
import { parseScript as meriyahParse } from 'meriyah';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const LIBS_DIR = join(__dirname, '..', 'real-world-libs');

// Parse command line arguments: node benchmark-real-world.js [warmup] [measurement]
const args = process.argv.slice(2);
const WARMUP_ITERATIONS = args[0] ? parseInt(args[0], 10) : 5;
const MEASUREMENT_ITERATIONS = args[1] ? parseInt(args[1], 10) : 10;

/**
 * Load a library file
 */
function loadLibrary(filename) {
    const path = join(LIBS_DIR, filename);
    const code = readFileSync(path, 'utf-8');
    return { filename, code, size: code.length };
}

/**
 * High-precision benchmark function
 */
function benchmark(name, fn, iterations) {
    const start = process.hrtime.bigint();

    for (let i = 0; i < iterations; i++) {
        fn();
    }

    const end = process.hrtime.bigint();
    const totalNanos = Number(end - start);
    const avgMillis = (totalNanos / iterations) / 1_000_000;

    return {
        name,
        iterations,
        totalNanos,
        avgMillis: avgMillis.toFixed(3)
    };
}

/**
 * Run benchmark suite for a specific library
 */
function runBenchmarkSuite(library) {
    console.log(`\n${'='.repeat(80)}`);
    console.log(`Library: ${library.filename}`);
    console.log(`Size: ${(library.size / 1024).toFixed(1)} KB (${library.size.toLocaleString()} bytes)`);
    console.log(`${'='.repeat(80)}\n`);

    const parsers = [
        {
            name: 'Acorn',
            parse: () => acorn.parse(library.code, { ecmaVersion: 'latest' })
        },
        {
            name: '@babel/parser',
            parse: () => babelParse(library.code)
        },
        {
            name: 'Meriyah',
            parse: () => meriyahParse(library.code, { next: true })
        }
    ];

    const results = [];

    for (const parser of parsers) {
        console.log(`Warming up ${parser.name}...`);

        // Warmup
        try {
            for (let i = 0; i < WARMUP_ITERATIONS; i++) {
                parser.parse();
            }
        } catch (error) {
            console.log(`  ⚠️  ${parser.name} failed: ${error.message}`);
            continue;
        }

        // Benchmark
        console.log(`Benchmarking ${parser.name}...`);
        try {
            const result = benchmark(parser.name, parser.parse, MEASUREMENT_ITERATIONS);
            results.push(result);
        } catch (error) {
            console.log(`  ⚠️  ${parser.name} failed: ${error.message}`);
        }
    }

    if (results.length === 0) {
        console.log('\n⚠️  All parsers failed for this library\n');
        return null;
    }

    // Sort by performance (fastest first)
    results.sort((a, b) => parseFloat(a.avgMillis) - parseFloat(b.avgMillis));

    // Print results
    console.log('\nResults (sorted by performance):');
    console.log('-'.repeat(80));
    console.log(`${'Parser'.padEnd(20)} | ${'Avg Time (ms)'.padStart(15)} | ${'vs Fastest'.padStart(15)} | ${'Throughput'.padStart(15)}`);
    console.log('-'.repeat(80));

    const fastest = parseFloat(results[0].avgMillis);

    for (const result of results) {
        const avg = parseFloat(result.avgMillis);
        const ratio = (avg / fastest).toFixed(2);
        const throughput = (library.size / avg / 1024).toFixed(1); // KB/ms
        const indicator = result === results[0] ? '🥇' :
                         result === results[1] ? '🥈' :
                         result === results[2] ? '🥉' : '  ';

        console.log(
            `${indicator} ${result.name.padEnd(18)} | ${result.avgMillis.padStart(15)} | ${ratio.padStart(15)}x | ${throughput.padStart(12)} KB/ms`
        );
    }

    return results;
}

/**
 * Main benchmark runner
 */
async function main() {
    console.log('\n' + '═'.repeat(80));
    console.log('Real-World JavaScript Library Parser Benchmarks');
    console.log('═'.repeat(80));
    console.log(`Warmup iterations: ${WARMUP_ITERATIONS}`);
    console.log(`Measurement iterations: ${MEASUREMENT_ITERATIONS}`);
    console.log(`Node.js version: ${process.version}`);
    console.log('═'.repeat(80));

    const libraries = [
        loadLibrary('react.production.min.js'),
        loadLibrary('vue.global.prod.js'),
        loadLibrary('react-dom.production.min.js'),
        loadLibrary('lodash.js'),
        loadLibrary('three.js'),
        loadLibrary('typescript.js'),
    ];

    const allResults = {};

    for (const library of libraries) {
        const results = runBenchmarkSuite(library);
        if (results) {
            allResults[library.filename] = results;
        }
    }

    // Summary
    console.log('\n' + '═'.repeat(80));
    console.log('SUMMARY');
    console.log('═'.repeat(80));

    for (const [libName, results] of Object.entries(allResults)) {
        if (results.length > 0) {
            console.log(`\n${libName}:`);
            console.log(`  🥇 ${results[0].name}: ${results[0].avgMillis} ms`);
        }
    }

    console.log('\n' + '═'.repeat(80));
}

main().catch(console.error);
