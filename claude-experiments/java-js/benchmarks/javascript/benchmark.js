#!/usr/bin/env node

/**
 * JavaScript Parser Benchmarks
 *
 * Compares performance of JavaScript parsers written in JavaScript:
 * - Acorn (widely used, fast)
 * - Esprima (reference implementation)
 * - @babel/parser (most popular)
 * - Meriyah (optimized for speed)
 */

import * as acorn from 'acorn';
import * as esprima from 'esprima';
import { parse as babelParse } from '@babel/parser';
import { parseScript as meriyahParse } from 'meriyah';
import { SMALL_FUNCTION, SMALL_CLASS, MEDIUM_ASYNC_MODULE, LARGE_MODULE } from './test-data.js';

const WARMUP_ITERATIONS = 1000;
const MEASUREMENT_ITERATIONS = 10000;

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
    const avgMicros = (totalNanos / iterations) / 1000;

    return {
        name,
        iterations,
        totalNanos,
        avgMicros: avgMicros.toFixed(3)
    };
}

/**
 * Run benchmark suite for a specific code sample
 */
function runBenchmarkSuite(suiteName, code) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Benchmark: ${suiteName}`);
    console.log(`Code size: ${code.length} characters`);
    console.log(`${'='.repeat(60)}\n`);

    const parsers = [
        {
            name: 'Acorn',
            parse: () => acorn.parse(code, { ecmaVersion: 'latest' })
        },
        {
            name: 'Esprima',
            parse: () => esprima.parseScript(code)
        },
        {
            name: '@babel/parser',
            parse: () => babelParse(code)
        },
        {
            name: 'Meriyah',
            parse: () => meriyahParse(code, { next: true })
        }
    ];

    const results = [];

    for (const parser of parsers) {
        // Warmup
        console.log(`Warming up ${parser.name}...`);
        for (let i = 0; i < WARMUP_ITERATIONS; i++) {
            parser.parse();
        }

        // Benchmark
        console.log(`Benchmarking ${parser.name}...`);
        const result = benchmark(parser.name, parser.parse, MEASUREMENT_ITERATIONS);
        results.push(result);
    }

    // Sort by performance (fastest first)
    results.sort((a, b) => parseFloat(a.avgMicros) - parseFloat(b.avgMicros));

    // Print results
    console.log('\nResults (sorted by performance):');
    console.log('-'.repeat(60));
    console.log(`${'Parser'.padEnd(20)} | ${'Avg Time (µs)'.padStart(15)} | ${'vs Fastest'.padStart(15)}`);
    console.log('-'.repeat(60));

    const fastest = parseFloat(results[0].avgMicros);

    for (const result of results) {
        const avg = parseFloat(result.avgMicros);
        const ratio = (avg / fastest).toFixed(2);
        const indicator = result === results[0] ? '🥇' :
                         result === results[1] ? '🥈' :
                         result === results[2] ? '🥉' : '  ';

        console.log(
            `${indicator} ${result.name.padEnd(18)} | ${result.avgMicros.padStart(15)} | ${ratio.padStart(15)}x`
        );
    }

    return results;
}

/**
 * Main benchmark runner
 */
async function main() {
    console.log('\n' + '═'.repeat(60));
    console.log('JavaScript Parser Benchmarks (JavaScript-based parsers)');
    console.log('═'.repeat(60));
    console.log(`Warmup iterations: ${WARMUP_ITERATIONS}`);
    console.log(`Measurement iterations: ${MEASUREMENT_ITERATIONS}`);
    console.log(`Node.js version: ${process.version}`);
    console.log('═'.repeat(60));

    const allResults = {
        'Small Function': runBenchmarkSuite('Small Function', SMALL_FUNCTION),
        'Small Class': runBenchmarkSuite('Small Class', SMALL_CLASS),
        'Medium Async Module': runBenchmarkSuite('Medium Async Module', MEDIUM_ASYNC_MODULE),
        'Large Module': runBenchmarkSuite('Large Module', LARGE_MODULE)
    };

    // Summary
    console.log('\n' + '═'.repeat(60));
    console.log('SUMMARY');
    console.log('═'.repeat(60));

    for (const [suiteName, results] of Object.entries(allResults)) {
        console.log(`\n${suiteName}:`);
        console.log(`  🥇 ${results[0].name}: ${results[0].avgMicros} µs`);
    }

    // Export results as JSON for integration with Java benchmarks
    const jsonResults = {};
    for (const [suiteName, results] of Object.entries(allResults)) {
        jsonResults[suiteName] = results.map(r => ({
            parser: r.name,
            avgMicros: parseFloat(r.avgMicros)
        }));
    }

    console.log('\n' + '═'.repeat(60));
    console.log('JSON Results (for integration):');
    console.log(JSON.stringify(jsonResults, null, 2));
}

main().catch(console.error);
