import * as fs from 'fs';
const ast = JSON.parse(fs.readFileSync('/tmp/our-webpack-ast.json', 'utf-8'));

function countNodes(node, counts) {
    if (node === null || node === undefined || typeof node !== 'object') return counts;
    if (node.type) {
        counts[node.type] = (counts[node.type] || 0) + 1;
    }
    for (const key of Object.keys(node)) {
        if (key === 'loc') continue;
        const val = node[key];
        if (Array.isArray(val)) {
            val.forEach(v => countNodes(v, counts));
        } else if (typeof val === 'object') {
            countNodes(val, counts);
        }
    }
    return counts;
}

const counts = countNodes(ast, {});
const sorted = Object.entries(counts).sort((a,b) => b[1] - a[1]);
console.log('Top node types (Our parser):');
sorted.slice(0, 20).forEach(([type, count]) => console.log('  ' + type + ': ' + count));
console.log('Total unique types:', sorted.length);
