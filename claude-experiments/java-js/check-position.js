const fs = require('fs');
const acorn = require('acorn');

const file = process.argv[2];
const source = fs.readFileSync(file, 'utf-8');
const ast = acorn.parse(source, { ecmaVersion: 2020, sourceType: 'module' });

// Navigate to body[26].declarations[0]
try {
    const node = ast.body[26].declarations[0];
    console.log("Node type:", node.type);
    console.log("Node start:", node.start);
    console.log("Node end:", node.end);
    console.log("\nContext around end (chars " + (node.end - 20) + " to " + (node.end + 20) + "):");
    console.log(source.substring(node.end - 20, node.end + 20));
    console.log("\nChar at end position:", JSON.stringify(source[node.end - 1]));
    console.log("Char after end:", JSON.stringify(source[node.end]));
} catch (e) {
    console.error("Error:", e.message);
}
