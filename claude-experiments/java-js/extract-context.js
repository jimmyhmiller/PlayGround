const fs = require('fs');
const acorn = require('acorn');

const file = process.argv[2];
const source = fs.readFileSync(file, 'utf-8');
const ast = acorn.parse(source, { ecmaVersion: 2020 });

// Navigate to body[0].expression.callee.object.body.body[248].expression.right
try {
    const node = ast.body[0].expression.callee.object.body.body[248].expression.right;
    console.log("Found node:");
    console.log("Type:", node.type);
    console.log("Value:", node.value);
    console.log("Raw:", node.raw);
    console.log("\nContext (chars " + (node.start - 50) + " to " + (node.end + 50) + "):");
    console.log(source.substring(node.start - 50, node.end + 50));
} catch (e) {
    console.error("Error navigating to node:", e.message);
}
