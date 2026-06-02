// A tiny expression interpreter dispatching with switch.
// Specializing it against the static `ast` makes the interpreter vanish.
function evalNode(node, env) {
    switch (node.op) {
        case "lit": return node.val;
        case "var": return env[node.name];
        case "add": return evalNode(node.l, env) + evalNode(node.r, env);
        case "mul": return evalNode(node.l, env) * evalNode(node.r, env);
        default: return 0;
    }
}

function main(x) {
    // (x + 3) * (x + x)
    let ast = {
        op: "mul",
        l: { op: "add", l: { op: "var", name: "x" }, r: { op: "lit", val: 3 } },
        r: { op: "add", l: { op: "var", name: "x" }, r: { op: "var", name: "x" } }
    };
    let env = { x: x };
    return evalNode(ast, env);
}
