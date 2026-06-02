// A Brainfuck interpreter. Specializing it against the static `program`
// (with input fed to cell 0 via `,`) makes the interpreter and AST vanish.
function exec(node, tape, ptr, out, input, inptr) {
    switch (node.op) {
        case "add":  { tape[ptr] = tape[ptr] + node.n; break; }
        case "move": { ptr = ptr + node.n; break; }
        case "out":  { out.push(tape[ptr]); break; }
        case "in":   { tape[ptr] = input[inptr]; inptr = inptr + 1; break; }
        case "loop": {
            while (tape[ptr] !== 0) {
                let r = exec(node.seq, tape, ptr, out, input, inptr);
                ptr = r[0]; inptr = r[1];
            }
            break;
        }
        case "seq": {
            let i = 0;
            while (i < node.body.length) {
                let r = exec(node.body[i], tape, ptr, out, input, inptr);
                ptr = r[0]; inptr = r[1];
                i = i + 1;
            }
            break;
        }
    }
    return [ptr, inptr];
}

function main(x) {
    let tape = [0, 0, 0, 0, 0, 0, 0, 0];
    let out = [];
    let input = [x];
    // ,++++++.>+++++++++++++[->+++++<]>.
    let program = { op: "seq", body: [
        { op: "in" }, { op: "add", n: 6 }, { op: "out" },
        { op: "move", n: 1 }, { op: "add", n: 13 },
        { op: "loop", seq: { op: "seq", body: [
            { op: "add", n: -1 }, { op: "move", n: 1 },
            { op: "add", n: 5 }, { op: "move", n: -1 } ] } },
        { op: "move", n: 1 }, { op: "out" }
    ] };
    let r = exec(program, tape, 0, out, input, 0);
    return out;
}
