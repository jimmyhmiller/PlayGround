function run(program) {
    let pc = 0;
    let stack = [];
    while (pc < program.length) {
        switch (program[pc]) {
            case 0:
                // PUSH: push next value onto stack
                stack.push(program[pc + 1]);
                pc += 2;
                break;
            case 1:
                // ADD: pop two, push sum
                let a = stack.pop();
                let b = stack.pop();
                stack.push(a + b);
                pc += 1;
                break;
        }
    }
    return stack;
}

// PUSH 5, PUSH 3, ADD => 8
var x = run([0, 5, 0, 3, 1, 0, 42, 0, 52, 0, 32, 1, 1, 1]);
var y = run([0, 5, 0, 3, 1, 0, 42]);

