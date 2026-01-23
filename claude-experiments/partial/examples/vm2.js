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
            case 2:
                // MUL: pop two, push product
                let x = stack.pop();
                let y = stack.pop();
                stack.push(x * y);
                pc += 1;
                break;
        }
    }
    return stack[0];
}

// (2 + 3) * 10 = 50
run([0, 2, 0, 3, 1, 0, 10, 2]);
