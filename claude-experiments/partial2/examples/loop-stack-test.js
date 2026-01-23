function run(program) {
    let pc = 0;
    let stack = [];
    let sp = 0;

    while (pc < program.length) {
        switch (program[pc]) {
            case 0: // PUSH
                stack[sp++] = program[pc + 1];
                pc += 2;
                break;
            case 1: // ADD
                let a = stack[--sp];
                let b = stack[--sp];
                stack[sp++] = a + b;
                pc += 1;
                break;
        }
    }
    return stack[0];
}

// Push 2, Push 3, ADD -> result should be 5
var result = run([0, 2, 0, 3, 1]);
console.log(result);
