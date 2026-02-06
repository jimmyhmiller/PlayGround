// A complex stack-based bytecode VM with DYNAMIC conditional branches.
// The bytecode program is KNOWN at partial eval time.
// The input is DYNAMIC (event.clientX).
// The partial evaluator should:
//   1. Eliminate the entire dispatch loop (while + switch)
//   2. Eliminate all stack/locals/sp/pc machinery
//   3. SPLIT on the dynamic branch (x > 10), emitting a residual if-statement
//
// Algorithm encoded in bytecode:
//   x = input
//   t = x * x
//   if (x > 10)        <<< DYNAMIC - depends on runtime input!
//     result = t + 42
//   else
//     result = t * 3 + 7
//   console.log(result)
//   return result

var OP_CONST   = 0;   // push constants[arg]          (2 bytes)
var OP_INPUT   = 1;   // push the dynamic input       (1 byte)
var OP_ADD     = 2;   // pop b, pop a, push a+b       (1 byte)
var OP_MUL     = 4;   // pop b, pop a, push a*b       (1 byte)
var OP_DUP     = 7;   // duplicate top of stack        (1 byte)
var OP_STORE   = 8;   // pop and store in locals[arg]  (2 bytes)
var OP_LOAD    = 9;   // push locals[arg]              (2 bytes)
var OP_CMP_GT  = 10;  // pop b, pop a, push (a>b?1:0) (1 byte)
var OP_JZ      = 11;  // pop, if falsy set pc = arg    (2 bytes)
var OP_JMP     = 12;  // set pc = arg                  (2 bytes)
var OP_PRINT   = 13;  // print top of stack (no pop)   (1 byte)
var OP_RET     = 14;  // return top of stack           (1 byte)

var constants = [10, 42, 3, 7];

// Bytecode - every index verified:
//
//  [0]  INPUT                           -> pc=1
//  [1]  DUP                             -> pc=2
//  [2]  STORE 0      locals[0] = x      -> pc=4
//  [4]  DUP                             -> pc=5
//  [5]  MUL          x * x              -> pc=6
//  [6]  STORE 1      locals[1] = x*x    -> pc=8
//  [8]  LOAD 0       push x             -> pc=10
// [10]  CONST 0      push 10            -> pc=12
// [12]  CMP_GT       x > 10 ?  DYNAMIC  -> pc=13
// [13]  JZ 24        if false -> 24     -> pc=15 or 24
// --- true branch ---
// [15]  LOAD 1       push x*x           -> pc=17
// [17]  CONST 1      push 42            -> pc=19
// [19]  ADD          x*x + 42           -> pc=20
// [20]  STORE 1      locals[1] = result  -> pc=22
// [22]  JMP 34       skip false branch  -> pc=34
// --- false branch ---
// [24]  LOAD 1       push x*x           -> pc=26
// [26]  CONST 2      push 3             -> pc=28
// [28]  MUL          x*x * 3            -> pc=29
// [29]  CONST 3      push 7             -> pc=31
// [31]  ADD          x*x*3 + 7          -> pc=32
// [32]  STORE 1      locals[1] = result  -> pc=34
// --- merge ---
// [34]  LOAD 1       push result        -> pc=36
// [36]  PRINT        log result         -> pc=37
// [37]  RET          return result

var program = [
    1,                 //  [0]: INPUT
    7,                 //  [1]: DUP
    8, 0,              //  [2]: STORE 0
    7,                 //  [4]: DUP
    4,                 //  [5]: MUL
    8, 1,              //  [6]: STORE 1
    9, 0,              //  [8]: LOAD 0
    0, 0,              // [10]: CONST 0  (push 10)
    10,                // [12]: CMP_GT
    11, 24,            // [13]: JZ -> 24
    9, 1,              // [15]: LOAD 1
    0, 1,              // [17]: CONST 1  (push 42)
    2,                 // [19]: ADD
    8, 1,              // [20]: STORE 1
    12, 34,            // [22]: JMP -> 34
    9, 1,              // [24]: LOAD 1
    0, 2,              // [26]: CONST 2  (push 3)
    4,                 // [28]: MUL
    0, 3,              // [29]: CONST 3  (push 7)
    2,                 // [31]: ADD
    8, 1,              // [32]: STORE 1
    9, 1,              // [34]: LOAD 1
    13,                // [36]: PRINT
    14                 // [37]: RET
];

function runVM(input) {
    var stack = [];
    var locals = [0, 0, 0, 0];
    var sp = 0;
    var pc = 0;

    while (pc < program.length) {
        var op = program[pc];
        switch (op) {
            case 0:  // CONST
                stack[sp] = constants[program[pc + 1]];
                sp = sp + 1;
                pc = pc + 2;
                break;
            case 1:  // INPUT
                stack[sp] = input;
                sp = sp + 1;
                pc = pc + 1;
                break;
            case 2:  // ADD
                sp = sp - 1;
                stack[sp - 1] = stack[sp - 1] + stack[sp];
                pc = pc + 1;
                break;
            case 4:  // MUL
                sp = sp - 1;
                stack[sp - 1] = stack[sp - 1] * stack[sp];
                pc = pc + 1;
                break;
            case 7:  // DUP
                stack[sp] = stack[sp - 1];
                sp = sp + 1;
                pc = pc + 1;
                break;
            case 8:  // STORE
                sp = sp - 1;
                locals[program[pc + 1]] = stack[sp];
                pc = pc + 2;
                break;
            case 9:  // LOAD
                stack[sp] = locals[program[pc + 1]];
                sp = sp + 1;
                pc = pc + 2;
                break;
            case 10: // CMP_GT
                sp = sp - 1;
                stack[sp - 1] = stack[sp - 1] > stack[sp] ? 1 : 0;
                pc = pc + 1;
                break;
            case 11: // JZ
                sp = sp - 1;
                if (stack[sp] === 0) {
                    pc = program[pc + 1];
                } else {
                    pc = pc + 2;
                }
                break;
            case 12: // JMP
                pc = program[pc + 1];
                break;
            case 13: // PRINT
                console.log(stack[sp - 1]);
                pc = pc + 1;
                break;
            case 14: // RET
                return stack[sp - 1];
        }
    }
    return stack[sp - 1];
}

document.addEventListener("click", function(event) {
    runVM(event.clientX);
});
