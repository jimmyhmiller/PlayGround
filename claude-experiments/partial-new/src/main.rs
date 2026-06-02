//! A principled, generic partial evaluator, demonstrated by "compiling"
//! Brainfuck via the first Futamura projection.
//!
//! `engine.rs` is the entire generic specializer. `bf.rs` is one client. The
//! engine never mentions Brainfuck; a second client would reuse it verbatim.

use partial::{bf, engine, fun, imp, js, residual};

use partial::imp::{bin, int, var, BinOp::*, Stmt::*};

fn demo(name: &str, src: &str, input: &[u8], show_residual: bool) {
    println!("================================================================");
    println!("  {name}");
    println!("================================================================");

    let bf = bf::Bf::new(src);
    let prog = engine::specialize(&bf, bf::State::start());

    println!(
        "source commands : {}\nresidual blocks : {}\nresidual ops    : {}",
        bf.prog_len(),
        prog.blocks.len(),
        prog.op_count(),
    );

    if show_residual {
        println!("\n--- residual program (no pc, no dispatch; pure tape ops) ---");
        print!("{prog}");
    }

    // Prove the partial evaluation preserved semantics.
    let reference = bf::run_reference(src, input);
    let residual = bf::run_residual(&prog, input);
    assert_eq!(
        reference, residual,
        "residual output diverged from the reference interpreter!"
    );

    println!("\noutput          : {:?}", String::from_utf8_lossy(&residual));
    println!("matches oracle  : yes ({} bytes)\n", residual.len());
}

fn main() {
    // 1. A run of `+` coalesces into one residual op (partially-static cell).
    demo("constant: print '0'", &("+".repeat(48) + "."), &[], true);

    // 2. A loop: 3 * 3 = 9 (tab). The dispatch melts; the BF loop survives as a
    //    residual loop whose back-edge was tied by memoization.
    demo("loop: 3*3 via [>+++<-]", "+++[>+++<-]>.", &[], true);

    // 3. Pointer motion coalesces across the run (partially-static pointer).
    demo("pointer coalescing", ">>>+++<<<++.>>>.", &[], true);

    // 4. The classic: Hello World. Many nested loops, all tied off; output is
    //    verified byte-for-byte against the reference interpreter.
    let hello = "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]\
                 >>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.";
    demo("Hello World", hello, &[], false);

    // 5. echo: read until 0, write back. Demonstrates I/O residualization and a
    //    data-dependent loop over dynamic input.
    demo("cat (echo input)", ",[.,]", b"partial!\0", false);

    imp_demos();
    fun_demos();
    js_demos();

    println!("All demos matched the reference interpreter.");
}

fn js_demo(name: &str, program: &[js::FuncDef], inputs: &[i64]) {
    println!("================================================================");
    println!("  JS (objects/arrays/closures): {name}");
    println!("================================================================");

    let vm = js::Js::new(program);
    let mut prog = engine::specialize(&vm, vm.start());
    residual::simplify(&mut prog);

    println!(
        "residual blocks : {}\nresidual ops    : {}\n",
        prog.blocks.len(),
        prog.op_count()
    );
    print!("{}", vm.dump(&prog));

    for &inp in inputs {
        let reference = vm.run_reference(inp);
        let got = vm.run_residual(&prog, inp);
        assert_eq!(reference, got, "residual diverged on input {inp}");
        println!("  input {inp} -> {got:?}  (matches oracle)");
    }
    println!();
}

fn js_demos() {
    use partial::js::{arr, bin, call, closure, func, get, index, num, obj, str_, var, Bop::*, FuncDef};
    use partial::js::Stmt::*;

    // 1. A static object is scalar-replaced away: property access folds.
    //    main(_) = { let p = {x:3, y:4}; return p.x + p.y; }  ->  return 7
    js_demo(
        "static object folds away",
        &[FuncDef {
            name: "main",
            nslots: 2,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["input", "p"],
            body: vec![
                Let(1, obj(vec![("x", num(3)), ("y", num(4))])),
                Return(bin(Add, get(var(1), "x"), get(var(1), "y"))),
            ],
        }],
        &[0, 99],
    );

    // 2. Partially-static object: shape known, one field dynamic.
    //    main(a) = { let p = {x:a, y:10}; return p.x * p.y; }  ->  return (a * 10)
    js_demo(
        "partially-static object: {x:a, y:10}.x * .y",
        &[FuncDef {
            name: "main",
            nslots: 2,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["a", "p"],
            body: vec![
                Let(1, obj(vec![("x", var(0)), ("y", num(10))])),
                Return(bin(Mul, get(var(1), "x"), get(var(1), "y"))),
            ],
        }],
        &[2, 7],
    );

    // 3. Static array + static loop: array scalar-replaced, loop unrolled.
    //    sum([10,20,30]) -> return 60
    js_demo(
        "static array sum unrolls to a constant",
        &[FuncDef {
            name: "main",
            nslots: 4,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["input", "xs", "s", "i"],
            body: vec![
                Let(1, arr(vec![num(10), num(20), num(30)])),
                Let(2, num(0)),
                Let(3, num(0)),
                While(
                    bin(Lt, var(3), num(3)),
                    vec![
                        Set(2, bin(Add, var(2), index(var(1), var(3)))),
                        Set(3, bin(Add, var(3), num(1))),
                    ],
                ),
                Return(var(2)),
            ],
        }],
        &[0],
    );

    // 4. Closures: a captured static value inlines the higher-order call.
    //    adder(n) = (x) => x + n;  main(input) = adder(5)(input)  ->  return (input + 5)
    js_demo(
        "closure specialization: adder(5)(x) -> x + 5",
        &[
            FuncDef {
                name: "adder",
                nslots: 1,
                ncaptured: 0,
                nparams: 1,
                slot_names: vec!["n"],
                body: vec![Return(closure(1, vec![var(0)]))],
            },
            FuncDef {
                name: "addN",
                nslots: 2,
                ncaptured: 1,
                nparams: 1,
                slot_names: vec!["n", "x"],
                body: vec![Return(bin(Add, var(1), var(0)))],
            },
            FuncDef {
                name: "main",
                nslots: 2,
                ncaptured: 0,
                nparams: 1,
                slot_names: vec!["input", "add5"],
                body: vec![
                    Let(1, call(func(0), vec![num(5)])),
                    Return(call(var(1), vec![var(0)])),
                ],
            },
        ],
        &[3, 10],
    );

    // 5. THE Futamura demo: specialize an expression interpreter against a
    //    static AST (objects) with a dynamic environment. The interpreter, the
    //    AST objects, and the op dispatch all vanish.
    //    ast = (x * 2) + 1, env = {x: input}  ->  return ((input * 2) + 1)
    js_demo(
        "interpreter specialization: eval(static AST, {x:input}) -> (x*2)+1",
        &[
            FuncDef {
                name: "eval",
                nslots: 2,
                ncaptured: 0,
                nparams: 2,
                slot_names: vec!["node", "env"],
                body: vec![
                    If(
                        bin(Eq, get(var(0), "op"), str_("lit")),
                        vec![Return(get(var(0), "val"))],
                        vec![If(
                            bin(Eq, get(var(0), "op"), str_("var")),
                            vec![Return(index(var(1), get(var(0), "name")))],
                            vec![If(
                                bin(Eq, get(var(0), "op"), str_("add")),
                                vec![Return(bin(
                                    Add,
                                    call(func(0), vec![get(var(0), "l"), var(1)]),
                                    call(func(0), vec![get(var(0), "r"), var(1)]),
                                ))],
                                vec![Return(bin(
                                    Mul,
                                    call(func(0), vec![get(var(0), "l"), var(1)]),
                                    call(func(0), vec![get(var(0), "r"), var(1)]),
                                ))],
                            )],
                        )],
                    ),
                ],
            },
            FuncDef {
                name: "main",
                nslots: 1,
                ncaptured: 0,
                nparams: 1,
                slot_names: vec!["input"],
                body: vec![Return(call(
                    func(0),
                    vec![
                        // ast: (x * 2) + 1
                        obj(vec![
                            ("op", str_("add")),
                            (
                                "l",
                                obj(vec![
                                    ("op", str_("mul")),
                                    ("l", obj(vec![("op", str_("var")), ("name", str_("x"))])),
                                    ("r", obj(vec![("op", str_("lit")), ("val", num(2))])),
                                ]),
                            ),
                            ("r", obj(vec![("op", str_("lit")), ("val", num(1))])),
                        ]),
                        // env: { x: input }
                        obj(vec![("x", var(0))]),
                    ],
                ))],
            },
        ],
        &[0, 5, 21],
    );

    // 6. Object mutation through the precise abstract heap, plus a dynamic
    //    branch. The object never escapes, so it folds; the mutation is tracked.
    //    main(a) = { let p = {count:0}; p.count = a - 1;
    //                if p.count > 0 { return p.count } else { return 0 } }
    js_demo(
        "object mutation + dynamic branch",
        &[FuncDef {
            name: "main",
            nslots: 2,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["a", "p"],
            body: vec![
                Let(1, obj(vec![("count", num(0))])),
                SetProp(var(1), "count".to_string(), bin(Sub, var(0), num(1))),
                If(
                    bin(Gt, get(var(1), "count"), num(0)),
                    vec![Return(get(var(1), "count"))],
                    vec![Return(num(0))],
                ),
            ],
        }],
        &[0, 1, 9],
    );

    // 7. Dynamic loop still merges cleanly alongside the heap machinery.
    //    main(n) = sum 0..n
    js_demo(
        "dynamic loop merges: sum 0..n",
        &[FuncDef {
            name: "main",
            nslots: 3,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["n", "s", "i"],
            body: vec![
                Let(1, num(0)),
                Let(2, num(0)),
                While(
                    bin(Lt, var(2), var(0)),
                    vec![
                        Set(1, bin(Add, var(1), var(2))),
                        Set(2, bin(Add, var(2), num(1))),
                    ],
                ),
                Return(var(1)),
            ],
        }],
        &[0, 3, 5],
    );

    // 8. Object ESCAPE: the result is an object, so it cannot be scalar-replaced.
    //    It is materialized as residual construction (partial escape analysis).
    //    main(a) = return { doubled: a*2, plus1: a+1 };
    js_demo(
        "object escape: return a constructed object",
        &[FuncDef {
            name: "main",
            nslots: 1,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["a"],
            body: vec![Return(obj(vec![
                ("doubled", bin(Mul, var(0), num(2))),
                ("plus1", bin(Add, var(0), num(1))),
            ]))],
        }],
        &[5, 10],
    );

    // 9. Escape across a dynamic branch: each arm returns a different object.
    //    main(a) = if a > 0 { return {sign:1, val:a} } else { return {sign:-1, val:0-a} }
    js_demo(
        "object escape across a dynamic branch",
        &[FuncDef {
            name: "main",
            nslots: 1,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["a"],
            body: vec![If(
                bin(Gt, var(0), num(0)),
                vec![Return(obj(vec![("sign", num(1)), ("val", var(0))]))],
                vec![Return(obj(vec![
                    ("sign", num(-1)),
                    ("val", bin(Sub, num(0), var(0))),
                ]))],
            )],
        }],
        &[7, 0, -4],
    );

    // 10. Nested escape: an object containing an array and a sub-object, all
    //     materialized depth-first.
    //     main(a) = return { pair: [a, a*a], meta: {x: a} };
    js_demo(
        "nested object/array escape",
        &[FuncDef {
            name: "main",
            nslots: 1,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["a"],
            body: vec![Return(obj(vec![
                ("pair", arr(vec![var(0), bin(Mul, var(0), var(0))])),
                ("meta", obj(vec![("x", var(0))])),
            ]))],
        }],
        &[3, 6],
    );

    // 11. map written IN the subset, specialized over a static-length array with
    //     a known callback: the loop unrolls, the callback inlines, the result
    //     array escapes.  map([a, a+1, a+2], x => x*2)  ->  [a*2, (a+1)*2, (a+2)*2]
    js_demo(
        "map([a,a+1,a+2], double) unrolls + escapes",
        &[
            FuncDef {
                name: "double",
                nslots: 1,
                ncaptured: 0,
                nparams: 1,
                slot_names: vec!["x"],
                body: vec![Return(bin(Mul, var(0), num(2)))],
            },
            FuncDef {
                name: "map",
                nslots: 4,
                ncaptured: 0,
                nparams: 2,
                slot_names: vec!["xs", "f", "result", "i"],
                body: vec![
                    Let(2, arr(vec![])),
                    Let(3, num(0)),
                    While(
                        bin(Lt, var(3), get(var(0), "length")),
                        vec![
                            Push(var(2), call(var(1), vec![index(var(0), var(3))])),
                            Set(3, bin(Add, var(3), num(1))),
                        ],
                    ),
                    Return(var(2)),
                ],
            },
            FuncDef {
                name: "main",
                nslots: 1,
                ncaptured: 0,
                nparams: 1,
                slot_names: vec!["a"],
                body: vec![Return(call(
                    func(1),
                    vec![
                        arr(vec![var(0), bin(Add, var(0), num(1)), bin(Add, var(0), num(2))]),
                        func(0),
                    ],
                ))],
            },
        ],
        &[5, 8],
    );

    // 12. reduce written in the subset: unrolls to a left fold; scalar result.
    //     reduce([a,a+1,a+2], (acc,x)=>acc+x, 0)  ->  (((0+a)+(a+1))+(a+2))
    js_demo(
        "reduce([a,a+1,a+2], add, 0) -> left fold",
        &[
            FuncDef {
                name: "add",
                nslots: 2,
                ncaptured: 0,
                nparams: 2,
                slot_names: vec!["acc", "x"],
                body: vec![Return(bin(Add, var(0), var(1)))],
            },
            FuncDef {
                name: "reduce",
                nslots: 5,
                ncaptured: 0,
                nparams: 3,
                slot_names: vec!["xs", "f", "init", "acc", "i"],
                body: vec![
                    Let(3, var(2)),
                    Let(4, num(0)),
                    While(
                        bin(Lt, var(4), get(var(0), "length")),
                        vec![
                            Set(3, call(var(1), vec![var(3), index(var(0), var(4))])),
                            Set(4, bin(Add, var(4), num(1))),
                        ],
                    ),
                    Return(var(3)),
                ],
            },
            FuncDef {
                name: "main",
                nslots: 1,
                ncaptured: 0,
                nparams: 1,
                slot_names: vec!["a"],
                body: vec![Return(call(
                    func(1),
                    vec![
                        arr(vec![var(0), bin(Add, var(0), num(1)), bin(Add, var(0), num(2))]),
                        func(0),
                        num(0),
                    ],
                ))],
            },
        ],
        &[5, 100],
    );

    // 13. filter over STATIC elements: the predicate is decided at PE time, so
    //     the whole thing folds to a constant array.
    //     filter([1,2,3,4,5], x => x > 2)  ->  [3, 4, 5]
    js_demo(
        "filter([1..5], >2) folds to a constant array",
        &[
            FuncDef {
                name: "gt2",
                nslots: 1,
                ncaptured: 0,
                nparams: 1,
                slot_names: vec!["x"],
                body: vec![Return(bin(Gt, var(0), num(2)))],
            },
            FuncDef {
                name: "filter",
                nslots: 4,
                ncaptured: 0,
                nparams: 2,
                slot_names: vec!["xs", "pred", "result", "i"],
                body: vec![
                    Let(2, arr(vec![])),
                    Let(3, num(0)),
                    While(
                        bin(Lt, var(3), get(var(0), "length")),
                        vec![
                            If(
                                call(var(1), vec![index(var(0), var(3))]),
                                vec![Push(var(2), index(var(0), var(3)))],
                                vec![],
                            ),
                            Set(3, bin(Add, var(3), num(1))),
                        ],
                    ),
                    Return(var(2)),
                ],
            },
            FuncDef {
                name: "main",
                nslots: 1,
                ncaptured: 0,
                nparams: 1,
                slot_names: vec!["a"],
                body: vec![Return(call(
                    func(1),
                    vec![
                        arr(vec![num(1), num(2), num(3), num(4), num(5)]),
                        func(0),
                    ],
                ))],
            },
        ],
        &[0],
    );

    // 14. DYNAMIC-LENGTH array: built in a dynamic loop. The array's length is a
    //     runtime value, so it cannot be scalar-replaced; it escapes to a
    //     residual array variable and the pushes become residual `push` ops.
    //     main(n) = { let r = []; let i = 0; while i < n { r.push(i); i=i+1 } return r }
    js_demo(
        "dynamic-length array built in a dynamic loop: range(n)",
        &[FuncDef {
            name: "main",
            nslots: 3,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["n", "r", "i"],
            body: vec![
                Let(1, arr(vec![])),
                Let(2, num(0)),
                While(
                    bin(Lt, var(2), var(0)),
                    vec![
                        Push(var(1), var(2)),
                        Set(2, bin(Add, var(2), num(1))),
                    ],
                ),
                Return(var(1)),
            ],
        }],
        &[0, 1, 4, 7],
    );

    // 15. Same, but each pushed element is a dynamic computation (squares),
    //     showing residual expressions flow into the residual push.
    //     main(n) = { let r = []; let i = 0; while i < n { r.push(i*i); i=i+1 } return r }
    js_demo(
        "dynamic-length array of squares",
        &[FuncDef {
            name: "main",
            nslots: 3,
            ncaptured: 0,
            nparams: 1,
            slot_names: vec!["n", "r", "i"],
            body: vec![
                Let(1, arr(vec![])),
                Let(2, num(0)),
                While(
                    bin(Lt, var(2), var(0)),
                    vec![
                        Push(var(1), bin(Mul, var(2), var(2))),
                        Set(2, bin(Add, var(2), num(1))),
                    ],
                ),
                Return(var(1)),
            ],
        }],
        &[0, 5],
    );

    // 16. filter over DYNAMIC elements: the predicate is runtime, so each push
    //     is conditional. The result array escapes BEFORE the branch, the pushes
    //     residualize, and the arms merge -> a linear residual, not 2^n paths.
    //     main(a) = filter([a, a-1, a+2, a-3], x => x > 0)
    js_demo(
        "filter over dynamic elements (conditional residual push)",
        &[
            FuncDef {
                name: "ispos",
                nslots: 1,
                ncaptured: 0,
                nparams: 1,
                slot_names: vec!["x"],
                body: vec![Return(bin(Gt, var(0), num(0)))],
            },
            FuncDef {
                name: "filter",
                nslots: 4,
                ncaptured: 0,
                nparams: 2,
                slot_names: vec!["xs", "pred", "result", "i"],
                body: vec![
                    Let(2, arr(vec![])),
                    Let(3, num(0)),
                    While(
                        bin(Lt, var(3), get(var(0), "length")),
                        vec![
                            If(
                                call(var(1), vec![index(var(0), var(3))]),
                                vec![Push(var(2), index(var(0), var(3)))],
                                vec![],
                            ),
                            Set(3, bin(Add, var(3), num(1))),
                        ],
                    ),
                    Return(var(2)),
                ],
            },
            FuncDef {
                name: "main",
                nslots: 1,
                ncaptured: 0,
                nparams: 1,
                slot_names: vec!["a"],
                body: vec![Return(call(
                    func(1),
                    vec![
                        arr(vec![
                            var(0),
                            bin(Sub, var(0), num(1)),
                            bin(Add, var(0), num(2)),
                            bin(Sub, var(0), num(3)),
                        ]),
                        func(0),
                    ],
                ))],
            },
        ],
        &[5, 2, 0, -1],
    );
}

fn fun_demo(name: &str, program: &[fun::FunDef], inputs: &[i64]) {
    use partial::fun::Fun;
    println!("================================================================");
    println!("  FUN (interpreter specialization): {name}");
    println!("================================================================");

    let vm = Fun::new(program);
    let prog = vm.specialize_program();

    println!(
        "residual functions : {}\nresidual blocks    : {}\nresidual ops       : {}\n",
        prog.funcs.len(),
        prog.block_count(),
        prog.op_count()
    );
    print!("{}", vm.dump(&prog));

    for &inp in inputs {
        let reference = vm.run_reference(inp);
        let got = vm.run_residual(&prog, inp);
        assert_eq!(reference, got, "residual diverged on input {inp}");
        println!("  input {inp} -> {got:?}  (matches oracle)");
    }
    println!();
}

fn fun_demos() {
    use partial::fun::Stmt::*;
    use partial::fun::{bin, call, int, var, FunDef};
    use partial::imp::BinOp::*;

    // 1. Non-recursive calls inline: the VM dispatch loop AND the call machinery
    //    disappear, leaving straight-line residual arithmetic over the input.
    //    square(n) = n*n ; main(x) = print square(x); print square(x+1)
    fun_demo(
        "inline non-recursive calls: square",
        &[
            FunDef {
                name: "square",
                nslots: 1,
                slot_names: vec!["n"],
                body: vec![Return(bin(Mul, var(0), var(0)))],
            },
            FunDef {
                name: "main",
                nslots: 1,
                slot_names: vec!["x"],
                body: vec![
                    Print(call(0, vec![var(0)])),
                    Print(call(0, vec![bin(Add, var(0), int(1))])),
                ],
            },
        ],
        &[0, 3, 10],
    );

    // 2. THE classic: specialize the interpreter to power(x, 5). The recursion
    //    is driven by the STATIC exponent, so it unrolls completely: x^5 as
    //    straight-line multiplies. No loop, no call, no dispatch in the residual.
    //    power(b,e) = if e==0 then 1 else b * power(b, e-1)
    fun_demo(
        "specialize recursion on static arg: power(x, 5) -> x^5",
        &[
            FunDef {
                name: "power",
                nslots: 2,
                slot_names: vec!["b", "e"],
                body: vec![If(
                    bin(Eq, var(1), int(0)),
                    vec![Return(int(1))],
                    vec![Return(bin(
                        Mul,
                        var(0),
                        call(0, vec![var(0), bin(Sub, var(1), int(1))]),
                    ))],
                )],
            },
            FunDef {
                name: "main",
                nslots: 1,
                slot_names: vec!["x"],
                body: vec![Print(call(0, vec![var(0), int(5)]))],
            },
        ],
        &[2, 3, 5],
    );

    // 3. Dynamic loop with an inlined call: the call inlines, the loop stays.
    //    The loop-modified variables merge at the header, so the residual is the
    //    textbook compiled loop with `add` inlined to `s := s + i`.
    //    add(a,b) = a+b
    //    main(n) = s=0; i=0; while i<n { s = add(s,i); i=i+1 }; print s
    fun_demo(
        "dynamic loop + inlined call: sum 0..n via add()",
        &[
            FunDef {
                name: "add",
                nslots: 2,
                slot_names: vec!["a", "b"],
                body: vec![Return(bin(Add, var(0), var(1)))],
            },
            FunDef {
                name: "main",
                nslots: 3,
                slot_names: vec!["n", "s", "i"],
                body: vec![
                    Set(1, int(0)), // s = 0
                    Set(2, int(0)), // i = 0
                    While(
                        bin(Lt, var(2), var(0)),
                        vec![
                            Set(1, call(0, vec![var(1), var(2)])), // s = add(s, i)
                            Set(2, bin(Add, var(2), int(1))),      // i = i + 1
                        ],
                    ),
                    Print(var(1)),
                ],
            },
        ],
        &[0, 1, 3, 5],
    );

    // 4. Dynamic-DEPTH recursion: fib(x) for dynamic x cannot be inlined. It is
    //    cut into a residual function that calls itself. The interpreter is gone,
    //    but the object program's recursion survives as residual recursion.
    //    fib(n) = if n<2 then n else fib(n-1) + fib(n-2)
    fun_demo(
        "dynamic-depth recursion -> residual function: fib(x)",
        &[
            FunDef {
                name: "fib",
                nslots: 1,
                slot_names: vec!["n"],
                body: vec![If(
                    bin(Lt, var(0), int(2)),
                    vec![Return(var(0))],
                    vec![Return(bin(
                        Add,
                        call(0, vec![bin(Sub, var(0), int(1))]),
                        call(0, vec![bin(Sub, var(0), int(2))]),
                    ))],
                )],
            },
            FunDef {
                name: "main",
                nslots: 1,
                slot_names: vec!["x"],
                body: vec![Print(call(0, vec![var(0)]))],
            },
        ],
        &[0, 1, 5, 10, 15],
    );

    // 5. Dynamic-IF join merging: both arms assign `y`, then a shared
    //    continuation uses it. Without merging, the continuation duplicates into
    //    both arms; with it, `y` is materialized at the join and the
    //    continuation is a single block.
    //    main(x) = y=0; if x<10 { y=x*2 } else { y=x+100 }; print y; print y+1; print y+2
    fun_demo(
        "dynamic-if join merging: shared continuation stays single",
        &[FunDef {
            name: "main",
            nslots: 2,
            slot_names: vec!["x", "y"],
            body: vec![
                Set(1, int(0)),
                If(
                    bin(Lt, var(0), int(10)),
                    vec![Set(1, bin(Mul, var(0), int(2)))],
                    vec![Set(1, bin(Add, var(0), int(100)))],
                ),
                Print(var(1)),
                Print(bin(Add, var(1), int(1))),
                Print(bin(Add, var(1), int(2))),
            ],
        }],
        &[3, 9, 10, 25],
    );
}

fn imp_demo(name: &str, names: &[&str], body: &[imp::Stmt], inputs: &[&[i64]]) {
    println!("================================================================");
    println!("  IMP: {name}");
    println!("================================================================");

    let prog = imp::Imp::new(names, body);
    let mut residual = engine::specialize(&prog, prog.start());
    residual::simplify(&mut residual);

    println!(
        "residual blocks : {}\nresidual ops    : {}\n",
        residual.blocks.len(),
        residual.op_count()
    );
    print!("{}", prog.dump(&residual));

    // Verify against the reference interpreter across several inputs.
    for inp in inputs {
        let reference = prog.run_reference(inp);
        let got = prog.run_residual(&residual, inp);
        assert_eq!(
            reference, got,
            "residual diverged from reference on input {inp:?}"
        );
        println!("  input {inp:?} -> {got:?}  (matches oracle)");
    }
    println!();
}

fn imp_demos() {
    // Variable indices (per-demo name arrays give them meaning).
    const X: usize = 0;
    const Y: usize = 1;
    const N: usize = 2;
    const I: usize = 3;

    // 1. Everything static: a whole expression folds to a constant print.
    imp_demo(
        "full static fold: print 2*3+1",
        &["x", "y"],
        &[
            Assign(X, int(2)),
            Assign(Y, int(3)),
            Print(bin(Add, bin(Mul, var(X), var(Y)), int(1))),
        ],
        &[&[]],
    );

    // 2. STATIC loop: sum 0..3 unrolls completely; no residual loop survives,
    //    it collapses to `print 3`.
    imp_demo(
        "static loop unrolls to a constant: sum i in 0..3",
        &["i", "s"],
        &[
            Assign(0, int(0)), // i
            Assign(1, int(0)), // s
            While(
                bin(Lt, var(0), int(3)),
                vec![
                    Assign(1, bin(Add, var(1), var(0))),
                    Assign(0, bin(Add, var(0), int(1))),
                ],
            ),
            Print(var(1)),
        ],
        &[&[]],
    );

    // 3. DYNAMIC loop: the bound is runtime input, so the engine cannot unroll.
    //    The loop variable `i` is loop-modified under a dynamic condition, so it
    //    is materialized to dynamic at the header; every path into the header
    //    merges and the loop residualizes as one clean block (no peeling).
    imp_demo(
        "dynamic loop merges at the header: print 0..n",
        &["x", "y", "n", "i"],
        &[
            Input(N),
            Assign(I, int(0)),
            While(
                bin(Lt, var(I), var(N)),
                vec![Print(var(I)), Assign(I, bin(Add, var(I), int(1)))],
            ),
        ],
        &[&[0], &[1], &[3], &[5]],
    );

    // 4. Dynamic branch with static arms: only the live computation residualizes.
    imp_demo(
        "dynamic branch: sign of x",
        &["x"],
        &[
            Input(X),
            If(
                bin(Lt, var(X), int(0)),
                vec![Print(int(-1))],
                vec![Print(int(1))],
            ),
        ],
        &[&[-7], &[0], &[42]],
    );

    // 5. Dynamic-if join merging: both arms assign `y` (here, abs(x)), then a
    //    shared continuation uses it. The arms merge at the join, so the
    //    continuation is a single block instead of being duplicated.
    imp_demo(
        "dynamic-if join merging: abs then shared continuation",
        &["x", "y"],
        &[
            Input(X),
            If(
                bin(Lt, var(X), int(0)),
                vec![Assign(1, bin(Sub, int(0), var(X)))],
                vec![Assign(1, var(X))],
            ),
            Print(var(1)),
            Print(bin(Add, var(1), int(100))),
        ],
        &[&[-7], &[0], &[42]],
    );
}
