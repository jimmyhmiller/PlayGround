// Higher-order function: make-adder pattern
let makeAdder = (n) => (x) => x + n;
let add5 = makeAdder(5);
add5(input)
