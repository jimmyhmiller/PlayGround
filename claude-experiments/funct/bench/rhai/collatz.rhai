fn steps(n) {
    let x = n;
    let c = 0;
    while x > 1 {
        if x % 2 == 0 { x /= 2; } else { x = 3 * x + 1; }
        c += 1;
    }
    c
}
fn run() {
    let total = 0;
    for n in 1..=100000 {
        total += steps(n);
    }
    total
}
print(run());
