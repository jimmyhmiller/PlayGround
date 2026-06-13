fn run() {
    let total = 0;
    for i in 0..2000 {
        for j in 0..2000 {
            total += (i * j) % 7;
        }
    }
    total
}
print(run());
