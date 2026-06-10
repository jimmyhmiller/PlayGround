fn classify(mut n: i32) -> i32 {
    let mut acc = 0;

    while n > 0 {
        if n % 3 == 0 {
            acc += n * 2;
        } else if n % 2 == 0 {
            acc -= n;
        } else {
            acc += 1;
        }
        n -= 1;
    }

    acc
}

fn main() {
    println!("{}", classify(8));
}
