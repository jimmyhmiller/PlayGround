fn main() {
    println!("Welcome to the echo program!");
    loop {
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        print!("{}", input);
    }
}
