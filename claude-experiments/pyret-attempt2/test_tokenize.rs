use pyret_attempt2::tokenizer::Tokenizer;

fn main() {
    let inputs = vec!["f(x)", "f (x)", "(42)", "f(x, y)"];
    
    for input in inputs {
        println!("\n--- Testing: '{}' ---", input);
        let mut tokenizer = Tokenizer::new(input);
        let tokens = tokenizer.tokenize();
        
        for token in &tokens {
            println!("{:?}: {:?}", token.token_type, token.value);
        }
    }
}
