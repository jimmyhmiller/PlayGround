fn main() {
    // Simulate what maxX might be
    let max_x_a: f64 = 437.5 + 20.0;  // Path endpoint + CONTENT_PADDING
    let max_x_b: f64 = 441.0;  // Some other calculation
    
    println!("max_x (path + padding): {}", max_x_a);
    println!("As i32: {}", max_x_a as i32);
    println!("With +20: {}", (max_x_a + 20.0) as i32);
    println!("Ceiling: {}", max_x_a.ceil() as i32);
    
    println!("\nmax_x (other): {}", max_x_b);
    println!("As i32: {}", max_x_b as i32);
    println!("With +20: {}", (max_x_b + 20.0) as i32);
    println!("Ceiling: {}", max_x_b.ceil() as i32);
    println!("Ceiling +20: {}", (max_x_b.ceil() + 20.0) as i32);
}
