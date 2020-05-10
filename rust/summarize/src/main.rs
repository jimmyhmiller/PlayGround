use std::collections::HashMap;
use std::time::{Instant};

fn summarize(nums : &Vec<HashMap<String, String>>, cols : & Vec<String>) -> HashMap<String, f64> {
    let mut total : HashMap<String, f64> = HashMap::new();
    for row in nums {
        for col in cols {
            if let Some(val) = total.get_mut(col) {
                *val += row[col].parse::<f64>().unwrap();
            } else {
                total.insert(col.clone(), row[col].parse().unwrap());
            }
        }
    }
    total
}

fn main() {
    let mut example = HashMap::new();
    example.insert("a".to_string(), "20".to_string());
    example.insert("b".to_string(), "10".to_string());
    example.insert("c".to_string(), "13.0".to_string());
    example.insert("d".to_string(), "10".to_string());
    example.insert("f".to_string(), "9.0".to_string());

    let examples : Vec<_> = std::iter::repeat(example).take(1_000_000).collect();

    let now = Instant::now();

    println!("{:?}", summarize(&examples, &vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string(), "f".to_string()]));
    println!("{}", now.elapsed().as_millis());

}
