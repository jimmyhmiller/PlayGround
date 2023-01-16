#![allow(dead_code)]
use rand::Rng;

#[derive(Debug)]
enum Value {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Null,
    Pair(Box<Value>, Box<Value>),
}

#[derive(Debug)]
struct Item {
    key: String,
    value: Value,
}

#[derive(Debug)]
struct Database {
    data: Vec<Item>,
}

impl Database {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn insert_sorted(&mut self, key: String, value: Value) {
        let index = self.data.binary_search_by(|item| item.key.cmp(&key));
        match index {
            // This would update instead of keep multiples. Not sure I want that.
            Ok(index) => self.data[index] = Item { key, value },
            Err(index) => self.data.insert(index, Item { key, value }),
        }
    }

    fn get(&self, key: String) -> Option<&Value> {
        self.data
            .binary_search_by_key(&key, |item| item.key.clone())
            .ok()
            .map(|index| &self.data[index].value)
    }
}

pub fn main() {
    let mut db = Database::new();
    // how to generate random integer in rust
    // https://stackoverflow.com/questions/27769784/how-to-generate-random-integer-in-rust
    let mut rng = rand::thread_rng();

    for _ in 0..10000 {
        let key = format!("id {}", rng.gen_range(1..100000));
        let num = rng.gen::<i64>();
        db.insert_sorted(key, Value::Integer(num));
    }

    println!("{:#?}", db.data);
}
