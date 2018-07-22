#[macro_use]
extern crate serde_json;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate rand;
extern crate rayon;
extern crate itertools;

use std::sync::atomic::{AtomicUsize, Ordering, ATOMIC_USIZE_INIT};

use std::io::Write;
use std::io::BufRead;
use rand::prelude::*;
use std::iter;
use std::fs::File;
use std::io::BufWriter;
use std::io::BufReader;
use rayon::prelude::*;

// Reimplementation of this blog post.
// https://tech.grammarly.com/blog/building-etl-pipelines-with-clojure
// Benchmark shows an order of magnitude speed increase.


#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
#[derive(Debug)]
#[serde(rename_all = "lowercase")]
enum JsonObject {
    String { string: String},
    Number { number: serde_json::Number },
    Empty {},
}


fn rand_obj() -> serde_json::Value {

    let mut rng = thread_rng();

    match rng.gen_range(0,3) {
        0 => json!({"type": "number", "number": rng.gen_range(0,1000)}),
        1 => json!({"type": "string", 
                    "string": (iter::repeat_with(|| rng.gen_range(b'A', b'z') as char)
                                    .take(30)
                                    .collect::<String>())}),
        2 => json!({"type": "empty"}),
        _ => panic!("{:?}", "Random out of range"),
    }
}

fn valid_entry(log_entry : & JsonObject) -> bool {
    match log_entry {
        JsonObject::String {string: _} => true,
        JsonObject::Number {number: _} => true,
        JsonObject::Empty {} => false,
    }
}

static DB: AtomicUsize = ATOMIC_USIZE_INIT;


fn transform_entry_if_relavant(log_entry : JsonObject) -> Option<JsonObject> {
    match log_entry {
        JsonObject::Number {number: n} => if n.as_i64().unwrap() > 900 {
          (n.as_f64()
            .map(|x| x.ln())
            .and_then(|x| serde_json::Number::from_f64(x))
            .map(|x| JsonObject::Number {number : x}))
        } else { None },
        JsonObject::String {string: s} => if s.contains("a") {
            Some(JsonObject::String { string: s + "-improved"})
        } else { None },
        JsonObject::Empty {} => None,
    }
}

fn read_json(path : &str) -> impl ParallelIterator<Item=JsonObject> {
    let f = File::open(path).expect("Failed");
    let reader = BufReader::new(f);

    let lines = reader.lines()
        .filter_map(Result::ok)
        .map(|s| serde_json::from_str(&s))
        .filter_map(Result::ok)
        .par_bridge();

    lines
}

fn process(files : Vec<String>) -> () {
    files.into_par_iter()
        .flat_map(|s| read_json(&s))
        .filter(valid_entry)
        .filter_map(transform_entry_if_relavant)
        .for_each(|_c| {
            DB.fetch_add(1, Ordering::SeqCst);
     });
}

fn write_dummy_file () {
    let f = File::create("/tmp/dummy-small.json").expect("Unable to create file");
    let mut f = BufWriter::new(f);

    for _i in 0..1000000 {
        f.write_all((rand_obj().to_string() + "\n").as_bytes()).expect("Unable to write data");
    }
}

fn main() {

    // write_dummy_file();
    let _processed = process(
        iter::repeat("/tmp/dummy-small.json".to_string()).take(100).collect()
    );

    
    println!("{:?}", DB.fetch_add(0, Ordering::SeqCst));
}
