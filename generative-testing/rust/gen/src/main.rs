#[macro_use]
extern crate quickcheck;

#[cfg(test)]
mod tests {
    use main;
    quickcheck! {
        fn prop(xs: Vec<u32>) -> bool {
            xs == main::reverse(&main::reverse(&xs))
        }
    }
}


pub mod main {

    pub fn reverse(xs: &[u32]) -> Vec<u32> {
        let mut rev = vec!();
        for x in xs.iter() {
            // if *x != 42 {
                rev.insert(0, x.clone())
            // }
        }
        rev
    }
}

fn main() {
    println!("Hello, world!");
}
