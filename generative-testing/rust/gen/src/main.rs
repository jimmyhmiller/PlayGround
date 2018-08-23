use quickcheck::Arbitrary;

use quickcheck::Gen;

#[macro_use]
extern crate quickcheck;

#[cfg(test)]
mod tests {
    use Point;
    use main;

    quickcheck! {
        fn prop(xs: Vec<u32>) -> bool {
            xs == main::reverse(&main::reverse(&xs))
        }
    }

    quickcheck! {
        fn is_pos(point : Point) -> bool {
            point.x > 0 && point.y > 0
        }
    }
}

#[derive(Debug)]
#[derive(Clone)]
pub struct Point {
    x: i32,
    y: i32
}

impl Arbitrary for Point {
    fn arbitrary<G: Gen>(g: &mut G) -> Point {
        let x : i32 = i32::arbitrary(g);
        let y : i32 = i32::arbitrary(g);
        Point { x, y } 
    }
}

pub mod main {

    use Point;

    pub fn pos_point(point : Point) -> Point {
        Point { 
            x: point.x.abs(),
            y: point.y.abs()
        }
    }


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
