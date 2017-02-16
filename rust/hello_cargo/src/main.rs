// use std::env;
#![feature(step_by)]
mod euler;

#[allow(dead_code)]
fn euler73() {
    let fracs = euler::farey(12000);
    let third = fracs.iter().position(|&x| x == (1,3));
    let half = fracs.iter().position(|&x| x == (1,2));
    println!("{:?}", half.unwrap() - third.unwrap() - 1);
}



fn main() {
    println!("{:?}", 2);
    
}
