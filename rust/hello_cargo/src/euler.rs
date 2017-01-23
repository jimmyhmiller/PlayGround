
use std::vec::Vec;

// 

pub fn farey(n: usize) -> Vec<(usize, usize)> {
    let mut elems = Vec::new();
    let mut a = 0;
    let mut b = 1;
    let mut c = 1;
    let mut d = n;
    while c <= n {
        let k = (n + b) / d;
        let p = k * c - a;
        let q = k * d - b;
        elems.push((c, d));
        a = c;
        b = d;
        c = p;
        d = q;
    }
    return elems;
}


#[allow(dead_code)]
pub fn list_of_primes(n: usize) -> Vec<usize> {
    let mut sieve = vec![true; n/2];
    let sqrt = (n as f64).sqrt() as usize;
    for i in (3..sqrt + 1).step_by(2) {
        if sieve[i/2] == true {
            for j in (i*i/2..n/2).step_by(i) {
                sieve[j] = false;
            }
        }
    }
    let mut answer = Vec::new();
    answer.push(2);
    for (i, item) in sieve.iter().enumerate() {
        if *item == true && i*2 >= 2 {
            answer.push(i*2+1);
        }
    }
    return answer;  
}



#[allow(dead_code)]
fn digits(mut n: u32) -> Vec<u32> {
    let mut digits = Vec::new();
    while n != 0 {
        let a = n%10;
        n /= 10;
        digits.push(a);
    }
    return digits;
}

#[allow(dead_code)]
fn bouncy(x: u32) -> bool {
    let digs = digits(x);
    let mut up = false;
    let mut down = false;
    for i in 1..digs.len() {
        if digs.get(i-1) < digs.get(i) {
            down = true;
        } else if digs.get(i-1) > digs.get(i) {
            up = true;
        }
        if up && down {
            return true;
        }
    }
    return false;
}

#[allow(dead_code)]
pub fn check_bouncy() {
   let mut total : f64 = 0.0;
   for i in 1..10000000 {
        if bouncy(i) {
            total += 1.0;
        }
        if total/i as f64 >= 0.99 {
            print!("{:?} {:?}", i, total);
            break;
        }
   }
}
