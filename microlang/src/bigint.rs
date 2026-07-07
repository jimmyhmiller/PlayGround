//! A small, dependency-free arbitrary-precision integer — enough to carry the
//! numeric tower past `i128` without pulling in a crate. Sign-magnitude, with
//! the magnitude stored little-endian in base 10^9 so decimal printing is a
//! trivial digit join. Schoolbook add/sub/mul (O(n^2)); fine for the sizes a
//! toy Scheme reaches. Only reached when the `i128` fast path overflows, so it
//! is never on the hot arithmetic path.

use std::cmp::Ordering;

const BASE: u64 = 1_000_000_000;
const BASE_DIGITS: usize = 9;

/// Sign-magnitude big integer. `mag` is little-endian base-10^9 with no trailing
/// zero limbs; zero is represented as an empty `mag` with `neg == false` (so
/// there is exactly one representation of every value).
#[derive(Clone, Debug)]
pub struct BigInt {
    neg: bool,
    mag: Vec<u32>,
}

impl BigInt {
    pub fn from_i128(x: i128) -> Self {
        let neg = x < 0;
        let mut u = x.unsigned_abs(); // handles i128::MIN without overflow
        let mut mag = Vec::new();
        while u > 0 {
            mag.push((u % BASE as u128) as u32);
            u /= BASE as u128;
        }
        BigInt { neg, mag }.normalized()
    }

    /// The exact value as `i128`, or `None` if it does not fit. Accumulates with
    /// the sign applied so `i128::MIN` (whose magnitude is not a valid positive
    /// `i128`) round-trips.
    pub fn to_i128(&self) -> Option<i128> {
        let mut acc: i128 = 0;
        for &limb in self.mag.iter().rev() {
            acc = acc.checked_mul(BASE as i128)?;
            acc = if self.neg {
                acc.checked_sub(limb as i128)?
            } else {
                acc.checked_add(limb as i128)?
            };
        }
        Some(acc)
    }

    /// An approximate `f64` (for mixed int/float arithmetic and comparison).
    pub fn to_f64(&self) -> f64 {
        let mut acc = 0.0f64;
        for &limb in self.mag.iter().rev() {
            acc = acc * BASE as f64 + limb as f64;
        }
        if self.neg {
            -acc
        } else {
            acc
        }
    }

    fn is_zero(&self) -> bool {
        self.mag.is_empty()
    }

    fn normalized(mut self) -> Self {
        while self.mag.last() == Some(&0) {
            self.mag.pop();
        }
        if self.mag.is_empty() {
            self.neg = false; // canonical zero
        }
        self
    }

    pub fn neg(&self) -> BigInt {
        if self.is_zero() {
            self.clone()
        } else {
            BigInt { neg: !self.neg, mag: self.mag.clone() }
        }
    }

    pub fn add(&self, other: &BigInt) -> BigInt {
        if self.neg == other.neg {
            BigInt { neg: self.neg, mag: mag_add(&self.mag, &other.mag) }.normalized()
        } else {
            // Different signs: subtract the smaller magnitude from the larger.
            match mag_cmp(&self.mag, &other.mag) {
                Ordering::Equal => BigInt { neg: false, mag: Vec::new() },
                Ordering::Greater => {
                    BigInt { neg: self.neg, mag: mag_sub(&self.mag, &other.mag) }.normalized()
                }
                Ordering::Less => {
                    BigInt { neg: other.neg, mag: mag_sub(&other.mag, &self.mag) }.normalized()
                }
            }
        }
    }

    pub fn sub(&self, other: &BigInt) -> BigInt {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &BigInt) -> BigInt {
        BigInt {
            neg: self.neg != other.neg,
            mag: mag_mul(&self.mag, &other.mag),
        }
        .normalized()
    }

    /// Signed comparison.
    pub fn cmp(&self, other: &BigInt) -> Ordering {
        match (self.neg, other.neg) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => mag_cmp(&self.mag, &other.mag),
            (true, true) => mag_cmp(&other.mag, &self.mag), // both negative: reverse
        }
    }

    pub fn to_string(&self) -> String {
        if self.mag.is_empty() {
            return "0".to_string();
        }
        let mut s = String::new();
        if self.neg {
            s.push('-');
        }
        // Most-significant limb without padding, the rest zero-padded to 9 digits.
        let (last, rest) = self.mag.split_last().unwrap();
        s.push_str(&last.to_string());
        for limb in rest.iter().rev() {
            s.push_str(&format!("{limb:0width$}", width = BASE_DIGITS));
        }
        s
    }
}

impl PartialEq for BigInt {
    fn eq(&self, other: &Self) -> bool {
        self.neg == other.neg && self.mag == other.mag
    }
}
impl Eq for BigInt {}

fn mag_cmp(a: &[u32], b: &[u32]) -> Ordering {
    if a.len() != b.len() {
        return a.len().cmp(&b.len());
    }
    for (x, y) in a.iter().rev().zip(b.iter().rev()) {
        match x.cmp(y) {
            Ordering::Equal => continue,
            ord => return ord,
        }
    }
    Ordering::Equal
}

fn mag_add(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len().max(b.len()) + 1);
    let mut carry = 0u64;
    for i in 0..a.len().max(b.len()) {
        let x = *a.get(i).unwrap_or(&0) as u64;
        let y = *b.get(i).unwrap_or(&0) as u64;
        let s = x + y + carry;
        out.push((s % BASE) as u32);
        carry = s / BASE;
    }
    if carry > 0 {
        out.push(carry as u32);
    }
    out
}

/// Precondition: `a >= b` (by magnitude).
fn mag_sub(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len());
    let mut borrow = 0i64;
    for i in 0..a.len() {
        let x = a[i] as i64;
        let y = *b.get(i).unwrap_or(&0) as i64;
        let mut d = x - y - borrow;
        if d < 0 {
            d += BASE as i64;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out.push(d as u32);
    }
    out
}

fn mag_mul(a: &[u32], b: &[u32]) -> Vec<u32> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0u64; a.len() + b.len()];
    for (i, &x) in a.iter().enumerate() {
        let mut carry = 0u64;
        for (j, &y) in b.iter().enumerate() {
            let cur = out[i + j] + x as u64 * y as u64 + carry;
            out[i + j] = cur % BASE;
            carry = cur / BASE;
        }
        out[i + b.len()] += carry;
    }
    out.iter().map(|&limb| limb as u32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big(s: &str) -> BigInt {
        // Small helper: parse a decimal string (handles the values we test).
        let neg = s.starts_with('-');
        let digits = s.trim_start_matches('-');
        let mut acc = BigInt::from_i128(0);
        let ten = BigInt::from_i128(10);
        for ch in digits.chars() {
            acc = acc.mul(&ten).add(&BigInt::from_i128((ch as u8 - b'0') as i128));
        }
        if neg {
            acc.neg()
        } else {
            acc
        }
    }

    #[test]
    fn roundtrip_and_print() {
        for v in [0i128, 1, -1, 42, -42, i128::MAX, i128::MIN, 1_000_000_000, 999_999_999] {
            let b = BigInt::from_i128(v);
            assert_eq!(b.to_i128(), Some(v), "roundtrip {v}");
            assert_eq!(b.to_string(), v.to_string(), "print {v}");
        }
    }

    #[test]
    fn mul_beyond_i128() {
        // 10^20 * 10^20 = 10^40, well beyond i128 (~1.7e38).
        let r = big("100000000000000000000").mul(&big("100000000000000000000"));
        assert_eq!(r.to_string(), "10000000000000000000000000000000000000000");
        assert_eq!(r.to_i128(), None, "10^40 must not fit i128");
    }

    #[test]
    fn signed_add_sub() {
        assert_eq!(big("-5").add(&big("3")).to_string(), "-2");
        assert_eq!(big("5").add(&big("-3")).to_string(), "2");
        assert_eq!(big("3").sub(&big("5")).to_string(), "-2");
        assert_eq!(big("5").sub(&big("5")).to_string(), "0");
        assert_eq!(big("-5").mul(&big("-3")).to_string(), "15");
        assert_eq!(big("-5").mul(&big("3")).to_string(), "-15");
        // carry/borrow across limb boundaries
        assert_eq!(big("1000000000").sub(&big("1")).to_string(), "999999999");
        assert_eq!(big("999999999").add(&big("1")).to_string(), "1000000000");
    }

    #[test]
    fn ordering() {
        use Ordering::*;
        assert_eq!(big("-1").cmp(&big("1")), Less);
        assert_eq!(big("1").cmp(&big("-1")), Greater);
        assert_eq!(big("-5").cmp(&big("-3")), Less);
        assert_eq!(big("100").cmp(&big("100")), Equal);
        assert_eq!(
            big("10000000000000000000000000000000000000000")
                .cmp(&big("9999999999999999999999999999999999999999")),
            Greater
        );
    }
}
