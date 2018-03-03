fn main() {
	let print = &Print {};
	let eval = &Eval {};

	println!("{}", make_expr(print));
	println!("{}", make_expr(eval))
}

trait ExpAlg<E> {
	fn lit(&self, n: i32) -> E;
	fn add(&self, l: E, r: E) -> E;
}

#[derive(Debug)]
struct Eval {}

impl ExpAlg<i32> for Eval {
	fn lit(&self, n: i32) -> i32 {
		return n;
	}

	fn add(&self, l: i32, r: i32) -> i32 {
		return l + r;
	}
}

#[derive(Debug)]
struct Print {}

impl ExpAlg<String> for Print {
	fn lit(&self, n: i32) -> String {
		return n.to_string();
	}

	fn add(&self, l: String, r: String) -> String {
		return l + " + " + &r;
	}
}

trait MulAlg<E>: ExpAlg<E> {
    fn mul(&self, l: E, r: E) -> E;
}


impl MulAlg<i32> for Eval {
	fn mul(&self, l: i32, r :i32) -> i32 {
		return l * r;
	}
}

impl MulAlg<String> for Print {
	fn mul(&self, l: String, r: String) -> String {
		return l + " * " + &r;
	}
}

fn make_expr<E>(alg: &MulAlg<E>) -> E {
	return alg.mul(alg.lit(3), alg.lit(4));
}