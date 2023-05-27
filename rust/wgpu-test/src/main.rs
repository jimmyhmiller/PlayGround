mod core;
use crate::core::run;
fn main() {
    pollster::block_on(run());
}