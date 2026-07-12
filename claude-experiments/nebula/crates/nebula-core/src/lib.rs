//! `nebula-core` — graph representation, generators, algorithms, and I/O.
//!
//! Everything downstream (layout, rendering) is built on the types here. The
//! design bias throughout is *scale*: structure-of-arrays, u32 node ids, u64
//! edge counts, parallel construction, and formats that upload to the GPU with
//! zero reshaping.

pub mod algorithms;
pub mod generate;
pub mod graph;
pub mod io;

pub use graph::{Csr, Graph, NodeId};

/// A 2D position, laid out for direct GPU upload (`vec2<f32>`).
pub type Pos = [f32; 2];
