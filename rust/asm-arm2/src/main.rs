use mmap_rs::MmapOptions;
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::{self, File},
    io::Write,
    mem,
    process::Command,
    str::from_utf8,
};

use roxmltree::{Document, Node};
use serde::{Deserialize, Serialize};

use arm::{Asm, Register};

use crate::arm::{Size, X0, X1};

mod arm;


// TODO:
// ORGANIZE!!!!


// We have a nice enum format
// but we need a none enum format

// We need better documentation on which
// instructions we actually like and care about.
// We need to abstract over all the different "add" and "mov"
// functions so I don't look dumb on stream

// We need to build a real build system

// Start writing some code for the real language
