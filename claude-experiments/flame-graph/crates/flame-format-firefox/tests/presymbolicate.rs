//! The `.syms.json` sidecar path, against a real
//! `samply record --unstable-presymbolicate` pair under `samples/`.
//!
//! The recorded program (`inl`) is a Rust binary whose hot loop lives in
//! `inl::outer_hot`, called from `inl::main`, with `middle_hot`/`inner_hot`
//! inlined into it.

use std::collections::HashSet;

use flame_core::{Profile, ProfileBuilder, TraceSource};
use flame_format_firefox::{FirefoxSource, PrecogSymbols};

const PROFILE: &str = "../../samples/samply-presym.json";
const SIDECAR: &str = "../../samples/samply-presym.syms.json";

fn load_plain() -> Profile {
    let bytes = std::fs::read(PROFILE).unwrap();
    let mut b = ProfileBuilder::new();
    FirefoxSource.load(&bytes, &mut b).unwrap();
    b.finish()
}

fn load_symbolicated() -> Profile {
    let bytes = std::fs::read(PROFILE).unwrap();
    let syms = PrecogSymbols::parse(&std::fs::read(SIDECAR).unwrap()).unwrap();
    let mut b = ProfileBuilder::new();
    FirefoxSource
        .load_with_symbols(&bytes, Some(&syms), &mut b)
        .unwrap();
    b.finish()
}

fn slice_names(p: &Profile) -> HashSet<String> {
    p.slices
        .name
        .iter()
        .map(|&id| p.strings.get(id).to_string())
        .collect()
}

#[test]
fn without_the_sidecar_frames_are_bare_addresses() {
    let names = slice_names(&load_plain());
    assert!(
        names.iter().all(|n| n.starts_with("0x")),
        "expected only hex names, got {names:?}"
    );
    assert!(
        !names.iter().any(|n| n.contains("outer_hot")),
        "the profile itself should carry no symbols"
    );
}

#[test]
fn the_sidecar_names_the_frames() {
    let names = slice_names(&load_symbolicated());
    for expected in ["inl::main", "inl::outer_hot", "main"] {
        assert!(names.contains(expected), "missing {expected} in {names:?}");
    }
    assert!(
        !names.iter().any(|n| n.starts_with("0x")),
        "every frame in this profile has a lib, so none should stay hex: {names:?}"
    );
}

#[test]
fn inlined_calls_become_their_own_frames() {
    let p = load_symbolicated();
    let names = slice_names(&p);
    // Address 0x1a23f resolves to an eight-deep inline chain ending in
    // `std::sys::pal::unix::stack_overflow::imp::init`; without expansion only
    // the outer `std::rt::lang_start_internal` would appear.
    assert!(names.contains("std::rt::lang_start_internal"), "{names:?}");
    assert!(
        names.contains("std::sys::pal::unix::stack_overflow::imp::init"),
        "inlinee missing from {names:?}"
    );
    // The inlinees must sit below their caller, not beside it.
    let depth_of = |name: &str| -> u16 {
        let id = p.strings.lookup(name).expect("name interned");
        p.slices
            .name
            .iter()
            .position(|&n| n == id)
            .map(|i| p.slices.depth[i])
            .expect("slice for name")
    };
    assert!(
        depth_of("std::sys::pal::unix::stack_overflow::imp::init")
            > depth_of("std::rt::lang_start_internal"),
        "inlinee should be deeper than the function it was inlined into"
    );
}

#[test]
fn addresses_in_one_function_collapse_to_one_frame() {
    // 0x914, 0x91c and 0x92c are three return addresses inside `inl::outer_hot`.
    // Unsymbolicated they are three separate names, so consecutive samples
    // never extend a slice; symbolicated they are one.
    let plain = slice_names(&load_plain());
    let hot: Vec<_> = ["0x914", "0x91c", "0x92c"]
        .iter()
        .filter(|a| plain.contains(**a))
        .collect();
    assert!(hot.len() >= 2, "fixture should contain several hot addresses, got {hot:?}");

    let p = load_symbolicated();
    let id = p.strings.lookup("inl::outer_hot").expect("symbolicated");
    let depths: HashSet<u16> = p
        .slices
        .name
        .iter()
        .enumerate()
        .filter(|(_, &n)| n == id)
        .map(|(i, _)| p.slices.depth[i])
        .collect();
    assert_eq!(
        depths.len(),
        1,
        "all outer_hot addresses should land on one row, got depths {depths:?}"
    );
}

#[test]
fn lookup_is_keyed_by_lib_identity() {
    let syms = PrecogSymbols::parse(&std::fs::read(SIDECAR).unwrap()).unwrap();
    // The recorded binary, in Breakpad form (uppercase, no dashes, age suffix)
    // exactly as the profile's `libs[].breakpadId` spells it.
    let inl = "77DCCB24C9EA386DA2BA5D14C5EE7E620";
    let frames = syms
        .lookup(Some(inl), None, 0x91c)
        .expect("0x91c is inside inl::outer_hot");
    assert_eq!(frames.last().unwrap().name, "inl::outer_hot");

    // A debug id from some other build must not borrow these symbols, and an
    // address past the end of every known function must not either.
    assert!(syms
        .lookup(Some("11111111222233334444555555555555"), None, 0x91c)
        .is_none());
    assert!(syms.lookup(Some(inl), None, 0xffff_ff00).is_none());
}
