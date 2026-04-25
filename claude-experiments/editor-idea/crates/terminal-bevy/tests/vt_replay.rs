//! Replay tests against `libghostty_vt::Terminal` via our shared
//! `vt::build_terminal` helper.
//!
//! These exist because the `libghostty-vt` Rust binding panics
//! ("no handler set but callback is still called") whenever the VT
//! parser dispatches into a callback slot the user didn't register.
//! `vt::build_terminal` is supposed to register *every* slot with a
//! safe default — these tests feed realistic escape sequences that
//! exercise each slot and assert the terminal doesn't panic.
//!
//! If any real program in the wild crashes our terminal (e.g. on start,
//! like `claude --dangerously-skip-permissions` did), drop a test here
//! reproducing the sequence that crashed. The bar is: the terminal
//! should swallow arbitrary bytes without killing the process.

use terminal_bevy::vt::{self, CellPx};

const CELL_PX: CellPx = CellPx {
    width: 8,
    height: 18,
};

fn fresh() -> (
    Box<libghostty_vt::Terminal<'static, 'static>>,
    std::rc::Rc<std::cell::RefCell<Vec<u8>>>,
) {
    vt::build_terminal(80, 24, 1000, CELL_PX)
}

// ---------- Per-slot coverage: each handler slot must not panic ----------

#[test]
fn da1_query_does_not_panic_and_produces_response() {
    let (mut term, resp) = fresh();
    // DA1: CSI c → primary device attributes.
    term.vt_write(b"\x1b[c");
    let out = resp.borrow();
    assert!(
        !out.is_empty(),
        "DA1 query should produce a response, got empty"
    );
    // Response must start with CSI ? (xterm-style DA1 reply prefix).
    assert!(
        out.starts_with(b"\x1b[?"),
        "unexpected DA1 reply: {:?}",
        out
    );
}

#[test]
fn da2_query_does_not_panic() {
    let (mut term, resp) = fresh();
    // DA2: CSI > c → secondary device attributes.
    term.vt_write(b"\x1b[>c");
    assert!(!resp.borrow().is_empty(), "DA2 should respond");
}

#[test]
fn da3_query_does_not_panic() {
    let (mut term, resp) = fresh();
    // DA3: CSI = c → tertiary device attributes. Response is a DCS
    // sequence, not a CSI one; we only care that we didn't panic.
    term.vt_write(b"\x1b[=c");
    let _ = resp.borrow().len();
}

#[test]
fn xtversion_query_does_not_panic() {
    let (mut term, resp) = fresh();
    // XTVERSION: CSI > 0 q → version string response.
    term.vt_write(b"\x1b[>0q");
    // Some apps key off a non-empty XTVERSION reply; don't require it
    // as a hard invariant, just that the terminal didn't crash.
    let _ = resp.borrow().len();
}

#[test]
fn xtwinops_size_queries_do_not_panic() {
    let (mut term, _resp) = fresh();
    // CSI 14 t / 16 t / 18 t — pixel size / cell size / char size reports.
    term.vt_write(b"\x1b[14t");
    term.vt_write(b"\x1b[16t");
    term.vt_write(b"\x1b[18t");
}

#[test]
fn color_scheme_query_does_not_panic() {
    let (mut term, _resp) = fresh();
    // CSI ? 996 n — "what's the current color scheme?" Used by tools
    // that want to adapt to dark vs light.
    term.vt_write(b"\x1b[?996n");
}

#[test]
fn title_change_does_not_panic() {
    let (mut term, _resp) = fresh();
    // OSC 0 and OSC 2 both set the window title. Both must dispatch
    // through on_title_changed without crashing.
    term.vt_write(b"\x1b]0;hello\x1b\\");
    term.vt_write(b"\x1b]2;claude terminal\x07");
}

#[test]
fn bel_and_enq_do_not_panic() {
    let (mut term, _resp) = fresh();
    // BEL (0x07) and ENQ (0x05) dispatch through on_bell / on_enquiry.
    term.vt_write(b"hello\x07\x05world");
}

// ---------- End-to-end replay of realistic startup sequences ----------

#[test]
fn bash_prompt_startup_sequence() {
    // Representative bash/readline startup: reset, title set, DA1 probe,
    // cursor position report request, then a prompt.
    let (mut term, _resp) = fresh();
    term.vt_write(b"\x1bc"); // RIS (full reset)
    term.vt_write(b"\x1b]0;bash\x07");
    term.vt_write(b"\x1b[c");
    term.vt_write(b"\x1b[6n"); // CPR — position query
    term.vt_write(b"\x1b[?2004h"); // bracketed paste on
    term.vt_write(b"user@host:~$ ");
}

#[test]
fn vim_style_capability_probes() {
    let (mut term, _resp) = fresh();
    // vim/tmux/neovim commonly chain these on startup to figure out
    // what the terminal can do. If any of them routes into an unset
    // callback slot we'll panic here.
    term.vt_write(b"\x1b[>c"); // DA2
    term.vt_write(b"\x1b[=c"); // DA3
    term.vt_write(b"\x1b[>0q"); // XTVERSION
    term.vt_write(b"\x1b[?2026$p"); // synchronized output query (DECRQM)
    term.vt_write(b"\x1b[?1049h"); // alternate screen
    term.vt_write(b"\x1b[2J\x1b[H"); // clear + home
    term.vt_write(b"\x1b[?1049l"); // leave alternate screen
}

#[test]
fn claude_code_startup_style_sequence() {
    // Stand-in for the probe burst `claude --dangerously-skip-permissions`
    // emits on startup. We don't have the real captured bytes here, so
    // this approximates the shape: XTVERSION, color scheme, DA1, kitty
    // keyboard probe, OSC 10/11 (fg/bg queries), bracketed paste, and
    // then some 24-bit-color output.
    let (mut term, _resp) = fresh();
    term.vt_write(b"\x1b[>0q"); // XTVERSION
    term.vt_write(b"\x1b[?996n"); // color scheme probe
    term.vt_write(b"\x1b[c"); // DA1
    term.vt_write(b"\x1b[>c"); // DA2
    term.vt_write(b"\x1b[?u"); // kitty kb query
    term.vt_write(b"\x1b]10;?\x07"); // fg color query
    term.vt_write(b"\x1b]11;?\x07"); // bg color query
    term.vt_write(b"\x1b[?2004h"); // bracketed paste
    term.vt_write(b"\x1b[?25l"); // hide cursor
    term.vt_write(b"\x1b[38;2;255;128;0mclaude\x1b[0m ready\r\n");
    term.vt_write(b"\x1b[?25h"); // show cursor
}

#[test]
fn long_mixed_stream_does_not_panic() {
    // Fuzzy-but-plausible mixed output stream. A bunch of SGR,
    // cursor moves, bell, line feeds, UTF-8, and occasional queries.
    let (mut term, _resp) = fresh();
    for _ in 0..200 {
        term.vt_write(b"\x1b[1;31mred\x1b[0m ");
        term.vt_write(b"\x1b[2;34mblue\x1b[0m ");
        term.vt_write("ünicode café 日本語 🚀\r\n".as_bytes());
        term.vt_write(b"\x07"); // BEL
        term.vt_write(b"\x1b[H"); // cursor home
        term.vt_write(b"\x1b[c"); // DA1 mid-stream
    }
}

// ---------- Move-survival regression tests ----------
//
// These assert that moving a `Box<Terminal>` around (into structs,
// Vec entries, HashMap entries, across HashMap rehashes) does not
// invalidate the registered callbacks. They exist because the
// `libghostty-vt` 0.1.1 binding crashes (SIGBUS / "no handler set")
// when the *inner* Terminal is relocated after registration, for
// several of its callback slots. Our `vt::build_terminal` contract is
// to return `Box<Terminal>` — the Terminal never moves again, only
// the Box's 8-byte pointer does. A regression in any of these tests
// means every shell-emitted title change / bell / DA reply will
// crash the Bevy app.

#[test]
fn terminal_survives_single_move_then_title_change() {
    let (term, _resp) = fresh();
    let mut moved = term;
    moved.vt_write(b"\x1b]0;hello\x07");
}

#[test]
fn terminal_survives_wrap_in_struct_then_all_effects() {
    struct Holder {
        term: Box<libghostty_vt::Terminal<'static, 'static>>,
    }
    let (term, _resp) = fresh();
    let mut holder = Holder { term };
    holder.term.vt_write(b"\x07"); // BEL → on_bell
    holder.term.vt_write(b"\x05"); // ENQ → on_enquiry
    holder.term.vt_write(b"\x1b[c"); // DA1 → on_device_attributes
    holder.term.vt_write(b"\x1b[>0q"); // XTVERSION
    holder.term.vt_write(b"\x1b[14t"); // XTWINOPS → on_size
    holder.term.vt_write(b"\x1b[?996n"); // color scheme
    holder.term.vt_write(b"\x1b]0;hi\x07"); // title
}

#[test]
fn terminal_survives_vec_reallocation() {
    let mut v: Vec<Box<libghostty_vt::Terminal<'static, 'static>>> = Vec::new();
    let (term, _resp) = fresh();
    v.push(term);
    for _ in 0..32 {
        let (t, _) = vt::build_terminal(80, 24, 100, CELL_PX);
        v.push(t);
    }
    v[0].vt_write(b"\x1b]0;hello\x07");
    v[0].vt_write(b"\x07\x05\x1b[c");
}

/// Mirror the `spawn_terminal` flow in lib.rs: build, wrap in an Entry
/// that holds peer fields, insert into a HashMap keyed by entity id,
/// trigger many rehashes, then fire every effect. This is the test
/// that would have caught the Claude Code crash — if it regresses,
/// the live Bevy app will die as soon as anything sends a title or bell.
#[test]
fn terminal_survives_spawn_terminal_style_flow() {
    use std::collections::HashMap;
    struct Entry {
        term: Box<libghostty_vt::Terminal<'static, 'static>>,
        _cols: u16,
        _rows: u16,
    }
    let mut map: HashMap<u32, Entry> = HashMap::new();
    let (term, _resp) = fresh();
    map.insert(
        1,
        Entry {
            term,
            _cols: 80,
            _rows: 24,
        },
    );
    for i in 2..128 {
        let (t, _) = vt::build_terminal(80, 24, 100, CELL_PX);
        map.insert(
            i,
            Entry {
                term: t,
                _cols: 80,
                _rows: 24,
            },
        );
    }
    let entry = map.get_mut(&1).unwrap();
    entry.term.vt_write(b"\x1b]0;hello\x07");
    entry.term.vt_write(b"\x1b]2;claude\x1b\\");
    entry.term.vt_write(b"\x07\x05");
    entry.term.vt_write(b"\x1b[c\x1b[>c\x1b[=c");
}

#[test]
fn resize_during_stream_does_not_panic() {
    let (mut term, _resp) = fresh();
    term.vt_write(b"some output\r\n");
    term.resize(120, 40, CELL_PX.width, CELL_PX.height)
        .expect("resize");
    term.vt_write(b"more output after resize\r\n");
    term.resize(40, 10, CELL_PX.width, CELL_PX.height)
        .expect("shrink");
    term.vt_write(b"and after shrink\r\n");
    term.vt_write(b"\x1b[c"); // still-working DA query
}
