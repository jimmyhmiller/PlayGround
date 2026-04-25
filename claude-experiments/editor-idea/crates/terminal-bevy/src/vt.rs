//! Shared libghostty-vt setup: building a Terminal with every effect
//! handler registered.
//!
//! The `libghostty-vt` Rust bindings panic if a callback slot fires
//! without a registered handler ("no handler set but callback is still
//! called"). Real programs — notably `claude --dangerously-skip-permissions`,
//! vim, tmux, htop, bash readline — probe terminal capability via many
//! escape sequences during startup, so we *must* install a handler for
//! every slot the VT parser might dispatch. This module centralises that
//! so both `spawn_terminal` in lib.rs and the replay tests share the
//! same well-behaved construction path.
//!
//! The default responses are deliberately modest (VT220 w/ ANSI colour,
//! no sixel/kitty images, identifying ourselves as "terminal-bevy").
//! Title/bell/enquiry are silenced. Nothing here is load-bearing for
//! correctness — the important property is that every slot is non-None.
//!
//! ## Move safety
//!
//! The `libghostty-vt` 0.1.1 binding stores a USERDATA pointer into the
//! `Box<VTable>` inside a Terminal at registration time. In principle a
//! `Box`'s heap pointer is stable across moves of the outer Terminal, so
//! this should work — and upstream has a single test
//! (`callbacks_survive_explicit_relocation`) that claims so. In practice
//! we've seen SIGBUS / "no handler set" panics when the Terminal is moved
//! into a `HashMap` / `Vec` / `Box` **after** handler registration, for
//! any handler whose closure body didn't actually dereference its captures.
//! The replay tests in `tests/vt_replay.rs` reproduce it reliably.
//!
//! The fix is simple: register handlers only once the Terminal is in its
//! final heap home. `build_terminal` returns `Box<Terminal>` and the
//! handlers are registered on the boxed value, so subsequent moves of
//! the Box (or anything containing it) don't relocate the Terminal
//! itself.

use std::cell::RefCell;
use std::rc::Rc;

use libghostty_vt::{
    terminal::{
        ConformanceLevel, DeviceAttributeFeature, DeviceAttributes, DeviceType, SizeReportSize,
        PrimaryDeviceAttributes, SecondaryDeviceAttributes, TertiaryDeviceAttributes,
    },
    Terminal, TerminalOptions,
};

pub const APP_NAME: &str = "terminal-bevy";

/// Pixel size of one cell, reported back in XTWINOPS size queries.
/// Not used elsewhere in this module — the actual grid is driven by
/// `TerminalOptions` and `Terminal::resize`.
#[derive(Copy, Clone, Debug)]
pub struct CellPx {
    pub width: u32,
    pub height: u32,
}

/// Construct a Terminal with every handler slot populated.
///
/// Returns the Terminal in a `Box` so its heap address is stable — the
/// binding has move-after-registration issues (see module docs). Callers
/// should keep it boxed and only move the Box.
///
/// The returned `pty_response` buffer collects bytes the VT parser wants
/// written back to the child (DA replies, etc.); callers drain and
/// forward it.
pub fn build_terminal(
    cols: u16,
    rows: u16,
    max_scrollback: usize,
    cell_px: CellPx,
) -> (Box<Terminal<'static, 'static>>, Rc<RefCell<Vec<u8>>>) {
    let mut terminal = Box::new(
        Terminal::new(TerminalOptions {
            cols,
            rows,
            max_scrollback,
        })
        .expect("Terminal::new failed"),
    );
    terminal
        .resize(cols, rows, cell_px.width, cell_px.height)
        .expect("initial resize");

    // Register against the boxed terminal so USERDATA points into its
    // final heap home — subsequent moves of the Box don't relocate it.
    let pty_response = register_all_handlers(&mut *terminal, cell_px, cols, rows);
    (terminal, pty_response)
}

/// Install a handler for every VT effect slot so no dispatch ever fires
/// into an unset callback. Returns the pty_response buffer.
pub fn register_all_handlers(
    terminal: &mut Terminal<'static, 'static>,
    cell_px: CellPx,
    cols: u16,
    rows: u16,
) -> Rc<RefCell<Vec<u8>>> {
    let pty_response: Rc<RefCell<Vec<u8>>> = Rc::new(RefCell::new(Vec::with_capacity(128)));

    {
        let sink = pty_response.clone();
        terminal
            .on_pty_write(move |_term, data| {
                sink.borrow_mut().extend_from_slice(data);
            })
            .expect("on_pty_write");
    }

    terminal.on_bell(|_term| {}).expect("on_bell");
    terminal
        .on_enquiry(|_term| Some(APP_NAME))
        .expect("on_enquiry");
    terminal
        .on_xtversion(|_term| Some(APP_NAME))
        .expect("on_xtversion");
    terminal
        .on_title_changed(|_term| {})
        .expect("on_title_changed");

    terminal
        .on_size(move |_term| {
            Some(SizeReportSize {
                rows,
                columns: cols,
                cell_width: cell_px.width,
                cell_height: cell_px.height,
            })
        })
        .expect("on_size");

    // Color scheme — we don't query the OS, so report nothing. Programs
    // that care (e.g. bat autodetect) fall back to env heuristics.
    terminal
        .on_color_scheme(|_term| None)
        .expect("on_color_scheme");

    // DA1/DA2/DA3 — identify as VT220 w/ ANSI colour. Matches ghostling's
    // defaults, enough to keep readline + bash + claude happy.
    terminal
        .on_device_attributes(|_term| {
            Some(DeviceAttributes {
                primary: PrimaryDeviceAttributes::new(
                    ConformanceLevel::VT220,
                    [
                        DeviceAttributeFeature::COLUMNS_132,
                        DeviceAttributeFeature::SELECTIVE_ERASE,
                        DeviceAttributeFeature::ANSI_COLOR,
                    ],
                ),
                secondary: SecondaryDeviceAttributes {
                    device_type: DeviceType::VT220,
                    firmware_version: 1,
                    rom_cartridge: 0,
                },
                tertiary: TertiaryDeviceAttributes::default(),
            })
        })
        .expect("on_device_attributes");

    pty_response
}
