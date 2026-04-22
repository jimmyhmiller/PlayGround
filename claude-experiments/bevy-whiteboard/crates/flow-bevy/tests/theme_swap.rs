//! Clicking the footer "Theme" button cycles through the three poster-ui
//! presets (iso50 → original → dark → iso50 …). This test doesn't verify
//! every visual surface re-skins — that's covered by poster-ui's re-skin
//! plugin — just that the `Theme` resource itself cycles.

mod common;

use common::make_app;
use flow_bevy::palette::ActionBtn;
use poster_ui::Theme;
use poster_ui::testing::click_by_marker;

#[test]
fn theme_button_cycles_presets() {
    let mut app = make_app();
    let start = app.world().resource::<Theme>().name;

    let clicked = click_by_marker::<ActionBtn, _>(&mut app, |a| matches!(a, ActionBtn::NextTheme));
    assert!(clicked, "no NextTheme button in palette");

    let after_one = app.world().resource::<Theme>().name;
    assert_ne!(start, after_one, "theme didn't change after one click");

    click_by_marker::<ActionBtn, _>(&mut app, |a| matches!(a, ActionBtn::NextTheme));
    click_by_marker::<ActionBtn, _>(&mut app, |a| matches!(a, ActionBtn::NextTheme));
    let after_three = app.world().resource::<Theme>().name;
    assert_eq!(
        start, after_three,
        "three clicks through the 3-preset cycle should return to start ({} ≠ {})",
        start, after_three
    );
}
