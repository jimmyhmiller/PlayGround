//! The ported chess widget (examples/widgets/chess.ft) driven end to end:
//! real clicks through the real layout math, real legality, real SAN/PGN —
//! and a REAL UCI engine (Stockfish) as a child process for the bot and
//! review-analysis tests.
//!
//! Board indices: 0 = a8 (top-left) … 63 = h1.

mod common;

use common::{install_widget_host, WidgetHost};
use funct::{Funct, Value};

const CHESS_SRC: &str = include_str!("../examples/widgets/chess.ft");

fn setup() -> (Funct, WidgetHost) {
    let mut vm = Funct::new();
    let host = install_widget_host(&mut vm);
    vm.set_global("canvas_w", Value::Float(800.0));
    vm.set_global("canvas_h", Value::Float(640.0));
    vm.eval(CHESS_SRC).expect("chess.ft must compile and run");
    vm.call("on_init", vec![]).expect("on_init");
    (vm, host)
}

fn xy(vm: &mut Funct, f: &str, arg: Value) -> (f64, f64) {
    let v = vm.call(f, vec![arg]).unwrap();
    match v {
        Value::Tuple(t) => {
            let fx = |v: &Value| match v {
                Value::Float(f) => *f,
                Value::Int(i) => *i as f64,
                other => panic!("expected number, got {:?}", other),
            };
            (fx(&t[0]), fx(&t[1]))
        }
        other => panic!("expected (x, y) tuple, got {:?}", other),
    }
}

fn click_at(vm: &mut Funct, x: f64, y: f64) {
    vm.call(
        "on_click",
        vec![
            Value::Float(x),
            Value::Float(y),
            Value::Bool(false),
            Value::Bool(false),
            Value::str(""),
        ],
    )
    .unwrap();
}

fn click_sq(vm: &mut Funct, idx: i64) {
    let (x, y) = xy(vm, "sq_click_xy", Value::Int(idx));
    click_at(vm, x, y);
}

fn click_btn(vm: &mut Funct, name: &str) {
    let (x, y) = xy(vm, "btn_xy", Value::str(name));
    click_at(vm, x, y);
}

fn mv(vm: &mut Funct, from: i64, to: i64) {
    click_sq(vm, from);
    click_sq(vm, to);
}

fn gs(vm: &mut Funct) -> serde_json::Value {
    vm.call("game_state", vec![]).unwrap().to_json().unwrap()
}

fn fen(vm: &mut Funct) -> String {
    match vm.call("current_fen", vec![]).unwrap() {
        Value::Str(s) => s.to_string(),
        other => panic!("expected fen string, got {:?}", other),
    }
}

fn board_at(vm: &mut Funct, i: i64) -> String {
    match vm.call("board_at", vec![Value::Int(i)]).unwrap() {
        Value::Str(s) => s.to_string(),
        other => panic!("expected piece string, got {:?}", other),
    }
}

fn sans(vm: &mut Funct) -> Vec<String> {
    match vm.call("history_san", vec![]).unwrap() {
        Value::List(items) => items
            .iter()
            .map(|v| match v {
                Value::Str(s) => s.to_string(),
                other => panic!("expected san string, got {:?}", other),
            })
            .collect(),
        other => panic!("expected san list, got {:?}", other),
    }
}

/// Pump on_frame like the host's animation tick until `pred` or timeout.
fn pump_until(vm: &mut Funct, secs: f64, mut pred: impl FnMut(&mut Funct) -> bool) -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(secs);
    while std::time::Instant::now() < deadline {
        vm.call("on_frame", vec![Value::Float(0.016)]).unwrap();
        if pred(vm) {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    false
}

// square shorthand used in the tests:
//   a2=48 a4=32 a6=16 a7=8 a8=0 b5=25 b7=9 b8=1
//   e2=52 e4=36 e5=28 e6=20 e7=12 f1=61 c4=34 c6=18 d1=59 h5=31 g8=6 f6=21 f7=13

#[test]
fn initial_render_and_fen() {
    let (mut vm, host) = setup();
    assert_eq!(
        fen(&mut vm),
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    );
    assert!(host.take_render_request());
    let frame = vm
        .call("render", vec![Value::Float(800.0), Value::Float(640.0)])
        .unwrap();
    let j = frame.to_json().unwrap();
    assert_eq!(j["kind"], "canvas");
    let children = j["children"].as_array().unwrap();
    assert!(
        children.len() > 100,
        "expected a full frame, got {} items",
        children.len()
    );
    let sprites = children.iter().filter(|c| c["kind"] == "sprite").count();
    assert_eq!(sprites, 32, "all pieces drawn");
    assert!(children.iter().any(|c| c["id"] == "sq_0"));
    assert!(children.iter().any(|c| c["id"] == "btn_new"));
    // piece sprites resolve through widget_asset
    assert!(children
        .iter()
        .any(|c| c["kind"] == "sprite" && c["path"].as_str().unwrap().ends_with("chess/kdt.png")));
}

#[test]
fn click_to_move_updates_board_fen_and_san() {
    let (mut vm, _host) = setup();
    click_sq(&mut vm, 52); // select e2
    assert_eq!(gs(&mut vm)["selected"], 52);
    click_sq(&mut vm, 36); // e4
    let g = gs(&mut vm);
    assert_eq!(g["moves"], 1);
    assert_eq!(g["turn"], "b");
    assert_eq!(board_at(&mut vm, 36), "wP");
    assert_eq!(board_at(&mut vm, 52), "");
    assert_eq!(sans(&mut vm), vec!["e4"]);
    // full FEN including the en-passant square
    assert_eq!(
        fen(&mut vm),
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    );
}

#[test]
fn illegal_moves_are_ignored() {
    let (mut vm, _host) = setup();
    click_sq(&mut vm, 52); // select e2
    click_sq(&mut vm, 35); // d4 — not reachable from e2
    let g = gs(&mut vm);
    assert_eq!(g["moves"], 0);
    assert_eq!(g["turn"], "w");
    // clicking an empty non-legal square also clears the selection
    assert_eq!(g["selected"], -1);
    // black piece can't be selected on white's turn
    click_sq(&mut vm, 12);
    assert_eq!(gs(&mut vm)["selected"], -1);
}

#[test]
fn scholars_mate_ends_the_game() {
    let (mut vm, _host) = setup();
    mv(&mut vm, 52, 36); // 1. e4
    mv(&mut vm, 12, 28); // 1... e5
    mv(&mut vm, 61, 34); // 2. Bc4
    mv(&mut vm, 1, 18); //  2... Nc6
    mv(&mut vm, 59, 31); // 3. Qh5
    mv(&mut vm, 6, 21); //  3... Nf6??
    mv(&mut vm, 31, 13); // 4. Qxf7#
    let g = gs(&mut vm);
    assert_eq!(g["over"], true);
    assert_eq!(g["status"], "White wins by checkmate");
    let s = sans(&mut vm);
    assert_eq!(s, vec!["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"]);
    // game over: further board clicks do nothing
    mv(&mut vm, 51, 35);
    assert_eq!(gs(&mut vm)["moves"], 7);
}

#[test]
fn promotion_picker_flow() {
    let (mut vm, _host) = setup();
    mv(&mut vm, 48, 32); // 1. a4
    mv(&mut vm, 9, 25); //  1... b5
    mv(&mut vm, 32, 25); // 2. axb5
    mv(&mut vm, 8, 16); //  2... a6
    mv(&mut vm, 25, 16); // 3. bxa6
    mv(&mut vm, 12, 20); // 3... e6
    mv(&mut vm, 16, 8); //  4. a7
    mv(&mut vm, 20, 28); // 4... e5
    mv(&mut vm, 8, 1); //   5. a7xb8 -> promotion picker opens
    let g = gs(&mut vm);
    assert_eq!(g["promo_pending"], true);
    assert_eq!(g["moves"], 8, "move not committed until a piece is picked");
    // the render shows the picker strip
    let frame = vm
        .call("render", vec![Value::Float(800.0), Value::Float(640.0)])
        .unwrap();
    let j = frame.to_json().unwrap();
    assert!(j["children"]
        .as_array()
        .unwrap()
        .iter()
        .any(|c| c["id"] == "promo_dim"));
    // pick cell 1 = rook (strip order Q, R, B, N)
    let (x, y) = xy(&mut vm, "promo_cell_xy", Value::Int(1));
    click_at(&mut vm, x, y);
    let g = gs(&mut vm);
    assert_eq!(g["promo_pending"], false);
    assert_eq!(g["moves"], 9);
    assert_eq!(board_at(&mut vm, 1), "wR");
    assert_eq!(sans(&mut vm).last().unwrap(), "axb8=R");
}

#[test]
fn promotion_cancel_by_clicking_outside() {
    let (mut vm, _host) = setup();
    mv(&mut vm, 48, 32);
    mv(&mut vm, 9, 25);
    mv(&mut vm, 32, 25);
    mv(&mut vm, 8, 16);
    mv(&mut vm, 25, 16);
    mv(&mut vm, 12, 20);
    mv(&mut vm, 16, 8);
    mv(&mut vm, 20, 28);
    mv(&mut vm, 8, 1); // picker opens
    assert_eq!(gs(&mut vm)["promo_pending"], true);
    click_sq(&mut vm, 36); // e4: outside the a/b-file strip -> cancel
    let g = gs(&mut vm);
    assert_eq!(g["promo_pending"], false);
    assert_eq!(g["moves"], 8, "promotion was cancelled, not committed");
    assert_eq!(board_at(&mut vm, 8), "wP", "pawn still on a7");
}

#[test]
fn bar_buttons_flip_new_level() {
    let (mut vm, _host) = setup();
    mv(&mut vm, 52, 36);
    click_btn(&mut vm, "flip");
    // flipped: clicking the same screen spot now selects the mirrored square;
    // verify through sq_click_xy round trip instead: e4 pawn still there
    assert_eq!(board_at(&mut vm, 36), "wP");
    click_btn(&mut vm, "new");
    let g = gs(&mut vm);
    assert_eq!(g["moves"], 0);
    assert_eq!(g["turn"], "w");
    assert_eq!(
        fen(&mut vm),
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    );
}

#[test]
fn pgn_copy_to_clipboard() {
    let (mut vm, host) = setup();
    mv(&mut vm, 52, 36); // e4
    mv(&mut vm, 12, 28); // e5
    click_btn(&mut vm, "copy");
    assert_eq!(gs(&mut vm)["copied"], true);
    let pgn = host.clipboard.lock().unwrap().clone();
    assert!(pgn.contains("[Event \"Casual Game\"]"), "{}", pgn);
    assert!(pgn.contains("1. e4 e5 *"), "{}", pgn);
}

#[test]
fn snapshot_round_trip_preserves_the_game() {
    let (mut vm, _host) = setup();
    mv(&mut vm, 52, 36); // e4
    mv(&mut vm, 12, 28); // e5
    let st = funct::VmState {
        frames: vec![],
        stack: vec![],
        status: funct::Status::Done(Value::Unit),
    };
    let saved = vm.save_state(&st).unwrap();

    // fresh engine, same host surface registered, restore, keep playing
    let mut vm2 = Funct::new();
    let _host2 = install_widget_host(&mut vm2);
    vm2.set_global("canvas_w", Value::Float(800.0));
    vm2.set_global("canvas_h", Value::Float(640.0));
    vm2.restore_state(&saved).unwrap();
    vm2.call("on_init", vec![]).unwrap();
    assert_eq!(sans(&mut vm2), vec!["e4", "e5"]);
    assert_eq!(
        fen(&mut vm2),
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 1"
    );
    mv(&mut vm2, 62, 45); // 2. Nf3 still works after restore
    assert_eq!(sans(&mut vm2).last().unwrap(), "Nf3");
}

// ---------- real UCI engine (Stockfish) ----------

/// Enable the bot via the real bar button; None if no UCI engine binary
/// could be spawned on this machine (test then skips loudly).
fn enable_bot(vm: &mut Funct) -> Option<()> {
    click_btn(vm, "bot"); // off -> bot plays Black
    let g = gs(vm);
    if g["engine_ok"] == false {
        eprintln!("SKIP: no UCI engine binary found (install stockfish to run this test)");
        return None;
    }
    assert_eq!(g["bot_side"], "b");
    assert!(g["eng"].as_i64().unwrap() > 0);
    Some(())
}

#[test]
fn plays_against_real_stockfish() {
    let (mut vm, host) = setup();
    if enable_bot(&mut vm).is_none() {
        return;
    }
    // human (White) plays e4 — the widget kicks the engine itself:
    // position fen ... / go movetime ... over the real pipe
    mv(&mut vm, 52, 36);
    let g = gs(&mut vm);
    assert_eq!(g["thinking"], true);
    assert!(host.animating.load(std::sync::atomic::Ordering::SeqCst));

    // pump the animation tick until the engine's bestmove is applied
    let done = pump_until(&mut vm, 30.0, |vm| gs(vm)["moves"] == 2);
    assert!(done, "engine never replied with a legal bestmove");
    let g = gs(&mut vm);
    assert_eq!(g["turn"], "w", "back to the human after the bot moved");
    assert_eq!(g["thinking"], false);
    assert_eq!(g["over"], false);
    let s = sans(&mut vm);
    assert_eq!(s.len(), 2);
    eprintln!("stockfish replied to 1. e4 with: {}", s[1]);

    // second exchange to prove the loop is stable: 2. Nf3
    mv(&mut vm, 62, 45);
    assert!(
        pump_until(&mut vm, 30.0, |vm| gs(vm)["moves"] == 4),
        "no reply to 2. Nf3"
    );
    assert_eq!(gs(&mut vm)["turn"], "w");
}

#[test]
fn review_mode_runs_real_analysis_sweep() {
    let (mut vm, _host) = setup();
    mv(&mut vm, 52, 36); // 1. e4
    mv(&mut vm, 12, 28); // 1... e5
    click_btn(&mut vm, "review");
    let g = gs(&mut vm);
    if g["engine_ok"] == false {
        eprintln!("SKIP: no UCI engine binary found (install stockfish to run this test)");
        return;
    }
    assert_eq!(g["review"], true);
    assert_eq!(g["sweeping"], true);
    assert_eq!(g["ply"], 2);

    // the sweep analyzes plies 0..=2 at ~300ms each, full strength
    let done = pump_until(&mut vm, 60.0, |vm| gs(vm)["sweeping"] == false);
    assert!(done, "analysis sweep never finished");

    // the review frame shows the eval bar, graph, and a best-move arrow
    let frame = vm
        .call("render", vec![Value::Float(800.0), Value::Float(640.0)])
        .unwrap();
    let j = frame.to_json().unwrap();
    let children = j["children"].as_array().unwrap();
    assert!(
        children.iter().any(|c| c["id"] == "eb_bg"),
        "eval bar present"
    );
    assert!(
        children.iter().any(|c| c["id"] == "gr_bg"),
        "eval graph present"
    );
    assert!(
        children.iter().any(|c| c["id"] == "bm_shaft"),
        "best-move arrow present"
    );

    // stepping back works through the real prev button
    click_btn(&mut vm, "prev");
    assert_eq!(gs(&mut vm)["ply"], 1);
    click_btn(&mut vm, "exit");
    let g = gs(&mut vm);
    assert_eq!(g["review"], false);
}
