// chess.rhai — playable chess board with real piece images.
//
// Click a piece of the side-to-move, then click a destination square.
// Legal-move dots show on the empty target squares. Top bar: white/black
// turn dot on the left, "Flip board" rect, "New game" rect on the right.
//
// Piece images are loaded from
//   crates/widget-bevy/assets/chess/  (Wikimedia CBurnett set, CC BY-SA 3.0)
//
// IMPORTANT — Rhai scoping: user-defined `fn`s are PURE. They cannot
// read the outer `state` variable except when they are called directly
// via `engine.call_fn` (host-invoked handlers like `on_init`, `on_click`,
// `render`). Helper fns that need board data take it as a parameter,
// and state mutation happens only inside handler bodies.

fn new_board() {
    [
        "bR","bN","bB","bQ","bK","bB","bN","bR",
        "bP","bP","bP","bP","bP","bP","bP","bP",
        "","","","","","","","",
        "","","","","","","","",
        "","","","","","","","",
        "","","","","","","","",
        "wP","wP","wP","wP","wP","wP","wP","wP",
        "wR","wN","wB","wQ","wK","wB","wN","wR",
    ]
}

fn initial_castle() { #{ wK: true, wQ: true, bK: true, bQ: true } }

if !("board" in state) {
    state.board = new_board();
    state.turn = "w";
    state.selected = -1;
    state.last_from = -1;
    state.last_to = -1;
    state.castle = initial_castle();
    state.ep = -1;
    state.flipped = false;
    state.over = false;
    state.status = "";
}
// Pending-promotion handle: -1 = none, otherwise #{from, to}.
if !("promo" in state) { state.promo = -1; }
// Drag state. drag_from = -1 when no press is held. drag_active flips
// to true the first time the cursor moves between press and release
// (so a pure click won't trigger drag-style release semantics).
if !("drag_from" in state) {
    state.drag_from = -1;
    state.drag_active = false;
    state.drag_x = 0.0;
    state.drag_y = 0.0;
}
// Hover indicator for bar buttons: "", "flip", "new", "bot", "level".
// The host sends on_hover whenever the cursor moves over the pane
// without a button held, and `x = infinity` when it leaves.
if !("hover_btn" in state) { state.hover_btn = ""; }

// --- UCI bot play -------------------------------------------------
// `bot_side` is "" (two-human game), "w", or "b" — the color the
// engine plays. `level_idx` selects a think-time from level_ms().
// `thinking` is true while a search is in flight; on_frame polls the
// engine and applies the reply. `engine_ok` caches whether a UCI
// engine binary could be found (so the bar can show it's unavailable).
if !("bot_side" in state) {
    state.bot_side = "";
    state.level_idx = 1;
    state.thinking = false;
    state.engine_ok = true;
}
// `eng` is the subprocess handle for the UCI engine (-1 = not spawned).
// Separate guard so panes persisted by an earlier build (which had
// bot_side but no eng) migrate cleanly on reload.
if !("eng" in state) { state.eng = -1; }

// --- Game review --------------------------------------------------
// `history` is the move list ({from,to,promo,san}) for PGN + replay.
// `review` mode shows a past position (`ply` half-moves in) and runs
// full-strength Stockfish analysis: `best_from/to/san` is the engine's
// recommendation, `eval_str` the score (white-relative). `an_kind/val`
// accumulate the latest `info ... score` while a search streams.
if !("history" in state) { state.history = []; }
if !("review" in state) {
    state.review = false;
    state.ply = 0;
    // Per-ply analysis cache (index = ply): #{done, kind, wr, from, to, san}.
    // `wr` is white-relative centipawns (or mate-in-N white-relative).
    // Filled by a background sweep; `an_ply` is the ply being analyzed.
    state.evals = [];
    state.sweeping = false;
    state.an_ply = -1;
    state.copied = false;
}

fn on_init() {
    // A persisted handle never survives a worker restart (the process
    // was killed on close), so start clean and respawn on demand.
    state.eng = -1;
    state.thinking = false;
    // Come up on the live game, not a stale review session.
    state.review = false;
    state.sweeping = false;
    state.an_ply = -1;
    state.evals = [];
    state.copied = false;
    // If we come up mid-game and it's the bot's move, resume thinking.
    if bot_to_move(state.bot_side, state.over, state.turn) {
        state.eng = ensure_engine(state.eng);
        if state.eng == -1 {
            state.engine_ok = false;
        } else {
            kick_engine(state.eng, state.board, state.turn, state.castle, state.ep, state.level_idx);
            state.thinking = true;
            set_animating(true);
        }
    }
    request_render();
}
fn on_resize(w, h) { request_render(); }

// Arrow keys step through the game while reviewing (Left/Right = one
// ply, Home/End = jump to start/end). The host only sends these when
// this widget is focused, so they don't fight the board during play.
fn on_key(key) {
    if !state.review { return; }
    let n = state.history.len();
    let np = state.ply;
    if key == "ArrowLeft" || key == "ArrowDown" { np = state.ply - 1; }
    else if key == "ArrowRight" || key == "ArrowUp" { np = state.ply + 1; }
    else if key == "Home" { np = 0; }
    else if key == "End" { np = n; }
    else { return; }
    if np < 0 { np = 0; }
    if np > n { np = n; }
    if np != state.ply { state.ply = np; request_render(); }
}

fn on_hover(x, y) {
    let L = compute_layout(canvas_w, canvas_h, state.bot_side != "", state.review);
    let new_hover = "";
    // INFINITY x means the cursor left the pane.
    if x < 1e9 && y >= 0.0 && y < L.bar_h {
        if state.review {
            if x >= L.btn_prev_x && x <= L.btn_prev_x + L.btn_prev_w { new_hover = "prev"; }
            else if x >= L.btn_next_x && x <= L.btn_next_x + L.btn_next_w { new_hover = "next"; }
            else if x >= L.btn_exit_x && x <= L.btn_exit_x + L.btn_exit_w { new_hover = "exit"; }
        } else {
            if x >= L.btn_flip_x && x <= L.btn_flip_x + L.btn_flip_w {
                new_hover = "flip";
            } else if x >= L.btn_new_x && x <= L.btn_new_x + L.btn_new_w {
                new_hover = "new";
            } else if x >= L.btn_bot_x && x <= L.btn_bot_x + L.btn_bot_w {
                new_hover = "bot";
            } else if state.bot_side != "" && x >= L.btn_lvl_x && x <= L.btn_lvl_x + L.btn_lvl_w {
                new_hover = "level";
            } else if x >= L.btn_review_x && x <= L.btn_review_x + L.btn_review_w {
                new_hover = "review";
            } else if x >= L.btn_copy_x && x <= L.btn_copy_x + L.btn_copy_w {
                new_hover = "copy";
            }
        }
    }
    // Moving the mouse clears the transient "Copied" flash.
    if state.copied && new_hover != "copy" { state.copied = false; request_render(); }
    if state.hover_btn != new_hover {
        state.hover_btn = new_hover;
        request_render();
    }
}

fn pc(p) { if p == "" { "" } else { p.sub_string(0, 1) } }
fn pt(p) { if p == "" { "" } else { p.sub_string(1) } }

fn in_bounds(r, c) { r >= 0 && r < 8 && c >= 0 && c < 8 }
fn rc_to_idx(r, c) { r * 8 + c }
fn idx_r(i) { i / 8 }
fn idx_c(i) { i % 8 }

// Squares a piece attacks. Pawn attacks differ from pawn forward moves;
// castling is never an attack (would recurse into is_attacked).
fn attacks_from(board, from) {
    let p = board[from];
    if p == "" { return []; }
    let color = pc(p);
    let kind = pt(p);
    let r = idx_r(from); let c = idx_c(from);
    let out = [];
    if kind == "P" {
        let dir = if color == "w" { -1 } else { 1 };
        let dcs = [-1, 1];
        for dc in dcs {
            let rr = r + dir; let cc = c + dc;
            if in_bounds(rr, cc) { out.push(rc_to_idx(rr, cc)); }
        }
    } else if kind == "N" {
        let jumps = [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]];
        for j in jumps {
            let rr = r + j[0]; let cc = c + j[1];
            if in_bounds(rr, cc) { out.push(rc_to_idx(rr, cc)); }
        }
    } else if kind == "B" || kind == "R" || kind == "Q" {
        let dirs = [];
        if kind != "R" {
            dirs.push([-1,-1]); dirs.push([-1,1]);
            dirs.push([1,-1]);  dirs.push([1,1]);
        }
        if kind != "B" {
            dirs.push([-1,0]); dirs.push([1,0]);
            dirs.push([0,-1]); dirs.push([0,1]);
        }
        for d in dirs {
            let rr = r + d[0]; let cc = c + d[1];
            while in_bounds(rr, cc) {
                let t = rc_to_idx(rr, cc);
                out.push(t);
                if board[t] != "" { break; }
                rr += d[0]; cc += d[1];
            }
        }
    } else if kind == "K" {
        for dr in [-1, 0, 1] {
            for dc in [-1, 0, 1] {
                if dr == 0 && dc == 0 { continue; }
                let rr = r + dr; let cc = c + dc;
                if in_bounds(rr, cc) { out.push(rc_to_idx(rr, cc)); }
            }
        }
    }
    out
}

fn is_attacked(board, sq, by_color) {
    let i = 0;
    while i < 64 {
        let p = board[i];
        if p != "" && pc(p) == by_color {
            let atks = attacks_from(board, i);
            for a in atks { if a == sq { return true; } }
        }
        i += 1;
    }
    false
}

fn king_sq(board, color) {
    let target = if color == "w" { "wK" } else { "bK" };
    let i = 0;
    while i < 64 {
        if board[i] == target { return i; }
        i += 1;
    }
    -1
}

// --- Material count -----------------------------------------------
// Standard piece values (king = 0, it's never captured).
fn piece_value(k) {
    if k == "P" { 1 }
    else if k == "N" || k == "B" { 3 }
    else if k == "R" { 5 }
    else if k == "Q" { 9 }
    else { 0 }
}
// White material minus Black material on `board` (the lead, in points).
fn material_adv(board) {
    let diff = 0;
    let i = 0;
    while i < 64 {
        let p = board[i];
        if p != "" {
            let v = piece_value(pt(p));
            if pc(p) == "w" { diff += v; } else { diff -= v; }
        }
        i += 1;
    }
    diff
}

// --- UCI helpers --------------------------------------------------
// Board index 0 = a8 (top-left), 63 = h1. UCI squares are file+rank,
// so file = idx % 8 (a..h) and rank = 8 - idx/8 (8..1).

// Difficulty is governed by Stockfish's UCI_Elo limiter, not by
// think-time. Think-time is kept modest and roughly constant so the
// engine just plays at its capped strength.
fn level_elo(idx) {
    if idx == 0 { 1350 }
    else if idx == 1 { 1850 }
    else if idx == 2 { 2350 }
    else { 2850 }
}
fn level_ms(idx) {
    if idx == 0 { 300 }
    else if idx == 1 { 500 }
    else if idx == 2 { 700 }
    else { 1000 }
}
// Button shows the actual Elo so the strength is unambiguous.
fn level_label(idx) { level_elo(idx).to_string() }

fn files_arr() { ["a","b","c","d","e","f","g","h"] }

fn sq_name(idx) {
    let f = idx % 8;
    let rank = 8 - (idx / 8);
    files_arr()[f] + rank.to_string()
}

// Map a single rank-digit string "1".."8" to its integer.
fn digit_val(ch) {
    let digits = ["1","2","3","4","5","6","7","8"];
    let i = 0;
    while i < 8 { if digits[i] == ch { return i + 1; } i += 1; }
    0
}

// "e2" -> board index. Returns -1 on a malformed square.
fn name_to_idx(s) {
    if s.len() < 2 { return -1; }
    let fc = s.sub_string(0, 1);
    let rank = digit_val(s.sub_string(1, 1));
    if rank == 0 { return -1; }
    let files = files_arr();
    let col = -1;
    let i = 0;
    while i < 8 { if files[i] == fc { col = i; break; } i += 1; }
    if col == -1 { return -1; }
    let row = 8 - rank;
    row * 8 + col
}

// Full FEN for the current position. Halfmove/fullmove counters are
// fixed at "0 1" — the widget doesn't track them and they only affect
// the 50-move rule, which is irrelevant for casual play vs a bot.
fn board_to_fen(board, turn, castle, ep) {
    let fen = "";
    let r = 0;
    while r < 8 {
        let empty = 0;
        let c = 0;
        while c < 8 {
            let p = board[r * 8 + c];
            if p == "" {
                empty += 1;
            } else {
                if empty > 0 { fen += empty.to_string(); empty = 0; }
                let letter = pt(p);
                if pc(p) == "w" { fen += letter; } else { fen += letter.to_lower(); }
            }
            c += 1;
        }
        if empty > 0 { fen += empty.to_string(); }
        if r < 7 { fen += "/"; }
        r += 1;
    }
    let cr = "";
    if castle.wK { cr += "K"; }
    if castle.wQ { cr += "Q"; }
    if castle.bK { cr += "k"; }
    if castle.bQ { cr += "q"; }
    if cr == "" { cr = "-"; }
    let ep_str = if ep == -1 { "-" } else { sq_name(ep) };
    fen + " " + turn + " " + cr + " " + ep_str + " 0 1"
}

// "e2e4" / "e7e8q" -> #{from, to, promo}. promo is "" or "Q/R/B/N".
// Returns #{from:-1,...} for the engine's "(none)" / "0000" no-move.
fn parse_engine_move(mv) {
    if mv.len() < 4 { return #{ from: -1, to: -1, promo: "" }; }
    let from = name_to_idx(mv.sub_string(0, 2));
    let to = name_to_idx(mv.sub_string(2, 2));
    let promo = "";
    if mv.len() >= 5 { promo = mv.sub_string(4, 1).to_upper(); }
    #{ from: from, to: to, promo: promo }
}

// Is it the engine's move right now?
fn bot_to_move(bot_side, over, turn) {
    bot_side != "" && !over && turn == bot_side
}

// Engine binaries to try, in order. $CHESS_UCI_ENGINE wins if set.
// Absolute paths matter: a bundled macOS app launched via Finder gets
// a minimal PATH that omits /opt/homebrew/bin, so bare "stockfish"
// alone would fail there.
fn engine_candidates() {
    let c = [];
    let env = host_env("CHESS_UCI_ENGINE");
    if env != "" { c.push(env); }
    c.push("stockfish");
    c.push("/opt/homebrew/bin/stockfish");
    c.push("/usr/local/bin/stockfish");
    c.push("/usr/bin/stockfish");
    c.push("lc0");
    c.push("/opt/homebrew/bin/lc0");
    c
}

// Ensure a live engine process, returning its handle (-1 on failure).
// Reuses `cur` if it's still alive, otherwise spawns the first engine
// candidate that starts and runs the UCI handshake. Pure helper (only
// calls host fns); the caller stores the returned handle into state.
fn ensure_engine(cur) {
    if cur != -1 && proc_alive(cur) { return cur; }
    let id = -1;
    for cand in engine_candidates() {
        let h = proc_spawn(cand);
        if h != -1 { id = h; break; }
    }
    if id != -1 {
        proc_write(id, "uci");
        proc_write(id, "isready");
        proc_write(id, "ucinewgame");
    }
    id
}

// Kick off an async search for the current position at the selected
// Elo. Pure helper (only host fns); the caller flips state.thinking
// and set_animating. First sends `stop` and drains any buffered
// output so a reply from an aborted prior search can't be mistaken
// for this one.
fn kick_engine(eng, board, turn, castle, ep, level_idx) {
    proc_write(eng, "stop");
    loop { if proc_read(eng) == "" { break; } }
    let fen = board_to_fen(board, turn, castle, ep);
    proc_write(eng, "setoption name UCI_LimitStrength value true");
    proc_write(eng, "setoption name UCI_Elo value " + level_elo(level_idx).to_string());
    proc_write(eng, "position fen " + fen);
    proc_write(eng, "go movetime " + level_ms(level_idx).to_string());
}

// Full-strength analysis (limiter OFF) of the position at `ply`,
// thinking for `movetime` ms. Drains stale output first, like
// kick_engine. Pure helper.
fn analyze_ply(eng, history, ply, movetime) {
    proc_write(eng, "stop");
    loop { if proc_read(eng) == "" { break; } }
    proc_write(eng, "setoption name UCI_LimitStrength value false");
    let v = replay_to(history, ply);
    let fen = board_to_fen(v.board, v.turn, v.castle, v.ep);
    proc_write(eng, "position fen " + fen);
    proc_write(eng, "go movetime " + movetime.to_string());
}

// Logistic win-probability (0=Black winning, 1=White winning) from a
// white-relative score. Used by the eval bar and graph.
fn win_prob(kind, wr) {
    if kind == "mate" { if wr >= 0 { 0.98 } else { 0.02 } }
    else if kind == "cp" { 1.0 / (1.0 + exp(-(wr.to_float()) / 350.0)) }
    else { 0.5 }
}

// Per-review think time per ply (ms). Keep short so the sweep fills the
// graph in a few seconds; full-strength even at this depth is strong.
fn review_movetime() { 300 }

// Parse "info ... score cp N ..." / "score mate M ...". Returns
// #{kind:"cp"/"mate"/"none", val}. Score is side-to-move-relative.
fn parse_info_score(line) {
    let toks = line.split(' ');
    let i = 0;
    while i < toks.len() {
        if toks[i] == "score" && i + 2 < toks.len() {
            return #{ kind: toks[i + 1], val: parse_int(toks[i + 2]) };
        }
        i += 1;
    }
    #{ kind: "none", val: 0 }
}

// White-relative eval string from a side-to-move-relative score.
fn eval_display(kind, val, turn) {
    let sign = if turn == "b" { -1 } else { 1 };
    if kind == "mate" {
        let m = val * sign;
        if m >= 0 { "+M" + m.to_string() } else { "-M" + (-m).to_string() }
    } else if kind == "cp" {
        let cp = val * sign;
        let acp = if cp < 0 { -cp } else { cp };
        let whole = acp / 100;
        let frac = acp % 100;
        let fs = if frac < 10 { "0" + frac.to_string() } else { frac.to_string() };
        (if cp >= 0 { "+" } else { "-" }) + whole.to_string() + "." + fs
    } else { "" }
}

// --- SAN + PGN ----------------------------------------------------

fn is_check(board, color) {
    let k = king_sq(board, color);
    if k == -1 { return false; }
    let o = if color == "w" { "b" } else { "w" };
    is_attacked(board, k, o)
}

// SAN disambiguation: which of file/rank (or both) is needed because
// another same-type piece could also reach `to`.
fn san_disambig(board, castle, ep, from, to) {
    let me = pc(board[from]);
    let kind = pt(board[from]);
    let others = [];
    let i = 0;
    while i < 64 {
        if i != from && board[i] != "" && pc(board[i]) == me && pt(board[i]) == kind {
            let l = legal_from(board, castle, ep, i);
            for m in l { if m == to { others.push(i); break; } }
        }
        i += 1;
    }
    if others.len() == 0 { return ""; }
    let files = files_arr();
    let same_file = false;
    let same_rank = false;
    for o in others {
        if idx_c(o) == idx_c(from) { same_file = true; }
        if idx_r(o) == idx_r(from) { same_rank = true; }
    }
    if !same_file { return files[idx_c(from)]; }
    if !same_rank { return (8 - idx_r(from)).to_string(); }
    files[idx_c(from)] + (8 - idx_r(from)).to_string()
}

// SAN without the +/# suffix (computed from the pre-move board).
fn move_san_core(board, castle, ep, turn, from, to, promo) {
    let kind = pt(board[from]);
    if kind == "K" {
        let dc = idx_c(to) - idx_c(from);
        if dc == 2 { return "O-O"; }
        if dc == -2 { return "O-O-O"; }
    }
    let dest = sq_name(to);
    let is_cap = board[to] != "" || (kind == "P" && to == ep && ep != -1);
    if kind == "P" {
        let s = if is_cap { files_arr()[idx_c(from)] + "x" + dest } else { dest };
        if idx_r(to) == 0 || idx_r(to) == 7 {
            let pk = if promo == "" { "Q" } else { promo };
            s = s + "=" + pk;
        }
        return s;
    }
    let disamb = san_disambig(board, castle, ep, from, to);
    kind + disamb + (if is_cap { "x" } else { "" }) + dest
}

// Full SAN: core + check/mate suffix derived from the post-move result.
fn full_san(board, castle, ep, turn, from, to, promo, res_board, res_turn, res_over, res_status) {
    let core = move_san_core(board, castle, ep, turn, from, to, promo);
    if res_over && res_status.contains("checkmate") { return core + "#"; }
    if is_check(res_board, res_turn) { return core + "+"; }
    core
}

// Build a PGN string from the recorded history.
fn build_pgn(history, over, status, bot_side, level_idx) {
    let result = "*";
    if over {
        if status.contains("White wins") { result = "1-0"; }
        else if status.contains("Black wins") { result = "0-1"; }
        else { result = "1/2-1/2"; }
    }
    let white = "Player";
    let black = "Player";
    if bot_side == "w" {
        white = "Stockfish " + level_elo(level_idx).to_string();
        black = "You";
    } else if bot_side == "b" {
        white = "You";
        black = "Stockfish " + level_elo(level_idx).to_string();
    }
    let pgn = "[Event \"Casual Game\"]\n";
    pgn += "[Site \"editor-idea chess widget\"]\n";
    pgn += "[Date \"????.??.??\"]\n";
    pgn += "[White \"" + white + "\"]\n";
    pgn += "[Black \"" + black + "\"]\n";
    pgn += "[Result \"" + result + "\"]\n\n";
    let body = "";
    let i = 0;
    while i < history.len() {
        if i % 2 == 0 { body += ((i / 2) + 1).to_string() + ". "; }
        body += history[i].san + " ";
        i += 1;
    }
    body += result;
    pgn + body + "\n"
}

// --- Review replay ------------------------------------------------

// Apply a move without mate/stalemate detection (cheap; for replay).
fn apply_quiet(board, castle, ep, turn, from, to, promo) {
    let mover = board[from];
    let kind = pt(mover);
    let new_ep = -1;
    if kind == "P" && (idx_r(from) - idx_r(to)).abs() == 2 {
        new_ep = rc_to_idx((idx_r(from) + idx_r(to)) / 2, idx_c(from));
    }
    let b2 = make_move_clone(board, ep, from, to, promo);
    let cr = castle;
    if kind == "K" {
        if pc(mover) == "w" { cr.wK = false; cr.wQ = false; }
        else { cr.bK = false; cr.bQ = false; }
    }
    if kind == "R" {
        if from == 56 { cr.wQ = false; }
        if from == 63 { cr.wK = false; }
        if from == 0  { cr.bQ = false; }
        if from == 7  { cr.bK = false; }
    }
    if to == 56 { cr.wQ = false; }
    if to == 63 { cr.wK = false; }
    if to == 0  { cr.bQ = false; }
    if to == 7  { cr.bK = false; }
    #{ board: b2, castle: cr, ep: new_ep, turn: if turn == "w" { "b" } else { "w" } }
}

// Reconstruct the position after `ply` half-moves from the start.
fn replay_to(history, ply) {
    let board = new_board();
    let castle = initial_castle();
    let ep = -1;
    let turn = "w";
    let lf = -1;
    let lt = -1;
    let i = 0;
    while i < ply && i < history.len() {
        let mv = history[i];
        let r = apply_quiet(board, castle, ep, turn, mv.from, mv.to, mv.promo);
        board = r.board; castle = r.castle; ep = r.ep; turn = r.turn;
        lf = mv.from; lt = mv.to;
        i += 1;
    }
    #{ board: board, castle: castle, ep: ep, turn: turn, last_from: lf, last_to: lt }
}

// --- Arrow geometry -----------------------------------------------

fn pi_const() { 3.14159265358979 }

fn atan2_deg(y, x) {
    let pi = pi_const();
    let r = 0.0;
    if x > 0.0 { r = atan(y / x); }
    else if x < 0.0 && y >= 0.0 { r = atan(y / x) + pi; }
    else if x < 0.0 && y < 0.0 { r = atan(y / x) - pi; }
    else if x == 0.0 && y > 0.0 { r = pi / 2.0; }
    else if x == 0.0 && y < 0.0 { r = -pi / 2.0; }
    else { r = 0.0; }
    r * 180.0 / pi
}

// Screen-space center of a board square, honoring perspective flip.
fn sq_center(L, flipped, idx) {
    let vr = if flipped { 7 - idx_r(idx) } else { idx_r(idx) };
    let vc = if flipped { 7 - idx_c(idx) } else { idx_c(idx) };
    #{ x: L.board_x + vc.to_float() * L.sq + L.sq / 2.0,
       y: L.board_y + vr.to_float() * L.sq + L.sq / 2.0 }
}

fn gen_pseudo(board, castle, ep, from) {
    let p = board[from];
    if p == "" { return []; }
    let me = pc(p);
    let kind = pt(p);
    let opp = if me == "w" { "b" } else { "w" };
    let r = idx_r(from); let c = idx_c(from);
    let out = [];

    if kind == "P" {
        let dir = if me == "w" { -1 } else { 1 };
        let start_r = if me == "w" { 6 } else { 1 };
        let r1 = r + dir;
        if in_bounds(r1, c) && board[rc_to_idx(r1, c)] == "" {
            out.push(rc_to_idx(r1, c));
            let r2 = r + 2*dir;
            if r == start_r && in_bounds(r2, c) && board[rc_to_idx(r2, c)] == "" {
                out.push(rc_to_idx(r2, c));
            }
        }
        let dcs = [-1, 1];
        for dc in dcs {
            let cc = c + dc;
            if in_bounds(r1, cc) {
                let t = rc_to_idx(r1, cc);
                let tgt = board[t];
                if tgt != "" && pc(tgt) == opp {
                    out.push(t);
                } else if t == ep && ep != -1 {
                    out.push(t);
                }
            }
        }
    } else if kind == "N" {
        let jumps = [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]];
        for j in jumps {
            let rr = r + j[0]; let cc = c + j[1];
            if in_bounds(rr, cc) {
                let t = rc_to_idx(rr, cc);
                if board[t] == "" || pc(board[t]) == opp { out.push(t); }
            }
        }
    } else if kind == "B" || kind == "R" || kind == "Q" {
        let dirs = [];
        if kind != "R" {
            dirs.push([-1,-1]); dirs.push([-1,1]);
            dirs.push([1,-1]);  dirs.push([1,1]);
        }
        if kind != "B" {
            dirs.push([-1,0]); dirs.push([1,0]);
            dirs.push([0,-1]); dirs.push([0,1]);
        }
        for d in dirs {
            let rr = r + d[0]; let cc = c + d[1];
            while in_bounds(rr, cc) {
                let t = rc_to_idx(rr, cc);
                if board[t] == "" {
                    out.push(t);
                } else {
                    if pc(board[t]) == opp { out.push(t); }
                    break;
                }
                rr += d[0]; cc += d[1];
            }
        }
    } else if kind == "K" {
        for dr in [-1, 0, 1] {
            for dc in [-1, 0, 1] {
                if dr == 0 && dc == 0 { continue; }
                let rr = r + dr; let cc = c + dc;
                if in_bounds(rr, cc) {
                    let t = rc_to_idx(rr, cc);
                    if board[t] == "" || pc(board[t]) == opp { out.push(t); }
                }
            }
        }
        let home_r = if me == "w" { 7 } else { 0 };
        if r == home_r && c == 4 {
            let kk = if me == "w" { "wK" } else { "bK" };
            let qq = if me == "w" { "wQ" } else { "bQ" };
            if castle[kk]
                && board[rc_to_idx(home_r, 5)] == ""
                && board[rc_to_idx(home_r, 6)] == ""
                && !is_attacked(board, rc_to_idx(home_r, 4), opp)
                && !is_attacked(board, rc_to_idx(home_r, 5), opp)
                && !is_attacked(board, rc_to_idx(home_r, 6), opp) {
                out.push(rc_to_idx(home_r, 6));
            }
            if castle[qq]
                && board[rc_to_idx(home_r, 1)] == ""
                && board[rc_to_idx(home_r, 2)] == ""
                && board[rc_to_idx(home_r, 3)] == ""
                && !is_attacked(board, rc_to_idx(home_r, 4), opp)
                && !is_attacked(board, rc_to_idx(home_r, 3), opp)
                && !is_attacked(board, rc_to_idx(home_r, 2), opp) {
                out.push(rc_to_idx(home_r, 2));
            }
        }
    }
    out
}

// Apply candidate move to a clone of the board. Auto-promotes pawns to
// queens, handles castling rook hop and en-passant capture.
// `promo` is "" (auto-queen, used for legality testing where the piece
// choice doesn't affect king safety) or one of "Q","R","B","N" (used
// when applying the user's promotion pick).
fn make_move_clone(board, ep, from, to, promo) {
    let b = board;
    let mover = b[from];
    let kind = pt(mover);
    let me = pc(mover);
    b[from] = "";
    b[to] = mover;
    if kind == "P" && to == ep && ep != -1 {
        let cap_r = idx_r(from);
        let cap_c = idx_c(to);
        b[rc_to_idx(cap_r, cap_c)] = "";
    }
    if kind == "K" {
        let dc = idx_c(to) - idx_c(from);
        let r0 = idx_r(from);
        if dc == 2 {
            b[rc_to_idx(r0, 5)] = b[rc_to_idx(r0, 7)];
            b[rc_to_idx(r0, 7)] = "";
        } else if dc == -2 {
            b[rc_to_idx(r0, 3)] = b[rc_to_idx(r0, 0)];
            b[rc_to_idx(r0, 0)] = "";
        }
    }
    if kind == "P" && (idx_r(to) == 0 || idx_r(to) == 7) {
        let pick = if promo == "" { "Q" } else { promo };
        b[to] = me + pick;
    }
    b
}

// True if applying this move means a pawn lands on its promotion rank.
fn is_promotion_move(board, from, to) {
    let p = board[from];
    if p == "" { return false; }
    if pt(p) != "P" { return false; }
    idx_r(to) == 0 || idx_r(to) == 7
}

fn legal_from(board, castle, ep, from) {
    let p = board[from];
    if p == "" { return []; }
    let me = pc(p);
    let opp = if me == "w" { "b" } else { "w" };
    let pseudo = gen_pseudo(board, castle, ep, from);
    let legal = [];
    for to in pseudo {
        let b2 = make_move_clone(board, ep, from, to, "");
        let ksq = king_sq(b2, me);
        if ksq == -1 { continue; }
        if !is_attacked(b2, ksq, opp) { legal.push(to); }
    }
    legal
}

fn any_legal_for(board, castle, ep, color) {
    let i = 0;
    while i < 64 {
        let p = board[i];
        if p != "" && pc(p) == color {
            let l = legal_from(board, castle, ep, i);
            if l.len() > 0 { return true; }
        }
        i += 1;
    }
    false
}

// Pure: compute the next state given a legal move. `promo` is the
// chosen promotion piece ("Q"/"R"/"B"/"N") or "" for non-promotion
// moves (and as a fallback that auto-queens).
fn make_result(board, castle, ep, turn, from, to, promo) {
    let mover = board[from];
    let kind = pt(mover);
    let new_ep = -1;
    if kind == "P" && (idx_r(from) - idx_r(to)).abs() == 2 {
        new_ep = rc_to_idx((idx_r(from) + idx_r(to)) / 2, idx_c(from));
    }
    let b2 = make_move_clone(board, ep, from, to, promo);
    let cr = castle;
    if kind == "K" {
        if pc(mover) == "w" { cr.wK = false; cr.wQ = false; }
        else { cr.bK = false; cr.bQ = false; }
    }
    if kind == "R" {
        if from == 56 { cr.wQ = false; }
        if from == 63 { cr.wK = false; }
        if from == 0  { cr.bQ = false; }
        if from == 7  { cr.bK = false; }
    }
    if to == 56 { cr.wQ = false; }
    if to == 63 { cr.wK = false; }
    if to == 0  { cr.bQ = false; }
    if to == 7  { cr.bK = false; }

    let next_turn = if turn == "w" { "b" } else { "w" };
    let over = false;
    let status = "";
    if !any_legal_for(b2, cr, new_ep, next_turn) {
        let ksq = king_sq(b2, next_turn);
        let opp = if next_turn == "w" { "b" } else { "w" };
        over = true;
        if ksq != -1 && is_attacked(b2, ksq, opp) {
            status = (if next_turn == "w" { "Black" } else { "White" }) + " wins by checkmate";
        } else {
            status = "Stalemate";
        }
    }
    #{
        board: b2,
        castle: cr,
        ep: new_ep,
        last_from: from,
        last_to: to,
        turn: next_turn,
        over: over,
        status: status,
    }
}

fn piece_path(p) {
    let prefix = "";
    let k = pt(p);
    if k == "K" { prefix = "k"; }
    else if k == "Q" { prefix = "q"; }
    else if k == "R" { prefix = "r"; }
    else if k == "B" { prefix = "b"; }
    else if k == "N" { prefix = "n"; }
    else { prefix = "p"; }
    let suffix = if pc(p) == "w" { "lt" } else { "dt" };
    widget_asset("chess/" + prefix + suffix + ".png")
}

// Bar layout is mode-aware: in review mode the bar carries the
// stepper (◀ ▶ Exit); otherwise the play controls (Copy / Review /
// Lvl / Bot / Flip / New). Buttons are laid out right-to-left from a
// cursor so a hidden Lvl button (bot off) leaves no gap.
fn compute_layout(w, h, bot_on, review) {
    let bar_h = 32.0;
    let pad = 8.0;
    // Review reserves a left strip (eval bar) and a bottom strip (graph),
    // so the board shrinks to make room.
    let eb_w = if review { 16.0 } else { 0.0 };
    let eb_gap = if review { 10.0 } else { 0.0 };
    let gr_h = if review { 58.0 } else { 0.0 };
    let gr_gap = if review { 10.0 } else { 0.0 };
    let avail_w = w - 2.0 * pad - eb_w - eb_gap;
    let avail_h = h - bar_h - 2.0 * pad - gr_h - gr_gap;
    let sq = (if avail_w < avail_h { avail_w } else { avail_h }) / 8.0;
    if sq < 8.0 { sq = 8.0; }
    let board_w = sq * 8.0;
    let region_x = pad + eb_w + eb_gap;
    let region_w = w - region_x - pad;
    let board_x = region_x + (region_w - board_w) / 2.0;
    if board_x < region_x { board_x = region_x; }
    let region_y = bar_h + pad;
    let region_h = h - bar_h - 2.0 * pad - gr_h - gr_gap;
    let board_y = region_y + (region_h - board_w) / 2.0;
    if board_y < region_y { board_y = region_y; }

    let gap = 8.0;
    let L = #{ bar_h: bar_h, sq: sq, board_x: board_x, board_y: board_y,
        eb_x: board_x - eb_gap - eb_w, eb_w: eb_w, eb_y: board_y, eb_h: board_w,
        gr_x: board_x, gr_y: board_y + board_w + gr_gap, gr_w: board_w, gr_h: gr_h };
    let cursor = w - gap;
    if review {
        let exit_w = 76.0; let step_w = 40.0;
        L.btn_exit_x = cursor - exit_w; L.btn_exit_w = exit_w; cursor = L.btn_exit_x - gap;
        L.btn_next_x = cursor - step_w; L.btn_next_w = step_w; cursor = L.btn_next_x - gap;
        L.btn_prev_x = cursor - step_w; L.btn_prev_w = step_w; cursor = L.btn_prev_x - gap;
    } else {
        let new_w = 84.0; let flip_w = 48.0; let bot_w = 70.0;
        let lvl_w = 52.0; let rev_w = 76.0; let copy_w = 84.0;
        L.btn_new_x = cursor - new_w; L.btn_new_w = new_w; cursor = L.btn_new_x - gap;
        L.btn_flip_x = cursor - flip_w; L.btn_flip_w = flip_w; cursor = L.btn_flip_x - gap;
        L.btn_bot_x = cursor - bot_w; L.btn_bot_w = bot_w; cursor = L.btn_bot_x - gap;
        if bot_on {
            L.btn_lvl_x = cursor - lvl_w; L.btn_lvl_w = lvl_w; cursor = L.btn_lvl_x - gap;
        } else {
            L.btn_lvl_x = -1000.0; L.btn_lvl_w = lvl_w;
        }
        L.btn_review_x = cursor - rev_w; L.btn_review_w = rev_w; cursor = L.btn_review_x - gap;
        L.btn_copy_x = cursor - copy_w; L.btn_copy_w = copy_w; cursor = L.btn_copy_x - gap;
    }
    L
}

// Visual-row/col coordinates for a board square idx, given perspective.
fn vis_row_of(flipped, idx) { if flipped { 7 - idx_r(idx) } else { idx_r(idx) } }
fn vis_col_of(flipped, idx) { if flipped { 7 - idx_c(idx) } else { idx_c(idx) } }

// 4-cell strip for the promotion picker, in visual (row, col) coords.
// Always extends from the destination square toward the board center,
// so the strip stays on-board regardless of perspective.
fn promo_strip_cells(flipped, to_idx) {
    let vr = vis_row_of(flipped, to_idx);
    let vc = vis_col_of(flipped, to_idx);
    let dir = if vr == 0 { 1 } else { -1 };
    [
        #{ vr: vr,             vc: vc },
        #{ vr: vr + dir,       vc: vc },
        #{ vr: vr + 2 * dir,   vc: vc },
        #{ vr: vr + 3 * dir,   vc: vc },
    ]
}

fn promo_pieces() { ["Q", "R", "B", "N"] }

fn on_click(x, y, shift, cmd, id) {
    let L = compute_layout(canvas_w, canvas_h, state.bot_side != "", state.review);

    // Pending-promotion overlay takes precedence over every other
    // click. It draws a 4-cell strip from the destination square; a
    // click inside picks that piece, a click outside cancels.
    if type_of(state.promo) == "map" {
        let cells = promo_strip_cells(state.flipped, state.promo.to);
        let pieces = promo_pieces();
        let bx = x - L.board_x;
        let by = y - L.board_y;
        let inside = bx >= 0.0 && by >= 0.0 && bx < L.sq * 8.0 && by < L.sq * 8.0;
        if inside {
            let cv = (bx / L.sq).to_int();
            let rv = (by / L.sq).to_int();
            let i = 0;
            while i < 4 {
                let cell = cells[i];
                if cell.vr == rv && cell.vc == cv {
                    let from = state.promo.from;
                    let to = state.promo.to;
                    let pick = pieces[i];
                    let res = make_result(state.board, state.castle, state.ep, state.turn, from, to, pick);
                    state.history.push(#{ from: from, to: to, promo: pick,
                        san: full_san(state.board, state.castle, state.ep, state.turn, from, to, pick, res.board, res.turn, res.over, res.status) });
                    state.board = res.board;
                    state.castle = res.castle;
                    state.ep = res.ep;
                    state.last_from = res.last_from;
                    state.last_to = res.last_to;
                    state.turn = res.turn;
                    state.over = res.over;
                    state.status = res.status;
                    state.selected = -1;
                    state.promo = -1;
                    if bot_to_move(state.bot_side, state.over, state.turn) {
                        state.eng = ensure_engine(state.eng);
                        if state.eng == -1 {
                            state.engine_ok = false;
                        } else {
                            kick_engine(state.eng, state.board, state.turn, state.castle, state.ep, state.level_idx);
                            state.thinking = true;
                            set_animating(true);
                        }
                    }
                    request_render();
                    return;
                }
                i += 1;
            }
        }
        // Click outside the strip → cancel the pending promotion.
        state.promo = -1;
        state.selected = -1;
        request_render();
        return;
    }

    if y < L.bar_h {
        // Review-mode bar: stepper + exit. Stepping just moves the
        // marker; the background sweep (on_frame) prioritizes whatever
        // ply is being viewed, and already-evaluated plies show their
        // cached result instantly.
        if state.review {
            if x >= L.btn_prev_x && x <= L.btn_prev_x + L.btn_prev_w {
                let np = state.ply - 1;
                if np < 0 { np = 0; }
                if np != state.ply { state.ply = np; }
                request_render();
                return;
            }
            if x >= L.btn_next_x && x <= L.btn_next_x + L.btn_next_w {
                let np = state.ply + 1;
                if np > state.history.len() { np = state.history.len(); }
                if np != state.ply { state.ply = np; }
                request_render();
                return;
            }
            if x >= L.btn_exit_x && x <= L.btn_exit_x + L.btn_exit_w {
                state.review = false;
                state.sweeping = false;
                state.an_ply = -1;
                state.evals = [];
                set_animating(false);
                if state.eng != -1 {
                    proc_write(state.eng, "stop");
                    loop { if proc_read(state.eng) == "" { break; } }
                }
                request_render();
                return;
            }
            return;
        }
        if x >= L.btn_new_x && x <= L.btn_new_x + L.btn_new_w {
            state.board = new_board();
            state.turn = "w";
            state.selected = -1;
            state.last_from = -1;
            state.last_to = -1;
            state.castle = initial_castle();
            state.ep = -1;
            state.over = false;
            state.status = "";
            state.promo = -1;
            state.thinking = false;
            state.history = [];
            state.copied = false;
            // If the bot plays White it opens the new game.
            if bot_to_move(state.bot_side, state.over, state.turn) {
                state.eng = ensure_engine(state.eng);
                if state.eng == -1 {
                    state.engine_ok = false;
                } else {
                    kick_engine(state.eng, state.board, state.turn, state.castle, state.ep, state.level_idx);
                    state.thinking = true;
                    set_animating(true);
                }
            }
            request_render();
            return;
        }
        if x >= L.btn_flip_x && x <= L.btn_flip_x + L.btn_flip_w {
            state.flipped = !state.flipped;
            state.selected = -1;
            request_render();
            return;
        }
        // Bot button cycles: off -> bot plays Black -> bot plays White -> off.
        if x >= L.btn_bot_x && x <= L.btn_bot_x + L.btn_bot_w {
            let next = if state.bot_side == "" { "b" }
                else if state.bot_side == "b" { "w" }
                else { "" };
            if next == "" {
                // Turning the bot off — free the engine process.
                if state.eng != -1 { proc_kill(state.eng); state.eng = -1; }
                state.bot_side = "";
                state.thinking = false;
                set_animating(false);
                state.selected = -1;
                request_render();
                return;
            }
            // Turning the bot on — make sure an engine is live first.
            state.eng = ensure_engine(state.eng);
            if state.eng == -1 {
                state.engine_ok = false; // no engine binary found; stay off
                request_render();
                return;
            }
            state.engine_ok = true;
            state.bot_side = next;
            state.selected = -1;
            state.thinking = false;
            // Newly the bot's turn? Start thinking right away.
            if bot_to_move(state.bot_side, state.over, state.turn) {
                kick_engine(state.eng, state.board, state.turn, state.castle, state.ep, state.level_idx);
                state.thinking = true;
                set_animating(true);
            }
            request_render();
            return;
        }
        // Level button (only meaningful while a bot is active).
        if state.bot_side != "" && x >= L.btn_lvl_x && x <= L.btn_lvl_x + L.btn_lvl_w {
            state.level_idx = (state.level_idx + 1) % 4;
            request_render();
            return;
        }
        // Review button — enter review at the final position and kick
        // off the background per-ply eval sweep.
        if x >= L.btn_review_x && x <= L.btn_review_x + L.btn_review_w {
            if state.history.len() == 0 { return; } // nothing to review yet
            state.review = true;
            state.thinking = false;
            state.selected = -1;
            state.ply = state.history.len();
            // One eval slot per ply, 0..N (ply 0 = starting position).
            state.evals = [];
            let i = 0;
            while i <= state.history.len() {
                state.evals.push(#{ done: false, kind: "none", wr: 0, from: -1, to: -1, san: "" });
                i += 1;
            }
            state.eng = ensure_engine(state.eng);
            if state.eng == -1 {
                state.engine_ok = false;
                state.sweeping = false;
                state.an_ply = -1;
            } else {
                state.engine_ok = true;
                // Analyze the viewed ply first so its eval/arrow appears
                // immediately; the sweep fills the rest in on_frame.
                state.an_ply = state.ply;
                analyze_ply(state.eng, state.history, state.an_ply, review_movetime());
                state.sweeping = true;
                set_animating(true);
            }
            request_render();
            return;
        }
        // Copy PGN to the system clipboard.
        if x >= L.btn_copy_x && x <= L.btn_copy_x + L.btn_copy_w {
            if state.history.len() > 0 {
                let pgn = build_pgn(state.history, state.over, state.status, state.bot_side, state.level_idx);
                state.copied = clipboard_set(pgn);
            }
            request_render();
            return;
        }
        return;
    }
    // Board is read-only while reviewing — but clicking the eval graph
    // jumps to that ply.
    if state.review {
        if state.history.len() > 0
            && x >= L.gr_x && x <= L.gr_x + L.gr_w
            && y >= L.gr_y && y <= L.gr_y + L.gr_h {
            let n = state.history.len();
            let np = ((x - L.gr_x) / L.gr_w * n.to_float() + 0.5).to_int();
            if np < 0 { np = 0; }
            if np > n { np = n; }
            if np != state.ply { state.ply = np; request_render(); }
        }
        return;
    }
    let bx = x - L.board_x;
    let by = y - L.board_y;
    if bx < 0.0 || by < 0.0 || bx >= L.sq * 8.0 || by >= L.sq * 8.0 { return; }
    let col_v = (bx / L.sq).to_int();
    let row_v = (by / L.sq).to_int();
    let r = if state.flipped { 7 - row_v } else { row_v };
    let c = if state.flipped { 7 - col_v } else { col_v };
    let idx = rc_to_idx(r, c);
    if state.over { return; }
    // It's the bot's move (or it's thinking) — ignore board input.
    if state.bot_side != "" && state.turn == state.bot_side { return; }

    if state.selected != -1 {
        let legal = legal_from(state.board, state.castle, state.ep, state.selected);
        let is_legal = false;
        for m in legal { if m == idx { is_legal = true; break; } }
        if is_legal {
            if is_promotion_move(state.board, state.selected, idx) {
                state.promo = #{ from: state.selected, to: idx };
                request_render();
                return;
            }
            let res = make_result(state.board, state.castle, state.ep, state.turn, state.selected, idx, "");
            state.history.push(#{ from: state.selected, to: idx, promo: "",
                san: full_san(state.board, state.castle, state.ep, state.turn, state.selected, idx, "", res.board, res.turn, res.over, res.status) });
            state.board = res.board;
            state.castle = res.castle;
            state.ep = res.ep;
            state.last_from = res.last_from;
            state.last_to = res.last_to;
            state.turn = res.turn;
            state.over = res.over;
            state.status = res.status;
            state.selected = -1;
            if bot_to_move(state.bot_side, state.over, state.turn) {
                state.eng = ensure_engine(state.eng);
                if state.eng == -1 {
                    state.engine_ok = false;
                } else {
                    kick_engine(state.eng, state.board, state.turn, state.castle, state.ep, state.level_idx);
                    state.thinking = true;
                    set_animating(true);
                }
            }
            request_render();
            return;
        }
    }
    let p = state.board[idx];
    if p != "" && pc(p) == state.turn {
        state.selected = idx;
        state.drag_from = idx;
        state.drag_active = false;
        state.drag_x = x;
        state.drag_y = y;
    } else {
        state.selected = -1;
        state.drag_from = -1;
    }
    request_render();
}

fn on_drag(x, y) {
    if state.drag_from == -1 { return; }
    state.drag_active = true;
    state.drag_x = x;
    state.drag_y = y;
    request_render();
}

fn on_release(x, y) {
    if state.drag_from == -1 { return; }
    if !state.drag_active {
        // Pure click — leave selection in place so click-to-select +
        // click-to-move still works. Just reset drag tracking.
        state.drag_from = -1;
        return;
    }
    let L = compute_layout(canvas_w, canvas_h, state.bot_side != "", state.review);
    let from = state.drag_from;
    state.drag_from = -1;
    state.drag_active = false;

    // Promotion picker open? Treat release like a click on the picker.
    if type_of(state.promo) == "map" {
        on_click(x, y, false, false, "");
        return;
    }

    let bx = x - L.board_x;
    let by = y - L.board_y;
    let inside = bx >= 0.0 && by >= 0.0 && bx < L.sq * 8.0 && by < L.sq * 8.0;
    if !inside {
        request_render();
        return;
    }
    let col_v = (bx / L.sq).to_int();
    let row_v = (by / L.sq).to_int();
    let r = if state.flipped { 7 - row_v } else { row_v };
    let c = if state.flipped { 7 - col_v } else { col_v };
    let to = rc_to_idx(r, c);
    if to == from {
        request_render();
        return;
    }
    if state.over { request_render(); return; }

    let legal = legal_from(state.board, state.castle, state.ep, from);
    let is_legal = false;
    for m in legal { if m == to { is_legal = true; break; } }
    if !is_legal {
        request_render();
        return;
    }
    if is_promotion_move(state.board, from, to) {
        state.promo = #{ from: from, to: to };
        request_render();
        return;
    }
    let res = make_result(state.board, state.castle, state.ep, state.turn, from, to, "");
    state.history.push(#{ from: from, to: to, promo: "",
        san: full_san(state.board, state.castle, state.ep, state.turn, from, to, "", res.board, res.turn, res.over, res.status) });
    state.board = res.board;
    state.castle = res.castle;
    state.ep = res.ep;
    state.last_from = res.last_from;
    state.last_to = res.last_to;
    state.turn = res.turn;
    state.over = res.over;
    state.status = res.status;
    state.selected = -1;
    if bot_to_move(state.bot_side, state.over, state.turn) {
        state.eng = ensure_engine(state.eng);
        if state.eng == -1 {
            state.engine_ok = false;
        } else {
            kick_engine(state.eng, state.board, state.turn, state.castle, state.ep, state.level_idx);
            state.thinking = true;
            set_animating(true);
        }
    }
    request_render();
}

// Animation tick — only sent while we've asked for it (during a bot
// search). Drain the engine's stdout; when a `bestmove` that's legal
// for the current position lands, apply it and stop animating. Stale
// replies (illegal for the live board, e.g. from an aborted search)
// are skipped, so we keep waiting for the right one.
fn on_frame(dt) {
    // Review-mode background sweep: evaluate one ply at a time (full
    // strength), storing white-relative score + best move per ply.
    // Prioritizes the ply being viewed so stepping feels responsive.
    if state.sweeping {
        if state.eng == -1 || state.an_ply < 0 {
            state.sweeping = false; state.an_ply = -1; set_animating(false); return;
        }
        let n = state.history.len();
        let cur = state.evals[state.an_ply];
        let kind = cur.kind;
        let wr = cur.wr;
        let bf = cur.from;
        let bt = cur.to;
        let bsan = cur.san;
        let v = replay_to(state.history, state.an_ply);
        let got_best = false;
        loop {
            let line = proc_read(state.eng);
            if line == "" { break; }
            if line.starts_with("info ") && line.contains(" score ") {
                let sc = parse_info_score(line);
                if sc.kind != "none" {
                    kind = sc.kind;
                    wr = if v.turn == "b" { -sc.val } else { sc.val }; // white-relative
                }
            } else if line.starts_with("bestmove ") {
                let rest = line.sub_string(9);
                let sp = rest.index_of(" ");
                let mv = if sp == -1 { rest } else { rest.sub_string(0, sp) };
                let m = parse_engine_move(mv);
                if m.from != -1 && m.to != -1 {
                    bf = m.from; bt = m.to;
                    bsan = move_san_core(v.board, v.castle, v.ep, v.turn, m.from, m.to, m.promo);
                }
                got_best = true;
                break;
            }
        }
        state.evals[state.an_ply] = #{ done: got_best, kind: kind, wr: wr, from: bf, to: bt, san: bsan };
        if got_best {
            // Next ply: the viewed one if still pending, else the lowest
            // pending index; -1 means the whole game is evaluated.
            let nxt = -1;
            if !state.evals[state.ply].done { nxt = state.ply; }
            else {
                let i = 0;
                while i <= n { if !state.evals[i].done { nxt = i; break; } i += 1; }
            }
            if nxt == -1 {
                state.sweeping = false; state.an_ply = -1; set_animating(false);
            } else {
                state.an_ply = nxt;
                analyze_ply(state.eng, state.history, nxt, review_movetime());
            }
        }
        request_render();
        return;
    }
    if !state.thinking { set_animating(false); return; }
    if state.eng == -1 { state.thinking = false; set_animating(false); return; }

    let done = false;
    loop {
        let line = proc_read(state.eng);
        if line == "" { break; } // nothing more buffered this tick

        if line.starts_with("bestmove ") {
            let rest = line.sub_string(9);
            let sp = rest.index_of(" ");
            let mv = if sp == -1 { rest } else { rest.sub_string(0, sp) };
            let m = parse_engine_move(mv);
            if m.from == -1 || m.to == -1 {
                // "(none)" / "0000" — engine has no move. Our own
                // mate/stalemate detection already handled state.over.
                done = true;
                break;
            }
            // Guard against a stale reply for a different position.
            let legal = legal_from(state.board, state.castle, state.ep, m.from);
            let ok = false;
            for mm in legal { if mm == m.to { ok = true; break; } }
            if !ok { continue; } // skip, keep reading / waiting

            let res = make_result(state.board, state.castle, state.ep, state.turn, m.from, m.to, m.promo);
            state.history.push(#{ from: m.from, to: m.to, promo: m.promo,
                san: full_san(state.board, state.castle, state.ep, state.turn, m.from, m.to, m.promo, res.board, res.turn, res.over, res.status) });
            state.board = res.board;
            state.castle = res.castle;
            state.ep = res.ep;
            state.last_from = res.last_from;
            state.last_to = res.last_to;
            state.turn = res.turn;
            state.over = res.over;
            state.status = res.status;
            state.selected = -1;
            done = true;
            break;
        }
    }
    if done {
        state.thinking = false;
        set_animating(false);
        request_render();
    }
}

fn render(cw, ch) {
    let L = compute_layout(cw, ch, state.bot_side != "", state.review);
    let items = [];

    items.push(#{ type: "rect", id: "bg",
        x: 0.0, y: 0.0, w: cw, h: ch,
        color: "#222831", anchor: "top-left", z: 0.0 });

    items.push(#{ type: "rect", id: "bar",
        x: 0.0, y: 0.0, w: cw, h: L.bar_h,
        color: "#2b323d", anchor: "top-left", z: 0.1 });

    // Position to draw: the live game, or the reviewed past position.
    let view = if state.review {
        replay_to(state.history, state.ply)
    } else {
        #{ board: state.board, castle: state.castle, ep: state.ep, turn: state.turn,
           last_from: state.last_from, last_to: state.last_to }
    };
    let vb = view.board;
    let vturn = view.turn;
    let vlf = view.last_from;
    let vlt = view.last_to;
    let vcastle = view.castle;
    let vep = view.ep;
    let vsel = if state.review { -1 } else { state.selected };

    // Current-ply analysis (review): eval + best move from the cache.
    let cur = if state.review && state.ply < state.evals.len() {
        state.evals[state.ply]
    } else {
        #{ done: false, kind: "none", wr: 0, from: -1, to: -1, san: "" }
    };
    let cur_from = cur.from;
    let cur_to = cur.to;
    let cur_san = cur.san;
    let cur_eval = if cur.kind != "none" { eval_display(cur.kind, cur.wr, "w") }
        else if state.sweeping { "…" } else { "" };

    let dot_border = if vturn == "w" { "#1a1a1a" } else { "#f0d9b5" };
    items.push(#{ type: "rect", id: "turn_b",
        x: 9.0, y: 7.0, w: 18.0, h: 18.0,
        color: dot_border, anchor: "top-left", z: 0.15 });
    let dot_color = if vturn == "w" { "#f0d9b5" } else { "#1a1a1a" };
    items.push(#{ type: "rect", id: "turn",
        x: 10.0, y: 8.0, w: 16.0, h: 16.0,
        color: dot_color, anchor: "top-left", z: 0.2 });
    let turn_label = if !state.engine_ok { "No UCI engine found" }
        else if state.review { "Move " + state.ply.to_string() + "/" + state.history.len().to_string() }
        else if state.over { state.status }
        else if state.thinking { "Bot thinking…" }
        else if vturn == "w" { "White to move" }
        else { "Black to move" };
    // Top-left anchor + computed Y so the text sits on the bar
    // baseline. Bevy 0.18's Text2d center-anchor lands the glyphs
    // outside short button rects, so every label here uses top-left.
    items.push(#{ type: "text", id: "turn_lbl",
        x: 36.0, y: 8.0,
        value: turn_label,
        color: "#e6e8ef", size: 13.0,
        anchor: "top-left", z: 0.3 });

    // Material lead (live + review), shown beside the turn indicator.
    let madv = material_adv(vb);
    if madv != 0 {
        let mtxt = if madv > 0 { "White +" + madv.to_string() }
            else { "Black +" + (-madv).to_string() };
        items.push(#{ type: "text", id: "mat_lbl",
            x: 160.0, y: 8.0, value: mtxt,
            color: "#d8c38a", size: 12.0, anchor: "top-left", z: 0.3 });
    }

    // Review: eval + engine's best move (further right, past material).
    if state.review {
        let info = "";
        if cur_eval != "" && cur_eval != "…" { info += "eval " + cur_eval; }
        if cur_san != "" { info += "    best " + cur_san; }
        else if state.sweeping { info += "    analyzing…"; }
        items.push(#{ type: "text", id: "rev_info",
            x: 260.0, y: 8.0, value: info,
            color: "#bfe3c4", size: 12.0, anchor: "top-left", z: 0.3 });
    }

    // Mono-glyph width ≈ 0.6 * font_size, used here to center button
    // labels manually. Off by a couple px is fine; the buttons leave
    // 30+ px of horizontal slack around the centered text.
    if state.review {
        // Stepper: < (prev)  > (next)  Exit
        let prev_bg = if state.hover_btn == "prev" { "#4a546a" } else { "#3a4252" };
        items.push(#{ type: "rect", id: "btn_prev",
            x: L.btn_prev_x, y: 4.0, w: L.btn_prev_w, h: L.bar_h - 8.0,
            color: prev_bg, anchor: "top-left", z: 0.2 });
        items.push(#{ type: "text", id: "btn_prev_lbl",
            x: L.btn_prev_x + L.btn_prev_w / 2.0 - 4.0, y: 7.0,
            value: "<", color: "#e6e8ef", size: 15.0, anchor: "top-left", z: 0.3 });
        let next_bg = if state.hover_btn == "next" { "#4a546a" } else { "#3a4252" };
        items.push(#{ type: "rect", id: "btn_next",
            x: L.btn_next_x, y: 4.0, w: L.btn_next_w, h: L.bar_h - 8.0,
            color: next_bg, anchor: "top-left", z: 0.2 });
        items.push(#{ type: "text", id: "btn_next_lbl",
            x: L.btn_next_x + L.btn_next_w / 2.0 - 4.0, y: 7.0,
            value: ">", color: "#e6e8ef", size: 15.0, anchor: "top-left", z: 0.3 });
        let exit_bg = if state.hover_btn == "exit" { "#7a5a5a" } else { "#5a4646" };
        let exit_tw = 4.0 * 12.0 * 0.6;
        items.push(#{ type: "rect", id: "btn_exit",
            x: L.btn_exit_x, y: 4.0, w: L.btn_exit_w, h: L.bar_h - 8.0,
            color: exit_bg, anchor: "top-left", z: 0.2 });
        items.push(#{ type: "text", id: "btn_exit_lbl",
            x: L.btn_exit_x + (L.btn_exit_w - exit_tw) / 2.0, y: 9.0,
            value: "Exit", color: "#f4f4f4", size: 12.0, anchor: "top-left", z: 0.3 });
    } else {
        let flip_text_w = 4.0 * 12.0 * 0.6;
        let flip_bg = if state.hover_btn == "flip" { "#4a546a" } else { "#3a4252" };
        items.push(#{ type: "rect", id: "btn_flip",
            x: L.btn_flip_x, y: 4.0, w: L.btn_flip_w, h: L.bar_h - 8.0,
            color: flip_bg, anchor: "top-left", z: 0.2 });
        items.push(#{ type: "text", id: "btn_flip_lbl",
            x: L.btn_flip_x + (L.btn_flip_w - flip_text_w) / 2.0,
            y: 9.0,
            value: "Flip", color: "#e6e8ef", size: 12.0,
            anchor: "top-left", z: 0.3 });

        let new_text_w = 8.0 * 12.0 * 0.6;
        let new_base = if state.over { "#c0703e" } else { "#5a4646" };
        let new_hover = if state.over { "#d68d52" } else { "#7a5a5a" };
        let new_color = if state.hover_btn == "new" { new_hover } else { new_base };
        items.push(#{ type: "rect", id: "btn_new",
            x: L.btn_new_x, y: 4.0, w: L.btn_new_w, h: L.bar_h - 8.0,
            color: new_color, anchor: "top-left", z: 0.2 });
        items.push(#{ type: "text", id: "btn_new_lbl",
            x: L.btn_new_x + (L.btn_new_w - new_text_w) / 2.0,
            y: 9.0,
            value: "New game", color: "#f4f4f4", size: 12.0,
            anchor: "top-left", z: 0.3 });

        // Bot toggle: off / plays Black / plays White. Tinted green when on.
        let bot_label = if state.bot_side == "w" { "Bot: W" }
            else if state.bot_side == "b" { "Bot: B" }
            else { "Bot off" };
        let bot_on = state.bot_side != "";
        let bot_base = if bot_on { "#3d6b46" } else { "#3a4252" };
        let bot_hov = if bot_on { "#4c8358" } else { "#4a546a" };
        let bot_color = if state.hover_btn == "bot" { bot_hov } else { bot_base };
        items.push(#{ type: "rect", id: "btn_bot",
            x: L.btn_bot_x, y: 4.0, w: L.btn_bot_w, h: L.bar_h - 8.0,
            color: bot_color, anchor: "top-left", z: 0.2 });
        let bot_text_w = 6.0 * 12.0 * 0.6;
        items.push(#{ type: "text", id: "btn_bot_lbl",
            x: L.btn_bot_x + (L.btn_bot_w - bot_text_w) / 2.0,
            y: 9.0,
            value: bot_label, color: "#f4f4f4", size: 12.0,
            anchor: "top-left", z: 0.3 });

        // Level button — only shown while a bot is active.
        if bot_on {
            let lvl_color = if state.hover_btn == "level" { "#4a546a" } else { "#3a4252" };
            items.push(#{ type: "rect", id: "btn_lvl",
                x: L.btn_lvl_x, y: 4.0, w: L.btn_lvl_w, h: L.bar_h - 8.0,
                color: lvl_color, anchor: "top-left", z: 0.2 });
            let lvl_text_w = 4.0 * 12.0 * 0.6;
            items.push(#{ type: "text", id: "btn_lvl_lbl",
                x: L.btn_lvl_x + (L.btn_lvl_w - lvl_text_w) / 2.0,
                y: 9.0,
                value: level_label(state.level_idx), color: "#e6e8ef", size: 12.0,
                anchor: "top-left", z: 0.3 });
        }

        // Review button.
        let rev_tw = 6.0 * 12.0 * 0.6;
        let rev_bg = if state.hover_btn == "review" { "#4a546a" } else { "#3a4252" };
        items.push(#{ type: "rect", id: "btn_review",
            x: L.btn_review_x, y: 4.0, w: L.btn_review_w, h: L.bar_h - 8.0,
            color: rev_bg, anchor: "top-left", z: 0.2 });
        items.push(#{ type: "text", id: "btn_review_lbl",
            x: L.btn_review_x + (L.btn_review_w - rev_tw) / 2.0, y: 9.0,
            value: "Review", color: "#e6e8ef", size: 12.0, anchor: "top-left", z: 0.3 });

        // Copy PGN button (flashes "Copied!" after a successful copy).
        let copy_label = if state.copied { "Copied!" } else { "Copy PGN" };
        let copy_tw = copy_label.len().to_float() * 12.0 * 0.6;
        let copy_bg = if state.copied { "#3d6b46" }
            else if state.hover_btn == "copy" { "#4a546a" } else { "#3a4252" };
        items.push(#{ type: "rect", id: "btn_copy",
            x: L.btn_copy_x, y: 4.0, w: L.btn_copy_w, h: L.bar_h - 8.0,
            color: copy_bg, anchor: "top-left", z: 0.2 });
        items.push(#{ type: "text", id: "btn_copy_lbl",
            x: L.btn_copy_x + (L.btn_copy_w - copy_tw) / 2.0, y: 9.0,
            value: copy_label, color: "#f4f4f4", size: 12.0, anchor: "top-left", z: 0.3 });
    }

    let legal = if vsel != -1 {
        legal_from(vb, vcastle, vep, vsel)
    } else { [] };

    let in_check = false;
    let check_sq = -1;
    let opp = if vturn == "w" { "b" } else { "w" };
    let ksq = king_sq(vb, vturn);
    if ksq != -1 && is_attacked(vb, ksq, opp) {
        in_check = true; check_sq = ksq;
    }

    let row = 0;
    while row < 8 {
        let col = 0;
        while col < 8 {
            let dr = if state.flipped { 7 - row } else { row };
            let dc = if state.flipped { 7 - col } else { col };
            let idx = rc_to_idx(dr, dc);
            let light = (row + col) % 2 == 0;
            let base = if light { "#f0d9b5" } else { "#b58863" };
            let sx = L.board_x + col.to_float() * L.sq;
            let sy = L.board_y + row.to_float() * L.sq;
            items.push(#{ type: "rect", id: `sq_${idx}`,
                x: sx, y: sy, w: L.sq, h: L.sq,
                color: base, anchor: "top-left", z: 1.0 });

            if idx == vlf || idx == vlt {
                items.push(#{ type: "rect", id: `lm_${idx}`,
                    x: sx, y: sy, w: L.sq, h: L.sq,
                    color: "#cdd26b55", anchor: "top-left", z: 1.1 });
            }
            if idx == vsel {
                items.push(#{ type: "rect", id: `sel_${idx}`,
                    x: sx, y: sy, w: L.sq, h: L.sq,
                    color: "#f6f66988", anchor: "top-left", z: 1.2 });
            }
            if in_check && idx == check_sq {
                items.push(#{ type: "rect", id: `chk_${idx}`,
                    x: sx, y: sy, w: L.sq, h: L.sq,
                    color: "#ff3a3a66", anchor: "top-left", z: 1.2 });
            }
            // Review: tint the best-move from/to squares green.
            if state.review && (idx == cur_from || idx == cur_to) {
                items.push(#{ type: "rect", id: `bm_${idx}`,
                    x: sx, y: sy, w: L.sq, h: L.sq,
                    color: "#3fb95033", anchor: "top-left", z: 1.15 });
            }
            let is_legal = false;
            for m in legal { if m == idx { is_legal = true; break; } }
            if is_legal {
                let has_piece = vb[idx] != "";
                let dot_r = if has_piece { L.sq * 0.46 } else { L.sq * 0.16 };
                items.push(#{ type: "rect", id: `lg_${idx}`,
                    x: sx + L.sq / 2.0 - dot_r,
                    y: sy + L.sq / 2.0 - dot_r,
                    w: dot_r * 2.0, h: dot_r * 2.0,
                    color: "#22222255", anchor: "top-left", z: 1.3 });
            }
            col += 1;
        }
        row += 1;
    }

    let lifted = state.drag_from != -1 && state.drag_active;
    let row2 = 0;
    while row2 < 8 {
        let col2 = 0;
        while col2 < 8 {
            let dr = if state.flipped { 7 - row2 } else { row2 };
            let dc = if state.flipped { 7 - col2 } else { col2 };
            let idx = rc_to_idx(dr, dc);
            let p = vb[idx];
            if p != "" && !(lifted && idx == state.drag_from) {
                items.push(#{
                    type: "sprite",
                    id: `pc_${idx}`,
                    x: L.board_x + col2.to_float() * L.sq + L.sq / 2.0,
                    y: L.board_y + row2.to_float() * L.sq + L.sq / 2.0,
                    w: L.sq * 0.92,
                    h: L.sq * 0.92,
                    image: "path",
                    path: piece_path(p),
                    anchor: "center",
                    z: 2.0,
                });
            }
            col2 += 1;
        }
        row2 += 1;
    }

    if lifted {
        let p = vb[state.drag_from];
        if p != "" {
            items.push(#{
                type: "sprite",
                id: "pc_lifted",
                x: state.drag_x,
                y: state.drag_y,
                w: L.sq * 1.05,
                h: L.sq * 1.05,
                image: "path",
                path: piece_path(p),
                anchor: "center",
                z: 5.0,
            });
        }
    }

    // Best-move arrow (review mode): a rotated-rect shaft + V head.
    if state.review && cur_from != -1 && cur_to != -1 {
        let p0 = sq_center(L, state.flipped, cur_from);
        let p1 = sq_center(L, state.flipped, cur_to);
        let dx = p1.x - p0.x;
        let dy = p1.y - p0.y;
        let dist = sqrt(dx * dx + dy * dy);
        if dist > 1.0 {
            let ang = atan2_deg(dy, dx);
            let pi = pi_const();
            let ux = dx / dist;
            let uy = dy / dist;
            let head_len = L.sq * 0.38;
            let thick = L.sq * 0.15;
            let col = "#2fa84fdd";
            let shaft_len = dist - head_len * 0.6;
            if shaft_len < 2.0 { shaft_len = 2.0; }
            let scx = p0.x + ux * shaft_len / 2.0;
            let scy = p0.y + uy * shaft_len / 2.0;
            items.push(#{ type: "rect", id: "bm_shaft",
                x: scx, y: scy, w: shaft_len, h: thick,
                color: col, anchor: "center", z: 3.5, rotation: ang });
            let back = ang + 180.0;
            let spread = 32.0;
            let a1 = (back - spread) * pi / 180.0;
            let a2 = (back + spread) * pi / 180.0;
            items.push(#{ type: "rect", id: "bm_h1",
                x: p1.x + cos(a1) * head_len / 2.0, y: p1.y + sin(a1) * head_len / 2.0,
                w: head_len, h: thick, color: col, anchor: "center", z: 3.6,
                rotation: back - spread });
            items.push(#{ type: "rect", id: "bm_h2",
                x: p1.x + cos(a2) * head_len / 2.0, y: p1.y + sin(a2) * head_len / 2.0,
                w: head_len, h: thick, color: col, anchor: "center", z: 3.6,
                rotation: back + spread });
        }
    }

    // Eval bar (left of board) + eval graph (below board), review only.
    if state.review {
        // --- Eval bar: white fill = White's win probability ---
        let wp = win_prob(cur.kind, cur.wr);
        items.push(#{ type: "rect", id: "eb_bg",
            x: L.eb_x, y: L.eb_y, w: L.eb_w, h: L.eb_h,
            color: "#23272d", anchor: "top-left", z: 1.0 });
        let wh = L.eb_h * wp;
        let wy = if state.flipped { L.eb_y } else { L.eb_y + L.eb_h - wh };
        items.push(#{ type: "rect", id: "eb_white",
            x: L.eb_x, y: wy, w: L.eb_w, h: wh,
            color: "#e8e8e8", anchor: "top-left", z: 1.05 });
        items.push(#{ type: "rect", id: "eb_mid",
            x: L.eb_x, y: L.eb_y + L.eb_h / 2.0 - 0.5, w: L.eb_w, h: 1.0,
            color: "#8892a0", anchor: "top-left", z: 1.1 });

        // --- Eval graph: area chart of White win prob over plies ---
        let n = state.history.len();
        items.push(#{ type: "rect", id: "gr_bg",
            x: L.gr_x, y: L.gr_y, w: L.gr_w, h: L.gr_h,
            color: "#1b1f24", anchor: "top-left", z: 1.0 });
        let mid = L.gr_y + L.gr_h / 2.0;
        items.push(#{ type: "rect", id: "gr_mid",
            x: L.gr_x, y: mid - 0.5, w: L.gr_w, h: 1.0,
            color: "#3a424d", anchor: "top-left", z: 1.05 });
        if n > 0 {
            let colw = L.gr_w / (n + 1).to_float();
            let i = 0;
            while i <= n {
                let e = state.evals[i];
                if e.kind != "none" {
                    let p = win_prob(e.kind, e.wr);
                    let dev = p - 0.5;
                    let colx = L.gr_x + i.to_float() * colw;
                    let barh = (if dev < 0.0 { -dev } else { dev }) * L.gr_h;
                    if dev >= 0.0 {
                        items.push(#{ type: "rect", id: `gc_${i}`,
                            x: colx, y: mid - barh, w: colw + 0.6, h: barh,
                            color: "#cfd6dd", anchor: "top-left", z: 1.06 });
                    } else {
                        items.push(#{ type: "rect", id: `gc_${i}`,
                            x: colx, y: mid, w: colw + 0.6, h: barh,
                            color: "#454d57", anchor: "top-left", z: 1.06 });
                    }
                }
                i += 1;
            }
            let mx = L.gr_x + state.ply.to_float() * colw;
            items.push(#{ type: "rect", id: "gr_cursor",
                x: mx - 1.0, y: L.gr_y, w: 2.0, h: L.gr_h,
                color: "#f0c050dd", anchor: "top-left", z: 1.2 });
        }
    }

    if type_of(state.promo) == "map" {
        let cells = promo_strip_cells(state.flipped, state.promo.to);
        let pieces = promo_pieces();
        items.push(#{ type: "rect", id: "promo_dim",
            x: L.board_x, y: L.board_y,
            w: L.sq * 8.0, h: L.sq * 8.0,
            color: "#0a0c1099", anchor: "top-left", z: 4.0 });
        let i = 0;
        while i < 4 {
            let cell = cells[i];
            let sx = L.board_x + cell.vc.to_float() * L.sq;
            let sy = L.board_y + cell.vr.to_float() * L.sq;
            items.push(#{ type: "rect", id: `promo_bg_${i}`,
                x: sx, y: sy, w: L.sq, h: L.sq,
                color: "#dfe2ea", anchor: "top-left", z: 4.1 });
            items.push(#{ type: "rect", id: `promo_border_${i}`,
                x: sx, y: sy, w: L.sq, h: 3.0,
                color: "#7c8da6", anchor: "top-left", z: 4.15 });
            items.push(#{
                type: "sprite",
                id: `promo_pc_${i}`,
                x: sx + L.sq / 2.0,
                y: sy + L.sq / 2.0,
                w: L.sq * 0.92, h: L.sq * 0.92,
                image: "path",
                path: piece_path(state.turn + pieces[i]),
                anchor: "center",
                z: 4.2,
            });
        i += 1;
        }
    }

    let files = ["a","b","c","d","e","f","g","h"];
    let lbl_size = if L.sq >= 36.0 { 11.0 } else { 9.0 };
    let inset = if L.sq >= 36.0 { 3.0 } else { 2.0 };
    let i = 0;
    while i < 8 {
        let file_idx = if state.flipped { 7 - i } else { i };
        let rank_idx = if state.flipped { i + 1 } else { 8 - i };
        let file_col = if (7 + i) % 2 == 0 { "#74522f" } else { "#dcbb84" };
        items.push(#{ type: "text", id: `flbl_${i}`,
            x: L.board_x + i.to_float() * L.sq + L.sq - inset,
            y: L.board_y + 8.0 * L.sq - inset,
            value: files[file_idx],
            color: file_col, size: lbl_size,
            anchor: "top-left", z: 3.0 });
        let rank_col = if i % 2 == 0 { "#74522f" } else { "#dcbb84" };
        items.push(#{ type: "text", id: `rlbl_${i}`,
            x: L.board_x + inset,
            y: L.board_y + i.to_float() * L.sq + inset,
            value: rank_idx.to_string(),
            color: rank_col, size: lbl_size,
            anchor: "top-left", z: 3.0 });
        i += 1;
    }

    #{ type: "canvas", children: items }
}
