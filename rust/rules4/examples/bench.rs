/// Benchmark: measures the actual hot path that's slow —
/// typing keystrokes in the TodoMVC input after a todo exists.
///
/// Run with: cargo run --release --example bench

use std::time::Instant;
use rules4::*;

const TODOMVC_PRELUDE: &str = r##"
# ── DOM helpers ──
fn div(?a, ?c) = element("div", ?a, ?c)
fn span(?a, ?c) = element("span", ?a, ?c)
fn section(?a, ?c) = element("section", ?a, ?c)
fn header(?a, ?c) = element("header", ?a, ?c)
fn footer(?a, ?c) = element("footer", ?a, ?c)
fn h1(?a, ?c) = element("h1", ?a, ?c)
fn ul(?a, ?c) = element("ul", ?a, ?c)
fn li(?a, ?c) = element("li", ?a, ?c)
fn label(?a, ?c) = element("label", ?a, ?c)
fn button(?a, ?c) = element("button", ?a, ?c)
fn input_el(?a) = element("input", ?a, [])
fn a_el(?a, ?c) = element("a", ?a, ?c)
fn strong(?c) = element("strong", [], ?c)

fn cls(?v) = attr("class", ?v)
fn on(?ev, ?handler) = attr("on-" ++ ?ev, ?handler)
fn placeholder(?v) = attr("placeholder", ?v)
fn type_attr(?v) = attr("type", ?v)
fn href(?v) = attr("href", ?v)
fn value(?v) = attr("value", ?v)
fn checked_attr() = attr("checked", "true")
fn autofocus() = attr("autofocus", "true")
fn for_attr(?v) = attr("for", ?v)

fn len(nil) = 0
fn len(cons(?h, ?t)) = 1 + len(?t)
fn map(?f, nil) = nil
fn map(?f, cons(?h, ?t)) = cons(?f(?h), map(?f, ?t))
fn filter_list(?p, nil) = nil
fn filter_list(?p, cons(?h, ?t)) = if ?p(?h) then cons(?h, filter_list(?p, ?t)) else filter_list(?p, ?t)

fn all_todos() = query_all(entry)
fn active_todos() = filter_list(is_active, all_todos())
fn completed_todos() = filter_list(is_completed, all_todos())
fn items_left() = len(active_todos())

fn is_active(entry(?id, ?body, false)) = true
fn is_active(entry(?id, ?body, true)) = false
fn is_completed(entry(?id, ?body, true)) = true
fn is_completed(entry(?id, ?body, false)) = false

fn visible_todos() = if filter == all then all_todos() else if filter == active then active_todos() else completed_todos()
fn has_todos() = if len(all_todos()) > 0 then true else false
fn has_completed() = if len(completed_todos()) > 0 then true else false
fn all_completed() = items_left() == 0

fn handle_event(add_todo, keydown("Enter")) = if input_value == "" then 0 else {
  rule(todo(next_id), entry(next_id, input_value, false))
  rule(next_id, next_id + 1)
  rule(input_value, "")
}
fn handle_event(add_todo, keydown(?k)) = 0
fn handle_event(update_input, input_event(?val)) = rule(input_value, ?val)
fn handle_event(toggle(?id), ?ev) = toggle_entry(?id)
fn toggle_entry(?id) = toggle_with(?id, todo(?id))
fn toggle_with(?id, entry(?i, ?body, true)) = rule(todo(?id), entry(?i, ?body, false))
fn toggle_with(?id, entry(?i, ?body, false)) = rule(todo(?id), entry(?i, ?body, true))
fn handle_event(delete(?id), ?ev) = retract(todo(?id))
fn handle_event(set_filter(?f), ?ev) = rule(filter, ?f)
fn handle_event(clear_completed, ?ev) = clear_each(completed_ids(all_todos()))
fn completed_ids(nil) = nil
fn completed_ids(cons(entry(?id, ?body, true), ?t)) = cons(?id, completed_ids(?t))
fn completed_ids(cons(entry(?id, ?body, false), ?t)) = completed_ids(?t)
fn clear_each(nil) = 0
fn clear_each(cons(?id, ?rest)) = { retract(todo(?id)) clear_each(?rest) }
fn handle_event(toggle_all, ?ev) = if all_completed() then set_all(false, all_todos()) else set_all(true, all_todos())
fn set_all(?done, nil) = 0
fn set_all(?done, cons(entry(?id, ?body, ?old), ?rest)) = { rule(todo(?id), entry(?id, ?body, ?done)) set_all(?done, ?rest) }

fn render() = @dom if has_todos() then render_app() else render_empty()
fn render_empty() = section([cls("todoapp")], [render_header()])
fn render_app() = section([cls("todoapp")],
  [render_header(), render_main(), render_footer()])
fn render_header() = header([cls("header")],
  [h1([], [text("todos")]),
   input_el([cls("new-todo"), placeholder("What needs to be done?"),
             on("keydown", add_todo), on("input", update_input),
             value(input_value), autofocus()])])
fn render_main() = section([cls("main")],
  [input_el([attr("id", "toggle-all"), cls("toggle-all"),
             type_attr("checkbox"), on("click", toggle_all)]),
   label([for_attr("toggle-all")], [text("Mark all as complete")]),
   ul([cls("todo-list")], map(render_todo_item, visible_todos()))])
fn render_todo_item(entry(?id, ?body, ?done)) = li([cls(todo_item_class(?done))],
  [div([cls("view")],
    [render_checkbox(?id, ?done),
     label([], [text(?body)]),
     button([cls("destroy"), on("click", delete(?id))], [])])])
fn render_checkbox(?id, true) = input_el(
  [cls("toggle"), type_attr("checkbox"), checked_attr(), on("click", toggle(?id))])
fn render_checkbox(?id, false) = input_el(
  [cls("toggle"), type_attr("checkbox"), on("click", toggle(?id))])
fn todo_item_class(true) = "completed"
fn todo_item_class(false) = ""
fn render_footer() = footer([cls("footer")],
  [span([cls("todo-count")],
     [strong([text(items_left())]), text(items_left_text())]),
   ul([cls("filters")],
     [render_filter("All", all),
      render_filter("Active", active),
      render_filter("Completed", completed)]),
   render_clear_button()])
fn items_left_text() = if items_left() == 1 then " item left" else " items left"
fn render_filter(?label, ?value) = li([],
  [a_el([cls(filter_class(?value)), href("#"), on("click", set_filter(?value))],
     [text(?label)])])
fn filter_class(?f) = if filter == ?f then "selected" else ""
fn render_clear_button() = if has_completed()
  then button([cls("clear-completed"), on("click", clear_completed)],
    [text("Clear completed")])
  else text("")

fn init() = {
  rule(next_id, 1)
  rule(filter, all)
  rule(input_value, "")
}

init()
"##;

fn fresh_engine() -> Engine {
    let mut store = TermStore::new();
    let tokens = Lexer::new(TODOMVC_PRELUDE).tokenize();
    let program = Parser::new(tokens, &mut store).parse_program();
    let term = pattern_to_term(&mut store, &program.expr);
    let mut engine = Engine::new(store, program.rules, program.meta_rules);
    engine.eval(term);
    engine
}

fn simulate_add_todo(engine: &mut Engine, text: &str) {
    // handle_event(update_input, input_event("text"))
    let he = engine.make_sym("handle_event");
    let ui = engine.make_sym("update_input");
    let ie_sym = engine.make_sym("input_event");
    let val = engine.make_sym(text);
    let ie = engine.make_call(ie_sym, &[val]);
    let call = engine.make_call(he, &[ui, ie]);
    engine.eval(call);

    // handle_event(add_todo, keydown("Enter"))
    let at = engine.make_sym("add_todo");
    let kd_sym = engine.make_sym("keydown");
    let enter = engine.make_sym("Enter");
    let kd = engine.make_call(kd_sym, &[enter]);
    let call = engine.make_call(he, &[at, kd]);
    engine.eval(call);
}

fn simulate_render(engine: &mut Engine) -> TermId {
    let render_sym = engine.make_sym("render");
    let call = engine.make_call(render_sym, &[]);
    engine.eval(call)
}

fn simulate_keystroke(engine: &mut Engine, text: &str) {
    let he = engine.make_sym("handle_event");
    let ui = engine.make_sym("update_input");
    let ie_sym = engine.make_sym("input_event");
    let val = engine.make_sym(text);
    let ie = engine.make_call(ie_sym, &[val]);
    let call = engine.make_call(he, &[ui, ie]);
    engine.eval(call);
}

fn bench<F: FnMut()>(name: &str, iters: usize, mut f: F) {
    // warmup
    for _ in 0..3 { f(); }

    let start = Instant::now();
    for _ in 0..iters { f(); }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iters as u32;
    println!("{name:40} {iters:6} iters  {per_iter:>10.2?}/iter  ({elapsed:.2?} total)");
}

fn main() {
    println!("=== Rules4 TodoMVC Benchmarks ===\n");

    // Bench 1: render with 0 todos (empty state)
    {
        let mut engine = fresh_engine();
        bench("render (0 todos)", 1000, || {
            simulate_render(&mut engine);
        });
        println!("  step_count: {}", engine.step_count);
    }

    // Bench 2: render with 1 todo
    {
        let mut engine = fresh_engine();
        simulate_add_todo(&mut engine, "buy milk");
        engine.step_count = 0;
        bench("render (1 todo)", 1000, || {
            simulate_render(&mut engine);
        });
        println!("  step_count after 1000 renders: {}", engine.step_count);
    }

    // Bench 3: render with 5 todos
    {
        let mut engine = fresh_engine();
        for t in &["buy milk", "clean house", "write code", "walk dog", "read book"] {
            simulate_add_todo(&mut engine, t);
        }
        engine.step_count = 0;
        bench("render (5 todos)", 1000, || {
            simulate_render(&mut engine);
        });
        println!("  step_count after 1000 renders: {}", engine.step_count);
    }

    // Bench 4: keystroke + render (the actual hot path)
    {
        let mut engine = fresh_engine();
        simulate_add_todo(&mut engine, "buy milk");
        engine.step_count = 0;
        let keys = ["h","he","hel","hell","hello"," ","w","wo","wor","worl","world"];
        let mut i = 0;
        bench("keystroke+render (1 todo)", 1000, || {
            let key = keys[i % keys.len()];
            simulate_keystroke(&mut engine, key);
            simulate_render(&mut engine);
            i += 1;
        });
        println!("  step_count after 1000 keystroke+render: {}", engine.step_count);
    }

    // Bench 5: keystroke + render with 10 todos
    {
        let mut engine = fresh_engine();
        for i in 0..10 {
            simulate_add_todo(&mut engine, &format!("todo item {i}"));
        }
        engine.step_count = 0;
        let keys = ["h","he","hel","hell","hello"," ","w","wo","wor","worl","world"];
        let mut i = 0;
        bench("keystroke+render (10 todos)", 1000, || {
            let key = keys[i % keys.len()];
            simulate_keystroke(&mut engine, key);
            simulate_render(&mut engine);
            i += 1;
        });
        println!("  step_count after 1000 keystroke+render: {}", engine.step_count);
    }

    // Bench 6: single render step count (for profiling)
    {
        let mut engine = fresh_engine();
        simulate_add_todo(&mut engine, "buy milk");
        engine.step_count = 0;
        simulate_render(&mut engine);
        println!("\n--- Single render profile (1 todo) ---");
        println!("  reductions: {}", engine.step_count);
    }

    // Bench 6b: count eval calls for a single render (10 todos)
    {
        let mut engine = fresh_engine();
        for i in 0..10 {
            simulate_add_todo(&mut engine, &format!("todo item {i}"));
        }
        engine.step_count = 0;
        engine.eval_calls = 0;
        simulate_render(&mut engine);
        println!("  eval_calls for 1 render (10 todos): {}", engine.eval_calls);
        println!("  reductions for 1 render (10 todos): {}", engine.step_count);
    }

    // Bench 7: just eval fact(10) as a baseline
    {
        let mut store = TermStore::new();
        let src = "fn fact(0) = 1\nfn fact(?n) = ?n * fact(?n - 1)\n0";
        let tokens = Lexer::new(src).tokenize();
        let program = Parser::new(tokens, &mut store).parse_program();
        let term = pattern_to_term(&mut store, &program.expr);
        let mut engine = Engine::new(store, program.rules, program.meta_rules);
        engine.eval(term);
        let fact_sym = engine.make_sym("fact");
        let ten = engine.make_num(10);
        let fact10 = engine.make_call(fact_sym, &[ten]);
        bench("fact(10) baseline", 10000, || {
            engine.eval(fact10);
        });
    }
}
