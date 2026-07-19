//! `ai-lang` — the generic runner.
//!
//! A program is a hash in a content-addressed codebase. Names are
//! mutable surface aliases. The codebase lives on disk as a directory:
//!
//!   defs/<hex>.def     canonical bytes per def
//!   types/<hex>.type   cached TypeScheme per def
//!   names.txt          name → hash
//!
//! ## Commands
//!
//!   ai-lang add <file.ail>                   ingest source into the codebase
//!   ai-lang ls                                list named defs
//!   ai-lang run <name> [--nodes=N] [-- ...]   JIT and invoke `<name>() -> Int`
//!   ai-lang serve                             become a worker for `at()` calls
//!
//! ## Codebase location
//!
//! Defaults to `./.ai-lang`. Override with `--codebase <path>` or env
//! `AI_LANG_CODEBASE=<path>`.

#[path = "../edit_server.rs"]
mod edit_server;

use ai_lang::Hash;
use ai_lang::ast::{Def, Type};
use ai_lang::codebase::Codebase;
use ai_lang::depindex::DependencyIndex;
use ai_lang::effects::EffectSet;
use ai_lang::edit::{self, Direction};
use ai_lang::printer::print_def;
use ai_lang::codegen::{
    CompiledModule, IncrementalJit, Jit, def_symbol, init_native_target,
};
use ai_lang::io_externs::{set_user_args, set_worker_nodes};
use ai_lang::knowledge::KnowledgeBase;
use ai_lang::namespace::{self, MergeResult};
use ai_lang::codec::encode_extern;
use ai_lang::deploy::{
    DeployManager, DeployRequest, StateAction, deploy_on_channel, invoke_on_channel,
    rollback_on_channel, serve_deploy_turn,
};
use ai_lang::net::{
    ItemKind, bind, build_at_runtime_binding, clear_at_conn_cache, clear_current_at_binding,
    clear_current_knowledge_base, clear_current_runtime, client_authenticate,
    install_current_at_binding, install_current_knowledge_base, install_current_runtime,
    server_expect_auth,
};
use ai_lang::parser::parse_module;
use ai_lang::resolve::{AtBinding, ExternSig, ResolvedDef, ResolvedModule, resolve_module};
use ai_lang::slice::{self, SliceError};
use ai_lang::runtime::{Runtime, Thread};
use ai_lang::stdlib::SOURCE as STDLIB;
use ai_lang::typecheck::{TypeCache, TypeScheme, typecheck_module};
use inkwell::context::Context;
use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

// =============================================================================
// CLI
// =============================================================================

fn usage(code: i32) -> ! {
    let s = "\
ai-lang — generic runner for content-addressed ai-lang programs

usage:
  ai-lang add <file.ail> [--prefix <ns>]   ingest source into the codebase.
                                           --prefix <ns> prepends a namespace
                                           (e.g. `Bar::`) to every def name.
  ai-lang ls [<prefix>] [--tag <tag>]       list named defs, optionally filtered by
                                           namespace prefix or metadata tag
  ai-lang test [--tag <tag>] [--prefix <ns>]  run tagged tests (default: ai:test).
                                           Pure tests are cached by hash —
                                           once passed, never re-run.
                                           --prefix filters to a namespace.
  ai-lang serve                             become a worker for at() (empty runtime)
       [--bind <addr>] [--token <t>]       --bind = standalone node (no parent
                                           watchdog); --token = shared-secret auth

live deploy (typed, instant rollback; nodes keep every version resident):
  ai-lang deploy <file.ail> <addr>          ship a new version to a running node.
       --bind <binding>=<def>              create/flip a named entry point (its
                                           signature is pinned at creation)
       [--migrate <state>=<def>]           required when a state's type changed;
                                           must be fn(OldT) -> NewT, checked on
                                           the node against the LIVE cell's type
       [--allow-state-drop]                permit live states absent from the
                                           new version (cells stay resident)
       [--token <t>]                       auth for token-protected nodes
  ai-lang rollback <addr> <binding>         instant flip to the previous version
  ai-lang invoke <addr> <binding> [i...]    call a binding by name (Int args, v1)

  ai-lang serve-edit                        start the JSONL structural-edit server
                                           (one JSON request per line on stdin,
                                           one JSON response per line on stdout)

structural editing (Phase 1):
  ai-lang rename <from> <to>               INSTANT, unbreakable rename. Moves a
                                           name alias only; recompiles nothing and
                                           breaks no callers (refs are by hash).
  ai-lang alias <from> <to>                 like rename, but keeps the old name.
                                           Both names now point to the same hash.
  ai-lang tag <name> <tag>                  add a tag to a definition's metadata
  ai-lang untag <name> <tag>                remove a tag
  ai-lang meta <name>                       print metadata as JSON
  ai-lang meta set <name> <key> <value>     set a user key in meta
  ai-lang meta unset <name> <key>           remove a user key from meta
  ai-lang update <name> <file> [--dry-run] replace <name>'s def with <file>'s source.
       [--propagate]                      Dependents are NOT touched by default —
                                           they still reference the old hash.
                                           --dry-run reports impact without committing.
                                           --propagate rewrites the whole dependency
                                           cone (same-type only; errors on type change).
  ai-lang usages <name|hashprefix>         exact reverse-dependency lookup (no grep)
  ai-lang deps <name|hashprefix>           forward dependencies of a definition
       [--reverse] [--transitive]         --reverse = who depends on it;
                                           --transitive = walk to a fixpoint
  ai-lang view <name|hashprefix>           project a stored def back to source
  ai-lang effects <name|hashprefix>        show inferred effects + guarantees
                                           (pure / mobile / cacheable)
  ai-lang eval <name> [<args>...]          JIT-run a def and print its value as
                                           JSON. Args are JSON values matching
                                           the def's parameter types.
  ai-lang cli                               list all entry points (tagged ai:cli)
  ai-lang cli <name> [args...] [--help]    run an entry point with CLI arg parsing
                                           -- flags declared in metadata.

  ai-lang propagate <name> [--dry-run]    verify that every transitive dependent of
                                           <name> typechecks against its current hash;
                                           reports the cone. Same-type safety check.
  ai-lang todos                            list the worklist of dependents that still
                                           reference an old hash after type changes

structural refactors (Phase 3, all on the canonical AST, then propagate):
  ai-lang move <from> <to>                 namespace-path move (like rename; the
                                           target may be a dotted path). O(1),
                                           callers unaffected (refs are by hash).
  ai-lang inline <name> [--dry-run]        beta-reduce every call to <name> into
                                           its callers (capture-correct de Bruijn),
                                           then propagate. Errors clearly if <name>
                                           is self-recursive or used as a value.
  ai-lang reorder-params <name> <perm>     permute <name>'s params (<perm> e.g. 1,0)
       [--dry-run]                         and rewrite every call site so behavior
                                           is preserved; then propagate.
  ai-lang extract <name> <sel> <new_name>  lift a sub-expression of <name> into a
       [--dry-run]                         new def <new_name> and call it. <sel> is
                                           `let-value` or `body`. Then propagate.

branches + history (Phase 4, causal namespaces; defs are shared, never copied):
  ai-lang branch <name> [<from>]           O(1) fork: new branch head points at the
                                           same causal node as <from> (default: the
                                           current branch). No defs copied.
  ai-lang branches                          list branches (current marked with *)
  ai-lang switch <name>                     change active branch; reload names.txt
                                           from that branch's head snapshot
  ai-lang diff <a> <b>                      structural diff of two branch snapshots
                                           (added / removed / changed names)
  ai-lang history                           causal-hash chain of the current branch
  ai-lang undo                              move the current branch head back to its
                                           parent (reverts the last name change;
                                           defs are immutable, nothing is destroyed)
  ai-lang merge <from> <into>               3-way merge over the name layer; a name
                                           changed differently on both sides is a
                                           reported conflict, never silently resolved

options (all commands):
  --codebase <path>      override codebase root (default ./.ai-lang)
                         also honoured via AI_LANG_CODEBASE env
  --json                 emit machine-readable JSON (rename/usages/deps/view/
                         update/move/inline/reorder-params/extract)

run-only:
  --nodes=<N>            spawn N `ai-lang serve` subprocesses, expose them
                         to the program via `node_count()` / `get_node_port(i)`
";
    let _ = io::stderr().write_all(s.as_bytes());
    std::process::exit(code);
}

fn cb_root(opt: Option<&String>) -> PathBuf {
    if let Some(p) = opt {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("AI_LANG_CODEBASE") {
        return PathBuf::from(p);
    }
    PathBuf::from(".ai-lang")
}

/// Flags shared across subcommands, parsed out of `args` in place.
struct Flags {
    codebase: Option<String>,
    nodes: Option<usize>,
    json: bool,
    reverse: bool,
    transitive: bool,
    /// Output path for `export` (`--out <path>`).
    out: Option<String>,
    post_dashdash: Vec<String>,
}

/// Parse the shared flags (`--codebase`, `--nodes`, `--json`, `--reverse`,
/// `--transitive`, and `-- ...`) out of `args` in place. Anything else is left
/// as a positional argument for the subcommand to consume.
fn split_flags(args: &mut Vec<String>) -> Flags {
    let mut codebase: Option<String> = None;
    let mut nodes: Option<usize> = None;
    let mut json = false;
    let mut reverse = false;
    let mut transitive = false;
    let mut out: Option<String> = None;
    let mut post_dashdash: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "--" {
            post_dashdash = args.split_off(i + 1);
            args.pop(); // drop the "--" itself
            break;
        } else if a == "--json" {
            json = true;
            args.remove(i);
        } else if a == "--reverse" {
            reverse = true;
            args.remove(i);
        } else if a == "--transitive" {
            transitive = true;
            args.remove(i);
        } else if a == "--codebase" {
            if i + 1 >= args.len() {
                eprintln!("--codebase: missing value");
                std::process::exit(2);
            }
            codebase = Some(args[i + 1].clone());
            args.drain(i..i + 2);
        } else if let Some(rest) = a.strip_prefix("--codebase=") {
            codebase = Some(rest.to_owned());
            args.remove(i);
        } else if a == "--out" {
            if i + 1 >= args.len() {
                eprintln!("--out: missing value");
                std::process::exit(2);
            }
            out = Some(args[i + 1].clone());
            args.drain(i..i + 2);
        } else if let Some(rest) = a.strip_prefix("--out=") {
            out = Some(rest.to_owned());
            args.remove(i);
        } else if let Some(rest) = a.strip_prefix("--nodes=") {
            nodes = Some(rest.parse().unwrap_or_else(|_| {
                eprintln!("--nodes=<N>: bad integer");
                std::process::exit(2);
            }));
            args.remove(i);
        } else if a == "--nodes" {
            if i + 1 >= args.len() {
                eprintln!("--nodes: missing value");
                std::process::exit(2);
            }
            nodes = Some(args[i + 1].parse().unwrap_or_else(|_| {
                eprintln!("--nodes <N>: bad integer");
                std::process::exit(2);
            }));
            args.drain(i..i + 2);
        } else {
            i += 1;
        }
    }
    Flags {
        codebase,
        nodes,
        json,
        reverse,
        transitive,
        out,
        post_dashdash,
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage(2);
    }
    // args[0] is the binary path; args[1] is the subcommand.
    let subcommand = args.remove(1);
    // Drop binary path.
    args.remove(0);
    let flags = split_flags(&mut args);
    let cb_path = cb_root(flags.codebase.as_ref());

    match subcommand.as_str() {
        "add" => {
            // <file> [--prefix <ns>]
            let mut prefix: Option<String> = None;
            let mut positional: Vec<&String> = Vec::new();
            let mut i = 0;
            while i < args.len() {
                if args[i] == "--prefix" {
                    i += 1;
                    if i < args.len() {
                        prefix = Some(args[i].clone());
                    }
                } else {
                    positional.push(&args[i]);
                }
                i += 1;
            }
            if positional.len() != 1 {
                usage(2);
            }
            cmd_add(&positional[0], prefix.as_deref(), &cb_path);
        }
        "ls" => {
            let mut prefix: Option<&str> = None;
            let mut tag_filter: Option<&str> = None;
            let mut i = 0;
            while i < args.len() {
                if args[i] == "--tag" {
                    i += 1;
                    if i < args.len() {
                        tag_filter = Some(&args[i]);
                    }
                } else if prefix.is_none() {
                    prefix = Some(&args[i]);
                }
                i += 1;
            }
            cmd_ls(prefix, tag_filter, &cb_path);
        }
        "run" => {
            if args.is_empty() {
                usage(2);
            }
            cmd_run(
                &args[0],
                flags.nodes.unwrap_or(0),
                flags.post_dashdash,
                &cb_path,
            );
        }
        "test" => {
            // [--tag <tag>] [--prefix <ns>] [--all]
            let mut prefix: Option<String> = None;
            let mut tag: Option<String> = None;
            let mut all = false;
            let mut i = 0;
            while i < args.len() {
                if args[i] == "--tag" {
                    i += 1;
                    if i < args.len() {
                        tag = Some(args[i].clone());
                    }
                } else if args[i] == "--prefix" {
                    i += 1;
                    if i < args.len() {
                        prefix = Some(args[i].clone());
                    }
                } else if args[i] == "--all" {
                    all = true;
                }
                i += 1;
            }
            cmd_test(prefix.as_deref(), tag.as_deref(), all, flags.json, &cb_path);
        }
        "rename" => {
            if args.len() != 2 {
                usage(2);
            }
            cmd_rename(&args[0], &args[1], flags.json, &cb_path);
        }
        "alias" => {
            if args.len() != 2 {
                usage(2);
            }
            cmd_alias(&args[0], &args[1], flags.json, &cb_path);
        }
        "tag" => {
            if args.len() != 2 {
                usage(2);
            }
            cmd_tag(&args[0], &args[1], &cb_path);
        }
        "untag" => {
            if args.len() != 2 {
                usage(2);
            }
            cmd_untag(&args[0], &args[1], &cb_path);
        }
        "meta" => {
            // meta <name>
            // meta set <name> <key> <value>
            // meta unset <name> <key>
            if args.is_empty() || args.len() > 4 {
                usage(2);
            }
            if args.len() == 1 {
                cmd_meta_show(&args[0], flags.json, &cb_path);
            } else if args.len() == 4 && args[0] == "set" {
                cmd_meta_set(&args[1], &args[2], &args[3], &cb_path);
            } else if args.len() == 3 && args[0] == "unset" {
                cmd_meta_unset(&args[1], &args[2], &cb_path);
            } else {
                usage(2);
            }
        }
        "update" => {
            // Positional: <name> <file>. --dry-run and --propagate may appear anywhere.
            let mut dry_run = false;
            let mut propagate = false;
            let positional: Vec<&String> = args
                .iter()
                .filter(|a| {
                    if a.as_str() == "--dry-run" {
                        dry_run = true;
                        false
                    } else if a.as_str() == "--propagate" {
                        propagate = true;
                        false
                    } else {
                        true
                    }
                })
                .collect();
            if positional.len() != 2 {
                usage(2);
            }
            cmd_update(positional[0], positional[1], dry_run, propagate, flags.json, &cb_path);
        }
        "move" => {
            if args.len() != 2 {
                usage(2);
            }
            cmd_move(&args[0], &args[1], flags.json, &cb_path);
        }
        "inline" => {
            // <name> [--dry-run]
            let mut dry_run = false;
            let positional: Vec<&String> = args
                .iter()
                .filter(|a| {
                    if a.as_str() == "--dry-run" {
                        dry_run = true;
                        false
                    } else {
                        true
                    }
                })
                .collect();
            if positional.len() != 1 {
                usage(2);
            }
            cmd_inline(positional[0], dry_run, flags.json, &cb_path);
        }
        "reorder-params" => {
            // <name> <perm> [--dry-run]
            let mut dry_run = false;
            let positional: Vec<&String> = args
                .iter()
                .filter(|a| {
                    if a.as_str() == "--dry-run" {
                        dry_run = true;
                        false
                    } else {
                        true
                    }
                })
                .collect();
            if positional.len() != 2 {
                usage(2);
            }
            cmd_reorder_params(positional[0], positional[1], dry_run, flags.json, &cb_path);
        }
        "extract" => {
            // <name> <selector> <new_name> [--dry-run]
            let mut dry_run = false;
            let positional: Vec<&String> = args
                .iter()
                .filter(|a| {
                    if a.as_str() == "--dry-run" {
                        dry_run = true;
                        false
                    } else {
                        true
                    }
                })
                .collect();
            if positional.len() != 3 {
                usage(2);
            }
            cmd_extract(
                positional[0],
                positional[1],
                positional[2],
                dry_run,
                flags.json,
                &cb_path,
            );
        }
        "propagate" => {
            let mut dry_run = false;
            let positional: Vec<&String> = args
                .iter()
                .filter(|a| {
                    if a.as_str() == "--dry-run" {
                        dry_run = true;
                        false
                    } else {
                        true
                    }
                })
                .collect();
            if positional.len() != 1 {
                usage(2);
            }
            cmd_propagate(positional[0], dry_run, flags.json, &cb_path);
        }
        "todos" => {
            cmd_todos(flags.json, &cb_path);
        }
        "usages" => {
            if args.len() != 1 {
                usage(2);
            }
            cmd_usages(&args[0], flags.json, &cb_path);
        }
        "deps" => {
            if args.len() != 1 {
                usage(2);
            }
            cmd_deps(
                &args[0],
                flags.reverse,
                flags.transitive,
                flags.json,
                &cb_path,
            );
        }
        "view" => {
            if args.len() != 1 {
                usage(2);
            }
            cmd_view(&args[0], flags.json, &cb_path);
        }
        "effects" => {
            if args.len() != 1 {
                usage(2);
            }
            cmd_effects(&args[0], flags.json, &cb_path);
        }
        "eval" => {
            if args.is_empty() {
                usage(2);
            }
            cmd_eval(&args[0], &args[1..], flags.json, &cb_path);
        }
        "cli" => {
            if args.is_empty() {
                cmd_cli_list(&cb_path);
            } else if args.len() == 2 && args[1] == "--help" {
                cmd_cli_help(&args[0], &cb_path);
            } else {
                cmd_cli_run(&args[0], &args[1..], &cb_path);
            }
        }
        "export" => {
            // <name...> --out <path>
            if args.is_empty() {
                usage(2);
            }
            cmd_export(&args, &flags);
        }
        "import" => {
            // <path>
            if args.len() != 1 {
                usage(2);
            }
            cmd_import(&args, &flags);
        }
        "find-by-type" => {
            // <signature...>
            if args.is_empty() {
                usage(2);
            }
            cmd_find_by_type(&args, &flags);
        }
        "branch" => {
            // <name> [<from>]
            if args.is_empty() || args.len() > 2 {
                usage(2);
            }
            let from = args.get(1).map(|s| s.as_str());
            cmd_branch(&args[0], from, flags.json, &cb_path);
        }
        "branches" => {
            cmd_branches(flags.json, &cb_path);
        }
        "switch" => {
            if args.len() != 1 {
                usage(2);
            }
            cmd_switch(&args[0], flags.json, &cb_path);
        }
        "diff" => {
            if args.len() != 2 {
                usage(2);
            }
            cmd_diff(&args[0], &args[1], flags.json, &cb_path);
        }
        "history" => {
            cmd_history(flags.json, &cb_path);
        }
        "undo" => {
            cmd_undo(flags.json, &cb_path);
        }
        "merge" => {
            if args.len() != 2 {
                usage(2);
            }
            cmd_merge(&args[0], &args[1], flags.json, &cb_path);
        }
        "serve" => {
            let (mut bind_addr, mut token) = (None::<String>, None::<String>);
            let mut i = 0;
            while i < args.len() {
                if args[i] == "--bind" && i + 1 < args.len() {
                    bind_addr = Some(args[i + 1].clone());
                    i += 2;
                } else if let Some(rest) = args[i].strip_prefix("--bind=") {
                    bind_addr = Some(rest.to_owned());
                    i += 1;
                } else if args[i] == "--token" && i + 1 < args.len() {
                    token = Some(args[i + 1].clone());
                    i += 2;
                } else if let Some(rest) = args[i].strip_prefix("--token=") {
                    token = Some(rest.to_owned());
                    i += 1;
                } else {
                    eprintln!("serve: unexpected argument {}", args[i]);
                    usage(2);
                }
            }
            cmd_serve(bind_addr.as_deref(), token.as_deref());
        }
        "deploy" => {
            cmd_deploy_cli(&args);
        }
        "rollback" => {
            let (positional, token) = take_token(&args);
            if positional.len() != 2 {
                eprintln!("usage: ai-lang rollback <addr> <binding> [--token <t>]");
                std::process::exit(2);
            }
            cmd_rollback(&positional[0], &positional[1], token.as_deref());
        }
        "invoke" => {
            let (positional, token) = take_token(&args);
            if positional.len() < 2 {
                eprintln!("usage: ai-lang invoke <addr> <binding> [<int> ...] [--token <t>]");
                std::process::exit(2);
            }
            let ints: Vec<i64> = positional[2..]
                .iter()
                .map(|s| {
                    s.parse().unwrap_or_else(|_| {
                        eprintln!("invoke: argument `{}` is not an Int", s);
                        std::process::exit(2);
                    })
                })
                .collect();
            cmd_invoke(&positional[0], &positional[1], &ints, token.as_deref());
        }
        "serve-edit" => {
            edit_server::serve(&cb_path);
        }
        "lambda-worker" => {
            cmd_lambda_worker();
        }
        "at-serve" => {
            // `at-serve`: a GENERIC warm node. It starts with only the stdlib
            // resident and serves self-contained KIND_BUNDLE frames forever
            // over a localhost TCP socket — each request ships its own code,
            // which we install on demand (a no-op once warm) and run. The
            // JIT'd code stays resident across calls. Prints `READY <port>`.
            cmd_at_serve();
        }
        "at-handle" => {
            // `at-handle <app.ail>`: compile the app (with stdlib), read ONE
            // request frame (`[KIND_CALL] ++ closure`) from stdin, run it,
            // write the reply frame (`[KIND_RESULT] ++ value`) to stdout.
            // This is the node front-end for a request/response transport
            // (HTTP / a Vercel function): the bridge hands us the body, we
            // hand back the body. The app must define the same source the
            // client compiled, so the shipped closure's code is resident.
            if args.is_empty() {
                eprintln!("usage: ai-lang at-handle <app.ail>");
                std::process::exit(2);
            }
            cmd_at_handle(&args[0]);
        }
        other => {
            eprintln!("unknown subcommand: {}", other);
            usage(2);
        }
    }
}

// =============================================================================
// `lambda-worker` — a fully generic AWS Lambda custom runtime.
//
// The function has NO baked-in logic. On each invocation it is HANDED the
// code to run, in the invoke payload, compiles it, runs it, and returns
// the result. Deploy this image once; the work is decided per-invocation
// by whoever invokes it.
//
// Payload: {"src64":"<base64 ai-lang source defining `task(input: String)
// -> String`>","in64":"<base64 input string>"}. The worker compiles the
// source (with the stdlib), calls `task(input)`, and returns the String.
// =============================================================================

/// A server-side `Channel` that replays one request frame and captures the
/// one reply frame — lets `serve_one` process a frame handed to us out of
/// band (here: an HTTP/FaaS request body on stdin).
struct OneShotChannel {
    request: Option<Vec<u8>>,
    response: Option<Vec<u8>>,
}
impl ai_lang::net::Channel for OneShotChannel {
    fn read_frame(&mut self) -> Result<Vec<u8>, ai_lang::net::NetError> {
        self.request
            .take()
            .ok_or(ai_lang::net::NetError::ConnectionClosed)
    }
    fn write_frame(&mut self, body: &[u8]) -> Result<(), ai_lang::net::NetError> {
        self.response = Some(body.to_vec());
        Ok(())
    }
}

/// `at-serve <app.ail>`: compile the app once, then serve `at` frames
/// forever on a localhost TCP socket, a thread per connection. The runtime
/// (JIT'd code + any node `state`) is shared and stays resident across all
/// calls — so a warm instance never re-compiles, and `state` persists for
/// the process's lifetime.
fn cmd_at_serve() {
    init_native_target().expect("init native target");

    // A GENERIC node: start with only the stdlib in a warm IncrementalJit;
    // each request ships its own program in a self-contained KIND_BUNDLE,
    // which we install on demand (idempotent — a no-op once warm) and run.
    let ctx = Context::create();
    let empty_r = resolve_module(&parse_module("").unwrap()).unwrap();
    let empty_cm = CompiledModule::build(&ctx, &empty_r).expect("build empty");
    let mut rt = Runtime::new_with_metadata(
        empty_cm.closure_type_infos.clone(),
        empty_cm.shape_registry.clone(),
        empty_cm.shape_meta.clone(),
        empty_cm.shape_by_type_id.clone(),
    );
    let mut jit = IncrementalJit::new(empty_cm, &rt).expect("incremental jit");

    let stdlib_m = parse_module(STDLIB).expect("parse stdlib");
    let stdlib_r = resolve_module(&stdlib_m).expect("resolve stdlib");
    let stdlib_kb = KnowledgeBase::build(&stdlib_r);
    let stdlib_roots: Vec<Hash> = stdlib_r.defs.iter().map(|d| d.hash).collect();
    if let Err(e) = install_from_kb(&mut jit, &mut rt, &stdlib_kb, &stdlib_roots) {
        eprintln!("at-serve: stdlib install failed: {}", e);
        std::process::exit(1);
    }
    install_current_runtime(&rt);
    install_current_knowledge_base(&stdlib_kb);

    // The effect policy: the set of effects shipped code is ALLOWED to
    // perform. The node infers each bundle's effects and refuses any that
    // exceed this (never installing or running them). Default is the safe
    // "compute + node-state" set — every effect EXCEPT the dangerous ones
    // (IO, Net, FFI): shipped code can compute and touch this node's state,
    // but cannot call arbitrary C, hit the network, or do process I/O.
    // Override with `AI_LANG_AT_EFFECTS` (e.g. `all`, `pure`, `io,net`).
    let policy = match std::env::var("AI_LANG_AT_EFFECTS") {
        Ok(s) => EffectSet::from_tokens(&s).unwrap_or_else(|e| {
            eprintln!("at-serve: bad AI_LANG_AT_EFFECTS: {}", e);
            std::process::exit(2);
        }),
        Err(_) => EffectSet::ALL.without(EffectSet::IO | EffectSet::NET | EffectSet::FFI),
    };
    // Precompute the resident stdlib's effects once; seeds the per-request
    // inference so shipped `TopRef`s into the stdlib resolve correctly.
    let stdlib_effects = ai_lang::knowledge::transitive_effects(
        &stdlib_r.defs,
        &[],
        &std::collections::HashMap::new(),
    );
    let policy_tokens = policy.tokens();
    eprintln!(
        "[at-serve] effect policy: allow [{}] (set AI_LANG_AT_EFFECTS to change)",
        if policy_tokens.is_empty() { "pure".to_string() } else { policy_tokens.join(",") }
    );

    let listener = ai_lang::net::bind("127.0.0.1:0").unwrap_or_else(|e| {
        eprintln!("at-serve: bind: {}", e);
        std::process::exit(1);
    });
    let port = listener.local_addr().unwrap().port();
    println!("READY {}", port);
    let _ = io::stdout().flush();

    // Single-threaded on the home thread (the proven serve model). Each
    // accepted connection is served frame-by-frame until the peer
    // disconnects; the proxy opens a fresh short connection per request.
    loop {
        match listener.accept() {
            Ok((mut stream, _)) => {
                let _ = stream.set_nodelay(true);
                loop {
                    match unsafe {
                        ai_lang::net::serve_bundle_one(
                            &mut rt,
                            &mut jit,
                            &mut stream,
                            policy,
                            &stdlib_effects,
                        )
                    } {
                        Ok(()) => continue,
                        Err(_) => break,
                    }
                }
            }
            Err(e) => {
                eprintln!("at-serve: accept: {}", e);
                break;
            }
        }
    }
}

/// `at-handle <app.ail>`: compile the app (+ stdlib), then process exactly
/// one `at` request frame from stdin and write the reply frame to stdout.
/// The node front-end for a request/response transport — a Vercel function's
/// Rust handler spawns this per request, piping the HTTP body through.
fn cmd_at_handle(app_path: &str) {
    use std::io::{Read, Write};
    init_native_target().expect("init native target");

    let app_src = std::fs::read_to_string(app_path).unwrap_or_else(|e| {
        eprintln!("at-handle: cannot read {}: {}", app_path, e);
        std::process::exit(1);
    });
    let full = format!("{}\n{}", STDLIB, app_src);
    let m = parse_module(&full).unwrap_or_else(|e| {
        eprintln!("at-handle: parse: {:?}", e);
        std::process::exit(1);
    });
    let r = resolve_module(&m).unwrap_or_else(|e| {
        eprintln!("at-handle: resolve: {:?}", e);
        std::process::exit(1);
    });
    let ctx = Context::create();
    let cm = CompiledModule::build(&ctx, &r).unwrap_or_else(|e| {
        eprintln!("at-handle: build: {:?}", e);
        std::process::exit(1);
    });
    let rt = Runtime::new_with_metadata(
        cm.closure_type_infos.clone(),
        cm.shape_registry.clone(),
        cm.shape_meta.clone(),
        cm.shape_by_type_id.clone(),
    );
    let _jit = Jit::new(cm, &rt).unwrap_or_else(|e| {
        eprintln!("at-handle: jit: {:?}", e);
        std::process::exit(1);
    });
    install_current_runtime(&rt);
    let kb = KnowledgeBase::build(&r);
    install_current_knowledge_base(&kb);

    let mut frame = Vec::new();
    if let Err(e) = std::io::stdin().read_to_end(&mut frame) {
        eprintln!("at-handle: read stdin: {}", e);
        std::process::exit(1);
    }
    let mut ch = OneShotChannel {
        request: Some(frame),
        response: None,
    };
    if let Err(e) = unsafe { ai_lang::net::serve_one(&rt, &mut ch) } {
        eprintln!("at-handle: serve_one: {}", e);
        std::process::exit(1);
    }
    let resp = ch.response.unwrap_or_default();
    let mut out = std::io::stdout();
    out.write_all(&resp).and_then(|_| out.flush()).unwrap_or_else(|e| {
        eprintln!("at-handle: write stdout: {}", e);
        std::process::exit(1);
    });
}

fn cmd_lambda_worker() {
    init_native_target().expect("init native target");
    let api = std::env::var("AWS_LAMBDA_RUNTIME_API")
        .expect("AWS_LAMBDA_RUNTIME_API not set (run inside Lambda)");

    // Compile the stdlib ONCE, at init. The base engine is empty; we
    // install the whole stdlib into a warm IncrementalJit, so subsequent
    // invocations only need to JIT the few defs the handed code adds.
    let ctx = Context::create();
    let empty_r = resolve_module(&parse_module("").unwrap()).unwrap();
    let empty_cm = CompiledModule::build(&ctx, &empty_r).expect("build empty");
    let mut rt = Runtime::new_with_metadata(
        empty_cm.closure_type_infos.clone(),
        empty_cm.shape_registry.clone(),
        empty_cm.shape_meta.clone(),
        empty_cm.shape_by_type_id.clone(),
    );
    let mut jit = IncrementalJit::new(empty_cm, &rt).expect("incremental jit");

    let stdlib_m = parse_module(STDLIB).expect("parse stdlib");
    let stdlib_r = resolve_module(&stdlib_m).expect("resolve stdlib");
    let stdlib_kb = KnowledgeBase::build(&stdlib_r);
    let stdlib_roots: Vec<Hash> = stdlib_r.defs.iter().map(|d| d.hash).collect();
    let stdlib_set: HashSet<Hash> = stdlib_roots.iter().copied().collect();
    let t0 = std::time::Instant::now();
    if let Err(e) = install_from_kb(&mut jit, &mut rt, &stdlib_kb, &stdlib_roots) {
        eprintln!("[lambda-worker] FATAL: stdlib install failed: {}", e);
        std::process::exit(1);
    }
    // Effect policy: shipped code may only perform allowed effects (default
    // = everything except IO/Net/FFI). Override with `AI_LANG_AT_EFFECTS`.
    let policy = match std::env::var("AI_LANG_AT_EFFECTS") {
        Ok(s) => EffectSet::from_tokens(&s).unwrap_or_else(|e| {
            eprintln!("[lambda-worker] bad AI_LANG_AT_EFFECTS: {}", e);
            std::process::exit(2);
        }),
        Err(_) => EffectSet::ALL.without(EffectSet::IO | EffectSet::NET | EffectSet::FFI),
    };
    let pt = policy.tokens();
    eprintln!(
        "[ai-lang lambda-worker] generic runtime ready ({} stdlib defs JIT'd in {:.2}s); effect policy allow [{}]; awaiting code",
        stdlib_roots.len(),
        t0.elapsed().as_secs_f64(),
        if pt.is_empty() { "pure".to_string() } else { pt.join(",") },
    );

    loop {
        let (req_id, event) = match runtime_next(&api) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[lambda-worker] /next failed: {} (retrying)", e);
                std::thread::sleep(std::time::Duration::from_millis(200));
                continue;
            }
        };
        let outcome = run_handed_code(&mut jit, &mut rt, &stdlib_set, policy, &event);
        let (path_kind, body) = match outcome {
            Ok(result) => ("response", result),
            Err(msg) => (
                "error",
                format!("{{\"errorMessage\":{:?},\"errorType\":\"AiLangError\"}}", msg),
            ),
        };
        if let Err(e) = runtime_reply(&api, &req_id, path_kind, &body) {
            eprintln!("[lambda-worker] reply failed: {}", e);
        }
    }
}

/// Install every def/lambda reachable from `roots` (and their extern
/// requirements) into the warm JIT. Already-installed defs are skipped.
fn install_from_kb(
    jit: &mut IncrementalJit<'_>,
    rt: &mut Runtime,
    kb: &KnowledgeBase,
    roots: &[Hash],
) -> Result<(), String> {
    let order = kb
        .collect_transitive_deps(roots)
        .map_err(|e| format!("deps: {}", e))?;
    let mut items: Vec<(ItemKind, Hash, Vec<u8>)> = order
        .iter()
        .map(|h| {
            let (k, b) = kb.lookup(h).expect("dep in kb");
            (*k, *h, b.clone())
        })
        .collect();
    for (name, sig) in kb.extern_requirements(&order) {
        let h = Hash::of_bytes(name.as_bytes());
        items.push((
            ItemKind::Extern,
            h,
            encode_extern(&name, &sig.params, &sig.ret, sig.library.as_deref(), sig.variadic),
        ));
    }
    jit.install(rt, items).map_err(|e| format!("install: {:?}", e))
}

/// Compile + run the code handed in `event` (a JSON payload carrying
/// base64 source + base64 input). Only the defs NOT already installed
/// (i.e. the task itself, not the stdlib) are JIT'd. Returns
/// `task(input)`'s String result.
fn run_handed_code(
    jit: &mut IncrementalJit<'_>,
    rt: &mut Runtime,
    stdlib_set: &HashSet<Hash>,
    policy: EffectSet,
    event: &[u8],
) -> Result<String, String> {
    let text = std::str::from_utf8(event).map_err(|_| "event not UTF-8".to_string())?;
    let source = b64_field(text, "src64").ok_or("payload missing src64")?;
    let input = b64_field(text, "in64").unwrap_or_default();
    let source = String::from_utf8(source).map_err(|_| "src64 not UTF-8".to_string())?;
    let input = String::from_utf8(input).map_err(|_| "in64 not UTF-8".to_string())?;

    let full = format!("{}\n{}", STDLIB, source);
    let m = parse_module(&full).map_err(|e| format!("parse: {:?}", e))?;
    let r = resolve_module(&m).map_err(|e| format!("resolve: {:?}", e))?;
    let mut tc = TypeCache::new();
    typecheck_module(&r, &mut tc).map_err(|e| format!("typecheck: {:?}", e))?;
    let task_hash = r
        .defs
        .iter()
        .find(|d| d.name == "task")
        .map(|d| d.hash)
        .ok_or("source must define `task(input: String) -> String`")?;
    let _ = stdlib_set; // stdlib defs are deduped by install via its own set.

    // Effect gate (BEFORE install/run): infer `task`'s effects (over the
    // whole module, so stdlib calls resolve) and refuse anything beyond the
    // node's policy. `r` already includes the stdlib, so an empty seed is
    // complete here.
    let effects = ai_lang::knowledge::transitive_effects(
        &r.defs,
        &[],
        &std::collections::HashMap::new(),
    );
    let task_eff = effects.get(&task_hash).copied().unwrap_or(EffectSet::EMPTY);
    let excess = task_eff.without(policy);
    if !excess.is_empty() {
        let allow = policy.tokens();
        return Err(format!(
            "effect policy [{}] forbids effect(s) [{}] in shipped code",
            if allow.is_empty() { "pure".to_string() } else { allow.join(",") },
            excess.tokens().join(","),
        ));
    }

    // Install only the delta (install skips defs already in the engine).
    let kb = KnowledgeBase::build(&r);
    install_from_kb(jit, rt, &kb, &[task_hash])?;

    let task = unsafe {
        jit.engine
            .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8) -> *mut u8>(
                &def_symbol(&task_hash),
            )
            .map_err(|_| "task is not a `(String) -> String` fn".to_string())?
    };
    let input_heap = unsafe { ai_lang::ffi::owned_str_to_heap(rt.thread_ptr(), &input) };
    // No panic channel: a contract violation in the task aborts the
    // process before this returns.
    let out = unsafe { task.call(rt.thread_ptr(), input_heap) };
    let result = unsafe { ai_lang::ffi::heap_str_to_owned(out) };
    Ok(result)
}

/// Extract a base64 JSON string field by name and decode it. The values
/// are base64 (no quotes/backslashes), so a quote-delimited scan is safe.
fn b64_field(json: &str, key: &str) -> Option<Vec<u8>> {
    let marker = format!("\"{}\":\"", key);
    let start = json.find(&marker)? + marker.len();
    let rest = &json[start..];
    let end = rest.find('"')?;
    b64_decode(&rest[..end])
}

fn b64_decode(s: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let bytes: Vec<u8> = s.bytes().filter(|&c| c != b'\n' && c != b'\r').collect();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let mut acc = 0u32;
        let mut pad = 0;
        for (i, &c) in chunk.iter().enumerate() {
            if c == b'=' {
                pad += 1;
                acc <<= 6;
            } else {
                acc = (acc << 6) | val(c)?;
            }
            let _ = i;
        }
        if chunk.len() < 4 {
            for _ in 0..(4 - chunk.len()) {
                acc <<= 6;
            }
        }
        out.push((acc >> 16) as u8);
        if pad < 2 {
            out.push((acc >> 8) as u8);
        }
        if pad < 1 {
            out.push(acc as u8);
        }
    }
    Some(out)
}

/// GET the next invocation from the Lambda Runtime API. Returns
/// (request-id, event-body).
fn runtime_next(api: &str) -> Result<(String, Vec<u8>), String> {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(api).map_err(|e| e.to_string())?;
    let req = format!(
        "GET /2018-06-01/runtime/invocation/next HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        api
    );
    stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    let split = find_subslice(&buf, b"\r\n\r\n").ok_or("no header/body split")?;
    let head = String::from_utf8_lossy(&buf[..split]);
    let body = buf[split + 4..].to_vec();
    let req_id = head
        .lines()
        .find_map(|l| {
            let l = l.trim();
            let lower = l.to_ascii_lowercase();
            if lower.starts_with("lambda-runtime-aws-request-id:") {
                Some(l[l.find(':').unwrap() + 1..].trim().to_string())
            } else {
                None
            }
        })
        .ok_or("no request-id header")?;
    Ok((req_id, body))
}

/// POST the result (or error) back to the Runtime API.
fn runtime_reply(api: &str, req_id: &str, kind: &str, body: &str) -> Result<(), String> {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(api).map_err(|e| e.to_string())?;
    let path = format!("/2018-06-01/runtime/invocation/{}/{}", req_id, kind);
    let req = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        path, api, body.len(), body
    );
    stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    let _ = stream.read_to_end(&mut buf);
    Ok(())
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

// =============================================================================
// `add`
// =============================================================================

fn cmd_add(file: &str, prefix: Option<&str>, cb_path: &PathBuf) {
    init_native_target().expect("init native target");
    let user_src = std::fs::read_to_string(file)
        .unwrap_or_else(|e| {
            eprintln!("read {}: {}", file, e);
            std::process::exit(1);
        });
    // Parse the user source separately to know which names are user-defined.
    let user_m = parse_module(&user_src).unwrap_or_else(|e| {
        eprintln!("parse: {:?}", e);
        std::process::exit(1);
    });
    let user_names: std::collections::HashSet<String> =
        user_m.defs.iter().map(|d| d.name.clone()).collect();

    let full_src = format!("{}\n{}", STDLIB, user_src);
    let m = parse_module(&full_src).unwrap_or_else(|e| {
        eprintln!("parse: {:?}", e);
        std::process::exit(1);
    });
    let mut r = resolve_module(&m).unwrap_or_else(|e| {
        eprintln!("resolve: {:?}", e);
        std::process::exit(1);
    });
    let mut tc = TypeCache::new();
    typecheck_module(&r, &mut tc).unwrap_or_else(|e| {
        eprintln!("typecheck: {:?}", e);
        std::process::exit(1);
    });

    // Apply namespace prefix: only to user-defined names, not stdlib.
    if let Some(ns) = prefix {
        let ns = ns.trim_end_matches(':');
        let ns_prefix = if ns.is_empty() {
            String::new()
        } else {
            format!("{}::", ns)
        };
        for rd in &mut r.defs {
            if user_names.contains(&rd.name) {
                rd.name = format!("{}{}", ns_prefix, rd.name);
            }
        }
    }

    let mut cb = Codebase::open(cb_path).expect("open codebase");
    cb.store_resolved_module(&r).expect("store resolved module");
    // Persist local names so `view` shows readable parameter/let names.
    // Use the full module (which includes stdlib) for name resolution.
    if let Ok(names) = ai_lang::resolve::local_names_for_module(&m) {
        let _ = cb.store_local_names_batch(&names);
    }
    let typed = cb.store_typecache(&tc).expect("store typecache");

    // Rebuild + persist the dependency index so `deps`/`usages` are fast and
    // correct immediately after `add`. The persisted index is a pure cache; a
    // full rebuild from `defs/` is cheap and always correct (every new def's
    // forward + reverse edges are captured).
    match DependencyIndex::rebuild_from_codebase(&cb) {
        Ok(index) => {
            if let Err(e) = index.save(cb.root()) {
                eprintln!("warning: failed to persist dependency index: {}", e);
            }
        }
        Err(e) => eprintln!("warning: failed to rebuild dependency index: {}", e),
    }

    eprintln!(
        "added {} defs to {} ({} new typeschemes cached)",
        r.defs.len(),
        cb_path.display(),
        typed,
    );
}

// =============================================================================
// `ls`
// =============================================================================

fn cmd_ls(prefix: Option<&str>, tag_filter: Option<&str>, cb_path: &PathBuf) {
    let cb = Codebase::open(cb_path).expect("open codebase");
    let mut entries: Vec<(&String, &Hash)> = cb.names().iter().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    let mut count = 0;
    for (name, hash) in entries {
        if let Some(p) = prefix {
            if !name.starts_with(p) {
                continue;
            }
        }
        if let Some(tag) = tag_filter {
            if !ai_lang::metadata::has_tag(&cb, hash, tag) {
                continue;
            }
        }
        println!("{:<32}  {}", name, &hash.to_hex()[..16]);
        count += 1;
    }
    let total = cb.names().len();
    if let Some(tag) = tag_filter {
        eprintln!("\n  {} name(s) tagged `{}` ({} total in {})", count, tag, total, cb_path.display());
    } else if let Some(p) = prefix {
        eprintln!("\n  {} name(s) matching `{}` ({} total in {})", count, p, total, cb_path.display());
    } else {
        eprintln!("\n  {} names in {}", total, cb_path.display());
    }
}

fn cmd_alias(from: &str, to: &str, json: bool, cb_path: &PathBuf) {
    let mut cb = Codebase::open(cb_path).expect("open codebase");
    let hash = match cb.get_name(from) {
        Some(h) => h,
        None => {
            eprintln!("alias: unknown name: {}", from);
            std::process::exit(1);
        }
    };
    cb.set_name(to.to_owned(), hash).unwrap_or_else(|e| {
        eprintln!("alias: {}", e);
        std::process::exit(1);
    });
    if json {
        println!(
            "{{\"ok\":true,\"aliased\":{{\"from\":\"{}\",\"to\":\"{}\",\"hash\":\"{}\"}}}}",
            json_escape(from),
            json_escape(to),
            hash.to_hex()
        );
    } else {
        println!("aliased {} -> {}  (hash {} unchanged)", from, to, &hash.to_hex()[..16]);
    }
}

fn cmd_tag(name: &str, tag: &str, cb_path: &PathBuf) {
    let cb = Codebase::open(cb_path).expect("open codebase");
    let hash = cb.get_name(name).unwrap_or_else(|| {
        eprintln!("tag: unknown name: {}", name);
        std::process::exit(1);
    });
    ai_lang::metadata::add_tag(&cb, &hash, tag).unwrap_or_else(|e| {
        eprintln!("tag: {}", e);
        std::process::exit(1);
    });
    println!("tagged {} with `{}`", name, tag);
}

fn cmd_untag(name: &str, tag: &str, cb_path: &PathBuf) {
    let cb = Codebase::open(cb_path).expect("open codebase");
    let hash = cb.get_name(name).unwrap_or_else(|| {
        eprintln!("untag: unknown name: {}", name);
        std::process::exit(1);
    });
    ai_lang::metadata::remove_tag(&cb, &hash, tag).unwrap_or_else(|e| {
        eprintln!("untag: {}", e);
        std::process::exit(1);
    });
    println!("removed tag `{}` from {}", tag, name);
}

fn cmd_meta_show(name: &str, json: bool, cb_path: &PathBuf) {
    let cb = Codebase::open(cb_path).expect("open codebase");
    let hash = cb.get_name(name).unwrap_or_else(|| {
        eprintln!("meta: unknown name: {}", name);
        std::process::exit(1);
    });
    match ai_lang::metadata::load(&cb, &hash) {
        Some(meta) => println!("{}", meta.to_string()),
        None => {
            if json {
                println!("{{\"tags\":[],\"meta\":{{}}}}");
            } else {
                println!("(no metadata)");
            }
            std::process::exit(1);
        }
    }
}

fn cmd_meta_set(name: &str, key: &str, value: &str, cb_path: &PathBuf) {
    let cb = Codebase::open(cb_path).expect("open codebase");
    let hash = cb.get_name(name).unwrap_or_else(|| {
        eprintln!("meta set: unknown name: {}", name);
        std::process::exit(1);
    });
    // Try to parse as JSON, fall back to string.
    let v = match ai_lang::jsonl::parse(value) {
        Ok(j) => j,
        Err(_) => ai_lang::jsonl::Json::Str(value.to_string()),
    };
    ai_lang::metadata::set_meta(&cb, &hash, key, &v).unwrap_or_else(|e| {
        eprintln!("meta set: {}", e);
        std::process::exit(1);
    });
    println!("set meta.{} on {}", key, name);
}

fn cmd_meta_unset(name: &str, key: &str, cb_path: &PathBuf) {
    let cb = Codebase::open(cb_path).expect("open codebase");
    let hash = cb.get_name(name).unwrap_or_else(|| {
        eprintln!("meta unset: unknown name: {}", name);
        std::process::exit(1);
    });
    ai_lang::metadata::unset_meta(&cb, &hash, key).unwrap_or_else(|e| {
        eprintln!("meta unset: {}", e);
        std::process::exit(1);
    });
    println!("unset meta.{} on {}", key, name);
}

// =============================================================================
// Structural editing: `rename`, `usages`, `deps`, `view`
// =============================================================================

/// Escape a string for inclusion in a JSON double-quoted string. Hand-rolled
/// (no serde): handles the characters the JSON spec requires.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Render an `Option<String>` name as a JSON value: a quoted string or `null`.
fn json_opt_name(name: &Option<String>) -> String {
    match name {
        Some(n) => format!("\"{}\"", json_escape(n)),
        None => "null".to_owned(),
    }
}

/// Render a `DefRef` as a compact JSON object `{"name":...,"hash":"..."}`.
fn json_defref(d: &ai_lang::edit::DefRef) -> String {
    format!(
        "{{\"name\":{},\"hash\":\"{}\"}}",
        json_opt_name(&d.name),
        d.hash.to_hex()
    )
}

/// Render a list of `DefRef`s as a compact JSON array.
fn json_defref_array(items: &[ai_lang::edit::DefRef]) -> String {
    let parts: Vec<String> = items.iter().map(json_defref).collect();
    format!("[{}]", parts.join(","))
}

/// `ai-lang rename <from> <to>` — instant, unbreakable namespace rename.
fn cmd_rename(from: &str, to: &str, json: bool, cb_path: &PathBuf) {
    let mut cb = Codebase::open(cb_path).expect("open codebase");
    let result = match edit::rename(&mut cb, from, to) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("rename: {}", e);
            std::process::exit(1);
        }
    };
    if json {
        let parts: Vec<String> = result
            .renamed
            .iter()
            .map(|(f, t, h)| {
                format!(
                    "{{\"from\":\"{}\",\"to\":\"{}\",\"hash\":\"{}\"}}",
                    json_escape(f),
                    json_escape(t),
                    h.to_hex()
                )
            })
            .collect();
        println!("{{\"renamed\":[{}]}}", parts.join(","));
    } else {
        for (f, t, h) in &result.renamed {
            println!("renamed {} -> {}  (hash {} unchanged; no callers broken)", f, t, &h.to_hex()[..16]);
        }
    }
}

/// Open the codebase + load-or-rebuild the dependency index, exiting on error.
fn open_with_index(cb_path: &PathBuf) -> (Codebase, DependencyIndex) {
    let cb = Codebase::open(cb_path).expect("open codebase");
    let index = match edit::load_or_rebuild_index(&cb) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("index: {}", e);
            std::process::exit(1);
        }
    };
    (cb, index)
}

/// `ai-lang usages <name|hashprefix>` — exact reverse-dependency lookup.
fn cmd_usages(target: &str, json: bool, cb_path: &PathBuf) {
    let (cb, index) = open_with_index(cb_path);
    let users = match edit::find_usages(&cb, &index, target) {
        Ok(u) => u,
        Err(e) => {
            eprintln!("usages: {}", e);
            std::process::exit(1);
        }
    };
    if json {
        println!("{{\"target\":\"{}\",\"usages\":{}}}", json_escape(target), json_defref_array(&users));
    } else if users.is_empty() {
        println!("no usages of {}", target);
    } else {
        for d in &users {
            let name = d.name.as_deref().unwrap_or("<unnamed>");
            println!("{:<32}  {}", name, &d.hash.to_hex()[..16]);
        }
        eprintln!("\n  {} usage(s) of {}", users.len(), target);
    }
}

/// `ai-lang deps <name|hashprefix> [--reverse] [--transitive]`.
fn cmd_deps(target: &str, reverse: bool, transitive: bool, json: bool, cb_path: &PathBuf) {
    let (cb, index) = open_with_index(cb_path);
    let direction = if reverse {
        Direction::Reverse
    } else {
        Direction::Forward
    };
    let result = match edit::deps(&cb, &index, target, direction, transitive) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("deps: {}", e);
            std::process::exit(1);
        }
    };
    if json {
        let dir = if reverse { "reverse" } else { "forward" };
        println!(
            "{{\"target\":\"{}\",\"direction\":\"{}\",\"transitive\":{},\"deps\":{}}}",
            json_escape(target),
            dir,
            transitive,
            json_defref_array(&result)
        );
    } else if result.is_empty() {
        println!("no dependencies");
    } else {
        for d in &result {
            let name = d.name.as_deref().unwrap_or("<unnamed>");
            println!("{:<32}  {}", name, &d.hash.to_hex()[..16]);
        }
    }
}

/// `ai-lang effects <name|hashprefix>` — show a def's inferred effect
/// signature and the guarantees derived from it (pure / mobile / cacheable).
fn cmd_effects(target: &str, json: bool, cb_path: &PathBuf) {
    use ai_lang::effects::infer_effects;

    let cb = Codebase::open(cb_path).expect("open codebase");
    let root_hash = match edit::resolve_target(&cb, target) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("effects: {}", e);
            std::process::exit(1);
        }
    };

    // Reconstruct a ResolvedModule from the transitive closure of the
    // target, so every referenced def is present and its `TopRef`s resolve
    // (mirrors `cmd_run`'s walk). Lambdas are inline (load_def → MissingDef);
    // skip them — they're analyzed within their enclosing def's body.
    let mut wanted: Vec<Hash> = Vec::new();
    let mut seen: std::collections::HashSet<Hash> = std::collections::HashSet::new();
    let mut frontier: Vec<Hash> = vec![root_hash];
    seen.insert(root_hash);
    while let Some(h) = frontier.pop() {
        let def = match cb.load_def(&h) {
            Ok(d) => d,
            Err(ai_lang::codebase::CodebaseError::MissingDef(_)) => continue,
            Err(e) => {
                eprintln!("effects: load_def {}: {:?}", h, e);
                std::process::exit(1);
            }
        };
        wanted.push(h);
        let mut deps: Vec<Hash> = Vec::new();
        let mut local_seen: std::collections::HashSet<Hash> = std::collections::HashSet::new();
        ai_lang::knowledge::walk_def_deps(&def, &mut deps, &mut local_seen);
        for d in deps {
            if seen.insert(d) {
                frontier.push(d);
            }
        }
    }
    let mut defs: Vec<ResolvedDef> = Vec::with_capacity(wanted.len());
    for h in wanted {
        let def = cb.load_def(&h).expect("load_def");
        let name = cb
            .names()
            .iter()
            .find(|(_, hh)| **hh == h)
            .map(|(n, _)| n.clone())
            .unwrap_or_else(|| format!("def_{}", &h.to_hex()[..8]));
        defs.push(ResolvedDef { name, hash: h, def });
    }
    let rm = ResolvedModule {
        defs,
        at_binding: None,
        externs: std::collections::HashMap::new(),
    };

    let sigs = infer_effects(&rm);
    let sig = match sigs.get(&root_hash) {
        Some(s) => *s,
        None => {
            eprintln!("effects: {} has no inferable body (a struct/enum?)", target);
            std::process::exit(1);
        }
    };

    // Render which parameters the effect is polymorphic over.
    let dep_indices: Vec<usize> = (0..64).filter(|j| sig.param_deps & (1u64 << j) != 0).collect();

    if json {
        let effs: Vec<&str> = [
            (EffectSet::IO, "IO"),
            (EffectSet::NET, "Net"),
            (EffectSet::STATE, "State"),
            (EffectSet::ATOM, "Atom"),
            (EffectSet::MUT, "Mut"),
            (EffectSet::FFI, "FFI"),
        ]
        .into_iter()
        .filter(|(bit, _)| sig.concrete.contains(*bit))
        .map(|(_, n)| n)
        .collect();
        let arr = |xs: &[String]| format!("[{}]", xs.join(","));
        println!(
            "{{\"target\":\"{}\",\"effects\":[{}],\"param_deps\":[{}],\"pure\":{},\"mobile\":{},\"cacheable\":{}}}",
            json_escape(target),
            effs.iter().map(|e| format!("\"{}\"", e)).collect::<Vec<_>>().join(","),
            dep_indices.iter().map(|j| j.to_string()).collect::<Vec<_>>().join(","),
            sig.is_pure(),
            sig.is_mobile(),
            sig.cacheable(),
        );
        let _ = arr;
    } else {
        let yn = |b: bool| if b { "yes" } else { "no" };
        println!("{}", target);
        println!("  effects:   {:?}", sig.concrete);
        if !dep_indices.is_empty() {
            let ps: Vec<String> = dep_indices.iter().map(|j| format!("#{j}")).collect();
            println!(
                "  + the effect of its argument(s): {}  (effect-polymorphic)",
                ps.join(", ")
            );
        }
        println!("  pure:      {}", yn(sig.is_pure()));
        println!("  mobile:    {}", yn(sig.is_mobile()));
        println!("  cacheable: {}", yn(sig.cacheable()));
        println!();
        println!("  (mobile = safe to ship to another thread or node;");
        println!("   cacheable = deterministic, safe to memoize across at())");
    }
}

/// `ai-lang view <name|hashprefix>` — project a stored def back to source.
fn cmd_view(target: &str, json: bool, cb_path: &PathBuf) {
    let cb = Codebase::open(cb_path).expect("open codebase");
    let hash = match edit::resolve_target(&cb, target) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("view: {}", e);
            std::process::exit(1);
        }
    };
    let source = match print_def(&cb, hash) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("view: {}", e);
            std::process::exit(1);
        }
    };
    if json {
        println!(
            "{{\"hash\":\"{}\",\"source\":\"{}\"}}",
            hash.to_hex(),
            json_escape(&source)
        );
    } else {
        println!("{}", source);
    }
}

// =============================================================================
// `eval`
// =============================================================================

fn cmd_eval(name: &str, arg_strs: &[String], _json: bool, cb_path: &PathBuf) {
    init_native_target().expect("init native target");
    let cb = Codebase::open(cb_path).expect("open codebase");
    let hash = match cb.get_name(name) {
        Some(h) => h,
        None => {
            eprintln!("eval: unknown name: {} (try `ai-lang ls`)", name);
            std::process::exit(1);
        }
    };

    // Look up the def's type.
    let scheme = match cb.types().get(&hash) {
        Some(s) => s.clone(),
        None => {
            eprintln!("eval: no cached type for {}; run `ai-lang add` first", name);
            std::process::exit(1);
        }
    };
    let (params, ret) = match &scheme {
        ai_lang::typecheck::TypeScheme::Fn { params, ret, .. } => (params.clone(), ret.clone()),
        _ => {
            eprintln!("eval: `{}` is not a function", name);
            std::process::exit(1);
        }
    };

    // Parse JSON args from the command line.
    let mut args_json: Vec<ai_lang::jsonl::Json> = Vec::new();
    for (i, s) in arg_strs.iter().enumerate() {
        match parse_json_literal(s) {
            Ok(j) => args_json.push(j),
            Err(e) => {
                eprintln!("eval: arg {} is not valid JSON: {}", i, e);
                std::process::exit(1);
            }
        }
    }

    match ai_lang::evalrun::eval(&cb, hash, &params, &ret, &args_json) {
        Ok(result) => {
            println!("{}", result.value.to_string());
        }
        Err(e) => {
            eprintln!("eval: {}: {}", e.kind(), e.message());
            std::process::exit(1);
        }
    }
}

/// Parse a simple JSON literal from a string.
fn parse_json_literal(s: &str) -> Result<ai_lang::jsonl::Json, String> {
    use ai_lang::jsonl::Json;
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') {
        return Ok(Json::Str(s[1..s.len()-1].to_owned()));
    }
    if s == "true" { return Ok(Json::Bool(true)); }
    if s == "false" { return Ok(Json::Bool(false)); }
    if s == "null" { return Ok(Json::Null); }
    if let Ok(n) = s.parse::<i64>() { return Ok(Json::Int(n)); }
    if let Ok(f) = s.parse::<f64>() { return Ok(Json::Float(f)); }
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len()-1];
        if inner.trim().is_empty() {
            return Ok(Json::Array(Vec::new()));
        }
        let mut arr = Vec::new();
        for part in split_json_array(inner) {
            arr.push(parse_json_literal(part)?);
        }
        return Ok(Json::Array(arr));
    }
    // Object (delegate to the JSON parser for robustness)
    if s.starts_with('{') && s.ends_with('}') {
        return match ai_lang::jsonl::parse(s) {
            Ok(j) => Ok(j),
            Err(e) => Err(format!("invalid JSON object: {}", e)),
        };
    }
    Err(format!("cannot parse as JSON: {}", s))
}

/// Split a JSON array body by top-level commas (respecting nesting).
fn split_json_array(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '[' | '{' => depth += 1,
            ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = s[start..].trim();
    if !last.is_empty() {
        parts.push(last);
    }
    parts
}

// =============================================================================
// `cli` — entry points with CLI arg parsing
// =============================================================================

fn cmd_cli_list(cb_path: &PathBuf) {
    let cb = Codebase::open(cb_path).expect("open codebase");
    let tag = "ai:cli";
    let hashes = match ai_lang::metadata::hashes_with_tag(cb.root(), tag) {
        Ok(hs) => hs,
        Err(e) => {
            eprintln!("cli: cannot scan metadata: {}", e);
            std::process::exit(1);
        }
    };
    if hashes.is_empty() {
        println!("no entry points found (tag a def with `ai-lang tag <name> ai:cli`)");
        return;
    }
    println!("{:<32}  {}", "NAME", "DESCRIPTION");
    println!("{}", "-".repeat(64));
    for h in &hashes {
        let name = cb.names().iter()
            .find(|(_, hh)| *hh == h)
            .map(|(n, _)| n.clone())
            .unwrap_or_else(|| format!("def_{}", &h.to_hex()[..8]));
        let loaded = ai_lang::metadata::load(&cb, h);
        let doc: &str = match &loaded {
            Some(m) => {
                if let Some(d) = m.get("doc").and_then(|d| d.as_str()) {
                    d
                } else if let Some(mm) = m.get("meta") {
                    mm.get("doc").and_then(|d| d.as_str()).unwrap_or("")
                } else {
                    ""
                }
            }
            None => "",
        };
        println!("{:<32}  {}", name, doc);
    }
}

fn cmd_cli_help(name: &str, cb_path: &PathBuf) {
    let cb = Codebase::open(cb_path).expect("open codebase");
    let hash = match cb.get_name(name) {
        Some(h) => h,
        None => {
            eprintln!("cli: unknown name: {}", name);
            std::process::exit(1);
        }
    };
    let scheme = match cb.types().get(&hash) {
        Some(s) => s.clone(),
        None => {
            eprintln!("cli: no type for {}", name);
            std::process::exit(1);
        }
    };
    let (params, _ret) = match &scheme {
        ai_lang::typecheck::TypeScheme::Fn { params, ret, .. } => (params.clone(), ret.clone()),
        _ => {
            eprintln!("cli: `{}` is not a function", name);
            std::process::exit(1);
        }
    };
    // Get param names from localnames side-car.
    let param_names: Vec<String> = cb.load_local_names(&hash).ok().flatten().unwrap_or_else(|| {
        (0..params.len()).map(|i| format!("arg{}", i)).collect()
    });
    let param_names: Vec<String> = (0..params.len())
        .map(|i| param_names.get(i).cloned().unwrap_or_else(|| format!("arg{}", i)))
        .collect();

    let meta = ai_lang::metadata::load(&cb, &hash);
    let doc: &str = if let Some(ref m) = meta {
        if let Some(d) = m.get("doc").and_then(|d| d.as_str()) {
            d
        } else if let Some(mm) = m.get("meta") {
            mm.get("doc").and_then(|d| d.as_str()).unwrap_or("")
        } else {
            ""
        }
    } else {
        ""
    };
    let param_meta = meta.as_ref()
        .and_then(|m| m.get("meta"))
        .and_then(|m| m.get("params"));

    if !doc.is_empty() {
        println!("{} — {}", name, doc);
    } else {
        println!("{}", name);
    }
    println!();
    print!("Usage: ai-lang cli {} ", name);
    for (i, _p) in params.iter().enumerate() {
        let pname = &param_names[i];
        let is_flag = param_meta
            .and_then(|pm| pm.get(pname))
            .and_then(|p| p.get("flag").and_then(|f| f.as_bool()))
            .unwrap_or(false);
        let has_default = param_meta
            .and_then(|pm| pm.get(pname))
            .and_then(|p| p.get("default"))
            .is_some();
        if is_flag {
            print!("[--{}] ", pname);
        } else if has_default {
            print!("[<{}>] ", pname);
        } else {
            print!("<{}> ", pname);
        }
    }
    println!();
    println!();
    println!("Arguments:");
    for (i, _p) in params.iter().enumerate() {
        let pname = &param_names[i];
        let info = param_meta.and_then(|pm| pm.get(pname));
        let pdoc = info.and_then(|i| i.get("doc").and_then(|d| d.as_str())).unwrap_or("");
        let is_flag = info.and_then(|i| i.get("flag").and_then(|f| f.as_bool())).unwrap_or(false);
        let defval = info.and_then(|i| i.get("default").and_then(|d| d.as_str()));
        if is_flag {
            println!("  --{:<20} {}", pname, pdoc);
        } else if let Some(dv) = defval {
            println!("  --{:<10} (default: {:<8}) {}", pname, dv, pdoc);
        } else {
            println!("  {:<12} {}", pname, pdoc);
        }
    }
}

fn cmd_cli_run(name: &str, arg_strs: &[String], cb_path: &PathBuf) {
    init_native_target().expect("init native target");
    let cb = Codebase::open(cb_path).expect("open codebase");
    let hash = match cb.get_name(name) {
        Some(h) => h,
        None => {
            eprintln!("cli: unknown name: {}", name);
            std::process::exit(1);
        }
    };
    let scheme = match cb.types().get(&hash) {
        Some(s) => s.clone(),
        None => {
            eprintln!("cli: no type for {}", name);
            std::process::exit(1);
        }
    };
    let (params, ret) = match &scheme {
        ai_lang::typecheck::TypeScheme::Fn { params, ret, .. } => (params.clone(), ret.clone()),
        _ => {
            eprintln!("cli: `{}` is not a function", name);
            std::process::exit(1);
        }
    };

    let meta = ai_lang::metadata::load(&cb, &hash);
    // Get param names from localnames side-car.
    let param_names: Vec<String> = cb.load_local_names(&hash).ok().flatten().unwrap_or_else(|| {
        (0..params.len()).map(|i| format!("arg{}", i)).collect()
    });
    // But localnames has ALL locals (params + lets). Truncate to param count.
    let param_names: Vec<String> = param_names.into_iter().take(params.len()).collect();
    // Pad if localnames had fewer entries than params.
    let param_names: Vec<String> = (0..params.len())
        .map(|i| param_names.get(i).cloned().unwrap_or_else(|| format!("arg{}", i)))
        .collect();

    let param_meta = meta.as_ref()
        .and_then(|m| m.get("meta"))
        .and_then(|m| m.get("params"));

    // Build default values from metadata.
    let mut values: Vec<Option<ai_lang::jsonl::Json>> = vec![None; params.len()];

    // Parse CLI args.
    let mut pos = 0usize;
    let mut i = 0usize;
    while i < arg_strs.len() {
        let a = &arg_strs[i];
        if a == "--help" {
            cmd_cli_help(name, cb_path);
            return;
        }
        if let Some(flag) = a.strip_prefix("--") {
            // Find which param has this name.
            if let Some(idx) = param_names.iter().position(|n| n == flag) {
                let ty = params.get(idx).cloned().unwrap_or(ai_lang::ast::Type::Builtin("Bool".to_string()));
                // Bool flag: map Bool→Int for Int params, Bool→Bool otherwise.
                let val = match &ty {
                    ai_lang::ast::Type::Builtin(b) if b == "Int" => ai_lang::jsonl::Json::Int(1),
                    _ => ai_lang::jsonl::Json::Bool(true),
                };
                values[idx] = Some(val);
            } else {
                // --key value style: next arg is the value.
                i += 1;
                if i < arg_strs.len() {
                    if let Some(idx) = param_names.iter().position(|n| n == flag) {
                        let ty = params.get(idx).cloned().unwrap_or(ai_lang::ast::Type::Builtin("String".to_string()));
                        values[idx] = Some(type_parse(&arg_strs[i], &ty));
                    }
                }
            }
            i += 1;
            continue;
        }
        // Positional arg: assign to next unfilled param.
        while pos < params.len() && values[pos].is_some() {
            pos += 1;
        }
        if pos < params.len() {
            values[pos] = Some(type_parse(a, &params[pos]));
            pos += 1;
        }
        i += 1;
    }

    // Fill defaults from metadata.
    for idx in 0..params.len() {
        if values[idx].is_none() {
            let pname = &param_names[idx];
            let defval = param_meta
                .and_then(|pm| pm.get(pname))
                .and_then(|p| p.get("default"));
            if let Some(dv) = defval {
                values[idx] = Some(dv.clone());
            }
        }
    }

    // Check required args are filled.
    let args_json: Vec<ai_lang::jsonl::Json> = values.iter().enumerate().map(|(idx, v)| {
        match v {
            Some(j) => j.clone(),
            None => {
                eprintln!("cli: missing required argument: {}", param_names[idx]);
                eprintln!("  run `ai-lang cli {} --help` for usage", name);
                std::process::exit(1);
            }
        }
    }).collect();

    match ai_lang::evalrun::eval(&cb, hash, &params, &ret, &args_json) {
        Ok(result) => {
            println!("{}", result.value.to_string());
        }
        Err(e) => {
            eprintln!("cli: {}: {}", e.kind(), e.message());
            std::process::exit(1);
        }
    }
}

fn type_parse(s: &str, ty: &ai_lang::ast::Type) -> ai_lang::jsonl::Json {
    use ai_lang::ast::Type;
    use ai_lang::jsonl::Json;
    match ty {
        Type::Builtin(name) => match name.as_str() {
            "Int" => Json::Int(s.parse().unwrap_or(0)),
            "Float" => Json::Float(s.parse().unwrap_or(0.0)),
            "Bool" => Json::Bool(s == "true" || s == "1"),
            "String" | _ => Json::Str(s.to_string()),
        },
        _ => Json::Str(s.to_string()),
    }
}

// =============================================================================
// `propagate`
// =============================================================================

fn cmd_propagate(name: &str, dry_run: bool, json: bool, cb_path: &PathBuf) {
    let mut cb = Codebase::open(cb_path).expect("open codebase");
    let res = if dry_run {
        edit::propagate_dry_run(&mut cb, name)
    } else {
        edit::propagate(&mut cb, name)
    };
    match res {
        Ok(r) => {
            if json {
                print_update_json(&r, dry_run);
            } else if r.propagated.is_empty() {
                println!("`{}` has no dependents to propagate to", name);
            } else {
                println!("`{}` is reachable from {} dependent(s):", name, r.propagated.len());
                for c in &r.propagated {
                    let nm = c.name.as_deref().unwrap_or("<unnamed>");
                    println!("  {}  {}", nm, &c.old.to_hex()[..12]);
                }
            }
        }
        Err(e) => {
            if json {
                println!(
                    "{{\"ok\":false,\"error\":{{\"message\":\"{}\"}}}}",
                    json_escape(&e.to_string())
                );
            } else {
                eprintln!("propagate: {}", e);
            }
            std::process::exit(1);
        }
    }
}

// =============================================================================
// `todos`
// =============================================================================

fn cmd_todos(json: bool, cb_path: &PathBuf) {
    use ai_lang::todostore::TodoStore;
    let cb = Codebase::open(cb_path).expect("open codebase");
    let branch = ai_lang::namespace::current_branch(&cb).unwrap_or_else(|_| "main".to_string());
    let store = match TodoStore::load(cb.root(), &branch) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("todos: cannot load todo store: {}", e);
            std::process::exit(1);
        }
    };
    let todos = store.list();
    if json {
        let items: Vec<String> = todos
            .iter()
            .map(|t| {
                format!(
                    "{{\"hash\":\"{}\",\"name\":{},\"message\":\"{}\"}}",
                    t.hash.to_hex(),
                    json_opt_name(&t.name),
                    json_escape(&t.message)
                )
            })
            .collect();
        println!("{{\"branch\":\"{}\",\"todos\":[{}]}}", json_escape(&branch), items.join(","));
    } else if todos.is_empty() {
        println!("no pending todos on branch `{}`", branch);
    } else {
        println!("{} todo(s) on branch `{}`:", todos.len(), branch);
        for t in &todos {
            let nm = t.name.as_deref().unwrap_or("<unnamed>");
            println!("  {}  {}  {}", nm, &t.hash.to_hex()[..12], t.message);
        }
    }
}

// =============================================================================
// `run`
// =============================================================================

fn cmd_run(name: &str, num_nodes: usize, user_args: Vec<String>, cb_path: &PathBuf) {
    init_native_target().expect("init native target");

    let cb = Codebase::open(cb_path).expect("open codebase");
    let root_hash = match cb.get_name(name) {
        Some(h) => h,
        None => {
            eprintln!("unknown name: {} (try `ai-lang ls`)", name);
            std::process::exit(1);
        }
    };

    // Reconstruct a ResolvedModule by walking the transitive closure
    // of `root_hash`. Loading "all currently-named defs" would be
    // wrong: names are mutable, so a prior `add` of a different
    // program can leave dangling references — e.g. `step_grid`'s
    // body has a fixed TopRef(HASH_G) to gol's compute_cell_bit,
    // but names["compute_cell_bit"] now points to mandel's HASH_M.
    // Walking from the root means we load exactly the defs the
    // entry point actually references, by hash. Other programs in
    // the same codebase can coexist without colliding.
    //
    // We assign each def a name based on the current names table
    // (looking up by hash); defs that don't have a name in the
    // current table get a synthetic `def_<8hex>` name so the
    // resolver/codegen can still dispatch.
    // The walker also surfaces lambda content hashes (lambdas are
    // referenced inline from def bodies but stored ONLY inline,
    // not as separate `.def` files). Skip hashes whose `load_def`
    // returns MissingDef — those are lambdas, not real defs.
    let mut wanted: Vec<Hash> = Vec::new();
    let mut seen: std::collections::HashSet<Hash> = std::collections::HashSet::new();
    let mut frontier: Vec<Hash> = vec![root_hash];
    seen.insert(root_hash);
    while let Some(h) = frontier.pop() {
        let def = match cb.load_def(&h) {
            Ok(d) => d,
            Err(ai_lang::codebase::CodebaseError::MissingDef(_)) => continue,
            Err(e) => {
                eprintln!("load_def {}: {:?}", h, e);
                std::process::exit(1);
            }
        };
        wanted.push(h);
        let mut deps: Vec<Hash> = Vec::new();
        let mut local_seen: std::collections::HashSet<Hash> =
            std::collections::HashSet::new();
        ai_lang::knowledge::walk_def_deps(&def, &mut deps, &mut local_seen);
        for d in deps {
            if seen.insert(d) {
                frontier.push(d);
            }
        }
    }
    // Build a hash → name reverse map from current names. Multiple
    // names may collide on a hash (idempotent re-adds); pick the
    // shortest-then-lexicographically-smallest deterministically.
    let mut name_for: HashMap<Hash, String> = HashMap::new();
    for (n, h) in cb.names().iter() {
        match name_for.get(h) {
            None => {
                name_for.insert(*h, n.clone());
            }
            Some(existing)
                if (n.len(), n.as_str()) < (existing.len(), existing.as_str()) =>
            {
                name_for.insert(*h, n.clone());
            }
            _ => {}
        }
    }
    let mut defs: Vec<ResolvedDef> = Vec::with_capacity(wanted.len());
    // Sort by hash for deterministic ordering — codegen does its
    // own dep ordering internally so the input order isn't load-bearing.
    wanted.sort_by(|a, b| a.to_hex().cmp(&b.to_hex()));
    for h in wanted {
        let def = cb.load_def(&h).expect("load_def");
        let name = name_for
            .get(&h)
            .cloned()
            .unwrap_or_else(|| format!("def_{}", &h.to_hex()[..8]));
        defs.push(ResolvedDef { name, hash: h, def });
    }

    let externs = stdlib_externs();
    let at_binding = at_binding_from_codebase(&cb);
    let rm = ResolvedModule {
        defs,
        at_binding,
        externs,
    };
    let wanted_hashes: Vec<Hash> = rm.defs.iter().map(|d| d.hash).collect();

    // Spawn workers (if requested) and publish their ports to the
    // global table that `get_node_port(i)` reads from.
    let mut servers: Vec<ServerChild> = Vec::new();
    if num_nodes > 0 {
        eprintln!("spawning {} worker processes...", num_nodes);
        let t0 = std::time::Instant::now();
        for i in 0..num_nodes {
            servers.push(spawn_worker(i).expect("spawn worker"));
        }
        let ports: Vec<u16> = servers.iter().map(|s| s.port).collect();
        eprintln!(
            "  {} workers ready in {:.1}s (first 6 PIDs: {:?})",
            num_nodes,
            t0.elapsed().as_secs_f64(),
            &servers.iter().map(|s| s.child.id()).take(6).collect::<Vec<_>>(),
        );
        set_worker_nodes(ports);
    }

    // Publish user args.
    set_user_args(user_args.clone());

    // JIT the reconstructed module and call <name>() -> Int.
    let ctx = Context::create();
    let cache_hit = ai_lang::codegen::load_bitcode_cache(&ctx, cb.root(), &wanted_hashes);
    let (cm, _from_cache) = if let Some(cached) = cache_hit {
        // Codegen was skipped, so dlsym-resolve the C externs it would
        // have registered (clock_gettime, malloc, ... — anything the
        // program calls through the C FFI).
        ai_lang::codegen::register_referenced_c_externs(&rm).expect("register C externs");
        (cached, true)
    } else {
        let built = CompiledModule::build(&ctx, &rm).expect("build module");
        let _ = ai_lang::codegen::write_bitcode_cache(&built, cb.root(), &wanted_hashes);
        (built, false)
    };
    // Diagnostic: dump the (optimized) LLVM IR of the module, or of one
    // def whose visible name is given (`AI_LANG_DUMP_IR=bfib`).
    if let Ok(filter) = std::env::var("AI_LANG_DUMP_IR") {
        let ir = cm.ir();
        if filter.is_empty() || filter == "1" {
            eprintln!("{}", ir);
        } else if let Some(h) = cb.get_name(&filter) {
            let sym = def_symbol(&h);
            let mut emit = false;
            for line in ir.lines() {
                if line.starts_with("define ") && line.contains(&sym) {
                    emit = true;
                }
                if emit {
                    eprintln!("{}", line);
                    if line == "}" {
                        break;
                    }
                }
            }
        }
    }
    let rt = Runtime::new_with_metadata(
        cm.closure_type_infos.clone(),
        cm.shape_registry.clone(),
        cm.shape_meta.clone(),
        cm.shape_by_type_id.clone(),
    );
    // The entry point's dual-register `{i64, i64}` Result ABI (if any),
    // captured before `cm` moves into the JIT so we call it correctly.
    let root_abi = cm.def_result_abi.get(&root_hash).copied();
    let jit = Jit::new(cm, &rt).expect("jit");

    install_current_runtime(&rt);
    // For at() to work end-to-end, the client also needs a knowledge
    // base to ship to servers on NeedCode. Build it from the reconstructed
    // resolved module — this captures every def + every reachable lambda.
    let kb = KnowledgeBase::build(&rm);
    install_current_knowledge_base(&kb);
    let _rb_storage;
    if let Some(rb) = rm
        .at_binding
        .as_ref()
        .and_then(|ab| build_at_runtime_binding(&rt, ab))
    {
        _rb_storage = rb;
        install_current_at_binding(&_rb_storage);
    }

    // No panic channel: a contract violation in the program aborts the
    // process with its message before this returns.
    //
    // A `-> Result<T, E>` entry uses the dual-register `{i64, i64}`
    // (tag, payload) ABI, so it must be called with the matching
    // signature. We print the Ok payload register on success and a
    // `Err(<bits>)` marker otherwise (a dev convenience — `ai-lang eval`
    // renders the full structured value).
    let sym = def_symbol(&root_hash);
    if let Some(abi) = root_abi {
        #[repr(C)]
        struct ResultPair {
            tag: i64,
            payload: i64,
        }
        let entry = unsafe {
            jit.engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> ResultPair>(&sym)
                .unwrap_or_else(|_| {
                    eprintln!("`{}` is not a callable entry function", name);
                    std::process::exit(1);
                })
        };
        let r = unsafe { entry.call(rt.thread_ptr()) };
        if r.tag == abi.ok_index as i64 {
            println!("{}", r.payload);
        } else {
            println!("Err({})", r.payload);
        }
    } else {
        let entry = unsafe {
            jit.engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&sym)
                .unwrap_or_else(|_| {
                    eprintln!("`{}` is not a `() -> Int` function", name);
                    std::process::exit(1);
                })
        };
        let result = unsafe { entry.call(rt.thread_ptr()) };
        println!("{}", result);
    }

    clear_at_conn_cache();
    clear_current_runtime();
    clear_current_knowledge_base();
    clear_current_at_binding();
    drop(servers); // kills workers
}

// =============================================================================
// `test`
// =============================================================================

fn cmd_test(prefix: Option<&str>, tag: Option<&str>, _all: bool, json: bool, cb_path: &PathBuf) {
    init_native_target().expect("init native target");

    let cb = Codebase::open(cb_path).expect("open codebase");

    let test_tag = tag.unwrap_or("ai:test");
    let mut test_names: Vec<(String, Hash)> = Vec::new();
    if let Ok(tagged_hashes) = ai_lang::metadata::hashes_with_tag(cb.root(), test_tag) {
        for hash in tagged_hashes {
            // Find the name(s) for this hash.
            let names: Vec<&String> = cb.names().iter()
                .filter(|(_, h)| **h == hash)
                .map(|(n, _)| n)
                .collect();
            let name = match names.first() {
                Some(n) => (*n).clone(),
                None => continue,
            };
            // Only zero-arg Int-returning functions.
            let is_test_fn = match cb.types().get(&hash) {
                Some(ai_lang::typecheck::TypeScheme::Fn { params, ret, .. }) => {
                    params.is_empty() && *ret == ai_lang::ast::Type::Builtin("Int".to_string())
                }
                _ => false,
            };
            if is_test_fn {
                if let Some(p) = prefix {
                    if !name.starts_with(p) {
                        continue;
                    }
                }
                test_names.push((name, hash));
            }
        }
    }
    test_names.sort_by(|a, b| a.0.cmp(&b.0));

    if test_names.is_empty() {
        println!("no tests found");
        return;
    }

    // Compute effects for all tests to know which are pure.
    let rm = {
        let mut all_hashes: Vec<Hash> = test_names.iter().map(|(_, h)| *h).collect();
        all_hashes.sort_by_key(|h| h.to_hex());
        all_hashes.dedup();
        let mut seen: std::collections::HashSet<Hash> = std::collections::HashSet::new();
        let mut frontier: Vec<Hash> = all_hashes.clone();
        for h in &all_hashes {
            seen.insert(*h);
        }
        while let Some(h) = frontier.pop() {
            if let Ok(def) = cb.load_def(&h) {
                let mut deps: Vec<Hash> = Vec::new();
                let mut local_seen = std::collections::HashSet::new();
                ai_lang::knowledge::walk_def_deps(&def, &mut deps, &mut local_seen);
                for d in deps {
                    if seen.insert(d) {
                        frontier.push(d);
                    }
                }
            }
        }
        let defs: Vec<ai_lang::resolve::ResolvedDef> = seen
            .iter()
            .filter_map(|h| {
                cb.load_def(h).ok().map(|def| {
                    let name = cb
                        .names()
                        .iter()
                        .find(|(_, hh)| *hh == h)
                        .map(|(n, _)| n.clone())
                        .unwrap_or_else(|| format!("def_{}", &h.to_hex()[..8]));
                    ai_lang::resolve::ResolvedDef {
                        name,
                        hash: *h,
                        def,
                    }
                })
            })
            .collect();
        ai_lang::resolve::ResolvedModule {
            defs,
            at_binding: None,
            externs: std::collections::HashMap::new(),
        }
    };
    let effect_sigs = ai_lang::effects::infer_effects(&rm);

    // Ensure test cache directory exists.
    let cache_dir = cb.root().join("test-cache");
    let _ = std::fs::create_dir_all(&cache_dir);

    // Build the full module for JIT (shared across all tests), with cache.
    let ctx = Context::create();
    let wanted: Vec<Hash> = rm.defs.iter().map(|d| d.hash).collect();
    let cache_hit = ai_lang::codegen::load_bitcode_cache(&ctx, cb.root(), &wanted);
    let (cm, _from_cache) = if let Some(cached) = cache_hit {
        ai_lang::codegen::register_referenced_c_externs(&rm).expect("register C externs");
        (cached, true)
    } else {
        let built = CompiledModule::build(&ctx, &rm).expect("build module");
        let _ = ai_lang::codegen::write_bitcode_cache(&built, cb.root(), &wanted);
        (built, false)
    };
    let rt = Runtime::new_with_metadata(
        cm.closure_type_infos.clone(),
        cm.shape_registry.clone(),
        cm.shape_meta.clone(),
        cm.shape_by_type_id.clone(),
    );
    let jit = Jit::new(cm, &rt).expect("jit");
    install_current_runtime(&rt);

    let mut passed = 0u32;
    let mut failed = 0u32;
    let mut skipped = 0u32;
    let mut json_results: Vec<String> = Vec::new();

    for (name, hash) in &test_names {
        let is_pure = effect_sigs.get(hash).map(|s| s.is_pure()).unwrap_or(false);
        let cache_path = cache_dir.join(format!("{}.passed", hash.to_hex()));

        if is_pure && cache_path.exists() {
            skipped += 1;
            if !json {
                println!("  PASS  {}  (cached)", name);
            }
            json_results.push(format!(
                "{{\"name\":\"{}\",\"hash\":\"{}\",\"result\":\"pass\",\"cached\":true}}",
                json_escape(name),
                &hash.to_hex()[..16]
            ));
            continue;
        }

        let entry = unsafe {
            match jit
                .engine
                .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(&def_symbol(hash))
            {
                Ok(f) => f,
                Err(_) => {
                    eprintln!("  FAIL  {}  (not a () -> Int function)", name);
                    failed += 1;
                    continue;
                }
            }
        };

        // No panic channel: a contract violation in a test aborts the
        // whole run with its message (a bug is a bug).
        let result = unsafe { entry.call(rt.thread_ptr()) };
        if result == 0 {
            passed += 1;
            if !json {
                println!("  PASS  {}", name);
            }
            if is_pure {
                let _ = std::fs::write(&cache_path, "");
            }
            json_results.push(format!(
                "{{\"name\":\"{}\",\"hash\":\"{}\",\"result\":\"pass\",\"pure\":{}}}",
                json_escape(name),
                &hash.to_hex()[..16],
                is_pure
            ));
        } else {
            failed += 1;
            if !json {
                println!("  FAIL  {}  (returned {})", name, result);
            }
            json_results.push(format!(
                "{{\"name\":\"{}\",\"hash\":\"{}\",\"result\":\"fail\",\"returned\":{},\"pure\":{}}}",
                json_escape(name),
                &hash.to_hex()[..16],
                result,
                is_pure
            ));
        }
    }

    clear_current_runtime();

    if json {
        println!(
            "{{\"passed\":{},\"failed\":{},\"skipped\":{},\"total\":{},\"results\":[{}]}}",
            passed,
            failed,
            skipped,
            test_names.len(),
            json_results.join(",")
        );
    } else {
        eprintln!(
            "\n  {} passed, {} failed, {} cached",
            passed, failed, skipped
        );
    }

    if failed > 0 {
        std::process::exit(1);
    }
}

// =============================================================================
// `serve`
// =============================================================================

/// Become a node. Two modes:
///
/// - **Child worker** (no `--bind`): loopback ephemeral port, exits when
///   the parent's stdin pipe closes. This is what `run --nodes=N`
///   spawns.
/// - **Standalone** (`--bind ADDR`): binds the given address, no
///   watchdog, optional `--token` shared-secret auth per connection.
///
/// Both modes serve the full deploy-capable protocol: `at()` calls with
/// the code-fetch handshake, plus deploy / rollback / invoke.
fn cmd_serve(bind_addr: Option<&str>, token: Option<&str>) {
    init_native_target().expect("init native target");

    let ctx = Context::create();
    let empty_m = parse_module("").expect("parse empty");
    let empty_r = resolve_module(&empty_m).expect("resolve empty");
    let empty_cm = CompiledModule::build(&ctx, &empty_r).expect("build empty");
    let mut rt = Runtime::new_with_metadata(
        empty_cm.closure_type_infos.clone(),
        empty_cm.shape_registry.clone(),
        empty_cm.shape_meta.clone(),
        empty_cm.shape_by_type_id.clone(),
    );
    let mut jit = IncrementalJit::new(empty_cm, &rt).expect("incremental jit");
    let mut mgr = DeployManager::default();

    let standalone = bind_addr.is_some();
    let listener = bind(bind_addr.unwrap_or("127.0.0.1:0")).expect("bind");
    let port = listener.local_addr().unwrap().port();
    println!("READY {}", port);
    let _ = io::stdout().flush();

    if !standalone {
        // Watchdog: parent's stdin pipe closes when parent dies → we exit.
        std::thread::spawn(|| {
            let stdin = io::stdin();
            let mut buf = [0u8; 64];
            loop {
                match stdin.lock().read(&mut buf) {
                    Ok(0) | Err(_) => std::process::exit(0),
                    Ok(_) => {}
                }
            }
        });
    }

    eprintln!(
        "[ai-lang serve pid={} port={}{}{}] empty runtime, awaiting code/deploys",
        std::process::id(),
        port,
        if standalone { " standalone" } else { "" },
        if token.is_some() { " auth" } else { "" },
    );

    loop {
        let mut stream = match listener.accept() {
            Ok((s, _)) => s,
            Err(e) => {
                eprintln!("[ai-lang serve] accept: {}", e);
                break;
            }
        };
        let _ = stream.set_nodelay(true);
        if let Some(tok) = token {
            if server_expect_auth(&mut stream, tok).is_err() {
                // Drop silently: don't tell a prober why.
                eprintln!("[ai-lang serve] rejected unauthenticated connection");
                continue;
            }
        }
        loop {
            match unsafe { serve_deploy_turn(&mut rt, &mut jit, &mut mgr, &mut stream) } {
                Ok(()) => continue,
                // Peer closed cleanly between frames — normal shutdown,
                // not an error. Silently drop the connection and go
                // back to accepting.
                Err(ai_lang::net::NetError::ConnectionClosed) => break,
                Err(e) => {
                    eprintln!(
                        "[ai-lang serve pid={}] error on connection: {}",
                        std::process::id(),
                        e
                    );
                    break;
                }
            }
        }
    }
}

/// Pull `--token <t>` / `--token=<t>` out of `args`; everything else is
/// returned as positionals.
fn take_token(args: &[String]) -> (Vec<String>, Option<String>) {
    let mut positional = Vec::new();
    let mut token = None;
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--token" && i + 1 < args.len() {
            token = Some(args[i + 1].clone());
            i += 2;
        } else if let Some(rest) = args[i].strip_prefix("--token=") {
            token = Some(rest.to_owned());
            i += 1;
        } else {
            positional.push(args[i].clone());
            i += 1;
        }
    }
    (positional, token)
}

/// Connect to a node, authenticating if a token was given.
fn connect_node(addr: &str, token: Option<&str>) -> std::net::TcpStream {
    let mut stream = std::net::TcpStream::connect(addr).unwrap_or_else(|e| {
        eprintln!("cannot connect to node {}: {}", addr, e);
        std::process::exit(1);
    });
    let _ = stream.set_nodelay(true);
    if let Some(tok) = token {
        if let Err(e) = client_authenticate(&mut stream, tok) {
            eprintln!("auth to {} failed: {}", addr, e);
            std::process::exit(1);
        }
    }
    stream
}

fn short(h: &Hash) -> String {
    h.to_hex()[..8].to_string()
}

fn print_deploy_report(report: &ai_lang::deploy::DeployReport) {
    for (name, action) in &report.states {
        match action {
            StateAction::Keep => println!("state {}: kept (cell already live)", name),
            StateAction::Fresh => println!("state {}: fresh (initializer ran)", name),
            StateAction::Carryover { from } => {
                println!("state {}: carried over from {} (initializer skipped)", name, short(from))
            }
            StateAction::Migrate { from, via } => {
                println!("state {}: migrated from {} via {}", name, short(from), short(via))
            }
        }
    }
    for name in &report.dropped_states {
        println!("state {}: dropped (cell stays resident, untracked)", name);
    }
    for (name, new, prev) in &report.rebound {
        match prev {
            Some(p) => println!("binding {} -> {} (was {})", name, short(new), short(p)),
            None => println!("binding {} -> {} (created)", name, short(new)),
        }
    }
}

/// Parse + run `ai-lang deploy <file.ail> <addr> --bind B=DEF
/// [--migrate STATE=DEF] [--allow-state-drop] [--token T]`.
fn cmd_deploy_cli(args: &[String]) {
    let mut positional: Vec<String> = Vec::new();
    let mut binds: Vec<(String, String)> = Vec::new();
    let mut migrates: Vec<(String, String)> = Vec::new();
    let mut allow_drop = false;
    let mut token: Option<String> = None;

    fn split_eq(flag: &str, v: &str) -> (String, String) {
        match v.split_once('=') {
            Some((a, b)) if !a.is_empty() && !b.is_empty() => (a.to_string(), b.to_string()),
            _ => {
                eprintln!("{} expects NAME=DEF, got `{}`", flag, v);
                std::process::exit(2);
            }
        }
    }

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a == "--bind" && i + 1 < args.len() {
            binds.push(split_eq("--bind", &args[i + 1]));
            i += 2;
        } else if let Some(rest) = a.strip_prefix("--bind=") {
            binds.push(split_eq("--bind", rest));
            i += 1;
        } else if a == "--migrate" && i + 1 < args.len() {
            migrates.push(split_eq("--migrate", &args[i + 1]));
            i += 2;
        } else if let Some(rest) = a.strip_prefix("--migrate=") {
            migrates.push(split_eq("--migrate", rest));
            i += 1;
        } else if a == "--allow-state-drop" {
            allow_drop = true;
            i += 1;
        } else if a == "--token" && i + 1 < args.len() {
            token = Some(args[i + 1].clone());
            i += 2;
        } else if let Some(rest) = a.strip_prefix("--token=") {
            token = Some(rest.to_owned());
            i += 1;
        } else {
            positional.push(a.clone());
            i += 1;
        }
    }
    if positional.len() != 2 {
        eprintln!(
            "usage: ai-lang deploy <file.ail> <addr> --bind <binding>=<def> \
             [--migrate <state>=<def>] [--allow-state-drop] [--token <t>]"
        );
        std::process::exit(2);
    }
    cmd_deploy(&positional[0], &positional[1], &binds, &migrates, allow_drop, token.as_deref());
}

/// Build a deploy from a source file (stdlib in scope) and ship it.
/// Client-side: parse, resolve, typecheck (fail fast — the node would
/// refuse anyway), collect the transitive closure of the rebind targets
/// + migrations + every `state` in the program, and send. No JIT runs
/// here; the client is bytes-only.
fn cmd_deploy(
    file: &str,
    addr: &str,
    binds: &[(String, String)],
    migrates: &[(String, String)],
    allow_drop: bool,
    token: Option<&str>,
) {
    let source = std::fs::read_to_string(file).unwrap_or_else(|e| {
        eprintln!("deploy: cannot read {}: {}", file, e);
        std::process::exit(1);
    });
    let full = format!("{}\n{}", STDLIB, source);
    let m = parse_module(&full).unwrap_or_else(|e| {
        eprintln!("deploy: parse failed: {:?}", e);
        std::process::exit(1);
    });
    let r = resolve_module(&m).unwrap_or_else(|e| {
        eprintln!("deploy: resolve failed: {}", e);
        std::process::exit(1);
    });
    let mut tc = TypeCache::new();
    if let Err(e) = typecheck_module(&r, &mut tc) {
        eprintln!("deploy: program does not typecheck: {:?}", e);
        std::process::exit(1);
    }

    let by_name: HashMap<&str, Hash> =
        r.defs.iter().map(|d| (d.name.as_str(), d.hash)).collect();
    let lookup = |flag: &str, def_name: &str| -> Hash {
        *by_name.get(def_name).unwrap_or_else(|| {
            eprintln!("deploy: {} references `{}`, which is not a def in {}", flag, def_name, file);
            std::process::exit(1);
        })
    };

    let rebinds: Vec<(String, Hash)> = binds
        .iter()
        .map(|(b, def)| (b.clone(), lookup("--bind", def)))
        .collect();
    let migrations: HashMap<String, Hash> = migrates
        .iter()
        .map(|(state, def)| (state.clone(), lookup("--migrate", def)))
        .collect();

    // Ship the transitive closure of: rebind targets, migrations, and
    // every `state` in the program (states are roots even when no
    // shipped fn references them yet).
    let mut state_names: HashMap<String, Hash> = HashMap::new();
    let mut roots: Vec<Hash> = rebinds.iter().map(|(_, h)| *h).collect();
    roots.extend(migrations.values().copied());
    for d in &r.defs {
        if matches!(d.def, Def::State { .. }) {
            state_names.insert(d.name.clone(), d.hash);
            roots.push(d.hash);
        }
    }

    let kb = KnowledgeBase::build(&r);
    let order = kb.collect_transitive_deps(&roots).unwrap_or_else(|e| {
        eprintln!("deploy: dependency collection failed: {}", e);
        std::process::exit(1);
    });
    let mut items: Vec<(ItemKind, Hash, Vec<u8>)> = order
        .iter()
        .map(|h| {
            let (k, b) = kb.lookup(h).expect("dep in kb");
            (*k, *h, b.clone())
        })
        .collect();
    for (name, sig) in kb.extern_requirements(&order) {
        items.push((
            ItemKind::Extern,
            Hash::of_bytes(name.as_bytes()),
            encode_extern(&name, &sig.params, &sig.ret, sig.library.as_deref(), sig.variadic),
        ));
    }

    let req = DeployRequest {
        items,
        state_names,
        migrations,
        rebinds,
        allow_state_drop: allow_drop,
    };

    let mut stream = connect_node(addr, token);
    match deploy_on_channel(&mut stream, &req) {
        Ok(report) => {
            println!("deployed to {}", addr);
            print_deploy_report(&report);
        }
        Err(ai_lang::net::NetError::DeployRefused(msg)) => {
            eprintln!("deploy refused (node unchanged): {}", msg);
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("deploy failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn cmd_rollback(addr: &str, binding: &str, token: Option<&str>) {
    let mut stream = connect_node(addr, token);
    match rollback_on_channel(&mut stream, binding) {
        Ok(report) => {
            println!("rolled back on {}", addr);
            print_deploy_report(&report);
        }
        Err(ai_lang::net::NetError::DeployRefused(msg)) => {
            eprintln!("rollback refused: {}", msg);
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("rollback failed: {}", e);
            std::process::exit(1);
        }
    }
}

fn cmd_invoke(addr: &str, binding: &str, args: &[i64], token: Option<&str>) {
    let mut stream = connect_node(addr, token);
    match invoke_on_channel(&mut stream, binding, args) {
        Ok(v) => println!("{}", v),
        Err(ai_lang::net::NetError::DeployRefused(msg)) => {
            eprintln!("invoke refused: {}", msg);
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("invoke failed: {}", e);
            std::process::exit(1);
        }
    }
}

// =============================================================================
// Codebase helpers
// =============================================================================

/// Re-resolve the stdlib in isolation to recover its `externs` map.
/// User-declared externs are not yet supported by the codebase
/// runner — they'd need their own on-disk table.
fn stdlib_externs() -> HashMap<String, ExternSig> {
    let m = parse_module(STDLIB).expect("parse stdlib");
    let r = resolve_module(&m).expect("resolve stdlib");
    r.externs
}

/// Rebuild the resolver's `AtBinding` from named enum/struct defs in the
/// codebase. Returns `None` if any of `Result`/`Failure`/`Node` is
/// missing (the program doesn't use `at()`).
fn at_binding_from_codebase(cb: &Codebase) -> Option<AtBinding> {
    let result_hash = cb.get_name("Result")?;
    let failure_hash = cb.get_name("Failure")?;
    let node_hash = cb.get_name("Node")?;
    let result_def = cb.load_def(&result_hash).ok()?;
    let failure_def = cb.load_def(&failure_hash).ok()?;
    let result_variants = match result_def {
        Def::Enum { variants, .. } => variants,
        _ => return None,
    };
    let failure_variants = match failure_def {
        Def::Enum { variants, .. } => variants,
        _ => return None,
    };
    let find = |vs: &[(String, _)], n: &str| -> Option<u32> {
        vs.iter().position(|(name, _)| name == n).map(|i| i as u32)
    };
    let (decode_error_hash, decode_tm_idx, decode_mf_idx) = match cb.get_name("DecodeError") {
        Some(h) => match cb.load_def(&h) {
            Ok(Def::Enum { variants, .. }) => (
                Some(h),
                find(&variants, "TypeMismatch").unwrap_or(0),
                find(&variants, "Malformed").unwrap_or(0),
            ),
            _ => (None, 0, 0),
        },
        None => (None, 0, 0),
    };
    Some(AtBinding {
        result_hash,
        failure_hash,
        node_hash,
        ok_variant_index: find(&result_variants, "Ok")?,
        err_variant_index: find(&result_variants, "Err")?,
        unreachable_variant_index: find(&failure_variants, "Unreachable")?,
        crashed_variant_index: find(&failure_variants, "Crashed")?,
        code_missing_variant_index: find(&failure_variants, "CodeMissing")?,
        cancelled_variant_index: find(&failure_variants, "Cancelled")?,
        timed_out_variant_index: find(&failure_variants, "TimedOut"),
        decode_error_hash,
        decode_type_mismatch_index: decode_tm_idx,
        decode_malformed_index: decode_mf_idx,
    })
}

// =============================================================================
// Worker spawn
// =============================================================================

struct ServerChild {
    child: Child,
    port: u16,
}

impl Drop for ServerChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_worker(_idx: usize) -> io::Result<ServerChild> {
    let exe = std::env::current_exe()?;
    let mut child = Command::new(exe)
        .arg("serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdout = child.stdout.take().expect("stdout piped");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let port: u16 = line
        .trim()
        .strip_prefix("READY ")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("worker bad READY line: {:?}", line),
            )
        })?;
    drop(reader);
    Ok(ServerChild { child, port })
}

// =============================================================================
// Library slicing commands (Phase 5)
// =============================================================================

/// Open the codebase and load (or rebuild) the dependency index, exiting with a
/// clear message on failure. Mirrors the helper used by the read commands.
fn open_cb_and_index(flags: &Flags) -> (Codebase, DependencyIndex) {
    let root = cb_root(flags.codebase.as_ref());
    let cb = Codebase::open(&root).unwrap_or_else(|e| {
        eprintln!("open codebase: {}", e);
        std::process::exit(1);
    });
    let index = match DependencyIndex::load(cb.root()) {
        Ok(idx) => idx,
        Err(_) => DependencyIndex::rebuild_from_codebase(&cb).unwrap_or_else(|e| {
            eprintln!("index: {}", e);
            std::process::exit(1);
        }),
    };
    (cb, index)
}

fn slice_die(e: SliceError) -> ! {
    eprintln!("{}", e);
    std::process::exit(1);
}

/// `ai-lang export <name...> --out <path>`
///
/// Resolve each NAME to a hash, compute its transitive stored-def closure, and
/// write a portable bundle (def bytes + type bytes + name manifest) to the
/// `--out` path. Partial pull: only the chosen names and their dependencies
/// travel; unrelated defs do not.
fn cmd_export(args: &[String], flags: &Flags) {
    if args.is_empty() {
        eprintln!("export: need <name...> --out <path>");
        std::process::exit(2);
    }
    let out = flags.out.clone().unwrap_or_else(|| {
        eprintln!("export: --out <path> is required");
        std::process::exit(2);
    });
    let (cb, index) = open_cb_and_index(flags);
    let manifest =
        slice::export(&cb, &index, args, std::path::Path::new(&out)).unwrap_or_else(|e| slice_die(e));

    if flags.json {
        let mut s = String::from("{\"out\":");
        s.push_str(&json_str(&out));
        s.push_str(",\"defs\":[");
        for (i, h) in manifest.defs.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&json_str(&h.to_hex()));
        }
        s.push_str("],\"types\":[");
        for (i, h) in manifest.types.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&json_str(&h.to_hex()));
        }
        s.push_str("],\"roots\":[");
        for (i, (name, hash)) in manifest.names.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str("{\"name\":");
            s.push_str(&json_str(name));
            s.push_str(",\"hash\":");
            s.push_str(&json_str(&hash.to_hex()));
            s.push('}');
        }
        s.push_str("]}");
        println!("{}", s);
    } else {
        println!(
            "exported {} def(s) ({} with types) to {}",
            manifest.defs.len(),
            manifest.types.len(),
            out
        );
        for (name, hash) in &manifest.names {
            println!("  root {} = {}", name, &hash.to_hex()[..12]);
        }
    }
}

/// `ai-lang import <path>`
///
/// Read a bundle into the local store (idempotent: hashes already present are
/// no-ops — automatic dedup), then add the manifest names. Reports how many
/// defs/types were new vs already present.
fn cmd_import(args: &[String], flags: &Flags) {
    let path = args.first().unwrap_or_else(|| {
        eprintln!("import: need <path>");
        std::process::exit(2);
    });
    let root = cb_root(flags.codebase.as_ref());
    let mut cb = Codebase::open(&root).unwrap_or_else(|e| {
        eprintln!("open codebase: {}", e);
        std::process::exit(1);
    });
    let report =
        slice::import(&mut cb, std::path::Path::new(path)).unwrap_or_else(|e| slice_die(e));

    // The index is a pure cache; rebuild + persist so later read commands see
    // the freshly-imported defs.
    let index = DependencyIndex::rebuild_from_codebase(&cb).unwrap_or_else(|e| {
        eprintln!("index: {}", e);
        std::process::exit(1);
    });
    if let Err(e) = index.save(cb.root()) {
        eprintln!("warning: could not persist index: {}", e);
    }

    if flags.json {
        let mut s = String::from("{");
        s.push_str(&format!("\"new_defs\":{},", report.new_defs));
        s.push_str(&format!("\"existing_defs\":{},", report.existing_defs));
        s.push_str(&format!("\"new_types\":{},", report.new_types));
        s.push_str(&format!("\"existing_types\":{},", report.existing_types));
        s.push_str("\"names\":[");
        for (i, (name, hash)) in report.names.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str("{\"name\":");
            s.push_str(&json_str(name));
            s.push_str(",\"hash\":");
            s.push_str(&json_str(&hash.to_hex()));
            s.push('}');
        }
        s.push_str("]}");
        println!("{}", s);
    } else {
        println!(
            "imported: {} new def(s), {} already present; {} new type(s), {} already present",
            report.new_defs, report.existing_defs, report.new_types, report.existing_types
        );
        for (name, hash) in &report.names {
            println!("  name {} = {}", name, &hash.to_hex()[..12]);
        }
    }
}

/// `ai-lang find-by-type <signature>`
///
/// Parse a function signature like `(Int, Int) -> Int` (or a generic
/// `(T) -> T`) into a query `TypeScheme::Fn`, then return every stored def
/// whose cached scheme matches it (alpha-equivalent over type variables).
fn cmd_find_by_type(args: &[String], flags: &Flags) {
    if args.is_empty() {
        eprintln!("find-by-type: need <signature>, e.g. \"(Int) -> Int\"");
        std::process::exit(2);
    }
    // Allow the signature to be passed as several shell words.
    let sig = args.join(" ");
    let query = parse_signature(&sig).unwrap_or_else(|e| {
        eprintln!("find-by-type: {}", e);
        std::process::exit(2);
    });
    let root = cb_root(flags.codebase.as_ref());
    let cb = Codebase::open(&root).unwrap_or_else(|e| {
        eprintln!("open codebase: {}", e);
        std::process::exit(1);
    });
    let hits = slice::find_by_type(&cb, &query);

    if flags.json {
        let mut s = String::from("[");
        for (i, (name, hash)) in hits.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str("{\"name\":");
            match name {
                Some(n) => s.push_str(&json_str(n)),
                None => s.push_str("null"),
            }
            s.push_str(",\"hash\":");
            s.push_str(&json_str(&hash.to_hex()));
            s.push('}');
        }
        s.push(']');
        println!("{}", s);
    } else if hits.is_empty() {
        println!("(no matches)");
    } else {
        for (name, hash) in &hits {
            let nm = name.as_deref().unwrap_or("(anon)");
            println!("{}  {}", &hash.to_hex()[..12], nm);
        }
    }
}

/// Minimal signature parser for `find-by-type`: `(T1, T2, ...) -> R`.
/// Type atoms are either a builtin/uppercase name treated as a builtin type, or
/// a single uppercase letter used as a fresh type variable (positional). This
/// is intentionally small but real — it builds a `TypeScheme::Fn` that actually
/// matches stored schemes. Anything it cannot parse is a clear error.
fn parse_signature(sig: &str) -> Result<TypeScheme, String> {
    let s = sig.trim();
    let arrow = s
        .find("->")
        .ok_or_else(|| format!("signature must contain `->`: {:?}", sig))?;
    let (lhs, rhs) = s.split_at(arrow);
    let rhs = &rhs[2..]; // drop "->"
    let lhs = lhs.trim();
    let rhs = rhs.trim();

    let inner = lhs
        .strip_prefix('(')
        .and_then(|x| x.strip_suffix(')'))
        .ok_or_else(|| format!("params must be parenthesized, e.g. (Int): {:?}", lhs))?;

    // A shared map from type-variable name -> assigned TypeVar index, so `(T) -> T`
    // uses the same TypeVar for both occurrences.
    let mut tvars: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut next_tv: u32 = 0;

    let mut params: Vec<Type> = Vec::new();
    if !inner.trim().is_empty() {
        for atom in inner.split(',') {
            let atom = atom.trim();
            if atom.is_empty() {
                return Err(format!("empty parameter in {:?}", lhs));
            }
            params.push(parse_type_atom(atom, &mut tvars, &mut next_tv)?);
        }
    }
    let ret = parse_type_atom(rhs, &mut tvars, &mut next_tv)?;

    Ok(TypeScheme::Fn {
        params,
        ret,
        // `wire` is not part of the search match; pick a value.
        wire: true,
    })
}

/// Parse one type atom. A single bare uppercase letter (e.g. `T`, `U`) is a
/// type variable; any other identifier is treated as a builtin type name
/// (`Int`, `Bool`, `String`, `Float`, `Bytes`, ...). Matching is over builtin
/// names exactly, so a typo'd builtin simply won't match anything.
fn parse_type_atom(
    atom: &str,
    tvars: &mut std::collections::HashMap<String, u32>,
    next_tv: &mut u32,
) -> Result<Type, String> {
    let atom = atom.trim();
    if atom.is_empty() {
        return Err("empty type atom".to_owned());
    }
    let is_single_upper =
        atom.len() == 1 && atom.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false);
    if is_single_upper {
        let idx = *tvars.entry(atom.to_owned()).or_insert_with(|| {
            let i = *next_tv;
            *next_tv += 1;
            i
        });
        Ok(Type::TypeVar(idx))
    } else {
        Ok(Type::Builtin(atom.to_owned()))
    }
}

/// JSON-escape a string and wrap it in double quotes. The CLI builds JSON by
/// hand (no serde dependency), matching the existing `--json` commands.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// =============================================================================
// `update` (Phase 2) — content edit with cone propagation.
// =============================================================================

/// `ai-lang update <name> <file> [--dry-run] [--json]` — replace the def
/// `<name>` with the source read from `<file>` and propagate the new hash up
/// its dependency cone. `--dry-run` reports the impact without committing.
fn cmd_update(name: &str, file: &str, dry_run: bool, propagate: bool, json: bool, cb_path: &PathBuf) {
    init_native_target().expect("init native target");
    let source = if file == "-" {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf).unwrap_or_else(|e| {
            eprintln!("update: cannot read stdin: {}", e);
            std::process::exit(1);
        });
        buf
    } else {
        std::fs::read_to_string(file).unwrap_or_else(|e| {
            eprintln!("update: cannot read source file {}: {}", file, e);
            std::process::exit(1);
        })
    };

    let mut cb = Codebase::open(cb_path).expect("open codebase");
    let res = match (dry_run, propagate) {
        (false, false) => edit::update(&mut cb, name, &source),
        (true, false) => edit::update_dry_run(&mut cb, name, &source),
        (false, true) => edit::update_propagate(&mut cb, name, &source),
        (true, true) => edit::update_dry_run_propagate(&mut cb, name, &source),
    };
    let res = match res {
        Ok(r) => r,
        Err(e) => {
            if json {
                println!(
                    "{{\"ok\":false,\"error\":{{\"message\":\"{}\"}}}}",
                    json_escape(&e.to_string())
                );
            } else {
                eprintln!("update: {}", e);
            }
            std::process::exit(1);
        }
    };

    // Persist todos to the on-disk store so `ai-lang todos` can show them.
    if !res.todos.is_empty() {
        let branch = ai_lang::namespace::current_branch(&cb).unwrap_or_else(|_| "main".to_string());
        if let Ok(mut store) = ai_lang::todostore::TodoStore::load(cb.root(), &branch) {
            store.record(&res.todos);
            store.clear_resolved(&cb);
            let _ = store.save();
        }
    }

    if json {
        print_update_json(&res, dry_run);
    } else {
        print_update_text(&res, dry_run);
    }
}

/// Render an `edit::Change` as a compact JSON object.
fn json_change(c: &ai_lang::edit::Change) -> String {
    format!(
        "{{\"name\":{},\"old\":\"{}\",\"new\":\"{}\"}}",
        json_opt_name(&c.name),
        c.old.to_hex(),
        c.new.to_hex()
    )
}

fn print_update_json(res: &ai_lang::edit::EditResult, dry_run: bool) {
    let updated: Vec<String> = res.updated.iter().map(json_change).collect();
    let propagated: Vec<String> = res.propagated.iter().map(json_change).collect();
    let todos: Vec<String> = res
        .todos
        .iter()
        .map(|t| {
            format!(
                "{{\"hash\":\"{}\",\"name\":{},\"message\":\"{}\"}}",
                t.hash.to_hex(),
                json_opt_name(&t.name),
                json_escape(&t.message)
            )
        })
        .collect();
    println!(
        "{{\"ok\":true,\"dry_run\":{},\"no_op\":{},\"updated\":[{}],\"propagated\":[{}],\"todos\":[{}]}}",
        dry_run,
        res.no_op,
        updated.join(","),
        propagated.join(","),
        todos.join(",")
    );
}

fn print_update_text(res: &ai_lang::edit::EditResult, dry_run: bool) {
    if res.no_op {
        println!("no change (new source is identical to the current definition)");
        return;
    }
    if dry_run {
        println!("[dry-run] no names were moved");
    }
    for c in &res.updated {
        let nm = c.name.as_deref().unwrap_or("<unnamed>");
        println!(
            "updated {}: {} -> {}",
            nm,
            &c.old.to_hex()[..12],
            &c.new.to_hex()[..12]
        );
    }
    for c in &res.propagated {
        let nm = c.name.as_deref().unwrap_or("<unnamed>");
        println!(
            "  propagated {}: {} -> {}",
            nm,
            &c.old.to_hex()[..12],
            &c.new.to_hex()[..12]
        );
    }
    if res.todos.is_empty() {
        println!("todos: none");
    } else {
        println!("todos ({}):", res.todos.len());
        for t in &res.todos {
            let nm = t.name.as_deref().unwrap_or("<unnamed>");
            println!("  - {} ({}): {}", nm, &t.hash.to_hex()[..12], t.message);
        }
    }
}

// =============================================================================
// Phase 3 refactor subcommands (move / inline / reorder-params / extract).
// Each is a thin wrapper over `edit.rs`; structured output reuses the same
// `print_update_{json,text}` helpers as `update` (the changelog shape is shared).
// =============================================================================

fn open_cb_or_die(cb_path: &PathBuf) -> Codebase {
    Codebase::open(cb_path).unwrap_or_else(|e| {
        eprintln!("cannot open codebase at {}: {}", cb_path.display(), e);
        std::process::exit(1);
    })
}

/// Render an `EditResult` (or error) for a Phase 3 refactor, reusing the shared
/// `update` changelog printers. Refactors that produce a new hash for the edited
/// def need the JIT initialized for any later `run`, but the refactor itself is a
/// pure AST + namespace operation, so no target init is required here.
fn report_edit(
    res: Result<ai_lang::edit::EditResult, ai_lang::edit::EditError>,
    op: &str,
    dry_run: bool,
    json: bool,
) {
    match res {
        Ok(r) => {
            if json {
                print_update_json(&r, dry_run);
            } else {
                print_update_text(&r, dry_run);
            }
        }
        Err(e) => {
            if json {
                println!("{{\"ok\":false,\"error\":{{\"message\":\"{}\"}}}}", json_escape(&e.to_string()));
            } else {
                eprintln!("{}: {}", op, e);
            }
            std::process::exit(1);
        }
    }
}

/// `ai-lang move <from> <to>` — namespace-path move (flat names this phase).
fn cmd_move(from: &str, to: &str, json: bool, cb_path: &PathBuf) {
    let mut cb = open_cb_or_die(cb_path);
    match edit::move_def(&mut cb, from, to) {
        Ok(r) => {
            let (f, t, h) = &r.renamed[0];
            if json {
                println!(
                    "{{\"ok\":true,\"moved\":{{\"from\":\"{}\",\"to\":\"{}\",\"hash\":\"{}\"}}}}",
                    json_escape(f),
                    json_escape(t),
                    h.to_hex()
                );
            } else {
                println!("moved {} -> {}  (hash {} unchanged; no callers broken)", f, t, &h.to_hex()[..16]);
            }
        }
        Err(e) => {
            if json {
                println!("{{\"ok\":false,\"error\":{{\"message\":\"{}\"}}}}", json_escape(&e.to_string()));
            } else {
                eprintln!("move: {}", e);
            }
            std::process::exit(1);
        }
    }
}

/// `ai-lang inline <name> [--dry-run] [--json]`.
fn cmd_inline(name: &str, dry_run: bool, json: bool, cb_path: &PathBuf) {
    let mut cb = open_cb_or_die(cb_path);
    let res = if dry_run {
        edit::inline_dry_run(&mut cb, name)
    } else {
        edit::inline(&mut cb, name)
    };
    report_edit(res, "inline", dry_run, json);
}

/// `ai-lang reorder-params <name> <perm> [--dry-run] [--json]`. `<perm>` is a
/// comma-separated permutation of `0..arity`, e.g. `1,0`.
fn cmd_reorder_params(name: &str, perm_str: &str, dry_run: bool, json: bool, cb_path: &PathBuf) {
    let perm: Result<Vec<usize>, _> =
        perm_str.split(',').map(|s| s.trim().parse::<usize>()).collect();
    let perm = match perm {
        Ok(p) if !p.is_empty() => p,
        _ => {
            eprintln!("reorder-params: <perm> must be a comma-separated permutation of 0..arity, e.g. 1,0");
            std::process::exit(2);
        }
    };
    let mut cb = open_cb_or_die(cb_path);
    let res = if dry_run {
        edit::reorder_params_dry_run(&mut cb, name, &perm)
    } else {
        edit::reorder_params(&mut cb, name, &perm)
    };
    report_edit(res, "reorder-params", dry_run, json);
}

// =============================================================================
// Phase 4: branches + history (branch / branches / switch / diff / history /
// undo / merge). Thin wrappers over `namespace.rs`; `--json` where sensible.
// =============================================================================

fn ns_die(op: &str, e: namespace::NamespaceError, json: bool) -> ! {
    if json {
        println!(
            "{{\"ok\":false,\"error\":{{\"message\":\"{}\"}}}}",
            json_escape(&e.to_string())
        );
    } else {
        eprintln!("{}: {}", op, e);
    }
    std::process::exit(1);
}

/// `ai-lang branch <name> [<from>]` — O(1) fork.
fn cmd_branch(name: &str, from: Option<&str>, json: bool, cb_path: &PathBuf) {
    let cb = open_cb_or_die(cb_path);
    let head = match namespace::branch(&cb, name, from) {
        Ok(h) => h,
        Err(e) => ns_die("branch", e, json),
    };
    if json {
        println!(
            "{{\"ok\":true,\"branch\":\"{}\",\"head\":\"{}\"}}",
            json_escape(name),
            head.to_hex()
        );
    } else {
        println!(
            "created branch {} at {} (defs shared, none copied)",
            name,
            &head.to_hex()[..16]
        );
    }
}

/// `ai-lang branches` — list branches, marking the active one.
fn cmd_branches(json: bool, cb_path: &PathBuf) {
    let cb = open_cb_or_die(cb_path);
    let current = namespace::current_branch(&cb).unwrap_or_else(|e| ns_die("branches", e, json));
    let list = namespace::branches(&cb).unwrap_or_else(|e| ns_die("branches", e, json));
    if json {
        let parts: Vec<String> = list
            .iter()
            .map(|b| {
                format!(
                    "{{\"name\":\"{}\",\"current\":{}}}",
                    json_escape(b),
                    b == &current
                )
            })
            .collect();
        println!(
            "{{\"current\":\"{}\",\"branches\":[{}]}}",
            json_escape(&current),
            parts.join(",")
        );
    } else {
        for b in &list {
            let mark = if b == &current { "*" } else { " " };
            println!("{} {}", mark, b);
        }
    }
}

/// `ai-lang switch <name>` — change active branch + reload names.
fn cmd_switch(name: &str, json: bool, cb_path: &PathBuf) {
    let mut cb = open_cb_or_die(cb_path);
    if let Err(e) = namespace::switch(&mut cb, name) {
        ns_die("switch", e, json);
    }
    if json {
        println!("{{\"ok\":true,\"current\":\"{}\"}}", json_escape(name));
    } else {
        println!("switched to branch {}", name);
    }
}

/// `ai-lang diff <a> <b>` — structural diff of two branch snapshots.
fn cmd_diff(a: &str, b: &str, json: bool, cb_path: &PathBuf) {
    let cb = open_cb_or_die(cb_path);
    let d = namespace::diff(&cb, a, b).unwrap_or_else(|e| ns_die("diff", e, json));
    if json {
        let added: Vec<String> = d
            .added
            .iter()
            .map(|(n, h)| format!("{{\"name\":\"{}\",\"hash\":\"{}\"}}", json_escape(n), h.to_hex()))
            .collect();
        let removed: Vec<String> = d
            .removed
            .iter()
            .map(|(n, h)| format!("{{\"name\":\"{}\",\"hash\":\"{}\"}}", json_escape(n), h.to_hex()))
            .collect();
        let changed: Vec<String> = d
            .changed
            .iter()
            .map(|(n, oa, ob)| {
                format!(
                    "{{\"name\":\"{}\",\"a\":\"{}\",\"b\":\"{}\"}}",
                    json_escape(n),
                    oa.to_hex(),
                    ob.to_hex()
                )
            })
            .collect();
        println!(
            "{{\"a\":\"{}\",\"b\":\"{}\",\"added\":[{}],\"removed\":[{}],\"changed\":[{}]}}",
            json_escape(a),
            json_escape(b),
            added.join(","),
            removed.join(","),
            changed.join(",")
        );
    } else if d.is_empty() {
        println!("no differences between {} and {}", a, b);
    } else {
        for (n, h) in &d.added {
            println!("+ {:<28}  {}", n, &h.to_hex()[..16]);
        }
        for (n, h) in &d.removed {
            println!("- {:<28}  {}", n, &h.to_hex()[..16]);
        }
        for (n, oa, ob) in &d.changed {
            println!(
                "~ {:<28}  {} -> {}",
                n,
                &oa.to_hex()[..12],
                &ob.to_hex()[..12]
            );
        }
    }
}

/// `ai-lang history` — causal-hash chain of the current branch.
fn cmd_history(json: bool, cb_path: &PathBuf) {
    let cb = open_cb_or_die(cb_path);
    let branch = namespace::current_branch(&cb).unwrap_or_else(|e| ns_die("history", e, json));
    let chain = namespace::history(&cb, None).unwrap_or_else(|e| ns_die("history", e, json));
    if json {
        let parts: Vec<String> = chain.iter().map(|h| format!("\"{}\"", h.to_hex())).collect();
        println!(
            "{{\"branch\":\"{}\",\"history\":[{}]}}",
            json_escape(&branch),
            parts.join(",")
        );
    } else {
        for (i, h) in chain.iter().enumerate() {
            let marker = if i == 0 { "HEAD" } else { "    " };
            println!("{} {}", marker, &h.to_hex()[..16]);
        }
        eprintln!("\n  {} commit(s) on {}", chain.len(), branch);
    }
}

/// `ai-lang undo` — move the current branch head back to its parent.
fn cmd_undo(json: bool, cb_path: &PathBuf) {
    let mut cb = open_cb_or_die(cb_path);
    let parent = namespace::undo(&mut cb).unwrap_or_else(|e| ns_die("undo", e, json));
    if json {
        println!("{{\"ok\":true,\"head\":\"{}\"}}", parent.to_hex());
    } else {
        println!(
            "undid last change; head is now {} (defs are immutable, none destroyed)",
            &parent.to_hex()[..16]
        );
    }
}

/// `ai-lang merge <from> <into>` — 3-way merge over the name layer.
fn cmd_merge(from: &str, into: &str, json: bool, cb_path: &PathBuf) {
    let mut cb = open_cb_or_die(cb_path);
    let res = namespace::merge(&mut cb, from, into).unwrap_or_else(|e| ns_die("merge", e, json));
    match res {
        MergeResult::Merged { new_head, brought } => {
            if json {
                let parts: Vec<String> = brought
                    .iter()
                    .map(|(n, h)| {
                        format!("{{\"name\":\"{}\",\"hash\":\"{}\"}}", json_escape(n), h.to_hex())
                    })
                    .collect();
                println!(
                    "{{\"ok\":true,\"merged\":true,\"head\":\"{}\",\"brought\":[{}]}}",
                    new_head.to_hex(),
                    parts.join(",")
                );
            } else {
                println!(
                    "merged {} into {} ({} name(s) brought across); head {}",
                    from,
                    into,
                    brought.len(),
                    &new_head.to_hex()[..16]
                );
                for (n, h) in &brought {
                    println!("  {} = {}", n, &h.to_hex()[..12]);
                }
            }
        }
        MergeResult::Conflicts(cs) => {
            if json {
                let parts: Vec<String> = cs
                    .iter()
                    .map(|c| {
                        format!(
                            "{{\"name\":\"{}\",\"base\":{},\"into\":\"{}\",\"from\":\"{}\"}}",
                            json_escape(&c.name),
                            match &c.base {
                                Some(h) => format!("\"{}\"", h.to_hex()),
                                None => "null".to_owned(),
                            },
                            c.into_hash.to_hex(),
                            c.from_hash.to_hex()
                        )
                    })
                    .collect();
                println!(
                    "{{\"ok\":false,\"merged\":false,\"conflicts\":[{}]}}",
                    parts.join(",")
                );
            } else {
                eprintln!(
                    "merge of {} into {} has {} conflict(s); NOTHING was committed:",
                    from,
                    into,
                    cs.len()
                );
                for c in &cs {
                    eprintln!(
                        "  {}: into={} from={}",
                        c.name,
                        &c.into_hash.to_hex()[..12],
                        &c.from_hash.to_hex()[..12]
                    );
                }
            }
            std::process::exit(1);
        }
    }
}

/// `ai-lang extract <name> <selector> <new_name> [--dry-run] [--json]`.
/// `<selector>` is `let-value` or `body`.
fn cmd_extract(name: &str, selector_str: &str, new_name: &str, dry_run: bool, json: bool, cb_path: &PathBuf) {
    let selector = match ai_lang::edit::ExtractSelector::parse(selector_str) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(2);
        }
    };
    let mut cb = open_cb_or_die(cb_path);
    let res = if dry_run {
        edit::extract_dry_run(&mut cb, name, selector, new_name)
    } else {
        edit::extract(&mut cb, name, selector, new_name)
    };
    report_edit(res, "extract", dry_run, json);
}
