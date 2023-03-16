use std::{path::Path, process::Command};

fn add_indention(text: &str, indention: usize) -> String {
    let mut new_text = String::new();
    for line in text.lines() {
        new_text.push_str(&" ".repeat(indention));
        new_text.push_str(line);
        new_text.push_str("\n");
    }
    new_text
}

fn preamble() -> String {
    "Here are the stats of railsbench and liquid-render before and after these changes".to_string()
}

fn template_single_bench(bench_name: &str, stats: Vec<u8>) -> String {
    // This indention business is a bit ugly
    let stats = add_indention(&String::from_utf8_lossy(&stats), 8);
    format!("
        <summary>{bench_name}</summary>
        <details>

        ```ruby
         {stats}
        ```
        </details>
    ")
}

fn base_command(ruby_path: &str) -> Command {
    let mut command = Command::new(ruby_path);

    command
        .env("WARMUP_ITRS", "0")
        .env("MIN_BENCH_ITRS", "1")
        .env("MIN_BENCH_TIME", "0")
        .env("PATH", Path::new(ruby_path).parent().unwrap().to_str().unwrap())
        .arg("-Iharness");

    command
}


fn run_liquid_render(ruby_path: &str) -> std::process::Output {
    eprintln!("Running liquid-render");
    base_command(ruby_path)
        .arg("--yjit")
        .arg("--yjit-call-threshold=1")
        .arg("--yjit-stats")
        .arg("benchmarks/liquid-render/benchmark.rb")
        .output()
        .expect("failed to execute process")
}

fn run_rails_bench(ruby_path: &str) -> std::process::Output {
    eprintln!("Running railsbench");
    base_command(ruby_path)
        .arg("--yjit")
        .arg("--yjit-call-threshold=1")
        .arg("--yjit-stats")
        .arg("benchmarks/railsbench/benchmark.rb")
        .output()
        .expect("failed to execute process")
}

// finds the first non_empty line and returns the number of spaces before it
fn find_indention(text: &str) -> usize {
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        let indention = line.chars().take_while(|c| c.is_whitespace()).count();
        if indention != 0 {
            return indention
        }
    }
    panic!("No indention found");
}

// a function to remove indention based on the first line
fn remove_indention(text: &str) -> String {
    let indention = find_indention(text);
    let mut new_text = String::new();
    for line in text.lines() {
        if line.len() > indention {
            new_text.push_str(&line[indention..]);
        } else {
            new_text.push_str(line);
        }
        new_text.push_str("\n");
    }
    new_text
}


fn main() {

    let current_ruby_stats = "/Users/jimmyhmiller/.rubies/ruby-yjit-stats/bin/ruby";
    let master_ruby_stats = "/Users/jimmyhmiller/.rubies/yjit-master-stats/bin/ruby";

    // yjit bench directory
    let yjit_bench_dir = Path::new("/Users/jimmyhmiller/Documents/Code/yjit-bench");
    // change directory
    std::env::set_current_dir(yjit_bench_dir).unwrap();

    eprintln!("Switching to yjit-master-stats");

    // TODO:
    // Auto pull on master
    // Build current and master on stats


    let rails_bench_master = run_rails_bench(master_ruby_stats);
    let liquid_render_master = run_liquid_render(master_ruby_stats);

    eprintln!("Switching to ruby-yjit-stats");
    let rails_bench_current = run_rails_bench(current_ruby_stats);
    let liquid_render_current = run_liquid_render(current_ruby_stats);

    let preamble = preamble();
    let rails_before = template_single_bench("Rails Bench Before", rails_bench_master.stderr);
    let rails_after = template_single_bench("Rails Bench After", rails_bench_current.stderr);

    let liquid_before = template_single_bench("Liquid Render Before", liquid_render_master.stderr);
    let liquid_after = template_single_bench("Liquid Render After", liquid_render_current.stderr);


    let full_message = format!("
        {preamble}
        {rails_before}
        {rails_after}
        {liquid_before}
        {liquid_after}
    ");

    println!("{}", remove_indention(&full_message));
}




