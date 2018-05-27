extern crate termion;

// http://ticki.github.io/blog/making-terminal-applications-in-rust-with-termion/

use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use std::io::{Write, stdout, stdin};

fn main() {
    // Get the standard input stream.
    let stdin = stdin();
    // Get the standard output stream and go to raw mode.
    let mut stdout = stdout().into_raw_mode().unwrap();
    let mut column = 1;
    let mut line = 1;

    write!(stdout, "{}{}Ctrl+ c exit.",
           // Clear the screen.
           termion::clear::All,
           // Goto (1,1).
           termion::cursor::Goto(1, 1)).unwrap();
    // Flush stdout (i.e. make the output appear).
    stdout.flush().unwrap();
    let (width, _height) = termion::terminal_size().unwrap();

    for c in stdin.keys() {
        // Clear the current line.
        write!(stdout, "{}", termion::cursor::Goto(column, line)).unwrap();

        if column == width {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
        // Print the key we type...
        match c.unwrap() {
            // Exit.
            Key::Char(c)   => print!("{}", c),
            Key::Alt(c)    => print!("Alt-{}", c),
            Key::Ctrl(_c)   => break,
            Key::Left      => column -= 2,
            Key::Right     => column += 2,
            Key::Up        => { line -= 1; column -= 1},
            Key::Down      => { line += 1; column -= 1 },
            _              => print!("Other"),
        }

        // Flush again.
        stdout.flush().unwrap();
    }

    // Show the cursor again before we exit.
    write!(stdout, "{}", termion::cursor::Show).unwrap();
}