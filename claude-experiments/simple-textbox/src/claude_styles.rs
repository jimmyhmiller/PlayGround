use minime::renderer::styles::{Footer, Header, Margin};
use minime::editor::Editor;
use std::io::Write;
use crossterm::{
    terminal::{Clear, ClearType},
    QueueableCommand,
};

/// Claude Code style header - just the top border
pub struct ClaudeHeader;

impl<W: Write> Header<W> for ClaudeHeader {
    fn rows(&self) -> usize {
        1
    }

    fn draw(&mut self, w: &mut W, _: &Editor) -> minime::Result<()> {
        // Get terminal width to draw full border
        let (width, _) = crossterm::terminal::size().unwrap_or((80, 24));
        
        // Top border: ╭─────...─────╮
        write!(w, "╭")?;
        for _ in 1..width-1 {
            write!(w, "─")?;
        }
        write!(w, "╮")?;
        
        Ok(())
    }
}

/// Claude Code style margin - simple prompt indicator
pub struct ClaudeMargin;

impl<W: Write> Margin<W> for ClaudeMargin {
    fn width(&self) -> usize {
        4 // "│ > " width
    }

    fn draw(&mut self, write: &mut W, line_idx: usize, data: &Editor) -> minime::Result<()> {
        if line_idx == 0 {
            // First line gets the prompt
            write!(write, "│ > ")?;
        } else {
            // Continuation lines
            write!(write, "│   ")?;
        }
        Ok(())
    }
}

/// Claude Code style footer - just the bottom border  
pub struct ClaudeFooter;

impl<W: Write> Footer<W> for ClaudeFooter {
    fn rows(&self) -> usize {
        1
    }

    fn draw(&mut self, w: &mut W, _: &Editor) -> minime::Result<()> {
        // Get terminal width to draw full border
        let (width, _) = crossterm::terminal::size().unwrap_or((80, 24));
        
        // Bottom border: ╰─────...─────╯
        write!(w, "╰")?;
        for _ in 1..width-1 {
            write!(w, "─")?;
        }
        write!(w, "╯")?;
        
        w.queue(Clear(ClearType::UntilNewLine))?;
        Ok(())
    }
}

/// Alternative with placeholder text on empty input
pub struct ClaudeMarginWithPlaceholder;

impl<W: Write> Margin<W> for ClaudeMarginWithPlaceholder {
    fn width(&self) -> usize {
        4 // "│ > " width
    }

    fn draw(&mut self, write: &mut W, line_idx: usize, data: &Editor) -> minime::Result<()> {
        if line_idx == 0 {
            write!(write, "│ > ")?;
            
            // Show placeholder if empty and this is the first line
            if data.char_count() == 0 {
                write!(write, "\x1b[90mTry \"fix typecheck errors\"\x1b[0m")?;
            }
        } else {
            write!(write, "│   ")?;
        }
        Ok(())
    }
}