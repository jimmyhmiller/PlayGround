use minime::editor::{keybindings::Keybinding, Editor};
use minime::Result;
use crossterm::event::{read, Event, KeyCode, KeyEvent, KeyModifiers};

/// Custom keybinding that only allows newlines with Alt+Enter (Option+Enter on Mac)
pub struct ClaudeKeybinding;

impl Keybinding for ClaudeKeybinding {
    fn read(&self, editor: &mut Editor) -> Result<bool> {
        let key_event = read()?;
        match key_event {
            Event::Key(k) => Self::process_key_event(editor, k),
            _ => Ok(true),
        }
    }
}

impl ClaudeKeybinding {
    pub fn process_key_event(editor: &mut Editor, event: KeyEvent) -> Result<bool> {
        let code = event.code;
        let ln_count = editor.line_count();
        let shifted = event.modifiers.contains(KeyModifiers::SHIFT);
        let alt = event.modifiers.contains(KeyModifiers::ALT);
        let control = event.modifiers.contains(KeyModifiers::CONTROL);

        match code {
            KeyCode::Down => editor.move_down(shifted),
            KeyCode::Up => editor.move_up(shifted),
            KeyCode::Left => editor.move_left(shifted),
            KeyCode::Right => editor.move_right(shifted),

            KeyCode::PageDown => editor.move_to_bottom(),
            KeyCode::PageUp => editor.move_to_top(),
            KeyCode::Home => {
                let leading_spaces = editor
                    .curr_ln_chars()
                    .take_while(|c| c.is_whitespace())
                    .count();
                if editor.selection.focus.col == leading_spaces {
                    editor.move_to_col(0, shifted);
                } else {
                    editor.move_to_col(leading_spaces, shifted);
                }
            }
            KeyCode::End => editor.move_to_line_end(shifted),

            KeyCode::Backspace => editor.backspace(),
            KeyCode::Char('h') if control => editor.backspace(),
            KeyCode::Delete => editor.delete(),

            KeyCode::F(12) => {
                editor.altscreen = !editor.altscreen;
            }

            #[cfg(feature = "unstable")]
            KeyCode::Char('c') if control => {
                if let Ok(mut clipboard) = arboard::Clipboard::new() {
                    if let Some(txt) = editor.curr_sel() {
                        clipboard.set_text(txt.to_string()).unwrap();
                    } else {
                        clipboard.set_text(editor.curr_ln().to_string()).unwrap();
                    }
                }
            }
            #[cfg(feature = "unstable")]
            KeyCode::Char('x') if control => {
                if let Ok(mut clipboard) = arboard::Clipboard::new() {
                    if let Some(txt) = editor.curr_sel() {
                        clipboard.set_text(txt.to_string()).unwrap();
                        editor.delete();
                    } else {
                        clipboard
                            .set_text(editor.remove_line(editor.selection.focus.ln))
                            .unwrap();
                    }
                }
            }
            #[cfg(feature = "unstable")]
            KeyCode::Char('v') if control => {
                if let Ok(mut clipboard) = arboard::Clipboard::new() {
                    if let Ok(txt) = clipboard.get_text() {
                        editor.insert_str(&txt);
                    }
                }
            }

            KeyCode::Tab => {
                editor.clamp();
                let soft = 4 - editor.selection.focus.col % 4;
                for _ in 0..soft {
                    editor.insert_char(0, ' ');
                }
                editor.selection.focus.col += soft;
            }
            KeyCode::BackTab => {
                editor.clamp();

                let leading_spaces = editor
                    .curr_ln_chars()
                    .take(4)
                    .take_while(|c| c.is_whitespace())
                    .count();

                // Delete leading whitespace characters one by one
                for _ in 0..leading_spaces {
                    editor.move_to_col(0, false);
                    editor.delete_char(0);
                }
            }
            KeyCode::Esc => return Ok(false),
            // Only allow newlines when Alt (Option) is pressed
            KeyCode::Enter => {
                if alt {
                    // Alt+Enter: insert newline
                    editor.type_char('\n');
                } else {
                    // Regular Enter: exit editor
                    return Ok(false);
                }
            }
            KeyCode::Char(c) => editor.type_char(c),
            _ => { /* ignored */ }
        }
        Ok(true)
    }
}