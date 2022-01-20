use std::{cmp::min, str::from_utf8};

use sdl2::clipboard::ClipboardUtil;

use crate::{text_buffer::TextBuffer, transaction::EditAction, scroller::Scroller, renderer::EditorBounds};



#[derive(Debug, Copy, Clone)]
pub struct Cursor(pub usize, pub usize);

impl Cursor {

    pub fn start_of_line(&mut self) {
        self.1 = 0;
    }

    pub fn move_up(&mut self, text_buffer: &TextBuffer) -> EditAction {
        let new_line = self.0.saturating_sub(1);
        self.0 = new_line;
        self.1 = min(self.1, text_buffer.line_length(new_line));
        EditAction::CursorPosition(*self)
        // *self = Cursor(new_line, min(cursor_column, line_length(line_range[new_line])));

        // Need to deal with line_fraction
        // Need to use this output to deal with scrolling up
        // if cursor_line < lines_above_fold {
        //     offset_y -= letter_height as i32;
        // }
    }

    pub fn move_down(&mut self, text_buffer: &TextBuffer) -> EditAction  {
        self.0 = min(self.0 + 1, text_buffer.line_count() -1);
        self.1 = min(self.1, text_buffer.line_length(self.0));
        EditAction::CursorPosition(*self)

        // Need to use this output to deal with scrolling down
    }


    pub fn move_left(&mut self, text_buffer: &TextBuffer) -> EditAction  {
        let Cursor(cursor_line, cursor_column) = *self;
        if cursor_column == 0 && cursor_line != 0 {
            let length = text_buffer.line_length(cursor_line - 1);
            *self = Cursor(cursor_line.saturating_sub(1), length);
        } else {
            *self = Cursor(cursor_line, cursor_column.saturating_sub(1));
        }
        // Could need to deal with scrolling left
        EditAction::CursorPosition(*self)
    }

    pub fn move_right(&mut self, text_buffer: &TextBuffer) -> EditAction  {
        let Cursor(cursor_line, cursor_column) = *self;
        if let Some((_line_start, _line_end)) = text_buffer.get_line(cursor_line) {
            let length = text_buffer.line_length(cursor_line);
            if cursor_column >= length {
                if cursor_line + 1 < text_buffer.line_count() {
                    *self = Cursor(cursor_line + 1, 0);
                }
            } else {
                *self = Cursor(cursor_line, cursor_column + 1);
            }
        }
         // Could need to deal with scrolling right
        EditAction::CursorPosition(*self)
    }

}




#[derive(Debug, Clone)]
pub struct CursorContext {
    pub cursor: Option<Cursor>,
    pub selection: Option<((usize, usize), (usize, usize))>,
    pub mouse_down: Option<Cursor>,
}

impl CursorContext {

    pub fn new() -> Self {
        CursorContext {
            cursor: None,
            mouse_down: None,
            selection: None,
        }
    }

    pub fn move_up(&mut self, text_buffer: &TextBuffer) -> EditAction {
        self.cursor
            .as_mut()
            .map(|cursor| cursor.move_up(text_buffer))
            .unwrap_or(EditAction::Noop)
    }
    pub fn move_down(&mut self, text_buffer: &TextBuffer) -> EditAction {
        self.cursor
            .as_mut()
            .map(|cursor| cursor.move_down(text_buffer))
            .unwrap_or(EditAction::Noop)
    }
    pub fn move_left(&mut self, text_buffer: &TextBuffer) -> EditAction  {
        self.cursor
            .as_mut()
            .map(|cursor| cursor.move_left(text_buffer))
            .unwrap_or(EditAction::Noop)
    }
    pub fn move_right(&mut self, text_buffer: &TextBuffer) -> EditAction  {
        self.cursor
            .as_mut()
            .map(|cursor| cursor.move_right(text_buffer))
            .unwrap_or(EditAction::Noop)
    }
    pub fn start_of_line(&mut self) {
        if let Some(cursor) = self.cursor.as_mut() {
            cursor.start_of_line();
        }
    }

    pub fn end_of_line(&mut self, text_buffer: &TextBuffer) {
        if let Some(cursor) = self.cursor.as_mut() {
            cursor.1 = text_buffer.line_length(cursor.0);
        }
    }

    pub fn set_cursor(&mut self, cursor: Cursor) {
        self.cursor = Some(cursor);
    }

    pub fn clear_selection(&mut self) {
        self.selection = None;
    }

    pub fn set_selection(&mut self, selection: ((usize, usize), (usize, usize))) {
        self.selection = Some(selection);
    }

    pub fn fix_cursor(&mut self, text_buffer: &TextBuffer) {
        // Need to do sanity checks for cursor column
        if let Some(Cursor(cursor_line, cursor_column)) = self.cursor {
            match text_buffer.get_line(cursor_line) {
                Some((_start, end)) => {
                    if cursor_column > *end {
                        self.cursor = Some(Cursor(cursor_line, text_buffer.line_length(cursor_line)));
                    }
                }
                None => {
                    self.cursor = text_buffer.last_line().map(|(line, column)| Cursor(*line, *column));
                }
            }
        }
    }

    pub fn copy_selection(&self, clipboard: &ClipboardUtil, text_buffer: &TextBuffer) -> Option<()> {
        if let Some(((start_line, start_column), (end_line, end_column))) = self.selection {
            let start = text_buffer.char_position_from_line_column(start_line, start_column)?;
            let end = text_buffer.char_position_from_line_column(end_line, end_column)?;
            clipboard.set_clipboard_text(from_utf8(&text_buffer.chars[start..end]).unwrap()).ok()?;
        }

        Some(())
    }

    pub fn mouse_down(&mut self) {
        self.mouse_down = self.cursor;
    }

    pub fn clear_mouse_down(&mut self) {
        self.mouse_down = None;
    }

    pub fn move_cursor_from_screen_position(&mut self, scroller: &Scroller, x: usize, y: usize, text_buffer: &TextBuffer, bounds: &EditorBounds) {
        self.cursor = scroller.text_space_from_screen_space(x, y, text_buffer, bounds);
    }

    pub fn handle_insert(&mut self, to_insert: &[u8], text_buffer : &mut TextBuffer) -> EditAction {
        if let Some(cursor) = self.cursor {
            let main_action = text_buffer.insert_char(cursor, to_insert);
            let cursor_action = if to_insert == b"\n" {
                self.move_down(text_buffer)
            } else {
                self.move_right(text_buffer)
            };
            
            main_action.combine_insert_and_cursor(cursor_action)
        } else {
            EditAction::Noop
        }

    }

    pub fn paste(&mut self, clipboard: &ClipboardUtil, text_buffer: &mut TextBuffer) -> Option<String> {
        if let Some(Cursor(line, column)) = self.cursor {
            let position = text_buffer.char_position_from_line_column(line, column)?;
            // I could probably just use splice
            let before = &text_buffer.chars[..position];
            let after = &text_buffer.chars[position..];
            let from_clipboard = clipboard.clipboard_text().ok()?;
            let new_chars = [before, from_clipboard.as_bytes(), after].concat();
            text_buffer.chars = new_chars;
            text_buffer.parse_lines();
            if let Some(cursor) = text_buffer.line_column_from_char_position(position + from_clipboard.len()) {
                self.set_cursor(cursor);
            }
            Some(from_utf8(from_clipboard.as_bytes()).unwrap().to_string())
        } else {
            None
        }
    }
}
