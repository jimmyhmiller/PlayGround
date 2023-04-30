use core::cmp::min;
use core::fmt::Debug;
use std::collections::HashMap;
// TODO: I probably do need to return actions here
// for every time the cursor moves.

pub struct LineIter<'a, Item> {
    current_position: usize,
    items: &'a [Item],
    newline: &'a Item,
}
impl<'a, Item> Iterator for LineIter<'a, Item>
where
    Item: PartialEq + Copy,
{
    type Item = &'a [Item];
    fn next(&mut self) -> Option<Self::Item> {
        let original_position = self.current_position;
        while self.current_position < self.items.len() {
            let byte = self.items[self.current_position];
            if byte == *self.newline {
                let line = &self.items[original_position..self.current_position];
                self.current_position += 1;
                return Some(line);
            }
            self.current_position += 1;
        }
        if self.current_position != original_position {
            let line = &self.items[original_position..self.current_position];
            return Some(line);
        }
        None
    }
}

pub trait TextBuffer {
    type Item;
    fn line_length(&self, line: usize) -> usize;
    fn line_count(&self) -> usize;
    // Rethink bytes because of utf8
    fn insert_bytes(&mut self, line: usize, column: usize, text: &[Self::Item]);
    fn byte_at_pos(&self, line: usize, column: usize) -> Option<&Self::Item>;
    fn delete_char(&mut self, line: usize, column: usize);
    fn lines(&self) -> LineIter<Self::Item>;
    fn last_line(&self) -> usize {
        self.line_count().saturating_sub(1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SimpleTextBuffer {
    pub bytes: Vec<u8>,
}

// These are really bad implementations
// probably need to actually know where lines start
// and stop in some data structure.
// But trying to start with the simplest thing that works
impl SimpleTextBuffer {
    pub fn new() -> Self {
        SimpleTextBuffer { bytes: Vec::new() }
    }

    pub fn new_with_contents(contents: &[u8]) -> Self {
        SimpleTextBuffer {
            bytes: contents.to_vec(),
        }
    }

    pub fn set_contents(&mut self, contents: &[u8]) {
        self.bytes = contents.to_vec();
    }

    pub fn line_start(&self, line: usize) -> usize {
        let mut line_start = 0;
        let mut lines_seen = 0;

        for byte in self.bytes.iter() {
            if lines_seen == line {
                break;
            }
            if *byte == b'\n' {
                lines_seen += 1;
            }
            line_start += 1;
        }
        line_start
    }
}

impl TextBuffer for SimpleTextBuffer {
    type Item = u8;

    fn line_length(&self, line: usize) -> usize {
        let line_start = self.line_start(line);

        let mut length = 0;
        for byte in self.bytes.iter().skip(line_start) {
            if *byte == b'\n' {
                break;
            }
            length += 1;
        }
        length
    }

    fn line_count(&self) -> usize {
        self.bytes.iter().filter(|&&byte| byte == b'\n').count() + 1
    }

    fn insert_bytes(&mut self, line: usize, column: usize, text: &[u8]) {
        let start = self.line_start(line) + column;
        self.bytes.splice(start..start, text.iter().cloned());
    }

    fn byte_at_pos(&self, line: usize, column: usize) -> Option<&u8> {
        self.bytes.get(self.line_start(line) + column)
    }

    fn delete_char(&mut self, line: usize, column: usize) {
        let start = self.line_start(line) + column;
        self.bytes.remove(start);
    }

    fn lines(&self) -> LineIter<Self::Item> {
        LineIter {
            current_position: 0,
            items: &self.bytes,
            newline: &b'\n',
        }
    }
}

pub trait VirtualCursor: Clone + Debug {
    fn move_to(&mut self, line: usize, column: usize);
    fn line(&self) -> usize;
    fn column(&self) -> usize;
    fn new(line: usize, column: usize) -> Self;

    fn move_to_bounded<T: TextBuffer>(&mut self, line: usize, column: usize, buffer: &T) {
        let line = min(buffer.last_line(), line);
        let column = min(buffer.line_length(line), column);
        self.move_to(line, column);
    }

    fn move_up<T: TextBuffer>(&mut self, buffer: &T) {
        let previous_line = self.line().saturating_sub(1);
        self.move_to(
            previous_line,
            min(self.column(), buffer.line_length(previous_line)),
        );
    }

    fn move_down<T: TextBuffer>(&mut self, buffer: &T) {
        let next_line = self.line().saturating_add(1);
        let last_line = buffer.last_line();
        if next_line > last_line {
            return
        }
        self.move_to(
            min(next_line, last_line),
            min(
                self.column(),
                buffer.line_length(next_line),
            ),
        );
    }

    fn move_left<T: TextBuffer>(&mut self, buffer: &T) {
        if self.column() == 0 && self.line() != 0 {
            let new_line = self.line().saturating_sub(1);
            let length = buffer.line_length(new_line);
            self.move_to(new_line, length);
        } else {
            self.move_to(self.line(), self.column().saturating_sub(1));
        }
    }

    fn move_right<T: TextBuffer>(&mut self, buffer: &T) {
        let length = buffer.line_length(self.line());
        if self.column() >= length {
            if self.line().saturating_add(1) < buffer.line_count() {
                self.move_to(self.line().saturating_add(1), 0);
            }
        } else {
            self.move_to(self.line(), self.column().saturating_add(1));
        }
    }

    fn start_of_line(&mut self) {
        self.move_to(self.line(), 0);
    }

    fn end_of_line<T: TextBuffer>(&mut self, buffer: &T) {
        self.move_to(self.line(), buffer.line_length(self.line()));
    }

    fn move_in_buffer<T: TextBuffer>(&mut self, line: usize, column: usize, buffer: &T) {
        if line < buffer.line_count() {
            self.move_to(line, min(column, buffer.line_length(line)));
        }
    }

    fn right_of<T: TextBuffer>(&self, buffer: &T) -> Self {
        let mut cursor = self.clone();
        cursor.move_right(buffer);
        cursor
    }

    fn left_of<T: TextBuffer>(&self, buffer: &T) -> Self {
        let mut cursor = self.clone();
        cursor.move_left(buffer);
        cursor
    }

    fn above<T: TextBuffer<Item = u8>>(&self, buffer: &T) -> Self {
        let mut cursor = self.clone();
        cursor.move_up(buffer);
        cursor
    }

    fn below<T: TextBuffer>(&self, buffer: &T) -> Self {
        let mut cursor = self.clone();
        cursor.move_down(buffer);
        cursor
    }

    fn auto_bracket_insert<T: TextBuffer<Item = u8>>(&mut self, buffer: &mut T, to_insert: &[u8]) {
        let to_insert = match to_insert {
            b"(" => b"()",
            b"[" => b"[]",
            b"{" => b"{}",
            b"\"" => b"\"\"",
            _ => to_insert,
        };

        buffer.insert_bytes(self.line(), self.column(), to_insert);
        self.move_right(buffer);
    }

    fn insert_normal_text<T: TextBuffer<Item = u8>>(&mut self, to_insert: &[u8], buffer: &mut T) {
        buffer.insert_bytes(self.line(), self.column(), to_insert);
        if to_insert == b"\n" {
            self.move_down(buffer);
            self.move_to(self.line(), 0);
        } else {
            // TODO: Do this more efficiently
            for _ in 0..(to_insert.len() - 1) {
                self.move_right(buffer);
            }
            self.move_right(buffer)
        };
    }

    fn delete_char<T: TextBuffer>(&mut self, buffer: &mut T) {
        self.move_left(buffer);
        buffer.delete_char(self.line(), self.column());
    }

    fn is_open_bracket(byte: &[u8]) -> bool {
        match byte {
            b"(" | b"[" | b"{" | b"\"" => true,
            _ => false,
        }
    }

    fn is_close_bracket(byte: &[u8]) -> bool {
        match byte {
            b")" | b"]" | b"}" | b"\"" => true,
            _ => false,
        }
    }

    fn handle_insert<T: TextBuffer<Item = u8>>(&mut self, to_insert: &[u8], buffer: &mut T) {
        if Self::is_open_bracket(to_insert) {
            // Would need to have a setting for this
            self.auto_bracket_insert(buffer, to_insert);
        } else if Self::is_close_bracket(to_insert) {
            let right_of_cursor = buffer.byte_at_pos(self.line(), self.column());

            match right_of_cursor {
                Some(right) if Self::is_close_bracket(&[*right]) => self.move_right(buffer),
                _ => self.insert_normal_text(to_insert, buffer),
            }
        } else {
            self.insert_normal_text(to_insert, buffer)
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Cursor {
    line: usize,
    column: usize,
}

impl VirtualCursor for Cursor {
    fn new(line: usize, column: usize) -> Self {
        Self { line, column }
    }

    fn line(&self) -> usize {
        self.line
    }

    fn column(&self) -> usize {
        self.column
    }

    fn move_to(&mut self, line: usize, column: usize) {
        self.line = line;
        self.column = column;
    }
}

// For right now it is a simple linear history
// Probably want it to be a tree
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
struct CursorWithHistory {
    cursor: Cursor,
    history: Vec<Cursor>,
}

impl VirtualCursor for CursorWithHistory {
    fn new(line: usize, column: usize) -> Self {
        Self {
            cursor: Cursor::new(line, column),
            history: Vec::new(),
        }
    }

    fn line(&self) -> usize {
        self.cursor.line()
    }

    fn column(&self) -> usize {
        self.cursor.column()
    }

    fn move_to(&mut self, line: usize, column: usize) {
        self.history.push(self.cursor.clone());
        self.cursor.move_to(line, column);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MultiCursor<C: VirtualCursor> {
    cursors: Vec<C>,
}

impl MultiCursor<Cursor> {
    pub fn new() -> Self {
        Self { cursors: vec![] }
    }

    pub fn add_cursor(&mut self, cursor: Cursor) {
        self.cursors.push(cursor);
    }
}

impl<C: VirtualCursor> VirtualCursor for MultiCursor<C> {
    fn move_up<T: TextBuffer>(&mut self, buffer: &T) {
        for cursor in &mut self.cursors {
            cursor.move_up(buffer);
        }
    }

    fn move_down<T: TextBuffer>(&mut self, buffer: &T) {
        for cursor in &mut self.cursors {
            cursor.move_down(buffer);
        }
    }

    fn move_left<T: TextBuffer>(&mut self, buffer: &T) {
        for cursor in &mut self.cursors {
            cursor.move_left(buffer);
        }
    }

    fn move_right<T: TextBuffer>(&mut self, buffer: &T) {
        for cursor in &mut self.cursors {
            cursor.move_right(buffer);
        }
    }

    fn start_of_line(&mut self) {
        for cursor in &mut self.cursors {
            cursor.start_of_line();
        }
    }

    fn handle_insert<T: TextBuffer<Item = u8>>(&mut self, to_insert: &[u8], buffer: &mut T) {
        for cursor in &mut self.cursors {
            cursor.handle_insert(to_insert, buffer);
        }
    }

    fn move_to(&mut self, line: usize, column: usize) {
        self.cursors = vec![C::new(line, column)];
    }

    fn line(&self) -> usize {
        self.cursors.get(0).map(C::line).unwrap_or(0)
    }

    fn column(&self) -> usize {
        self.cursors.get(0).map(C::column).unwrap_or(0)
    }

    fn new(line: usize, column: usize) -> Self {
        Self {
            cursors: vec![C::new(line, column)],
        }
    }

    fn end_of_line<T: TextBuffer>(&mut self, buffer: &T) {
        for cursor in &mut self.cursors {
            cursor.end_of_line(buffer);
        }
    }

    fn move_in_buffer<T: TextBuffer>(&mut self, line: usize, column: usize, buffer: &T) {
        let mut c = C::new(line, column);
        c.move_in_buffer(line, column, buffer);
        self.cursors = vec![c];
    }

    fn right_of<T: TextBuffer>(&self, buffer: &T) -> Self {
        Self {
            cursors: self.cursors.iter().map(|c| c.right_of(buffer)).collect(),
        }
    }

    fn left_of<T: TextBuffer>(&self, buffer: &T) -> Self {
        Self {
            cursors: self.cursors.iter().map(|c| c.left_of(buffer)).collect(),
        }
    }

    fn above<T: TextBuffer<Item = u8>>(&self, buffer: &T) -> Self {
        Self {
            cursors: self.cursors.iter().map(|c| c.above(buffer)).collect(),
        }
    }

    fn below<T: TextBuffer>(&self, buffer: &T) -> Self {
        Self {
            cursors: self.cursors.iter().map(|c| c.below(buffer)).collect(),
        }
    }

    fn auto_bracket_insert<T: TextBuffer<Item = u8>>(&mut self, buffer: &mut T, to_insert: &[u8]) {
        for cursor in &mut self.cursors {
            cursor.auto_bracket_insert(buffer, to_insert);
        }
    }

    fn insert_normal_text<T: TextBuffer<Item = u8>>(&mut self, to_insert: &[u8], buffer: &mut T) {
        for cursor in &mut self.cursors {
            cursor.insert_normal_text(to_insert, buffer);
        }
    }

    fn delete_char<T: TextBuffer>(&mut self, buffer: &mut T) {
        for cursor in &mut self.cursors {
            cursor.delete_char(buffer);
        }
    }
}

// TODO:
// To make this a fully headless editor I need to handle all the basic interactions
// Selections
// Undo/redo
// Copy/cut/paste
// Text decorations?
// Text annotation?
// What text is visible
// Real Text Buffer
// Save to File?
// Load from File?
// Single line support

// TODO: Make a fake text buffer impl
// test it by doing compensating actions
// and making sure state is always reset.

struct EventTextBuffer {
    text_positions: HashMap<(usize, usize), u8>,
    bytes: Vec<u8>,
}

// This is a weird implementation because I can insert anywhere
#[cfg(test)]
impl EventTextBuffer {
    fn new() -> Self {
        Self {
            text_positions: HashMap::new(),
            bytes: Vec::new(),
        }
    }
}

impl TextBuffer for EventTextBuffer {
    type Item = u8;
    fn line_length(&self, _line: usize) -> usize {
        80
    }

    fn line_count(&self) -> usize {
        80
    }

    fn insert_bytes(&mut self, line: usize, column: usize, text: &[u8]) {
        for (i, byte) in text.iter().enumerate() {
            self.text_positions.insert((line, column + i), *byte);
        }
        let mut text_positions: Vec<(&(usize, usize), &u8)> = self.text_positions.iter().collect();
        text_positions.sort_by_key(|x| x.0);
        self.bytes = text_positions.iter().map(|x| *x.1).collect();
    }

    fn byte_at_pos(&self, line: usize, column: usize) -> Option<&u8> {
        self.text_positions.get(&(line, column))
    }

    fn delete_char(&mut self, line: usize, column: usize) {
        self.text_positions.remove(&(line, column));
    }

    fn lines(&self) -> LineIter<'_, Self::Item> {
        LineIter {
            current_position: 0,
            items: &self.bytes,
            newline: &b'\n',
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn adding_and_remove_basic() {
        let mut cursor = Cursor::new(0, 0);
        let mut buffer = EventTextBuffer::new();
        let my_string = b"Hello World";
        cursor.handle_insert(my_string, &mut buffer);
        for i in 0..my_string.len() {
            assert_eq!(buffer.byte_at_pos(0, i), Some(&my_string[i]));
        }
        for _ in 0..my_string.len() {
            cursor.delete_char(&mut buffer);
        }

        assert!(buffer.text_positions.is_empty());
    }

    #[test]
    fn adding_and_remove_basic_multi() {
        let cursor = Cursor::new(0, 0);
        let mut buffer = EventTextBuffer::new();
        let cursor_below = cursor.below(&buffer);
        let my_string = b"Hello World";

        let mut multi_cursor = MultiCursor::new();
        multi_cursor.add_cursor(cursor);
        multi_cursor.add_cursor(cursor_below);

        multi_cursor.handle_insert(my_string, &mut buffer);

        for cursor in multi_cursor.cursors.iter() {
            for i in 0..my_string.len() {
                assert_eq!(buffer.byte_at_pos(cursor.line(), i), Some(&my_string[i]));
            }
        }

        for _ in 0..my_string.len() {
            multi_cursor.delete_char(&mut buffer);
        }

        assert!(buffer.text_positions.is_empty());
    }

    #[test]
    fn test_lines() {
        let mut cursor = Cursor::new(0, 0);
        let mut buffer = EventTextBuffer::new();
        let my_string = b"Hello World";

        cursor.handle_insert(my_string, &mut buffer);
        cursor.handle_insert(b"\n", &mut buffer);
        cursor.handle_insert(b"Hello World", &mut buffer);
        cursor.handle_insert(b"\n", &mut buffer);

        for line in buffer.lines() {
            assert!(line == my_string)
        }
    }
}
