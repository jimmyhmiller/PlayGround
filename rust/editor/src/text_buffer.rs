use std::{ops::{Index, IndexMut, RangeBounds}, str::from_utf8};

use crate::{tokenizer::Tokenizer, Cursor, EditAction};


#[derive(Debug, Clone)]
pub struct TextBuffer {
    pub line_range: Vec<(usize, usize)>,
    pub chars: Vec<u8>,
    pub max_line_width_cache: usize,
    pub tokenizer: Tokenizer,
}

impl Index<usize> for TextBuffer {
    type Output = (usize, usize);
    fn index(&self, index: usize) -> &Self::Output {
        &self.line_range[index]
    }
}

impl IndexMut<usize> for TextBuffer {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.line_range[index]
    }
}

impl TextBuffer {

    pub fn new(text: &str) -> Self {
        let mut buffer = TextBuffer {
            chars: text.as_bytes().to_vec(),
            line_range: vec![],
            max_line_width_cache: 0,
            tokenizer: Tokenizer::new(),
        };
        buffer.parse_lines();
        buffer
    }

    pub fn get_line(&self, line: usize) -> Option<&(usize,usize)> {
        self.line_range.get(line)
    }

    pub fn line_count(&self) -> usize {
        self.line_range.len()
    }

    pub fn line_length(&self, line: usize) -> usize {
        if let Some(line_range) = self.get_line(line) {
            line_range.1 - line_range.0
        } else {
            // Should this be 0?
            0
        }
    }

    pub fn delete_line(&mut self, line: usize) {
        self.line_range.remove(line);
    }

    pub fn insert(&mut self, index: usize, line: (usize, usize)) {
        self.line_range.insert(index, line);
    }

    pub fn splice<R: RangeBounds<usize>, I: IntoIterator<Item=(usize,usize)>>(&mut self, index: R, lines: I) {
        self.line_range.splice(index, lines);
    }

    pub fn lines_iter_mut(&mut self) -> impl Iterator<Item=&mut (usize,usize)> {
        self.line_range.iter_mut()
    }

    pub fn last_line(&self) -> Option<&(usize,usize)> {
        self.line_range.last()
    }

    pub fn parse_lines(&mut self) {
        let mut line_start = 0;
        let mut line_range = Vec::<(usize,usize)>::with_capacity(self.chars.len()/60);
        for (line_end, char) in self.chars.iter().enumerate() {
            if *char == b'\n'{
                line_range.push((line_start, line_end));
                line_start = line_end + 1;
            }
            if line_end == self.chars.len() - 1 {
                line_range.push((line_start, line_end + 1));
            }
        }
        if line_range.is_empty() {
            line_range.push((0, 0));
        }
        self.line_range = line_range;

    }

    pub fn set_contents(&mut self, contents: &[u8]) {
        self.chars = contents.to_vec();
        self.parse_lines();
    }

    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    pub fn char_length(&self) -> usize {
        self.chars.len()
    }

    pub fn insert_char(&mut self, cursor: Cursor, to_insert: &[u8]) -> EditAction {
        // This is assuming that to_insert is a single character.
        let Cursor(cursor_line, cursor_column) = cursor;
        // TODO: If this line doesn't exist, I need to make it exist.
        if self.line_range.len() <= cursor_line {
            self.chars.extend(format!("\nerror {:?}, {}, {:?}, {:?}", self.line_range, cursor_line, cursor, from_utf8(to_insert)).as_bytes().to_vec());
            self.parse_lines();
            return EditAction::Noop;
        }
        let line_start = self[cursor_line].0;
        let char_pos = line_start + cursor_column;
        // TODO:
        // I have panic here. I think it is when I delete a selection
        // and try to undo because I don't capture deleting selections in the transaction system.
        self.chars.splice(char_pos..char_pos, to_insert.to_vec());

        let mut lines_to_skip = 1;
        if to_insert == [b'\n'] {
            let (start, end) = self[cursor_line];
            if char_pos >= end && cursor_column != 0 {
                self.insert(cursor_line + 1, (char_pos+1, char_pos+1));
            } else if cursor_column == 0 {
                self.splice(cursor_line..cursor_line+1, [(start,char_pos), (start+1, end+1)]);
            } else {
                self.splice(cursor_line..cursor_line + 1, [(start, char_pos), (char_pos+1, end+1)]);
            }
            lines_to_skip = 2;
        } else {
            self[cursor_line] = (line_start, self[cursor_line].1 + 1);
        }

        for mut line in self.lines_iter_mut().skip(cursor_line + lines_to_skip) {
            line.0 += 1;
            line.1 += 1;
        }

        EditAction::Insert((cursor_line, cursor_column), std::str::from_utf8(to_insert).unwrap().to_string())
    }

    pub fn remove_char(&mut self, cursor: Cursor) -> EditAction {

        // TODO: Clean Up
        let Cursor(cursor_line, cursor_column) = cursor;

        let line = self.get_line(cursor_line);
        if line.is_none() {
            return EditAction::Noop;
        }
        let line = line.unwrap();

        let char_pos = (line.0 + cursor_column).saturating_sub(1);

        if self.is_empty() || char_pos >= self.char_length( ){
            return EditAction::Noop;
        }
        if cursor_line == 0 && cursor_column == 0 {
            return EditAction::Noop;
        }

        let result = self.chars.remove(char_pos);

        self[cursor_line].1 = self[cursor_line].1.saturating_sub(1);
        let mut line_erased = false;
        if cursor_column == 0 {
            self[cursor_line - 1] = (self[cursor_line - 1].0, self[cursor_line].1);
            self.delete_line(cursor_line);
            line_erased = true;
        }

        for mut line in self.lines_iter_mut().skip(cursor_line + if line_erased { 0} else {1}) {
            line.0 = line.0.saturating_sub(1);
            line.1 = line.1.saturating_sub(1);
        }
        // TODO: I had a crash here
        EditAction::Delete((cursor_line, cursor_column),
                            from_utf8(&[result]).unwrap().to_string())
    }

    pub fn max_line_width(&mut self) -> usize {
        // TODO: Update cache on edit!
        if self.max_line_width_cache != 0 {
            return self.max_line_width_cache;
        }
        self.max_line_width_cache = self.line_range.iter().map(|(start,end)| end - start).max().unwrap_or(0);
        self.max_line_width_cache
    }

    pub fn char_position_from_line_column(&self, line_number: usize, column_number: usize) -> Option<usize> {
        if let Some((start_line, end_line))  = self.line_range.get(line_number) {
            if (end_line - start_line) < column_number {
                return None;
            }
            Some(start_line + column_number)
        } else {
            None
        }

    }

    pub fn line_column_from_char_position(&self, position: usize) -> Option<Cursor> {
        for (i, (start, end)) in self.line_range.iter().enumerate() {
            if position >= *start && position <= *end {
                return Some(Cursor(i, position - start));
            }
        }
        None
    }

    pub fn get_text(&self) -> &str {
        from_utf8(&self.chars).unwrap()
    }

}