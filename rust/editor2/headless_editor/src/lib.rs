#![allow(clippy::single_match)]
use core::cmp::min;
use core::fmt::Debug;
use std::collections::HashMap;
use std::str::from_utf8;
// TODO: I probably do need to return actions here
// for every time the cursor moves.

pub mod transaction;

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
    fn line_start(&self, line: usize) -> usize;
    fn line_count(&self) -> usize;
    // Rethink bytes because of utf8
    fn insert_bytes(&mut self, line: usize, column: usize, text: &[Self::Item]);
    fn byte_at_pos(&self, line: usize, column: usize) -> Option<&Self::Item>;
    fn delete_char(&mut self, line: usize, column: usize);
    fn lines(&self) -> LineIter<Self::Item>;
    fn last_line(&self) -> usize {
        self.line_count().saturating_sub(1)
    }
    fn get_line(&self, index: usize) -> Option<&[Self::Item]>;
    fn get_line_start_end(&self, index: usize) -> Option<(usize, usize)> {
        let start = self.line_start(index);
        let length = self.line_length(index);
        if length == 0 {
            return None;
        }
        Some((start, start + length))
    }
    fn set_contents(&mut self, contents: &[Self::Item]);
    fn contents(&self) -> &[Self::Item];
    fn max_line_length(&self) -> usize {
        let mut max = 0;
        for i in 0..self.line_count() {
            max = max.max(self.line_length(i));
        }
        max
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Token {
    pub delta_line: usize,
    pub delta_start: usize,
    pub length: usize,
    pub kind: usize,
    pub modifiers: usize,
}

impl From<&[u64]> for Token {
    fn from(chunk: &[u64]) -> Self {
        assert!(
            chunk.len() == 5,
            "Expected chunk to be of length 5, but was {}",
            chunk.len(),
        );
        Token {
            delta_line: chunk[0] as usize,
            delta_start: chunk[1] as usize,
            length: chunk[2] as usize,
            kind: chunk[3] as usize,
            modifiers: chunk[4] as usize,
        }
    }
}

pub fn parse_tokens(tokens: &[u64]) -> Vec<Token> {
    tokens.chunks(5).map(Token::from).collect()
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Edit {
    Insert(usize, usize, Vec<u8>),
    Delete(usize, usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct EditEvent {
    pub edit: Edit,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TokenTextBuffer<T: TextBuffer> {
    pub tokens: Vec<Token>,
    pub underlying_text_buffer: T,
    pub edits: Vec<EditEvent>,
    pub document_version: usize,
    pub token_actions: Vec<TokenAction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TokenAction {
    SplitToken {
        index: usize,
        offset: usize,
    },
    MergeToken,
    DeleteToken,
    CreateToken,
    NewLine {
        line: usize,
        column: usize,
        index: usize,
    },
    DeleteNewLine {
        line: usize,
        column: usize,
        index: usize,
    },
    OffsetToken {
        index: usize,
        length: isize,
    },
    ChangeTokenLength {
        index: usize,
        length: isize,
    },
    JoinLine {
        line: usize,
        column: usize,
        index: usize,
    },
    NewLineAbove {
        line: usize,
        column: usize,
        index: usize,
    },
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TokenWindowKind {
    Left { index: usize, offset: usize },
    Inside { index: usize, offset: usize },
    Above { index: usize },
}

impl TokenWindowKind {
    fn get_index(&self) -> usize {
        match self {
            TokenWindowKind::Left { index, .. } => *index,
            TokenWindowKind::Inside { index, .. } => *index,
            TokenWindowKind::Above { index } => *index,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TokenWindow {
    pub kind: Option<TokenWindowKind>,
    pub index: usize,
    pub line: usize,
    pub column: usize,
    pub left: Option<Token>,
    pub center: Option<Token>,
    pub right: Option<Token>,
}

impl<T> TokenTextBuffer<T>
where
    T: TextBuffer<Item = u8>,
{
    // TODO: Make this an iterator instead
    pub fn decorated_lines(&self, skip: usize, take: usize) -> Vec<Vec<(&[u8], Option<&Token>)>> {
        if self.tokens.is_empty() {
            return self
                .lines()
                .skip(skip)
                .take(take)
                .map(|line| vec![(line, None)])
                .collect();
        }

        let mut result = vec![];
        for (relative_line_number, (line, tokens)) in self
            .lines()
            .skip(skip)
            .take(take)
            .zip(self.tokens.token_lines().skip(skip).take(take))
            .enumerate()
        {
            let _line_number = relative_line_number + skip;
            result.push(self.decorated_line(line, tokens));
        }
        result
    }

    fn decorated_line<'a>(
        &self,
        line: &'a [u8],
        tokens: &'a [Token],
    ) -> Vec<(&'a [u8], Option<&'a Token>)> {
        // TODO: Need to account for edits

        let mut result = vec![];
        let mut current_position = 0;
        let mut last_end = 0;
        for token in tokens.iter() {
            current_position += token.delta_start;
            if current_position > last_end {
                let non_token_end = current_position;
                let non_token_end = std::cmp::min(non_token_end, line.len());
                let non_token_range = last_end..non_token_end;
                result.push((&line[non_token_range], None));
            }
            let end = current_position + token.length;
            let end = std::cmp::min(end, line.len());
            let start = std::cmp::min(current_position, end);
            let token_range = start..end;
            result.push((&line[token_range], Some(token)));
            last_end = end;
        }
        if last_end < line.len() {
            let non_token_range = last_end..line.len();
            result.push((&line[non_token_range], None));
        }

        result
    }
    pub fn drain_edits(&mut self) -> Vec<EditEvent> {
        std::mem::take(&mut self.edits)
    }

    pub fn set_tokens(&mut self, tokens: Vec<Token>) {
        self.tokens = tokens;
    }

    fn update_tokens_insert(&mut self, line: usize, column: usize, text: &[u8]) {
        let window: TokenWindow = self.find_token(line, column);
        let actions = self.resolve_token_action_insert(window, line, column, text);
        self.token_actions.extend(actions.clone());

        for action in actions.iter() {
            self.apply_token_action(action);
        }
    }

    fn update_tokens_delete(&mut self, line: usize, column: usize, text: &[u8]) {
        // We need to know where the cursor was,
        // not the character.
        let mut cursor = SimpleCursor::new(line, column);
        cursor.move_right(self);

        let window: TokenWindow = self.find_token(cursor.line(), cursor.column());
        let actions = self.result_token_action_delete(window, line, column, text);
        self.token_actions.extend(actions.clone());
        for action in actions.iter() {
            self.apply_token_action(action);
        }
    }

    pub fn find_token(&self, target_line: usize, target_column: usize) -> TokenWindow {
        let mut total_tokens = 0;
        for (line_number, line) in self
            .decorated_lines(0, self.line_count())
            .iter()
            .enumerate()
        {
            if line_number == target_line {
                let token_window_kind: TokenWindowKind =
                    self.find_token_in_line(line, target_column, total_tokens);
                let index = token_window_kind.get_index();
                // TODO: Only have center if in inside

                let mut center = self.tokens.get(index).cloned();
                let left = if index == 0 {
                    None
                } else {
                    self.tokens.get(index.saturating_sub(1)).cloned()
                };
                let mut right = self.tokens.get(index + 1).cloned();

                match token_window_kind {
                    TokenWindowKind::Left { .. } => {
                        right = center;
                        center = None;
                    }
                    TokenWindowKind::Above { .. } => {
                        right = center;
                        center = None;
                    }
                    TokenWindowKind::Inside { .. } => {}
                }

                return TokenWindow {
                    kind: Some(token_window_kind),
                    index,
                    line: target_line,
                    column: target_column,
                    left,
                    center,
                    right,
                };
            }
            total_tokens += line.iter().filter(|(_, token)| token.is_some()).count();
        }

        TokenWindow {
            kind: None,
            index: 0,
            line: target_line,
            column: target_column,
            left: None,
            center: None,
            right: None,
        }
    }

    fn resolve_token_action_insert(
        &self,
        window: TokenWindow,
        line: usize,
        column: usize,
        text: &[u8],
    ) -> Vec<TokenAction> {
        let mut actions = vec![];

        if window.kind.is_none() {
            return vec![];
        }
        // TODO: Extend tokens based on offset
        match window.kind.unwrap() {
            TokenWindowKind::Left { .. } => {
                for char in text {
                    match char {
                        b'\n' => {
                            actions.push(TokenAction::NewLine {
                                line,
                                column,
                                index: window.index,
                            });
                        }
                        _ => actions.push(TokenAction::OffsetToken {
                            index: window.index,
                            length: 1,
                        }),
                    }
                }
            }
            TokenWindowKind::Inside { offset, .. } => {
                let mut token_splits = 0;
                for char in text {
                    match char {
                        b'\n' => {
                            actions.push(TokenAction::SplitToken {
                                index: window.index + token_splits,
                                offset,
                            });
                            token_splits += 1;
                            actions.push(TokenAction::NewLine {
                                line,
                                column,
                                index: window.index + token_splits,
                            });
                        }
                        _ => actions.push(TokenAction::ChangeTokenLength {
                            index: window.index + token_splits,
                            length: 1,
                        }),
                    }
                }
            }
            TokenWindowKind::Above { .. } => {
                // TODO: A different action other than new line
                // so we know not to change the offset
                for char in text {
                    match char {
                        b'\n' => {
                            actions.push(TokenAction::NewLineAbove {
                                line,
                                column,
                                index: window.index,
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
        actions
    }

    fn result_token_action_delete(
        &self,
        window: TokenWindow,
        line: usize,
        column: usize,
        text: &[u8],
    ) -> Vec<TokenAction> {
        if window.kind.is_none() {
            return vec![];
        }
        let mut actions = vec![];
        match window.kind.unwrap() {
            TokenWindowKind::Left { offset, .. } => {
                let mut changed_offset: usize = 0;
                for char in text {
                    match char {
                        b'\n' => {
                            actions.push(TokenAction::JoinLine {
                                line,
                                column,
                                index: window.index,
                            });
                        }
                        _ => {
                            if offset.saturating_sub(changed_offset) == 0 {
                                actions.push(TokenAction::ChangeTokenLength {
                                    index: window.index.saturating_sub(1),
                                    length: -1,
                                });
                                changed_offset += 1;
                            } else {
                                actions.push(TokenAction::OffsetToken {
                                    index: window.index,
                                    length: -1,
                                })
                            }
                        }
                    }
                }
            }
            TokenWindowKind::Inside { .. } => {
                for char in text {
                    match char {
                        // TODO: Is this possible?
                        b'\n' => {
                            actions.push(TokenAction::DeleteNewLine {
                                line,
                                column,
                                index: window.index,
                            });
                        }
                        _ => actions.push(TokenAction::ChangeTokenLength {
                            index: window.index,
                            length: -1,
                        }),
                    }
                }
            }
            TokenWindowKind::Above { .. } => {
                for char in text {
                    match char {
                        b'\n' => {
                            actions.push(TokenAction::DeleteNewLine {
                                line,
                                column,
                                index: window.index,
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
        actions
    }

    // TODO: Backspace .
    // Add at end of token (Do I need right again?)

    fn apply_token_action(&mut self, action: &TokenAction) {
        match action {
            TokenAction::SplitToken { index, offset } => {
                if let Some(token) = self.tokens.get_mut(*index) {
                    token.length = *offset;
                    let remaining_length = token.length - offset;
                    let new_token = Token {
                        delta_start: token.delta_start,
                        length: remaining_length,
                        ..*token
                    };
                    self.tokens.insert(*index + 1, new_token);
                }
            }
            TokenAction::MergeToken => todo!(),
            TokenAction::DeleteToken => todo!(),
            TokenAction::CreateToken => todo!(),
            TokenAction::OffsetToken { index, length } => {
                if let Some(token) = self.tokens.get_mut(*index) {
                    if length.is_negative() {
                        token.delta_start = token.delta_start.saturating_sub(length.unsigned_abs());
                    } else {
                        token.delta_start += *length as usize;
                    }
                }
            }
            TokenAction::ChangeTokenLength { index, length } => {
                if let Some(token) = self.tokens.get_mut(*index) {
                    // TODO: Is this right?
                    token.length = token.length.checked_add_signed(*length).unwrap_or(0);
                }
                if let Some(token) = self.tokens.get_mut(index + 1) {
                    if token.delta_line == 0 {
                        token.delta_start =
                            token.delta_start.checked_add_signed(*length).unwrap_or(0)
                    }
                }
            }
            TokenAction::NewLine {
                line: _,
                column: _,
                index,
            } => {
                if let Some(token) = self.tokens.get_mut(*index) {
                    token.delta_line += 1;
                    token.delta_start = 0;
                }
            }
            TokenAction::NewLineAbove {
                line: _,
                column: _,
                index,
            } => {
                if let Some(token) = self.tokens.get_mut(*index) {
                    token.delta_line += 1;
                }
            }
            TokenAction::DeleteNewLine {
                line,
                column: _,
                index,
            } => {
                if self.tokens.is_empty() {
                    return;
                }
                let line = self.tokens.token_lines().nth(*line);
                assert!(line.is_some(), "Expected line to be found");
                if let Some(token) = self.tokens.get_mut(*index) {
                    token.delta_line -= 1;
                }
            }
            TokenAction::JoinLine {
                line,
                column: _,
                index,
            } => {
                let decorated_line = self.decorated_lines(*line, 1).first().cloned();
                let mut extra_delta = 0;
                // TODO: Need to get the last two
                // So if the last isn't a token I can get the the second to last entry
                // for a token length
                if let Some(line) = decorated_line {
                    let line: Vec<_> = line.iter().rev().take(2).rev().collect();
                    match (line.get(0), line.get(1)) {
                        (None, None) => {}
                        (None, Some((text, None))) => extra_delta = text.len(),
                        (Some((text, None)), None) => extra_delta = text.len(),
                        (None, Some((_, Some(token)))) => extra_delta = token.length,
                        (Some((_, Some(token))), None) => extra_delta = token.length,
                        (_, Some((_, Some(token)))) => extra_delta = token.length,
                        (Some((_, Some(token))), Some((text, None))) => {
                            extra_delta = token.length + text.len()
                        }
                        (Some((_, None)), Some((_, None))) => {
                            unreachable!("Should not have this")
                        }
                    }
                }
                if let Some(token) = self.tokens.get_mut(*index) {
                    token.delta_line -= 1;
                    token.delta_start += extra_delta;
                }
            }
        }
    }

    fn find_token_in_line(
        &self,
        line: &[(&[u8], Option<&Token>)],
        target_column: usize,
        starting_index: usize,
    ) -> TokenWindowKind {
        // TODO: offset is wrong on the first line when deleting spaces
        // What exactly is offset supposed to be?
        // I think I'm defining it as offset from last token
        // but it should be last token on line
        let mut current_column = 0;
        let mut current_index = starting_index;
        for (index, (text, token)) in line.iter().enumerate() {
            let end = current_column + text.len();
            if target_column < end && target_column > current_column {
                if token.is_some() {
                    let offset = target_column.saturating_sub(current_column);
                    return TokenWindowKind::Inside {
                        index: current_index,
                        offset,
                    };
                } else {
                    if index == line.len() - 1 {
                        return TokenWindowKind::Above {
                            index: current_index,
                        };
                    }
                    let offset = target_column - current_column;
                    return TokenWindowKind::Left {
                        index: current_index,
                        offset,
                    };
                }
            } else if target_column == current_column {
                if index == line.len() - 1 && token.is_none() {
                    return TokenWindowKind::Above {
                        index: current_index,
                    };
                }

                let offset = if let Some(token) = token {
                    if let Some(left_token) = self.tokens.get(current_index.saturating_sub(1)) {
                        if token.delta_line == 0 {
                            token.delta_start.saturating_sub(left_token.length)
                        } else {
                            token.delta_start
                        }
                    } else {
                        0
                    }
                } else {
                    0
                };
                return TokenWindowKind::Left {
                    index: current_index,
                    offset,
                };
            }
            if token.is_some() {
                current_index += 1;
            }
            current_column += text.len();
        }
        // TODO: Is this right?
        TokenWindowKind::Above {
            index: current_index,
        }
    }

    fn add_edit_action(&mut self, event: EditEvent) {
        self.document_version += 1;
        self.edits.push(event);
    }
}

impl<T> TextBuffer for TokenTextBuffer<T>
where
    T: TextBuffer<Item = u8>,
{
    type Item = u8;

    fn line_length(&self, line: usize) -> usize {
        self.underlying_text_buffer.line_length(line)
    }

    fn line_start(&self, line: usize) -> usize {
        self.underlying_text_buffer.line_start(line)
    }

    fn line_count(&self) -> usize {
        self.underlying_text_buffer.line_count()
    }

    fn insert_bytes(&mut self, line: usize, column: usize, text: &[Self::Item]) {
        self.update_tokens_insert(line, column, text);
        self.underlying_text_buffer.insert_bytes(line, column, text);

        let event = EditEvent {
            edit: Edit::Insert(line, column, text.to_vec()),
        };

        self.add_edit_action(event);
    }

    fn byte_at_pos(&self, line: usize, column: usize) -> Option<&Self::Item> {
        self.underlying_text_buffer.byte_at_pos(line, column)
    }

    fn delete_char(&mut self, line: usize, column: usize) {
        if let Some(byte) = self.underlying_text_buffer.byte_at_pos(line, column) {
            self.update_tokens_delete(line, column, &[*byte]);
            self.underlying_text_buffer.delete_char(line, column);
            self.add_edit_action(EditEvent {
                edit: Edit::Delete(line, column),
            });
        }
    }

    fn lines(&self) -> LineIter<Self::Item> {
        self.underlying_text_buffer.lines()
    }

    fn get_line(&self, index: usize) -> Option<&[Self::Item]> {
        self.underlying_text_buffer.get_line(index)
    }

    fn set_contents(&mut self, contents: &[Self::Item]) {
        self.underlying_text_buffer.set_contents(contents);
    }

    fn contents(&self) -> &[Self::Item] {
        self.underlying_text_buffer.contents()
    }
}

impl TokenTextBuffer<SimpleTextBuffer> {
    pub fn new_with_contents(contents: &[u8]) -> Self {
        let underlying_text_buffer = SimpleTextBuffer::new_with_contents(contents);
        Self {
            document_version: 0,
            tokens: vec![],
            underlying_text_buffer,
            edits: vec![],
            token_actions: vec![],
        }
    }
}

pub struct TokenLineIter<'a> {
    current_position: usize,
    tokens: &'a [Token],
    empty_lines: usize,
    inital_empty_lines: bool,
}

impl<'a> Iterator for TokenLineIter<'a> {
    type Item = &'a [Token];
    fn next(&mut self) -> Option<Self::Item> {
        let original_position = self.current_position;

        if self.empty_lines > 0 {
            self.empty_lines -= 1;
            return Some(&[]);
        }
        while self.current_position < self.tokens.len() {
            let token = &self.tokens[self.current_position];
            if self.current_position != original_position && token.delta_line == 1 {
                self.empty_lines = 0;
                return Some(&self.tokens[original_position..self.current_position]);
            } else if self.current_position != original_position && token.delta_line > 1 {
                self.empty_lines = token.delta_line - 1;
                return Some(&self.tokens[original_position..self.current_position]);
            } else if self.current_position == 0
                && token.delta_line >= 1
                && !self.inital_empty_lines
            {
                self.empty_lines = token.delta_line - 1;
                self.inital_empty_lines = true;
                return Some(&[]);
            }
            self.current_position += 1;
        }
        if self.current_position != original_position {
            let line = &self.tokens[original_position..self.current_position];
            return Some(line);
        }
        None
    }
}

trait TokenLinerIterExt<'a> {
    fn token_lines(self) -> TokenLineIter<'a>;
}

impl<'a> TokenLinerIterExt<'a> for &'a [Token] {
    fn token_lines(self) -> TokenLineIter<'a> {
        TokenLineIter {
            current_position: 0,
            tokens: self,
            empty_lines: 0,
            inital_empty_lines: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SimpleTextBuffer {
    pub bytes: Vec<u8>,
}

impl Default for SimpleTextBuffer {
    fn default() -> Self {
        Self::new()
    }
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
}

impl TextBuffer for SimpleTextBuffer {
    type Item = u8;

    fn line_start(&self, line: usize) -> usize {
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
        // check range
        if start > self.bytes.len() {
            return;
        }
        self.bytes.splice(start..start, text.iter().cloned());
    }

    fn byte_at_pos(&self, line: usize, column: usize) -> Option<&u8> {
        self.bytes.get(self.line_start(line) + column)
    }

    fn delete_char(&mut self, line: usize, column: usize) {
        let start = self.line_start(line) + column;
        if start >= self.bytes.len() {
            return;
        }
        self.bytes.remove(start);
    }

    fn lines(&self) -> LineIter<Self::Item> {
        LineIter {
            current_position: 0,
            items: &self.bytes,
            newline: &b'\n',
        }
    }

    fn get_line(&self, index: usize) -> Option<&[Self::Item]> {
        self.lines().nth(index)
    }

    fn set_contents(&mut self, contents: &[Self::Item]) {
        self.bytes = contents.to_vec();
    }

    fn contents(&self) -> &[Self::Item] {
        &self.bytes
    }
}

pub trait VirtualCursor: Clone + Debug {
    fn move_to(&mut self, line: usize, column: usize);
    fn line(&self) -> usize;
    fn column(&self) -> usize;
    fn selection(&self) -> Option<((usize, usize), (usize, usize))>;
    fn set_selection(&mut self, selection: Option<((usize, usize), (usize, usize))>);
    fn new(line: usize, column: usize) -> Self;

    fn set_selection_ordered(&mut self, selection: Option<((usize, usize), (usize, usize))>) {
        set_selection_ordered(self, selection)
    }
    fn set_selection_movement(&mut self, new_location: (usize, usize)) {
        set_selection_movement(self, new_location)
    }
    fn remove_empty_selection(&mut self) {
        remove_empty_selection(self)
    }
    fn move_to_bounded<T: TextBuffer>(&mut self, line: usize, column: usize, buffer: &T) {
        move_to_bounded(self, line, column, buffer)
    }
    fn set_selection_bounded<T: TextBuffer>(
        &mut self,
        selection: Option<((usize, usize), (usize, usize))>,
        text_buffer: &T,
    ) {
        set_selection_bounded(self, selection, text_buffer)
    }
    fn move_up<T: TextBuffer>(&mut self, buffer: &T) {
        move_up(self, buffer)
    }
    fn move_down<T: TextBuffer>(&mut self, buffer: &T) {
        move_down(self, buffer)
    }
    fn move_left<T: TextBuffer>(&mut self, buffer: &T) {
        move_left(self, buffer)
    }
    fn move_right<T: TextBuffer>(&mut self, buffer: &T) {
        move_right(self, buffer)
    }
    fn start_of_line(&mut self) {
        start_of_line(self)
    }
    fn end_of_line<T: TextBuffer>(&mut self, buffer: &T) {
        end_of_line(self, buffer)
    }
    fn move_in_buffer<T: TextBuffer>(&mut self, line: usize, column: usize, buffer: &T) {
        move_in_buffer(self, line, column, buffer)
    }
    fn line_at<T: TextBuffer<Item = u8>>(&mut self, line: usize, buffer: &T) -> Option<String> {
        line_at(self, line, buffer)
    }
    fn right_of<T: TextBuffer>(&self, buffer: &T) -> Self {
        right_of(self, buffer)
    }
    fn left_of<T: TextBuffer>(&self, buffer: &T) -> Self {
        left_of(self, buffer)
    }
    fn above<T: TextBuffer>(&self, buffer: &T) -> Self {
        above(self, buffer)
    }
    fn below<T: TextBuffer>(&self, buffer: &T) -> Self {
        below(self, buffer)
    }
    fn auto_bracket_insert<T: TextBuffer<Item = u8>>(&mut self, buffer: &mut T, to_insert: &[u8]) {
        auto_bracket_insert(self, buffer, to_insert)
    }
    fn insert_normal_text<T: TextBuffer<Item = u8>>(&mut self, to_insert: &[u8], buffer: &mut T) {
        insert_normal_text(self, to_insert, buffer)
    }
    fn delete<T: TextBuffer<Item = u8>>(&mut self, buffer: &mut T) {
        delete(self, buffer)
    }
    fn delete_chars<T: TextBuffer<Item = u8>>(
        &mut self,
        buffer: &mut T,
        start: (usize, usize),
        end: (usize, usize),
    ) {
        delete_chars(self, buffer, start, end)
    }
    fn delete_selection<T: TextBuffer<Item = u8>>(&mut self, buffer: &mut T) {
        delete_selection(self, buffer)
    }
    fn delete_char<T: TextBuffer<Item = u8>>(&mut self, buffer: &mut T) {
        delete_char(self, buffer)
    }

    fn get_last_line<T: TextBuffer<Item = u8>>(&mut self, buffer: &mut T) -> Option<String> {
        get_last_line(self, buffer)
    }
    fn auto_indent<T: TextBuffer<Item = u8>>(&mut self, to_insert: &[u8], buffer: &mut T) {
        auto_indent(self, to_insert, buffer)
    }
    fn handle_insert<T: TextBuffer<Item = u8>>(&mut self, to_insert: &[u8], buffer: &mut T) {
        handle_insert(self, to_insert, buffer)
    }

    fn nearest_text_position<T: TextBuffer>(
        &mut self,
        line: usize,
        column: usize,
        buffer: &T,
    ) -> Self {
        nearest_text_position(self, line, column, buffer)
    }
}

fn set_selection_ordered<Cursor: VirtualCursor>(
    cursor: &mut Cursor,
    selection: Option<((usize, usize), (usize, usize))>,
) {
    if let Some(((line1, column1), (line2, column2))) = selection {
        if line1 > line2 || (line1 == line2 && column1 > column2) {
            cursor.set_selection(Some(((line2, column2), (line1, column1))));
        } else {
            cursor.set_selection(Some(((line1, column1), (line2, column2))));
        }
    } else {
        cursor.set_selection(selection);
    }
}

// TODO: This isn't quite right, my cursor should be at the start
// of the selection and I should base things on the end
fn set_selection_movement<Cursor: VirtualCursor>(
    cursor: &mut Cursor,
    new_location: (usize, usize),
) {
    let (start_line, start_column) = new_location;
    let new_start_line = start_line.min(cursor.line());
    let line = cursor.line().max(start_line);
    let mut column = cursor.column();
    let new_start_column =
        if new_start_line != start_line || start_line == line && start_column > column {
            std::mem::replace(&mut column, start_column)
        } else {
            start_column
        };

    cursor.set_selection(Some(((new_start_line, new_start_column), (line, column))));
}

fn remove_empty_selection<Cursor: VirtualCursor>(cursor: &mut Cursor) {
    if let Some(((l1, c1), (l2, c2))) = cursor.selection() {
        if l1 == l2 && c1 == c2 {
            cursor.set_selection(None)
        }
    }
}

fn move_to_bounded<T: TextBuffer, Cursor: VirtualCursor>(
    cursor: &mut Cursor,
    line: usize,
    column: usize,
    buffer: &T,
) {
    let line = min(buffer.last_line(), line);
    let column = min(buffer.line_length(line), column);
    cursor.move_to(line, column);
}

fn set_selection_bounded<T: TextBuffer, Cursor: VirtualCursor>(
    cursor: &mut Cursor,
    selection: Option<((usize, usize), (usize, usize))>,
    text_buffer: &T,
) {
    cursor.set_selection_ordered(selection);
    if let Some(selection) = cursor.selection() {
        let line1 = selection.0 .0.min(text_buffer.line_count());
        let line2 = selection.1 .0.min(text_buffer.line_count());
        let column1 = selection.0 .1.min(text_buffer.line_length(line1));
        let column2 = selection.1 .1.min(text_buffer.line_length(line2));
        cursor.set_selection(Some(((line1, column1), (line2, column2))));
        cursor.remove_empty_selection();
    }
}

fn nearest_text_position<T: TextBuffer, Cursor: VirtualCursor>(
    _cursor: &mut Cursor,
    line: usize,
    column: usize,
    buffer: &T,
) -> Cursor {
    let mut new_cursor = Cursor::new(line, column);
    new_cursor.move_to_bounded(line, column, buffer);
    new_cursor
}

fn move_up<T: TextBuffer, Cursor: VirtualCursor>(cursor: &mut Cursor, buffer: &T) {
    let previous_line = cursor.line().saturating_sub(1);
    cursor.move_to(
        previous_line,
        min(cursor.column(), buffer.line_length(previous_line)),
    );
}

fn move_down<T: TextBuffer, Cursor: VirtualCursor>(cursor: &mut Cursor, buffer: &T) {
    let next_line = cursor.line().saturating_add(1);
    let last_line = buffer.last_line();
    if next_line > last_line {
        return;
    }
    cursor.move_to(
        min(next_line, last_line),
        min(cursor.column(), buffer.line_length(next_line)),
    );
}

fn move_left<T: TextBuffer, Cursor: VirtualCursor>(cursor: &mut Cursor, buffer: &T) {
    if cursor.column() == 0 && cursor.line() != 0 {
        let new_line = cursor.line().saturating_sub(1);
        let length = buffer.line_length(new_line);
        cursor.move_to(new_line, length);
    } else {
        cursor.move_to(cursor.line(), cursor.column().saturating_sub(1));
    }
}

fn move_right<T: TextBuffer, Cursor: VirtualCursor>(cursor: &mut Cursor, buffer: &T) {
    let length = buffer.line_length(cursor.line());
    if cursor.column() >= length {
        if cursor.line().saturating_add(1) < buffer.line_count() {
            cursor.move_to(cursor.line().saturating_add(1), 0);
        }
    } else {
        cursor.move_to(cursor.line(), cursor.column().saturating_add(1));
    }
}

fn start_of_line<Cursor: VirtualCursor>(cursor: &mut Cursor) {
    cursor.move_to(cursor.line(), 0);
}

fn end_of_line<T: TextBuffer, Cursor: VirtualCursor>(cursor: &mut Cursor, buffer: &T) {
    cursor.move_to(cursor.line(), buffer.line_length(cursor.line()));
}

fn move_in_buffer<T: TextBuffer, Cursor: VirtualCursor>(
    cursor: &mut Cursor,
    line: usize,
    column: usize,
    buffer: &T,
) {
    if line < buffer.line_count() {
        cursor.move_to(line, min(column, buffer.line_length(line)));
    }
}

fn line_at<T: TextBuffer<Item = u8>, Cursor: VirtualCursor>(
    _cursor: &mut Cursor,
    line: usize,
    buffer: &T,
) -> Option<String> {
    let found_line = buffer.lines().nth(line);
    found_line
        .and_then(|x| from_utf8(x).ok())
        .map(|x| x.to_string())
}

fn right_of<T: TextBuffer, Cursor: VirtualCursor>(cursor: &Cursor, buffer: &T) -> Cursor {
    let mut cursor = cursor.clone();
    cursor.move_right(buffer);
    cursor
}

fn left_of<T: TextBuffer, Cursor: VirtualCursor>(cursor: &Cursor, buffer: &T) -> Cursor {
    let mut cursor = cursor.clone();
    cursor.move_left(buffer);
    cursor
}

fn above<T: TextBuffer, Cursor: VirtualCursor>(cursor: &Cursor, buffer: &T) -> Cursor {
    let mut cursor = cursor.clone();
    cursor.move_up(buffer);
    cursor
}

fn below<T: TextBuffer, Cursor: VirtualCursor>(cursor: &Cursor, buffer: &T) -> Cursor {
    let mut cursor = cursor.clone();
    cursor.move_down(buffer);
    cursor
}

fn auto_bracket_insert<T: TextBuffer<Item = u8>, Cursor: VirtualCursor>(
    cursor: &mut Cursor,
    buffer: &mut T,
    to_insert: &[u8],
) {
    let to_insert = match to_insert {
        b"(" => b"()",
        b"[" => b"[]",
        b"{" => b"{}",
        b"\"" => b"\"\"",
        _ => to_insert,
    };

    cursor.insert_normal_text(to_insert, buffer);
    cursor.move_right(buffer);
}

fn insert_normal_text<T: TextBuffer<Item = u8>, Cursor: VirtualCursor>(
    cursor: &mut Cursor,
    to_insert: &[u8],
    buffer: &mut T,
) {
    
    buffer.insert_bytes(cursor.line(), cursor.column(), to_insert);
    if to_insert == b"\n" {
        cursor.move_down(buffer);
        cursor.move_to(cursor.line(), 0);
    } else {
        // TODO: Do this more efficiently
        for _ in 0..(to_insert.len().saturating_sub(1)) {
            cursor.move_right(buffer);
        }
        cursor.move_right(buffer)
    };
}

fn delete<T: TextBuffer<Item = u8>, Cursor: VirtualCursor>(cursor: &mut Cursor, buffer: &mut T) {
    if cursor.selection().is_some() {
        cursor.delete_selection(buffer);
    } else {
        cursor.delete_char(buffer);
    }
}

fn delete_chars<T: TextBuffer<Item = u8>, Cursor: VirtualCursor>(
    cursor: &mut Cursor,
    buffer: &mut T,
    start: (usize, usize),
    end: (usize, usize),
) {
    cursor.move_to(end.0, end.1);
    while cursor.line() != start.0 || cursor.column() != start.1 {
        cursor.delete_char(buffer);
    }
}

fn delete_selection<T: TextBuffer<Item = u8>, Cursor: VirtualCursor>(
    cursor: &mut Cursor,
    buffer: &mut T,
) {
    if let Some((start, end)) = cursor.selection() {
        cursor.set_selection(None);
        cursor.delete_chars(buffer, start, end);
    }
}

fn delete_char<T: TextBuffer<Item = u8>, Cursor: VirtualCursor>(
    cursor: &mut Cursor,
    buffer: &mut T,
) {
    cursor.move_left(buffer);
    buffer.delete_char(cursor.line(), cursor.column());
}

fn is_open_bracket(byte: &[u8]) -> bool {
    matches!(byte, b"(" | b"[" | b"{" | b"\"")
}

// TODO: I actually need to use an matching_close_bracket
// This logic makes it so I do the wrong thing if we have
// something like {(}
fn is_close_bracket(byte: &[u8]) -> bool {
    matches!(byte, b")" | b"]" | b"}" | b"\"")
}

fn is_new_line(byte: &[u8]) -> bool {
    matches!(byte, b"\n")
}

fn get_last_line<T: TextBuffer<Item = u8>, Cursor: VirtualCursor>(
    cursor: &mut Cursor,
    buffer: &mut T,
) -> Option<String> {
    if cursor.line() == 0 {
        return None;
    }
    let above = cursor.above(buffer);
    cursor.line_at(above.line(), buffer)
}

/// Broken
fn auto_indent<T: TextBuffer<Item = u8>, Cursor: VirtualCursor>(
    cursor: &mut Cursor,
    to_insert: &[u8],
    buffer: &mut T,
) {
    let last_line: Option<String> = cursor.get_last_line(buffer);
    let current_line: Option<String> = cursor.line_at(cursor.line(), buffer);
    if let (Some(_last_line), Some(current_line)) = (last_line, current_line) {
        let indent = get_indent(&current_line);
        if let Some((last_character_index, last_character)) =
            last_non_whitespace_character(&current_line)
        {
            if cursor.column() < last_character_index {
                cursor.insert_normal_text(to_insert, buffer);
                cursor.handle_insert(indent.as_bytes(), buffer);
            } else {
                match last_character {
                    b'{' => {
                        cursor.insert_normal_text(to_insert, buffer);
                        cursor.handle_insert(increase_indent(indent).as_bytes(), buffer);
                    }
                    b'}' => {
                        cursor.insert_normal_text(to_insert, buffer);
                        cursor.handle_insert(indent.as_bytes(), buffer);
                    }
                    _ => {
                        cursor.insert_normal_text(to_insert, buffer);
                        cursor.handle_insert(indent.as_bytes(), buffer);
                    }
                };
            }
        } else {
            cursor.insert_normal_text(to_insert, buffer);
            cursor.handle_insert(indent.as_bytes(), buffer);
        }
    } else {
        cursor.insert_normal_text(to_insert, buffer);
    }
}

fn handle_insert<T: TextBuffer<Item = u8>, Cursor: VirtualCursor>(
    cursor: &mut Cursor,
    to_insert: &[u8],
    buffer: &mut T,
) {
    if to_insert.is_empty() {
        return;
    }
    if is_open_bracket(to_insert) {
        // Would need to have a setting for this
        cursor.auto_bracket_insert(buffer, to_insert);
    } else if is_close_bracket(to_insert) {
        let right_of_cursor = buffer.byte_at_pos(cursor.line(), cursor.column());

        match right_of_cursor {
            Some(right) if is_close_bracket(&[*right]) => cursor.move_right(buffer),
            _ => cursor.insert_normal_text(to_insert, buffer),
        }
    } else if is_new_line(to_insert) {
        cursor.auto_indent(to_insert, buffer);
    } else {
        cursor.insert_normal_text(to_insert, buffer)
    }
}

fn _decrease_indent(indent: String) -> String {
    if let Some(stripped) = indent.strip_prefix("    ") {
        stripped.to_string()
    } else {
        indent
    }
}

fn increase_indent(indent: String) -> String {
    indent + "    "
}

fn last_non_whitespace_character(line: &String) -> Option<(usize, u8)> {
    // last non_whitespace
    for (index, byte) in line.as_bytes().iter().enumerate().rev() {
        if !byte.is_ascii_whitespace() {
            return Some((index, *byte));
        }
    }
    None
}

fn get_indent(last_line: &String) -> String {
    let mut indent = String::new();
    for byte in last_line.as_bytes() {
        if byte.is_ascii_whitespace() {
            indent.push(*byte as char);
        } else {
            break;
        }
    }
    indent
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SimpleCursor {
    line: usize,
    column: usize,
    selection: Option<((usize, usize), (usize, usize))>,
}

impl VirtualCursor for SimpleCursor {
    fn new(line: usize, column: usize) -> Self {
        Self {
            line,
            column,
            selection: None,
        }
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

    fn selection(&self) -> Option<((usize, usize), (usize, usize))> {
        self.selection
    }

    fn set_selection(&mut self, selection: Option<((usize, usize), (usize, usize))>) {
        self.selection = selection;
    }
}

// For right now it is a simple linear history
// Probably want it to be a tree
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
struct CursorWithHistory {
    cursor: SimpleCursor,
    history: Vec<SimpleCursor>,
}

impl VirtualCursor for CursorWithHistory {
    fn new(line: usize, column: usize) -> Self {
        Self {
            cursor: SimpleCursor::new(line, column),
            history: Vec::new(),
        }
    }

    fn line(&self) -> usize {
        self.cursor.line()
    }

    fn column(&self) -> usize {
        self.cursor.column()
    }

    fn selection(&self) -> Option<((usize, usize), (usize, usize))> {
        self.cursor.selection()
    }

    fn set_selection(&mut self, selection: Option<((usize, usize), (usize, usize))>) {
        self.cursor.set_selection(selection);
    }

    fn move_to(&mut self, line: usize, column: usize) {
        self.history.push(self.cursor);
        self.cursor.move_to(line, column);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MultiCursor<C: VirtualCursor> {
    cursors: Vec<C>,
}

impl Default for MultiCursor<SimpleCursor> {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiCursor<SimpleCursor> {
    pub fn new() -> Self {
        Self { cursors: vec![] }
    }

    pub fn add_cursor(&mut self, cursor: SimpleCursor) {
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

    // TODO: These don't really make sense
    fn line(&self) -> usize {
        self.cursors.get(0).map(C::line).unwrap_or(0)
    }

    fn column(&self) -> usize {
        self.cursors.get(0).map(C::column).unwrap_or(0)
    }

    fn selection(&self) -> Option<((usize, usize), (usize, usize))> {
        self.cursors.get(0).and_then(C::selection)
    }

    fn set_selection(&mut self, selection: Option<((usize, usize), (usize, usize))>) {
        for cursor in &mut self.cursors {
            cursor.set_selection(selection);
        }
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

    fn above<T: TextBuffer>(&self, buffer: &T) -> Self {
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

    fn delete_char<T: TextBuffer<Item = u8>>(&mut self, buffer: &mut T) {
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

    fn line_start(&self, line: usize) -> usize {
        line * self.line_length(line)
    }

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

    fn get_line(&self, index: usize) -> Option<&[Self::Item]> {
        self.lines().nth(index)
    }

    fn set_contents(&mut self, contents: &[Self::Item]) {
        self.bytes = contents.to_vec();
    }

    fn contents(&self) -> &[Self::Item] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn adding_and_remove_basic() {
        let mut cursor = SimpleCursor::new(0, 0);
        let mut buffer = EventTextBuffer::new();
        let my_string = b"Hello World";
        cursor.handle_insert(my_string, &mut buffer);
        for (i, item) in my_string.iter().enumerate() {
            assert_eq!(buffer.byte_at_pos(0, i), Some(item));
        }
        for _ in 0..my_string.len() {
            cursor.delete_char(&mut buffer);
        }

        assert!(buffer.text_positions.is_empty());
    }

    #[test]
    fn adding_and_remove_basic_multi() {
        let cursor = SimpleCursor::new(0, 0);
        let mut buffer = EventTextBuffer::new();
        let cursor_below = cursor.below(&buffer);
        let my_string = b"Hello World";

        let mut multi_cursor = MultiCursor::new();
        multi_cursor.add_cursor(cursor);
        multi_cursor.add_cursor(cursor_below);

        multi_cursor.handle_insert(my_string, &mut buffer);

        for cursor in multi_cursor.cursors.iter() {
            for (i, item) in my_string.iter().enumerate() {
                assert_eq!(buffer.byte_at_pos(cursor.line(), i), Some(item));
            }
        }

        for _ in 0..my_string.len() {
            multi_cursor.delete_char(&mut buffer);
        }

        assert!(buffer.text_positions.is_empty());
    }

    #[test]
    fn test_lines() {
        let mut cursor = SimpleCursor::new(0, 0);
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

// TODO:
// I need auto indent
