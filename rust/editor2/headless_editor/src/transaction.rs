use serde::{Deserialize, Serialize};

use crate::{TextBuffer, VirtualCursor};


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Transaction<Cursor: VirtualCursor> {
    pub transaction_number: usize,
    pub parent_pointer: usize,
    pub action: EditAction<Cursor>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TransactionManager<Cursor: VirtualCursor> {
    pub transactions: Vec<Transaction<Cursor>>,
    pub current_transaction: usize,
    pub transaction_pointer: usize,
}

// I think I want to control when transactions start and end
// for the most part. I am sure there are cases where I will
// need to let the caller decide.
// I will have to think more about how to make those two work.
impl<Cursor: VirtualCursor> TransactionManager<Cursor> {
    pub fn new() -> TransactionManager<Cursor> {
        TransactionManager {
            transactions: Vec::new(),
            current_transaction: 1,
            transaction_pointer: 0,
        }
    }

    pub fn get_last_action_ignoring_cursor(&self) -> Option<&EditAction<Cursor>> {
        let mut transaction_pointer = self.transaction_pointer;
        let current_transaction_number = self.current_transaction;
        loop {
            let last_transaction = self.transactions.get(transaction_pointer)?;
            if last_transaction.transaction_number != current_transaction_number {
                return None
            }
            if !matches!(last_transaction.action, EditAction::CursorPosition(_)) {
                return Some(&last_transaction.action);
            }
            if transaction_pointer == last_transaction.parent_pointer {
                return None
            }
            transaction_pointer = last_transaction.parent_pointer;
        }
    }

    pub fn add_action(&mut self, action: EditAction<Cursor>) {

        match &action {
            EditAction::Noop => {},
            EditAction::InsertWithCursor(_, _, _) => {
                let (a, b) = action.split_insert_and_cursor();
                self.add_action(a);
                self.add_action(b);
                return;
            }
            EditAction::Insert(_, s) => {
                if s.iter().all(|x| x.is_ascii_whitespace()) {
                    self.current_transaction += 1;
                }
            }
            EditAction::Delete(_, _, _) => {
                match self.get_last_action_ignoring_cursor() {
                    None => {},
                    Some(EditAction::Delete(_, _, _)) => {},
                    Some(_) => {
                        self.current_transaction += 1;
                    }
                }
            }
            EditAction::CursorPosition(_) => {}
        }

        self.transactions.push(Transaction {
            transaction_number: self.current_transaction,
            parent_pointer: self.transaction_pointer,
            action,
        });

        self.transaction_pointer = self.transactions.len() - 1;
    }

    pub fn undo<Buffer: TextBuffer<Item = u8>>(&mut self, cursor: &mut Cursor, text_buffer: &mut Buffer) {
        if self.transaction_pointer == 0 {
           return;
        }
        let last_transaction = self.transactions[self.transaction_pointer].transaction_number;
        let mut i = self.transaction_pointer;
        while self.transactions[i].transaction_number == last_transaction {
            self.transactions[i].action.undo(cursor, text_buffer);

            if i == 0 {
                break;
            }
            i = self.transactions[i].parent_pointer;
        }
        self.transaction_pointer = i;
        self.current_transaction += 1;

    }


    pub fn redo<Buffer: TextBuffer<Item = u8>>(&mut self, cursor: &mut Cursor, text_buffer: &mut Buffer) {

        if self.transaction_pointer == self.transactions.len() - 1 {
            return;
        }

        let last_undo = self.transactions.iter()
            .rev()
            .find(|t| t.parent_pointer == self.transaction_pointer);

        if let Some(Transaction{ transaction_number: last_transaction, ..}) = last_undo {
            for (i, transaction) in self.transactions.iter().enumerate() {
                if transaction.transaction_number == *last_transaction {
                    self.transactions[i].action.redo(cursor, text_buffer);
                    self.transaction_pointer = i;
                }
                if transaction.transaction_number > *last_transaction {
                    break;
                }
            }
        }
        self.current_transaction += 1;
    }

    pub fn next_transaction(&mut self) {
        self.current_transaction += 1;
    }

}


#[derive(Serialize, Deserialize, Debug, Clone,)]
pub enum EditAction<Cursor: VirtualCursor>{
    Insert((usize, usize), Vec<u8>),
    Delete((usize, usize), (usize, usize), Vec<u8>),
    // These only get recorded as part of these other actions.
    // They would be in the same transaction as other actions
    CursorPosition(Cursor),
    InsertWithCursor((usize, usize), Vec<u8>, Cursor),
    Noop,
}

impl<Cursor: VirtualCursor> EditAction<Cursor>  {
    pub fn undo<Buffer: TextBuffer<Item = u8>>(&self, cursor: &mut Cursor, text_buffer: &mut Buffer) {
        match self {
            EditAction::Insert((line, column), text_to_insert) => {
                let new_position = Cursor::new(*line, *column);
                // TODO: Make faster
                // for _ in 0..text_to_insert.len() {
                //     new_position.move_right(text_buffer);
                // }
                for _ in 0..text_to_insert.len() {
                    text_buffer.delete_char(new_position.line(), new_position.column());
                }
                cursor.move_to(new_position.line(), new_position.column());
            },
            EditAction::Delete(start, _end, text_to_delete) => {
                cursor.move_to(start.0, start.1);
                cursor.move_left(text_buffer);
                cursor.insert_normal_text(text_to_delete, text_buffer);
            },
            EditAction::CursorPosition(old_cursor) => {
                cursor.move_to(old_cursor.line(), old_cursor.column());
            }
            EditAction::InsertWithCursor(location, text_to_insert, cursor ) => {
                EditAction::Insert(*location, text_to_insert.clone()).undo(&mut cursor.clone(), text_buffer);
                EditAction::CursorPosition(cursor.clone()).undo(&mut cursor.clone(), text_buffer);
            }
            EditAction::Noop => {}
        }
    }

    pub fn redo<Buffer: TextBuffer<Item = u8>>(&self, cursor: &mut Cursor, text_buffer: &mut Buffer) -> Option<()> {

        match self {
            EditAction::Insert((line, column), text_to_insert) => {
                let new_position = Cursor::new(*line, *column);
                cursor.move_to(new_position.line(), new_position.column());
                cursor.handle_insert(text_to_insert, text_buffer);
            },
            EditAction::Delete(start, end, _text_to_delete) => {
                cursor.move_to(start.0, start.1);
                cursor.delete_chars(text_buffer, *start, *end);
                cursor.move_to(start.0, start.1);
            },
            EditAction::CursorPosition(new_cursor) => {
                cursor.move_to(new_cursor.line(), new_cursor.column());
            }
            EditAction::InsertWithCursor(location,text_to_insert, cursor ) => {
                EditAction::Insert(*location, text_to_insert.clone()).redo(&mut cursor.clone(), text_buffer);
                EditAction::CursorPosition(cursor.clone()).redo(&mut cursor.clone(), text_buffer);
            }
            EditAction::Noop => {}
        }
        Some(())
    }

    pub fn combine_insert_and_cursor(self, cursor_action: EditAction<Cursor>) -> EditAction<Cursor> {
        match (self, cursor_action) {
            (EditAction::Insert(location, text_to_insert), EditAction::CursorPosition(cursor)) => {
                EditAction::InsertWithCursor(location, text_to_insert, cursor)
            }
            x => panic!("Can't combine these actions {:?}", x)
        }
    }

    pub fn split_insert_and_cursor(&self) -> (EditAction<Cursor>, EditAction<Cursor>) {
        match self {
            EditAction::InsertWithCursor(location, text_to_insert, cursor) => {
                (EditAction::Insert(*location, text_to_insert.clone()), EditAction::CursorPosition(cursor.clone()))
            }
            x => panic!("Can't split these actions {:?}", x)
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TransactingVirtualCursor<Cursor: VirtualCursor> {
    cursor: Cursor,
    transaction_manager: TransactionManager<Cursor>,
}

impl<Cursor: VirtualCursor> VirtualCursor for TransactingVirtualCursor<Cursor> {
    fn move_to(&mut self, line: usize, column: usize) {
        self.cursor.move_to(line, column);
        self.transaction_manager.add_action(EditAction::CursorPosition(self.cursor.clone()));
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

    // Kind of ugly reimplementation
    fn delete<T: TextBuffer<Item=u8>>(&mut self, buffer: &mut T) {
        if self.selection().is_some() {
            self.delete_selection(buffer);
        } else {
            self.delete_char(buffer);
        }
    }

    fn delete_chars<T: TextBuffer<Item=u8>>(&mut self, buffer: &mut T, start: (usize, usize), end: (usize, usize)) { 
        self.move_to(end.0, end.1);
        while self.line() != start.0 || self.column() != start.1 {
            self.delete_char(buffer);
        }
    }

    fn delete_selection<T: TextBuffer<Item=u8>>(&mut self, buffer: &mut T) {
        if let Some(((start_line, start_column), (end_line, end_column))) = self.selection() {
            self.delete_chars(buffer, (start_line, start_column), (end_line, end_column));
        }
        self.set_selection(None);
    }

    fn delete_char<T: TextBuffer<Item = u8>>(&mut self, buffer: &mut T) {
        let mut cursor = self.cursor.clone();
        cursor.move_left(buffer);
        if let Some(current_text) = buffer.byte_at_pos(cursor.line(), cursor.column()) {
            self.transaction_manager.add_action(EditAction::Delete((self.line(), self.column()), (self.line(), self.column() + 1), vec![*current_text]));
            self.cursor.delete_char(buffer);
        } else {
            panic!("Can't delete char, no character found at position");
        }
    }
    
    // TODO: My original implementation has a lot more going on. Need to test this out
    // and remember why I had all that stuff
    fn insert_normal_text<T: TextBuffer<Item = u8>>(&mut self, to_insert: &[u8], buffer: &mut T) {
        self.transaction_manager.add_action(EditAction::Insert((self.line(), self.column()), to_insert.to_vec()));
        self.cursor.insert_normal_text(to_insert, buffer);
    }

    fn set_selection(&mut self, selection: Option<((usize, usize), (usize, usize))>) {
        self.cursor.set_selection(selection);
    }

    fn new(line: usize, column: usize) -> Self {
        TransactingVirtualCursor { 
            cursor: Cursor::new(line, column),
            transaction_manager: TransactionManager::new() 
        }
    }
}


impl<Cursor: VirtualCursor> TransactingVirtualCursor<Cursor> {
    pub fn undo<T: TextBuffer<Item = u8>>(&mut self, text_buffer: &mut T) {
        self.transaction_manager.undo(&mut self.cursor, text_buffer);
    }
    pub fn redo<T: TextBuffer<Item = u8>>(&mut self, text_buffer: &mut T) {
        self.transaction_manager.redo(&mut self.cursor, text_buffer);
    }

    pub fn get_transactions(&self) -> &Vec<Transaction<Cursor>> {
        &self.transaction_manager.transactions
    }
}