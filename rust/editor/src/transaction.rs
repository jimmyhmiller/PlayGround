use crate::{CursorContext, text_buffer::TextBuffer, Cursor};


#[derive(Debug, Clone)]
pub struct Transaction {
    pub transaction_number: usize,
    pub parent_pointer: usize,
    pub action: EditAction,
}

#[derive(Debug, Clone)]
pub struct TransactionManager {
    pub transactions: Vec<Transaction>,
    pub current_transaction: usize,
    pub transaction_pointer: usize,
}

// I think I want to control when transactions start and end
// for the most part. I am sure there are cases where I will
// need to let the caller decide.
// I will have to think more about how to make those two work.
impl TransactionManager {
    pub fn new() -> TransactionManager {
        TransactionManager {
            transactions: Vec::new(),
            current_transaction: 1,
            transaction_pointer: 0,
        }
    }

    pub fn add_action(&mut self, action: EditAction) {

        match &action {
            EditAction::Noop => {},
            EditAction::InsertWithCursor(_, _, _) => {
                let (a, b) = action.split_insert_and_cursor();
                self.add_action(a);
                self.add_action(b);
                return;
            }
            EditAction::Insert(_, s) => {
                if s.trim().is_empty() {
                    self.current_transaction += 1;
                }
            }
            EditAction::Delete(_, _) => {
                // Delete isn't quite right. I kind of want strings
                // of deletes to coalesce.
                // Maybe I should have some compact functions?
                self.current_transaction += 1;
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

    pub fn undo(&mut self, cursor_context: &mut CursorContext, text_buffer: &mut TextBuffer) {
        if self.transaction_pointer == 0 {
           return;
        }
        let last_transaction = self.transactions[self.transaction_pointer].transaction_number;
        let mut i = self.transaction_pointer;
        while self.transactions[i].transaction_number == last_transaction {
            self.transactions[i].action.undo(cursor_context, text_buffer);

            if i == 0 {
                break;
            }
            i = self.transactions[i].parent_pointer;
        }
        self.transaction_pointer = i;
        self.current_transaction += 1;

    }

    // How do I redo?

    pub fn redo(&mut self, cursor_context: &mut CursorContext, text_buffer: &mut TextBuffer) {

        if self.transaction_pointer == self.transactions.len() - 1 {
            return;
        }

        let last_undo = self.transactions.iter()
            .rev()
            .find(|t| t.parent_pointer == self.transaction_pointer);

        if let Some(Transaction{ transaction_number: last_transaction, ..}) = last_undo {
            for (i, transaction) in self.transactions.iter().enumerate() {
                if transaction.transaction_number == *last_transaction {
                    self.transactions[i].action.redo(cursor_context, text_buffer);
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


#[derive(Debug, Clone)]
pub enum EditAction {
    Insert((usize, usize), String),
    Delete((usize,usize), String),
    // These only get recorded as part of these other actions.
    // They would be in the same transaction as other actions
    CursorPosition(Cursor),
    InsertWithCursor((usize, usize), String, Cursor),
    Noop,
}

impl EditAction  {
    pub fn undo(&self, cursor_context: &mut CursorContext, text_buffer: &mut TextBuffer) {
        match self {
            EditAction::Insert((line, column), text_to_insert) => {
                let mut new_position = Cursor(*line, *column);
                // TODO: Make faster
                for _ in 0..text_to_insert.len() {
                    new_position.move_right(text_buffer);
                }
                for _ in 0..text_to_insert.len() {
                    text_buffer.remove_char(new_position);
                    new_position.move_left(text_buffer);
                }
                cursor_context.set_cursor(new_position);
            },
            EditAction::Delete((line, column), text_to_delete) => {
                let mut new_position = Cursor(*line, *column);
                new_position.move_left(text_buffer);
                // I have a panic here
                text_buffer.insert_char(new_position, text_to_delete.as_bytes());
                new_position.move_right(text_buffer);
                cursor_context.set_cursor(new_position);
            },
            EditAction::CursorPosition(old_cursor) => {
                cursor_context.set_cursor(*old_cursor);
            }
            EditAction::InsertWithCursor(location, text_to_insert, cursor ) => {
                EditAction::Insert(*location, text_to_insert.clone()).undo(cursor_context, text_buffer);
                EditAction::CursorPosition(*cursor).undo(cursor_context, text_buffer);
            }
            EditAction::Noop => {}
        }
    }

    pub fn redo(&self, cursor_context: &mut CursorContext, text_buffer: &mut TextBuffer) {

        // TODO:
        // Are these start and end?
        // Or are they line and column?
        // We call them start and end here,
        // But insert_char takes a cursor, which is line and column.
        match self {
            EditAction::Insert((start, end), text_to_insert) => {
                // TODO: We can panic here if we paste do some undo and redo stuff.
                text_buffer.insert_char(Cursor(*start, *end), text_to_insert.as_bytes());
            },
            EditAction::Delete((start, end), _text_to_delete) => {
                text_buffer.remove_char(Cursor(*start, *end));
            },
            EditAction::CursorPosition(new_cursor) => {
                cursor_context.set_cursor(*new_cursor);
            }
            EditAction::InsertWithCursor(location,text_to_insert, cursor ) => {
                EditAction::Insert(*location, text_to_insert.clone()).redo(cursor_context, text_buffer);
                EditAction::CursorPosition(*cursor).redo(cursor_context, text_buffer);
            }
            EditAction::Noop => {}
        }
    }

    pub fn combine_insert_and_cursor(self, cursor_action: EditAction) -> EditAction {
        match (self, cursor_action) {
            (EditAction::Insert(location, text_to_insert), EditAction::CursorPosition(cursor)) => {
                EditAction::InsertWithCursor(location, text_to_insert, cursor)
            }
            x => panic!("Can't combine these actions {:?}", x)
        }
    }

    pub fn split_insert_and_cursor(&self) -> (EditAction, EditAction) {
        match self {
            EditAction::InsertWithCursor(location, text_to_insert, cursor) => {
                (EditAction::Insert(*location, text_to_insert.clone()), EditAction::CursorPosition(*cursor))
            }
            x => panic!("Can't split these actions {:?}", x)
        }
    }
}
