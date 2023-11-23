use std::future;
use std::str::from_utf8;

use quickcheck::quickcheck;
use quickcheck::Arbitrary;
use rand::{
    distributions::{Alphanumeric, Distribution, Standard},
    Rng,
};
use serde::{Deserialize, Serialize};
use standard_dist::StandardDist;

use crate::delete_char;
use crate::handle_insert;
use crate::insert_normal_text;
use crate::TokenTextBuffer;
use crate::{SimpleCursor, SimpleTextBuffer, TextBuffer, VirtualCursor};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Transaction<Cursor: VirtualCursor> {
    pub transaction_number: usize,
    pub parent_pointer: Option<usize>,
    pub action: EditAction<Cursor>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct TransactionManager<Cursor: VirtualCursor> {
    pub transactions: Vec<Transaction<Cursor>>,
    pub current_transaction: usize,
    pub last_undo_pointer: Option<usize>,
    pub transaction_pointer: Option<usize>,
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
            transaction_pointer: None,
            last_undo_pointer: None,
        }
    }

    pub fn get_last_action_ignoring_cursor(&self) -> Option<&EditAction<Cursor>> {
        if let Some(transaction_pointer) = self.transaction_pointer {
            let mut transaction_pointer = transaction_pointer;
            let current_transaction_number = self.current_transaction;
            loop {
                let last_transaction = self.transactions.get(transaction_pointer)?;
                if last_transaction.transaction_number != current_transaction_number {
                    return None;
                }
                if !matches!(last_transaction.action, EditAction::CursorPosition(_)) {
                    return Some(&last_transaction.action);
                }
                if Some(transaction_pointer) == last_transaction.parent_pointer {
                    return None;
                }
                if let Some(parent_pointer) = last_transaction.parent_pointer {
                    transaction_pointer = parent_pointer;
                } else {
                    return None;
                }
            }
        }
        None
    }

    pub fn add_action(&mut self, action: EditAction<Cursor>) {
        match &action {
            EditAction::Noop => {}
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
            EditAction::Delete(_, _, _) => match self.get_last_action_ignoring_cursor() {
                None => {}
                Some(EditAction::Delete(_, _, _)) => {}
                Some(_) => {
                    self.current_transaction += 1;
                }
            },
            EditAction::CursorPosition(_) => {}
        }

        self.transactions.push(Transaction {
            transaction_number: self.current_transaction,
            parent_pointer: self.transaction_pointer,
            action,
        });

        self.transaction_pointer = Some(self.transactions.len() - 1);
    }

    // TODO: Rewrite this code so I understand it
    // pub fn undo<Buffer: TextBuffer<Item = u8>>(&mut self, cursor: &mut Cursor, text_buffer: &mut Buffer) {
    //     if let Some(transaction_pointer) = self.transaction_pointer {
    //         if let Some(last_transaction) = self.transactions.get(transaction_pointer).map(|x| x.transaction_number) {
    //             let mut i = transaction_pointer;
    //             let mut next_pointer = None;
    //             while self.transactions[i].transaction_number == last_transaction {
    //                 self.transactions[i].action.undo(cursor, text_buffer);
    //                 self.last_undo_pointer = Some(i);

    //                 next_pointer = self.transactions[i].parent_pointer;
    //                 if let Some(next_pointer) = next_pointer {
    //                     i = next_pointer;
    //                 } else {
    //                     break;
    //                 }
    //             }
    //             self.transaction_pointer = next_pointer;
    //             self.current_transaction += 1;
    //         }
    //     }
    // }

    pub fn undo<Buffer: TextBuffer<Item = u8>>(
        &mut self,
        cursor: &mut Cursor,
        text_buffer: &mut Buffer,
    ) {
        println!("undo {:?}", self.transaction_pointer);
        if let Some(transaction_pointer) = self.transaction_pointer {
            if let Some(last_transaction) = self.transactions.get(transaction_pointer) {
                let transaction_number = last_transaction.transaction_number;

                let mut transaction = last_transaction;
                let mut transaction_pointer = Some(transaction_pointer);

                loop {
                    transaction.action.undo(cursor, text_buffer);
                    println!("Undoing {:?}", transaction.action);
                    println!("Result {:?}", from_utf8(text_buffer.contents()));
                    println!("Cursor {:?}", cursor);
                    self.last_undo_pointer = transaction_pointer;

                    let last_transaction = transaction;
                    if let Some(parent_pointer) = last_transaction.parent_pointer {
                        transaction_pointer = Some(parent_pointer);
                        transaction = self.transactions.get(parent_pointer).unwrap();
                    } else {
                        self.transaction_pointer = None;
                        break;
                    }

                    if transaction.transaction_number != transaction_number {
                        self.transaction_pointer = transaction_pointer;
                        break;
                    }
                    
                }

            }
        }
    }

    // I need to start from the last undo I think
    pub fn redo<Buffer: TextBuffer<Item = u8>>(
        &mut self,
        cursor: &mut Cursor,
        text_buffer: &mut Buffer,
    ) {
        // if self.last_undo_pointer.is_none() {
        //     return;
        // }
        if let Some(last_undo_pointer) = self.last_undo_pointer {
            if let Some(Transaction {
                transaction_number: last_transaction,
                parent_pointer,
                ..
            }) = self.transactions.get(last_undo_pointer)
            {
                self.last_undo_pointer = *parent_pointer;
                for (i, transaction) in self.transactions.iter().enumerate() {
                    if transaction.transaction_number == *last_transaction {
                        self.transactions[i].action.redo(cursor, text_buffer);
                        println!("Redoing {:?}", self.transactions[i].action);
                        println!("Result {:?}", from_utf8(text_buffer.contents()));
                        println!("Cursor {:?}", cursor);
                        self.transaction_pointer = self.transactions[i].parent_pointer;
                    }
                    if transaction.transaction_number > *last_transaction {
                        break;
                    }
                }
            }
        }

        self.current_transaction += 1;
    }

    pub fn next_transaction(&mut self) {
        self.current_transaction += 1;
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum EditAction<Cursor: VirtualCursor> {
    Insert((usize, usize), Vec<u8>),
    Delete((usize, usize), (usize, usize), Vec<u8>),
    // These only get recorded as part of these other actions.
    // They would be in the same transaction as other actions
    CursorPosition(Cursor),
    InsertWithCursor((usize, usize), Vec<u8>, Cursor),
    Noop,
}

impl<Cursor: VirtualCursor> EditAction<Cursor> {
    // Everything is broken. And I'm adding cursor movement too much
    // I need to figure out how to do this better
    pub fn undo<Buffer: TextBuffer<Item = u8>>(
        &self,
        cursor: &mut Cursor,
        text_buffer: &mut Buffer,
    ) {
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
            }
            EditAction::Delete(start, _end, text_to_delete) => {
                cursor.move_to(start.0, start.1);
                // cursor.move_left(text_buffer);
                cursor.insert_normal_text(text_to_delete, text_buffer);
            }
            EditAction::CursorPosition(old_cursor) => {
                cursor.move_to(old_cursor.line(), old_cursor.column());
            }
            EditAction::InsertWithCursor(location, text_to_insert, cursor) => {
                EditAction::Insert(*location, text_to_insert.clone())
                    .undo(&mut cursor.clone(), text_buffer);
                EditAction::CursorPosition(cursor.clone()).undo(&mut cursor.clone(), text_buffer);
            }
            EditAction::Noop => {}
        }
    }

    pub fn redo<Buffer: TextBuffer<Item = u8>>(
        &self,
        cursor: &mut Cursor,
        text_buffer: &mut Buffer,
    ) -> Option<()> {
        match self {
            EditAction::Insert((line, column), text_to_insert) => {
                let new_position = Cursor::new(*line, *column);
                cursor.move_to(new_position.line(), new_position.column());
                cursor.handle_insert(text_to_insert, text_buffer);
            }
            EditAction::Delete(start, end, _text_to_delete) => {
                cursor.move_to(start.0, start.1);
                cursor.delete_chars(text_buffer, *start, *end);
                cursor.move_to(start.0, start.1);
            }
            EditAction::CursorPosition(new_cursor) => {
                cursor.move_to(new_cursor.line(), new_cursor.column());
            }
            EditAction::InsertWithCursor(location, text_to_insert, cursor) => {
                EditAction::Insert(*location, text_to_insert.clone())
                    .redo(&mut cursor.clone(), text_buffer);
                EditAction::CursorPosition(cursor.clone()).redo(&mut cursor.clone(), text_buffer);
            }
            EditAction::Noop => {}
        }
        Some(())
    }

    pub fn combine_insert_and_cursor(
        self,
        cursor_action: EditAction<Cursor>,
    ) -> EditAction<Cursor> {
        match (self, cursor_action) {
            (EditAction::Insert(location, text_to_insert), EditAction::CursorPosition(cursor)) => {
                EditAction::InsertWithCursor(location, text_to_insert, cursor)
            }
            x => panic!("Can't combine these actions {:?}", x),
        }
    }

    pub fn split_insert_and_cursor(&self) -> (EditAction<Cursor>, EditAction<Cursor>) {
        match self {
            EditAction::InsertWithCursor(location, text_to_insert, cursor) => (
                EditAction::Insert(*location, text_to_insert.clone()),
                EditAction::CursorPosition(cursor.clone()),
            ),
            x => panic!("Can't split these actions {:?}", x),
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
        self.transaction_manager
            .add_action(EditAction::CursorPosition(self.cursor.clone()));
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
    fn delete<T: TextBuffer<Item = u8>>(&mut self, buffer: &mut T) {
        if self.selection().is_some() {
            self.delete_selection(buffer);
        } else {
            self.delete_char(buffer);
        }
    }

    fn delete_chars<T: TextBuffer<Item = u8>>(
        &mut self,
        buffer: &mut T,
        start: (usize, usize),
        end: (usize, usize),
    ) {
        self.move_to(end.0, end.1);
        while self.line() != start.0 || self.column() != start.1 {
            self.delete_char(buffer);
        }
    }

    fn delete_selection<T: TextBuffer<Item = u8>>(&mut self, buffer: &mut T) {
        if let Some(((start_line, start_column), (end_line, end_column))) = self.selection() {
            self.delete_chars(buffer, (start_line, start_column), (end_line, end_column));
        }
        self.set_selection(None);
    }

    fn delete_char<T: TextBuffer<Item = u8>>(&mut self, buffer: &mut T) {
        let mut cursor = self.cursor.clone();
        if cursor.line() == 0 && cursor.column() == 0 && cursor.selection().is_none() {
            return;
        }
        cursor.move_left(buffer);
        if let Some(current_text) = buffer.byte_at_pos(cursor.line(), cursor.column()) {
            self.transaction_manager.add_action(EditAction::Delete(
                (cursor.line(), cursor.column()),
                (cursor.line(), cursor.column() + 1),
                vec![*current_text],
            ));
            delete_char(self, buffer);
        } else {
            // TODO: Need to handle errors better
            // panic!("Can't delete char, no character found at position");
        }
    }

    // TODO: My original implementation has a lot more going on. Need to test this out
    // and remember why I had all that stuff
    fn insert_normal_text<T: TextBuffer<Item = u8>>(&mut self, to_insert: &[u8], buffer: &mut T) {
        self.transaction_manager.add_action(EditAction::Insert(
            (self.line(), self.column()),
            to_insert.to_vec(),
        ));
        println!("Inserting {:?}", from_utf8(to_insert));
        insert_normal_text(self, to_insert, buffer);
    }

    fn handle_insert<T: TextBuffer<Item = u8>>(&mut self, to_insert: &[u8], buffer: &mut T) {
        println!("Calling handle insert");
        handle_insert(self, to_insert, buffer)
    }

    fn set_selection(&mut self, selection: Option<((usize, usize), (usize, usize))>) {
        self.cursor.set_selection(selection);
    }

    fn new(line: usize, column: usize) -> Self {
        TransactingVirtualCursor {
            cursor: Cursor::new(line, column),
            transaction_manager: TransactionManager::new(),
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

    pub fn get_transaction_manager(&self) -> &TransactionManager<Cursor> {
        &self.transaction_manager
    }
}

#[derive(Debug, Clone)]
struct RandomString {
    string: String,
}

impl Distribution<RandomString> for Standard {
    fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> RandomString {
        let random_length = rng.gen_range(0..100);

        let random_string: String = (0..random_length)
            .map(|_| rng.sample(Alphanumeric))
            .map(char::from)
            .collect();

        RandomString {
            string: random_string,
        }
    }
}

// impl Distribution for RandomString {
//     fn sample<R: rand::prelude::Rng + ?Sized>(&self, rng: &mut R) -> T {
//         let random_length = rng.gen_range(0..100);
//         let random_string: String = rng
//             .sample_iter(&Alphanumeric)
//             .take(random_length)
//             .map(char::from)
//             .collect();
//         RandomString {
//             length: random_length,
//             string: random_string,
//         }
//     }
// }

#[derive(StandardDist, Debug, Clone)]
enum Actions {
    #[weight(10)]
    Delete,
    #[weight(1000)]
    Insert(RandomString),
    #[weight(3)]
    Move(MoveAction),
    Select((usize, usize), (usize, usize)),
    InsertSpace,
    InsertNewLine,
}

impl Arbitrary for Actions {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let mut action = rand::random::<Actions>();
        match &mut action {
            Actions::Insert(string) => {
                let mut rng = rand::thread_rng();
                let size = g.size();
                let random_string: String = (0..size)
                    .map(|_| rng.sample(Alphanumeric))
                    .map(char::from)
                    .collect();
                *string = RandomString {
                    string: random_string,
                };
            }
            _ => {}
        }
        action
    }
}

#[derive(StandardDist, Debug, Clone)]
enum MoveAction {
    Left,
    Right,
    Up,
    Down,
    Location(usize, usize),
}

fn interpret_action<T: TextBuffer<Item = u8>>(
    action: &Actions,
    cursor: &mut TransactingVirtualCursor<SimpleCursor>,
    text_buffer: &mut T,
) -> bool {
    match action {
        Actions::Delete => {
            cursor.delete(text_buffer);
            !(cursor.line() == 0 && cursor.column() == 0)
        }
        Actions::Insert(string) => {
            // random string
            cursor.insert_normal_text(string.string.as_bytes(), text_buffer);
            true
        }
        Actions::Move(move_action) => match move_action {
            MoveAction::Left => {
                cursor.move_left(text_buffer);
                false
            }
            MoveAction::Right => {
                cursor.move_right(text_buffer);
                false
            }
            MoveAction::Up => {
                cursor.move_up(text_buffer);
                false
            }
            MoveAction::Down => {
                cursor.move_down(text_buffer);
                false
            }
            MoveAction::Location(line, column) => {
                cursor.move_to_bounded(*line, *column, text_buffer);
                false
            }
        },
        Actions::Select(from, to) => {
            cursor.set_selection_bounded(Some((*from, *to)), text_buffer);
            false
        }
        Actions::InsertSpace => {
            cursor.insert_normal_text(&[b' '], text_buffer);
            true
        }
        Actions::InsertNewLine => {
            cursor.insert_normal_text(&[b'\n'], text_buffer);
            true
        }
    }
}

#[test]
fn example_test() {
    use Actions::*;
    use MoveAction::*;
    let initial_contents = "hello".as_bytes();
    let mut text_buffer = SimpleTextBuffer::new();
    text_buffer.set_contents(initial_contents);

    let mut cursor: TransactingVirtualCursor<SimpleCursor> = TransactingVirtualCursor::new(0, 0);

    let actions = vec![Move(Right), Delete];
    for action in actions.iter() {
        let changed = interpret_action(&action, &mut cursor, &mut text_buffer);
    }
    // for _ in 0..actions.len() {
    //     cursor.undo(&mut text_buffer);
    // }
    // for _ in 0..actions.len() {
    //     cursor.redo(&mut text_buffer);
    // }
    // for _ in 0..actions.len() {
    //     cursor.undo(&mut text_buffer);
    // }
    assert!(
        text_buffer.contents() == initial_contents,
        "Contents: {:?}",
        from_utf8(text_buffer.contents())
    );
}

quickcheck! {
    fn prop(actions: Vec<Actions>) -> bool {
        let initial_contents = "hello".as_bytes();
        let mut text_buffer = SimpleTextBuffer::new();
        text_buffer.set_contents(initial_contents);

        let mut cursor: TransactingVirtualCursor<SimpleCursor> = TransactingVirtualCursor::new(0, 0);

        for action in actions.iter() {
            let changed = interpret_action(&action, &mut cursor, &mut text_buffer);
            if changed {
               if text_buffer.contents() == initial_contents {
                   return false
               }
            }
        }
        for _ in 0..actions.len() {
            cursor.undo(&mut text_buffer);
        }
        text_buffer.contents() == initial_contents
    }
}

quickcheck! {
    fn prop2(actions: Vec<Actions>) -> bool {
        let initial_contents = "hello".as_bytes();
        let mut text_buffer = SimpleTextBuffer::new();
        text_buffer.set_contents(initial_contents);

        let mut cursor: TransactingVirtualCursor<SimpleCursor> = TransactingVirtualCursor::new(0, 0);

        for action in actions.iter() {
            println!("action {:?}", action);
            interpret_action(&action, &mut cursor, &mut text_buffer);
            // println!("Contents: {:?}", from_utf8(text_buffer.contents()));
        }
        
        // for _ in 0..actions.len() {
        //     cursor.undo(&mut text_buffer);
        // }

        // for _ in 0..actions.len() {
        //     cursor.redo(&mut text_buffer);
        // }
        // for _ in 0..actions.len() {
        //     cursor.undo(&mut text_buffer);
        // }
        text_buffer.contents() == initial_contents
    }
}

quickcheck! {
    fn prop3(actions: Vec<Actions>) -> bool {
        let initial_contents = "hello".as_bytes();
        let mut text_buffer = TokenTextBuffer::new_with_contents("hello".as_bytes());

        let mut cursor: TransactingVirtualCursor<SimpleCursor> = TransactingVirtualCursor::new(0, 0);

        for action in actions.iter() {
            interpret_action(&action, &mut cursor, &mut text_buffer);
            println!("Contents: {:?}", from_utf8(text_buffer.contents()));
        }
        for _ in 0..actions.len() {
            cursor.undo(&mut text_buffer);
        }
        text_buffer.contents() == initial_contents
    }
}

quickcheck! {
    fn prop4(actions: Vec<Actions>) -> bool {
        let initial_contents = "hello".as_bytes();
        let mut text_buffer = TokenTextBuffer::new_with_contents("hello".as_bytes());

        let mut cursor: TransactingVirtualCursor<SimpleCursor> = TransactingVirtualCursor::new(0, 0);

        for action in actions.iter() {
            println!("action {:?}", action);
            interpret_action(&action, &mut cursor, &mut text_buffer);
            cursor.undo(&mut text_buffer);
        }
        text_buffer.contents() == initial_contents
    }
}
