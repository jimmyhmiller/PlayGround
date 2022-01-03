use std::{cmp::{max, min}, fs, ops::{Index, IndexMut, Neg, RangeBounds}, str::from_utf8, time::Instant, convert::TryInto, process::{Command, Stdio, ChildStdout, Child}};
use std::fmt::Debug;

use nonblock::NonBlockingReader;
use rand::Rng;
use sdl2::{event::{Event, WindowEvent}, keyboard::{Keycode, Mod, Scancode}, mouse::SystemCursor, pixels::{Color}, rect::Rect, render::{Canvas, Texture}, video::{self}, sys, clipboard::{ClipboardUtil}};
use tokenizer::{Tokenizer, rust_specific_pass, RustSpecific, Token};
use sdl2::gfx::primitives::DrawRenderer;


// cargo build --message-format=json | jq 'select(.reason == "compiler-message")' | jq '.message' | jq '.spans' | jq ".[]" | jq '"rect canvas \(.line_start) \(.column_start) \(.line_end) \(.column_end) 1"'  | tr -d '"'

mod native;
mod sdl;
mod tokenizer;

const BACKGROUND_COLOR: Color = Color::RGB(0x29, 0x2D, 0x3E);
const STANDARD_TEXT_COLOR: Color = Color::RGB(0x95, 0x9D, 0xCB);
const PURPLE_TEXT_COLOR: Color = Color::RGB(0xC7, 0x92, 0xEA);
const BLUE_TEXT_COLOR: Color = Color::RGB(0x82,0xAA,0xFF);
const LIGHT_BLUE_TEXT_COLOR: Color = Color::RGB(0x89,0xDD,0xFF);
const GREEN_TEXT_COLOR: Color = Color::RGB(0xC3,0xE8, 0x8D);
const COMMENT_TEXT_COLOR: Color = Color::RGB(0x67, 0x6E, 0x95);
const ORANGE_TEXT_COLOR: Color = Color::RGB(0xF7, 0x8C, 0x6C);
const CURSOR_COLOR: Color = Color::RGBA(255, 204, 0, 255);

// I really want so debugging panels.
// Should probably invest in that.
// Make it work automatically with Debug.

#[derive(Debug, Clone)]
struct Transaction {
    transaction_number: usize,
    parent_pointer: usize,
    action: EditAction,
}

#[derive(Debug, Clone)]
struct TransactionManager {
    transactions: Vec<Transaction>,
    current_transaction: usize,
    transaction_pointer: usize,
}

// I think I want to control when transactions start and end
// for the most part. I am sure there are cases where I will
// need to let the caller decide.
// I will have to think more about how to make those two work.
impl TransactionManager {
    fn new() -> TransactionManager {
        TransactionManager {
            transactions: Vec::new(),
            current_transaction: 1,
            transaction_pointer: 0,
        }
    }

    fn add_action(&mut self, action: EditAction) {

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

    fn undo(&mut self, cursor_context: &mut CursorContext, text_buffer: &mut TextBuffer) {
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

    fn redo(&mut self, cursor_context: &mut CursorContext, text_buffer: &mut TextBuffer) {

        if self.transaction_pointer == self.transactions.len() - 1 {
            return;
        }

        let last_undo = self.transactions.iter()
            .rev()
            .find(|t| t.parent_pointer == self.transaction_pointer);

        // My cursor is one off! But this seems to be close to correct for the small cases I tried.
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

}

#[derive(Debug)]
enum SideEffectAction {
    Play(String),
}

enum PerFrameAction {
    ReadCommand(String, Child, NonBlockingReader<ChildStdout>),
    DisplayError(String, String),
}


enum PerFrameActionResult {
    RemoveAction(usize),
    Noop,
}

#[derive(Debug, Clone)]
enum EditAction {
    Insert((usize, usize), String),
    Delete((usize,usize), String),
    // These only get recorded as part of these other actions.
    // They would be in the same transaction as other actions
    CursorPosition(Cursor),
    InsertWithCursor((usize, usize), String, Cursor),
    Noop,
}

impl EditAction  {
    fn undo(&self, cursor_context: &mut CursorContext, text_buffer: &mut TextBuffer) {
        match self {
            EditAction::Insert((start, end), text_to_insert) => {
                let mut new_position = Cursor(*start, *end);
                new_position.move_right(text_buffer);
                for _ in 0..text_to_insert.len() {
                    text_buffer.remove_char(new_position);
                    new_position.move_left(text_buffer);
                }
                cursor_context.set_cursor(new_position);
            },
            EditAction::Delete((start, end), text_to_delete) => {
                let mut new_position = Cursor(*start, *end);
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

    fn redo(&self, cursor_context: &mut CursorContext, text_buffer: &mut TextBuffer) {

        match self {
            EditAction::Insert((start, end), text_to_insert) => {
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

    fn combine_insert_and_cursor(self, cursor_action: EditAction) -> EditAction {
        match (self, cursor_action) {
            (EditAction::Insert(location, text_to_insert), EditAction::CursorPosition(cursor)) => {
                EditAction::InsertWithCursor(location, text_to_insert, cursor)
            }
            x => panic!("Can't combine these actions {:?}", x)
        }
    }

    fn split_insert_and_cursor(&self) -> (EditAction, EditAction) {
        match self {
            EditAction::InsertWithCursor(location, text_to_insert, cursor) => {
                (EditAction::Insert(*location, text_to_insert.clone()), EditAction::CursorPosition(*cursor))
            }
            x => panic!("Can't split these actions {:?}", x)
        }
    }
}



#[derive(Debug, Copy, Clone)]
struct Cursor(usize, usize);

impl Cursor {

    fn start_of_line(&mut self) {
        self.1 = 0;
    }

    fn move_up(&mut self, text_buffer: &TextBuffer) -> EditAction {
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

    fn move_down(&mut self, text_buffer: &TextBuffer) -> EditAction  {
        self.0 = min(self.0 + 1, text_buffer.line_count() -1);
        self.1 = min(self.1, text_buffer.line_length(self.0));
        EditAction::CursorPosition(*self)

        // Need to use this output to deal with scrolling down
    }


    fn move_left(&mut self, text_buffer: &TextBuffer) -> EditAction  {
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

    fn move_right(&mut self, text_buffer: &TextBuffer) -> EditAction  {
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


fn digit_count(x: usize) -> usize {
    let mut count = 0;
    let mut x = x;
    while x > 0 {
        x /= 10;
        count += 1;
    }
    count
}


#[derive(Debug, Clone)]
struct EditorBounds {
    editor_left_margin: usize,
    line_number_gutter_width: usize,
    letter_height: usize,
    letter_width: usize,
}

impl EditorBounds{

    fn line_number_digits(&self, text_buffer: &TextBuffer) -> usize {
        digit_count(text_buffer.line_count())
    }
    fn line_number_padding(&self, text_buffer: &TextBuffer) -> usize {
        self.line_number_digits(text_buffer) * self.letter_width as usize
            + self.line_number_gutter_width
            + self.editor_left_margin
            + self.letter_width as usize
    }
}


// This probably belongs in scroller
fn text_space_from_screen_space(scroller: &Scroller, mut x: usize, y: usize, text_buffer: &TextBuffer, bounds: &EditorBounds) -> Option<Cursor> {
    // Slightly off probably due to rounding.
    // println!("{}", y as f32 / letter_height as f32);

    // Probably should move some/all of this to the scroller.
    let EditorBounds {letter_height, letter_width, ..} = *bounds;
    let line_number_padding = bounds.line_number_padding(text_buffer) as i32;

    // Still crash here
    // Didn't realize it could do that without the unwrap
    let line_number : usize = (((y as f32 / letter_height as f32)
        + (scroller.line_fraction_y(bounds) as f32 / bounds.letter_height as f32)).floor() as i32
        + scroller.lines_above_fold(bounds) as i32) as usize;


    if (x as i32) < line_number_padding && (x as i32) > line_number_padding - 20  {
        x = line_number_padding as usize;
    }
    if x < line_number_padding as usize {
        return None;
    }
    let mut column_number : usize = (x - line_number_padding as usize) / letter_width as usize;

    if let Some((line_start, line_end)) = text_buffer.get_line(line_number) {
        if column_number > line_end - line_start {
            column_number = text_buffer[line_number].1 - text_buffer[line_number].0;
        }
        return Some(Cursor(line_number, column_number));
    }
    if line_number > text_buffer.line_count() && text_buffer.last_line().is_some() {
        return Some(Cursor(text_buffer.line_count() - 1, text_buffer.line_length(text_buffer.line_count() - 1)));
    }
    None
}


// TODO LIST:
// Add some spacing between letters!
// It would be pretty cool to add a minimap
// Need toggle line numbers
// Need references to panes
// Need canvas scrolling?
// Need to think about undo and pane positions
// I need copy and paste
// I need to experiment with panes being a surface
// I need to experiment with non-text panes
// I also need to think about the coordinate system
// It being upside down is annoying
// I need to think about afterburner text decorations
// I could have queries and panes as the results of those queries
// Need a command interface. But what to do it enso style
// Multi line bash commands

struct FpsCounter {
    start_time: Instant,
    frame_count: usize,
    fps: usize,
}

impl FpsCounter {
    fn reset(&mut self) {
        self.start_time = Instant::now();
        self.frame_count = 0;
    }

    fn tick(&mut self) -> usize {
        self.frame_count += 1;
        if self.start_time.elapsed().as_secs() >= 1 {
            self.fps = self.frame_count;
            self.reset();
        }
        self.fps
    }

}


#[derive(Debug, Clone, Copy)]
struct Window {
    width: i32,
    height: i32,
}

impl Window {
    fn resize(&mut self, width: i32, height: i32) {
        self.width = width;
        self.height = height;
    }
}

#[derive(Debug, Clone)]
struct CursorContext {
    cursor: Option<Cursor>,
    selection: Option<((usize, usize), (usize, usize))>,
    mouse_down: Option<Cursor>,
}

impl CursorContext {
    fn move_up(&mut self, text_buffer: &TextBuffer) -> EditAction {
        self.cursor
            .as_mut()
            .map(|cursor| cursor.move_up(text_buffer))
            .unwrap_or(EditAction::Noop)
    }
    fn move_down(&mut self, text_buffer: &TextBuffer) -> EditAction {
        self.cursor
            .as_mut()
            .map(|cursor| cursor.move_down(text_buffer))
            .unwrap_or(EditAction::Noop)
    }
    fn move_left(&mut self, text_buffer: &TextBuffer) -> EditAction  {
        self.cursor
            .as_mut()
            .map(|cursor| cursor.move_left(text_buffer))
            .unwrap_or(EditAction::Noop)
    }
    fn move_right(&mut self, text_buffer: &TextBuffer) -> EditAction  {
        self.cursor
            .as_mut()
            .map(|cursor| cursor.move_right(text_buffer))
            .unwrap_or(EditAction::Noop)
    }
    fn start_of_line(&mut self) {
        if let Some(cursor) = self.cursor.as_mut() {
            cursor.start_of_line();
        }
    }

    fn end_of_line(&mut self, text_buffer: &TextBuffer) {
        if let Some(cursor) = self.cursor.as_mut() {
            cursor.1 = text_buffer.line_length(cursor.0);
        }
    }

    fn set_cursor(&mut self, cursor: Cursor) {
        self.cursor = Some(cursor);
    }

    fn clear_selection(&mut self) {
        self.selection = None;
    }

    fn set_selection(&mut self, selection: ((usize, usize), (usize, usize))) {
        self.selection = Some(selection);
    }

    fn fix_cursor(&mut self, text_buffer: &TextBuffer) {
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

    fn copy_selection(&self, clipboard: &ClipboardUtil, text_buffer: &TextBuffer) -> Option<()> {
        if let Some(((start_line, start_column), (end_line, end_column))) = self.selection {
            let start = text_buffer.char_position_from_line_column(start_line, start_column)?;
            let end = text_buffer.char_position_from_line_column(end_line, end_column)?;
            clipboard.set_clipboard_text(from_utf8(&text_buffer.chars[start..end]).unwrap()).ok()?;
        }

        Some(())
    }

    fn mouse_down(&mut self) {
        self.mouse_down = self.cursor;
    }

    fn clear_mouse_down(&mut self) {
        self.mouse_down = None;
    }

    fn move_cursor_from_screen_position(&mut self, scroller: &Scroller, x: usize, y: usize, text_buffer: &TextBuffer, bounds: &EditorBounds) {
        self.cursor = text_space_from_screen_space(scroller, x, y, text_buffer, bounds);
    }

    fn handle_insert(&mut self, to_insert: &[u8], text_buffer : &mut TextBuffer) -> EditAction {
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

    fn paste(&mut self, clipboard: &ClipboardUtil, text_buffer: &mut TextBuffer) -> Option<String> {
        if let Some(Cursor(line, column)) = self.cursor {
            let position = text_buffer.char_position_from_line_column(line, column)?;
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

#[derive(Debug, Clone)]
struct Scroller {
    offset_x: i32,
    offset_y: i32,
    scroll_speed: i32
}

impl Scroller {
    fn scroll_y(&mut self, height: usize, amount: i32, text_buffer: &TextBuffer, bounds: &EditorBounds) {
        if !self.at_end(height, text_buffer, bounds) || amount < 0 {
            self.offset_y += amount * self.scroll_speed;
        }
        // Need to set offsety if we are at the end
        // so that it stays consistent
        self.offset_y = max(0, self.offset_y);
    }

    fn scroll_x(&mut self, width: usize, amount: i32, text_buffer: &mut TextBuffer, bounds: &EditorBounds) {
        self.offset_x += amount * self.scroll_speed;
        self.offset_x = max(0, self.offset_x);
        let max_right = text_buffer.max_line_width() * bounds.letter_width;
        // I probably need something in pane to get this right.
        let viewable_width =  width - bounds.letter_width*2 - bounds.line_number_padding(text_buffer);
        if self.offset_x + viewable_width as i32 > max_right as i32{
            self.offset_x = (max_right.saturating_sub(viewable_width)) as i32;
        }
        // self.offset_x = min(self.offset_x, (text_buffer.max_line_width() * self.bounds.letter_width) as i32);
    }
    // Will need to do scroll fraction for x too.
    fn scroll_x_character(&self, bounds: &EditorBounds) -> usize {
        self.offset_x as usize / bounds.letter_width
    }

    fn viewing_lines(&self, height: usize, bounds: &EditorBounds) -> usize {
       height / bounds.letter_height
    }

    fn lines_above_fold(&self, bounds: &EditorBounds) -> usize {
        self.offset_y as usize / bounds.letter_height as usize
    }

    fn at_end(&self, height: usize, text_buffer: &TextBuffer, bounds: &EditorBounds) -> bool {
        self.lines_above_fold(bounds) + self.viewing_lines(height, bounds) >= text_buffer.line_count() + 3
    }

    fn move_to_the_top(&mut self) {
        self.offset_y = 0;
    }

    fn line_fraction_y(&self, bounds: &EditorBounds) -> usize {
        self.offset_y as usize % bounds.letter_height as usize
    }

    fn line_fraction_x(&self, bounds: &EditorBounds) -> usize {
        self.offset_x as usize % bounds.letter_width as usize
    }
}


struct Renderer<'a> {
    canvas: Canvas<video::Window>,
    texture: Texture<'a>,
    target: Rect,
    bounds: EditorBounds,
    system_cursor: sdl2::mouse::Cursor,
}


impl<'a> Renderer<'a> {

    fn draw_triangle(&mut self, x: i16, y: i16, color: Color) -> Result<(), String> {
        self.canvas.filled_trigon(x, y, x, y+10, x+5, y+5, color)
    }

    fn set_draw_color(&mut self, color: Color) {
        self.canvas.set_draw_color(color);
    }

    fn clear(&mut self) {
        self.canvas.clear();
    }

    fn set_color_mod(&mut self, color: Color) {
        let (r, g, b, _) = color.rgba();
        self.texture.set_color_mod(r, g, b);
    }

    fn set_initial_rendering_location(&mut self, scroller: &Scroller) {
        self.target = Rect::new(
            self.bounds.editor_left_margin as i32,
            (scroller.line_fraction_y(&self.bounds) as i32).neg() + self.bounds.letter_height as i32 * 2,
            self.bounds.letter_width as u32,
            self.bounds.letter_height as u32
        );
    }

    fn copy(&mut self, source: &Rect) -> Result<(), String> {
        self.canvas.copy(&self.texture, *source, self.target)
    }

    fn fill_rect(&mut self, rect: &Rect) -> Result<(), String> {
        self.canvas.fill_rect(*rect)
    }

    fn draw_rect(&mut self, rect: &Rect) -> Result<(), String> {
        self.canvas.draw_rect(*rect)
    }

    fn move_right(&mut self, padding: i32) {
        self.target.set_x(self.target.x() + padding);
    }
    fn move_left(&mut self, padding: i32) {
        self.target.set_x(self.target.x().saturating_sub(padding));
    }

    fn move_down(&mut self, padding: i32) {
        self.target.set_y(self.target.y() + padding);
    }

    fn move_right_one_char(&mut self) {
        self.move_right(self.bounds.letter_width as i32);
    }
    fn move_down_one_line(&mut self) {
        self.move_down(self.bounds.letter_height as i32);
    }

    fn set_x(&mut self, x: i32) {
        self.target.set_x(x);
    }

    fn set_y(&mut self, x: i32) {
        self.target.set_y(x);
    }


    fn char_position_in_atlas(&self, c: char) -> Rect {
        Rect::new(self.bounds.letter_width as i32 * (c as i32 - 33), 0, self.bounds.letter_width as u32, self.bounds.letter_height as u32)
    }

    fn draw_string(&mut self, text: &str) -> Result<(), String> {
        for char in text.chars() {
            self.move_right_one_char();
            self.copy(&self.char_position_in_atlas(char))?
        }
        Ok(())
    }

    fn draw_fps(&mut self, fps: &mut FpsCounter, window: &Window) -> Result<(), String> {
        let current_fps = fps.tick();
        // Do something better with this target
        self.target = Rect::new(window.width - (self.bounds.letter_width * 10) as i32, 0, self.bounds.letter_width as u32, self.bounds.letter_height as u32);
        self.set_color_mod(STANDARD_TEXT_COLOR);
        self.draw_string(&format!("fps: {}", current_fps))
    }

    fn draw_column_line(&mut self, pane_manager: &mut PaneManager) -> Result<(), String> {
        self.target = Rect::new(pane_manager.window.width - (self.bounds.letter_width * 22) as i32, pane_manager.window.height-self.bounds.letter_height as i32, self.bounds.letter_width as u32, self.bounds.letter_height as u32);
        if let Some(pane) = pane_manager.get_active_pane() {
            if let Some(Cursor(cursor_line, cursor_column)) = pane.cursor_context.cursor {
                self.draw_string( &format!("Line {}, Column {}", cursor_line, cursor_column))?;
            }
        }
        Ok(())
    }

    fn present(&mut self) {
        self.canvas.present();
    }

   fn set_cursor_pointer(&mut self) {
        self.system_cursor = sdl2::mouse::Cursor::from_system(SystemCursor::Hand).unwrap();
        self.system_cursor.set();
    }


   fn set_cursor_ibeam(&mut self) {
        self.system_cursor = sdl2::mouse::Cursor::from_system(SystemCursor::IBeam).unwrap();
        self.system_cursor.set();
    }


}

fn in_square(mouse_pos: (i32, i32), square_pos: (i32, i32), square_size: i32) -> bool {
    let (x, y) = mouse_pos;
    let (x_pos, y_pos) = square_pos;
    let size = square_size;
    x >= x_pos && x <= x_pos + size && y >= y_pos && y <= y_pos + size
}


#[derive(Debug, Clone)]
struct Pane {
    name: String,
    scroller: Scroller,
    cursor_context: CursorContext,
    text_buffer: TextBuffer,
    position: (i32, i32),
    width: usize,
    height: usize,
    active: bool,
    transaction_manager: TransactionManager,
    editing_name: bool,
    mouse_pos: Option<(i32, i32)>,
    draw_commands: Vec<DrawCommand>,
}

// Thoughts:
// I don't have any notion of bounds to stop rendering at.
// I need to think about that here.
impl Pane {

    fn adjust_position(&self, x: i32, y: i32, bounds: &EditorBounds) -> (usize, usize) {
        (max(0, x - self.position.0) as usize, max(0,y - self.position.1 - (bounds.letter_height * 2) as i32) as usize)
    }

    fn max_characters_per_line(&self, bounds: &EditorBounds) -> usize {
        let padding = bounds.line_number_padding(&self.text_buffer);
        let extra_padding = bounds.letter_width;
        if self.width < padding + extra_padding {
           0
        } else {
            ((self.width - padding - extra_padding) / bounds.letter_width) + 2
        }

    }

    fn max_lines_per_page(&self, bounds: &EditorBounds) -> usize {
        self.height / bounds.letter_height as usize
    }
    

    fn draw_with_texture(&mut self, renderer: &mut Renderer) -> Result<(), String> {
        let texture_creator = renderer.canvas.texture_creator();
        let texture = &mut texture_creator.create_texture_target(renderer.canvas.default_pixel_format(), self.width as u32, self.height as u32).unwrap();

        if renderer.canvas.render_target_supported() {
            // TODO: This is some ugly code I should isolate;
            let target = unsafe { sys::SDL_GetRenderTarget(renderer.canvas.raw()) };
            unsafe {
                if sys::SDL_SetRenderTarget(renderer.canvas.raw(), texture.raw()) != 0 {
                    panic!("Failed to set render target");
                }
            }

            self.draw(renderer)?;

            unsafe {
                if sys::SDL_SetRenderTarget(renderer.canvas.raw(), target) != 0 {
                    panic!("Failed to set render target");
                }
            }
        }
        renderer.canvas.copy(&texture, None, Rect::new(self.position.0, self.position.1, self.width as u32, self.height as u32))?;

        Ok(())
    }

    fn draw(&mut self, renderer: &mut Renderer) -> Result<(), String> {


        let mut line_decorations : Vec<(i32, i32, i32, i32, i32)> = vec![];
        for draw_command in self.draw_commands.iter() {
            match draw_command {
                &DrawCommand::RectOnPaneAtLocation(_, start_line, start_column, end_line, end_column, height) => {
                    line_decorations.push((start_line, start_column, end_line, end_column, height));
                }
                _ => {}
            }
        }


        // It would be great if we normalized the drawing here
        // to be relative to the pane itself.
        // Could simplify quite a lot.
        let editor_left_margin = renderer.bounds.editor_left_margin;
        let line_number_padding = renderer.bounds.line_number_padding(&self.text_buffer);
        let line_number_digits = renderer.bounds.line_number_digits(&self.text_buffer);

        renderer.set_draw_color(Color::RGBA(42, 45, 62, 255));
        renderer.set_y(0);
        renderer.fill_rect(
            &Rect::new(
                0,
                0,
                self.width as u32,
                self.height as u32
            )
        )?;




        renderer.set_initial_rendering_location(&self.scroller);
        // renderer.move_right(self.position.0 as i32);
        // renderer.move_down(self.position.1 as i32);

        let number_of_lines = min(self.scroller.lines_above_fold(&renderer.bounds) + self.max_lines_per_page(&renderer.bounds) + 2, self.text_buffer.line_count());
        let starting_line = self.scroller.lines_above_fold(&renderer.bounds);


        self.text_buffer.tokenizer.skip_lines(starting_line, &self.text_buffer.chars);

        for line in starting_line as usize..number_of_lines {
            renderer.set_color_mod(STANDARD_TEXT_COLOR);
            renderer.set_x(editor_left_margin as i32);

            if self.width > renderer.bounds.line_number_padding(&self.text_buffer)  {
                self.draw_line_numbers(renderer, line_number_digits, line)?;
            }

            // This really shouldn't be in this loop.
            // I could instead figure out if it is visbible.
            self.draw_cursor(renderer, line, line_number_padding)?;
            self.draw_selection(renderer, line, line_number_padding)?;

            renderer.move_left(self.scroller.line_fraction_x(&renderer.bounds) as i32);
            let initial_x_position = renderer.target.x;
            self.draw_code(renderer, line)?;
            let left_most_character = self.scroller.scroll_x_character(&renderer.bounds);

            // This is all bad. I need a way of dealing with scrolling and positions of things
            // It really shouldn't be that hard, but my brain doesn't seem to like it.
            for decoration in line_decorations
                    .iter_mut()
                    .filter(|(line_number, _, _, _, _)| *line_number == line as i32) {
                renderer.set_draw_color(CURSOR_COLOR);


                let (_line_start, start_column, _line_end, end_column, height) = *decoration;

                let start = if start_column <= left_most_character as i32 {
                    // Didnt want to think about why this is +1
                    // But it seems to work.
                    left_most_character + 1
                } else {
                    start_column as usize
                };

                let end = min(end_column as usize, start + self.max_characters_per_line(&renderer.bounds));
                let start = min(start, end);

                let characters_before_start = start.saturating_sub(left_most_character);
                
                let mut rect = Rect::new(0,0,0,height as u32);
                let target = renderer.target;
                rect.set_x(initial_x_position + characters_before_start as i32 * renderer.bounds.letter_width as i32);
                rect.set_y(target.y + rect.y);
                let letter_width = end.saturating_sub(start);
                if letter_width > 0 {
                    rect.set_width(letter_width as u32 * renderer.bounds.letter_width as u32);
                    renderer.draw_rect(&rect)?;
                }
                
                

            }

            renderer.move_down_one_line();
        }
        

        renderer.set_draw_color(CURSOR_COLOR);
        for draw_command in self.draw_commands.iter_mut() {
            match draw_command {
                DrawCommand::RectOnPane(_, rect) => {
                    rect.set_x(rect.x() - self.scroller.offset_x + renderer.bounds.letter_width as i32);
                    rect.set_y(rect.y() - self.scroller.offset_y + renderer.bounds.letter_height as i32);
                    renderer.draw_rect(rect)?;
                }
                _ => {}
            }
           
        }
        self.draw_commands.clear();

        renderer.set_draw_color(Color::RGBA(42, 45, 62, 255));

        // Ummm, why don't I just draw an unfilled rectangle?
        // Is it becasue I want some sides different?
        // probably not for part of this.
       
        // top
        renderer.fill_rect(
            &Rect::new(
                0,
                0,
                (self.width + renderer.bounds.letter_width) as u32,
                renderer.bounds.letter_height as u32))?;
        // bottom
        renderer.fill_rect(
            &Rect::new(
                0,
                self.height as i32 - renderer.bounds.letter_height as i32,
                self.width as u32,
                renderer.bounds.letter_height as u32))?;

        // Do I need this???
        // left
        // renderer.fill_rect(
        //     &Rect::new(
        //          renderer.bounds.line_number_padding(&self.text_buffer) as i32 - renderer.bounds.letter_width as i32,
        //         0,
        //         renderer.bounds.letter_width as u32,
        //         self.height as u32))?;

        // right
        renderer.fill_rect(&Rect::new(
            self.width as i32 - renderer.bounds.letter_width as i32,
            0,
            (renderer.bounds.letter_width * 2) as u32,
            self.height as u32 + (renderer.bounds.letter_height * 2) as u32))?;


        renderer.set_draw_color(Color::RGBA(36, 39, 55, 255));
        // Really need to think about split vs freefloating
        if self.position.0 != 0 {

            renderer.fill_rect(&Rect::new(0,  0, 2, self.height as u32))?;
            renderer.fill_rect(&Rect::new(self.width as i32 - 2, 0, 2, self.height as u32))?;
        }

        renderer.fill_rect(
                &Rect::new(
                    0,
                    0,
                    self.width as u32,
                    renderer.bounds.letter_height as u32))?;

        renderer.fill_rect(
            &Rect::new(
                0,
                self.height as i32 - renderer.bounds.letter_height as i32,
                self.width as u32,
                renderer.bounds.letter_height as u32))?;


        if self.editing_name {
            let padding = 5;
            renderer.set_draw_color(CURSOR_COLOR);
            renderer.draw_rect(&Rect::new(
                self.width as i32 - (renderer.bounds.letter_width * (self.name.len() + 2)) as i32 - padding,
                0,
                renderer.bounds.letter_width as u32 * (self.name.len() + 1) as u32 + padding as u32,
                renderer.bounds.letter_height as u32 + padding as u32))?;
        }
        renderer.set_color_mod(STANDARD_TEXT_COLOR);
        renderer.set_x(self.width as i32 - (renderer.bounds.letter_width * (self.name.len() + 3)) as i32);
        renderer.set_y(0);
        renderer.draw_string(&self.name)?;

        // I hate that this requires i16.
        // Will this cause issues if I allow scrolling?
        fn into_i16(x: i32) -> i16 {
            x.try_into().unwrap()
        }
        let play_button_x = renderer.bounds.letter_width as i32;
        let play_button_y = 4;

        renderer.draw_triangle(into_i16(play_button_x),into_i16(play_button_y), STANDARD_TEXT_COLOR)?;

        if let Some(mouse_pos) = self.mouse_pos {
            if in_square(mouse_pos, (play_button_x, play_button_y), 10) {
                renderer.set_cursor_pointer();
            }
        }



        // TODO: do something better
        self.text_buffer.tokenizer.position = 0;
        Ok(())
    }

    // I might want to move this out of here at some point
    // Not really sure what the responsibilities for this renderer should be.
    fn draw_line_numbers(&mut self, renderer: &mut Renderer, line_number_digits: usize, line: usize) -> Result<(), String> {
        // I want to pad this so that the offset by the line number never changes.
        // Really I should draw a line or something to make it look nicer.
        let left_padding_count = line_number_digits - digit_count(line + 1);
        let padding = left_padding_count * renderer.bounds.letter_width as usize;
        renderer.move_right(padding as i32);
        let line_number = (line + 1).to_string();
        renderer.draw_string(&line_number)?;
        renderer.move_right(renderer.bounds.line_number_gutter_width as i32);
        Ok(())
    }


    fn draw_cursor(&mut self, renderer: &mut Renderer, line: usize, line_number_padding: usize) -> Result<(), String> {
        if !self.active {
            return Ok(());
        }
        if let Some(cursor) = self.cursor_context.cursor {
            if cursor.0 == line {
                let cursor_x = cursor.1 as i32  * renderer.bounds.letter_width as i32 + line_number_padding as i32;
                let cursor_y = renderer.target.y();

                renderer.set_draw_color(CURSOR_COLOR);
                renderer.fill_rect(&Rect::new(cursor_x as i32, cursor_y as i32, 2, renderer.bounds.letter_height as u32))?
            }
        }
        Ok(())
    }

    fn draw_selection(&mut self, renderer: &mut Renderer, line: usize, line_number_padding: usize) -> Result<(), String> {


        // TODO: Really think about this and make it work properly.
        // I need some way of looking at a character and easily deciding where it is
        // if it is visible, etc.

        // Really this shouldn't be that different than drawing code right?
        // Maybe I should just record the character range instead of the line range?
        if let Some(((start_line, start_column), (end_line, end_column))) = self.cursor_context.selection {

            // TODO: This isn't properly cutting off at the pane boundary.
            if line >= start_line && line <= end_line {
                let mut start_x : i32 = if line == start_line {
                    (start_column * renderer.bounds.letter_width + line_number_padding) as i32
                } else {
                    line_number_padding as i32
                };
                start_x = start_x
                    - (self.scroller.scroll_x_character(&renderer.bounds) as i32 * renderer.bounds.letter_width as i32)
                    - self.scroller.line_fraction_x(&renderer.bounds) as i32;


                let mut width = if start_line == end_line {
                    (end_column - start_column) * renderer.bounds.letter_width
                } else if line == end_line {
                    end_column * renderer.bounds.letter_width as usize
                } else if line == start_line {
                    (self.text_buffer.line_length(line) - start_column) * renderer.bounds.letter_width as usize
                } else {
                    self.text_buffer.line_length(line)  * renderer.bounds.letter_width
                };

                width = if start_x < line_number_padding as i32 + renderer.bounds.letter_width as i32 {
                    width - min( width as i32, line_number_padding.saturating_sub(start_x as usize) as i32) as usize
                } else {
                    width
                };

                start_x = max(start_x, line_number_padding as i32);


                let start_y = renderer.target.y();
                renderer.set_draw_color(Color::RGBA(65, 70, 99, 255));
                // Need to deal with last line.
                if width != 0 {
                    renderer.fill_rect(&Rect::new(start_x as i32, start_y, width as u32, renderer.bounds.letter_height as u32))?
                }
            }

        };
        Ok(())
    }


    fn draw_code(&mut self, renderer: &mut Renderer, line: usize) -> Result<(), String> {

        let (line_start, line_end) = self.text_buffer[line];
        let start = min(line_start + self.scroller.scroll_x_character(&renderer.bounds), line_end);
        let end = min(line_end, start + self.max_characters_per_line(&renderer.bounds));
        // Super awkward hacky approach. But fine for now
        let mut position = line_start;
        let tokenizer = &mut self.text_buffer.tokenizer;

        while !tokenizer.at_end(&self.text_buffer.chars) && !tokenizer.is_newline(&self.text_buffer.chars) {
            if let Some(token) = tokenizer.parse_single(&self.text_buffer.chars) {
                let token = rust_specific_pass(token, &self.text_buffer.chars);
                let color = color_for_token(&token, &self.text_buffer.chars);
                let text = from_utf8(match token {
                    RustSpecific::Keyword((s, e)) => &self.text_buffer.chars[s..e],
                    RustSpecific::Token(t) => {
                        match t {
                            Token::OpenParen => &[b'('],
                            Token::CloseParen => &[b')'],
                            Token::OpenCurly => &[b'{'],
                            Token::CloseCurly => &[b'}'],
                            Token::OpenBracket => &[b'['],
                            Token::CloseBracket => &[b']'],
                            Token::SemiColon => &[b';'],
                            Token::Colon => &[b':'],
                            Token::Comma => &[b','],
                            Token::NewLine => &[],
                            Token::Comment((s, e))
                            | Token::Spaces((s, e))
                            | Token::String((s, e))
                            | Token::Integer((s, e))
                            | Token::Float((s, e))
                            | Token::Atom((s, e)) => &self.text_buffer.chars[s..e],
                        }
                    }
                }).unwrap();

                if position > end {
                    continue;
                }
                let token_start = {
                    if position < start && position + text.len() < start {
                        None
                    } else if position < start {
                        Some(start - position)
                    } else {
                        Some(0)
                    }
                };


                let token_end = {
                    if position + text.len() > end {
                        min(end - position, text.len())
                    } else {
                        text.len()
                    }
                };


                if let Some(token_start) = token_start {

                    let token = &text[token_start..token_end];
                    renderer.set_color_mod(color);
                    renderer.draw_string(token)?;
                }

                position += text.len();



            }

        }
        if !tokenizer.at_end(&self.text_buffer.chars) {
            tokenizer.consume();
        }
        Ok(())


    }

    fn is_mouse_over(&self, (x, y): (i32, i32), bounds: &EditorBounds) -> bool {
        x > self.position.0  &&
        x < self.position.0 + self.width as i32 &&
        y > self.position.1 - bounds.letter_height as i32 &&
        y < self.position.1 + self.height as i32 + bounds.letter_height  as i32
    }

    fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    fn mouse_over_play_button(&self, mouse_pos: (i32, i32), bounds: &EditorBounds) -> bool {
        // probably better to make this work with adjusted positions
        let play_button_x = self.position.0 + bounds.letter_width as i32;
        let play_button_y = self.position.1 - bounds.letter_height as i32 + 4;
        in_square(mouse_pos, (play_button_x, play_button_y), 10)
    }

    fn on_click(&mut self, mouse_pos: (i32, i32), bounds: &EditorBounds) -> Option<SideEffectAction> {
        if self.mouse_over_play_button(mouse_pos, bounds) {
            Some(SideEffectAction::Play(self.name.clone()))
        } else {
            let (x, y) = self.adjust_position(mouse_pos.0, mouse_pos.1, bounds);
            self.cursor_context.move_cursor_from_screen_position(&self.scroller, x, y, &self.text_buffer, bounds);
            self.cursor_context.mouse_down();
            self.cursor_context.clear_selection();
            None
        }

    }


}

fn draw(renderer: &mut Renderer, pane_manager: &mut PaneManager, fps: &mut FpsCounter) -> Result<(), String> {
    renderer.set_draw_color(BACKGROUND_COLOR);
    renderer.clear();
    handle_draw_panes(pane_manager, renderer)?;
    for pane in pane_manager.panes.iter_mut() {
        pane.draw_with_texture(renderer)?;
    }

    if pane_manager.create_pane_activated {
        renderer.set_draw_color(Color::RGBA(255, 255, 255, 255));
        // Need to deal with current < start

        let position_x = min(pane_manager.create_pane_start.0, pane_manager.create_pane_current.0);
        let position_y = min(pane_manager.create_pane_start.1, pane_manager.create_pane_current.1);
        let current_x = max(pane_manager.create_pane_start.0, pane_manager.create_pane_current.0);
        let current_y = max(pane_manager.create_pane_start.1, pane_manager.create_pane_current.1);
        let width = (current_x - position_x) as u32;
        let height = (current_y - position_y) as u32;


        renderer.draw_rect(&Rect::new(
            position_x,
            position_y,
            width,
            height))?;
    }

    // TODO:
    // Fix this whole scroller weirdness.
    // Really just need window here.
    renderer.draw_fps(fps, &pane_manager.window)?;
    // Does this belong in the pane?
    // Is it global?
    // Need to think about the UI
    renderer.draw_column_line(pane_manager)?;

    renderer.present();

    Ok(())
}




#[derive(Debug, Clone)]
struct TextBuffer {
    line_range: Vec<(usize, usize)>,
    chars: Vec<u8>,
    max_line_width_cache: usize,
    tokenizer: Tokenizer,
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
    fn get_line(&self, line: usize) -> Option<&(usize,usize)> {
        self.line_range.get(line)
    }

    fn line_count(&self) -> usize {
        self.line_range.len()
    }

    fn line_length(&self, line: usize) -> usize {
        if let Some(line_range) = self.get_line(line) {
            line_range.1 - line_range.0
        } else {
            // Should this be 0?
            0
        }
    }

    fn delete_line(&mut self, line: usize) {
        self.line_range.remove(line);
    }

    fn insert(&mut self, index: usize, line: (usize, usize)) {
        self.line_range.insert(index, line);
    }

    fn splice<R: RangeBounds<usize>, I: IntoIterator<Item=(usize,usize)>>(&mut self, index: R, lines: I) {
        self.line_range.splice(index, lines);
    }

    fn lines_iter_mut(&mut self) -> impl Iterator<Item=&mut (usize,usize)> {
        self.line_range.iter_mut()
    }

    fn last_line(&self) -> Option<&(usize,usize)> {
        self.line_range.last()
    }

    fn parse_lines(&mut self) {
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

    fn set_contents(&mut self, contents: &[u8]) {
        self.chars = contents.to_vec();
        self.parse_lines();
    }

    fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    fn char_length(&self) -> usize {
        self.chars.len()
    }

    fn insert_char(&mut self, cursor: Cursor, to_insert: &[u8]) -> EditAction {
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

    fn remove_char(&mut self, cursor: Cursor) -> EditAction {

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

    fn max_line_width(&mut self) -> usize {
        // TODO: Update cache on edit!
        if self.max_line_width_cache != 0 {
            return self.max_line_width_cache;
        }
        self.max_line_width_cache = self.line_range.iter().map(|(start,end)| end - start).max().unwrap_or(0);
        self.max_line_width_cache
    }

    fn char_position_from_line_column(&self, line_number: usize, column_number: usize) -> Option<usize> {
        if let Some((start_line, end_line))  = self.line_range.get(line_number) {
            if (end_line - start_line) < column_number {
                return None;
            }
            Some(start_line + column_number)
        } else {
            None
        }

    }

    fn line_column_from_char_position(&self, position: usize) -> Option<Cursor> {
        for (i, (start, end)) in self.line_range.iter().enumerate() {
            if position >= *start && position <= *end {
                return Some(Cursor(i, position - start));
            }
        }
        None
    }

}


// Need a draw order for z-index purposes.
struct PaneManager {
    panes: Vec<Pane>,
    active_pane: usize,
    scroll_active_pane: usize,
    window: Window,
    dragging_start: (i32, i32),
    dragging_pane: Option<usize>,
    dragging_pane_start: (i32, i32),
    resize_start: (i32, i32),
    resize_pane: Option<usize>,
    create_pane_activated: bool,
    create_pane_start: (i32, i32),
    create_pane_current: (i32, i32),
}

impl PaneManager {

    fn set_mouse_pos(&mut self, mouse_pos: (i32, i32)) {
        if let Some(pane) = self.panes.get_mut(self.scroll_active_pane) {
            pane.mouse_pos = Some(mouse_pos);
        }
    }

    fn delete_pane_at_mouse(&mut self, mouse_pos: (i32, i32), bounds: &EditorBounds) {
        if let Some(closest_pane) = self.get_pane_index_at_mouse(mouse_pos, bounds) {
            self.panes.remove(closest_pane);
        }
    }

    fn set_scroll_active_if_mouse_over(&mut self, mouse_pos: (i32, i32), bounds: &EditorBounds) {
        let mut new_active_pane = self.scroll_active_pane;
        for (i, pane) in self.panes.iter().enumerate() {
            if pane.is_mouse_over(mouse_pos, bounds) {
                new_active_pane = i;
            }
        }
        if new_active_pane != self.scroll_active_pane {
            self.scroll_active_pane = new_active_pane;
        }
        self.set_mouse_pos(mouse_pos)
    }

    fn set_active_from_click_coords(&mut self, mouse_pos: (i32, i32), bounds: &EditorBounds) {
        let old_active = self.active_pane;
        let mut new_active_pane = self.active_pane;
        for (i, pane) in self.panes.iter().enumerate() {
            if pane.is_mouse_over(mouse_pos, bounds) {
                new_active_pane = i;
            }
        }
        self.active_pane = new_active_pane;
        if let Some(pane) = self.panes.get_mut(self.active_pane){
            pane.set_active(true);
        }
        if new_active_pane != self.active_pane {
            if let Some(pane) = self.panes.get_mut(old_active) {
                pane.set_active(false);
            }
        }
    }

    fn get_pane_index_at_mouse(&mut self, mouse_pos: (i32, i32), bounds: &EditorBounds) -> Option<usize> {
        for (i, pane) in self.panes.iter().enumerate().rev() {
            if pane.is_mouse_over(mouse_pos, bounds) {
                return Some(i);
            }
        }
        None
    }

    fn get_pane_at_mouse_mut(&mut self, mouse_pos: (i32, i32), bounds: &EditorBounds) -> Option<&mut Pane> {
        if let Some(i) = self.get_pane_index_at_mouse(mouse_pos, bounds) {
            return self.panes.get_mut(i);
        }
        None
    }

    // TODO: This should be a Option<Pane> because there might not be any.
    fn get_active_pane_mut(&mut self) -> Option<&mut Pane> {
        self.panes.get_mut(self.active_pane)
    }

    fn get_scroll_active_pane_mut(&mut self) -> Option<&mut Pane> {
        self.panes.get_mut(self.scroll_active_pane)
    }


    fn get_active_pane(&mut self) -> Option<&Pane> {
        self.panes.get(self.active_pane)
    }

    fn get_dragging_pane_mut(&mut self) -> Option<&mut Pane> {
        if let Some(i) = self.dragging_pane {
            Some(&mut self.panes[i])
        } else {
            None
        }
    }

    fn get_resize_pane_mut(&mut self) -> Option<&mut Pane> {
        if let Some(i) = self.resize_pane {
            Some(&mut self.panes[i])
        } else {
            None
        }
    }

    fn set_dragging_start(&mut self, mouse_pos: (i32, i32), bounds: &EditorBounds) -> bool {
        if let Some(i) = self.get_pane_index_at_mouse(mouse_pos, bounds) {
            let pane = self.panes.remove(i);
            self.panes.push(pane);

            let new_i = self.panes.len() - 1;
            if self.active_pane == i {
                self.active_pane = new_i;
            }
            self.dragging_start = mouse_pos;
            self.dragging_pane = Some(new_i);
            self.dragging_pane_start = self.panes[new_i].position;
            return true
        }
        false
    }

    fn update_dragging_position(&mut self, mouse_pos: (i32, i32)) {
        let (x, y) = mouse_pos;
        let (x_diff, y_diff) = (x - self.dragging_start.0, y - self.dragging_start.1);
        let (pane_x, pane_y) = self.dragging_pane_start;
        if let Some(pane) = self.get_dragging_pane_mut() {
            pane.position = (pane_x as i32 + x_diff, pane_y as i32 + y_diff);
        }
    }

    fn stop_dragging(&mut self) {
        self.dragging_pane = None;
    }

    fn set_resize_start(&mut self, mouse_pos: (i32, i32), bounds: &EditorBounds) -> bool {
        if let Some(i) = self.get_pane_index_at_mouse(mouse_pos, bounds) {
            self.resize_start = mouse_pos;
            self.resize_pane = Some(i);
            self.update_resize_size(mouse_pos);
            true
        } else {
            false
        }
    }

    fn update_resize_size(&mut self, mouse_pos: (i32, i32)) {
        let (x, y) = mouse_pos;

        if let Some(pane) = self.get_resize_pane_mut() {

            let (current_x, current_y) = pane.position;

            if x < current_x || y < current_y {
                return;
            }

            let width = x - current_x;
            let height = y - current_y;

            pane.width = width as usize;
            pane.height = height as usize;
        }
    }

    fn stop_resizing(&mut self) {
        self.resize_pane = None;
    }

    fn set_create_start(&mut self, mouse_pos: (i32, i32)) {
        self.create_pane_activated = true;
        self.create_pane_start = mouse_pos;
        self.create_pane_current = mouse_pos;
    }

    fn update_create_pane(&mut self, mouse_pos: (i32, i32)) {
        if self.create_pane_activated {
            self.create_pane_current = mouse_pos;
        }
    }

    fn create_pane_raw(&mut self, pane_name: String, position: (i32, i32), width: usize, height: usize) -> usize{
        // This is duplicate code
        let scroller = Scroller {
            offset_y: 0,
            offset_x: 0,
            scroll_speed: 5,
        };

        let cursor_context = CursorContext {
            cursor: None,
            mouse_down: None,
            selection: None,
        };

        self.panes.push(Pane {
            name: pane_name,
            position,
            width,
            height,
            active: true,
            scroller,
            cursor_context,
            text_buffer: TextBuffer {
                line_range: vec![(0,0)],
                chars: vec![],
                max_line_width_cache: 0,
                tokenizer: Tokenizer::new(),
            },
            mouse_pos: None,
            transaction_manager: TransactionManager::new(),
            editing_name: false,
            draw_commands: vec![],
        });
        self.panes.len() - 1
    }

    fn create_pane(&mut self) {
        if self.create_pane_activated {
            self.create_pane_activated = false;


            let position_x = min(self.create_pane_start.0, self.create_pane_current.0);
            let position_y = min(self.create_pane_start.1, self.create_pane_current.1);
            let current_x = max(self.create_pane_start.0, self.create_pane_current.0);
            let current_y = max(self.create_pane_start.1, self.create_pane_current.1);
            let width = (current_x - position_x) as usize;
            let height = (current_y - position_y) as usize;

            let mut rng = rand::thread_rng();
            self.create_pane_raw(format!("temp-{}", rng.gen::<u32>()), (position_x, position_y), width, height);
            self.active_pane = self.panes.len() - 1;
        }
    }

    fn remove(&mut self, i: usize) -> Pane {
        // Should swap_remove but more complicated
        let pane = self.panes.remove(i);
        if i < self.active_pane {
            self.active_pane -= 1;
        }
        pane
    }

    fn insert(&mut self, i: usize, pane: Pane) {
        if i <= self.active_pane {
            self.active_pane += 1;
        }
        self.panes.insert(i, pane);
    }


    fn _get_pane_mut(&mut self, index: usize) -> Option<&mut Pane> {
        self.panes.get_mut(index)
    }

    fn get_pane_by_name_mut(&mut self, pane_name: String) -> Option<&mut Pane> {
        for pane in self.panes.iter_mut() {
            if pane.name.starts_with(&pane_name) {
                return Some(pane);
            }
        }
        None
    }

    fn get_pane_by_name(&mut self, pane_name: String) -> Option<&Pane> {
        for pane in self.panes.iter() {
            if pane.name.starts_with(&pane_name) {
                return Some(pane);
            }
        }
        None
    }


}


fn handle_events(event_pump: &mut sdl2::EventPump,
                 pane_manager: &mut PaneManager,
                 bounds: &EditorBounds,
                 clipboard: &ClipboardUtil) -> Vec<SideEffectAction> {
    let mut is_text_input = false;

    // Is allocating here bad?
    let mut actions: Vec<SideEffectAction> = vec![];

    // let text_buffer = &mut pane.text_buffer;
    // let cursor_context = &mut pane.cursor_context;
    // let scroller = &mut pane.scroller;

    // This whole way of handling things is wrong.
    // We probably need a pane manager.
    // Maybe good chance for a pun?

    let ctrl_is_pressed = event_pump.keyboard_state().is_scancode_pressed(Scancode::LCtrl);
    let alt_is_pressed = event_pump.keyboard_state().is_scancode_pressed(Scancode::LAlt);
    let cmd_is_pressed = event_pump.keyboard_state().is_scancode_pressed(Scancode::LGui);

    for event in event_pump.poll_iter() {
        // println!("frame: {}, event {:?}", frame_counter, event);
        match event {
            Event::Quit { .. } => ::std::process::exit(0),
            // Note: These work I can do enso style quasimodal input
            // Event::KeyUp {keycode, ..} => {
            //     println!("{:?}", keycode);
            // }
            // Event::KeyDown{keycode: Some(Keycode::Escape), ..} => {
            //     println!("{:?}", "yep");
            // }

            Event::KeyDown { keycode, keymod, .. } => {
                matches!(keycode, Some(_));
                match (keycode.unwrap(), keymod) {
                    (Keycode::Up, _) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            pane.cursor_context.move_up(&pane.text_buffer);
                        }
                    },
                    (Keycode::Down, _) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            pane.cursor_context.move_down(&pane.text_buffer);
                        }
                    },
                    (Keycode::Left, _) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            pane.cursor_context.move_left(&pane.text_buffer);
                        }
                    },
                    (Keycode::Right, _) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            pane.cursor_context.move_right(&pane.text_buffer);
                        }
                    },
                    (Keycode::Backspace, _) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {

                            if pane.editing_name {
                                pane.name.pop();
                                continue;
                            }
                            // Need to deal with this in a nicer way

                            if let Some(current_selection) = pane.cursor_context.selection {
                                let (start, end) = current_selection;
                                let (start_line, start_column) = start;
                                let (end_line, end_column) = end;
                                if let Some((line_start, _line_end)) = pane.text_buffer.get_line(start_line as usize) {
                                    let char_start_pos = line_start + start_column as usize ;
                                    if let Some((end_line_start, _line_end)) = pane.text_buffer.get_line(end_line as usize) {
                                        let char_end_pos = end_line_start + end_column as usize;
                                        pane.text_buffer.chars.drain(char_start_pos as usize..char_end_pos as usize);
                                        // Probably shouldn't reparse the whole file.

                                        pane.text_buffer.parse_lines();
                                        pane.cursor_context.clear_selection();
                                        pane.cursor_context.fix_cursor(&pane.text_buffer);
                                        continue;
                                    }

                                }
                            }


                            // Is there a better way to do this other than clone?
                            // Maybe a non-mutating method?
                            // How to deal with optional aspect here?
                            if let Some(current_cursor) = pane.cursor_context.cursor {
                                let mut old_cursor = current_cursor;
                                // We do this move_left first, because otherwise we might end up at the end
                                // of the new line we formed from the deletion, rather than the old end of the line.
                                let cursor_action = old_cursor.move_left(&pane.text_buffer);
                                let action = pane.text_buffer.remove_char(current_cursor);

                                pane.transaction_manager.add_action(action);
                                pane.transaction_manager.add_action(cursor_action);


                                pane.cursor_context.set_cursor(old_cursor);
                            }
                        }
                    }
                    (Keycode::Return, _) if cmd_is_pressed => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            actions.push(SideEffectAction::Play(pane.name.clone()))
                        }
                    }
                    (Keycode::Return, _) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {

                            if pane.editing_name {
                                pane.editing_name = false;
                                continue;
                            }
                            // refactor to be better
                            let action = pane.cursor_context.handle_insert(&[b'\n'], &mut pane.text_buffer);
                            pane.transaction_manager.add_action(action);
                            pane.cursor_context.start_of_line();
                        }
                    },


                    (Keycode::Z, key_mod) => {
                        // I should think about the fact that I might not have an active editor
                        // Should undo undo pane movement?
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            if key_mod == Mod::LGUIMOD || keymod == Mod::RGUIMOD {
                                pane.transaction_manager.undo(&mut pane.cursor_context, &mut pane.text_buffer);
                            } else if key_mod == (Mod::LSHIFTMOD | Mod::LGUIMOD) {
                                pane.transaction_manager.redo(&mut pane.cursor_context, &mut pane.text_buffer);
                            } else {
                                is_text_input = true
                            }
                        }

                    },

                    (Keycode::A, Mod::LCTRLMOD | Mod::RCTRLMOD) => {
                        
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            pane.cursor_context.start_of_line()
                        }
                    },
                    (Keycode::E, Mod::LCTRLMOD | Mod::RCTRLMOD) => {
                        
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            pane.cursor_context.end_of_line(&pane.text_buffer);
                        }
                    },

                    (Keycode::C, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            pane.cursor_context.copy_selection(&clipboard, &pane.text_buffer);
                        }
                    },
                    (Keycode::V, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            // TODO: I NEED UNDO!
                            if let Some(inserted_string) = pane.cursor_context.paste(&clipboard, &mut pane.text_buffer) {
                                if let Some(Cursor(cursor_line, cursor_column)) = pane.cursor_context.cursor {
                                    pane.transaction_manager.add_action(EditAction::Insert((cursor_line, cursor_column), inserted_string));
                                }
                            }
                        }
                    },

                    (Keycode::O, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            let text = native::open_file_dialog();
                            if let Some(text) = text {
                                pane.text_buffer.set_contents(text.as_bytes());
                                pane.scroller.move_to_the_top();
                            }
                        }
                    }
                    (Keycode::A, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                        // This is super ugly, fix.
                            pane.cursor_context.set_selection(((0,0), (pane.text_buffer.line_count()-1, pane.text_buffer.line_length(pane.text_buffer.line_count()-1))));
                        }
                    }

                    _ => is_text_input = true
                }
            }
            Event::TextInput{text, ..} => {
                if let Some(pane) = pane_manager.get_active_pane_mut() {
                    if is_text_input && pane.editing_name {
                        pane.name.push_str(&text);
                    } else if is_text_input {
                        // TODO: Replace with actually deleting the selection.
                        pane.cursor_context.clear_selection();

                        let action = pane.cursor_context.handle_insert(text.as_bytes(), &mut pane.text_buffer);
                        pane.transaction_manager.add_action(action);
                    }
                }
            }

            // Need to make selection work
            // Which probably means changing cursor representation


            Event::MouseButtonDown { x, y, .. } if ctrl_is_pressed && alt_is_pressed && cmd_is_pressed => {
                pane_manager.delete_pane_at_mouse((x, y), bounds);
            }

            Event::MouseButtonDown { x, y, .. }  if ctrl_is_pressed && cmd_is_pressed  => {
                if let Some(pane) = pane_manager.get_pane_at_mouse_mut((x,y), bounds) {
                    pane.editing_name = true;
                }
            }

            Event::MouseButtonDown { x, y, .. } if ctrl_is_pressed && alt_is_pressed => {
                let found = pane_manager.set_resize_start((x,y), bounds);
                if !found {
                    pane_manager.set_create_start((x,y));
                }
            }

            Event::MouseButtonDown { x, y, .. } if cmd_is_pressed && alt_is_pressed => {
                if let Some(i) = pane_manager.get_pane_index_at_mouse((x, y), bounds) {
                    let mut pane = pane_manager.panes[i].clone();
                    pane.position = (pane.position.0 + 20, pane.position.1 + 20);
                    pane_manager.panes.push(pane);
                }
            }

            Event::MouseButtonDown { x, y, .. } if ctrl_is_pressed => {
                let found = pane_manager.set_dragging_start((x, y), bounds);
                if !found {
                    pane_manager.set_create_start((x,y));
                }
            }

            Event::MouseButtonDown { x, y, .. } => {
                pane_manager.set_active_from_click_coords((x, y), bounds);
                if let Some(pane) = pane_manager.get_active_pane_mut() {
                    if let Some(action) = pane.on_click((x, y), bounds) {
                        actions.push(action);
                    }
                }

                if !(ctrl_is_pressed && cmd_is_pressed) {
                    // Really I want this to work even if this is pressed, just for all panes
                    // that are not at the mouse point
                    // But this is a temporary binding anyways.
                    for pane in pane_manager.panes.iter_mut() {
                        pane.editing_name = false;
                    }
                }

            }

            Event::MouseMotion{x, y, .. } => {

                pane_manager.update_dragging_position((x, y));
                pane_manager.update_resize_size((x, y));
                pane_manager.update_create_pane((x, y));

                // TODO:
                // distinguish between active and scrolling.
                // Mouse over is enough for scrolling, but not for active.
                pane_manager.set_scroll_active_if_mouse_over((x, y), bounds);


                if let Some(pane) = pane_manager.get_active_pane_mut() {
                    let (x2, y2) = pane.adjust_position(x, y, bounds);


                    if let Some(Cursor(start_line, mut start_column)) = pane.cursor_context.mouse_down {
                        pane.cursor_context.move_cursor_from_screen_position(&pane.scroller, x2, y2, &pane.text_buffer, bounds);
                        // TODO: Get my int types correct!
                        if let Some(Cursor(line, mut column)) = pane.cursor_context.cursor {
                            let new_start_line = start_line.min(line);
                            let line = line.max(start_line);
                            if new_start_line != start_line || start_line == line && start_column > column {
                                let temp = start_column;
                                start_column = column;
                                column = temp as usize;
                            }

                            // ugly refactor
                            pane.cursor_context.set_selection(((new_start_line, start_column), (line, column)));

                        }
                    }
                }
            }

            Event::MouseButtonUp{x, y, ..} => {
                pane_manager.stop_dragging();
                pane_manager.stop_resizing();
                pane_manager.create_pane();
                if let Some(pane) = pane_manager.get_active_pane_mut() {
                    let (x, y) = pane.adjust_position(x, y, bounds);
                    if let Some(Cursor(start_line, mut start_column)) = pane.cursor_context.mouse_down {
                        pane.cursor_context.move_cursor_from_screen_position(&pane.scroller, x, y, &pane.text_buffer, bounds);
                        if pane.cursor_context.selection.is_some() {
                            if let Some(Cursor(line, mut column)) = pane.cursor_context.cursor {
                                let new_start_line = start_line.min(line);
                                let line = line.max(start_line);
                                if new_start_line != start_line || start_line == line && start_column > column {
                                    let temp = start_column;
                                    start_column = column;
                                    column = temp as usize;
                                }

                                pane.cursor_context.set_selection(((new_start_line, start_column), (line, column)));
                                // TODO: Set Cursor
                            }
                        }

                    }

                    pane.cursor_context.clear_mouse_down();
                }
            }
            // Continuous resize in sdl2 is a bit weird
            // Would need to watch events or something
            Event::Window {win_event: WindowEvent::Resized(width, height), ..} => {

                // for pane in pane_manager.panes.iter_mut() {
                //     pane.width = (width / 2) as usize;
                //     pane.height = height as usize;
                //     if pane.position.0 != 0 {
                //         pane.position.0 = pane.width;
                //     }
                // }
                pane_manager.window.resize(width, height);

            }

            Event::MouseWheel {x, y, direction , timestamp: _, .. } => {
                // mouse state does not update when not focused.
                // Need to fix that some how.
                // So that I can scroll both panes unfocused.
                // pane_manager.set_scroll_active_if_mouse_over((mouse_state.x(), mouse_state.y()));
                let direction_multiplier = match direction {
                    sdl2::mouse::MouseWheelDirection::Normal => 1,
                    sdl2::mouse::MouseWheelDirection::Flipped => -1,
                    sdl2::mouse::MouseWheelDirection::Unknown(x) => x as i32
                };
                if let Some(pane) = pane_manager.get_scroll_active_pane_mut() {
                    pane.scroller.scroll_x(pane.width, x * direction_multiplier.neg(), &mut pane.text_buffer, bounds);
                    pane.scroller.scroll_y(pane.height, y * direction_multiplier, &pane.text_buffer, bounds);
                }

            }
            _ => {}
        }
        if let Some(pane) = pane_manager.get_active_pane_mut() {
            pane.cursor_context.fix_cursor(&pane.text_buffer);
        }
    }
    actions
}


fn color_for_token(token: &RustSpecific, input_bytes: &[u8]) -> Color {
    match token {
        RustSpecific::Keyword(_) => PURPLE_TEXT_COLOR,
        RustSpecific::Token(Token::Comment(_)) =>  COMMENT_TEXT_COLOR,
        RustSpecific::Token(t) => {
            match t {
                Token::Comment(_) => {
                    COMMENT_TEXT_COLOR
                },
                Token::OpenBracket |
                Token::CloseBracket |
                Token::OpenParen |
                Token::CloseParen |
                Token::OpenCurly |
                Token::CloseCurly |
                Token::Comma => {
                    LIGHT_BLUE_TEXT_COLOR
                },
                Token::Atom((s,_e)) => {
                    if input_bytes[*s].is_ascii_uppercase() {
                        BLUE_TEXT_COLOR
                    } else {
                        STANDARD_TEXT_COLOR
                    }
                },
                Token::String(_) => {
                    GREEN_TEXT_COLOR
                },
                Token::Integer(_) | Token::Float(_) => {
                    ORANGE_TEXT_COLOR
                }
                _ => {
                    STANDARD_TEXT_COLOR
                }
            }
        }
    }
}


fn main() -> Result<(), String> {
    native::set_smooth_scroll();

    let window = Window {
        width: 1200,
        height: 800,
    };

    let sdl::SdlContext {
        mut event_pump,
        canvas,
        texture_creator,
        ttf_context,
        video,
    } = sdl::setup_sdl(window.width as usize, window.height as usize)?;

    let (mut texture, letter_width, letter_height) = sdl::draw_font_texture(&texture_creator, ttf_context)?;
    texture.set_color_mod(167, 174, 210);

    // If this gets dropped, the cursor resets.
    let cursor = sdl2::mouse::Cursor::from_system(SystemCursor::IBeam).unwrap();
    cursor.set();

    let clipboard = video.clipboard();

    let bounds = EditorBounds {
        editor_left_margin: 10,
        line_number_gutter_width : 20,
        letter_height,
        letter_width,
    };
    let scroller = Scroller {
        offset_y: 0,
        offset_x: 0,
        scroll_speed: 5,
    };

    let text = fs::read_to_string("/Users/jimmyhmiller/Documents/Code/Playground/rust/editor/src/main.rs").unwrap();


    let mut text_buffer = TextBuffer {
        chars: text.as_bytes().to_vec(),
        line_range: vec![],
        max_line_width_cache: 0,
        tokenizer: Tokenizer::new(),
    };
    text_buffer.parse_lines();

    let mut text_buffer2 = TextBuffer {
        chars: "rect canvas 2 1 2 4 1".as_bytes().to_vec(),
        line_range: vec![],
        max_line_width_cache: 0,
        tokenizer: Tokenizer::new(),
    };
    text_buffer2.parse_lines();

    let cursor_context = CursorContext {
        cursor: None,
        mouse_down: None,
        selection: None,
    };

    let mut fps = FpsCounter{
        start_time: Instant::now(),
        frame_count: 0,
        fps: 0,
    };

    let pane1 = Pane {
        name: "test_draw".to_string(),
        scroller: scroller.clone(),
        cursor_context: cursor_context.clone(),
        text_buffer: text_buffer2,
        position: (100, 100),
        width: 500,
        height: 500,
        active: true,
        mouse_pos: None,
        transaction_manager: TransactionManager::new(),
        editing_name: false,
        draw_commands: vec![],
    };

    let pane2 = Pane {
        name: "canvas".to_string(),
        scroller,
        cursor_context,
        text_buffer,
        position: (650, 100),
        width: 500,
        height: 500,
        active: false,
        mouse_pos: None,
        transaction_manager: TransactionManager::new(),
        editing_name: false,
        draw_commands: vec![],
    };



    let mut renderer = Renderer {
        canvas,
        texture,
        target: Rect::new(0, 0, 0, 0),
        bounds,
        system_cursor: cursor,
    };

    let mut pane_manager = PaneManager {
        panes: vec![pane1, pane2],
        active_pane: 0,
        scroll_active_pane: 0,
        window,
        dragging_pane: None,
        dragging_start: (0, 0),
        dragging_pane_start: (0, 0),
        resize_pane: None,
        resize_start: (0, 0),
        create_pane_activated: false,
        create_pane_start: (0, 0),
        create_pane_current: (0, 0),
    };

    let mut per_frame_actions = vec![];


    loop {
        renderer.set_cursor_ibeam();
        handle_transaction_pane(&mut pane_manager);
        handle_token_pane(&mut pane_manager);

        draw(&mut renderer, &mut pane_manager, &mut fps)?;
        let side_effects = handle_events(&mut event_pump, &mut pane_manager, &renderer.bounds, &clipboard);



        for side_effect in side_effects {
            handle_side_effects(&mut pane_manager, side_effect, &mut per_frame_actions);
        }

        let mut per_frame_action_results = vec![];
        for (i, per_frame_action) in per_frame_actions.iter_mut().enumerate() {
            per_frame_action_results.push(
                handle_per_frame_actions(i, &mut pane_manager, per_frame_action)
            );
        }
        for per_frame_action_result in per_frame_action_results {
            match per_frame_action_result {
                PerFrameActionResult::RemoveAction(i) => {
                    per_frame_actions.swap_remove(i);
                },
                PerFrameActionResult::Noop => {}
            }
        }


    }
}

fn handle_per_frame_actions(index: usize, pane_manager: &mut PaneManager, per_frame_action: &mut PerFrameAction) -> PerFrameActionResult {
    match per_frame_action {
        PerFrameAction::ReadCommand(output_pane_name, _child, non_blocking_reader) => {
            let output_pane = pane_manager.get_pane_by_name_mut(output_pane_name.clone());
            if output_pane.is_none() {
                return PerFrameActionResult::Noop
            }
            let output_pane = output_pane.unwrap();

            let max_attempts = 100;
            let mut i = 0;
            let mut buf = String::new();
            while !non_blocking_reader.is_eof() {
                if i > max_attempts {
                    break;
                }
                let length = non_blocking_reader.read_available_to_string(&mut buf).unwrap();
                if length == 0 {
                    break;
                }
                i += 1;
            }
            if buf.contains('\x0c') {
                output_pane.text_buffer.chars.clear();
                buf = buf[0..buf.chars().position(|x| x == '\x0c').unwrap()].to_string();
            }

            if !buf.is_empty(){
                // TODO: Have mode for if it should clear or have history
                // Or maybe just don't clear and always have scroll history
                // output_pane.text_buffer.chars.clear();
                output_pane.text_buffer.chars.extend(buf.as_bytes().to_vec());
                output_pane.text_buffer.parse_lines();

            }
            if non_blocking_reader.is_eof() {
                PerFrameActionResult::RemoveAction(index)
            } else {
                PerFrameActionResult::Noop
            }
        }
        PerFrameAction::DisplayError(_pane_name, _error_text) => {
            PerFrameActionResult::Noop
        }
    }
}

fn handle_side_effects(pane_manager: &mut PaneManager, side_effect: SideEffectAction, per_frame_actions: &mut Vec<PerFrameAction>) {
    match side_effect {
        SideEffectAction::Play(pane_name) => {
            // TODO: think about multiple panes of same name
            let pane_contents = {
                if let Some(pane) = pane_manager.get_pane_by_name_mut(pane_name.clone()) {
                    Some(from_utf8(&pane.text_buffer.chars).unwrap().to_string())
                } else {
                    None
                }
            };

            if let Some(pane_contents) = pane_contents {

                let output_pane_name = format!("{}-output", pane_name);
                let output_pane = pane_manager.get_pane_by_name_mut(output_pane_name.clone());
                match output_pane {

                    None => {
                        let existing_pane = pane_manager.get_pane_by_name(pane_name);
                        let position = {
                            match existing_pane {
                                Some(pane) =>  {
                                    (pane.position.0 + pane.width as i32 + 10, pane.position.1)
                                }
                                None => (0, 0)
                            }
                        };
                        pane_manager.create_pane_raw(output_pane_name.to_string(), position, 300, 300);
                    }
                    Some(output_pane) => {
                        // This causes a flash to happen
                        // Which is actually useful from a user experience perspective
                        // but it was unintentional.
                        // Makes me think something is taking longer to render
                        // than I thought.
                        // I guess it makes sense in some ways though.
                        // ls for example takes some amount of time,
                        // and then I have to fetch that data and render.
                        output_pane.text_buffer.chars.clear();
                        output_pane.text_buffer.parse_lines();
                    }
                }

                let current_running_action = per_frame_actions.iter().enumerate().find(|(_i, x)|
                    if let PerFrameAction::ReadCommand(name, _, _) = x {
                        *name == output_pane_name
                    } else {
                        false
                    }
                );
                if let Some((i, _)) = current_running_action {
                    let action = per_frame_actions.swap_remove(i);
                    if let PerFrameAction::ReadCommand(_name, mut child, _) = action {
                        // TODO: Get rid of this unwrap!

                        child.kill().unwrap();
                    }
                }

                let command = pane_contents;
                // need to handle error
                let child = Command::new("bash")
                    .arg("-c")
                    .arg(command)
                    .stdout(Stdio::piped())
                    .spawn();

                match child {
                    Ok(mut child) => {
                        let stdout = child.stdout.take().unwrap();
                        let noblock_stdout = NonBlockingReader::from_fd(stdout).unwrap();
                        per_frame_actions.push(PerFrameAction::ReadCommand(output_pane_name, child, noblock_stdout))
                    }
                    Err(e) => {
                        per_frame_actions.push(PerFrameAction::DisplayError(output_pane_name, format!("error {:?}", e)))
                    }
                }
            }
        },
    }
}

// I need to think about how to generalize this
fn handle_transaction_pane(pane_manager: &mut PaneManager) {
    let mut transaction_pane_index = None;
    for (i, pane) in pane_manager.panes.iter().enumerate() {
        if pane.name == "transaction_pane" {
            transaction_pane_index = Some(i);
        }
    }
    if Some(pane_manager.active_pane) != transaction_pane_index {

        if let Some(i) = transaction_pane_index  {
            let mut transaction_pane = pane_manager.remove(i);
            transaction_pane.text_buffer.chars.clear();

            if let Some(active_pane) = pane_manager.get_active_pane_mut() {
                let transaction_manager = &active_pane.transaction_manager;

                transaction_pane.text_buffer.chars.extend(format!("current: {}, pointer: {}\n",
                    transaction_manager.current_transaction,
                    transaction_manager.transaction_pointer).as_bytes());

                for transaction in active_pane.transaction_manager.transactions.iter() {
                    transaction_pane.text_buffer.chars.extend(format!("{:?}\n", transaction).as_bytes());
                }
            }

            transaction_pane.text_buffer.parse_lines();
            pane_manager.insert(i, transaction_pane);
        }
    }
}



fn handle_token_pane(pane_manager: &mut PaneManager) {
    let mut token_pane_index = None;
    for (i, pane) in pane_manager.panes.iter().enumerate() {
        if pane.name == "token_pane" {
            token_pane_index = Some(i);
        }
    }
    if Some(pane_manager.active_pane) != token_pane_index {

        if let Some(i) = token_pane_index  {
            let mut token_pane = pane_manager.remove(i);
            token_pane.text_buffer.chars.clear();

            // I am doing this every frame.
            // I really need a nice way of handling non-every frame events
            if let Some(active_pane) = pane_manager.get_active_pane_mut() {
                let tokenizer = &mut active_pane.text_buffer.tokenizer;
                while !tokenizer.at_end(&active_pane.text_buffer.chars) {
                    let token = tokenizer.parse_single(&active_pane.text_buffer.chars);
                    if let Some(token) = token {
                        token_pane.text_buffer.chars.extend(format!("{:?} ", token).as_bytes());
                        if matches!(token, Token::NewLine) {
                            token_pane.text_buffer.chars.extend(b"\n");
                        }
                    }
                }
                tokenizer.position = 0;
            }

            token_pane.text_buffer.parse_lines();
            pane_manager.insert(i, token_pane);
        }
    }
}

fn get_i32_from_token(token: &Token, chars: &[u8]) -> Option<i32> {
    if let Token::Integer((s, e)) = token {
        let string_value = from_utf8(&chars[*s..*e]).ok()?;
        let int_value: i32 = string_value.parse().ok()?;
        Some(int_value)
    } else {
        None
    }

}

#[derive(Debug, Clone)]
enum DrawCommand {
    Rect(Rect),
    RectOnPane(String, Rect),
    RectOnPaneAtLocation(String, i32, i32, i32, i32, i32),
}


// rect canvas 10 100 10 10 100 100

fn parse_rect(tokenizer: &mut Tokenizer, chars: &[u8]) -> Option<DrawCommand> {

    let line = tokenizer.get_line(chars);
    match line.len() {
        8 => Some(
            DrawCommand::Rect(Rect::new(
            get_i32_from_token(&line[1], chars)?,
            get_i32_from_token(&line[3], chars)?,
            get_i32_from_token(&line[5], chars)? as u32,
            get_i32_from_token(&line[7], chars)? as u32,
        ))),
        10 => {
            let (s, e) = match line[1] {
                Token::Atom((s, e)) => Some((s, e)),
                _ => return None,
            }?;

            let pane_name = from_utf8(&chars[s..e]).ok()?.to_string();
            Some(
                DrawCommand::RectOnPane(
                    pane_name,
                    Rect::new(
                        get_i32_from_token(&line[3], chars)?,
                        get_i32_from_token(&line[5], chars)?,
                        get_i32_from_token(&line[7], chars)? as u32,
                        get_i32_from_token(&line[9], chars)? as u32,
                    )
                )
            )
        }
        12 => {
            let (s, e) = match line[1] {
                Token::Atom((s, e)) => Some((s, e)),
                _ => return None,
            }?;
            let pane_name = from_utf8(&chars[s..e]).ok()?.to_string();
            Some(DrawCommand::RectOnPaneAtLocation(
               pane_name,
                get_i32_from_token(&line[3], chars)?,
                get_i32_from_token(&line[5], chars)?,
                get_i32_from_token(&line[7], chars)?,
                get_i32_from_token(&line[9], chars)?,
                get_i32_from_token(&line[11], chars)?,
            ))
        }

        _ => None

    }

    

    // let _ = tokenizer.parse_single(chars)?;
    // let next_token = tokenizer.parse_single(chars)?;
    // let x_token;
    // if let Token::Atom((start, end)) = next_token {
    //     pane_name = Some(from_utf8(&chars[start..end]).ok()?.to_string());
    //     let _ = tokenizer.parse_single(chars)?;
    //     x_token = tokenizer.parse_single(chars)?;
    // } else {
    //     x_token = next_token;
    // }
    // let x = get_i32_from_token(&x_token, chars)?;
    // let _ = tokenizer.parse_single(chars)?;
    // let y_token = tokenizer.parse_single(chars)?;
    // let y = get_i32_from_token(&y_token, chars)?;
    // let _ = tokenizer.parse_single(chars)?;
    // let width_token = tokenizer.parse_single(chars)?;
    // let width = get_i32_from_token(&width_token, chars)?;
    // let _ = tokenizer.parse_single(chars)?;
    // let height_token = tokenizer.parse_single(chars)?;
    // let height = get_i32_from_token(&height_token, chars)?;
    // Some((pane_name, Rect::new(x, y, width as u32, height as u32)))
}

// This happens every frame. Can I do better?
// I tokenize yet again here
// I probably want to cache the tokens
// and only retokenize on change
fn handle_draw_panes(pane_manager: &mut PaneManager, renderer: &mut Renderer) -> Result<(), String> {
    let mut panes_with_rects = vec![];
    for pane in pane_manager.panes.iter_mut() {
        if !pane.name.ends_with("_draw") {
            continue;
        }
        renderer.set_draw_color(CURSOR_COLOR);
        let tokenizer = &mut pane.text_buffer.tokenizer;
        let chars = &pane.text_buffer.chars;
        while !tokenizer.at_end( chars) {
            if let Some(Token::Atom((start, end))) = tokenizer.parse_single(chars) {
                let atom = &chars[start..end];
                if atom == b"rect" {
                    if let Some(draw_command) = parse_rect(tokenizer, chars) {
                        match draw_command {
                            DrawCommand::Rect(rect) => renderer.draw_rect(&rect)?,
                            DrawCommand::RectOnPane(_, _) => panes_with_rects.push(draw_command),
                            DrawCommand::RectOnPaneAtLocation(_, _, _, _, _, _) => panes_with_rects.push(draw_command),
                        }
                        
                    }

                }
            }
        }

        tokenizer.position = 0;
    }
    
    // TODO: Get rid of these clones
    for draw_command in panes_with_rects.iter() {
        match draw_command {
            DrawCommand::RectOnPane(pane_name, _rect) => {
                if let Some(pane) = pane_manager.get_pane_by_name_mut(pane_name.clone()) {
                    pane.draw_commands.push(draw_command.clone())
                }
            }
            DrawCommand::RectOnPaneAtLocation(pane_name, _, _, _, _, _) => {
                if let Some(pane) = pane_manager.get_pane_by_name_mut(pane_name.clone()) {
                    pane.draw_commands.push(draw_command.clone())
                }
            }
            DrawCommand::Rect(_) => {}
        }
    }

    Ok(())
}


