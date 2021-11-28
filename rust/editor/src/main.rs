use std::{cmp::{max, min}, convert::TryInto, fs, ops::{Index, IndexMut, Neg, RangeBounds}, str::from_utf8, time::Instant};
use std::fmt::Debug;

use sdl2::{event::{Event, WindowEvent}, keyboard::{Keycode, Mod}, mouse::SystemCursor, pixels::Color, rect::Rect, render::{Canvas, Texture}, video::{self}};


mod native;
mod sdl;


// I really want so debugging panels.
// Should probably invest in that.
// Make it work automatically with Debug.

#[derive(Debug)]
struct Transaction {
    transaction_number: usize,
    parent_pointer: usize,
    action: EditAction,
}

#[derive(Debug)]
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
    }

}



#[derive(Debug)]
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
            EditAction::Insert((start, end), _text_to_insert) => {
                let mut new_position = Cursor(*start, *end);
                new_position.move_right(text_buffer);
                text_buffer.remove_char(new_position);
                new_position.move_left(text_buffer);
                cursor_context.set_cursor(new_position);
            },
            EditAction::Delete((start, end), text_to_delete) => {
                let mut new_position = Cursor(*start, *end);
                new_position.move_left(text_buffer);
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
        self.1 = min(self.1, line_length(text_buffer[new_line]));
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
        self.1 = min(self.1, line_length(text_buffer[self.0]));
        EditAction::CursorPosition(*self)

        // Need to use this output to deal with scrolling down
    }


    fn move_left(&mut self, text_buffer: &TextBuffer) -> EditAction  {
        let Cursor(cursor_line, cursor_column) = *self;
        if cursor_column == 0 && cursor_line != 0 {
            let previous_line = text_buffer[cursor_line - 1];
            let length = line_length(previous_line);
            *self = Cursor(cursor_line.saturating_sub(1), length);
        } else {
            *self = Cursor(cursor_line, cursor_column.saturating_sub(1));
        }
        // Could need to deal with scrolling left
        EditAction::CursorPosition(*self)
    }

    fn move_right(&mut self, text_buffer: &TextBuffer) -> EditAction  {
        let Cursor(cursor_line, cursor_column) = *self;
        if let Some((line_start, line_end)) = text_buffer.get_line(cursor_line) {
            let length = line_length((*line_start, *line_end));
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
        self.line_number_digits(text_buffer) * self.letter_width as usize + self.line_number_gutter_width + self.editor_left_margin + self.letter_width as usize
    }
}


// This probably belongs in scroller
fn text_space_from_screen_space(scroller: &Scroller, mut x: i32, y: i32, text_buffer: &TextBuffer) -> Option<Cursor> {
    // Slightly off probably due to rounding.
    // println!("{}", y as f32 / letter_height as f32);

    // Probably should move some/all of this to the scroller.
    let EditorBounds {letter_height, letter_width, ..} = scroller.bounds;
    let bounds = &scroller.bounds;
    let line_number_padding = bounds.line_number_padding(text_buffer) as i32;

    let line_number : usize = (((y as f32 / letter_height as f32) + (scroller.line_fraction() as f32 / bounds.letter_height as f32)).floor() as i32 + scroller.lines_above_fold() as i32).try_into().unwrap();
    if x < line_number_padding && x > line_number_padding - 20  {
        x = line_number_padding;
    }
    if x < line_number_padding {
        return None;
    }
    let mut column_number : usize = ((x - line_number_padding as i32) / letter_width as i32).try_into().unwrap();

    if let Some((line_start, line_end)) = text_buffer.get_line(line_number) {
        if column_number > line_end - line_start {
            column_number = text_buffer[line_number].1 - text_buffer[line_number].0;
        }
        return Some(Cursor(line_number, column_number));
    }
    if line_number > text_buffer.line_count() {
        if let Some((start, end)) = text_buffer.last_line() {
           return Some(Cursor(text_buffer.line_count() - 1, line_length((*start, *end))));
        }

    }
    None
}




// Move to TextBuffer
fn line_length(line: (usize, usize)) -> usize {
    line.1 - line.0
}




// TODO:
// Add some spacing between letters!
// Need to add a real parser or I can try messing with tree sitter.
// It would be pretty cool to add a minimap
// Also cool to just add my own scrollbar.
// Add horizontal scroll
// Add rendering bounds
// Add multiple panes
// Need to handle transactions per pane
// Need to handle active vs inactive pane

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
                Some((start, end)) => {
                    if cursor_column > *end {
                        self.cursor = Some(Cursor(cursor_line, line_length((*start, *end))));
                    }
                }
                None => {
                    self.cursor = text_buffer.last_line().map(|(line, column)| Cursor(*line, *column));
                }
            }
        }
    }
    fn mouse_down(&mut self) {
        self.mouse_down = self.cursor;
    }

    fn clear_mouse_down(&mut self) {
        self.mouse_down = None;
    }

    fn move_cursor_from_screen_position(&mut self, scroller: &Scroller, x: i32, y: i32, text_buffer: &TextBuffer) {
        self.cursor = text_space_from_screen_space(scroller, x, y, text_buffer);
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
}

#[derive(Debug, Clone)]
struct Scroller {
    offset_x: i32,
    offset_y: i32,
    scroll_speed: i32,
    bounds: EditorBounds,
}

impl Scroller {
    fn scroll_y(&mut self, height: usize, amount: i32, text_buffer: &TextBuffer) {
        if !self.at_end(height, text_buffer) || amount < 0 {
            self.offset_y += amount * self.scroll_speed;
        }
        // Need to set offsety if we are at the end
        // so that it stays consistent
        self.offset_y = max(0, self.offset_y);
    }

    fn scroll_x(&mut self, width: usize, amount: i32, text_buffer: &TextBuffer) {
        self.offset_x += amount * self.scroll_speed;
        self.offset_x = max(0, self.offset_x);
        let max_right = text_buffer.max_line_width() * self.bounds.letter_width;
        // I probably need something in pane to get this right.
        let viewable_width =  width - self.bounds.letter_width*2 - self.bounds.line_number_padding(text_buffer);
        if self.offset_x + viewable_width as i32 > max_right as i32{
            self.offset_x = (max_right.saturating_sub(viewable_width)) as i32;
        }
        // self.offset_x = min(self.offset_x, (text_buffer.max_line_width() * self.bounds.letter_width) as i32);
    }
    // Will need to do scroll fraction for x too.
    fn scroll_x_character(&self) -> usize {
        self.offset_x as usize / self.bounds.letter_width
    }

    fn viewing_lines(&self, height: usize) -> usize {
       height / self.bounds.letter_height
    }

    fn lines_above_fold(&self) -> usize {
        self.offset_y as usize / self.bounds.letter_height as usize
    }

    fn at_end(&self, height: usize, text_buffer: &TextBuffer) -> bool {
        self.lines_above_fold() + self.viewing_lines(height) >= text_buffer.line_count() + 3
    }

    fn to_the_top(&mut self) {
        self.offset_y = 0;
    }

    fn line_fraction(&self) -> usize {
        self.offset_y as usize % self.bounds.letter_height as usize
    }
}


struct Renderer<'a> {
    canvas: Canvas<video::Window>,
    texture: Texture<'a>,
    target: Rect,
    bounds: EditorBounds,
}

impl<'a> Renderer<'a> {
    fn set_draw_color(&mut self, color: Color) {
        self.canvas.set_draw_color(color);
    }

    fn clear(&mut self) {
        self.canvas.clear();
    }

    fn set_color_mod(&mut self, red: u8, green: u8, blue: u8) {
        self.texture.set_color_mod(red, green, blue);
    }

    fn set_initial_rendering_location(&mut self, scroller: &Scroller) {
        self.target = Rect::new(scroller.bounds.editor_left_margin as i32, (scroller.line_fraction() as i32).neg(),
               scroller.bounds.letter_width as u32, scroller.bounds.letter_height as u32)
    }

    fn copy(&mut self, source: &Rect) -> Result<(), String> {
        self.canvas.copy(&self.texture, *source, self.target)
    }

    fn fill_rect(&mut self, rect: &Rect) -> Result<(), String> {
        self.canvas.fill_rect(*rect)
    }

    fn move_right(&mut self, padding: i32) {
        self.target.set_x(self.target.x() + padding);
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
        self.draw_string(&format!("fps: {}", current_fps))
    }

    fn draw_column_line(&mut self, pane_manager: &mut PaneManager) -> Result<(), String> {
        self.target = Rect::new(pane_manager.window.width - (self.bounds.letter_width * 22) as i32, pane_manager.window.height-self.bounds.letter_height as i32, self.bounds.letter_width as u32, self.bounds.letter_height as u32);
        if let Some(Cursor(cursor_line, cursor_column)) = pane_manager.get_active_pane().cursor_context.cursor {
            self.draw_string( &format!("Line {}, Column {}", cursor_line, cursor_column))?;
        }
        Ok(())
    }

    // This makes me question what the responsibilities of this renderer should be.
    // If it is drawing things like the cursor, the line numbers, the text, the column line, etc,
    // then we need multiple of these, but now we are singularly presenting the canvas.
    // Feels a bit strange.
    // Also, the bounds now would need to change per drawing area, but we set the bounds.
    // Is there a way to make this not own the canvas?
    // Or should I move these drawing methods out to something like a "Pane" and just pass the renderer?
    // Really not sure yet.
    // After some thought, I definitely think pane is the correct place for these to live.
    fn present(&mut self) {
        self.canvas.present();
    }



}

#[derive(Debug, Clone)]
struct Pane {
    scroller: Scroller,
    cursor_context: CursorContext,
    text_buffer: TextBuffer,
    position: (usize, usize),
    width: usize,
    height: usize,
    active: bool,
}

// Thoughts:
// I don't have any notion of bounds to stop rendering at.
// I need to think about that here.
impl Pane {

    fn adjust_position(&self, x: i32, y: i32) -> (i32, i32) {
        (x.saturating_sub(self.position.0 as i32), y.saturating_sub(self.position.1 as i32))
    }

    fn max_characters_per_line(&self) -> usize {
        (self.width - self.scroller.bounds.line_number_padding(&self.text_buffer) - self.scroller.bounds.letter_width) / self.scroller.bounds.letter_width as usize
    }

    fn max_lines_per_page(&self) -> usize {
        self.height / self.scroller.bounds.letter_height as usize
    }

    fn draw(&mut self, renderer: &mut Renderer) -> Result<(), String> {
        let editor_left_margin = self.scroller.bounds.editor_left_margin;
        let line_number_padding = self.scroller.bounds.line_number_padding(&self.text_buffer);
        let line_number_digits = self.scroller.bounds.line_number_digits(&self.text_buffer);


        // Really need to think about spit vs freefloating
        if self.position.0 != 0 {
            renderer.set_draw_color(Color::RGBA(36, 39, 55, 255));
            renderer.fill_rect(&Rect::new(self.position.0 as i32, self.position.1 as i32, 2, self.height as u32))?;
        }


        renderer.set_initial_rendering_location(&self.scroller);
        renderer.move_right(self.position.0 as i32);
        renderer.move_down(self.position.1 as i32);

        let number_of_lines = min(self.scroller.lines_above_fold() + self.max_lines_per_page(), self.text_buffer.line_count());


        for line in self.scroller.lines_above_fold() as usize..number_of_lines {
            renderer.set_color_mod(167, 174, 210);
            renderer.set_x(editor_left_margin as i32 + self.position.0 as i32);

            self.draw_line_numbers(renderer, line_number_digits, line)?;
            // This really shouldn't be in this loop.
            // I could instead figure out if it is visbible.
            self.draw_cursor(renderer, line, line_number_padding)?;
            self.draw_selection(renderer, line, line_number_padding)?;

            self.draw_code(renderer, line)?;

            renderer.move_down_one_line();
        }
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
                let cursor_x = cursor.1 as i32  * renderer.bounds.letter_width as i32 + line_number_padding as i32 + self.position.0 as i32;
                let cursor_y = renderer.target.y();

                renderer.set_draw_color(Color::RGBA(255, 204, 0, 255));
                renderer.fill_rect(&Rect::new(cursor_x as i32, cursor_y as i32, 2, renderer.bounds.letter_height as u32))?
            }
        }
        Ok(())
    }

    fn draw_selection(&mut self, renderer: &mut Renderer, line: usize, line_number_padding: usize) -> Result<(), String> {
        if let Some(((start_line, start_column), (end_line, end_column))) = self.cursor_context.selection {
            if line >= start_line && line <= end_line {
                let mut start_x = if line == start_line {
                    start_column * renderer.bounds.letter_width + line_number_padding
                } else {
                    line_number_padding
                };
                let width = if start_line == end_line {
                    (end_column - start_column) * renderer.bounds.letter_width
                } else if line == end_line {
                    end_column * renderer.bounds.letter_width as usize
                } else if line == start_line {
                    (line_length(self.text_buffer[line]) - start_column) * renderer.bounds.letter_width as usize
                } else {
                    line_length(self.text_buffer[line]) * renderer.bounds.letter_width
                }.try_into().unwrap();

                start_x += self.position.0;

                let start_y = renderer.target.y();
                renderer.set_draw_color(Color::RGBA(65, 70, 99, 255));
                // Need to deal with last line.
                renderer.fill_rect(&Rect::new(start_x as i32, start_y, width, renderer.bounds.letter_height as u32))?
            }

        };
        Ok(())
    }


    fn draw_code(&mut self, renderer: &mut Renderer, line: usize) -> Result<(), String> {
        let (start, end) = self.text_buffer[line];
        let start = min(start + self.scroller.scroll_x_character(), end);
        let end = min(end, start + self.max_characters_per_line());
        renderer.draw_string(from_utf8(self.text_buffer.chars[start..end].as_ref()).unwrap())
    }

    fn is_mouse_over(&self, (x, y): (i32, i32)) -> bool {
        (x as usize) > self.position.0 &&
        (x as usize) < self.position.0 + self.width &&
        (y as usize) > self.position.1 &&
        (y as usize) < self.position.1 + self.height
    }

    fn set_active(&mut self, active: bool) {
        self.active = active;
    }


}

fn draw(renderer: &mut Renderer, pane_manager: &mut PaneManager, fps: &mut FpsCounter) -> Result<(), String> {
    renderer.set_draw_color(Color::RGBA(42, 45, 62, 255));
    renderer.clear();
    for pane in pane_manager.panes.iter_mut() {
        pane.draw(renderer)?;
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
        EditAction::Delete((cursor_line, cursor_column),
                            from_utf8(&[result]).unwrap().to_string())
    }

    fn max_line_width(&self) -> usize {
        self.line_range.iter().map(|(start,end)| end - start).max().unwrap_or(0)
    }
}



struct PaneManager {
    panes: Vec<Pane>,
    active_pane: usize,
    scroll_active_pane: usize,
    window: Window,
}

impl PaneManager {

    fn set_scroll_active_if_mouse_over(&mut self, mouse_pos: (i32, i32)) {
        let mut new_active_pane = self.scroll_active_pane;
        for (i, pane) in self.panes.iter().enumerate() {
            if pane.is_mouse_over(mouse_pos) {
                new_active_pane = i;
            }
        }
        if new_active_pane != self.scroll_active_pane {
            self.scroll_active_pane = new_active_pane;
        }
    }
    fn set_active_from_click_coords(&mut self, mouse_pos: (i32, i32)) {
        let old_active = self.active_pane;
        let mut new_active_pane = self.active_pane;
        for (i, pane) in self.panes.iter().enumerate() {
            if pane.is_mouse_over(mouse_pos) {
                new_active_pane = i;
            }
        }
        if new_active_pane != self.active_pane {
            self.active_pane = new_active_pane;
            self.panes.get_mut(self.active_pane).unwrap().set_active(true);
            self.panes.get_mut(old_active).unwrap().set_active(false);
        }
    }


    fn get_active_pane_mut(&mut self) -> &mut Pane {
        &mut self.panes[self.active_pane]
    }

    fn get_scroll_active_pane_mut(&mut self) -> &mut Pane {
        &mut self.panes[self.scroll_active_pane]
    }


    fn get_active_pane(&mut self) -> &Pane {
        &self.panes[self.active_pane]
    }
}


fn handle_events(event_pump: &mut sdl2::EventPump,
                 transaction_manager: &mut TransactionManager,
                 pane_manager: &mut PaneManager) {
    let mut is_text_input = false;

    // let text_buffer = &mut pane.text_buffer;
    // let cursor_context = &mut pane.cursor_context;
    // let scroller = &mut pane.scroller;

    // This whole way of handling things is wrong.
    // We probably need a pane manager.
    // Maybe good chance for a pun?


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
                        let pane = pane_manager.get_active_pane_mut();
                        pane.cursor_context.move_up(&pane.text_buffer);
                    },
                    (Keycode::Down, _) => {
                        let pane = pane_manager.get_active_pane_mut();
                        pane.cursor_context.move_down(&pane.text_buffer);
                    },
                    (Keycode::Left, _) => {
                        let pane = pane_manager.get_active_pane_mut();
                        pane.cursor_context.move_left(&pane.text_buffer);
                    },
                    (Keycode::Right, _) => {
                        let pane = pane_manager.get_active_pane_mut();
                        pane.cursor_context.move_right(&pane.text_buffer);
                    },
                    (Keycode::Backspace, _) => {
                        let pane = pane_manager.get_active_pane_mut();
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
                            // BUG:
                            // This is not working quite correctly.
                            // Delete at the end of a line with a non-empty above.
                            // Try undoing the delete and then redoing.
                            // The cursor is all wrong.
                            let mut old_cursor = current_cursor.clone();
                            // We do this move_left first, because otherwise we might end up at the end
                            // of the new line we formed from the deletion, rather than the old end of the line.
                            let cursor_action = old_cursor.move_left(&pane.text_buffer);
                            let action = pane.text_buffer.remove_char(current_cursor);

                            transaction_manager.add_action(action);
                            transaction_manager.add_action(cursor_action);


                            pane.cursor_context.set_cursor(old_cursor);
                        }
                    }
                    (Keycode::Return, _) => {
                        let pane = pane_manager.get_active_pane_mut();
                        // refactor to be better
                        let action = pane.cursor_context.handle_insert(&[b'\n'], &mut pane.text_buffer);
                        transaction_manager.add_action(action);
                        pane.cursor_context.start_of_line();
                    },


                    (Keycode::Z, key_mod) => {
                        let pane = pane_manager.get_active_pane_mut();
                        if key_mod == Mod::LGUIMOD || keymod == Mod::RGUIMOD {
                            transaction_manager.undo(&mut pane.cursor_context, &mut pane.text_buffer);
                        } else if key_mod == (Mod::LSHIFTMOD | Mod::LGUIMOD) {
                            transaction_manager.redo(&mut pane.cursor_context, &mut pane.text_buffer);
                        } else {
                            is_text_input = true
                        }

                    },

                    (Keycode::O, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        let pane = pane_manager.get_active_pane_mut();
                        let text = native::open_file_dialog();
                        if let Some(text) = text {
                            pane.text_buffer.set_contents(text.as_bytes());
                            pane.scroller.to_the_top();
                        }
                    }
                    (Keycode::A, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        let pane = pane_manager.get_active_pane_mut();
                        // This is super ugly, fix.
                        pane.cursor_context.set_selection(((0,0), (pane.text_buffer.line_count()-1, line_length(pane.text_buffer[pane.text_buffer.line_count()-1]))));
                    }

                    _ => is_text_input = true
                }
            }
            Event::TextInput{text, ..} => {
                let pane = pane_manager.get_active_pane_mut();
                if is_text_input {
                    // TODO: Replace with actually deleting the selection.
                    pane.cursor_context.clear_selection();

                    let action = pane.cursor_context.handle_insert(text.as_bytes(), &mut pane.text_buffer);
                    transaction_manager.add_action(action);
                }
            }

            // Need to make selection work
            // Which probably means changing cursor representation

            Event::MouseButtonDown { x, y, .. } => {
                pane_manager.set_active_from_click_coords((x, y));
                let pane = pane_manager.get_active_pane_mut();
                let (x, y) = pane.adjust_position(x, y);
                pane.cursor_context.move_cursor_from_screen_position(&pane.scroller, x, y, &pane.text_buffer);
                pane.cursor_context.mouse_down();
                pane.cursor_context.clear_selection();
            }

            Event::MouseMotion{x, y, .. } => {

                // TODO:
                // distinguish between active and scrolling.
                // Mouse over is enough for scrolling, but not for active.
                pane_manager.set_scroll_active_if_mouse_over((x, y));


                let pane = pane_manager.get_active_pane_mut();
                let (x, y) = pane.adjust_position(x, y);


                if let Some(Cursor(start_line, mut start_column)) = pane.cursor_context.mouse_down {
                    pane.cursor_context.move_cursor_from_screen_position(&pane.scroller, x, y, &pane.text_buffer);
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

            Event::MouseButtonUp{x, y, ..} => {
                let pane = pane_manager.get_active_pane_mut();
                let (x, y) = pane.adjust_position(x, y);
                if let Some(Cursor(start_line, mut start_column)) = pane.cursor_context.mouse_down {
                    pane.cursor_context.move_cursor_from_screen_position(&pane.scroller, x, y, &pane.text_buffer);
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
            // Continuous resize in sdl2 is a bit weird
            // Would need to watch events or something
            Event::Window {win_event: WindowEvent::Resized(width, height), ..} => {

                for pane in pane_manager.panes.iter_mut() {
                    pane.width = (width / 2) as usize;
                    pane.height = height as usize;
                    if pane.position.0 != 0 {
                        pane.position.0 = pane.width;
                    }
                }
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
                let pane = pane_manager.get_scroll_active_pane_mut();
                pane.scroller.scroll_x(pane.width, x * direction_multiplier * -1, &pane.text_buffer);
                pane.scroller.scroll_y(pane.height, y * direction_multiplier, &pane.text_buffer);
            }
            event => println!("{:?}", event)
        }
        let pane = pane_manager.get_active_pane_mut();
        pane.cursor_context.fix_cursor(&pane.text_buffer);
    }
}

fn main() -> Result<(), String> {
    native::set_smooth_scroll();

    let window = Window {
        width: 1200,
        height: 800,
    };

    let (ttf_context, canvas, mut event_pump, texture_creator) = sdl::setup_sdl(window.width as usize, window.height as usize)?;
    let (mut texture, letter_width, letter_height) = sdl::draw_font_texture(&texture_creator, ttf_context)?;
    texture.set_color_mod(167, 174, 210);

    // For some reason as soon as I extract this into a function, it doesn't work.
    let cursor = sdl2::mouse::Cursor::from_system(SystemCursor::IBeam).unwrap();
    cursor.set();


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
        bounds: bounds.clone(),
    };

    let text = fs::read_to_string("/Users/jimmyhmiller/Documents/Code/Playground/rust/editor/src/main.rs").unwrap();

    let mut text_buffer = TextBuffer {
        chars: text.as_bytes().to_vec(),
        line_range: vec![]
    };
    text_buffer.parse_lines();

    let cursor_context = CursorContext {
        cursor: None,
        mouse_down: None,
        selection: None,
    };
    let mut transaction_manager = TransactionManager::new();

    let mut fps = FpsCounter{
        start_time: Instant::now(),
        frame_count: 0,
        fps: 0,
    };

    let pane1 = Pane {
        scroller: scroller.clone(),
        cursor_context: cursor_context.clone(),
        text_buffer: text_buffer.clone(),
        position: (0, 0),
        width: 600,
        height: 800,
        active: true,
    };

    let pane2 = Pane {
        scroller,
        cursor_context,
        text_buffer,
        position: (600, 0),
        width: 600,
        height: 800,
        active: false,
    };

    let mut renderer = Renderer {
        canvas,
        texture,
        target: Rect::new(0, 0, 0, 0),
        bounds,
    };

    let mut pane_manager = PaneManager {
        panes: vec![pane1, pane2],
        active_pane: 0,
        scroll_active_pane: 0,
        window,
    };


    loop {
        draw(&mut renderer, &mut pane_manager, &mut fps)?;
        handle_events(&mut event_pump, &mut transaction_manager, &mut pane_manager);
    }
}


