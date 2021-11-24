use std::{cmp::{max, min}, convert::TryInto, fs, ops::Neg, time::Instant};
use std::fmt::Debug;

use native_dialog::FileDialog;
use sdl2::{event::*, keyboard::*, mouse::{SystemCursor}, pixels::Color, rect::Rect, render::*, video::{self, WindowContext}};



fn setup_sdl(window: &Window) -> Result<(sdl2::ttf::Sdl2TtfContext, Canvas<video::Window>, sdl2::EventPump, TextureCreator<WindowContext>), String> {
    let sdl_context = sdl2::init()?;
    let ttf_context = sdl2::ttf::init().map_err(|e| e.to_string())?;
    let sdl_window = sdl_context
        .video()?
        .window("Example", window.width as u32, window.height as u32)
        .resizable()
        .build()
        .unwrap();
        
    let cursor = sdl2::mouse::Cursor::from_system(SystemCursor::IBeam)
        .map_err(|err| format!("failed to load cursor: {}", err))?;
    cursor.set();
    
    let canvas: Canvas<video::Window> = sdl_window
        .into_canvas()
        .present_vsync()
        .build()
        .unwrap();

    let event_pump = sdl_context.event_pump()?;

    let texture_creator = canvas.texture_creator();

    Ok((ttf_context, canvas, event_pump, texture_creator))
}


fn set_smooth_scroll() {
    unsafe {
        use cocoa_foundation::foundation::NSUserDefaults;
        use cocoa_foundation::foundation::NSString;
        use cocoa_foundation::base::nil;
        let defaults = cocoa_foundation::base::id::standardUserDefaults();
        let key = NSString::alloc(nil).init_str("AppleMomentumScrollSupported");
        defaults.setBool_forKey_(cocoa_foundation::base::YES, key)
    }
}


// This wouldn't work for multiple cursors
// But could if I do transactions separately

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

    fn add_action_pair(&mut self, action: EditAction, cursor_action: EditAction) {

        if let EditAction::Insert(_, s) = &action {
            // This does not handle multiple spaces which we probably want to group
            if s.trim().is_empty() {
                self.current_transaction += 1;
            }
        }
        if let EditAction::Delete(_, _) = &action {
            // Delete isn't quite right. I kind of want strings
            // of deletes to coalesce.
            // Maybe I should have some compact functions?
            self.current_transaction += 1;
        }
        self.add_action(action);
        self.add_action(cursor_action);
    }

    fn add_action(&mut self, action: EditAction) {

        self.transactions.push(Transaction {
            transaction_number: self.current_transaction,
            parent_pointer: self.transaction_pointer,
            action,
        });

        self.transaction_pointer = self.transactions.len() - 1;
    }

    fn undo(&mut self, cursor_context: &mut CursorContext, chars: &mut Vec<u8>, line_range: &mut Vec<(usize, usize)>) {
        if self.transaction_pointer == 0 {
           return;
        }
        let last_transaction = self.transactions[self.transaction_pointer].transaction_number;
        let mut i = self.transaction_pointer;
        while self.transactions[i].transaction_number == last_transaction {
            self.transactions[i].action.undo(cursor_context, chars, line_range);

            if i == 0 {
                break;
            }
            i = self.transactions[i].parent_pointer;
        }
        self.transaction_pointer = i;

    }

    // How do I redo?

    fn redo(&mut self, cursor_context: &mut CursorContext, chars: &mut Vec<u8>, line_range: &mut Vec<(usize, usize)>) {

        if self.transaction_pointer == self.transactions.len() - 1 {
            return;
        }

        let last_undo = self.transactions.iter()
            .rev()
            .filter(|t| t.parent_pointer == self.transaction_pointer)
            .next();
        
        // My cursor is one off! But this seems to be close to correct for the small cases I tried.
        if let Some(Transaction{ transaction_number: last_transaction, ..}) = last_undo {
            for (i, transaction) in self.transactions.iter().enumerate() {
                if transaction.transaction_number == *last_transaction {
                    self.transactions[i].action.redo(cursor_context, chars, line_range);
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
}

// Copilot talked about an apply function
// That is an interesting idea
impl EditAction  {
    fn undo(&self, cursor_context: &mut CursorContext, chars: &mut Vec<u8>, line_range: &mut Vec<(usize, usize)>) {
        match self {
            EditAction::Insert((start, end), _text_to_insert) => {
                let mut new_position = Cursor(*start, *end);
                new_position.move_right(line_range);
                handle_delete(new_position, chars, line_range);
                new_position.move_left(line_range);
                cursor_context.set_cursor(new_position);
            },
            EditAction::Delete((start, end), text_to_delete) => {
                let mut new_position = Cursor(*start, *end);
                new_position.move_left(line_range);
                handle_insert(new_position, text_to_delete.as_bytes(), chars, line_range);
                new_position.move_right(line_range);
                cursor_context.set_cursor(new_position);
            },
            EditAction::CursorPosition(old_cursor) => {
                cursor_context.set_cursor(*old_cursor);
            }
        }
    }

    fn redo(&self, cursor_context: &mut CursorContext, chars: &mut Vec<u8>, line_range: &mut Vec<(usize, usize)>) {

        match self {
            EditAction::Insert((start, end), text_to_insert) => {
                handle_insert( Cursor(*start, *end), text_to_insert.as_bytes(), chars, line_range);
            },
            EditAction::Delete((start, end), _text_to_delete) => {
                handle_delete( Cursor(*start, *end), chars, line_range);
            },
            EditAction::CursorPosition(new_cursor) => {
                cursor_context.set_cursor(*new_cursor);
            }
        }
    }
}


#[derive(Debug, Copy, Clone)]
struct Cursor(usize, usize);

impl Cursor {
 
    fn start_of_line(&mut self) {
        self.1 = 0;
    }
    
    fn set_position(&mut self, cursor: Cursor) {
        self.0 = cursor.0;
        self.1 = cursor.1;
    }
    
    fn move_up(&mut self, line_range: &Vec<(usize, usize)>) -> EditAction {
        let Cursor(cursor_line, cursor_column) = *self;
        let new_line = cursor_line.saturating_sub(1);
        self.0 = new_line;
        self.1 = min(cursor_column, line_length(line_range[new_line]));
        EditAction::CursorPosition(*self)
        // *self = Cursor(new_line, min(cursor_column, line_length(line_range[new_line])));

        // Need to deal with line_fraction
        // Need to use this output to deal with scrolling up
        // if cursor_line < lines_above_fold {
        //     offset_y -= letter_height as i32;
        // }
    }

    fn move_down(&mut self, line_range: &Vec<(usize, usize)>) -> EditAction  {
        let Cursor(cursor_line, cursor_column) = *self;
        let new_line = cursor_line + 1;
        if let Some(line) = line_range.get(new_line) {
            *self = Cursor(new_line, min(cursor_column, line_length(*line)));
        }   
        EditAction::CursorPosition(*self)

        // Need to use this output to deal with scrolling down
    }

    
    fn move_left(&mut self, line_range: &Vec<(usize, usize)>) -> EditAction  {
        let Cursor(cursor_line, cursor_column) = *self;
        if cursor_column == 0 && cursor_line != 0 {
            let previous_line = line_range[cursor_line - 1];
            let length = line_length(previous_line);
            *self = Cursor(cursor_line.saturating_sub(1), length);
        } else {
            *self = Cursor(cursor_line, cursor_column.saturating_sub(1));
        }
        // Could need to deal with scrolling left
        EditAction::CursorPosition(*self)
    }

    fn move_right(&mut self, line_range: &Vec<(usize, usize)>) -> EditAction  {
        let Cursor(cursor_line, cursor_column) = *self;
        if let Some((line_start, line_end)) = line_range.get(cursor_line) {
            let length = line_length((*line_start, *line_end));
            if cursor_column >= length {
                if cursor_line + 1 < line_range.len() {
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

// There is no length here.
// We should do that.
// Also consider a direction
// Finally line_range should probably have some methods for doing what we do here.
fn handle_delete(cursor: Cursor, chars: &mut Vec<u8>, line_range: &mut Vec<(usize, usize)>) -> Option<EditAction> {
    let Cursor(cursor_line, cursor_column)= cursor;

    if let Some((line_start, _line_end)) = line_range.get(cursor_line) {
        let char_pos = (line_start + cursor_column).saturating_sub(1);
        
        if chars.is_empty() || char_pos >= chars.len() {
            return None;
        }
        let result = chars.remove(char_pos);
                                        
        line_range[cursor_line].1 = line_range[cursor_line].1.saturating_sub(1);
        let mut line_erased = false;
        if cursor_column == 0 {
            line_range[cursor_line - 1] = (line_range[cursor_line - 1].0, line_range[cursor_line].1);
            line_range.remove(cursor_line);
            line_erased = true;
        }

        for mut line in line_range.iter_mut().skip(cursor_line + if line_erased { 0} else {1}) {
            line.0 = line.0.saturating_sub(1);
            line.1 = line.1.saturating_sub(1);
        }
        return Some(EditAction::Delete((cursor_line, cursor_column), std::str::from_utf8(&[result]).unwrap().to_string()));
    }
    None
}

// Need to think about paste
fn handle_insert(cursor: Cursor, to_insert: &[u8], chars: &mut Vec<u8>, line_range: &mut Vec<(usize, usize)>) -> EditAction {
    // println!("Insert! {:?} {:?}", cursor, to_insert);
    let Cursor(cursor_line, cursor_column) = cursor;
    let line_start = line_range[cursor_line].0;
    let char_pos = line_start + cursor_column;
    chars.splice(char_pos..char_pos, to_insert.to_vec());

    let mut lines_to_skip = 1;
    if to_insert == [b'\n'] {
        let (start, end) = line_range[cursor_line];
        if char_pos >= end && cursor_column != 0 {
            line_range.insert(cursor_line + 1, (char_pos+1, char_pos+1));
        } else if cursor_column == 0 {
            line_range.splice(cursor_line..cursor_line+1, [(start,char_pos), (start+1, end+1)]);
        } else {
            line_range.splice(cursor_line..cursor_line + 1, [(start, char_pos), (char_pos+1, end+1)]);
        }
        lines_to_skip = 2;
    } else {
        line_range[cursor_line] = (line_start, line_range[cursor_line].1 + 1);
    }
 
    for mut line in line_range.iter_mut().skip(cursor_line + lines_to_skip) {
        line.0 += 1;
        line.1 += 1;
    }

    EditAction::Insert((cursor_line, cursor_column), std::str::from_utf8(to_insert).unwrap().to_string())
}





fn render_char(width: i32, height: i32, c: char) -> Rect {
    Rect::new(width * (c as i32 - 33), 0, width as u32, height as u32)
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


// This clearly is telling me I'm missing an abstraction
fn draw_string<'a>(canvas: & mut Canvas<video::Window>, target: &'a mut Rect, texture: &Texture, text: &str) -> &'a mut Rect {
    for char in text.chars() {
        let char_rect : Rect = render_char(target.width() as i32, target.height() as i32, char as char);
        target.set_x(target.x() + target.width() as i32);
        canvas.copy(texture, Some(char_rect), Some(*target)).unwrap();
    }
    target
}

fn parse_lines(chars : & Vec<u8>) ->  Vec<(usize, usize)> {
    let mut line_start = 0;
    let mut line_range = Vec::<(usize,usize)>::with_capacity(chars.len()/60);
    for (line_end, char) in chars.iter().enumerate() {
        if *char == b'\n'{
            line_range.push((line_start, line_end));
            line_start = line_end + 1;
        }
        if line_end == chars.len() - 1 {
            line_range.push((line_start, line_end + 1));
        }
    }
    if line_range.is_empty() {
        line_range.push((0, 0));
    }
    line_range
}


#[derive(Debug)]
struct EditorBounds {
    editor_left_margin: usize,
    line_number_gutter_width: usize,
    letter_height: usize,
    letter_width: usize,
}

impl EditorBounds{

    fn line_number_digits(&self, line_range: &Vec<(usize,usize)>) -> usize {
        digit_count(line_range.len())
    }
    fn line_number_padding(&self, line_range: &Vec<(usize,usize)>) -> usize {
        self.line_number_digits(line_range) * self.letter_width as usize + self.line_number_gutter_width + self.editor_left_margin + self.letter_width as usize
    }
}


fn text_space_from_screen_space(scroller: &Scroller, mut x: i32, y: i32, line_range: &Vec<(usize,usize)>) -> Option<Cursor> {
    // Slightly off probably due to rounding.
    // println!("{}", y as f32 / letter_height as f32);
    
    // Probably should move some/all of this to the scroller.
    let EditorBounds {letter_height, letter_width, ..} = scroller.bounds;
    let bounds = &scroller.bounds;
    let line_number_padding = bounds.line_number_padding(line_range) as i32;

    let line_number : usize = ((y as f32 / letter_height as f32).floor() as i32 + scroller.lines_above_fold() as i32).try_into().unwrap();
    if x < line_number_padding && x > line_number_padding - 20  {
        x = line_number_padding;
    } 
    if x < line_number_padding {
        return None;
    }
    let mut column_number : usize = ((x - line_number_padding as i32) / letter_width as i32).try_into().unwrap();

    if let Some((line_start, line_end)) = line_range.get(line_number) {
        if column_number > line_end - line_start {
            column_number = line_range[line_number].1 - line_range[line_number].0;
        }
        return Some(Cursor(line_number, column_number));
    }
    if line_number > line_range.len() {
        if let Some((start, end)) = line_range.last() {
           return Some(Cursor(line_range.len() - 1, line_length((*start, *end))));
        }
       
    }
    None
}



fn move_right(target: &mut Rect, padding: i32) -> &mut Rect {
    target.set_x(target.x() + padding);
    target
}

fn move_down(target: &mut Rect, padding: i32) -> &mut Rect {
    target.set_y(target.y() + padding);
    target
}

fn line_length(line: (usize, usize)) -> usize {
    line.1 - line.0
}




// I am editing in place right now. There are a lot of things wrong with the way I'm doing it
// but in general it is working.
// I need to fix it so that cursors can't end up in the middle of nowhere.
// I need to fix special symbols.
// I need to fix lots of things
// Editing a 1 gb file is slow. But do I care?
// This path might not be sustainable long term,
// but I it is getting my going.
// I need delete. I also think I am getting rid of \n characters
// at times and that might be awkward for delete.

// TODO:
// Add some spacing between letters!
// Change cursor
// Need to make some nice cursor movement function
// This is probably pretty important to do.

// TODO: 
// scroll on keypresses and selection
// Delete selections
// UNDO!
// Refactor things

// Need to add a real parser or I can try messing with tree sitter.
// But maybe I need to make text editable first?


// Bugs:
// completely empty file cannot have any content added


// It would be pretty cool to add a minimap
// Also cool to just add my own scrollbar.


// It would be great to make this like a real window
// But probably need a lot more abstraction before I get there.
fn draw_list<'a, T: Debug, I>(canvas: & mut Canvas<video::Window>, target: &'a mut Rect, texture: &Texture, line_height: i32, elements: I) -> ()
where I: IntoIterator<Item=T> {
    let start_x = target.x();
    for element in elements {
        draw_string(canvas, target, texture, &format!("{:?}", element));
        move_down(target, line_height);
        target.set_x(start_x);
    }
}


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

struct CursorContext {
    cursor: Option<Cursor>,
    selection: Option<((usize, usize), (usize, usize))>,
    mouse_down: Option<Cursor>,
}

impl CursorContext {
    fn move_up(&mut self, line_range: &Vec<(usize, usize)>) {
        self.cursor
            .as_mut()
            .map(|cursor| cursor.move_up(line_range));
    }
    fn move_down(&mut self, line_range: &Vec<(usize, usize)>) {
        self.cursor
            .as_mut()
            .map(|cursor| cursor.move_down(line_range));
    }
    fn move_left(&mut self, line_range: &Vec<(usize, usize)>) {
        self.cursor
            .as_mut()
            .map(|cursor| cursor.move_left(line_range));
    }
    fn move_right(&mut self, line_range: &Vec<(usize, usize)>) {
        self.cursor
            .as_mut()
            .map(|cursor| cursor.move_right(line_range));
    }
    fn set_cursor(&mut self, cursor: Cursor) {
        self.cursor = Some(cursor);
    }

    fn set_cursor_opt(&mut self, cursor: Option<Cursor>) {
        self.cursor = cursor;
    }

    fn clear_selection(&mut self) {
        self.selection = None;
    }

    fn set_selection(&mut self, selection: ((usize, usize), (usize, usize))) {
        self.selection = Some(selection);
    }

    fn fix_cursor(&mut self, line_range: &Vec<(usize, usize)>) {
        // Need to do sanity checks for cursor column
        if let Some(Cursor(cursor_line, cursor_column)) = self.cursor {
            match line_range.get(cursor_line) {
                Some((start, end)) => {
                    if cursor_column > *end {
                        self.cursor = Some(Cursor(cursor_line, line_length((*start, *end))));
                    }
                }
                None => {
                    self.cursor = line_range.last().map(|(line, column)| Cursor(*line, *column));
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

    fn move_cursor_from_screen_position(&mut self, scroller: &Scroller, x: i32, y: i32, line_range: &Vec<(usize,usize)>) {
        self.cursor = text_space_from_screen_space(scroller, x, y, line_range);
    }

    fn cursor_exists(&self) -> bool {
        self.cursor.is_some()
    }
}


struct Scroller {
    offset_y: i32,
    scroll_speed: i32,
    window: Window,
    bounds: EditorBounds,
}

impl Scroller {
    fn scroll(&mut self, amount: i32, line_range: &Vec<(usize, usize)>) {
        if !self.at_end(line_range) || amount < 0 {
            self.offset_y += amount * self.scroll_speed;
        }
        self.offset_y = max(0, self.offset_y);
    }

    fn viewing_lines(&self) -> usize {
       (self.window.height / self.bounds.letter_height as i32) as usize
    }
    
    fn lines_above_fold(&self) -> usize {
        self.offset_y as usize / self.bounds.letter_height as usize
    }
    
    fn at_end(&self, line_range: &Vec<(usize, usize)>) -> bool {
        self.lines_above_fold() + self.viewing_lines() >= line_range.len() + 3
    }

    fn to_the_top(&mut self) {
        self.offset_y = 0;
    }

    fn line_fraction(&self) -> usize {
        self.offset_y as usize % self.bounds.letter_height as usize
    }
}


fn draw_font_texture<'a>(texture_creator: &'a TextureCreator<WindowContext>, ttf_context: sdl2::ttf::Sdl2TtfContext) -> Result<(Texture<'a>, usize, usize), String> {
    let font_path = "/Users/jimmyhmiller/Library/Fonts/UbuntuMono-Regular.ttf";
    let font = ttf_context.load_font(font_path, 16)?;
    let mut text = String::new();
    for i  in 33..127 {
        text.push(i as u8 as char);
    }
    let surface = font
        .render(text.as_str())
        // This needs to be 255 if I want to change colors
        .blended(Color::RGBA(255, 255, 255, 255))
        .map_err(|e| e.to_string())?;
    let texture = texture_creator
        .create_texture_from_surface(&surface)
        .map_err(|e| e.to_string())?;
    let TextureQuery { width, height, .. } = texture.query();
    let width = (width / text.len() as u32).try_into().unwrap();
    Ok((texture, width, height.try_into().unwrap()))
}

fn draw(canvas: &mut Canvas<video::Window>, scroller: &Scroller, line_range: &Vec<(usize, usize)>, texture: &mut Texture, cursor_context: &CursorContext,  chars: &Vec<u8>, fps: &mut FpsCounter) -> Result<(), String> {
    canvas.set_draw_color(Color::RGBA(42, 45, 62, 255));
    canvas.clear();
    let editor_left_margin = scroller.bounds.editor_left_margin;
    let line_number_padding = scroller.bounds.line_number_padding(line_range);
    let line_number_digits = scroller.bounds.line_number_digits(line_range);
    let mut target = Rect::new(editor_left_margin as i32, (scroller.line_fraction() as i32).neg(), scroller.bounds.letter_width as u32, scroller.bounds.letter_height as u32);
    for line in scroller.lines_above_fold() as usize..min(scroller.lines_above_fold() + scroller.viewing_lines(), line_range.len()) {
        texture.set_color_mod(167, 174, 210);
        let (start, end) = line_range[line];
        target.set_x(editor_left_margin as i32);

        // I want to pad this so that the offset by the line number never changes.
        // Really I should draw a line or something to make it look nicer.
        let left_padding_count = line_number_digits - digit_count(line + 1);
        let padding = left_padding_count * scroller.bounds.letter_width as usize;
        move_right(&mut target, padding as i32);

        let line_number = (line + 1).to_string();

        let target = draw_string(canvas, &mut target, texture, &line_number);
        move_right(target, scroller.bounds.line_number_gutter_width as i32);

        if let Some(cursor) = cursor_context.cursor {
            if cursor.0 == line {
                let cursor_x = cursor.1 as i32  * scroller.bounds.letter_width as i32 + line_number_padding as i32;
                let cursor_y = target.y();
                canvas.set_draw_color(Color::RGBA(255, 204, 0, 255));
                canvas.fill_rect(Rect::new(cursor_x as i32, cursor_y as i32, 2, scroller.bounds.letter_height as u32))?;
            }
        }


        if let Some(((start_line, start_column), (end_line, end_column))) = cursor_context.selection {
            if line >= start_line.try_into().unwrap() && line <= end_line.try_into().unwrap() {
                let start_x = if line == start_line {
                    start_column * scroller.bounds.letter_width + line_number_padding
                } else {
                    line_number_padding
                };
                let width = if start_line == end_line {
                    ((end_column - start_column) * scroller.bounds.letter_width).try_into().unwrap()
                } else if line == end_line {
                    (end_column * scroller.bounds.letter_width as usize).try_into().unwrap()
                } else if line == start_line {
                    ((line_length(line_range[line]) - start_column) * scroller.bounds.letter_width as usize).try_into().unwrap()
                } else {
                    (line_length(line_range[line]) * scroller.bounds.letter_width).try_into().unwrap()
                };

                let start_y = target.y();
                canvas.set_draw_color(Color::RGBA(65, 70, 99, 255));
                // Need to deal with last line.
                canvas.fill_rect(Rect::new(start_x as i32, start_y, width, scroller.bounds.letter_height as u32))?;
            }

        };

        draw_string(canvas, target, texture, std::str::from_utf8(chars[start..end].as_ref()).unwrap());


        move_down(target, scroller.bounds.letter_height as i32);
    }
    let current_fps = fps.tick();
    let mut target = Rect::new(scroller.window.width - (scroller.bounds.letter_width * 10) as i32, 0, scroller.bounds.letter_width as u32, scroller.bounds.letter_height as u32);
    draw_string(canvas, &mut target, texture, &format!("fps: {}", current_fps));
    let mut target = Rect::new(scroller.window.width - (scroller.bounds.letter_width * 22) as i32, scroller.window.height-scroller.bounds.letter_height as i32, scroller.bounds.letter_width as u32, scroller.bounds.letter_height as u32);
    if let Some(Cursor(cursor_line, cursor_column)) = cursor_context.cursor {
        draw_string(canvas, &mut target, texture, &format!("Line {}, Column {}", cursor_line, cursor_column));
    }
    canvas.present();
    Ok(())
}


fn handle_events(event_pump: &mut sdl2::EventPump,
                
                cursor_context: &mut CursorContext,           

                line_range: &mut Vec<(usize, usize)>,
                chars: &mut Vec<u8>,
                
                transaction_manager: &mut TransactionManager,

                scroller: &mut Scroller) {
    let mut is_text_input = false;
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
                        cursor_context.move_up(line_range);
                    },
                    (Keycode::Down, _) => {
                        cursor_context.move_down(line_range);
                    },
                    (Keycode::Left, _) => {
                        cursor_context.move_left(line_range);
                    },
                    (Keycode::Right, _) => {
                        cursor_context.move_right(line_range);
                    },
                    (Keycode::Backspace, _) => {

                        // Need to deal with this in a nicer way
                        if let Some(current_selection) = cursor_context.selection {
                            let (start, end) = current_selection;
                            let (start_line, start_column) = start;
                            let (end_line, end_column) = end;
                            if let Some((line_start, _line_end)) = line_range.get(start_line as usize) {
                                let char_start_pos = line_start + start_column as usize ;
                                if let Some((end_line_start, _line_end)) = line_range.get(end_line as usize) {
                                    let char_end_pos = end_line_start + end_column as usize;
                                    chars.drain(char_start_pos as usize..char_end_pos as usize);
                                    // Probably shouldn't reparse the whole file.
                                    *line_range = parse_lines(chars);
                                    cursor_context.clear_selection();
                                    cursor_context.fix_cursor(line_range);
                                    continue;
                                }
                            
                            }
                        }

                    
                        // Is there a better way to do this other than clone?
                        // Maybe a non-mutating method?
                        // How to deal with optional aspect here?
                        if let Some(current_cursor) = cursor_context.cursor {
                            let mut old_cursor = current_cursor.clone();
                            // We do this move_left first, because otherwise we might end up at the end
                            // of the new line we formed from the deletion, rather than the old end of the line.
                            let cursor_action = old_cursor.move_left(line_range);
                            // move_left isn't option but this is. Probably some weird edge case here
                            let action = handle_delete(current_cursor, chars, line_range);
                            if action.is_some() {
                                transaction_manager.add_action_pair(action.unwrap(), cursor_action);
                            }


                            cursor_context.set_cursor(old_cursor);
                        }
                    }
                    (Keycode::Return, _) => {
                        // refactor to be better
                        // if cursor_context.cursor_exists() {
                        //     let action = handle_insert(Cursor(cursor_line, cursor_column), &[b'\n'], chars, line_range); 
                        // }
                        if let Some(current_cursor) = cursor_context.cursor.as_mut() {
                            let action = handle_insert(*current_cursor, &[b'\n'], chars, line_range);
                            let cursor_action = current_cursor.move_down(line_range);
                            transaction_manager.add_action_pair(action, cursor_action);
                        
                            current_cursor.start_of_line();
                        }
                    },


                    (Keycode::Z, key_mod) => {
                    
                        if key_mod == Mod::LGUIMOD || keymod == Mod::RGUIMOD {
                            transaction_manager.undo(cursor_context, chars, line_range);
                        } else if key_mod == (Mod::LSHIFTMOD | Mod::LGUIMOD) {
                            transaction_manager.redo(cursor_context, chars, line_range);
                        } else {
                            is_text_input = true
                        }
                   
                    },

                    (Keycode::O, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        let path = FileDialog::new()
                            .set_location("~/Documents")
                            .show_open_single_file()
                            .unwrap();

                        if path.is_none() {
                            continue;
                        }
                        let path = path.unwrap();
                        let path_str = path.to_str().unwrap();
                        let path_str = &path_str.replace("file://", "");

                        // Need to refactor into reusable function instead of just repeating here.
                        let text = fs::read_to_string(path_str).unwrap();
                        *chars = text.as_bytes().to_vec();
                
                        *line_range = parse_lines(chars);

                        scroller.to_the_top();

                    }
                    (Keycode::A, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        // This is super ugly, fix.
                        cursor_context.set_selection(((0,0), (line_range.len()-1, line_length(line_range[line_range.len()-1]))));
                    }

                    _ => is_text_input = true
                }
            }
            Event::TextInput{text, ..} => {
                if is_text_input {
                    // TODO: Replace with actually deleting the selection.
                    cursor_context.clear_selection();
                    if let Some(current_cursor) = cursor_context.cursor.as_mut() {
                        let to_insert = text.into_bytes();
                        let action = handle_insert(*current_cursor, to_insert.as_slice(), chars, line_range);
                        let cursor_action = current_cursor.move_right(line_range);
                    
                        transaction_manager.add_action_pair(action, cursor_action);
                    }
                }
            }

            // Need to make selection work
            // Which probably means changing cursor representation
       
            Event::MouseButtonDown { x, y, .. } => {
                cursor_context.move_cursor_from_screen_position(scroller, x, y, line_range);
                cursor_context.mouse_down();
                cursor_context.clear_selection();
            }

            Event::MouseMotion{x, y, .. } => {
                if let Some(Cursor(start_line, mut start_column)) = cursor_context.mouse_down {
                    cursor_context.move_cursor_from_screen_position(scroller, x, y, line_range);
                    // TODO: Get my int types correct!
                    if let Some(Cursor(line, mut column)) = cursor_context.cursor {
                        let new_start_line = start_line.min(line.try_into().unwrap());
                        let line = line.max(start_line.try_into().unwrap());
                        if new_start_line != start_line || start_line == line && start_column > column {
                            let temp = start_column;
                            start_column = column.try_into().unwrap();
                            column = temp as usize;
                        }

                        // ugly refactor
                        cursor_context.set_selection(((new_start_line, start_column), (line, column)));
                    
                    }
                }
            }

            Event::MouseButtonUp{x, y, ..} => {
                if let Some(Cursor(start_line, mut start_column)) = cursor_context.mouse_down {
                    cursor_context.move_cursor_from_screen_position(scroller, x, y, line_range);
                    if cursor_context.selection.is_some() {
                        if let Some(Cursor(line, mut column)) = cursor_context.cursor {
                            let new_start_line = start_line.min(line.try_into().unwrap());
                            let line = line.max(start_line.try_into().unwrap());
                            if new_start_line != start_line || start_line == line && start_column > column {
                                let temp = start_column;
                                start_column = column.try_into().unwrap();
                                column = temp as usize;
                            }
    
                            cursor_context.set_selection(((new_start_line, start_column), (line.try_into().unwrap(), column.try_into().unwrap())));
                            // TODO: Set Cursor
                        }
                    }
                    
                }
            
                cursor_context.clear_mouse_down();
            }
            // Continuous resize in sdl2 is a bit weird
            // Would need to watch events or something
            Event::Window {win_event: WindowEvent::Resized(width, height), ..} => {
                scroller.window.resize(width, height);
            }

            Event::MouseWheel {x: _, y, direction , timestamp: _, .. } => {
                let direction_multiplier = match direction {
                    sdl2::mouse::MouseWheelDirection::Normal => 1,
                    sdl2::mouse::MouseWheelDirection::Flipped => -1,
                    sdl2::mouse::MouseWheelDirection::Unknown(x) => x as i32
                };
                scroller.scroll( y * direction_multiplier, &line_range);
            }
            _ => {}
        }
        cursor_context.fix_cursor(line_range);
    }
}



fn main() -> Result<(), String> {
    set_smooth_scroll();

    let window = Window {
        width: 1200,
        height: 800,
    };

    let (ttf_context, mut canvas, mut event_pump, texture_creator) = setup_sdl(&window)?;
    let (mut texture, letter_width, letter_height) = draw_font_texture(&texture_creator, ttf_context)?;
    texture.set_color_mod(167, 174, 210);

    let mut scroller = Scroller {
        offset_y: 0,
        scroll_speed: 5,
        window,
        bounds: EditorBounds {
            editor_left_margin: 10,
            line_number_gutter_width : 20,
            letter_height,
            letter_width,
        },
    };

    let text = fs::read_to_string("/Users/jimmyhmiller/Documents/Code/Playground/rust/editor/src/main.rs").unwrap();
    let mut chars = text.as_bytes().to_vec();
    let mut line_range = parse_lines(&chars);

    let mut fps = FpsCounter{
        start_time: Instant::now(),
        frame_count: 0,
        fps: 0,
    };

    let mut cursor_context = CursorContext {
        cursor: None,
        mouse_down: None,
        selection: None,
    };
    let mut transaction_manager = TransactionManager::new();


    loop {
        draw(&mut canvas, &scroller, &line_range,  &mut texture, &cursor_context,  &chars, &mut fps)?;
        handle_events(&mut event_pump,  &mut cursor_context, &mut line_range, &mut chars, &mut transaction_manager, &mut scroller);
    }
}


