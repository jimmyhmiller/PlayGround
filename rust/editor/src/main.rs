use std::{cmp::{max, min}, convert::TryInto, fs, ops::Neg};
use std::fmt::Debug;

use native_dialog::FileDialog;
use sdl2::{event::*, keyboard::*, mouse::{SystemCursor}, pixels::Color, rect::Rect, render::*, video};


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

    fn undo(&mut self, cursor: &mut Cursor, chars: &mut Vec<u8>, line_range: &mut Vec<(usize, usize)>) {
        if self.transaction_pointer == 0 {
           return;
        }
        let last_transaction = self.transactions[self.transaction_pointer].transaction_number;
        let mut i = self.transaction_pointer;
        while self.transactions[i].transaction_number == last_transaction {
            println!("{:?}", self.transactions[i].action);
            self.transactions[i].action.undo(cursor, chars, line_range);

            if i == 0 {
                break;
            }
            i = self.transactions[i].parent_pointer;
        }
        self.transaction_pointer = i;

    }

    // How do I redo?

    fn redo(&mut self, cursor: &mut Cursor, chars: &mut Vec<u8>, line_range: &mut Vec<(usize, usize)>) {

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
                    self.transactions[i].action.redo(cursor, chars, line_range);
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
    fn undo(&self, cursor: &mut Cursor, chars: &mut Vec<u8>, line_range: &mut Vec<(usize, usize)>) {
        match self {
            EditAction::Insert((start, end), _text_to_insert) => {
                let mut new_position = Cursor(*start, *end);
                new_position.move_right(line_range);
                handle_delete(new_position, chars, line_range);
                new_position.move_left(line_range);
                *cursor = new_position;
            },
            EditAction::Delete((start, end), text_to_delete) => {
                let mut new_position = Cursor(*start, *end);
                new_position.move_left(line_range);
                handle_insert(new_position, text_to_delete.as_bytes(), chars, line_range);
                new_position.move_right(line_range);
                *cursor = new_position;
            },
            EditAction::CursorPosition(old_cursor) => {
                cursor.set_position(*old_cursor);
            }
        }
    }

    fn redo(&self, cursor: &mut Cursor, chars: &mut Vec<u8>, line_range: &mut Vec<(usize, usize)>) {

        match self {
            EditAction::Insert((start, end), text_to_insert) => {
                handle_insert( Cursor(*start, *end), text_to_insert.as_bytes(), chars, line_range);
            },
            EditAction::Delete((start, end), _text_to_delete) => {
                handle_delete( Cursor(*start, *end), chars, line_range);
            },
            EditAction::CursorPosition(new_cursor) => {
                cursor.set_position(*new_cursor);
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
    lines_above_fold: usize,
    letter_height: usize,
    letter_width: usize,
    line_number_padding: usize,
}


fn text_space_from_screen_space(EditorBounds {lines_above_fold, letter_height,line_number_padding, letter_width} : &EditorBounds, mut x: i32, y: i32, line_range: &Vec<(usize,usize)>) -> Option<Cursor> {
    // Slightly off probably due to rounding.
    // println!("{}", y as f32 / letter_height as f32);
    let line_number : usize = ((y as f32 / *letter_height as f32).floor() as i32 + *lines_above_fold as i32).try_into().unwrap();
    if x < *line_number_padding as i32 && x > *line_number_padding as i32 - 20  {
        x = *line_number_padding as i32;
    } 
    if x < *line_number_padding as i32 {
        return None;
    }
    let mut column_number : usize = ((x - *line_number_padding as i32) / *letter_width as i32).try_into().unwrap();

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



fn main() -> Result<(), String> {

    unsafe { 

        use cocoa_foundation::foundation::NSUserDefaults;
        use cocoa_foundation::foundation::NSString;
        use cocoa_foundation::base::nil;
        // [[NSUserDefaults standardUserDefaults] setBool: YES
        //                                        forKey: @"AppleMomentumScrollSupported"];

        let defaults = cocoa_foundation::base::id::standardUserDefaults();
        let key = NSString::alloc(nil).init_str("AppleMomentumScrollSupported");
        defaults.setBool_forKey_(cocoa_foundation::base::YES, key)
    }

    let sdl_context = sdl2::init()?;
    let ttf_context = sdl2::ttf::init().map_err(|e| e.to_string())?;

    let mut window = Window {
        width: 1200,
        height: 800,
    };

    let sdl_window = sdl_context
        .video()?
        .window("Example", window.width as u32, window.height as u32)
        .resizable()
        .build()
        .unwrap();

    let cursor = sdl2::mouse::Cursor::from_system(SystemCursor::IBeam)
        .map_err(|err| format!("failed to load cursor: {}", err))?;
    cursor.set();


    // Let's create a Canvas which we will use to draw in our Window
    let mut canvas: Canvas<video::Window> = sdl_window
        .into_canvas()
        .present_vsync()
        .build()
        .unwrap();
    let mut event_pump = sdl_context.event_pump()?;

    let texture_creator = canvas.texture_creator();
    let font_path = "/Users/jimmyhmiller/Library/Fonts/UbuntuMono-Regular.ttf";
    let font = ttf_context.load_font(font_path, 16)?;


    let mut text = String::new();
    for i  in 33..127 {
        text.push(i as u8 as char);
    }
    
    // println!("{}", text);
    // let text ="abcdefghijklmnopqrstuvwxyz";
    let surface = font
        .render(text.as_str())
        // This needs to be 255 if I want to change colors
        .blended(Color::RGBA(255, 255, 255, 255))
        .map_err(|e| e.to_string())?;

    let mut texture = texture_creator
        .create_texture_from_surface(&surface)
        .map_err(|e| e.to_string())?;


    let TextureQuery { width, height, .. } = texture.query();


    let letter_width = width / text.len() as u32;
    let letter_height = height;
    let letter_height_usize : usize = letter_height.try_into().unwrap();
    let letter_width_usize : usize = letter_width.try_into().unwrap();

    let start_time = std::time::Instant::now();
    // let mut text = fs::read_to_string("/Users/jimmyhmiller/Desktop/test/test.txt").unwrap();
    let text = fs::read_to_string("/Users/jimmyhmiller/Documents/Code/jml/src/jml/core.clj").unwrap();
    // let mut text = "test\nthing\nstuff".to_string();
    println!("read file in {} ms", start_time.elapsed().as_millis());
    let mut chars = text.as_bytes().to_vec();

    let mut line_range = parse_lines(&chars);
    
    // println!("{:?}", line_range);
    println!("parsed file in {} ms", start_time.elapsed().as_millis());

    println!("copied file");
    let mut offset_y = 0;
    let mut at_end = false;
    let scroll_speed : i32 = 5;
    let mut frame_counter = 0;
    let mut time_start = std::time::Instant::now();
    let mut fps = 0;
    let mut cursor_context = CursorContext {
        cursor: None,
        mouse_down: None,
        selection: None,
    };
    let mut transaction_manager = TransactionManager::new();
    
    

    texture.set_color_mod(167, 174, 210);
    loop {
        

        canvas.set_draw_color(Color::RGBA(42, 45, 62, 255));
        canvas.clear();
    
        let mut scroll_y : i32 = 0;

         // duplicated below because we need to recompute after
         // we update line count
        let editor_left_margin = 10;
        let line_number_digits = digit_count(line_range.len());
        let line_number_gutter_width = 20;
        // final letter width is because we write our string, we are in that letters position, then move more.
        let line_number_padding = line_number_digits * letter_width as usize + line_number_gutter_width + editor_left_margin + letter_width as usize;

        let lines_above_fold : usize = offset_y as usize / letter_height as usize;
        // Fix this to be less hacky.
        let viewing_window: usize = min((window.height / letter_height as i32).try_into().unwrap(), 1000);



        let editor_bounds = EditorBounds {
            lines_above_fold,
            letter_height: letter_height_usize,
            letter_width: letter_width_usize,
            line_number_padding,
        };

        handle_events(&mut event_pump,  &mut cursor_context, &mut line_range, &mut chars, &mut transaction_manager,editor_bounds, &mut window, &mut offset_y, &mut scroll_y, scroll_speed);


        // duplicated
        let editor_left_margin = 10;
        let line_number_digits = digit_count(line_range.len());
        let line_number_gutter_width = 20;
        // final letter width is because we write our string, we are in that letters position, then move more.
        let line_number_padding = line_number_digits * letter_width as usize + line_number_gutter_width + editor_left_margin + letter_width as usize;


        if !at_end || scroll_y < 0 {
            offset_y += scroll_y;
        }
        offset_y = max(0, offset_y);

        // Need to reset this after scroll
        // Need better handling of these things.
        let lines_above_fold : usize = offset_y as usize / letter_height as usize;

        let line_fraction = offset_y as usize % letter_height as usize;



        if lines_above_fold + viewing_window >= line_range.len() + 3 {
            at_end = true;
        } else {
            at_end = false;
        }



        let mut target = Rect::new(editor_left_margin as i32, (line_fraction as i32).neg(), letter_width, letter_height);

        // I got rid of line wrap in this refactor. Probably should add that back in.
        for line in lines_above_fold as usize..min(lines_above_fold + viewing_window, line_range.len()) {
            texture.set_color_mod(167, 174, 210);
            let (start, end) = line_range[line];
            target.set_x(editor_left_margin as i32);

            // I want to pad this so that the offset by the line number never changes.
            // Really I should draw a line or something to make it look nicer.
            let left_padding_count = line_number_digits - digit_count(line + 1);
            let padding = left_padding_count * letter_width as usize;
            move_right(&mut target, padding as i32);

            let line_number = (line + 1).to_string();

            let target = draw_string(&mut canvas, &mut target, &texture, &line_number);
            move_right(target, line_number_gutter_width as i32);
        
            if let Some(cursor) = cursor_context.cursor {
                if cursor.0 == line {
                    let cursor_x = cursor.1 as i32  * letter_width as i32 + line_number_padding as i32;
                    let cursor_y = target.y();
                    canvas.set_draw_color(Color::RGBA(255, 204, 0, 255));
                    canvas.fill_rect(Rect::new(cursor_x as i32, cursor_y as i32, 2, letter_height))?;
                }
            }


            if let Some(((start_line, start_column), (end_line, end_column))) = cursor_context.selection {
                if line >= start_line.try_into().unwrap() && line <= end_line.try_into().unwrap() {
                    let start_x = if line == start_line {
                        start_column * letter_width_usize + line_number_padding
                    } else {
                        line_number_padding
                    };
                    let width = if start_line == end_line {
                        ((end_column - start_column) * letter_width as usize).try_into().unwrap()
                    } else if line == end_line {
                        (end_column * letter_width as usize).try_into().unwrap()
                    } else if line == start_line {
                        ((line_length(line_range[line]) - start_column) * letter_width as usize).try_into().unwrap()
                    } else {
                        (line_length(line_range[line]) * letter_width_usize).try_into().unwrap()
                    };

                    let start_y = target.y();
                    canvas.set_draw_color(Color::RGBA(65, 70, 99, 255));
                    // Need to deal with last line.
                    canvas.fill_rect(Rect::new(start_x as i32, start_y, width, letter_height))?;
                }

            };

            draw_string(&mut canvas, target, &texture, std::str::from_utf8(chars[start..end].as_ref()).unwrap());


            move_down(target, letter_height as i32);
        }

        let mut target = Rect::new(window.width - (letter_width * 10) as i32, 0, letter_width, letter_height);
        draw_string(&mut canvas, &mut target, &texture, &format!("fps: {}", fps));
        frame_counter += 1;
        if time_start.elapsed().as_secs() >= 1 {
            fps = frame_counter;
            frame_counter = 0;
            time_start = std::time::Instant::now();
        }

        // Need to calculate this based on length
        let mut target = Rect::new(window.width - (letter_width * 22) as i32, window.height-letter_height as i32, letter_width, letter_height);
        if let Some(Cursor(cursor_line, cursor_column)) = cursor_context.cursor {
            draw_string(&mut canvas, &mut target, &texture, &format!("Line {}, Column {}", cursor_line, cursor_column));
        }

        // let mut target = Rect::new(window_width - (letter_width * 50) as i32, window_height-(letter_height*3) as i32, letter_width, letter_height);
        // draw_string(&mut canvas, &mut target, &texture, 
        //     &format!("#: {} len: {} ptr: {}", 
        //                     transaction_manager.current_transaction, 
        //                     transaction_manager.transactions.len(),
        //                     transaction_manager.transaction_pointer));

        // let mut target = Rect::new(window_width - (letter_width * 150) as i32, 10, letter_width, letter_height);
        // draw_list(&mut canvas, &mut target, &texture, letter_height as i32, transaction_manager.transactions.iter().skip(lines_above_fold));



        canvas.present();
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
            .map(|cursor| cursor.move_up(&*line_range));
    }
    fn move_down(&mut self, line_range: &Vec<(usize, usize)>) {
        self.cursor
            .as_mut()
            .map(|cursor| cursor.move_down(&*line_range));
    }
    fn move_left(&mut self, line_range: &Vec<(usize, usize)>) {
        self.cursor
            .as_mut()
            .map(|cursor| cursor.move_left(&*line_range));
    }
    fn move_right(&mut self, line_range: &Vec<(usize, usize)>) {
        self.cursor
            .as_mut()
            .map(|cursor| cursor.move_right(&*line_range));
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
 
}




fn handle_events(event_pump: &mut sdl2::EventPump,
                
                cursor_context: &mut CursorContext,           

                line_range: &mut Vec<(usize, usize)>,
                chars: &mut Vec<u8>,
                
                transaction_manager: &mut TransactionManager,

                editor_bounds: EditorBounds,

                window: &mut Window,

                offset_y: &mut i32,
                scroll_y: &mut i32,
                scroll_speed: i32) {
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
                                    *line_range = parse_lines(&*chars);
                                    cursor_context.clear_selection();
                                    cursor_context.fix_cursor(&*line_range);
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
                            let cursor_action = old_cursor.move_left(&*line_range);
                            // move_left isn't option but this is. Probably some weird edge case here
                            let action = handle_delete(current_cursor, chars, line_range);
                            if action.is_some() {
                                transaction_manager.add_action_pair(action.unwrap(), cursor_action);
                            }


                            cursor_context.set_cursor(old_cursor);
                        }
                    }
                    (Keycode::Return, _) => {
                        // Some weird things are happening here.
                        // Letters appear out of the void
                        if let Some(current_cursor) = cursor_context.cursor.as_mut() {
                            let Cursor(cursor_line, cursor_column) = *current_cursor;
                            let action = handle_insert(Cursor(cursor_line, cursor_column), &[b'\n'], chars, line_range);
                            let cursor_action = current_cursor.move_down(&*line_range);
                            transaction_manager.add_action_pair(action, cursor_action);
                        
                            current_cursor.start_of_line();
                        }
                    },


                    (Keycode::Z, key_mod) => {
                    
                        if key_mod == Mod::LGUIMOD || keymod == Mod::RGUIMOD {
                            // Do I need these cursors? Why doesn't redo and undo just set the cursor?
                            // Need to refactor.
                            let mut new_cursor = cursor_context.cursor.unwrap_or(Cursor(0, 0));
                            transaction_manager.undo(&mut new_cursor, chars, line_range);
                            cursor_context.set_cursor(new_cursor);
                        } else if key_mod == (Mod::LSHIFTMOD | Mod::LGUIMOD) {
                            // Do I need these cursors? Why doesn't redo and undo just set the cursor?
                            // Need to refactor.
                            let mut new_cursor = cursor_context.cursor.unwrap_or(Cursor(0, 0));
                            transaction_manager.redo(&mut new_cursor, chars, line_range);
                            cursor_context.set_cursor(new_cursor);
                        } else {
                            is_text_input = true
                        }
                   
                    },

                    (Keycode::O, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        let path = FileDialog::new()
                            .set_location("~/Documents")
                            .show_open_single_file()
                            .unwrap();
                        let start_time = std::time::Instant::now();
                        if path.is_none() {
                            continue;
                        }
                        let path = path.unwrap();
                        let path_str = path.to_str().unwrap();
                        let path_str = &path_str.replace("file://", "");

                        // Need to refactor into reusable function instead of just repeating here.
                        let text = fs::read_to_string(path_str).unwrap();
                        println!("read file in {} ms", start_time.elapsed().as_millis());
                        *chars = text.as_bytes().to_vec();
                
                        *line_range = parse_lines(&*chars);
                    
                        *offset_y = 0;
                        println!("parsed file in {} ms", start_time.elapsed().as_millis());
                    }
                    (Keycode::A, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        // This is super ugly, fix.
                        cursor_context.set_selection(((0,0), ((line_range.len()-1).try_into().unwrap(), line_length(line_range[line_range.len()-1]).try_into().unwrap())));
                    }

                    _ => is_text_input = true
                }
            }
            Event::TextInput{text, ..} => {
                if is_text_input {
                    // TODO: Replace with actually deleting the selection.
                    cursor_context.clear_selection();
                    if let Some(current_cursor) = cursor_context.cursor.as_mut() {
                        let Cursor(cursor_line, cursor_column) = *current_cursor;
                        let to_insert = text.into_bytes();
                        let action = handle_insert(Cursor(cursor_line, cursor_column), to_insert.as_slice(), chars, line_range);
                        let cursor_action = current_cursor.move_right(&*line_range);
                    
                        transaction_manager.add_action_pair(action, cursor_action);
                    }
                }
            }

            // Need to make selection work
            // Which probably means changing cursor representation
       
            Event::MouseButtonDown { x, y, .. } => {
                cursor_context.set_cursor_opt(
                    text_space_from_screen_space(
                        &editor_bounds,
                        x,
                        y,
                        &*line_range,
                    )
                );

                cursor_context.mouse_down();
                cursor_context.clear_selection();
            }

            Event::MouseMotion{x, y, .. } => {
                if let Some(Cursor(start_line, mut start_column)) = cursor_context.mouse_down {
                    cursor_context.set_cursor_opt(
                 text_space_from_screen_space(
                            &editor_bounds,
                            x,
                            y,
                            &*line_range,
                        )
                    );
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
                        cursor_context.set_selection(((new_start_line, start_column), (line.try_into().unwrap(), column.try_into().unwrap())));
                    
                    }
                }
            }

            Event::MouseButtonUp{x, y, ..} => {
                if let Some(Cursor(start_line, mut start_column)) = cursor_context.mouse_down {
                    cursor_context.set_cursor_opt(
                        text_space_from_screen_space(
                                   &editor_bounds,
                                   x,
                                   y,
                                   &*line_range,
                               )
                           );
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
                window.resize(width, height);
            }

            Event::MouseWheel {x: _, y, direction , timestamp: _, .. } => {
                let direction_multiplier = match direction {
                    sdl2::mouse::MouseWheelDirection::Normal => 1,
                    sdl2::mouse::MouseWheelDirection::Flipped => -1,
                    sdl2::mouse::MouseWheelDirection::Unknown(x) => x as i32
                };
                *scroll_y = y * direction_multiplier * scroll_speed;
            }
            _ => {}
        }
        cursor_context.fix_cursor(line_range);
    }
}



