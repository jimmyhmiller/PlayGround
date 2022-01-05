use std::cmp::max;

use crate::{TextBuffer, renderer::EditorBounds, cursor::Cursor};


#[derive(Debug, Clone)]
pub struct Scroller {
    pub offset_x: i32,
    pub offset_y: i32,
    pub scroll_speed: i32
}


impl Scroller {
    pub fn new() -> Self {
        Scroller {
            offset_y: 0,
            offset_x: 0,
            scroll_speed: 5,
        }
    }

    pub fn scroll_y(&mut self, height: usize, amount: i32, text_buffer: &TextBuffer, bounds: &EditorBounds) {
        if !self.at_end(height, text_buffer, bounds) || amount < 0 {
            self.offset_y += amount * self.scroll_speed;
        }
        // Need to set offsety if we are at the end
        // so that it stays consistent
        self.offset_y = max(0, self.offset_y);
    }

    pub fn scroll_x(&mut self, width: usize, amount: i32, text_buffer: &mut TextBuffer, bounds: &EditorBounds) {
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
    pub fn scroll_x_character(&self, bounds: &EditorBounds) -> usize {
        self.offset_x as usize / bounds.letter_width
    }

    pub fn viewing_lines(&self, height: usize, bounds: &EditorBounds) -> usize {
       height / bounds.letter_height
    }

    pub fn lines_above_fold(&self, bounds: &EditorBounds) -> usize {
        self.offset_y as usize / bounds.letter_height as usize
    }

    pub fn at_end(&self, height: usize, text_buffer: &TextBuffer, bounds: &EditorBounds) -> bool {
        self.lines_above_fold(bounds) + self.viewing_lines(height, bounds) >= text_buffer.line_count() + 3
    }

    pub fn move_to_the_top(&mut self) {
        self.offset_y = 0;
    }

    pub fn line_fraction_y(&self, bounds: &EditorBounds) -> usize {
        self.offset_y as usize % bounds.letter_height as usize
    }

    pub fn line_fraction_x(&self, bounds: &EditorBounds) -> usize {
        self.offset_x as usize % bounds.letter_width as usize
    }




// This probably belongs in scroller
pub fn text_space_from_screen_space(&self, mut x: usize, y: usize, text_buffer: &TextBuffer, bounds: &EditorBounds) -> Option<Cursor> {
    // Slightly off probably due to rounding.
    // println!("{}", y as f32 / letter_height as f32);

    // Probably should move some/all of this to the scroller.
    let EditorBounds {letter_height, letter_width, ..} = *bounds;
    let line_number_padding = bounds.line_number_padding(text_buffer) as i32;

    // Still crash here
    // Didn't realize it could do that without the unwrap
    let line_number : usize = (((y as f32 / letter_height as f32)
        + (self.line_fraction_y(bounds) as f32 / bounds.letter_height as f32)).floor() as i32
        + self.lines_above_fold(bounds) as i32) as usize;


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

}

