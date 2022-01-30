use std::ops::Neg;

use sdl2::{pixels::Color, rect::Rect, mouse::SystemCursor, render::{Canvas, Texture}, video};
use sdl2::gfx::primitives::DrawRenderer;

use crate::{TextBuffer, Scroller, Window, PaneManager, Cursor, color::STANDARD_TEXT_COLOR};



#[derive(Debug, Clone)]
pub struct EditorBounds {
    pub editor_left_margin: usize,
    pub line_number_gutter_width: usize,
    pub letter_height: usize,
    pub letter_width: usize,
}

impl EditorBounds{

     pub fn line_number_digits(&self, text_buffer: &TextBuffer) -> usize {
        digit_count(text_buffer.line_count())
    }
     pub fn line_number_padding(&self, text_buffer: &TextBuffer) -> usize {
        self.line_number_digits(text_buffer) * self.letter_width as usize
            + self.line_number_gutter_width
            + self.editor_left_margin
            + self.letter_width as usize
    }
}


pub fn digit_count(x: usize) -> usize {
    let mut count = 0;
    let mut x = x;
    while x > 0 {
        x /= 10;
        count += 1;
    }
    count
}




pub struct Renderer<'a> {
    pub canvas: Canvas<video::Window>,
    pub texture: Texture<'a>,
    pub target: Rect,
    pub bounds: EditorBounds,
    pub system_cursor: sdl2::mouse::Cursor,
}


impl<'a> Renderer<'a> {

    pub fn draw_triangle(&mut self, x: i16, y: i16, color: Color) -> Result<(), String> {
        self.canvas.filled_trigon(x, y, x, y+10, x+5, y+5, color)
    }

    pub fn set_draw_color(&mut self, color: Color) {
        self.canvas.set_draw_color(color);
    }

    pub fn clear(&mut self) {
        self.canvas.clear();
    }

    pub fn set_color_mod(&mut self, color: Color) {
        let (r, g, b, _) = color.rgba();
        self.texture.set_color_mod(r, g, b);
    }

    pub fn set_initial_rendering_location(&mut self, scroller: &Scroller) {
        // TODO:
        // Abstract out where we start drawing
        // so we could change pane look easily
        self.target = Rect::new(
            self.bounds.editor_left_margin as i32,
            (scroller.line_fraction_y(&self.bounds) as i32).neg() + self.bounds.letter_height as i32 * 2,
            self.bounds.letter_width as u32,
            self.bounds.letter_height as u32
        );
    }

    pub fn copy(&mut self, source: &Rect) -> Result<(), String> {
        self.canvas.copy(&self.texture, *source, self.target)
    }

    pub fn fill_rect(&mut self, rect: &Rect) -> Result<(), String> {
        self.canvas.fill_rect(*rect)
    }

    pub fn draw_rect(&mut self, rect: &Rect) -> Result<(), String> {
        self.canvas.draw_rect(*rect)
    }

    pub fn move_right(&mut self, padding: i32) {
        self.target.set_x(self.target.x() + padding);
    }
    pub fn move_left(&mut self, padding: i32) {
        self.target.set_x(self.target.x().saturating_sub(padding));
    }

    pub fn move_down(&mut self, padding: i32) {
        self.target.set_y(self.target.y() + padding);
    }

    pub fn move_right_one_char(&mut self) {
        self.move_right(self.bounds.letter_width as i32);
    }
    pub fn move_down_one_line(&mut self) {
        self.move_down(self.bounds.letter_height as i32);
    }

     pub fn set_x(&mut self, x: i32) {
        self.target.set_x(x);
    }

     pub fn set_y(&mut self, x: i32) {
        self.target.set_y(x);
    }


     pub fn char_position_in_atlas(&self, c: char) -> Rect {
        Rect::new(self.bounds.letter_width as i32 * (c as i32 - 33), 0, self.bounds.letter_width as u32, self.bounds.letter_height as u32)
    }

     pub fn draw_string(&mut self, text: &str) -> Result<(), String> {
        for char in text.chars() {
            self.move_right_one_char();
            self.copy(&self.char_position_in_atlas(char))?
        }
        Ok(())
    }

     pub fn draw_fps(&mut self, fps: usize, window: &Window) -> Result<(), String> {
        // Do something better with this target
        self.target = Rect::new(window.width - (self.bounds.letter_width * 10) as i32, 0, self.bounds.letter_width as u32, self.bounds.letter_height as u32);
        self.set_color_mod(STANDARD_TEXT_COLOR);
        self.draw_string(&format!("fps: {}", fps))
    }

     pub fn draw_column_line(&mut self, pane_manager: &mut PaneManager) -> Result<(), String> {
        self.target = Rect::new(pane_manager.window.width - (self.bounds.letter_width * 22) as i32, pane_manager.window.height-self.bounds.letter_height as i32, self.bounds.letter_width as u32, self.bounds.letter_height as u32);
        if let Some(pane) = pane_manager.get_active_pane() {
            if let Some(Cursor(cursor_line, cursor_column)) = pane.cursor_context.cursor {
                self.draw_string( &format!("Line {}, Column {}", cursor_line, cursor_column))?;
            }
        }
        Ok(())
    }

     pub fn present(&mut self) {
        self.canvas.present();
    }

    pub fn set_cursor_pointer(&mut self) {
        self.system_cursor = sdl2::mouse::Cursor::from_system(SystemCursor::Hand).unwrap();
        self.system_cursor.set();
    }


    pub fn set_cursor_ibeam(&mut self) {
        self.system_cursor = sdl2::mouse::Cursor::from_system(SystemCursor::IBeam).unwrap();
        self.system_cursor.set();
    }


}