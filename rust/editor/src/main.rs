use std::{cmp::{max, min}, fs, str::from_utf8, convert::TryInto};
use std::fmt::Debug;

use rand::Rng;
use sdl2::{mouse::SystemCursor, pixels::{Color}, rect::Rect, sys};
use tokenizer::{Tokenizer, rust_specific_pass, RustSpecific, Token};


use tiny_http::{Server, Response};
use matchit::Node;

// cargo build --message-format=json | jq 'select(.reason == "compiler-message")' | jq '.message' | jq '.spans' | jq ".[]" | jq '"rect canvas \(.line_start) \(.column_start) \(.line_end) \(.column_end) 1"'  | tr -d '"'

mod native;
mod sdl;
mod tokenizer;
mod renderer;
mod scroller;
mod text_buffer;
mod transaction;
mod cursor;
mod fps;
mod color;
mod event;

use renderer::{Renderer, EditorBounds, digit_count};
use scroller::Scroller;
use text_buffer::TextBuffer;
use transaction::{EditAction, TransactionManager};
use cursor::{Cursor, CursorContext};
use fps::FpsCounter;
use event::{SideEffectAction, handle_events, handle_side_effects, handle_per_frame_actions, PerFrameActionResult};

// I really want so debugging panels.
// Should probably invest in that.
// Make it work automatically with Debug.




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




#[derive(Debug, Clone, Copy)]
pub struct Window {
    width: i32,
    height: i32,
}

impl Window {
    fn resize(&mut self, width: i32, height: i32) {
        self.width = width;
        self.height = height;
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

    fn new(name: String, (x, y): (i32, i32), (width, height): (usize, usize), text: &str, active: bool) -> Self {

        Pane {
            name,
            scroller: Scroller::new(),
            cursor_context: CursorContext::new(),
            text_buffer: TextBuffer::new(text),
            position: (x, y),
            width,
            height,
            active,
            mouse_pos: None,
            transaction_manager: TransactionManager::new(),
            editing_name: false,
            draw_commands: vec![],
        }
    }

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
            match *draw_command {
                DrawCommand::RectOnPaneAtLocation(_, start_line, start_column, end_line, end_column, height) => {
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
            renderer.set_color_mod(color::STANDARD_TEXT_COLOR);
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
                renderer.set_draw_color(color::CURSOR_COLOR);


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
        

        renderer.set_draw_color(color::CURSOR_COLOR);
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
            renderer.set_draw_color(color::CURSOR_COLOR);
            renderer.draw_rect(&Rect::new(
                self.width as i32 - (renderer.bounds.letter_width * (self.name.len() + 2)) as i32 - padding,
                0,
                renderer.bounds.letter_width as u32 * (self.name.len() + 1) as u32 + padding as u32,
                renderer.bounds.letter_height as u32 + padding as u32))?;
        }
        renderer.set_color_mod(color::STANDARD_TEXT_COLOR);
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

        renderer.draw_triangle(into_i16(play_button_x),into_i16(play_button_y), color::STANDARD_TEXT_COLOR)?;

        // TODO: I translated play_button_x and play_button_y to local coords
        // But mouse_pos is not locally translated.
        // Need to do this better in general.
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

                renderer.set_draw_color(color::CURSOR_COLOR);
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
                let color = color::color_for_token(&token, &self.text_buffer.chars);
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
    renderer.set_draw_color(color::BACKGROUND_COLOR);
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
    renderer.draw_fps(fps.tick(), &pane_manager.window)?;
    // Does this belong in the pane?
    // Is it global?
    // Need to think about the UI
    renderer.draw_column_line(pane_manager)?;

    renderer.present();

    Ok(())
}
// Need a draw order for z-index purposes.
pub struct PaneManager {
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
        let scroller = Scroller::new();
        let cursor_context = CursorContext::new();

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
        renderer.set_draw_color(color::CURSOR_COLOR);
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
    let system_cursor = sdl2::mouse::Cursor::from_system(SystemCursor::IBeam).unwrap();
    system_cursor.set();

    let clipboard = video.clipboard();

    let bounds = EditorBounds {
        editor_left_margin: 10,
        line_number_gutter_width : 20,
        letter_height,
        letter_width,
    };


    let text = fs::read_to_string("/Users/jimmyhmiller/Documents/Code/Playground/rust/editor/src/main.rs").unwrap();

    let mut fps = FpsCounter::new();

    let pane1 = Pane::new("test_draw".to_string(), (100, 100), (500, 500), "rect canvas 2 1 2 4 1", true);
    let pane2 = Pane::new("canvas".to_string(), (650, 100), (500, 500), &text, false);

    let mut renderer = Renderer {
        canvas,
        texture,
        target: Rect::new(0, 0, 0, 0),
        bounds,
        system_cursor,
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


    let server = Server::http("0.0.0.0:8000").unwrap();


    enum HttpRoutes {
        GetPane
    }

    let mut matcher = Node::new();
    matcher.insert("/panes/:pane_name", HttpRoutes::GetPane).ok();

    loop {


        // Silly little pattern for dealing with a locally scoped
        // optional. Once try try blocks are stable don't need this iffe
        (|| {
            let request = server.try_recv().ok()??;
            let route = matcher.at(request.url()).ok();
            match route.map(|x| (x.value, x.params)) {
                Some((HttpRoutes::GetPane, params)) => {
                    let pane = pane_manager.get_pane_by_name(params.get("pane_name")?.to_string())?;
                    let response = Response::from_string(pane.text_buffer.get_text());
                    request.respond(response).ok()?;
                    Some(())
                }
                None => request.respond(Response::from_string("Not Found").with_status_code(404)).ok()
            }
        })();

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

