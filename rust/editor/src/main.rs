#![allow(clippy::single_match)]

use std::{cmp::{max, min}, fs, str::from_utf8, convert::TryInto};
use std::fmt::Debug;

use rand::Rng;
use sdl2::{pixels::{Color}, rect::Rect, sys};
use tokenizer::{Tokenizer, rust_specific_pass, Token};


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
use event::{Action, handle_events, handle_side_effects, handle_per_frame_actions, PerFrameAction};



// I really want so debugging panels.
// Should probably invest in that.
// Make it work automatically with Debug.
// Is there any generic way we could do this?
// Expose all of state (would that get into infinite loop territory?)
// What would that look like? Would it be useful?
// Also, can we have a generic tree view component easily?

// TODO LIST:
// Add some spacing between letters!
// It would be pretty cool to add a minimap
// Need toggle line numbers
// Need references to panes
// Need canvas scrolling?
// Need to think about undo and pane positions
// I need to experiment with non-text panes
// I also need to think about the coordinate system
// It being upside down is annoying
// I need to think about afterburner text decorations
// I could have queries and panes as the results of those queries
// Need a command interface. But what to do it enso style
// Select word via multiclick
// Think about auto indention
// paredit
// comment line
// cut
// paste isn't working first try everytime (or was this the active bug?)
// Highlight matching brackets
// Deindent
// At some point I made scroll not as smooth. There are no fractional top lines
// Scroll left and right with arrow keys
// LOTS of cpu usage. Need to debug and optimize


// Bug
// For some reason when running a program the stdout stops at a certain length
// Example program
// #!/usr/bin/env node


// const sleep = (millis) => {
//    return new Promise(resolve => {
//       setTimeout(resolve, millis)  
//    })
// }
// let size = 100;
// let width = 1000;
// let height = 1000;
// let x = 0;
// let y = 0;
// let vx = 4;
// let vy = 10;

// const myFunction = async () => {
//     while (true) {
//         x += vx;
//         y += vy;
//         if (x >= width || x < 0) { vx *= -1}
//         if (y >= width || y < 0) { vy *= -1}
//         await sleep(32)
//        // console.log('\x0c')
//         console.log(`rect ${x} ${y} 100 100`)
//    }
// }


// myFunction()




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
    id: usize,
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
    tokens: Vec<Token>,
}


// Thoughts:
// I don't have any notion of bounds to stop rendering at.
// I need to think about that here.
impl Pane {

    fn new(id: usize, name: String, (x, y): (i32, i32), (width, height): (usize, usize), text: &str, active: bool) -> Self {

        let mut pane = Pane {
            id,
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
            tokens: vec![],
        };
        pane.parse_all();
        pane
    }

    fn adjust_position(&self, x: i32, y: i32, bounds: &EditorBounds) -> (usize, usize) {
        (max(0, x - self.position.0) as usize, max(0, y - self.position.1 - (bounds.letter_height * 2) as i32) as usize)
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

    fn parse_all(&mut self) {
        // I could allocate less here.
        self.tokens = self.text_buffer.tokenizer.parse_all(&self.text_buffer.chars);
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


        // This whole lines vs tokenizer thing needs to go. I need to get rid of line parsing.

        // Now I could cache this if the pane doesn't change
        self.parse_all();
        
        let mut token_position = 0;
        let mut new_lines_found = 0;
        for token in self.tokens.iter() {
            if new_lines_found == starting_line {
                break;
            }
            if matches!(token, Token::NewLine) {
                new_lines_found += 1;
            }
            token_position += 1;
        }


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
            token_position = self.draw_code(renderer, line, token_position)?;
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
                    // I have a panic here not sure why
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


    fn draw_code(&mut self, renderer: &mut Renderer, line: usize, token_position: usize) -> Result<usize, String> {

        let tokens = &self.tokens;
        let (line_start, line_end) = self.text_buffer[line];
        let start = min(line_start + self.scroller.scroll_x_character(&renderer.bounds), line_end);
        let end = min(line_end, start + self.max_characters_per_line(&renderer.bounds));
        let mut position = line_start;

        // I need to handle multi line tokens. So, I would be looking at start and deciding to draw part of the token.
        // Will get there eventually.
        
        let mut is_multi_line = false;

        let mut token_position = token_position;
        let mut current_token = tokens.get(token_position);
        while current_token.is_some() && !matches!(current_token, Some(Token::NewLine)) && token_position < tokens.len() && !is_multi_line {
            let token = current_token.unwrap();
            let token = rust_specific_pass(*token, &self.text_buffer.chars);
            let color = color::color_for_token(&token, &self.text_buffer.chars);
            let mut text = token.to_string(&self.text_buffer.chars);
        

            let initial_token_start = token.get_token_start(position);

            if line_start > initial_token_start {
                text = from_utf8(&self.text_buffer.chars[line_start..token.get_token_end(initial_token_start)]).unwrap();
            }

            let token_start = {
                if position < start && position + text.len() < start {
                    None
                } else if position <= start {
                    Some(start - position)
                } else {
                    Some(0)
                }
            };
            
    
            // Revisit this
            let token_end = {
                if position + text.len() > end {
                    min(end.saturating_sub(position), text.len())
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
            if let Some(token_start) = token_start {
                if text[token_start..].contains("\n") {
                    is_multi_line = true;
                }
            }
            if !is_multi_line {
                token_position += 1;
                current_token = tokens.get(token_position);
            }
        }
        if matches!(current_token, Some(Token::NewLine)) {
            token_position += 1;
        }

        Ok(token_position)


    }

    fn is_mouse_over(&self, (x, y): (i32, i32), bounds: &EditorBounds) -> bool {
        x > self.position.0  &&
        x < self.position.0 + self.width as i32 &&
        y > self.position.1 - bounds.letter_height as i32 &&
        y < self.position.1 + self.height as i32 + bounds.letter_height  as i32
    }

    fn mouse_over_play_button(&self, mouse_pos: (i32, i32), bounds: &EditorBounds) -> bool {
        // probably better to make this work with adjusted positions
        let play_button_x = self.position.0 + bounds.letter_width as i32;
        let play_button_y = self.position.1 - bounds.letter_height as i32 + 4;
        in_square(mouse_pos, (play_button_x, play_button_y), 10)
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


#[derive(Debug, Clone)]
pub enum PaneSelector {
    Active,
    Id(usize),
    AtMouse((i32, i32)),
    Scroll,
}

// I should consider changing pane_manager to use ids
// instead of indexes
// Then should they be a map? Or is it still better
// to have an array?

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
    pane_id_counter: usize,
}

impl PaneManager {

    fn new(panes: Vec<Pane>, window: Window) -> Self {
        PaneManager {
            panes,
            window,
            active_pane: 0,
            scroll_active_pane: 0,
            dragging_pane: None,
            dragging_start: (0, 0),
            dragging_pane_start: (0, 0),
            resize_pane: None,
            resize_start: (0, 0),
            create_pane_activated: false,
            create_pane_start: (0, 0),
            create_pane_current: (0, 0),
            pane_id_counter: 0,
        }
    }

    fn get_pane_index_by_id(&self, pane_id: usize) -> Option<usize> {
        for (index, pane) in self.panes.iter().enumerate() {
            if pane.id == pane_id {
                return Some(index);
            }
        }
        None
    }

    fn delete_pane(&mut self, pane_id: usize) {
        if let Some(pane) = self.get_pane_index_by_id(pane_id) {
            self.panes.remove(pane);
        }
    }



    fn set_active_by_id(&mut self, pane_id: usize) {
        if let Some(i) = self.get_pane_index_by_id(pane_id) {
            self.active_pane = i;
            if let Some(pane) = self.panes.get_mut(i) {
                pane.active = true;
            }
            for pane in self.panes.iter_mut() {
                if pane.id != pane_id {
                    pane.active = false;
                }
            }
        }
    }

    fn set_scroll_active_by_id(&mut self, pane_id: usize) {
        if let Some(i) = self.get_pane_index_by_id(pane_id) {
            self.scroll_active_pane = i;
        }
    }

    fn get_pane_index_at_mouse(&self, mouse_pos: (i32, i32), bounds: &EditorBounds) -> Option<usize> {
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
    fn get_pane_at_mouse(&self, mouse_pos: (i32, i32), bounds: &EditorBounds) -> Option<&Pane> {
        if let Some(i) = self.get_pane_index_at_mouse(mouse_pos, bounds) {
            return self.panes.get(i);
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
    fn get_scroll_active_pane(&self) -> Option<&Pane> {
        self.panes.get(self.scroll_active_pane)
    }

    fn get_active_pane(&self) -> Option<&Pane> {
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

    fn set_resize_start(&mut self, mouse_pos: (i32, i32), pane_id: usize) -> bool {
        if let Some(i) = self.get_pane_index_by_id(pane_id) {
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

            pane.width = max(width as usize, 20);
            pane.height = max(height as usize, 20);
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

        let id = self.new_pane_id();


        // Can I use Pane::new here?
        self.panes.push(Pane {
            id,
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
            tokens: vec![],
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
            if width < 20 || height < 20 {
                return;
            }

            let mut rng = rand::thread_rng();
            self.create_pane_raw(format!("temp-{}", rng.gen::<u32>()), (position_x, position_y), width, height);
            self.active_pane = self.panes.len() - 1;
        }
    }

    fn _remove(&mut self, i: usize) -> Pane {
        // Should swap_remove but more complicated
        let pane = self.panes.remove(i);
        if i < self.active_pane {
            self.active_pane -= 1;
        }
        pane
    }

    fn _insert(&mut self, i: usize, pane: Pane) {
        if pane.active {
            self.active_pane = i;
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


    fn get_pane_by_id_mut(&mut self, pane_id: usize) -> Option<&mut Pane> {
        for pane in self.panes.iter_mut() {
            if pane.id == pane_id {
                return Some(pane);
            }
        }
        None
    }
    fn get_pane_by_id(&self, pane_id: usize) -> Option<&Pane> {
        for pane in self.panes.iter() {
            if pane.id == pane_id {
                return Some(pane);
            }
        }
        None
    }

    fn get_pane_by_name(&self, pane_name: &str) -> Option<&Pane> {
        for pane in self.panes.iter() {
            if pane.name.starts_with(&pane_name) {
                return Some(pane);
            }
        }
        None
    }

    fn new_pane_id(&mut self) -> usize {
        self.pane_id_counter += 1;
        self.pane_id_counter
    }

    fn get_pane_by_selector_mut(&mut self, pane_selector: &PaneSelector, editor_bounds: &EditorBounds) -> Option<&mut Pane> {
        match pane_selector {
            PaneSelector::Active => self.get_active_pane_mut(),
            PaneSelector::Id(id) => self.get_pane_by_id_mut(*id),
            PaneSelector::AtMouse(mouse_pos) => self.get_pane_at_mouse_mut(*mouse_pos, editor_bounds),
            PaneSelector::Scroll => self.get_scroll_active_pane_mut(),
        }
    }

    fn get_pane_by_selector(&self, pane_selector: &PaneSelector, editor_bounds: &EditorBounds) -> Option<&Pane> {
        match pane_selector {
            PaneSelector::Active => self.get_active_pane(),
            PaneSelector::Id(id) => self.get_pane_by_id(*id),
            PaneSelector::AtMouse(mouse_pos) => self.get_pane_at_mouse(*mouse_pos, editor_bounds),
            PaneSelector::Scroll => self.get_scroll_active_pane(),
        }
    }

    fn clear_active(&mut self) {
        self.active_pane = 0;
        for pane in self.panes.iter_mut() {
            pane.active = false;
        }
    }
}



// I can very easily generalize this.

fn handle_transaction_pane(pane_manager: &mut PaneManager) -> Option<()> {

    let mut chars: Vec<u8> = Vec::new();

    let transaction_pane_id = pane_manager.get_pane_by_name( "transaction_pane")?.id;

    let active_pane = pane_manager.get_active_pane_mut()?;
    let transaction_manager = &active_pane.transaction_manager;

    chars.extend(format!("current: {}, pointer: {}\n",
        transaction_manager.current_transaction,
        transaction_manager.transaction_pointer).as_bytes());

    for transaction in active_pane.transaction_manager.transactions.iter() {
        chars.extend(format!("{:?}\n", transaction).as_bytes());
    }
   
    let mut transaction_pane = pane_manager.get_pane_by_id_mut(transaction_pane_id)?;
    transaction_pane.text_buffer.chars = chars;
    transaction_pane.text_buffer.parse_lines();
    Some(())
}

// I need to think about how to generalize this
fn handle_action_pane(pane_manager: &mut PaneManager, actions: &[Action], editor_bounds: &EditorBounds) -> Option<()>{

    let mut chars: Vec<u8> = Vec::new();

    let action_pane_id = pane_manager.get_pane_by_name( "action_pane")?.id;
 
    for action in actions.iter() {
        if matches!(action, Action::MoveMouse(_) | Action::SetScrollPane(_)) {
            continue;
        }
       
        // TODO:
        // I need to resolve these ids but keep around the fact that
        // these were using meta-selectors
        // Otherwise these things will be wrong.
        if Some(action_pane_id) == action.pane_id(pane_manager, editor_bounds) {
            continue;
        }
        chars.extend(format!("{:?}\n", action).as_bytes());
    }

    let mut action_pane = pane_manager.get_pane_by_id_mut(action_pane_id)?;
    action_pane.text_buffer.chars = chars;
    action_pane.text_buffer.parse_lines();

    Some(())
}



fn handle_token_pane(pane_manager: &mut PaneManager) -> Option<()> {

    let mut chars: Vec<u8> = Vec::new();

    let token_pane_id = pane_manager.get_pane_by_name( "token_pane")?.id;
 
    let active_pane = pane_manager.get_active_pane_mut()?;
    let tokenizer = &mut active_pane.text_buffer.tokenizer;
    while !tokenizer.at_end(&active_pane.text_buffer.chars) {
        let token = tokenizer.parse_single(&active_pane.text_buffer.chars)?;
        chars.extend(format!("{:?} ", token).as_bytes());
        if matches!(token, Token::NewLine) {
            chars.extend(b"\n");
        }
    }
    tokenizer.position = 0;

    let mut token_pane = pane_manager.get_pane_by_id_mut(token_pane_id)?;
    token_pane.text_buffer.chars = chars;
    token_pane.text_buffer.parse_lines();

    Some(())
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


enum HttpRoutes {
    GetPane
}

fn process_http_request(server: &Server, matcher: &Node<HttpRoutes>, pane_manager: &mut PaneManager) -> Option<()> {
    let request = server.try_recv().ok()??;
    let route = matcher.at(request.url()).ok();
    match route.map(|x| (x.value, x.params)) {
        Some((HttpRoutes::GetPane, params)) => {
            let pane = pane_manager.get_pane_by_name(params.get("pane_name")?)?;
            let response = Response::from_string(pane.text_buffer.get_text());
            request.respond(response).ok()?;
            Some(())
        }
        None => request.respond(Response::from_string("Not Found").with_status_code(404)).ok()
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
        clipboard,
        system_cursor,
    } = sdl::setup_sdl(window.width as usize, window.height as usize)?;

    let (mut texture, letter_width, letter_height) = sdl::draw_font_texture(&texture_creator, ttf_context)?;
    texture.set_color_mod(167, 174, 210);

    // If this gets dropped, the cursor resets.
    let bounds = EditorBounds {
        editor_left_margin: 10,
        line_number_gutter_width : 20,
        letter_height,
        letter_width,
    };


    let text = fs::read_to_string("/Users/jimmyhmiller/Documents/Code/Playground/rust/editor/src/main.rs").unwrap();

    let mut fps = FpsCounter::new();

    // ids are large enough we shouldn't have duplicates here.
    // This is of course just test code.
    let pane1 = Pane::new(12352353, "transaction_pane".to_string(), (100, 100), (500, 500), "", true);
    let pane2 = Pane::new(12352353353, "canvas".to_string(), (650, 100), (500, 500), &text, false);


    let mut renderer = Renderer {
        canvas,
        texture,
        target: Rect::new(0, 0, 0, 0),
        bounds,
        system_cursor,
    };

    let mut pane_manager = PaneManager::new(
        vec![pane1, pane2],
        window,
    );

    let mut per_frame_actions: Vec<PerFrameAction> = vec![];

    // TODO: HTTP Server Struct?
    let server = Server::http("0.0.0.0:8000").unwrap();
    let mut matcher = Node::new();
    matcher.insert("/panes/:pane_name", HttpRoutes::GetPane).ok();


    // Might not want to keep all of these around forever.
    let mut all_actions: Vec<Action> = vec![];

    loop {

        process_http_request(&server, &matcher, &mut pane_manager);
      
        // Set this each frame so that something can change it if they want
        renderer.set_cursor_ibeam();

        handle_transaction_pane(&mut pane_manager);
        handle_token_pane(&mut pane_manager);
        handle_action_pane(&mut pane_manager, &all_actions, &renderer.bounds);
      

        draw(&mut renderer, &mut pane_manager, &mut fps)?;
        
        
        let mut actions = handle_events(&mut event_pump);
        let mut i = 0;
        while i < actions.len() {
            // I might need to resolve all of these selectors to id
            // selectors before I process? Or After? It isn't clear.
            // But I will need to record the ids at some point
            // So I can know which panes changed for my dependencies
            if let Some(new_actions) = actions[i].process(&mut pane_manager, &renderer.bounds, &clipboard) {
                for (j, action) in new_actions.into_iter().enumerate() {
                    actions.insert(i + j + 1, action);
                }
            }
            i += 1;
        }
        all_actions.extend(actions.clone());
        handle_side_effects(&mut pane_manager, &renderer.bounds, actions, &mut per_frame_actions);
        handle_per_frame_actions(&mut per_frame_actions, &mut pane_manager);
        

    }
}
