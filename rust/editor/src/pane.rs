use std::{cmp::{max, min}, str::from_utf8, convert::TryInto};

use sdl2::{sys, rect::Rect, pixels::Color};

use crate::{scroller::Scroller, cursor::CursorContext, text_buffer::TextBuffer, transaction::TransactionManager, DrawCommand, tokenizer::{Token, rust_specific_pass}, renderer::{EditorBounds, Renderer, digit_count}, color};


// TODO:
// I want to be able to have panes that aren't text panes
// From what I can tell in order for a pane_manager to manage panes
// we need to following properties:
// pane.active
// pane.id
// pane.is_mouse_over
// pane.name
// pane.position
// pane.width
// pane.height

// That is a pretty reasonable list. 
// Of course there are more uses of panes
// but since they are accessed through the pane_manager
// we can just make them go through a get_text_pane
// function and not return anything if they expect a text pane.




fn in_square(mouse_pos: (i32, i32), square_pos: (i32, i32), square_size: i32) -> bool {
    let (x, y) = mouse_pos;
    let (x_pos, y_pos) = square_pos;
    let size = square_size;
    x >= x_pos && x <= x_pos + size && y >= y_pos && y <= y_pos + size
}



#[derive(Debug, Clone)]
pub struct EmptyPane {
    pub name: String,
    pub id: usize,
    pub position: (i32, i32),
    pub width: usize,
    pub height: usize,
    pub active: bool,
}

#[derive(Debug, Clone)]
pub struct PickerPane {
    pub name: String,
    pub id: usize,
    pub position: (i32, i32),
    pub width: usize,
    pub height: usize,
    pub active: bool,
}

impl PickerPane {
    pub fn draw_with_texture(&self, renderer: &mut Renderer) -> Result<(), String> {
        let rect = Rect::new(self.position.0, self.position.1, self.width as u32, self.height as u32);
        renderer.set_draw_color(color::GREEN_TEXT_COLOR);
        renderer.fill_rect(&rect)?;
        Ok(())
    }
}



impl EmptyPane {
    pub fn draw_with_texture(&self, renderer: &mut Renderer) -> Result<(), String> {
        let rect = Rect::new(self.position.0, self.position.1, self.width as u32, self.height as u32);
        renderer.set_draw_color(color::CURSOR_COLOR);
        renderer.draw_rect(&rect)?;
        Ok(())
    }
}


// Trying out this trait to see if this is a good idea.
// Still not sure.
pub trait AdjustablePosition {
    fn position(&self) -> (i32, i32);
    fn adjust_position(&self, x: i32, y: i32, bounds: &EditorBounds) -> (usize, usize) {
        let position = self.position();
        (max(0, x - position.0) as usize, max(0, y - position.1 - (bounds.letter_height * 2) as i32) as usize)
    }
}


// Wouldn't it be great to be able to duck type here?
impl AdjustablePosition for Pane {
    fn position(&self) -> (i32, i32) {
        match self {
            Pane::Text(tp) => tp.position,
            Pane::_Empty(ep) => ep.position,
            Pane::_PanePicker(pp) => pp.position,
        }
    }
}




#[derive(Debug, Clone)]
pub enum Pane {
    Text(TextPane),
    _Empty(EmptyPane),
    _PanePicker(PickerPane)
}
// I could also do this as a trait instead
// Not sure which is better
// but shouldn't be hard to try both over time.

impl Pane {

    pub fn get_text_pane(&self) -> Option<&TextPane> {
        match self {
            Pane::Text(pane) => Some(pane),
            _ => None,
        }
    }

    pub fn get_text_pane_mut(&mut self) -> Option<&mut TextPane> {
        match self {
            Pane::Text(pane) => Some(pane),
            _ => None,
        }
    }

    pub fn editing_name(&self) -> bool {
        match self {
            Pane::Text(tp) => tp.editing_name,
            _ => false,
        }
    }

    pub fn id(&self) -> usize {
        match self {
            Pane::Text(tp) => tp.id,
            Pane::_Empty(ep) => ep.id,
            Pane::_PanePicker(pp) => pp.id,
        }
    }

    pub fn set_id(&mut self, id: usize) {
        match self {
            Pane::Text(tp) => tp.id = id,
            Pane::_Empty(ep) => ep.id = id,
            Pane::_PanePicker(pp) => pp.id = id,
        }
    }

    pub fn set_position(&mut self, x: i32, y: i32) {
        match self {
            Pane::Text(tp) => tp.position = (x, y),
            Pane::_Empty(ep) => ep.position = (x, y),
            Pane::_PanePicker(pp) => pp.position = (x, y),
        }
    }
    pub fn width(&self) -> usize {
        match self {
            Pane::Text(tp) => tp.width,
            Pane::_Empty(ep) => ep.width,
            Pane::_PanePicker(pp) => pp.width,
        }
    }

    pub fn set_width(&mut self, width: usize) {
        match self {
            Pane::Text(tp) => tp.width = width,
            Pane::_Empty(ep) => ep.width = width,
            Pane::_PanePicker(pp) => pp.width = width,
        }
    }

    pub fn height(&self) -> usize {
        match self {
            Pane::Text(tp) => tp.height,
            Pane::_Empty(ep) => ep.height,
            Pane::_PanePicker(pp) => pp.height,
        }
    }

    pub fn set_height(&mut self, height: usize) {
        match self {
            Pane::Text(tp) => tp.height = height,
            Pane::_Empty(ep) => ep.height = height,
            Pane::_PanePicker(pp) => pp.height = height,
        }
    }

    pub fn name(&self) -> String {
        match self {
            Pane::Text(tp) => tp.name.clone(),
            Pane::_Empty(ep) => ep.name.clone(),
            Pane::_PanePicker(pp) => pp.name.clone(),
        }
    }

    pub fn set_name(&mut self, name: String) {
        match self {
            Pane::Text(tp) => tp.name = name,
            Pane::_Empty(ep) => ep.name = name,
            Pane::_PanePicker(pp) => pp.name = name,
        }
    }

    pub fn draw_with_texture(&mut self, renderer: &mut Renderer) -> Result<(), String> {
        match self {
            Pane::Text(tp) => tp.draw_with_texture(renderer),
            Pane::_Empty(ep) => ep.draw_with_texture(renderer),
            Pane::_PanePicker(pp) => pp.draw_with_texture(renderer),
        }
    }

    pub fn _active(&self) -> bool {
        match self {
            Pane::Text(tp) => tp.active,
            Pane::_Empty(ep) => ep.active,
            Pane::_PanePicker(pp) => pp.active,
        }
    }

    pub fn set_active(&mut self, active: bool) {
        match self {
            Pane::Text(tp) => tp.active = active,
            Pane::_Empty(ep) => ep.active = active,
            Pane::_PanePicker(pp) => pp.active = active,
        }
    }


    pub fn is_mouse_over(&self, (x, y): (i32, i32), bounds: &EditorBounds, scale_factor: f32) -> bool {
        let x = x as f32;
        let y = y as f32;
        let position = self.position();
        let position_x = position.0 as f32; // * scale_factor;
        let position_y = position.1 as f32; // * scale_factor;
        let width = self.width() as f32; // * scale_factor;
        let height = self.height() as f32; // * scale_factor;
        
        x > position_x &&
        x < position_x + width &&
        y > position_y - bounds.letter_height as f32 * scale_factor &&
        y < position_y + height + bounds.letter_height  as f32 * scale_factor
    }
}



#[derive(Debug, Clone)]
pub struct TextPane {
    pub id: usize,
    pub name: String,
    pub scroller: Scroller,
    pub cursor_context: CursorContext,
    pub text_buffer: TextBuffer,
    pub position: (i32, i32),
    pub width: usize,
    pub height: usize,
    pub active: bool,
    pub transaction_manager: TransactionManager,
    pub editing_name: bool,
    pub mouse_pos: Option<(i32, i32)>,
    pub draw_commands: Vec<DrawCommand>,
    pub tokens: Vec<Token>,
}

impl TextPane {

    pub fn new(id: usize, name: String, (x, y): (i32, i32), (width, height): (usize, usize), text: &str, active: bool) -> Self {

        let mut pane = TextPane {
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

    pub fn max_characters_per_line(&self, bounds: &EditorBounds) -> usize {
        let padding = bounds.line_number_padding(&self.text_buffer);
        let extra_padding = bounds.letter_width;
        if self.width < padding + extra_padding {
           0
        } else {
            ((self.width - padding - extra_padding) / bounds.letter_width) + 2
        }

    }

    pub fn max_lines_per_page(&self, bounds: &EditorBounds) -> usize {
        self.height / bounds.letter_height as usize
    }
    

    pub fn draw_with_texture(&mut self, renderer: &mut Renderer) -> Result<(), String> {
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

    pub fn parse_all(&mut self) {
        // I could allocate less here.
        self.tokens = self.text_buffer.tokenizer.parse_all(&self.text_buffer.chars);
    }

    pub fn draw(&mut self, renderer: &mut Renderer) -> Result<(), String> {


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
        pub fn into_i16(x: i32) -> i16 {
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
    pub fn draw_line_numbers(&mut self, renderer: &mut Renderer, line_number_digits: usize, line: usize) -> Result<(), String> {
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


    pub fn draw_cursor(&mut self, renderer: &mut Renderer, line: usize, line_number_padding: usize) -> Result<(), String> {
        if !self.active {
            return Ok(());
        }

        let offset_x = self.scroller.offset_x;
        
        if let Some(cursor) = self.cursor_context.cursor {
            if cursor.0 == line {
                let cursor_x = cursor.1 as i32  * renderer.bounds.letter_width as i32 + line_number_padding as i32 - offset_x;
                let cursor_y = renderer.target.y();
                if cursor_x < line_number_padding as i32 {
                    return Ok(());
                }

                renderer.set_draw_color(color::CURSOR_COLOR);
                renderer.fill_rect(&Rect::new(cursor_x as i32, cursor_y as i32, 2, renderer.bounds.letter_height as u32))?
            }
        }
        Ok(())
    }

    pub fn draw_selection(&mut self, renderer: &mut Renderer, line: usize, line_number_padding: usize) -> Result<(), String> {


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
                    // had a panic. Changed to saturating sub
                    (self.text_buffer.line_length(line).saturating_sub(start_column)) * renderer.bounds.letter_width as usize
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


    pub fn draw_code(&mut self, renderer: &mut Renderer, line: usize, token_position: usize) -> Result<usize, String> {

        let tokens = &self.tokens;
        let (line_start, line_end) = self.text_buffer[line];
        let start = min(line_start + self.scroller.scroll_x_character(&renderer.bounds), line_end);
        let end = min(line_end, start + self.max_characters_per_line(&renderer.bounds));
        let mut position = line_start;

        // TODO: fix multiline
        // I handle multi line tokens
        // but not if they are more than the view pane.
        // Need to deal with that fact.
        // Might be from skipping lines incorrectly?
        // Need to look at that.
        
        
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

    pub fn mouse_over_play_button(&self, mouse_pos: (i32, i32), bounds: &EditorBounds) -> bool {
        // probably better to make this work with adjusted positions
        let play_button_x = self.position.0 + bounds.letter_width as i32;
        let play_button_y = self.position.1 - bounds.letter_height as i32 + 4;
        in_square(mouse_pos, (play_button_x, play_button_y), 10)
    }

    pub fn adjust_position(&self, x: i32, y: i32, bounds: &EditorBounds) -> (usize, usize) {
        (max(0, x - self.position.0) as usize, max(0, y - self.position.1 - (bounds.letter_height * 2) as i32) as usize)
    }



}
