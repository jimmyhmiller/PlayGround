#![allow(clippy::single_match)]

use std::{cmp::{max, min}, fs, str::from_utf8};
use std::fmt::Debug;

use pane::{TextPane, Pane};
use pane_manager::{PaneManager, PaneSelector};
use sdl2::{pixels::{Color}, rect::Rect};
use tokenizer::{Tokenizer, Token};


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
mod pane;
mod pane_manager;
mod ink;

use renderer::{Renderer, EditorBounds};
use scroller::Scroller;
use text_buffer::TextBuffer;
use transaction::{EditAction};
use cursor::{Cursor, CursorContext};
use fps::FpsCounter;
use event::{Action, handle_events, handle_per_frame_actions, PerFrameAction};



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
// Lots of cpu usage. Need to debug and optimize
// fix multiline
// capture stderr
// Got some weird undo around new lines
// Clean up this file
// Fix token pane
// Actually implment pane selection


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
    pane_manager.ink_manager.draw(renderer)?;

    // TODO:
    // Fix this whole scroller weirdness.
    // Really just need window here.
    renderer.draw_fps(fps.tick(), &pane_manager.window)?;
    // Does this belong in the pane?
    // Is it global?
    // Need to think about the UI
    renderer.draw_column_line(pane_manager)?;

    renderer.canvas.set_scale(pane_manager.scale_factor, pane_manager.scale_factor)?;

    renderer.present();

    Ok(())
}


// I should consider changing pane_manager to use ids
// instead of indexes
// Then should they be a map? Or is it still better
// to have an array?

// Need a draw order for z-index purposes.



fn special_pane_by_name(pane_manager: &mut PaneManager, name: &str, text_builder: fn(&mut TextPane, &mut Vec<u8>) -> Option<()>) -> Option<()> {
    let mut chars: Vec<u8> = Vec::new();
    let pane_id = pane_manager.get_pane_by_name( name)?.id();

    let active_pane = pane_manager.get_active_pane_mut()?;
    let active_pane = active_pane.get_text_pane_mut()?;

    text_builder(active_pane, &mut chars);

    let pane = pane_manager.get_pane_by_id_mut(pane_id)?;
    let pane = pane.get_text_pane_mut()?;
    pane.text_buffer.chars = chars;
    pane.text_buffer.parse_lines();

    Some(())
}



// I can very easily generalize this.

fn handle_transaction_pane(pane_manager: &mut PaneManager) -> Option<()> {

    special_pane_by_name(pane_manager, "transaction_pane", |pane, chars| {
        let transaction_manager = &pane.transaction_manager;
        chars.extend(format!("current: {}, pointer: {}\n",
        transaction_manager.current_transaction,
        transaction_manager.transaction_pointer).as_bytes());

        for transaction in pane.transaction_manager.transactions.iter() {
            chars.extend(format!("{:?}\n", transaction).as_bytes());
        }
        Some(())
    })
}



// TODO: THIS IS BROKEN!
fn handle_token_pane(pane_manager: &mut PaneManager) -> Option<()> {

    special_pane_by_name(pane_manager, "token_pane", |pane, chars| {

        let tokenizer = &mut pane.text_buffer.tokenizer;
        while !tokenizer.at_end(&pane.text_buffer.chars) {
            let token = tokenizer.parse_single(&pane.text_buffer.chars)?;
            chars.extend(format!("{:?} ", token).as_bytes());
            if matches!(token, Token::NewLine) {
                chars.extend(b"\n");
            }
        }
        tokenizer.position = 0;
        Some(())
    })
}

// Can't use special because I need actions
fn handle_action_pane(pane_manager: &mut PaneManager, actions: &[Action]) -> Option<()>{

    let mut chars: Vec<u8> = Vec::new();

    let action_pane_id = pane_manager.get_pane_by_name("action_pane")?.id();
 
    for action in actions.iter() {
        if matches!(action, Action::MoveMouse(_) | Action::SetScrollPane(_)) {
            continue;
        }
       
        // TODO:
        // I need to resolve these ids but keep around the fact that
        // these were using meta-selectors
        // Otherwise these things will be wrong.
        match action.pane_selector() {
            Some(PaneSelector::Id(id)) if id == action_pane_id  => {
                continue;
            }
            _ => {}
        }
        chars.extend(format!("{:?}\n", action).as_bytes());
    }

    let action_pane = pane_manager.get_pane_by_id_mut(action_pane_id)?;
    let action_pane = action_pane.get_text_pane_mut()?;
    action_pane.text_buffer.chars = chars;
    action_pane.text_buffer.parse_lines();

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
pub enum DrawCommand {
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
        if !pane.name().ends_with("_draw") {
            continue;
        }
        // Because we are tokenizing, this must be a text pane
        let pane = pane.get_text_pane_mut();
        if pane.is_none() {
            continue;
        }
        let pane = pane.unwrap();
        renderer.set_draw_color(color::CURSOR_COLOR);
        let tokenizer = &mut pane.text_buffer.tokenizer;
        let chars = &pane.text_buffer.chars;
        let mut filled = false;
        while !tokenizer.at_end( chars) {
            if let Some(Token::Atom((start, end))) = tokenizer.parse_single(chars) {
                let atom = &chars[start..end];
                if atom == b"filled" {
                    filled = true;
                } else if atom == b"unfilled" {
                    filled = false;
                } else if atom == b"rect" {
                    if let Some(draw_command) = parse_rect(tokenizer, chars) {
                        match draw_command {
                            DrawCommand::Rect(rect) => {
                                if filled { 
                                    renderer.fill_rect(&rect)?
                                } else {
                                    renderer.draw_rect(&rect)?
                                }
                            },
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
                if let Some(pane) = pane_manager.get_pane_by_name_mut(pane_name.clone()).and_then(|pane| pane.get_text_pane_mut()) {
                    pane.draw_commands.push(draw_command.clone())
                }
            }
            DrawCommand::RectOnPaneAtLocation(pane_name, _, _, _, _, _) => {
                if let Some(pane) = pane_manager.get_pane_by_name_mut(pane_name.clone()).and_then(|pane| pane.get_text_pane_mut()) {
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
            let pane = pane.get_text_pane()?;
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
    let pane1 = TextPane::new(12352353, "action_pane".to_string(), (100, 100), (500, 500), "", true);
    let pane2 = TextPane::new(12352353353, "canvas".to_string(), (650, 100), (500, 500), &text, false);


    let mut renderer = Renderer {
        canvas,
        texture,
        target: Rect::new(0, 0, 0, 0),
        bounds,
        system_cursor,
    };

    let mut pane_manager = PaneManager::new(
        vec![Pane::Text(pane1), Pane::Text(pane2)],
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
        handle_action_pane(&mut pane_manager, &all_actions);
      

        draw(&mut renderer, &mut pane_manager, &mut fps)?;
        
        
        let mut actions = handle_events(&mut event_pump);
        let mut i = 0;

        // By moving this up, I make sure to process the actions
        // on that same frame, instead of having to let them cross frames
        // that would get weird for dependency tracking.
        handle_per_frame_actions(&mut per_frame_actions, &mut pane_manager, &mut actions);

        while i < actions.len() {
            // Would love this to be outside the loop. Will deal for now
            let mut more_actions = vec![];
            // Side effects might also produce actions. Should consider passing the vector rather than making it?
            // That some times gets weird in a loop though.
            actions[i].process(&mut pane_manager, &renderer.bounds, &clipboard, &mut more_actions);
            actions[i].handle_side_effect(&mut pane_manager, &renderer.bounds, &mut per_frame_actions, &mut more_actions);
           
           
            for (j, action) in more_actions.into_iter().enumerate() {
                actions.insert(i + j + 1, action);
            }
            i += 1;
        }
        all_actions.extend(actions.clone());
        

    }
}
