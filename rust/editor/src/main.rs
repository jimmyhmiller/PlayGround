use std::{cmp::{max, min}, convert::TryInto, fs, ops::Neg};

use native_dialog::FileDialog;
use sdl2::{event::*, keyboard::*, mouse::{Cursor, SystemCursor}, pixels::Color, rect::Rect, render::*, video::*};


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
fn draw_string<'a>(canvas: & mut Canvas<Window>, target: &'a mut Rect, texture: &Texture, text: &str) -> &'a mut Rect {
    for char in text.chars() {
        let char_rect : Rect = render_char(target.width() as i32, target.height() as i32, char as char);
        target.set_x(target.x() + target.width() as i32);
        canvas.copy(texture, Some(char_rect), Some(*target)).unwrap();
    }
    target
}


#[derive(Debug)]
struct EditorBounds<'a> {
    lines_above_fold: usize,
    letter_height: usize,
    letter_width: usize,
    line_number_padding: usize,
    line_range: &'a Vec<(usize, usize)>
}


fn text_space_from_screen_space(EditorBounds {lines_above_fold, letter_height,line_number_padding, letter_width, line_range} : EditorBounds, mut x: i32, y: i32) -> Option<(usize, usize)> {
    let line_number : usize = (y / letter_height as i32 + lines_above_fold as i32).try_into().unwrap();
    if x < line_number_padding as i32 && x > line_number_padding as i32 - 20  {
        x = line_number_padding as i32;
    } 
    if x < line_number_padding as i32 {
        return None;
    }
    let mut column_number : usize = ((x - line_number_padding as i32) / letter_width as i32).try_into().unwrap();

    if let Some((line_start, line_end)) = line_range.get(line_number) {
        if column_number > line_end - line_start {
            column_number = line_range[line_number].1 - line_range[line_number].0;
        }
        return Some((line_number, column_number));
    }
    if line_number > line_range.len() {
        if let Some((start, end)) = line_range.last() {
           return Some((line_range.len() - 1, line_length((*start, *end))));
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


// It would be pretty cool to add a minimap
// Also cool to just add my own scrollbar.

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

    

    let mut window_width: i32 = 1200;
    let mut window_height: i32 = 800;

    let window = sdl_context
        .video()?
        .window("Example", window_width as u32, window_height as u32)
        .resizable()
        .build()
        .unwrap();

    let cursor = Cursor::from_system(SystemCursor::IBeam)
        .map_err(|err| format!("failed to load cursor: {}", err))?;
    cursor.set();


    // Let's create a Canvas which we will use to draw in our Window
    let mut canvas: Canvas<Window> = window
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
    // let mut text = fs::read_to_string("/Users/jimmyhmiller/Desktop/test/test2.txt").unwrap();
    let mut text = fs::read_to_string("/Users/jimmyhmiller/Documents/Code/jml/src/jml/core.clj").unwrap();
    // let mut text = "test\nthing\nstuff".to_string();
    println!("read file in {} ms", start_time.elapsed().as_millis());
    let mut chars = text.as_bytes().to_vec();

    let mut line_range = Vec::<(usize,usize)>::with_capacity(chars.len()/60);
    let mut line_start = 0;
    // This is slow. I don't actually have to do this for the whole file in one go.
    // I can do it for the first screen and then start doing more over time.
    // Need that concept in this app.
    // But while this is "slow", in release it is only about a second for a 1 GB file.
    let start_time = std::time::Instant::now();
    for (line_end, char) in chars.iter().enumerate() {
        if *char == b'\n'{
            line_range.push((line_start, line_end));
            line_start = line_end + 1;
        }
        if line_end == chars.len() - 1 {
            line_range.push((line_start, line_end + 1));
        }
    }
    
    // println!("{:?}", line_range);
    println!("parsed file in {} ms", start_time.elapsed().as_millis());

    println!("copied file");
    let mut offset_y = 0;
    let mut at_end = false;
    let scroll_speed : i32 = 5;
    let mut frame_counter = 0;
    let mut time_start = std::time::Instant::now();
    let mut fps = 0;
    let mut cursor: Option<(usize, usize)> = None;
    let mut mouse_down : Option<(i32, i32)> = None;
    let mut selection : Option<((i32, i32), (i32, i32))> = None;

    
    

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
        let viewing_window: usize = min((window_height / letter_height as i32).try_into().unwrap(), 1000);




        // Commands can stop this from being text input
        let mut is_text_input = false;
        for event in event_pump.poll_iter() {

            match event {
                Event::Quit { .. } => ::std::process::exit(0),

                Event::KeyDown { keycode, keymod, .. } => {
                    matches!(keycode, Some(_));
                    match (keycode.unwrap(), keymod) {
                        (Keycode::Up, _) => {
                            if let Some((cursor_line, cursor_column)) = cursor {
                                let new_line = cursor_line.saturating_sub(1);
                                cursor = Some((new_line, min(cursor_column, line_length(line_range[new_line]))));

                                // Need to deal with line_fraction
                                if cursor_line < lines_above_fold {
                                    offset_y -= letter_height as i32;
                                }
                            }
                        },
                        (Keycode::Down, _) => {

                            if let Some((cursor_line, cursor_column)) = cursor {
                                let new_line = cursor_line.saturating_add(1);
                                cursor = Some((new_line, min(cursor_column, line_length(line_range[new_line]))));

                                // // Need to deal with line_fraction
                                // if cursor_line > lines_above_fold {
                                //     offset_y += letter_height as i32;
                                // }
                            }
                        },
                        (Keycode::Right, _) => {
                            if let Some((cursor_line, cursor_column)) = cursor {
                                if let Some((line_start, line_end)) = line_range.get(cursor_line) {
                                    let length = line_length((*line_start, *line_end));
                                    if cursor_column >= length {
                                        if cursor_line + 1 < line_range.len() {
                                            cursor = Some((cursor_line + 1, 0));
                                        }
                                    } else {
                                        cursor = Some((cursor_line, cursor_column + 1));
                                    }
                                }
                            }
                        },
                        (Keycode::Left, _) => {
                            if let Some((cursor_line, cursor_column)) = cursor {
                                if cursor_column == 0 && cursor_line != 0 {
                                    let previous_line = line_range[cursor_line - 1];
                                    let length = line_length(previous_line);
                                    cursor = Some((cursor.unwrap().0.saturating_sub(1), length));
                                } else {
                                    cursor = Some((cursor_line, cursor_column.saturating_sub(1)));
                                }
                            }
                        },
                        (Keycode::Backspace, _) => {
                            // Need to deal with backspacing new line. Cursor is wrong.
                            if let Some((cursor_line, cursor_column)) = cursor {
                                if let Some((line_start, _line_end)) = line_range.get(cursor_line) {
                                    let char_pos = line_start + cursor_column - 1;
                                    let line_above_length = line_length(line_range[cursor_line - 1]);
                                    
                                    if chars.is_empty() || char_pos > chars.len() {
                                        continue;
                                    }
                                    chars.remove(char_pos);
                                    
                                    line_range[cursor_line].1 = line_range[cursor_line].1.saturating_sub(1);
                                    let mut line_erased = false;
                                    if cursor_column == 0 {
                                        line_range[cursor_line - 1] = (line_range[cursor_line - 1].0, line_range[cursor_line].1);
                                        line_range.remove(cursor_line);
                                        line_erased = true;
                                    }
                                    for mut line in line_range.iter_mut().skip(cursor_line + (if line_erased {0} else {1})) {
                                        line.0 = line.0.saturating_sub(1);
                                        line.1 = line.1.saturating_sub(1);
                                    }
        
                                    if cursor_column == 0 {
                                        cursor = Some((cursor_line.saturating_sub(1), line_above_length));
                                    } else {
                                        cursor = Some((cursor_line, cursor_column - 1));
                                    }
                                }
                            }
                        }
                        (Keycode::Return, _) => {
                            // Some weird things are happening here.
                            // Letters appear out of the void
                            if let Some((cursor_line, cursor_column)) = cursor {
                                let line_start = line_range[cursor_line].0;
                                let char_pos = line_start + cursor_column;

                                chars.splice(char_pos..char_pos, [b'\n']);
                                
                                let (start, end) = line_range[cursor_line];
                                if char_pos >= end && cursor_column != 0 {
                                    line_range.insert(cursor_line + 1, (char_pos, char_pos));
                                } else if cursor_column == 0 {
                                    line_range.splice(cursor_line..cursor_line+1, [(start,char_pos), (start+1, end+1)]);
                                } else {
                                    line_range.splice(cursor_line..cursor_line + 1, [(start, char_pos), (char_pos+1, end+1)]);
                                }
                                for mut line in line_range.iter_mut().skip(cursor_line + 2) {
                                    line.0 += 1;
                                    line.1 += 1;
                                }
                                cursor = Some((cursor_line+1, 0));
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
                            text = fs::read_to_string(path_str).unwrap();
                            println!("read file in {} ms", start_time.elapsed().as_millis());
                            chars = text.as_bytes().to_vec();
                        
                            line_range = Vec::<(usize,usize)>::with_capacity(chars.len()/60);
                            line_start = 0;
                            // This is slow. I don't actually have to do this for the whole file in one go.
                            // I can do it for the first screen and then start doing more over time.
                            // Need that concept in this app.
                            // But while this is "slow", in release it is only about a second for a 1 GB file.
                            let start_time = std::time::Instant::now();
                            for (line_end, char) in chars.iter().enumerate() {
                                if *char == b'\n'{
                                    line_range.push((line_start, line_end));
                                    line_start = line_end + 1;
                                }
                                if line_end == chars.len() - 1 {
                                    line_range.push((line_start, line_end + 1));
                                }
                            }
                            
                            offset_y = 0;
                            println!("parsed file in {} ms", start_time.elapsed().as_millis());
                        }

                        _ => is_text_input = true
                    }
                }
                Event::TextInput{text, ..} => {
                    if is_text_input {
                        if let Some((cursor_line, cursor_column)) = cursor {
                            let line_start = line_range[cursor_line].0;
                            let char_pos = line_start + cursor_column;
                            let char = text.into_bytes();
                            chars.splice(char_pos..char_pos, char);
                            line_range[cursor_line] = (line_start, line_range[cursor_line].1 + 1);
                            for mut line in line_range.iter_mut().skip(cursor_line + 1) {
                                line.0 += 1;
                                line.1 += 1;
                            }
                            cursor = Some((cursor_line, cursor_column + 1));
                        }
                    }
                }

                // Need to make selection work
                // Which probably means changing cursor representation
               
                Event::MouseButtonDown { x, y, .. } => {
                    cursor = text_space_from_screen_space(
                        EditorBounds {
                            lines_above_fold,
                            letter_height: letter_height_usize,
                            letter_width: letter_width_usize,
                            line_number_padding,
                            line_range: &line_range,
                        },
                        x,
                        y,
                    );
                    if let Some((x, y) ) = cursor {
                        mouse_down = Some((x.try_into().unwrap(), y.try_into().unwrap()));
                    }
                    
                    selection = None;
                }

                Event::MouseMotion{x, y, .. } => {
                    if let Some((start_line, mut start_column)) = mouse_down {
                        cursor = text_space_from_screen_space(
                            EditorBounds {
                                lines_above_fold,
                                letter_height: letter_height_usize,
                                letter_width: letter_width_usize,
                                line_number_padding,
                                line_range: &line_range,
                            },
                            x,
                            y,
                        );
                        // TODO: Get my int types correct!
                        if let Some((line, mut column)) = cursor {
                            let new_start_line = start_line.min(line.try_into().unwrap());
                            let line = line.max(start_line.try_into().unwrap());
                            if new_start_line != start_line || start_line == line as i32 && start_column > column as i32 {
                                let temp = start_column;
                                start_column = column.try_into().unwrap();
                                column = temp as usize;
                            }

                            selection = Some(((new_start_line, start_column), (line.try_into().unwrap(), column.try_into().unwrap())));
                            
                        }
                    }
                }

                Event::MouseButtonUp{x, y, ..} => {
                    if let Some((start_line, mut start_column)) = mouse_down {
                        cursor = text_space_from_screen_space(
                            EditorBounds {
                                lines_above_fold,
                                letter_height: letter_height_usize,
                                letter_width: letter_width_usize,
                                line_number_padding,
                                line_range: &line_range,
                            },
                            x,
                            y,
                        );
                        if selection.is_some() {
                            if let Some((line, mut column)) = cursor {
                                let new_start_line = start_line.min(line.try_into().unwrap());
                                let line = line.max(start_line.try_into().unwrap());
                                if new_start_line != start_line || start_line == line as i32 && start_column > column as i32{
                                    let temp = start_column;
                                    start_column = column.try_into().unwrap();
                                    column = temp as usize;
                                }
    
                                selection = Some(((new_start_line, start_column), (line.try_into().unwrap(), column.try_into().unwrap())));
                            }
                        }
                            
                    }
                    
                    mouse_down = None;
                }
                // Continuous resize in sdl2 is a bit weird
                // Would need to watch events or something
                Event::Window {win_event: WindowEvent::Resized(w, h), ..} => {
                    window_width = w;
                    window_height = h;
                }

                Event::MouseWheel {x: _, y, direction , timestamp: _, .. } => {
                    let direction_multiplier = match direction {
                        sdl2::mouse::MouseWheelDirection::Normal => 1,
                        sdl2::mouse::MouseWheelDirection::Flipped => -1,
                        sdl2::mouse::MouseWheelDirection::Unknown(x) => x as i32
                    };
                    scroll_y = y * direction_multiplier * scroll_speed;
                }
                _ => {}
            }
        }



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
        
            if let Some(cursor) = cursor {
                if cursor.0 == line {
                    let cursor_x = cursor.1 as i32  * letter_width as i32 + line_number_padding as i32;
                    let cursor_y = target.y();
                    canvas.set_draw_color(Color::RGBA(255, 204, 0, 255));
                    canvas.fill_rect(Rect::new(cursor_x as i32, cursor_y as i32, 2, letter_height))?;
                }
            }


            if let Some(((start_line, start_column), (end_line, end_column))) = selection {
                if line >= start_line.try_into().unwrap() && line <= end_line.try_into().unwrap() {
                    let start_x = if line as i32 == start_line {
                        start_column as i32 * letter_width as i32 + line_number_padding as i32
                    } else {
                        line_number_padding as i32
                    };
                    let width = if start_line == end_line {
                        ((end_column - start_column) * letter_width as i32).try_into().unwrap()
                    } else if line as i32 == end_line {
                        (end_column * letter_width as i32).try_into().unwrap()
                    } else if line as i32 == start_line {
                        ((line_length(line_range[line]) as i32 - start_column) * letter_width as i32).try_into().unwrap()
                    } else {
                        (line_length(line_range[line]) * letter_width_usize).try_into().unwrap()
                    };

                    let start_y = target.y();
                    canvas.set_draw_color(Color::RGBA(65, 70, 99, 255));
                    // Need to deal with last line.
                    canvas.fill_rect(Rect::new(start_x, start_y, width, letter_height))?;
                }

            };

            draw_string(&mut canvas, target, &texture, std::str::from_utf8(chars[start..end].as_ref()).unwrap());


            move_down(target, letter_height as i32);
        }

        let mut target = Rect::new(window_width - (letter_width * 10) as i32, 0, letter_width, letter_height);
        draw_string(&mut canvas, &mut target, &texture, &format!("fps: {}", fps));
        frame_counter += 1;
        if time_start.elapsed().as_secs() >= 1 {
            fps = frame_counter;
            frame_counter = 0;
            time_start = std::time::Instant::now();
        }

        // Need to calculate this based on length
        let mut target = Rect::new(window_width - (letter_width * 22) as i32 as i32, window_height-letter_height as i32, letter_width, letter_height);
        if let Some((cursor_line, cursor_column)) = cursor {
            draw_string(&mut canvas, &mut target, &texture, &format!("Line {}, Column {}", cursor_line, cursor_column));
        }



        canvas.present();
    }
}
