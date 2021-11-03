use std::{cmp::{max, min}, convert::TryInto, fs, ops::Neg};

use native_dialog::FileDialog;
use sdl2::{*, event::*, keyboard::*, libc::fpos_t, pixels::Color, rect::Rect, render::*, video::*};


fn render_char(width: i32, height: i32, c: char) -> Rect {
    let rect = Rect::new(width * (c as i32 - 33), 0, width as u32, height as u32);
    rect
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
    return target;
}



fn move_right(target: &mut Rect, padding: i32) -> &mut Rect {
    target.set_x(target.x() + padding);
    return target;
}


fn move_down(target: &mut Rect, padding: i32) -> &mut Rect {
    target.set_y(target.y() + padding);
    return target;
}

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

    let start_time = std::time::Instant::now();
    let mut text = fs::read_to_string("/Users/jimmyhmiller/Desktop/test/test3.txt").unwrap();
    println!("read file in {} ms", start_time.elapsed().as_millis());
    let mut chars = text.as_bytes();

    let mut line_range = Vec::<(usize,usize)>::with_capacity(chars.len()/60);
    let mut line_start = 0;
    // This is slow. I don't actually have to do this for the whole file in one go.
    // I can do it for the first screen and then start doing more over time.
    // Need that concept in this app.
    // But while this is "slow", in release it is only about a second for a 1 GB file.
    let start_time = std::time::Instant::now();
    for (line_end, char) in chars.into_iter().enumerate() {
        if *char == '\n' as u8 {
            line_range.push((line_start, line_end - 1));
            line_start = line_end + 1;
        }
    }
    println!("parsed file in {} ms", start_time.elapsed().as_millis());

    println!("copied file");
    let mut offset_y = 0;
    let mut at_end = false;
    let mut scroll_speed : i32 = 5;
    let mut frame_counter = 0;
    let mut time_start = std::time::Instant::now();
    let mut fps = 0;
    let mut cursor: Option<(usize, usize)> = None;
    

    texture.set_color_mod(167, 174, 210);
    loop {




        canvas.set_draw_color(Color::RGBA(42, 45, 62, 255));
        canvas.clear();
    
        let mut scroll_y : i32 = 0;

        let editor_left_margin = 10;
        let line_number_digits = digit_count(line_range.len());
        let line_number_gutter_width = 20;
        // final letter width is because we write our string, we are in that letters position, then move more.
        let line_number_padding = line_number_digits * letter_width as usize + line_number_gutter_width + editor_left_margin + letter_width as usize;

        let lines_above_fold : usize = offset_y as usize / letter_height as usize;
        // Fix this to be less hacky.
        let viewing_window: usize = min((window_height / letter_height as i32).try_into().unwrap(), 1000);

        
        match event_pump.poll_event() {
            Some(Event::Quit { .. }) => ::std::process::exit(0),
            Some(Event::KeyDown { keycode: Some(Keycode::Escape), .. }) => ::std::process::exit(0),
            // Play with scroll speed
            Some(Event::KeyDown { keycode: Some(Keycode::Up), .. }) => {
                if cursor.is_some() {
                    cursor = Some((cursor.unwrap().0 - 1, cursor.unwrap().1));
                    // Need to actually deal with line fractions here.
                    if cursor.unwrap().0 + 1 <= lines_above_fold {
                        offset_y -= letter_height as i32;;
                    }
                }
                // Need something to deal with stuff not existing
            }
            Some(Event::KeyDown { keycode: Some(Keycode::Down), .. }) => {
                if cursor.is_some() {
                    cursor = Some((cursor.unwrap().0 + 1, cursor.unwrap().1));
                    // Need to actually deal with line fractions here.
                    if cursor.unwrap().0 + 1 > lines_above_fold + viewing_window {
                        offset_y += letter_height as i32;
                    }
                }
                
                // Need something to deal with stuff not existing
            }
            Some(Event::KeyDown { keycode: Some(Keycode::Right), .. }) => {
                if cursor.is_some() {
                    cursor = Some((cursor.unwrap().0, cursor.unwrap().1 + 1));
                }
                // Need something to deal with stuff not existing
            }
            Some(Event::KeyDown { keycode: Some(Keycode::Left), .. }) => {
                if cursor.is_some() {
                    cursor = Some((cursor.unwrap().0, cursor.unwrap().1 - 1));
                }
                // Need something to deal with stuff not existing
            }
            Some(Event::MouseButtonUp { x, y, .. }) => {
                // Need to make sure I round up here so I can get the right line.
                let line_number : usize = (y / letter_height as i32 + lines_above_fold as i32).try_into().unwrap();
                if x < line_number_padding as i32 {
                    continue;
                }
                let mut column_number : usize = ((x - line_number_padding as i32) / letter_width as i32).try_into().unwrap();
                if column_number > line_range[line_number].1 - line_range[line_number].0 {
                    column_number = line_range[line_number].1 - line_range[line_number].0;
                }
                cursor = Some((line_number, column_number));
                println!("Column {}", column_number);
            }
            Some(Event::KeyDown { keycode: Some(Keycode::O), keymod: Mod::LGUIMOD | Mod:: RGUIMOD, .. }) => {  
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
                chars = text.as_bytes();
            
                line_range = Vec::<(usize,usize)>::with_capacity(chars.len()/60);
                line_start = 0;
                // This is slow. I don't actually have to do this for the whole file in one go.
                // I can do it for the first screen and then start doing more over time.
                // Need that concept in this app.
                // But while this is "slow", in release it is only about a second for a 1 GB file.
                let start_time = std::time::Instant::now();
                for (line_end, char) in chars.into_iter().enumerate() {
                    if *char == '\n' as u8 {
                        line_range.push((line_start, line_end - 1));
                        line_start = line_end + 1;
                    }
                }
                offset_y = 0;
                println!("parsed file in {} ms", start_time.elapsed().as_millis());
            }
            // Continuous resize in sdl2 is a bit weird
            // Would need to watch events or something
            Some(Event::Window {win_event: WindowEvent::Resized(w, h), ..}) => {
                window_width = w;
                window_height = h;
            }

            Some(Event::MouseWheel {x: _, y, direction , timestamp: _, .. }) => {
                let direction_multiplier = match direction {
                    sdl2::mouse::MouseWheelDirection::Normal => 1,
                    sdl2::mouse::MouseWheelDirection::Flipped => -1,
                    sdl2::mouse::MouseWheelDirection::Unknown(x) => x as i32
                };
                scroll_y = y * direction_multiplier * scroll_speed;
            }

            _ => {}
        }


        if !at_end || scroll_y < 0 {
            offset_y += scroll_y;
        }
        offset_y = max(0, offset_y);




        // I should be smartest than this. I can know what I need to render
        // based on offset and window size
        // let mut lines = 0;





        let line_fraction = offset_y as usize % letter_height as usize;

        let mut target = Rect::new(editor_left_margin as i32, (line_fraction as i32).neg(), letter_width, letter_height);
        
        if lines_above_fold + viewing_window >= line_range.len() + 3 {
            at_end = true;
        } else {
            at_end = false;
        }


        // TODO: Add some spacing between letters!

        // Need to add a real parser or I can try messing with tree sitter.
        // But maybe I need to make text editable first?

        // I got rid of line wrap in this refactor. Probably should add that back in.
        for i in lines_above_fold as usize..min(lines_above_fold + viewing_window, line_range.len()) {
            texture.set_color_mod(167, 174, 210);
            let (start, end) = line_range[i];
            target.set_x(editor_left_margin as i32);

            // I want to pad this so that the offset by the line number never changes.
            // Really I should draw a line or something to make it look nicer.
            let left_padding_count = line_number_digits - digit_count(i + 1);
            let padding = left_padding_count * letter_width as usize;
            move_right(&mut target, padding as i32);

            let line_number = (i + 1).to_string();

            let target = draw_string(&mut canvas, &mut target, &texture, &line_number);
            move_right(target, line_number_gutter_width as i32);
        
            if cursor.is_some() && cursor.unwrap().0 == i  {
                let cursor_x = cursor.unwrap().1 as i32  * letter_width as i32 + line_number_padding as i32;
                let cursor_y = target.y();
                canvas.set_draw_color(Color::RGBA(82, 135, 249, 255));
                canvas.fill_rect(Rect::new(cursor_x as i32, cursor_y as i32, 2, letter_height))?;
            }

            draw_string(&mut canvas, target, &texture, std::str::from_utf8(chars[start..=end].as_ref()).unwrap());
            move_down(target, letter_height as i32);
        }

        let mut target = Rect::new(window_width - (letter_width * 8) as i32, 0, letter_width, letter_height);
        draw_string(&mut canvas, &mut target, &texture, &format!("fps: {}", fps));
        frame_counter += 1;
        if time_start.elapsed().as_secs() == 1 {
            fps = frame_counter;
            frame_counter = 0;
            time_start = std::time::Instant::now();
        }

        canvas.present();
    }
}
