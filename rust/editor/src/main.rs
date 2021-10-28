use std::{cmp::{max, min}, convert::TryInto, fs, ops::Neg};

use sdl2::{*, event::*, keyboard::*, pixels::Color, rect::Rect, render::*, video::*};


fn render_char(width: i32, height: i32, c: char) -> Rect {
    let width = width / 94;
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

    

    let window_width: i32 = 500;
    let mut window_height: i32 = 500;

    let window = sdl_context
        .video()?
        .window("Example", window_width as u32, window_height as u32)
        .resizable()
        .build()
        .unwrap();

    // Let's create a Canvas which we will use to draw in our Window
    let mut canvas: Canvas<Window> = window.into_canvas().present_vsync().build().unwrap();
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
        .blended(Color::RGBA(0, 0, 0, 255))
        .map_err(|e| e.to_string())?;
    let texture = texture_creator
        .create_texture_from_surface(&surface)
        .map_err(|e| e.to_string())?;

    let TextureQuery { width, height, .. } = texture.query();
    let letter_width = width / text.len() as u32;
    let letter_height = height;

    let start_time = std::time::Instant::now();
    let text = fs::read_to_string("/Users/jimmyhmiller/Desktop/test/test3.txt").unwrap();
    println!("read file in {} ms", start_time.elapsed().as_millis());
    let chars = text.as_bytes();

    let mut line_range = Vec::<(usize,usize)>::with_capacity(chars.len()/60);
    let mut line_start = 0;
    let mut line_end = 0;
    // This is slow. I don't actually have to do this for the whole file in one go.
    // I can do it for the first screen and then start doing more over time.
    // Need that concept in this app.
    // But while this is "slow", in release it is only about a second for a 1 GB file.
    let start_time = std::time::Instant::now();
    for (line_end, char) in chars.into_iter().enumerate() {
        if *char == 10 {
            line_range.push((line_start, line_end - 1));
            line_start = line_end + 1;
        }
    }
    println!("parsed file in {} ms", start_time.elapsed().as_millis());

    println!("copied file");
    let mut offset_y = 0;
    let mut at_end = false;
    loop {
        let mut scroll_y = 0;
        match event_pump.poll_event() {
            Some(Event::Quit { .. }) => ::std::process::exit(0),
            // Continuous resize in sdl2 is a bit weird
            // Would need to watch events or something
            Some(Event::Window {win_event: WindowEvent::Resized(_w, h), ..}) => {
                // window_width = w;
                window_height = h;
            }
            Some(Event::MouseWheel {x: _, y, direction , timestamp: _, .. }) => {
                let direction_multiplier = match direction {
                    sdl2::mouse::MouseWheelDirection::Normal => 1,
                    sdl2::mouse::MouseWheelDirection::Flipped => -1,
                    sdl2::mouse::MouseWheelDirection::Unknown(x) => x as i32
                };
                scroll_y = y  * direction_multiplier * 5;
            }
            _ => {}
        }

        if !at_end || scroll_y < 0 {
            offset_y += scroll_y;
        }


        canvas.set_draw_color(Color::RGBA(255, 255, 255, 255));
        canvas.clear();

        // I should be smartest than this. I can know what I need to render
        // based on offset and window size
        // let mut lines = 0;

        offset_y = max(0, offset_y);


        let lines_above_fold : usize = offset_y as usize / letter_height as usize;
        let viewing_window: usize = min((window_height / letter_height as i32).try_into().unwrap(), 1000);
        
        let line_fraction = offset_y as usize % letter_height as usize;

        let mut target = Rect::new(10, (line_fraction as i32).neg(), letter_width, letter_height);
        
        if lines_above_fold + viewing_window >= line_range.len() + 3 {
            at_end = true;
        } else {
            at_end = false;
        }

        // I got rid of line wrap in this refactor. Probably should add that back in.
        for i in lines_above_fold as usize..min(lines_above_fold + viewing_window, line_range.len()) {
            let line_start = line_range[i as usize].0;
            let line_end = line_range[i as usize].1;
            let mut line_offset = 10;
            // I want to pad this so that the offset by the line number never changes.
            // Really I should draw a line or something to make it look nicer.
            line_offset += ((digit_count(line_range.len()) - digit_count(i + 1)) * 10) as i32;
            let line_number = (i + 1).to_string();
            for char in line_number.chars() {
                let char_rect = render_char(width as i32, height as i32, char as char);
                target.set_x(line_offset);
                canvas.copy(&texture, Some(char_rect), Some(target)).unwrap();
                line_offset += letter_width as i32;
            }

            line_offset += 20;
            for j in line_start..line_end+1 {
                let char = chars[j];
                let rect = render_char(width as i32, height as i32, char as char);
                target.set_x(line_offset as i32);
                canvas.copy(&texture, rect, target).unwrap();
                line_offset += letter_width as i32;
            }
            target.set_y(target.y + height as i32);
        }

        canvas.present();

        // handle_key_presses(&event_pump, &mut player, &world_map);
    }
}
