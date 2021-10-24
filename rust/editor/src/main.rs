use sdl2::{*, event::*, keyboard::*, pixels::Color, rect::Rect, render::*, video::*};


fn render_char(width: i32, height: i32, c: char) -> Rect {
    let width = width / 94;
    let rect = Rect::new(width * (c as i32 - 33), 0, width as u32, height as u32);
    // Interesting code from copilot, probably useful later
    // match c {
    //     ' ' => rect.set_x(width),
    //     '\n' => rect.set_y(height),
    //     '\t' => rect.set_x(width * 4),
    //     '\r' => rect.set_x(0),
    //     '\x08' => rect.set_x(width * -1),
    //     '\x0C' => rect.set_x(width * -2),
    //     _ => {}
    // }
    rect
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

    

    let mut window_width: i32 = 500;
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
    let mut font = ttf_context.load_font(font_path, 16)?;


    let mut text = String::new();
    for i  in 33..127 {
        text.push(i as u8 as char);
    }
    println!("{}", text);
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

    let mut offsetY = 0;
    loop {
        let mut scrollY = 0;
        match event_pump.poll_event() {
            Some(Event::Quit { .. }) => ::std::process::exit(0),
            // Continuous resize in sdl2 is a bit weird
            // Would need to watch events or something
            Some(Event::Window {win_event: WindowEvent::Resized(w, h), ..}) => {
                window_width = w;
                window_height = h;
            }
            Some(Event::MouseWheel {x, y, direction , timestamp: _, .. }) => {
                let direction_multiplier = match direction {
                    sdl2::mouse::MouseWheelDirection::Normal => -1,
                    sdl2::mouse::MouseWheelDirection::Flipped => 1,
                    sdl2::mouse::MouseWheelDirection::Unknown(x) => x as i32
                };
                scrollY = y  * direction_multiplier * 10;
            }
            _ => {}
        }

        
        offsetY += scrollY;

        let mut target = Rect::new(10 as i32, offsetY as i32, letter_width, height);

        canvas.set_draw_color(Color::RGBA(255, 255, 255, 255));
        canvas.clear();

        // I should be smartest than this. I can know what I need to render
        // based on offset and window size
        let mut lines = 0;
        for _ in 0..1000 {
            for (i, c) in text.chars().enumerate() {
                let rect = render_char(width as i32, height as i32, c);
                
                if target.x + letter_width as i32 > window_width as i32 {
                    target.set_x(10);
                    target.set_y(target.y + height as i32);
                    lines += 1;
                }
                canvas.copy(&texture, Some(rect), Some(target))?;
                target.offset(letter_width as i32, 0);
            }
        }

        // canvas.copy(&texture, Some(render_char(width as i32, height as i32, 'a')), Some(target))?;
        canvas.present();

        scrollY = 0;

        // handle_key_presses(&event_pump, &mut player, &world_map);
    }
}
