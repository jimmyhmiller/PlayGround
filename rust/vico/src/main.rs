use std::fs;

use sdl2::{event::Event, rect::Rect, pixels::Color, render::Canvas, video::{self}};


mod sdl;

#[derive(Debug, Clone, Copy)]
struct Window {
    width: i32,
    height: i32,
}

fn draw_word(canvas: &mut Canvas<video::Window>, word: &str) {
    let fonts = word.lines()
        .map(|line| line.trim()).map(|line| line.chars()
        .map(|c| c == '1').collect::<Vec<bool>>());

    for (j, line) in fonts.enumerate() {
        for (i, pixel) in line.iter().enumerate() {
            if *pixel {
                canvas.set_draw_color(Color::RGBA(0, 0, 0, 255));
            } else {
                canvas.set_draw_color(Color::RGBA(255, 255, 255, 255));
            }
            canvas.fill_rect(Rect::new(i as i32 * 1 + 100, j as i32 * 1 + 100, 1, 1)).unwrap();
        }
    }

}


fn handle_events(event_pump: &mut sdl2::EventPump){
    for event in event_pump.poll_iter() {
        // println!("frame: {}, event {:?}", frame_counter, event);
        match event {
            Event::Quit { .. } => ::std::process::exit(0),
            _ => {}
        }
    }
}


fn main() -> Result<(), String> {
    let window = Window {
        width: 1200,
        height: 800,
    };
    

    let palette = vec![
        Color::RGB(0x00, 0x00, 0x00),
        Color::RGB(0x1D, 0x2B, 0x53),
        Color::RGB(0x7E, 0x25, 0x53),
        Color::RGB(0x00, 0x87, 0x51),
        Color::RGB(0xAB, 0x52, 0x36),
        Color::RGB(0x5F, 0x57, 0x4F),
        Color::RGB(0xC2, 0xC3, 0xC7),
        Color::RGB(0xFF, 0xF1, 0xE8),
        Color::RGB(0xFF, 0x00, 0x4D),
        Color::RGB(0xFF, 0xA3, 0x00),
        Color::RGB(0xFF, 0xEC, 0x27),
        Color::RGB(0x00, 0xE4, 0x36),
        Color::RGB(0x29, 0xAD, 0xFF),
        Color::RGB(0x83, 0x76, 0x9C),
        Color::RGB(0xFF, 0x77, 0xA8),
        Color::RGB(0xFF, 0xCC, 0xAA),
    ];

    let sdl::SdlContext {
        mut event_pump,
        mut canvas,
        texture_creator: _,
        ttf_context: _,
        video: _,
    } = sdl::setup_sdl(window.width as usize, window.height as usize)?;

    loop {
        canvas.set_draw_color(Color::BLACK);
        canvas.clear();
        canvas.set_draw_color(Color::WHITE);

        for (i, color) in palette.iter().enumerate() {
            let offset = 2;
            let size = 40;
            let x = i as i32 / size;
            let y = i as i32 % size;
            canvas.set_draw_color(Color::WHITE);
            canvas.fill_rect(Rect::new(x * 10, y * size + offset, size as u32, size as u32))?;
            canvas.set_draw_color(*color);
            canvas.fill_rect(Rect::new(x * 10 + 1, y * size + 1 + offset, size as u32 - 2, size as u32 - 2))?;
        }
        canvas.set_draw_color(Color::WHITE);
        canvas.fill_rect(Rect::new(window.width - 140, 20, 128, 128)).unwrap();
        draw_word(&mut canvas, fs::read_to_string("font.txt").unwrap().as_str());
        canvas.present();
        handle_events(&mut event_pump);
    }


}
