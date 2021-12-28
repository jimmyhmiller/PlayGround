use std::convert::TryInto;

use sdl2::{pixels::Color, render::*, video::{self, WindowContext}, VideoSubsystem};

pub struct SdlContext {
    pub ttf_context: sdl2::ttf::Sdl2TtfContext, 
    pub canvas: Canvas<video::Window>, 
    pub event_pump: sdl2::EventPump,
    pub texture_creator: TextureCreator<WindowContext>,
    pub video: VideoSubsystem,
}

pub fn setup_sdl(width: usize, height: usize) -> Result<SdlContext, String> {
    let sdl_context = sdl2::init()?;
    let ttf_context = sdl2::ttf::init().map_err(|e| e.to_string())?;
    let video = sdl_context
        .video()?;
      
    let sdl_window =   
        video
        .window("Lith", width as u32, height as u32)
        // .opengl()
        .resizable()
        .build()
        .unwrap();

    
    let canvas: Canvas<video::Window> = sdl_window
        .into_canvas()
        .present_vsync()
        .build()
        .unwrap();


    let event_pump = sdl_context.event_pump()?;

    let texture_creator = canvas.texture_creator();


   Ok(SdlContext {
        ttf_context,
        canvas,
        event_pump,
        texture_creator,
        video,
    })
}

pub fn draw_font_texture(texture_creator: &TextureCreator<WindowContext>, ttf_context: sdl2::ttf::Sdl2TtfContext) -> Result<(Texture<>, usize, usize), String> {
    let font_path = "/Users/jimmyhmiller/Library/Fonts/UbuntuMono-Regular.ttf";
    let font = ttf_context.load_font(font_path, 16)?;
    let mut text = String::new();
    for i  in 33..127 {
        text.push(i as u8 as char);
    }
    let surface = font
        .render(text.as_str())
        // This needs to be 255 if I want to change colors
        .blended(Color::RGBA(255, 255, 255, 255))
        .map_err(|e| e.to_string())?;
    let texture = texture_creator
        .create_texture_from_surface(&surface)
        .map_err(|e| e.to_string())?;
    let TextureQuery { width, height, .. } = texture.query();
    let width = (width / text.len() as u32).try_into().unwrap();
    Ok((texture, width, height.try_into().unwrap()))
}
