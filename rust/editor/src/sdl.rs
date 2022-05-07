use std::ptr;
use std::{convert::TryInto};
use foreign_types::ForeignType;
use foreign_types::ForeignTypeRef;
// use sdl2_sys::*;
use metal::Device;
use metal::MetalLayer;
use metal::{MetalDrawableRef};

use sdl2::{pixels::Color, render::*, video::{self, WindowContext, Window}, clipboard::ClipboardUtil, mouse::SystemCursor};
use skia_safe::gpu::mtl;
use skia_safe::{gpu::{BackendRenderTarget, SurfaceOrigin, DirectContext, mtl::TextureInfo}, ColorType, Surface, ColorSpace};

pub struct SdlContext {
    pub ttf_context: sdl2::ttf::Sdl2TtfContext, 
    pub canvas: Canvas<video::Window>, 
    pub event_pump: sdl2::EventPump,
    pub texture_creator: TextureCreator<WindowContext>,
    pub clipboard: ClipboardUtil,
    pub system_cursor: sdl2::mouse::Cursor,
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

    let clipboard = video.clipboard();

    let system_cursor = sdl2::mouse::Cursor::from_system(SystemCursor::IBeam).unwrap();
    system_cursor.set();

    Ok(SdlContext {
        ttf_context,
        canvas,
        event_pump,
        texture_creator,
        clipboard,
        system_cursor,
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


// fn create_surface(
//     window_context: &Window,
//     fb_info: &FramebufferInfo,
//     gr_context: &mut skia_safe::gpu::DirectContext,
// ) -> skia_safe::Surface {

//     // let pixel_format = window_context.window_pixel_format();
//     let size = window_context.size();
//     let backend_render_target = BackendRenderTarget::new_gl(
//         (
//             size.0.try_into().unwrap(),
//             size.1.try_into().unwrap(),
//         ),
//         None,
//         8,
//         *fb_info,
//     );
//     Surface::from_backend_render_target(
//         gr_context,
//         &backend_render_target,
//         SurfaceOrigin::BottomLeft,
//         ColorType::RGBA8888,
//         None,
//         None,
//     )
//     .unwrap()
// }



pub struct SkiaContext<'a> {
    pub surface: skia_safe::Surface,
    pub drawable: &'a MetalDrawableRef,
    pub command_queue: metal::CommandQueue,
}

pub fn setup_skia<'a>(device: &'a Device, layer: &'a MetalLayer, canvas: &'a Canvas<Window>) -> SkiaContext<'a> {

	layer.set_device(&device);
	// layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
	// layer.set_presents_with_transaction(false);a
	// layer.display_sync_enabled();

	// layer.set_framebuffer_only(false);
	layer.set_opaque(true);

	// layer.set_maximum_drawable_count(3);


	// let draw_size = window.inner_size();
	// layer.set_drawable_size(CGSize::new(draw_size.width as f64, draw_size.height as f64));

	let queue = device.new_command_queue();

    let backend = unsafe {
        mtl::BackendContext::new(
            device.as_ptr() as mtl::Handle,
            queue.as_ptr() as mtl::Handle,
            ptr::null(),
        )
    };

    let drawable = layer.next_drawable().unwrap();

	let mut ctx = DirectContext::new_metal(&backend, None).expect("Unable to create direct context");
	
    let t_info = unsafe { TextureInfo::new(drawable.texture().as_ptr() as *const _) };
    let target = BackendRenderTarget::new_metal((canvas.window().size().0 as i32, canvas.window().size().1 as i32), 4, &t_info);

    let surface = Surface::from_backend_render_target(
        &mut ctx,
        &target,
        SurfaceOrigin::BottomLeft,
        ColorType::BGRA8888,
            ColorSpace::new_srgb(),
            None,
    );

    SkiaContext {
        surface: surface.unwrap(),
        drawable,
        command_queue: queue,
    }
   
}

