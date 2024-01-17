
use std::any::Any;

use metal::MetalLayerRef;
use raw_window_handle::HasWindowHandle;
use skia_safe::{gpu, scalar, ColorType};
use winit::{platform::macos::WindowBuilderExtMacOS, event_loop::EventLoopProxy};
use cocoa::{appkit::{NSView, NSViewLayerContentsPlacement}, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use foreign_types_shared::{ForeignType, ForeignTypeRef};
use metal::{Device, MTLPixelFormat, MetalLayer};
use objc::{rc::autoreleasepool, runtime::YES, msg_send, sel, sel_impl};
use skia_safe::gpu::{mtl, BackendRenderTarget, DirectContext, SurfaceOrigin};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub trait App {
    fn on_window_create(&mut self, event_loop_proxy: EventLoopProxy<()>, size: Size);
    // TODO: Consider user event
    fn add_event(&mut self, event: &Event<()>) -> bool;
    fn exit(&mut self);
    fn draw(&mut self, canvas: &skia_safe::Canvas);
    fn end_frame(&mut self);
    fn tick(&mut self);
    fn should_redraw(&mut self) -> bool;
    fn cursor_icon(&mut self) -> winit::window::CursorIcon;
    fn set_window_size(&mut self, size: Size);

    fn create_window(&mut self, title: &str, width: i32, height: i32) {

        let mut size = LogicalSize::new(width, height);
    
        let events_loop = EventLoop::new().unwrap();
    
        let event_loop_proxy = events_loop.create_proxy();
    
        let window = WindowBuilder::new()
            .with_inner_size(size)
            .with_title(title.to_string())
            .with_titlebar_transparent(true)
            .with_fullsize_content_view(true)
            .with_transparent(false)
            .build(&events_loop)
            .unwrap();
    
        self.on_window_create(
            event_loop_proxy,
            Size {
                width: size.width as f64,
                height: size.height as f64,
            },
        );
    
        let device = Device::system_default().expect("no device found");
    
        let metal_layer = {
            let draw_size = window.inner_size();
            let layer: MetalLayer = MetalLayer::new();
            layer.set_device(&device);
            unsafe {
                let mask = 1 << 4 | 1 << 1;
                let _ : () = msg_send![layer, setAutoresizingMask: mask];
                let _: () = msg_send![layer, setNeedsDisplayOnBoundsChange: YES];
            }
            layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            layer.set_presents_with_transaction(false);
            layer.set_opaque(true);
            layer.set_display_sync_enabled(true);
            layer.set_contents_scale(window.scale_factor() as f64);
            // change some stuff to make resizing nice
            // https://thume.ca/2019/06/19/glitchless-metal-window-resizing/
    
            let raw = window.window_handle().unwrap().as_raw();
            unsafe {
                match raw {
                    raw_window_handle::RawWindowHandle::AppKit(handle) => {
                        let view = handle.ns_view.as_ptr() as cocoa_id;
                        view.setWantsLayer(YES);
                        view.setLayer(
                            layer.as_ref() as *const MetalLayerRef as *mut objc::runtime::Object
                        );
                        let _: () = msg_send![view, setLayerContentsRedrawPolicy: 1];
                        view.setLayerContentsPlacement(NSViewLayerContentsPlacement::NSViewLayerContentsPlacementTopLeft);
                    }
                    _ => todo!(),
                }
            }
    
            layer.set_drawable_size(CGSize::new(draw_size.width as f64, draw_size.height as f64));
            layer
        };
    
        let command_queue = device.new_command_queue();
    
        let backend = unsafe {
            mtl::BackendContext::new(
                device.as_ptr() as mtl::Handle,
                command_queue.as_ptr() as mtl::Handle,
                std::ptr::null(),
            )
        };
    
        let mut context = DirectContext::new_metal(&backend, None).unwrap();
        let mut event_added = false;
    
        events_loop
            .run(move |event, event_window| {
                autoreleasepool(|| {
                    // TODO! NEEd to deal with control_flow
                    event_window.set_control_flow(ControlFlow::Wait);
                    let was_added = self.add_event(&event);
                    if was_added {
                        event_added = true;
                    }
                    match event {
                        Event::WindowEvent { event, .. } => match event {
                            WindowEvent::CloseRequested => {
                                self.exit();
                                event_window.exit();
                            }
                            WindowEvent::Resized(current_size) => {
                                metal_layer.set_drawable_size(CGSize::new(
                                    current_size.width as f64,
                                    current_size.height as f64,
                                ));
    
                                size.width = current_size.width as i32;
                                size.height = current_size.height as i32;
                                self.set_window_size(Size {
                                    width: size.width as f64,
                                    height: size.height as f64,
                                });
                                window.request_redraw();
                            }
                            WindowEvent::RedrawRequested => {
                                if let Some(drawable) = metal_layer.next_drawable() {
                                    let drawable_size = {
                                        let size = metal_layer.drawable_size();
                                        skia_safe::Size::new(size.width as scalar, size.height as scalar)
                                    };
    
                                    let mut surface = unsafe {
                                        let texture_info = mtl::TextureInfo::new(
                                            drawable.texture().as_ptr() as mtl::Handle,
                                        );
    
                                        let backend_render_target = BackendRenderTarget::new_metal(
                                            (drawable_size.width as i32, drawable_size.height as i32),
                                            &texture_info,
                                        );
    
                                        gpu::surfaces::wrap_backend_render_target(
                                            &mut context,
                                            &backend_render_target,
                                            SurfaceOrigin::TopLeft,
                                            ColorType::BGRA8888,
                                            None,
                                            None,
                                        )
                                        .unwrap()
                                    };
    
                                    self.draw(surface.canvas());
    
                                    context.flush_and_submit();
                                    drop(surface);
    
                                    let command_buffer = command_queue.new_command_buffer();
                                    command_buffer.present_drawable(drawable);
                                    command_buffer.commit();
                                }
                            }
                            _ => (),
                        },
                        Event::AboutToWait => {
                            // TODO: I would need to signal if there is any waiting
                            // work left to do from our wasm modules.
                            // If there is no work left, we don't need to do anything
                            // Well, as long as we define this properly. A module could
                            // need a tick, because they are totally driven by the editor
                            self.end_frame();
    
                            // This needs to happen unconditonally, because
                            // these are external things that can create events
                            self.tick();
    
    
                            if !self.should_redraw() {
                                return;
                            }
    
                            if self.should_redraw() {
                                window.set_cursor_icon(self.cursor_icon());
                                window.request_redraw();
                            }
                        }
    
                        _ => {}
                    }
                });
            })
            .unwrap();
    }
}


pub struct Size {
    pub width: f64,
    pub height: f64,
}



