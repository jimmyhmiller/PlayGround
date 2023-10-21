// use cacao::{webview::{WebView, WebViewConfig, WebViewDelegate}, layer::Layer, layout::LayoutAnchorX, view::View};

use metal_rs::MetalLayerRef;
use skia_safe::{scalar, ColorType, Size, Surface};
use winit::platform::macos::WindowBuilderExtMacOS;

use crate::{editor, widget};

pub fn setup_window(mut editor: editor::Editor) {
    use cocoa::{appkit::NSView, base::id as cocoa_id};

    use core_graphics_types::geometry::CGSize;

    use foreign_types_shared::{ForeignType, ForeignTypeRef};
    use metal_rs::{Device, MTLPixelFormat, MetalLayer};
    use objc::{rc::autoreleasepool, runtime::YES};

    use skia_safe::gpu::{mtl, BackendRenderTarget, DirectContext, SurfaceOrigin};

    use winit::{
        dpi::LogicalSize,
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        platform::macos::WindowExtMacOS,
        window::WindowBuilder,
    };

    let mut size = LogicalSize::new(1600_i32, 1600_i32);

    let events_loop = EventLoop::new();

    let event_loop_proxy = events_loop.create_proxy();

    let window = WindowBuilder::new()
        .with_inner_size(size)
        .with_title("Lith".to_string())
        .with_titlebar_transparent(true)
        .with_fullsize_content_view(true)
        .with_transparent(false)
        .build(&events_loop)
        .unwrap();

    editor.on_window_create(
        event_loop_proxy,
        widget::Size {
            width: (size.width as f64 * window.scale_factor()) as f32,
            height: (size.height as f64 * window.scale_factor()) as f32,
        },
    );

    let device = Device::system_default().expect("no device found");

    let metal_layer = {
        let draw_size = window.inner_size();
        let layer: MetalLayer = MetalLayer::new();
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        layer.set_presents_with_transaction(false);
        layer.set_opaque(true);
        // change some stuff to make resizing nice
        // https://thume.ca/2019/06/19/glitchless-metal-window-resizing/

        unsafe {
            let view = window.ns_view() as cocoa_id;
            view.setWantsLayer(YES);
            view.setLayer(layer.as_ref() as *const MetalLayerRef as *mut objc::runtime::Object);
        }
        layer.set_drawable_size(CGSize::new(draw_size.width as f64, draw_size.height as f64));
        layer
    };

    // #[derive(Default)]
    // pub struct WebViewInstance;

    // impl WebViewDelegate for WebViewInstance {
    //     fn did_load(&mut self, _webview: WebView) {
    //         println!("did load");
    //     }
    //     fn did_appear(&self) {
    //         println!("did appear");
    //     }
    // }

    // let mut webview = WebView::with(
    //     WebViewConfig::default(),
    //     WebViewInstance::default(),
    // );
    // webview.load_url("https://google.com");
    // webview.width.constraint_equal_to_constant(600.0);
    // webview.height.constraint_equal_to_constant(600.0);

    // webview.layer = Layer::wrap(metal_layer.as_ptr() as *mut _);

    let command_queue = device.new_command_queue();

    let backend = unsafe {
        mtl::BackendContext::new(
            device.as_ptr() as mtl::Handle,
            command_queue.as_ptr() as mtl::Handle,
            std::ptr::null(),
        )
    };

    let mut context = DirectContext::new_metal(&backend, None).unwrap();
    let mut needs_update = true;
    let mut event_added = false;

    events_loop.run(move |event, _, control_flow| {
        autoreleasepool(|| {
            *control_flow = ControlFlow::Wait;
            let was_added = editor.add_event(&event);
            if was_added {
                event_added = true;
            }
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        editor.exit();
                        *control_flow = ControlFlow::Exit
                    }
                    WindowEvent::Resized(current_size) => {
                        metal_layer.set_drawable_size(CGSize::new(
                            current_size.width as f64,
                            current_size.height as f64,
                        ));

                        size.width = current_size.width as i32;
                        size.height = current_size.height as i32;
                        editor.window.size.width = size.width as f32;
                        editor.window.size.height = size.height as f32;
                        window.request_redraw();
                    }
                    _ => (),
                },
                Event::MainEventsCleared => {
                    // TODO: I would need to signal if there is any waiting
                    // work left to do from our wasm modules.
                    // If there is no work left, we don't need to do anything
                    // Well, as long as we define this properly. A module could
                    // need a tick, because they are totally driven by the editor
                    editor.end_frame();

                    // This needs to happen unconditonally, because
                    // these are external things that can create events
                    editor.process_per_frame_actions();

                    if editor.events.events_for_frame().len() > 0 {
                        needs_update = true;
                    }
                    if editor.wasm_messenger.number_of_pending_requests() > 0 {
                        needs_update = true;
                    }

                    // TODO:
                    // if I want to have 0 cpu usage when I'm not actually
                    // doing anything I need to signal to winit when I am
                    // doing things. That means I need to have some other thread
                    // looking at if there is work to be done and signalling here
                    // Or maybe I'm just not think clearly enough about this
                    // But I think that is the case for my process stuff
                    // It would be nice to have something checking on that.
                    // But I also think I can make that part more event driven
                    // rather than per thread.

                    // if !event_added && !needs_update {
                    //     return;
                    // }
                    if event_added {
                        event_added = false;
                    }

                    if needs_update {
                        needs_update = editor.update();
                    }
                    window.set_cursor_icon(editor.cursor_icon);

                    // This messes up fps counter
                    // Not sure how I would fix that
                    // I guess I could separate the editor
                    // from the fps counter?
                    // Really not sure
                    if needs_update {
                        if editor.should_redraw() {
                            window.request_redraw();
                        }
                    }
                }
                Event::RedrawRequested(_) => {
                    // TODO: Determine if this is a good idea or not.
                    // I am also setting this with move. Maybe I shouldn't?
                    // This lets me drop things in the correct spot
                    // unsafe {
                    //     let size = window.inner_size();
                    //     let point = NSWindow::mouseLocationOutsideOfEventStream(window.ns_window() as cocoa_id);
                    //     let logical_height = size.to_logical::<i32>(window.scale_factor()).height;
                    //     let logical_point = LogicalPosition::new(point.x as i32, logical_height - point.y as i32);
                    //     let physical_point = logical_point.to_physical::<i32>(window.scale_factor());

                    //     editor.set_mouse_position(physical_point.x as f32, physical_point.y as f32);
                    // }

                    if let Some(drawable) = metal_layer.next_drawable() {
                        let drawable_size = {
                            let size = metal_layer.drawable_size();
                            Size::new(size.width as scalar, size.height as scalar)
                        };

                        let mut surface = unsafe {
                            let texture_info =
                                mtl::TextureInfo::new(drawable.texture().as_ptr() as mtl::Handle);

                            let backend_render_target = BackendRenderTarget::new_metal(
                                (drawable_size.width as i32, drawable_size.height as i32),
                                1,
                                &texture_info,
                            );

                            Surface::from_backend_render_target(
                                &mut context,
                                &backend_render_target,
                                SurfaceOrigin::TopLeft,
                                ColorType::BGRA8888,
                                None,
                                None,
                            )
                            .unwrap()
                        };

                        editor.draw(surface.canvas());

                        surface.flush_and_submit();

                        let command_buffer = command_queue.new_command_buffer();
                        command_buffer.present_drawable(drawable);
                        command_buffer.commit();
                    }
                    // editor.next_frame();
                }
                _ => {}
            }
        });
    });
}
