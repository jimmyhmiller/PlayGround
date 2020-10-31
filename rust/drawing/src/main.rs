use std::{thread::sleep, time::{Duration, SystemTime}};
use euclid::Angle;
use font_kit::{family_name::FamilyName, properties::Properties, source::SystemSource};
use mini_gl_fb::glutin::VirtualKeyCode;
use raqote::{DrawOptions, DrawTarget, PathBuilder, Point, SolidSource, Source, Transform};
const WIDTH: usize = 1000;
const HEIGHT: usize = 1000;


fn main() {
    let mut fb = mini_gl_fb::gotta_go_fast("Hello world!", WIDTH as f64, HEIGHT as f64);


    let font = SystemSource::new()
            .select_best_match(&[FamilyName::SansSerif], &Properties::new())
            .unwrap()
            .load()
            .unwrap();
    let mut dt = DrawTarget::new(WIDTH as i32, HEIGHT as i32);
    let mut x_velocity : f32 = 0.;
    let mut y_velocity : f32 = 5.;
    let mut rect_x = 0.;
    let mut rect_y = 0.;
    let rect_size = 50.;
    let mut events: Vec<&str> = vec![];

    // Got it drawing, but it is upside down :(

    fb.glutin_handle_basic_input(|fb, input| {
        if input.key_is_down(VirtualKeyCode::Left) && events.last().unwrap_or(&"nope") != &"left" {
            events.push("left");
        }
        if input.key_is_down(VirtualKeyCode::Right) && events.last().unwrap_or(&"nope") != &"right" {
            events.push("right");
        }
        if input.key_is_down(VirtualKeyCode::Up) && events.last().unwrap_or(&"nope") != &"up" {
            events.push("up");
        }
        if input.key_is_down(VirtualKeyCode::Down) && events.last().unwrap_or(&"nope") != &"down" {
            events.push("down");
        }

        dt.clear(SolidSource::from_unpremultiplied_argb(0xff, 0xff, 0xff, 0xff));
        let pos_string = format!("{:}", "test");
        dt.draw_text(&font, 
            36., 
            &pos_string, 
            Point::new(0., 500.),
            &Source::Solid(SolidSource::from_unpremultiplied_argb(0xff, 0, 0, 0)),
            &DrawOptions::new(),
        );
        // y_velocity = y_velocity.min(10. as f32);
        // x_velocity = x_velocity.min(10. as f32);

        rect_x += x_velocity as f32;
        rect_y += y_velocity as f32;
        rect_x = rect_x.rem_euclid(WIDTH as f32);
        rect_y = rect_y.rem_euclid(HEIGHT as f32);
        let mut pb = PathBuilder::new();
        pb.rect(rect_x, rect_y, rect_size, rect_size);
        let path = pb.finish();
        dt.fill(&path, &Source::Solid(SolidSource::from_unpremultiplied_argb(0xff, 0, 0xff, 0)), &DrawOptions::new());
        fb.update_buffer(&dt.get_data());
        fb.redraw();
        sleep(Duration::from_millis(16));
        true
    });
}
