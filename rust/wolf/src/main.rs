use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::Color;
use sdl2::rect::Point;
use sdl2::render::Canvas;
use std::collections::HashSet;

use sdl2::video::Window;

extern crate sdl2;

// code borrowed from https://lodev.org/cgtutor/raycasting.html

fn clear_canvas(canvas: &mut Canvas<Window>) {
    canvas.set_draw_color(Color::RGB(0, 0, 0));
    // fills the canvas with the color we set in `set_draw_color`.
    canvas.clear();
}

fn main() {
    let world_map = vec![
        vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ],
        vec![
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 1,
        ],
        vec![
            1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 1,
        ],
        vec![
            1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 0, 0, 0, 0, 0, 2, 2, 0, 2, 2, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 1,
        ],
        vec![
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 4, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 4, 0, 0, 0, 0, 5, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 4, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 4, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ],
        vec![
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ],
    ];

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let map_width: i32 = 1680;
    let map_height: i32 = 1050;

    let window = video_subsystem
        .window("Example", map_width as u32, map_height as u32)
        .fullscreen()
        .build()
        .unwrap();

    // Let's create a Canvas which we will use to draw in our Window
    let mut canvas: Canvas<Window> = window.into_canvas()
        .present_vsync() //< this means the screen cannot
        // render faster than your display rate (usually 60Hz or 144Hz)
        .build().unwrap();
    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut running = true;

    let mut timer = sdl_context.timer().unwrap();

    let mut pos_y: f64 = 22.0;
    let mut pos_x: f64 = 12.0;
    let mut dir_x: f64 = -1.0;
    let mut dir_y: f64 = 0.0;
    let mut plane_x: f64 = 0.0;
    let mut plane_y: f64 = 0.66;
    let mut time = timer.ticks();
    let mut old_time;

    while running {
        old_time = time;
        time = timer.ticks();
        let frame_time = (time - old_time) as f64 / 1000.0;
        let rot_speed: f64 = frame_time * 5.0;
        let move_speed: f64 = frame_time * 3.0;

        let keys: HashSet<Keycode> = event_pump
            .keyboard_state()
            .pressed_scancodes()
            .filter_map(Keycode::from_scancode)
            .collect();

        if keys.contains(&Keycode::Right) {
            let old_dir_x = dir_x;
            dir_x = dir_x * (-rot_speed).cos() - dir_y * (-rot_speed).sin();
            dir_y = old_dir_x * (-rot_speed).sin() + dir_y * (-rot_speed).cos();
            let old_plane_x = plane_x;
            plane_x = plane_x * (-rot_speed).cos() - plane_y * (-rot_speed).sin();
            plane_y = old_plane_x * (-rot_speed).sin() + plane_y * (-rot_speed).cos();
        }

        if keys.contains(&Keycode::Left) {
            let old_dir_x = dir_x;
            dir_x = dir_x * rot_speed.cos() - dir_y * rot_speed.sin();
            dir_y = old_dir_x * rot_speed.sin() + dir_y * rot_speed.cos();
            let old_plane_x = plane_x;
            plane_x = plane_x * rot_speed.cos() - plane_y * rot_speed.sin();
            plane_y = old_plane_x * rot_speed.sin() + plane_y * rot_speed.cos();
        }

        if keys.contains(&Keycode::Up) {
            if world_map[(pos_x + dir_x * move_speed) as usize][pos_y as usize] == 0 {
                pos_x += dir_x * move_speed;
            }
            if world_map[pos_x as usize][(pos_y + dir_y * move_speed) as usize] == 0 {
                pos_y += dir_y * move_speed;
            }
        }

        if keys.contains(&Keycode::Down) {
            if world_map[(pos_x - dir_x * move_speed) as usize][(pos_y) as usize] == 0 {
                pos_x -= dir_x * move_speed;
            }
            if world_map[(pos_x) as usize][(pos_y - dir_y * -move_speed) as usize] == 0 {
                pos_y -= dir_y * move_speed;
            }
        }

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => {
                    running = false;
                }

                Event::MouseButtonDown { x: _, y: _, .. } => {}
                _ => {}
            }
        }
        clear_canvas(&mut canvas);

        for x in 0..map_width {
            let camera_x = 2.0 * (x as f64) / (map_width as f64) - 1.0; //x-coordinate in camera space
            let ray_dir_x = dir_x + plane_x * camera_x;
            let ray_dir_y = dir_y + plane_y * camera_x;

            //which box of the map we're in
            let mut map_x: i32 = pos_x as i32;
            let mut map_y: i32 = pos_y as i32;

            //length of ray from current position to next x or y-side
            let mut side_dist_x: f64;
            let mut side_dist_y: f64;

            //length of ray from one x or y-side to next x or y-side
            let delta_dist_x = (1.0 / ray_dir_x).abs();
            let delta_dist_y = (1.0 / ray_dir_y).abs();
            let mut perp_wall_dist: f64;

            //what direction to step in x or y-direction (either +1 or -1)
            let step_x: i32;
            let step_y: i32;

            let mut hit = 0; //was there a wall hit?
            let mut side: i32 = 1; //was a NS or a EW wall hit?

            if ray_dir_x < 0.0 {
                step_x = -1;
                side_dist_x = (pos_x - (map_x as f64)) * delta_dist_x;
            } else {
                step_x = 1;
                side_dist_x = (map_x as f64 + 1.0 - pos_x) * delta_dist_x;
            }
            if ray_dir_y < 0.0 {
                step_y = -1;
                side_dist_y = (pos_y - map_y as f64) * delta_dist_y;
            } else {
                step_y = 1;
                side_dist_y = (map_y as f64 + 1.0 - pos_y) * delta_dist_y;
            }

            while hit == 0 {
                //jump to next map square, OR in x-direction, OR in y-direction
                if side_dist_x < side_dist_y {
                    side_dist_x += delta_dist_x;
                    map_x += step_x as i32;
                    side = 0;
                } else {
                    side_dist_y += delta_dist_y;
                    map_y += step_y as i32;
                    side = 1;
                }
                //Check if ray has hit a wall
                if world_map[map_x as usize][map_y as usize] > 0 {
                    hit = 1;
                }
            }
            if side == 0 {
                perp_wall_dist = (map_x as f64 - pos_x + (1.0 - step_x as f64) / 2.0) / ray_dir_x;
            } else {
                perp_wall_dist = (map_y as f64 - pos_y + (1.0 - step_y as f64) / 2.0) / ray_dir_y;
            }
            //Calculate height of line to draw on screen
            let line_height = map_height as f64 / perp_wall_dist;

            //calculate lowest and highest pixel to fill in current stripe
            let mut draw_start = -line_height / 2.0 + map_height as f64 / 2.0;
            if draw_start < 0.0 {
                draw_start = 0.0;
            }
            let mut draw_end = line_height / 2.0 + map_height as f64 / 2.0;
            if draw_end >= map_height as f64 {
                draw_end = map_height as f64 - 1.0;
            }

            let color = match world_map[map_x as usize][map_y as usize] {
                1 => Color::RGB(255, 0, 0),
                2 => Color::RGB(0, 255, 0),
                3 => Color::RGB(0, 0, 255),
                4 => Color::RGB(255, 255, 255),
                _ => Color::RGB(255, 255, 0),
            };

            canvas.set_draw_color(color);

            canvas
                .draw_line(
                    Point::new(x, draw_start as i32),
                    Point::new(x, draw_end as i32),
                )
                .unwrap();
        }

        canvas.present();
    }
}
