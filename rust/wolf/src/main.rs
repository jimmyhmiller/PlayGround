use std::collections::HashSet;
use sdl2::{render::*, video::*, event::*, keyboard::*, *};

use wolf::world_map::get_world_map;

#[derive(Debug, PartialEq, Clone)]
struct Point<T> {
    x: T,
    y: T,
}

#[derive(Debug, PartialEq, Clone)]
struct Player {
    facing: Point<f64>,
    position: Point<f64>,
    camera: Point<f64>,
    rotation_speed: f64,
    move_speed: f64,
}

fn rotate_point(point: &mut Point<f64>, rotation_speed: f64) {
    point.x = point.x * rotation_speed.cos() - point.y * rotation_speed.sin();
    point.y = point.x * rotation_speed.sin() + point.y * rotation_speed.cos();
}

fn rotate_player(player: &mut Player, dir: f64) {
    rotate_point(&mut player.facing, dir * player.rotation_speed);
    rotate_point(&mut player.camera, dir * player.rotation_speed);
}

fn move_position(world: &Vec<Vec<usize>>, player: &mut Player, delta: Point<f64>) {
    if world[(player.position.x + delta.x) as usize][(player.position.y + delta.y) as usize] == 0 {
        player.position.x += delta.x;
        player.position.y += delta.y;
    }
}

fn negative(point: Point<f64>) -> Point<f64> {
    Point { x: point.x * -1.0, y:  point.y * -1.0 }
}

fn handle_key_presses(event_pump: &EventPump, player: &mut Player, world_map: &Vec<Vec<usize>>) {
    let keys: HashSet<Keycode> = event_pump
        .keyboard_state()
        .pressed_scancodes()
        .filter_map(Keycode::from_scancode)
        .collect();

    let delta = Point {
        x: player.facing.x * player.move_speed,
        y: player.facing.y * player.move_speed,
    };

    if keys.contains(&Keycode::Right) {
        rotate_player(player, -1.0)
    } else if keys.contains(&Keycode::Left) {
        rotate_player(player, 1.0)
    } else if keys.contains(&Keycode::Up) {
        move_position(&world_map, player, delta)
    } else if keys.contains(&Keycode::Down) {
        move_position(&world_map, player, negative(delta))
    } else {
        
    }
}

fn main() -> Result<(), String> {
    let world_map = get_world_map();

    let sdl_context = sdl2::init()?;

    let map_width: i32 = 500;
    let map_height: i32 = 500;

    let window = sdl_context.video()?
        .window("Example", map_width as u32, map_height as u32)
        .build()
        .unwrap();

    // Let's create a Canvas which we will use to draw in our Window
    let _canvas: Canvas<Window> = window.into_canvas().present_vsync().build().unwrap();
    let mut event_pump = sdl_context.event_pump()?;

    let mut player = Player {
        facing: Point { x: -1.0, y: 0.0 },
        position: Point { x: 12.0, y: 22.0 },
        camera: Point { x: 0.0, y: 0.66 },
        rotation_speed: 0.0,
        move_speed: 0.0,
    };

    loop {
        match event_pump.poll_event() {
            Some(Event::Quit { .. }) => {
                ::std::process::exit(0)
            }
             _ => {}
        }

        handle_key_presses(&event_pump, &mut player, &world_map);

    }
}