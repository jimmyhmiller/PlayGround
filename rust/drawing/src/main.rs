use std::collections::VecDeque;
use std::{thread::sleep, time::{Duration, SystemTime}};
use euclid::Angle;
use font_kit::{family_name::FamilyName, properties::Properties, source::SystemSource};
use mini_gl_fb::glutin::VirtualKeyCode;
use raqote::{DrawOptions, DrawTarget, PathBuilder, Point, SolidSource, Source, Transform};
const WIDTH: usize = 800;
const HEIGHT: usize = 800;


#[derive(Debug)]
enum Direction {
    Left,
    Right,
    Up,
    Down,
}

#[derive(Debug)]
struct Coord {
    x: isize,
    y: isize,
}

#[derive(Debug)]
struct Snake {
    heading: Direction,
    body: VecDeque<Coord>,
}

fn draw_filled_tile(dt: &mut DrawTarget, coord: &Coord, tile_size: (usize, usize)) {
    let mut pb = PathBuilder::new();
    pb.rect(coord.x as f32 * tile_size.0 as f32, coord.y as f32 * tile_size.1 as f32, tile_size.0 as f32, tile_size.1 as f32);
    let path = pb.finish();
    dt.fill(&path, &Source::Solid(SolidSource::from_unpremultiplied_argb(0xff, 0, 0xff, 0)), &DrawOptions::new());
}

fn move_snake(snake: &mut Snake, grid_size: (usize, usize)) {
    match snake.heading {
        Direction::Left => {
            snake.body.pop_back();
            let front = snake.body.pop_front().unwrap();
            let new_front = Coord {x: (front.x - 1).rem_euclid(grid_size.0 as isize), y: front.y};
            snake.body.push_front(front);
            snake.body.push_front(new_front);
        }
        Direction::Right => {
            snake.body.pop_back();
            let front = snake.body.pop_front().unwrap();
            let new_front = Coord {x: (front.x + 1).rem_euclid(grid_size.0 as isize), y: front.y};
            snake.body.push_front(front);
            snake.body.push_front(new_front);
        }
        Direction::Up => {
            snake.body.pop_back();
            let front = snake.body.pop_front().unwrap();
            let new_front = Coord {x: front.x, y: (front.y + 1).rem_euclid(grid_size.1 as isize)};
            snake.body.push_front(front);
            snake.body.push_front(new_front);
        }
        Direction::Down => {
            snake.body.pop_back();
            let front = snake.body.pop_front().unwrap();
            let new_front = Coord {x: front.x, y: (front.y - 1).rem_euclid(grid_size.1 as isize)};
            snake.body.push_front(front);
            snake.body.push_front(new_front);
        }
    }
}

fn main() {
    let mut fb = mini_gl_fb::gotta_go_fast("Hello world!", WIDTH as f64, HEIGHT as f64);


    let grid_size = (50,50);
    let tile_size = (WIDTH/grid_size.0, HEIGHT/grid_size.1);
    let mut snake = Snake {
        heading: Direction::Up,
        body: VecDeque::new(),
    };

    snake.body.push_back(Coord { x: 10, y: 48});
    snake.body.push_back(Coord { x: 10, y: 49});
    let mut dt = DrawTarget::new(WIDTH as i32, HEIGHT as i32);

   
    // Got it drawing, but it is upside down :(

    fb.glutin_handle_basic_input(|fb, input| {

        if input.key_is_down(VirtualKeyCode::Up) {
            snake.heading = Direction::Up
        }
        if input.key_is_down(VirtualKeyCode::Down) {
            snake.heading = Direction::Down
        }
        if input.key_is_down(VirtualKeyCode::Left) {
            snake.heading = Direction::Left
        }
        if input.key_is_down(VirtualKeyCode::Right) {
            snake.heading = Direction::Right
        }
        dt.clear(SolidSource::from_unpremultiplied_argb(0xff, 0xff, 0xff, 0xff));
        for tile in &snake.body {
            draw_filled_tile(&mut dt, tile, tile_size);
        }
        // println!("{:?}", snake);
       

        fb.update_buffer(&dt.get_data());
        fb.redraw();
        move_snake(&mut snake, grid_size);
        sleep(Duration::from_millis(32));
        true
    });
}
