use std::collections::HashSet;
use std::collections::VecDeque;
use std::{thread::sleep, time::{Duration, SystemTime}};
use euclid::Angle;
use rand::Rng;
use font_kit::{family_name::FamilyName, properties::Properties, source::SystemSource};
use mini_gl_fb::glutin::VirtualKeyCode;
use raqote::{DrawOptions, DrawTarget, PathBuilder, Point, SolidSource, Source, Transform};
const WIDTH: usize = 800;
const HEIGHT: usize = 800;


#[derive(Debug, PartialEq, Eq)]
enum Direction {
    Left,
    Right,
    Up,
    Down,
}

#[derive(Debug, PartialEq, Eq, Hash, Copy, Clone)]
struct Coord {
    x: isize,
    y: isize,
}

#[derive(Debug)]
struct Snake {
    heading: Direction,
    body: VecDeque<Coord>,
    positions: HashSet<Coord>,
}

fn draw_filled_tile(dt: &mut DrawTarget, coord: &Coord, tile_size: (usize, usize), solid_source: SolidSource) {
    let mut pb = PathBuilder::new();
    pb.rect(coord.x as f32 * tile_size.0 as f32, coord.y as f32 * tile_size.1 as f32, tile_size.0 as f32, tile_size.1 as f32);
    let path = pb.finish();
    dt.fill(&path, &Source::Solid(solid_source), &DrawOptions::new());
}

fn add_one_heading(heading: &Direction, coord: &Coord, grid_size: (usize, usize)) -> Coord {
    match heading {
        Direction::Left => {
            Coord {x: (coord.x - 1).rem_euclid(grid_size.0 as isize), y: coord.y}
        }
        Direction::Right => {
            Coord {x: (coord.x + 1).rem_euclid(grid_size.0 as isize), y: coord.y}
        }
        Direction::Up => {
            Coord {x: coord.x, y: (coord.y + 1).rem_euclid(grid_size.1 as isize)}
        }
        Direction::Down => {
            Coord {x: coord.x, y: (coord.y - 1).rem_euclid(grid_size.1 as isize)}
        }
    }
}

fn move_snake(snake: &mut Snake, grid_size: (usize, usize)) -> bool {
    let back = snake.body.pop_back().unwrap();
    snake.positions.remove(&back);
    grow_snake(snake, grid_size)
}

fn grow_snake(snake: &mut Snake, grid_size: (usize, usize)) -> bool {
    let front = snake.body.pop_front().unwrap();
    let new_front = add_one_heading(&snake.heading, &front, grid_size);
    if snake.positions.contains(&new_front) {
        return false;
    }
    snake.body.push_front(front);
    snake.body.push_front(new_front);
    snake.positions.insert(new_front.clone());
    true
}

fn init_snake(x: isize, y: isize, grid_size: (usize, usize)) -> Snake {
    let mut s = Snake {
        heading: Direction::Up,
        body: VecDeque::new(),
        positions: HashSet::new(),
    };
    let coord = Coord {x: x, y: y};
    s.body.push_front(coord.clone());
    s.positions.insert(coord);
    grow_snake(&mut s, grid_size);
    s
}

fn draw_food(dt: &mut DrawTarget, coord: &Coord, tile_size: (usize, usize), grid_size: (usize, usize)) {
    draw_filled_tile(dt, &coord, tile_size, SolidSource::from_unpremultiplied_argb(0xff, 0xff, 0, 0))
}

fn gen_food_location(grid_size: (usize, usize)) -> Coord {
    let mut rng = rand::thread_rng();
    let coord = Coord { x: rng.gen_range(0, grid_size.0 as isize), y: rng.gen_range(0, grid_size.1 as isize)};
    coord
}


fn main() {
    let mut fb = mini_gl_fb::gotta_go_fast("Hello world!", WIDTH as f64, HEIGHT as f64);
    let snake_color = SolidSource::from_unpremultiplied_argb(0xff, 0, 0xff, 0);

    let mut is_start = true;

    let grid_size = (25,25);
    let tile_size = (WIDTH/grid_size.0, HEIGHT/grid_size.1);
    let mut snake = init_snake(12, 12, grid_size);
    let mut food_location = gen_food_location(grid_size);

    let mut dt = DrawTarget::new(WIDTH as i32, HEIGHT as i32);

   
    // Got it drawing, but it is upside down :(

    fb.glutin_handle_basic_input(|fb, input| {


        if input.key_is_down(VirtualKeyCode::Up) && snake.heading != Direction::Down {
            snake.heading = Direction::Up
        }
        if input.key_is_down(VirtualKeyCode::Down) && snake.heading != Direction::Up {
            snake.heading = Direction::Down
        }
        if input.key_is_down(VirtualKeyCode::Left) && snake.heading != Direction::Right  {
            snake.heading = Direction::Left
        }
        if input.key_is_down(VirtualKeyCode::Right) && snake.heading != Direction::Left  {
            snake.heading = Direction::Right
        }
        dt.clear(SolidSource::from_unpremultiplied_argb(0xff, 0xff, 0xff, 0xff));

        if is_start && input.key_is_down(VirtualKeyCode::Return) {
            is_start = false;
        } else if is_start {
            sleep(Duration::from_millis(64));
            return true
        }
        let moved = move_snake(&mut snake, grid_size);
        if !moved {
            is_start = true;
            snake = init_snake(12, 12, grid_size);
            // need to reset state
        }
        if snake.positions.contains(&food_location) {
            grow_snake(&mut snake, grid_size);
            food_location = gen_food_location(grid_size);
        }
        for tile in &snake.body {
            draw_filled_tile(&mut dt, tile, tile_size, snake_color);
        }
        draw_food(&mut dt, &food_location, tile_size, grid_size);
       

        fb.update_buffer(&dt.get_data());
        fb.redraw();
        sleep(Duration::from_millis(64));
        true
    });
}
