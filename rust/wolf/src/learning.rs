use sdl2::render::Canvas;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use sdl2::keyboard::Keycode;
use sdl2::event::Event;

use sdl2::video::Window;
use std::cmp;

extern crate sdl2;


#[derive(Debug)]
struct Player {
    x: i32,
    y: i32,
    yvelocity: i32,
    xvelocity: i32
}

fn clear_canvas(canvas : &mut Canvas<Window>) {
    canvas.set_draw_color(Color::RGB(0, 0, 0));
    // fills the canvas with the color we set in `set_draw_color`.
    canvas.clear();
}

fn draw_player(canvas : &mut Canvas<Window>, player : &mut Player) -> Result<(), String> {


    // change the color of our drawing with a gold-color ...
    canvas.set_draw_color(Color::RGB(255, 210, 0));
    // A draw a rectangle which almost fills our window with it !
    canvas.fill_rect(rect_for_player(player))?;

    return Result::Ok(())
}

fn rect_for_player(player: &Player) -> Rect {
    Rect::new(player.x, player.y, 20, 20)
}





fn on_ground(player : &mut Player) -> bool {
    let rect = rect_for_player(player);
    rect.bottom() >= 600
}


fn gravity(player : &mut Player, collide: bool) {
    if collide || on_ground(player) {
        player.yvelocity *= -1;
        player.yvelocity /= 2;
    } else {
        player.yvelocity += 1;
    }
}


fn collision(player1 : &Player, player2 : &Player) -> bool {
    let rect1 = rect_for_player(player1);
    let rect2 = rect_for_player(player2);
    !rect2.contains_rect(rect1) && rect2.has_intersection(rect1)
}

fn move_player(player : &mut Player, ground : i32) {
    player.x = cmp::min(player.x + player.xvelocity, 800);
    player.y = cmp::min(player.y + player.yvelocity, ground);
}



fn main() {

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem.window("Example", 800, 600).build().unwrap();

    // Let's create a Canvas which we will use to draw in our Window
    let mut canvas : Canvas<Window> = window.into_canvas()
        .present_vsync() //< this means the screen cannot
        // render faster than your display rate (usually 60Hz or 144Hz)
        .build().unwrap();
    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut running = true;

    let mut players : Vec<Player> = vec![];

    while running {

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} | Event::KeyDown {keycode: Some(Keycode::Escape), ..} => {
                    running = false;
                },
                Event::MouseButtonDown {x, y, ..} => {
                    players.push(Player{x: x, y: y, xvelocity: 0, yvelocity: 0})
                }
                _ => {}
            }
        }


        for i in 0..players.len() {
            let mut ground = None;
            for j in 0..players.len()  {
                if j != i && collision(&players[j], &players[i]) {
                    ground = Some(rect_for_player(&players[j]).top() + (rect_for_player(&players[i]).height() as i32));
                }
            }
            gravity(&mut players[i], ground.is_some());
            move_player(&mut players[i], ground.unwrap_or(580));
        }

        clear_canvas(&mut canvas);
        for mut player in &mut players {
            draw_player(&mut canvas, &mut player).unwrap();
        }
        canvas.present();
    }
}
