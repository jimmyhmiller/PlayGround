use std::io::{self, Write};
use std::thread;
use std::time::Duration;

// 8 retro colors
const COLORS: [&str; 8] = [" ", "█", "▓", "▒", "░", "·", "•", "●"];
const WIDTH: usize = 40;
const HEIGHT: usize = 20;

struct Grid {
    cells: [[usize; WIDTH]; HEIGHT],
}

impl Grid {
    fn new() -> Self {
        Self {
            cells: [[0; WIDTH]; HEIGHT],
        }
    }

    fn add_sand(&mut self, x: usize, y: usize) {
        if x < WIDTH && y < HEIGHT {
            self.cells[y][x] = 1; // Sand color
        }
    }

    fn update(&mut self) {
        // Process from bottom to top
        for y in (1..HEIGHT).rev() {
            for x in 0..WIDTH {
                if self.cells[y][x] == 0 && self.cells[y-1][x] != 0 {
                    // Move particle down if space is empty
                    self.cells[y][x] = self.cells[y-1][x];
                    self.cells[y-1][x] = 0;
                }
            }
        }
    }

    fn display(&self) {
        // Clear screen
        print!("\x1B[2J\x1B[1;1H");
        
        for row in &self.cells {
            for &cell in row {
                print!("{}", COLORS[cell]);
            }
            println!();
        }
        io::stdout().flush().unwrap();
    }
}

fn main() {
    let mut grid = Grid::new();
    let mut running = true;
    
    println!("Retro Sand Game!");
    println!("Click to add sand particles (Ctrl+C to quit)");
    thread::sleep(Duration::from_millis(1000));
    
    while running {
        grid.display();
        grid.update();
        
        // Add sand at random positions for demo
        let x = rand::random::<usize>() % WIDTH;
        grid.add_sand(x, 0);
        
        thread::sleep(Duration::from_millis(100));
    }
}