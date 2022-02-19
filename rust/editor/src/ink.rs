use crate::renderer::Renderer;


type Point = (i32, i32);

pub struct Ink {
    pub id: usize,
    pub points: Vec<Point>,
}

impl Ink {
    pub fn new(id: usize) -> Self {
        Self { 
            id,
            points: Vec::new()
        }
    }

    pub fn add_point(&mut self, point: Point) {
        self.points.push(point);
    }

}

pub struct InkManager {
    pub drawing_counter : usize,
    pub drawings: Vec<Ink>,
    pub current_drawing: Option<usize>,
}

impl InkManager {
    pub fn new() -> Self {
        Self {
            drawing_counter: 0,
            drawings: Vec::new(),
            current_drawing: None
        }
    }

    pub fn get_drawing_id(&mut self) -> usize {
        self.drawing_counter += 1;
        self.drawing_counter
    }

    pub fn start_drawing(&mut self, point: Point) {
        let id = self.get_drawing_id();
        let mut ink = Ink::new(id);
        ink.add_point(point);
        self.drawings.push(ink);
        self.current_drawing = Some(id);
    }

    pub fn end_drawing(&mut self) {
        self.current_drawing = None;
    }

    pub fn get_drawing_by_id_mut(&mut self, id: usize) -> Option<&mut Ink> {
        self.drawings.iter_mut().find(|ink| ink.id == id)
    }

    pub fn add_point(&mut self, id: usize, point: Point) -> Option<()> {
        let drawing = self.get_drawing_by_id_mut(id)?;
        drawing.add_point(point);
        Some(())
    }

    pub fn add_to_current_drawing(&mut self, point: Point) -> Option<()> {
        let id = self.current_drawing?;
        self.add_point(id, point)
    }

    pub fn draw(&self, renderer: &mut Renderer) -> Result<(), String> {
        // Apparently there is no way to set line width in SDL2
        // So I probably need to do something else here.
        for drawing in &self.drawings {
            for pair in drawing.points.windows(2).into_iter() {
                let p1 = pair[0];
                let p2 = pair[1];
                renderer.canvas.draw_line(p1, p2)?;
            }
        }

        Ok(())
    }
    
}