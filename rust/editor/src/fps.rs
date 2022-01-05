use std::time::Instant;

pub struct FpsCounter {
    pub start_time: Instant,
    pub frame_count: usize,
    pub fps: usize,
}

impl FpsCounter {

    pub fn new() -> Self {
        FpsCounter {
            start_time: Instant::now(),
            frame_count: 0,
            fps: 0,
        }
    }
    
    pub fn reset(&mut self) {
        self.start_time = Instant::now();
        self.frame_count = 0;
    }

    pub fn tick(&mut self) -> usize {
        self.frame_count += 1;
        if self.start_time.elapsed().as_secs() >= 1 {
            self.fps = self.frame_count;
            self.reset();
        }
        self.fps
    }

}