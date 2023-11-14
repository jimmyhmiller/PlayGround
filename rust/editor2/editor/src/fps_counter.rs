use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

pub struct FpsCounter {
    pub start_time: Instant,
    pub frame_count: usize,
    pub fps: usize,
    pub times: HashMap<String, Stats>,
}

pub struct Stats {
    pub average: Duration,
    pub times: [Duration; 100],
    pub min: Duration,
    pub max: Duration,
    pub count: usize,
}

impl Stats {
    fn update(&mut self, time: Duration) {
        // Average is a running average of the last 100 frames.
        self.count += 1;
        self.count %= 100;
        self.count = self.count.max(1);
        self.times[self.count] = time;
        self.average = self.times.iter().sum::<Duration>() / 100 as u32;
        self.min = self.min.min(time);
        self.max = self.max.max(time);
    }
}

impl FpsCounter {
    pub fn new() -> Self {
        FpsCounter {
            start_time: Instant::now(),
            frame_count: 0,
            fps: 0,
            times: HashMap::new(),
        }
    }

    pub fn add_time(&mut self, name: &str, time: Duration) {
        if let Some(stats) = self.times.get_mut(name) {
            stats.update(time);
        } else {
            self.times.insert(
                name.to_string(),
                Stats {
                    average: time,
                    times: [Duration::ZERO; 100],
                    min: time,
                    max: time,
                    count: 1,
                },
            );
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
