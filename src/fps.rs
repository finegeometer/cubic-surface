use std::collections::VecDeque;

pub struct FrameCounter {
    time: f64,
    recorded: VecDeque<f64>,
}

impl FrameCounter {
    pub fn new(time: f64) -> Self {
        Self {
            time,
            recorded: VecDeque::with_capacity(61),
        }
    }

    /// Tell the frame counter that a new frame has occurred, returning the time (in seconds) since the previous frame.
    pub fn frame(&mut self, time: f64) -> f64 {
        let old_time = self.time;
        let milliseconds = time - old_time;
        self.time = time;

        self.recorded.push_back(milliseconds);
        if self.recorded.len() > 60 {
            self.recorded.pop_front();
        }

        milliseconds * 1e-3
    }
}

impl std::fmt::Display for FrameCounter {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut iter = self.recorded.iter();
        if let Some(&t) = iter.next() {
            let mut min = t;
            let mut max = t;
            let mut sum = t;

            for &t in iter {
                min = min.min(t);
                max = max.max(t);
                sum += t;
            }

            let avg = sum / self.recorded.len() as f64;

            write!(
                f,
                "milliseconds per frame (min/max/avg): {:.2} {:.2} {:.2}",
                min, max, avg
            )
        } else {
            write!(f, "No data yet!")
        }
    }
}
