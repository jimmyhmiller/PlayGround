#![allow(dead_code)]
use std::f64::EPSILON;

// This whole file is a ChatGPT translation of perfect freehand
// The outline function didn't turn out well.
// I should try making my own ink instead.


use skia_safe::{Paint, Color, Canvas};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrokePoint {
    pub point: [f64; 2],
    pub pressure: f64,
    pub vector: [f64; 2],
    pub distance: f64,
    pub running_length: f64,
}

impl StrokePoint {
    fn equal(&self, other: &StrokePoint) -> bool {
        (self.point[0] - other.point[0]).abs() < EPSILON
            && (self.point[1] - other.point[1]).abs() < EPSILON
            && (self.pressure - other.pressure).abs() < EPSILON
            && (self.vector[0] - other.vector[0]).abs() < EPSILON
            && (self.vector[1] - other.vector[1]).abs() < EPSILON
            && (self.distance - other.distance).abs() < EPSILON
            && (self.running_length - other.running_length).abs() < EPSILON
    }
}

pub struct StrokeOptions {
    size: Option<f64>,
    thinning: Option<f64>,
    smoothing: Option<f64>,
    easing: Option<fn(f64) -> f64>,
    simulate_pressure: Option<bool>,
    start: Option<StrokeCapTaperEasing>,
    end: Option<StrokeCapTaperEasing>,
    last: Option<bool>,
}

impl StrokeOptions {
    pub fn new() -> Self {
        Self {
            size: None,
            thinning: None,
            smoothing: None,
            easing: None,
            simulate_pressure: None,
            start: None,
            end: None,
            last: None,
        }
    }
}

struct StrokeCapTaperEasing {
    cap: StrokeCap,
    taper: f64,
    easing: Option<fn(f64) -> f64>,
}

impl Default for StrokeCapTaperEasing {
    fn default() -> Self {
        Self {
            cap: StrokeCap::Round,
            taper: 0.0,
            easing: None,
        }
    }
}

enum StrokeCap {
    Round,
    Square,
    Butt,
}

impl Default for StrokeCap {
    fn default() -> Self {
        Self::Round
    }
}

pub fn get_stroke_points(points: Vec<[f64; 2]>, options: StrokeOptions) -> Vec<StrokePoint> {
    let streamline = options.smoothing.unwrap_or(0.5);
    let size = options.size.unwrap_or(16.0);
    let last = options.last.unwrap_or(false);

    if points.is_empty() {
        return Vec::new();
    }

    let t = 0.15 + (1.0 - streamline) * 0.85;

    let mut pts = points;
    if pts.len() == 2 {
        let last = pts[1];
        pts = pts[..pts.len() - 1].to_vec();
        for i in 1..5 {
            pts.push(lrp(pts[0], last, i as f64 / 4.0));
        }
    }

    if pts.len() == 1 {
        pts = vec![pts[0], add([pts[0][0], pts[0][1]], [1.0, 1.0])];
    }

    let mut stroke_points = vec![StrokePoint {
        point: pts[0],
        pressure: if pts[0][1] >= 0.0 { pts[0][1] } else { 0.25 },
        vector: [1.0, 1.0],
        distance: 0.0,
        running_length: 0.0,
    }];

    let mut has_reached_minimum_length = false;
    let mut running_length = 0.0;
    let mut prev = stroke_points[0];

    let max = pts.len() - 1;

    for (i, point) in pts.iter().enumerate().skip(1) {
        let point = if last && i == max {
            point[..2].to_vec()
        } else {
            lrp(prev.point, *point, t).to_vec()
        };

        let pressure = if point[1] >= 0.0 { point[1] } else { 0.25 };
        let vector = sub([point[0], point[1]], prev.point).to_vec();
        let distance = dist([point[0], point[1]], prev.point);
        running_length += distance;

        let stroke_point = StrokePoint {
            point: [point[0], point[1]],
            pressure,
            vector: [vector[0], vector[1]],
            distance,
            running_length,
        };

        if !has_reached_minimum_length {
            if running_length > size * 0.5 {
                has_reached_minimum_length = true;
            }
        } else {
            stroke_points.push(stroke_point);
        }

        prev = stroke_point;
    }

    if !has_reached_minimum_length && stroke_points.len() > 1 {
        stroke_points.pop();
    }

    if options.thinning.is_some() {
        let thinning = options.thinning.unwrap();
        for stroke_point in stroke_points.iter_mut() {
            let size = size * (1.0 - stroke_point.pressure * thinning);
            stroke_point.point = add(
                stroke_point.point,
                [
                    stroke_point.vector[0] * size * 0.5,
                    stroke_point.vector[1] * size * 0.5,
                ],
            );
            stroke_point.pressure = (stroke_point.pressure - 0.5) * 2.0;
        }
    }

    if options.smoothing.is_some() {
        let smoothing = options.smoothing.unwrap();
        for stroke_point in stroke_points.iter_mut() {
            let t = stroke_point.running_length / running_length;
            stroke_point.pressure = (options.easing.unwrap())(t);
        }

        if stroke_points.len() > 2 {
            let mut prev = stroke_points[0];
            let last = stroke_points[stroke_points.len() - 1].clone();
            let stroke_points_clone = stroke_points.clone();
            for (index, stroke_point) in stroke_points[1..].iter_mut().enumerate() {
                let next = if stroke_point.equal(&last) {
                    None
                } else {
                    Some(stroke_points_clone[index + 2])
                };

                stroke_point.point = lrp(prev.point, stroke_point.point, smoothing);
                if let Some(next) = next {
                    stroke_point.point = lrp(stroke_point.point, next.point, smoothing);
                }

                prev = stroke_point.clone();
            }
        }
    }

    if options.simulate_pressure.unwrap_or(false) {
        for stroke_point in stroke_points.iter_mut() {
            stroke_point.pressure = stroke_point.pressure.powf(stroke_point.distance / size);
        }
    }

    if options.start.is_some() {
        let start = options.start.unwrap();
        match start.cap {
            StrokeCap::Round => {
                let t = (1.0 - start.taper) * 0.5;
                let t = (start.easing.unwrap())(t);
                let t_inv = 1.0 - t;
                let t_inv_sq = t_inv * t_inv;
                let t_sq = t * t;

                let pressure = (t_inv_sq + 2.0 * t_inv * t + t_sq) * stroke_points[0].pressure;
                let point = lrp(stroke_points[0].point, stroke_points[1].point, t);

                stroke_points[0] = StrokePoint {
                    point,
                    pressure,
                    vector: stroke_points[0].vector,
                    distance: stroke_points[0].distance,
                    running_length: stroke_points[0].running_length,
                };
            }
            _ => {}
        }

        if options.end.is_some() {
            let end = options.end.unwrap();
            match end.cap {
                StrokeCap::Round => {
                    let t = (1.0 - end.taper) * 0.5;
                    let t = (end.easing.unwrap())(t);
                    let t_inv = 1.0 - t;
                    let t_inv_sq = t_inv * t_inv;
                    let t_sq = t * t;

                    let pressure = (t_inv_sq + 2.0 * t_inv * t + t_sq)
                        * stroke_points.last().unwrap().pressure;
                    let point = lrp(
                        stroke_points[stroke_points.len() - 2].point,
                        stroke_points.last().unwrap().point,
                        t,
                    );
                    let len = stroke_points.len();
                    stroke_points[len - 1] = StrokePoint {
                        point,
                        pressure,
                        vector: stroke_points.last().unwrap().vector,
                        distance: stroke_points.last().unwrap().distance,
                        running_length: stroke_points.last().unwrap().running_length,
                    };
                }
                _ => {}
            }
        }
    }
    stroke_points
}

fn add(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn is_equal(a: [f64; 2], b: [f64; 2]) -> bool {
    (a[0] - b[0]).abs() < EPSILON && (a[1] - b[1]).abs() < EPSILON
}

fn lrp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

fn mul(a: [f64; 2], b: f64) -> [f64; 2] {
    [a[0] * b, a[1] * b]
}

fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

pub fn draw_strokes(points: &[StrokePoint], canvas: &mut Canvas) {

    if points.is_empty() {
        return;
    }
    let mut paint = Paint::default();
    paint.set_color(Color::from_argb(255, 0, 0, 0)); // Set the color to black

    for point in points {
        canvas.draw_circle((point.point[0] as f32, point.point[1] as f32), 2.0, &paint);
    }

}

pub fn draw_points(points: &[[f64; 2]], canvas: &mut Canvas) {

    if points.is_empty() {
        return;
    }
    let mut paint = Paint::default();
    paint.set_color(Color::from_argb(255, 255, 0, 0)); // Set the color to black

    for point in points {
        canvas.draw_circle((point[0] as f32, point[1] as f32), 2.0, &paint);
    }

}

use std::cmp::min;

const RATE_OF_PRESSURE_CHANGE : f64 = 0.275;

fn get_stroke_radius(size: f64, thinning: f64, pressure: f64, easing: &dyn Fn(f64) -> f64) -> f64 {
    size * easing(0.5 - thinning * (0.5 - pressure))
}


fn find_corner(p1: &[f64; 2], p2: &[f64; 2], p3: &[f64; 2], v1: &[f64; 2], v2: &[f64; 2], min_distance: f64) -> Option<[f64; 2]> {
    let v1_len = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
    let v2_len = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();
    let v1 = [v1[0] / v1_len, v1[1] / v1_len];
    let _v2 = [v2[0] / v2_len, v2[1] / v2_len];
    let p = [p1[0] + v1[0], p1[1] + v1[1]];
    let mut corner = None;
    let mut dist = min_distance;
    let d1 = ((p3[0] - p[0]) * (p3[0] - p[0]) + (p3[1] - p[1]) * (p3[1] - p[1])).sqrt();
    if d1 > dist {
        corner = Some(p3.clone());
        dist = d1;
    }
    let d2 = ((p2[0] - p[0]) * (p2[0] - p[0]) + (p2[1] - p[1]) * (p2[1] - p[1])).sqrt();
    if d2 > dist {
        corner = Some(p2.clone());
    }
    corner
}


pub fn get_stroke_outline_points(
    points: &[StrokePoint],
    options: StrokeOptions,
) -> Vec<[f64; 2]> {

    // This code is a broken translation from ChatGPT

    if points.len() < 2 {
        return vec![];
    }
    let size = options.size.unwrap_or(16.0);
    let smoothing = options.smoothing.unwrap_or(0.5);
    let thinning = options.thinning.unwrap_or(0.5);
    let simulate_pressure = options.simulate_pressure.unwrap_or(true);
    let easing = options.easing.unwrap_or(|t| t);
    let start = options.start.unwrap_or_else(|| StrokeCapTaperEasing::default());
    let end = options.end.unwrap_or_else(|| StrokeCapTaperEasing::default());
    let is_complete = options.last.unwrap_or(false);
    let _cap_start = start.cap;
    let _taper_start_ease = start.easing.unwrap_or(|t| t * (2.0 - t));
    let _cap_end = end.cap;
    let _taper_end_ease = end.easing.unwrap_or(|t| t - 1.0 * t * t * t + 1.0);
    let total_length = points[points.len() - 1].running_length;
    let _taper_start = start.taper;
    let _taper_end = end.taper;
    let min_distance = f64::powf(size * smoothing, 2.0);
    let mut left_pts = Vec::new();
    let mut right_pts = Vec::new();
    let mut _prev_pressure = points[0..min(points.len(), 10)]
        .iter()
        .fold(points[0].pressure, |acc, curr| {
            let mut pressure = curr.pressure;
            if simulate_pressure {
                let sp = f64::min(1.0, curr.distance / size);
                let rp = f64::min(1.0, 1.0 - sp);
                pressure = f64::min(1.0, acc + (rp - acc) * (sp * RATE_OF_PRESSURE_CHANGE));
            }
            (acc + pressure) / 2.0
        });
    let mut radius;
    let mut first_radius: Option<f64> = None;
    let mut prev_vector = points[0].vector;
    let mut pl = points[0].point;
    let mut pr = pl;
    let mut tl = pl;
    let mut tr = pr;
    let mut is_prev_point_sharp_corner = false;
    let mut distance = 0.0;
    for (i, point) in points.iter().enumerate().skip(1).take(points.len() - 2) {
        let pressure = point.pressure;
        let ppoint = point.point;
        let vector = point.vector;
        distance += point.distance;
        let running_length = point.running_length;
        if i < points.len() - 1 && total_length - running_length < 3.0 {
            continue;
        }
        if thinning != 0.0 {
            radius = size / 2.0;
        } else {
            radius = get_stroke_radius(size, thinning, pressure, &easing);
        }
        if first_radius.is_none() {
            first_radius = Some(radius);
        }
        let radius = f64::max(radius, 0.5);
        let radius_t = f64::max(radius, 0.5);
        let p = pl;
        let mut corner = None;
        if is_prev_point_sharp_corner {
            is_prev_point_sharp_corner = false;
        } else {
            corner = find_corner(
                &pl,
                &pr,
                &ppoint,
                &prev_vector,
                &vector,
                f64::powf(min_distance / 2.0, 0.5),
            );
        }
        if let Some(corner) = corner {
            pl = corner;
            is_prev_point_sharp_corner = true;
        } else {
            let t = 0.5 + cross(vector, prev_vector) / 2.0;
            let r = 1.0 / (1.0 + t * radius_t);
            let tr = add(mul(prev_vector, r * radius_t), mul(vector, radius_t - radius_t * r));
            pl = add(p, tr);
        }
        pr = add(p,  mul(vector, radius));
        if distance > min_distance {
            left_pts.push(tl);
            right_pts.push(tr);
            prev_vector = vector;
            tl = pl;
            tr = pr;
            distance = 0.0;
        }
        _prev_pressure = pressure;
    }
    left_pts.push(tl);
    right_pts.push(tr);
    if points.len() > 1 {
        let _last = points[points.len() - 1].point;
        // if cap_end {
        //     let cap = get_cap(
        //         &last,
        //         &right_pts[right_pts.len() - 1],
        //         &vector,
        //         &taper_end_ease,
        //         taper_end,
        //         size,
        //         easing,
        //         &easing,
        //     );
        //     left_pts.extend(cap.0);

        //     right_pts.extend(cap.1);
        // } else {

        // }
    }
    if is_complete {
        let mut outline = Vec::new();
        outline.extend(right_pts.into_iter().rev());
        outline.extend(left_pts);
        outline
    } else {
        left_pts
    }
}

