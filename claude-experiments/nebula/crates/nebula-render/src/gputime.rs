//! Lightweight GPU pass timing via timestamp queries. Measures the edge-cull
//! compute pass and the main render pass so we can tell whether a frame is
//! vertex/compute-bound or raster/blend-bound. No-op (None) when the adapter
//! lacks TIMESTAMP_QUERY.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

const Q_CULL_BEGIN: u32 = 0;
const Q_CULL_END: u32 = 1;
const Q_MAIN_BEGIN: u32 = 2;
const Q_MAIN_END: u32 = 3;
const Q_COUNT: u32 = 4;

const IDLE: u8 = 0;
const MAPPING: u8 = 1;
const READY: u8 = 2;

pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    period_ns: f32,
    state: Arc<AtomicU8>,
    /// Whether the current frame is being timed (armed by `begin_frame`).
    armed: bool,
    /// Whether the cull pass actually ran in the armed frame.
    cull_timed: bool,
    /// Latest measured durations, milliseconds.
    pub cull_ms: Option<f32>,
    pub main_ms: Option<f32>,
}

impl GpuTimer {
    /// Returns None if the device wasn't created with TIMESTAMP_QUERY.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frame_timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: Q_COUNT,
        });
        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts_resolve"),
            size: Q_COUNT as u64 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts_staging"),
            size: Q_COUNT as u64 * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Some(GpuTimer {
            query_set,
            resolve_buf,
            staging,
            period_ns: queue.get_timestamp_period(),
            state: Arc::new(AtomicU8::new(IDLE)),
            armed: false,
            cull_timed: false,
            cull_ms: None,
            main_ms: None,
        })
    }

    /// Harvest a finished readback (if any) and decide whether to time this
    /// frame. Only one timing readback is in flight at a time.
    pub fn begin_frame(&mut self) -> bool {
        if self.state.load(Ordering::Acquire) == READY {
            let raw: Vec<u64> = {
                let view = self.staging.slice(..).get_mapped_range();
                bytemuck::cast_slice::<u8, u64>(&view).to_vec()
            };
            self.staging.unmap();
            self.state.store(IDLE, Ordering::Release);
            let ms = |a: u64, b: u64| {
                b.saturating_sub(a) as f32 * self.period_ns / 1_000_000.0
            };
            if self.cull_timed {
                self.cull_ms =
                    Some(ms(raw[Q_CULL_BEGIN as usize], raw[Q_CULL_END as usize]));
            }
            self.main_ms = Some(ms(raw[Q_MAIN_BEGIN as usize], raw[Q_MAIN_END as usize]));
        }
        self.armed = self.state.load(Ordering::Acquire) == IDLE;
        if self.armed {
            self.cull_timed = false;
        }
        self.armed
    }

    /// Timestamp writes for the edge-cull compute pass, if timing this frame.
    /// Call `set_cull_ran` afterwards with whether the pass actually ran.
    pub fn cull_writes(&self) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        if !self.armed {
            return None;
        }
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(Q_CULL_BEGIN),
            end_of_pass_write_index: Some(Q_CULL_END),
        })
    }

    pub fn set_cull_ran(&mut self, ran: bool) {
        if self.armed {
            self.cull_timed = ran;
        }
    }

    /// Timestamp writes for the main render pass, if timing this frame.
    pub fn main_writes(&self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        if !self.armed {
            return None;
        }
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(Q_MAIN_BEGIN),
            end_of_pass_write_index: Some(Q_MAIN_END),
        })
    }

    /// Encode query resolution. Call after the timed passes, before submit.
    pub fn resolve(&self, enc: &mut wgpu::CommandEncoder) {
        if !self.armed {
            return;
        }
        enc.resolve_query_set(&self.query_set, 0..Q_COUNT, &self.resolve_buf, 0);
        enc.copy_buffer_to_buffer(
            &self.resolve_buf,
            0,
            &self.staging,
            0,
            Q_COUNT as u64 * 8,
        );
    }

    /// Kick off the async readback. Call right after queue.submit().
    pub fn after_submit(&mut self) {
        if !self.armed {
            return;
        }
        self.state.store(MAPPING, Ordering::Release);
        let state = self.state.clone();
        self.staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            state.store(if r.is_ok() { READY } else { IDLE }, Ordering::Release);
        });
        self.armed = false;
    }
}
