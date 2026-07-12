//! Occasional GPU->CPU readback of positions (for exact camera fitting and,
//! later, picking). Deliberately gated to manageable sizes by callers — copying
//! and mapping a multi-gigabyte buffer every fit would defeat the point.

use crate::gpu::Gpu;
use crate::scene::GpuGraph;
use nebula_core::Pos;

pub fn read_positions(gpu: &Gpu, graph: &GpuGraph) -> Option<Vec<Pos>> {
    let n = graph.num_nodes as usize;
    if n == 0 {
        return Some(Vec::new());
    }
    let bytes = (n * std::mem::size_of::<Pos>()) as u64;

    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pos_readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
    enc.copy_buffer_to_buffer(&graph.positions, 0, &staging, 0, bytes);
    gpu.queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    // Block until the copy completes and the mapping is ready.
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    match rx.recv() {
        Ok(Ok(())) => {}
        _ => return None,
    }

    let data = slice.get_mapped_range();
    let out: Vec<Pos> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Some(out)
}
