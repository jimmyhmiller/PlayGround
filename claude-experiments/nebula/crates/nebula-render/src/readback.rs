//! Occasional GPU->CPU readback of positions (for exact camera fitting and,
//! later, picking). Deliberately gated to manageable sizes by callers — copying
//! and mapping a multi-gigabyte buffer every fit would defeat the point.

use crate::gpu::Gpu;
use crate::scene::GpuGraph;
use nebula_core::Pos;

/// Read back a single node's position (8 bytes). Cheap enough to call per frame
/// while a node is selected, so the selection marker tracks it live.
pub fn read_one_position(gpu: &Gpu, graph: &GpuGraph, index: u32) -> Option<Pos> {
    if index as u64 >= graph.num_nodes {
        return None;
    }
    let offset = index as u64 * 8;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("one_pos_readback"),
        size: 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("one_pos") });
    enc.copy_buffer_to_buffer(&graph.positions, offset, &staging, 0, 8);
    gpu.queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().ok()?.ok()?;
    let data = slice.get_mapped_range();
    let p: Pos = *bytemuck::from_bytes(&data[..8]);
    drop(data);
    staging.unmap();
    Some(p)
}

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
