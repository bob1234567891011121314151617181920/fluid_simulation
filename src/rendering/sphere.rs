use std::f32::consts::{PI, TAU};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SphereVertex {
    pub position: [f32; 3],
}

impl SphereVertex {
    pub fn describe() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }
}

pub fn create_uv_sphere(stacks: u32, slices: u32) -> (Vec<SphereVertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(((stacks + 1) * (slices + 1)) as usize);
    let mut indices = Vec::with_capacity((stacks * slices * 6) as usize);

    for stack in 0..=stacks {
        let phi = PI * stack as f32 / stacks as f32;
        let (sin_phi, cos_phi) = phi.sin_cos();
        for slice in 0..=slices {
            let theta = TAU * slice as f32 / slices as f32;
            let (sin_theta, cos_theta) = theta.sin_cos();
            vertices.push(SphereVertex {
                position: [sin_phi * cos_theta, cos_phi, sin_phi * sin_theta],
            });
        }
    }

    let row = slices + 1;
    for stack in 0..stacks {
        for slice in 0..slices {
            let top_left = stack * row + slice;
            let bottom_left = (stack + 1) * row + slice;
            indices.extend_from_slice(&[
                top_left,
                top_left + 1,
                bottom_left,
                top_left + 1,
                bottom_left + 1,
                bottom_left,
            ]);
        }
    }

    (vertices, indices)
}
