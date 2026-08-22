use super::grid::{CellType, Grid3D};

use glam::{UVec3, Vec3};

pub struct MacGrid3D {
    pub dimensions: UVec3,

    pub u_x: Grid3D<f32>,
    pub u_y: Grid3D<f32>,
    pub u_z: Grid3D<f32>,

    pub divergence: Grid3D<f32>,
    pub pressure: Grid3D<f32>,
    pub cell_type: Grid3D<CellType>,
    pub sdf: Grid3D<f32>,
}

impl Clone for MacGrid3D {
    fn clone(&self) -> Self {
        Self {
            dimensions: self.dimensions,
            u_x: self.u_x.clone(),
            u_y: self.u_y.clone(),
            u_z: self.u_z.clone(),

            divergence: self.divergence.clone(),
            pressure: self.pressure.clone(),
            cell_type: self.cell_type.clone(),
            sdf: self.sdf.clone(),
        }
    }
}

impl MacGrid3D {
    pub fn new(dimensions: UVec3) -> Self {
        let x = dimensions.x;
        let y = dimensions.y;
        let z = dimensions.z;
        Self {
            dimensions,
            u_x: Grid3D::new(x + 1, y, z, 0.0),
            u_y: Grid3D::new(x, y + 1, z, 0.0),
            u_z: Grid3D::new(x, y, z + 1, 0.0),

            divergence: Grid3D::new(x, y, z, 0.0),
            pressure: Grid3D::new(x, y, z, 0.0),
            cell_type: Grid3D::new(x, y, z, CellType::Air),
            sdf: Grid3D::new(x, y, z, 1.6),
        }
    }

    pub fn check_wall(&self, x: u32, y: u32, z: u32) -> f32 {
        let cell_type = self.cell_type.get(x, y, z);
        if cell_type == CellType::Solid {
            -1.0
        } else {
            1.0
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ParticleType {
    Fluid,
    Solid,
}

pub struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    //pub previous_position: Vec3,
    pub previous_velocity: Vec3,
    pub flip_velocity: Vec3,
    pub density: f32,
    pub mass: f32,
    pub particle_type: ParticleType,
}

impl Particle {
    pub fn new_fluid(position: Vec3, velocity: Vec3) -> Self {
        Self {
            position,
            velocity,
            previous_velocity: velocity,
            density: 0.0,
            mass: 1.0,
            particle_type: ParticleType::Fluid,
            flip_velocity: Vec3::ZERO,
        }
    }
}
