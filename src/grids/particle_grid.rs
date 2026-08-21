use glam::UVec3;

use super::grid::{CellType, Grid3D};
use super::mac_grid::{MacGrid3D, Particle, ParticleType};

const EMPTY: u32 = u32::MAX;
pub struct ParticleGrid {
    dimensions: UVec3,
    grid: Grid3D<u32>,
    pub cells: Vec<Vec<usize>>,
}

impl ParticleGrid {
    pub fn new(dimensions: UVec3) -> Self {
        Self {
            dimensions,
            grid: Grid3D::new(dimensions.x, dimensions.y, dimensions.z, EMPTY),
            cells: vec![],
        }
    }

    pub fn get_cell_neighbors(&self, index: UVec3, number_of_neighbors: UVec3) -> Vec<usize> {
        let mut neighbors = vec![];

        let min_x = index.x.saturating_sub(number_of_neighbors.x);
        let min_y = index.y.saturating_sub(number_of_neighbors.y);
        let min_z = index.z.saturating_sub(number_of_neighbors.z);

        let max_x = (index.x + number_of_neighbors.x).min(self.dimensions.x - 1);
        let max_y = (index.y + number_of_neighbors.y).min(self.dimensions.y - 1);
        let max_z = (index.z + number_of_neighbors.z).min(self.dimensions.z - 1);

        for x in min_x..=max_x {
            for y in min_y..=max_y {
                for z in min_z..=max_z {
                    let cell_index = self.grid.get(x, y, z);
                    if cell_index != EMPTY {
                        neighbors.extend_from_slice(&self.cells[cell_index as usize]);
                    }
                }
            }
        }
        neighbors
    }

    pub fn get_wall_neighbors(&self, index: UVec3, number_of_neighbors: UVec3) -> Vec<usize> {
        let mut neighbors = Vec::new();

        let min_x = index.x.saturating_sub(number_of_neighbors.x);
        let min_y = index.y.saturating_sub(number_of_neighbors.y);
        let min_z = index.z.saturating_sub(number_of_neighbors.z);

        // Exclusive upper bounds
        let max_x = index
            .x
            .saturating_add(number_of_neighbors.x)
            .min(self.dimensions.x);

        let max_y = index
            .y
            .saturating_add(number_of_neighbors.y)
            .min(self.dimensions.y);

        let max_z = index
            .z
            .saturating_add(number_of_neighbors.z)
            .min(self.dimensions.z);

        for x in min_x..max_x {
            for y in min_y..max_y {
                for z in min_z..max_z {
                    let cell_index = self.grid.get(x, y, z);

                    if cell_index != EMPTY {
                        neighbors.extend_from_slice(&self.cells[cell_index as usize]);
                    }
                }
            }
        }
        neighbors
    }

    fn cell_sdf(
        &self,
        x: u32,
        y: u32,
        z: u32,
        density: f32,
        particles: &[Particle],
        particle_type: ParticleType,
    ) -> f32 {
        let mut accumulator = 0.0;

        let cell_index = self.grid.get(x, y, z);

        if cell_index != EMPTY {
            for &particle_index in &self.cells[cell_index as usize] {
                let particle = &particles[particle_index];
                if particle.particle_type == particle_type {
                    accumulator += particle.density;
                } else {
                    return 1.0;
                }
            }
        }

        0.2 * (1.0 / density.powi(3)) - accumulator
    }

    fn build_sdf(&self, mac_grid: &mut MacGrid3D, density: f32, particles: &[Particle]) {
        let dimensions = self.dimensions;

        for x in 0..dimensions.x {
            for y in 0..dimensions.y {
                for z in 0..dimensions.z {
                    mac_grid.sdf.set(
                        x,
                        y,
                        z,
                        self.cell_sdf(x, y, z, density, particles, ParticleType::Fluid),
                    );
                }
            }
        }
    }

    pub fn mark_cell_types(
        &self,
        particles: &[Particle],
        cell_types: &mut Grid3D<CellType>,
        density: f32,
    ) {
        let dimensions = self.dimensions;
        for x in 0..dimensions.x {
            for y in 0..dimensions.y {
                for z in 0..dimensions.z {
                    let cell_index = self.grid.get(x, y, z);

                    let is_solid = cell_index != EMPTY
                        && self.cells[cell_index as usize]
                            .iter()
                            .any(|&particle_index| {
                                particles[particle_index].particle_type == ParticleType::Solid
                            });

                    let cell_type = if is_solid {
                        CellType::Solid
                    } else if self.cell_sdf(x, y, z, density, particles, ParticleType::Fluid) < 0.0
                    {
                        CellType::Fluid
                    } else {
                        CellType::Air
                    };

                    cell_types.set(x, y, z, cell_type);
                }
            }
        }
    }

    pub fn sort(&mut self, particles: &[Particle]) {
        self.grid.clear();
        self.cells.clear();

        let dimensions = self.dimensions.as_vec3();
        let max_dimension = dimensions.max_element();

        for (particle_index, particle) in particles.iter().enumerate() {
            let position = particle.position;

            let x = (max_dimension * position.x).clamp(0.0, dimensions.x - 1.0) as u32;
            let y = (max_dimension * position.y).clamp(0.0, dimensions.y - 1.0) as u32;
            let z = (max_dimension * position.z).clamp(0.0, dimensions.z - 1.0) as u32;

            let cell_index = self.grid.get(x, y, z);
            if cell_index == EMPTY {
                let new_cell_index = self.cells.len() as u32;
                self.grid.set(x, y, z, new_cell_index);
                self.cells.push(vec![particle_index]);
            } else {
                self.cells[cell_index as usize].push(particle_index);
            }
        }
    }
}
