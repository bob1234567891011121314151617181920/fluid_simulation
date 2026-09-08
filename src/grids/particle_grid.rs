use glam::UVec3;

use super::grid::{CellType, Grid3D};
use super::mac_grid::{Particle, ParticleType};

const EMPTY: u32 = u32::MAX;
pub struct ParticleGrid {
    pub dimensions: UVec3,
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

    pub fn for_get_cell_neighbors<F>(&self, index: UVec3, number_of_neighbors: UVec3, mut visit: F)
    where
        F: FnMut(usize),
    {
        let minimum = index.saturating_sub(number_of_neighbors);
        let end = (index + number_of_neighbors).min(self.dimensions - UVec3::ONE);

        for x in minimum.x..=end.x {
            for y in minimum.y..=end.y {
                for z in minimum.z..=end.z {
                    let cell_index = self.grid.get(x, y, z);
                    if cell_index == EMPTY {
                        continue;
                    }
                    for &particle_index in &self.cells[cell_index as usize] {
                        visit(particle_index);
                    }
                }
            }
        }
    }

    pub fn for_each_wall_neighbor<F>(&self, index: UVec3, extent: UVec3, mut visit: F)
    where
        F: FnMut(usize),
    {
        let minimum = index.saturating_sub(extent);

        let end = (index + extent).min(self.dimensions);

        for x in minimum.x..end.x {
            for y in minimum.y..end.y {
                for z in minimum.z..end.z {
                    let cell_index = self.grid.get(x, y, z);

                    if cell_index == EMPTY {
                        continue;
                    }

                    for &particle_index in &self.cells[cell_index as usize] {
                        visit(particle_index);
                    }
                }
            }
        }
    }

    pub fn mark_cell_types(&self, particles: &[Particle], cell_types: &mut Grid3D<CellType>) {
        let dimensions = self.dimensions;

        for x in 0..dimensions.x {
            for y in 0..dimensions.y {
                for z in 0..dimensions.z {
                    let cell_index = self.grid.get(x, y, z);

                    let cell_type = if cell_index == EMPTY {
                        CellType::Air
                    } else {
                        let particle_indices = &self.cells[cell_index as usize];

                        let contains_solid = particle_indices
                            .iter()
                            .any(|&index| particles[index].particle_type == ParticleType::Solid);

                        let contains_fluid = particle_indices
                            .iter()
                            .any(|&index| particles[index].particle_type == ParticleType::Fluid);

                        if contains_solid {
                            CellType::Solid
                        } else if contains_fluid {
                            CellType::Fluid
                        } else {
                            CellType::Air
                        }
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
