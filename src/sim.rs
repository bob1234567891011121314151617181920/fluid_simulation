use crate::grids::{CellType, Grid3D, MacGrid3D, Particle, ParticleGrid, ParticleType};

use glam::{IVec3, UVec3, Vec3, u32, vec3};

const GRAVITY: f32 = -9.81;

struct FlipSimulation {
    dimensions: UVec3,

    particles: Vec<Particle>,
    particle_grid: ParticleGrid,

    mac_grid: MacGrid3D,
    previous_mac_grid: MacGrid3D,

    density: f32,
    max_density: f32,

    step_size: f32,
    pic_flip_ratio: f32,
    density_threshold: f32,
}

impl FlipSimulation {
    pub fn new(dimensions: UVec3, density: f32, step_size: f32) -> Self {
        let particle_grid = ParticleGrid::new(dimensions);
        let mac_grid = MacGrid3D::new(dimensions);
        let previous_mac_grid = MacGrid3D::new(dimensions);

        let mut simulation = Self {
            dimensions,

            particles: Vec::new(),
            particle_grid,

            mac_grid,
            previous_mac_grid,

            density,
            max_density: 0.0,

            step_size,
            pic_flip_ratio: 0.95,
            density_threshold: 0.04,
        };

        simulation.init();
        simulation
    }

    fn compute_density(&mut self) {
        let max_dimension = self.dimensions.max_element() as f32;
        let kernel_radius = 4.0 * self.density / max_dimension;
        let mut calculated_densities = vec![0.0; self.particles.len()];

        for particle_index in 0..self.particles.len() {
            let particle = &self.particles[particle_index];

            if particle.particle_type == ParticleType::Solid {
                calculated_densities[particle_index] = 1.0;
                continue;
            }

            let scaled_position = (particle.position * max_dimension).floor();
            let cell = UVec3::new(
                (scaled_position.x as u32).min(self.dimensions.x - 1),
                (scaled_position.y as u32).min(self.dimensions.y - 1),
                (scaled_position.z as u32).min(self.dimensions.z - 1),
            );

            let neighbors = self.particle_grid.get_cell_neighbors(cell, UVec3::ONE);
            let mut weight_sum = 0.0;
            for &neighbor_index in neighbors.iter() {
                let neighbor = &self.particles[neighbor_index];

                let distance_squared = neighbor.position.distance_squared(particle.position);
                // TODO use a better approximation
                let weight = neighbor.mass
                    * (1.0 - distance_squared / (kernel_radius * kernel_radius)).max(0.0);

                weight_sum += weight;
            }
            calculated_densities[particle_index] = weight_sum / self.max_density;
        }

        for (particle, density) in self.particles.iter_mut().zip(calculated_densities) {
            particle.density = density;
        }
    }

    fn init(&mut self) {
        let max_dimension = self.dimensions.max_element() as f32;

        let cell_width = 1.0 / max_dimension;

        let particle_spacing = self.density * cell_width;
        self.particles.clear();

        for x in 0..10 {
            for y in 0..10 {
                for z in 0..10 {
                    let grid_position = Vec3::new(x as f32, y as f32, z as f32);

                    let position = (grid_position + Vec3::splat(0.5)) * particle_spacing;
                    self.particles
                        .push(Particle::new_fluid(position, Vec3::ZERO));
                }
            }
        }

        self.particle_grid.sort(&self.particles);

        self.max_density = 1.0;
        self.compute_density();

        self.max_density = self
            .particles
            .iter()
            .filter(|particle| particle.particle_type == ParticleType::Fluid)
            .map(|particle| particle.density)
            .fold(0.0_f32, f32::max);

        assert!(
            self.max_density > 0.0,
            "Density calibration produced zero density"
        );

        self.particles.clear();

        let domain_size = self.dimensions.as_vec3() / max_dimension;
        let fluid_min = domain_size * Vec3::new(0.05, 0.05, 0.05);
        let fluid_max = domain_size * Vec3::new(0.45, 0.85, 0.85);

        let particle_counts = ((fluid_max - fluid_min) / particle_spacing)
            .floor()
            .as_uvec3();

        for x in 0..particle_counts.x {
            for y in 0..particle_counts.y {
                for z in 0..particle_counts.z {
                    let offset = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5)
                        * particle_spacing;

                    let position = fluid_min + offset;
                    self.particles
                        .push(Particle::new_fluid(position, Vec3::ZERO));
                }
            }
        }

        self.particle_grid.sort(&self.particles);
        self.particle_grid.mark_cell_types(
            &self.particles,
            &mut self.mac_grid.cell_type,
            self.density,
        );
    }

    fn store_temporary_particle_state(&mut self) {
        for particle in self.particles.iter_mut() {
            particle.temporary_velocity = particle.velocity;
            particle.temporary_position = particle.position;
        }
    }

    fn apply_gravity(&mut self) {
        for particle in self.particles.iter_mut() {
            if particle.particle_type == ParticleType::Solid {
                continue;
            }
            particle.velocity.y += GRAVITY * self.step_size;
        }
    }

    fn transfer_particle_velocities_to_mac_grid(&mut self) {
        let radious = 1.4;

        let dimensions = self.dimensions;
        let max_dimension = dimensions.max_element() as f32;
        for x in 0..=dimensions.x {
            for y in 0..dimensions.y {
                for z in 0..dimensions.z {
                    let face_position = Vec3::new(x as f32, y as f32 + 0.5, z as f32 + 0.5);

                    let neighbors = self
                        .particle_grid
                        .get_wall_neighbors(UVec3::new(x, y, z), UVec3::new(1, 2, 2));

                    let mut weight_sum = 0.0;
                    let mut weighted_velocity_sum = 0.0;

                    for neighbor_index in neighbors {
                        let particle = &self.particles[neighbor_index];
                        if particle.particle_type != ParticleType::Fluid {
                            continue;
                        }

                        let grid_position = (particle.position * max_dimension)
                            .clamp(Vec3::ZERO, Vec3::splat(max_dimension));

                        let distance_squared = face_position.distance_squared(grid_position);
                        let weight = particle.mass
                            * (radious * radious / distance_squared.max(1.0e-5) - 1.0).max(0.0);

                        weighted_velocity_sum += weight * particle.velocity.x;
                        weight_sum += weight;
                    }

                    let face_velocity = if weight_sum > 0.0 {
                        weighted_velocity_sum / weight_sum
                    } else {
                        0.0
                    };

                    self.mac_grid.u_x.set(x, y, z, face_velocity);
                }
            }
        }

        for x in 0..dimensions.x {
            for y in 0..=dimensions.y {
                for z in 0..dimensions.z {
                    let face_position = Vec3::new(x as f32 + 0.5, y as f32, z as f32 + 0.5);

                    let neighbors = self
                        .particle_grid
                        .get_wall_neighbors(UVec3::new(x, y, z), UVec3::new(2, 1, 2));

                    let mut weight_sum = 0.0;
                    let mut weighted_velocity_sum = 0.0;

                    for neighbor_index in neighbors {
                        let particle = &self.particles[neighbor_index];
                        if particle.particle_type != ParticleType::Fluid {
                            continue;
                        }

                        let grid_position = (particle.position * max_dimension)
                            .clamp(Vec3::ZERO, Vec3::splat(max_dimension));

                        let distance_squared = face_position.distance_squared(grid_position);
                        let weight = particle.mass
                            * (radious * radious / distance_squared.max(1.0e-5) - 1.0).max(0.0);

                        weighted_velocity_sum += weight * particle.velocity.x;
                        weight_sum += weight;
                    }

                    let face_velocity = if weight_sum > 0.0 {
                        weighted_velocity_sum / weight_sum
                    } else {
                        0.0
                    };

                    self.mac_grid.u_y.set(x, y, z, face_velocity);
                }
            }
        }

        for x in 0..dimensions.x {
            for y in 0..dimensions.y {
                for z in 0..=dimensions.z {
                    let face_position = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32);

                    let neighbors = self
                        .particle_grid
                        .get_wall_neighbors(UVec3::new(x, y, z), UVec3::new(2, 2, 1));

                    let mut weight_sum = 0.0;
                    let mut weighted_velocity_sum = 0.0;

                    for neighbor_index in neighbors {
                        let particle = &self.particles[neighbor_index];
                        if particle.particle_type != ParticleType::Fluid {
                            continue;
                        }

                        let grid_position = (particle.position * max_dimension)
                            .clamp(Vec3::ZERO, Vec3::splat(max_dimension));

                        let distance_squared = face_position.distance_squared(grid_position);
                        let weight = particle.mass
                            * (radious * radious / distance_squared.max(1.0e-5) - 1.0).max(0.0);

                        weighted_velocity_sum += weight * particle.velocity.x;
                        weight_sum += weight;
                    }

                    let face_velocity = if weight_sum > 0.0 {
                        weighted_velocity_sum / weight_sum
                    } else {
                        0.0
                    };

                    self.mac_grid.u_z.set(x, y, z, face_velocity);
                }
            }
        }
    }

    fn compute_divergence(&mut self) {
        let dimensions = self.mac_grid.dimensions;
        let max_dimensions = dimensions.max_element() as f32;
        let h = 1.0 / max_dimensions;
        for x in 0..dimensions.x {
            for y in 0..dimensions.y {
                for z in 0..dimensions.z {
                    let divergence = (self.mac_grid.u_x.get(x + 1, y, z)
                        - self.mac_grid.u_x.get(x, y, z))
                        + (self.mac_grid.u_y.get(x, y + 1, z) - self.mac_grid.u_y.get(x, y, z))
                        + (self.mac_grid.u_z.get(x, y, z + 1) - self.mac_grid.u_z.get(x, y, z)) / h;

                    self.mac_grid.divergence.set(x, y, z, divergence);
                }
            }
        }
    }

    fn enforce_boundary_velocities(&mut self) {
        let dimensions = self.mac_grid.dimensions;
        for x in 0..=dimensions.x {
            for y in 0..dimensions.y {
                for z in 0..dimensions.z {
                    if x == 0 || x == dimensions.x {
                        self.mac_grid.u_x.set(x, y, z, 0.0);
                        continue;
                    }
                    if self.mac_grid.check_wall(x, y, z) * self.mac_grid.check_wall(x - 1, y, z)
                        > 0.0
                    {
                        self.mac_grid.u_x.set(x, y, z, 0.0);
                    }
                }
            }
        }

        for x in 0..dimensions.x {
            for y in 0..=dimensions.y {
                for z in 0..dimensions.z {
                    if y == 0 || y == dimensions.y {
                        self.mac_grid.u_y.set(x, y, z, 0.0);
                        continue;
                    }
                    if self.mac_grid.check_wall(x, y, z) * self.mac_grid.check_wall(x, y - 1, z)
                        > 0.0
                    {
                        self.mac_grid.u_y.set(x, y, z, 0.0);
                    }
                }
            }
        }

        for x in 0..dimensions.x {
            for y in 0..dimensions.y {
                for z in 0..=dimensions.z {
                    if z == 0 || z == dimensions.z {
                        self.mac_grid.u_z.set(x, y, z, 0.0);
                        continue;
                    }
                    if self.mac_grid.check_wall(x, y, z) * self.mac_grid.check_wall(x, y, z - 1)
                        > 0.0
                    {
                        self.mac_grid.u_z.set(x, y, z, 0.0);
                    }
                }
            }
        }
    }

    fn extrapolate_velocities(&mut self) {
        let dimensions = self.mac_grid.dimensions;

        let face_dimensions = [
            UVec3::new(dimensions.x + 1, dimensions.y, dimensions.z),
            UVec3::new(dimensions.x, dimensions.y + 1, dimensions.z),
            UVec3::new(dimensions.x, dimensions.y, dimensions.z + 1),
        ];

        let mut fluid_mark = [
            Grid3D::new(
                face_dimensions[0].x,
                face_dimensions[0].y,
                face_dimensions[0].z,
                false,
            ),
            Grid3D::new(
                face_dimensions[1].x,
                face_dimensions[1].y,
                face_dimensions[1].z,
                false,
            ),
            Grid3D::new(
                face_dimensions[2].x,
                face_dimensions[2].y,
                face_dimensions[2].z,
                false,
            ),
        ];

        let mut wall_mark = [
            Grid3D::new(
                face_dimensions[0].x,
                face_dimensions[0].y,
                face_dimensions[0].z,
                false,
            ),
            Grid3D::new(
                face_dimensions[1].x,
                face_dimensions[1].y,
                face_dimensions[1].z,
                false,
            ),
            Grid3D::new(
                face_dimensions[2].x,
                face_dimensions[2].y,
                face_dimensions[2].z,
                false,
            ),
        ];

        let mac_grid = &mut self.mac_grid;
        for x in 0..=dimensions.x {
            for y in 0..dimensions.y {
                for z in 0..dimensions.z {
                    fluid_mark[0].set(
                        x,
                        y,
                        z,
                        x > 0 && mac_grid.cell_type.get(x - 1, y, z) == CellType::Fluid
                            || x < dimensions.x
                                && mac_grid.cell_type.get(x, y, z) == CellType::Fluid,
                    );
                    let touches_solid = (x > 0
                        && mac_grid.cell_type.get(x - 1, y, z) == CellType::Solid)
                        || (x < dimensions.x && mac_grid.cell_type.get(x, y, z) == CellType::Solid);

                    let boundary = x == 0 || x == dimensions.x;

                    wall_mark[0].set(x, y, z, boundary || touches_solid);
                }
            }
        }

        for x in 0..dimensions.x {
            for y in 0..=dimensions.y {
                for z in 0..dimensions.z {
                    fluid_mark[1].set(
                        x,
                        y,
                        z,
                        y > 0 && mac_grid.cell_type.get(x, y - 1, z) == CellType::Fluid
                            || y < dimensions.y
                                && mac_grid.cell_type.get(x, y, z) == CellType::Fluid,
                    );
                    let touches_solid = (y > 0
                        && mac_grid.cell_type.get(x, y - 1, z) == CellType::Solid)
                        || (y < dimensions.y && mac_grid.cell_type.get(x, y, z) == CellType::Solid);

                    let boundary = y == 0 || y == dimensions.y;

                    wall_mark[1].set(x, y, z, boundary || touches_solid);
                }
            }
        }

        for x in 0..dimensions.x {
            for y in 0..dimensions.y {
                for z in 0..=dimensions.z {
                    fluid_mark[2].set(
                        x,
                        y,
                        z,
                        z > 0 && mac_grid.cell_type.get(x, y, z - 1) == CellType::Fluid
                            || z < dimensions.z
                                && mac_grid.cell_type.get(x, y, z) == CellType::Fluid,
                    );
                    let touches_solid = (z > 0
                        && mac_grid.cell_type.get(x, y, z - 1) == CellType::Solid)
                        || (z < dimensions.z && mac_grid.cell_type.get(x, y, z) == CellType::Solid);

                    let boundary = z == 0 || z == dimensions.z;

                    wall_mark[2].set(x, y, z, boundary || touches_solid);
                }
            }
        }

        for axis in 0..3 {
            let axis_dimensions = face_dimensions[axis];

            for x in 0..axis_dimensions.x {
                for y in 0..axis_dimensions.y {
                    for z in 0..axis_dimensions.z {
                        if fluid_mark[axis].get(x, y, z) || wall_mark[axis].get(x, y, z) {
                            continue;
                        }

                        let position = IVec3::new(x as i32, y as i32, z as i32);
                        let neighbor_positions = [
                            position + IVec3::new(-1, 0, 0),
                            position + IVec3::new(1, 0, 0),
                            position + IVec3::new(0, -1, 0),
                            position + IVec3::new(0, 1, 0),
                            position + IVec3::new(0, 0, -1),
                            position + IVec3::new(0, 0, 1),
                        ];

                        let mut velocity_sum = 0.0;
                        let mut weight_sum = 0;

                        for neighbor in neighbor_positions {
                            if neighbor.x < 0 || neighbor.y < 0 || neighbor.z < 0 {
                                continue;
                            }
                            let neighbor = neighbor.as_uvec3();

                            if neighbor.x >= axis_dimensions.x
                                || neighbor.y >= axis_dimensions.y
                                || neighbor.z >= axis_dimensions.z
                            {
                                continue;
                            }

                            if !fluid_mark[axis].get(neighbor.x, neighbor.y, neighbor.z) {
                                continue;
                            }

                            let neighbor_velocity = match axis {
                                0 => mac_grid.u_x.get(neighbor.x, neighbor.y, neighbor.z),
                                1 => mac_grid.u_y.get(neighbor.x, neighbor.y, neighbor.z),
                                2 => mac_grid.u_z.get(neighbor.x, neighbor.y, neighbor.z),
                                _ => unreachable!(),
                            };

                            velocity_sum += neighbor_velocity;
                            weight_sum += 1;
                        }

                        if weight_sum == 0 {
                            continue;
                        }

                        let extrapolated_velocity = velocity_sum / weight_sum as f32;

                        match axis {
                            0 => mac_grid.u_x.set(x, y, z, extrapolated_velocity),
                            1 => mac_grid.u_y.set(x, y, z, extrapolated_velocity),
                            2 => mac_grid.u_z.set(x, y, z, extrapolated_velocity),
                            _ => unreachable!(),
                        }
                    }
                }
            }
        }
    }

    fn subtract_grid_component(
        current: &Grid3D<f32>,
        previous: &mut Grid3D<f32>,
        dimensions: UVec3,
    ) {
        for x in 0..dimensions.x {
            for y in 0..dimensions.y {
                for z in 0..dimensions.z {
                    let difference = current.get(x, y, z) - previous.get(x, y, z);

                    previous.set(x, y, z, difference);
                }
            }
        }
    }

    fn subtract_previous_grid(&mut self) {
        let dimensions = self.mac_grid.dimensions;

        Self::subtract_grid_component(
            &self.mac_grid.u_x,
            &mut self.previous_mac_grid.u_x,
            dimensions,
        );
        Self::subtract_grid_component(
            &self.mac_grid.u_y,
            &mut self.previous_mac_grid.u_y,
            dimensions,
        );
        Self::subtract_grid_component(
            &self.mac_grid.u_z,
            &mut self.previous_mac_grid.u_z,
            dimensions,
        );
    }

    fn interpolate(grid: &Grid3D<f32>, position: Vec3, dimensions: UVec3) -> f32 {
        let position = position.clamp(Vec3::ZERO, dimensions.as_vec3() - Vec3::ONE);

        let x = position.x.clamp(0.0, dimensions.x as f32 - 1.0);
        let y = position.y.clamp(0.0, dimensions.y as f32 - 1.0);
        let z = position.z.clamp(0.0, dimensions.z as f32 - 1.0);

        let i = (x as u32).min((dimensions.x).saturating_sub(2));
        let j = (y as u32).min((dimensions.y).saturating_sub(2));
        let k = (z as u32).min((dimensions.z).saturating_sub(2));

        let tx = x - i as f32;
        let ty = y - j as f32;
        let tz = z - k as f32;

        let lerp = |a: f32, b: f32, t: f32| a + t * (b - a);

        lerp(
            lerp(
                lerp(grid.get(i, j, k), grid.get(i + 1, j, k), tx),
                lerp(grid.get(i, j + 1, k), grid.get(i + 1, j + 1, k), tx),
                ty,
            ),
            lerp(
                lerp(grid.get(i, j, k + 1), grid.get(i + 1, j, k + 1), tx),
                lerp(grid.get(i, j + 1, k + 1), grid.get(i + 1, j + 1, k + 1), tx),
                ty,
            ),
            tz,
        )
    }

    fn interpolate_velocities(mac_grid: &MacGrid3D, position: Vec3) -> Vec3 {
        let dimensions = mac_grid.dimensions;
        let scale = dimensions.max_element() as f32;

        let u_x_dimension = UVec3::new(dimensions.x + 1, dimensions.y, dimensions.z);
        let u_y_dimension = UVec3::new(dimensions.x, dimensions.y + 1, dimensions.z);
        let u_z_dimension = UVec3::new(dimensions.x, dimensions.y, dimensions.z + 1);

        vec3(
            Self::interpolate(
                &mac_grid.u_x,
                vec3(
                    scale * position.x,
                    scale * position.y - 0.5,
                    scale * position.z - 0.5,
                ),
                u_x_dimension,
            ),
            Self::interpolate(
                &mac_grid.u_y,
                vec3(
                    scale * position.x - 0.5,
                    scale * position.y,
                    scale * position.z - 0.5,
                ),
                u_y_dimension,
            ),
            Self::interpolate(
                &mac_grid.u_z,
                vec3(
                    scale * position.x - 0.5,
                    scale * position.y - 0.5,
                    scale * position.z,
                ),
                u_z_dimension,
            ),
        )
    }

    fn transfer_mac_grid_to_particles(particles: &mut [Particle], mac_grid: &MacGrid3D) {
        for particle in particles {
            particle.velocity = Self::interpolate_velocities(mac_grid, particle.position);
        }
    }

    fn solve_pic_flip(&mut self) {
        for particle in self.particles.iter_mut() {
            particle.previous_velocity = particle.velocity;
        }

        Self::transfer_mac_grid_to_particles(&mut self.particles, &self.previous_mac_grid);

        for particle in self.particles.iter_mut() {
            particle.previous_velocity += particle.velocity;
        }

        Self::transfer_mac_grid_to_particles(&mut self.particles, &self.mac_grid);

        for particle in self.particles.iter_mut() {
            let pic_velocity = particle.velocity;
            let flip_velocity = particle.previous_velocity;

            particle.velocity =
                (1.0 - self.pic_flip_ratio) * pic_velocity + self.pic_flip_ratio * flip_velocity;
        }
    }

    fn advance_particles(&mut self) {
        for particle in self.particles.iter_mut() {
            if particle.particle_type == ParticleType::Solid {
                continue;
            }
            let velocity = Self::interpolate_velocities(&self.mac_grid, particle.position);
            particle.position += velocity * self.step_size;
        }
    }

    fn constrain_particles_to_domain(&mut self) {
        let max_dimension = self.dimensions.max_element() as f32;
        let domain_size = self.dimensions.as_vec3() / max_dimension;

        // Small distance from the exact boundary.
        let margin = 0.01 / max_dimension;

        let minimum = Vec3::splat(margin);
        let maximum = domain_size - Vec3::splat(margin);

        for particle in &mut self.particles {
            if particle.position.x < minimum.x {
                particle.position.x = minimum.x;
                particle.velocity.x = particle.velocity.x.max(0.0);
            } else if particle.position.x > maximum.x {
                particle.position.x = maximum.x;
                particle.velocity.x = particle.velocity.x.min(0.0);
            }

            if particle.position.y < minimum.y {
                particle.position.y = minimum.y;
                particle.velocity.y = particle.velocity.y.max(0.0);
            } else if particle.position.y > maximum.y {
                particle.position.y = maximum.y;
                particle.velocity.y = particle.velocity.y.min(0.0);
            }

            if particle.position.z < minimum.z {
                particle.position.z = minimum.z;
                particle.velocity.z = particle.velocity.z.max(0.0);
            } else if particle.position.z > maximum.z {
                particle.position.z = maximum.z;
                particle.velocity.z = particle.velocity.z.min(0.0);
            }
        }
    }

    pub fn step(&mut self) {
        self.store_temporary_particle_state();
        self.particle_grid.sort(&self.particles);
        self.compute_density();
        self.apply_gravity();
        self.transfer_particle_velocities_to_mac_grid();
        self.particle_grid.mark_cell_types(
            &self.particles,
            &mut self.mac_grid.cell_type,
            self.density,
        );
        self.previous_mac_grid = self.mac_grid.clone();
        self.compute_divergence();
        self.enforce_boundary_velocities();
        self.extrapolate_velocities();
        self.subtract_previous_grid();
        self.solve_pic_flip();
        self.advance_particles();
        self.constrain_particles_to_domain();
        self.store_temporary_particle_state();
    }
}
