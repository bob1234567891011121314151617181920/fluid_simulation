use crate::{
    grids::{CellType, Grid3D, MacGrid3D, Particle, ParticleGrid, ParticleType},
    simulation::pcg_solver::pcg,
};

use glam::{IVec3, UVec3, Vec3, camera::rh, u32, vec3};

const GRAVITY: f32 = -9.81;

pub struct FlipSimulation {
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

        let search_radius = (4.0 * self.density).ceil() as u32;
        let neighbor_extent = UVec3::splat(search_radius);

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

            let mut weight_sum = 0.0;

            self.particle_grid
                .for_get_cell_neighbors(cell, neighbor_extent, |neighbor_index| {
                    let neighbor = &self.particles[neighbor_index];

                    let distance_squared = neighbor.position.distance_squared(particle.position);
                    // TODO use a better approximation
                    let weight = neighbor.mass
                        * (1.0 - distance_squared / (kernel_radius * kernel_radius)).max(0.0);

                    weight_sum += weight;
                });
            calculated_densities[particle_index] = weight_sum / self.max_density;
        }

        for (particle, density) in self.particles.iter_mut().zip(calculated_densities) {
            particle.density = density;
        }
    }

    pub fn init(&mut self) {
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
            .fold(0.0, f32::max);

        assert!(
            self.max_density > 0.0,
            "Density calibration produced zero density"
        );

        self.particles.clear();

        let domain_size = self.dimensions.as_vec3() / max_dimension;
        let fluid_min = domain_size * Vec3::new(0.05, 0.05, 0.05);
        let fluid_max = domain_size * Vec3::new(0.5, 0.15, 0.15);

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
        self.compute_density();
        self.particle_grid.mark_cell_types(
            &self.particles,
            &mut self.mac_grid.cell_type,
            self.density,
        );
        println!(
            "Initialized simulation with {} particles",
            self.particles.len()
        );
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
        let dimensions = self.dimensions;
        let max_dimension = dimensions.max_element() as f32;

        Self::transfer_velocity_component_to_grid(
            &self.particles,
            &self.particle_grid,
            &mut self.mac_grid.u_x,
            Vec3::new(0.0, 0.5, 0.5),
            UVec3::new(1, 2, 2),
            0,
            max_dimension,
        );
        Self::transfer_velocity_component_to_grid(
            &self.particles,
            &self.particle_grid,
            &mut self.mac_grid.u_y,
            Vec3::new(0.5, 0.0, 0.5),
            UVec3::new(2, 1, 2),
            1,
            max_dimension,
        );
        Self::transfer_velocity_component_to_grid(
            &self.particles,
            &self.particle_grid,
            &mut self.mac_grid.u_z,
            Vec3::new(0.5, 0.5, 0.0),
            UVec3::new(2, 2, 1),
            2,
            max_dimension,
        );
    }

    fn transfer_velocity_component_to_grid(
        particles: &[Particle],
        particle_grid: &ParticleGrid,
        velocity_grid: &mut Grid3D<f32>,
        face_offset: Vec3,
        neighbor_extent: UVec3,
        component: usize,
        max_dimension: f32,
    ) {
        const RADIUS: f32 = 1.4;

        let dimensions = velocity_grid.dimensions();
        for x in 0..dimensions.x {
            for y in 0..dimensions.y {
                for z in 0..dimensions.z {
                    let cell = UVec3::new(x, y, z);
                    let face_position = cell.as_vec3() + face_offset;
                    let mut weight_sum = 0.0;
                    let mut weighted_velocity_sum = 0.0;

                    particle_grid.for_each_wall_neighbor(cell, neighbor_extent, |neighbor_index| {
                        let particle = &particles[neighbor_index];
                        if particle.particle_type != ParticleType::Fluid {
                            return;
                        }

                        let grid_position = (particle.position * max_dimension)
                            .clamp(Vec3::ZERO, Vec3::splat(max_dimension));
                        let distance_squared = face_position.distance_squared(grid_position);
                        let weight = particle.mass
                            * (RADIUS * RADIUS / distance_squared.max(1.0e-5) - 1.0).max(0.0);

                        weighted_velocity_sum += weight * particle.velocity[component];
                        weight_sum += weight;
                    });

                    let face_velocity = if weight_sum > 0.0 {
                        weighted_velocity_sum / weight_sum
                    } else {
                        0.0
                    };
                    velocity_grid.set(x, y, z, face_velocity);
                }
            }
        }
    }

    fn pressure_diagonal(
        position: UVec3,
        dimensions: UVec3,
        cell_types: &Grid3D<CellType>,
        inverse_h_squared: f32,
    ) -> f32 {
        let position = position.as_ivec3();
        let neighbors = [
            position - IVec3::X,
            position + IVec3::X,
            position - IVec3::Y,
            position + IVec3::Y,
            position - IVec3::Z,
            position + IVec3::Z,
        ];
        let dimensions = dimensions.as_ivec3();

        let mut diagonal = 0.0;

        for neighbor in neighbors {
            if neighbor.cmplt(IVec3::ZERO).any() || neighbor.cmpge(dimensions).any() {
                continue;
            }

            let neighbor = neighbor.as_uvec3();
            match cell_types.get(neighbor.x, neighbor.y, neighbor.z) {
                CellType::Solid => {}
                CellType::Fluid | CellType::Air => {
                    diagonal += inverse_h_squared;
                }
            }
        }
        diagonal
    }

    fn solve_pressure(&mut self) {
        const MAX_ITERATIONS: usize = 50;
        const RELATIVE_TOLERANCE: f32 = 1.0e-5;
        const EPSILON: f32 = 1.0e-12;

        let mac_grid = &mut self.mac_grid;
        let dimensions = mac_grid.dimensions;
        let max_dimensions = dimensions.max_element() as f32;
        let h = 1.0 / max_dimensions;
        let inverse_h_squared = 1.0 / (h * h);

        let cell_types = &mac_grid.cell_type;
        let divergence = &mac_grid.divergence;
        let cell_count = (dimensions.x * dimensions.y * dimensions.z) as usize;
        let index = |x: u32, y: u32, z: u32| ((x * dimensions.y + y) * dimensions.z + z) as usize;
        let mut solution = vec![0.0; cell_count];
        let mut rhs = vec![0.0; cell_count];
        let mut inverse_diagonal = vec![0.0; cell_count];

        for x in 0..dimensions.x {
            for y in 0..dimensions.y {
                for z in 0..dimensions.z {
                    if cell_types.get(x, y, z) != CellType::Fluid {
                        continue;
                    }
                    let cell_index = index(x, y, z);
                    rhs[cell_index] = -divergence.get(x, y, z);
                    let diagonal = Self::pressure_diagonal(
                        UVec3::new(x, y, z),
                        dimensions,
                        cell_types,
                        inverse_h_squared,
                    );

                    if diagonal > EPSILON {
                        inverse_diagonal[cell_index] = 1.0 / diagonal;
                    }
                }
            }
        }

        let apply_pressure_matrix = |input: &[f32], output: &mut [f32]| {
            output.fill(0.0);
            for x in 0..dimensions.x {
                for y in 0..dimensions.y {
                    for z in 0..dimensions.z {
                        if cell_types.get(x, y, z) != CellType::Fluid {
                            continue;
                        }

                        let position = UVec3::new(x, y, z);
                        let cell_index = index(x, y, z);
                        let diagonal = FlipSimulation::pressure_diagonal(
                            position,
                            dimensions,
                            cell_types,
                            inverse_h_squared,
                        );
                        let mut result = diagonal * input[cell_index];
                        let neighbors = [
                            position.saturating_sub(UVec3::X),
                            position + UVec3::X,
                            position.saturating_sub(UVec3::Y),
                            position + UVec3::Y,
                            position.saturating_sub(UVec3::Z),
                            position + UVec3::Z,
                        ];

                        for neighbor in neighbors {
                            if neighbor.x >= dimensions.x
                                || neighbor.y >= dimensions.y
                                || neighbor.z >= dimensions.z
                            {
                                continue;
                            }

                            if cell_types.get(neighbor.x, neighbor.y, neighbor.z) == CellType::Fluid
                            {
                                result -= inverse_h_squared
                                    * input[index(neighbor.x, neighbor.y, neighbor.z)];
                            }
                        }
                        output[cell_index] = result;
                    }
                }
            }
        };
        let apply_preconditioner = |input: &[f32], output: &mut [f32]| {
            for ((output, input), inverse_diagonal) in
                output.iter_mut().zip(input).zip(&inverse_diagonal)
            {
                *output = input * inverse_diagonal;
            }
        };

        let converged = pcg(
            apply_pressure_matrix,
            apply_preconditioner,
            &mut solution,
            &rhs,
            MAX_ITERATIONS,
            RELATIVE_TOLERANCE,
        );

        if !converged {
            log::warn!("Warning: Pressure solve did not converge");
        }

        mac_grid.pressure.clear();
        for x in 0..dimensions.x {
            for y in 0..dimensions.y {
                for z in 0..dimensions.z {
                    if cell_types.get(x, y, z) == CellType::Fluid {
                        mac_grid.pressure.set(x, y, z, solution[index(x, y, z)]);
                    }
                }
            }
        }
    }
    fn project(&mut self) {
        let dimensions = self.mac_grid.dimensions;
        let mac_grid = &mut self.mac_grid;
        let max_dimensions = dimensions.max_element() as f32;
        let h = 1.0 / max_dimensions;
        for x in 0..dimensions.x {
            for y in 0..dimensions.y {
                for z in 0..dimensions.z {
                    let divergence = ((mac_grid.u_x.get(x + 1, y, z) - mac_grid.u_x.get(x, y, z))
                        + (mac_grid.u_y.get(x, y + 1, z) - mac_grid.u_y.get(x, y, z))
                        + (mac_grid.u_z.get(x, y, z + 1) - mac_grid.u_z.get(x, y, z)))
                        / h;

                    mac_grid.divergence.set(x, y, z, divergence);
                }
            }
        }

        self.solve_pressure();
        self.subtract_pressure_gradient();
    }

    fn subtract_pressure_gradient(&mut self) {
        let mac_grid = &mut self.mac_grid;
        let dimensions = mac_grid.dimensions;
        let max_dimensions = dimensions.max_element() as f32;
        let h = 1.0 / max_dimensions;

        Self::subtract_pressure_gradient_component(
            &mac_grid.pressure,
            &mut mac_grid.u_x,
            dimensions,
            0,
            h,
        );

        Self::subtract_pressure_gradient_component(
            &mac_grid.pressure,
            &mut mac_grid.u_y,
            dimensions,
            1,
            h,
        );

        Self::subtract_pressure_gradient_component(
            &mac_grid.pressure,
            &mut mac_grid.u_z,
            dimensions,
            2,
            h,
        );
    }

    fn subtract_pressure_gradient_component(
        pressure: &Grid3D<f32>,
        velocity: &mut Grid3D<f32>,
        dimensions: UVec3,
        component: usize,
        h: f32,
    ) {
        let mut start = UVec3::ZERO;
        start[component] = 1;

        for x in start.x..dimensions.x {
            for y in start.y..dimensions.y {
                for z in start.z..dimensions.z {
                    let forward = UVec3::new(x, y, z);
                    let mut backward = forward;
                    backward[component] -= 1;

                    let forward_pressure = pressure.get(forward.x, forward.y, forward.z);
                    let backward_pressure = pressure.get(backward.x, backward.y, backward.z);
                    let pressure_gradient = (forward_pressure - backward_pressure) / h;
                    let corrected_velocity =
                        velocity.get(forward.x, forward.y, forward.z) - pressure_gradient;

                    velocity.set(forward.x, forward.y, forward.z, corrected_velocity);
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
                    if self.mac_grid.check_wall(x, y, z) || self.mac_grid.check_wall(x - 1, y, z) {
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
                    if self.mac_grid.check_wall(x, y, z) || self.mac_grid.check_wall(x, y - 1, z) {
                        println!("pain");
                        self.mac_grid.u_x.set(x, y, z, 0.0);
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
                    if self.mac_grid.check_wall(x, y, z) || self.mac_grid.check_wall(x, y, z - 1) {
                        self.mac_grid.u_x.set(x, y, z, 0.0);
                    }
                }
            }
        }
    }

    fn extrapolate_velocity_component<F>(
        velocity: &mut Grid3D<f32>,
        cell_types: &Grid3D<CellType>,
        face_dimensions: UVec3,
        adjacent_cells: F,
    ) where
        F: Fn(UVec3) -> (Option<UVec3>, Option<UVec3>),
    {
        let mut fluid_mark = Grid3D::new(
            face_dimensions.x,
            face_dimensions.y,
            face_dimensions.z,
            false,
        );
        let mut wall_mark = Grid3D::new(
            face_dimensions.x,
            face_dimensions.y,
            face_dimensions.z,
            false,
        );

        for x in 0..face_dimensions.x {
            for y in 0..face_dimensions.y {
                for z in 0..face_dimensions.z {
                    let position = UVec3::new(x, y, z);
                    let (negative_cell, positive_cell) = adjacent_cells(position);
                    let negative_type =
                        negative_cell.map(|cell| cell_types.get(cell.x, cell.y, cell.z));
                    let positive_type =
                        positive_cell.map(|cell| cell_types.get(cell.x, cell.y, cell.z));

                    let touches_fluid = negative_type == Some(CellType::Fluid)
                        || positive_type == Some(CellType::Fluid);
                    let touches_solid = negative_type == Some(CellType::Solid)
                        || positive_type == Some(CellType::Solid);
                    let boundary = negative_cell.is_none() || positive_cell.is_none();

                    fluid_mark.set(x, y, z, touches_fluid);
                    wall_mark.set(x, y, z, boundary || touches_solid);
                }
            }
        }

        for x in 0..face_dimensions.x {
            for y in 0..face_dimensions.y {
                for z in 0..face_dimensions.z {
                    if fluid_mark.get(x, y, z) || wall_mark.get(x, y, z) {
                        continue;
                    }

                    let position = UVec3::new(x, y, z);
                    let neighbor_positions = [
                        position.saturating_sub(UVec3::X),
                        (position + UVec3::X).min(face_dimensions - UVec3::ONE),
                        position.saturating_sub(UVec3::Y),
                        (position + UVec3::Y).min(face_dimensions - UVec3::ONE),
                        position.saturating_sub(UVec3::Z),
                        (position + UVec3::Z).min(face_dimensions - UVec3::ONE),
                    ];

                    let mut velocity_sum = 0.0;
                    let mut weight_sum = 0;

                    for neighbor in neighbor_positions {
                        if !fluid_mark.get(neighbor.x, neighbor.y, neighbor.z) {
                            continue;
                        }

                        velocity_sum += velocity.get(neighbor.x, neighbor.y, neighbor.z);
                        weight_sum += 1;
                    }

                    if weight_sum > 0 {
                        velocity.set(x, y, z, velocity_sum / weight_sum as f32);
                    }
                }
            }
        }
    }

    fn extrapolate_velocities(&mut self) {
        let dimensions = self.mac_grid.dimensions;
        let cell_types = &self.mac_grid.cell_type;

        Self::extrapolate_velocity_component(
            &mut self.mac_grid.u_x,
            cell_types,
            dimensions + UVec3::X,
            |position| {
                (
                    (position.x > 0).then(|| position - UVec3::X),
                    (position.x < dimensions.x).then_some(position),
                )
            },
        );
        Self::extrapolate_velocity_component(
            &mut self.mac_grid.u_y,
            cell_types,
            dimensions + UVec3::Y,
            |position| {
                (
                    (position.y > 0).then(|| position - UVec3::Y),
                    (position.y < dimensions.y).then_some(position),
                )
            },
        );
        Self::extrapolate_velocity_component(
            &mut self.mac_grid.u_z,
            cell_types,
            dimensions + UVec3::Z,
            |position| {
                (
                    (position.z > 0).then(|| position - UVec3::Z),
                    (position.z < dimensions.z).then_some(position),
                )
            },
        );
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
            UVec3::new(dimensions.x + 1, dimensions.y, dimensions.z),
        );
        Self::subtract_grid_component(
            &self.mac_grid.u_y,
            &mut self.previous_mac_grid.u_y,
            UVec3::new(dimensions.x, dimensions.y + 1, dimensions.z),
        );
        Self::subtract_grid_component(
            &self.mac_grid.u_z,
            &mut self.previous_mac_grid.u_z,
            UVec3::new(dimensions.x, dimensions.y, dimensions.z + 1),
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
            particle.position += particle.velocity * self.step_size;
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
        self.particle_grid.sort(&self.particles);
        self.compute_density();
        self.apply_gravity();
        self.transfer_particle_velocities_to_mac_grid();
        self.particle_grid.mark_cell_types(
            &self.particles,
            &mut self.mac_grid.cell_type,
            self.density,
        );
        self.enforce_boundary_velocities();
        self.extrapolate_velocities();
        self.previous_mac_grid = self.mac_grid.clone();
        self.project();
        self.enforce_boundary_velocities();
        self.extrapolate_velocities();
        self.subtract_previous_grid();
        self.solve_pic_flip();
        self.advance_particles();
        self.constrain_particles_to_domain();
    }

    pub fn particle_positions(&self) -> impl Iterator<Item = Vec3> + '_ {
        self.particles
            .iter()
            .filter(|particle| particle.particle_type == ParticleType::Fluid)
            .map(|particle| particle.position)
    }
}
