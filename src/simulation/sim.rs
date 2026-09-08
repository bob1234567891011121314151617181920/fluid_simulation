use crate::{
    grids::{CellType, Grid3D, MacGrid3D, Particle, ParticleGrid, ParticleType},
    simulation::pcg_solver::pcg,
};

use glam::{IVec3, UVec3, Vec3, u32, uvec3, vec3};

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
            .fold(0.0, f32::max);

        if self.max_density == 0.0 {
            log::error!("Density calibration produced zero density");
        }

        self.particles.clear();

        let domain_size = self.dimensions.as_vec3() / max_dimension;
        let fluid_min = domain_size * Vec3::new(0.05, 0.1, 0.05);
        let fluid_max = domain_size * Vec3::new(0.8, 0.3, 0.8);

        let particle_counts = ((fluid_max - fluid_min) / particle_spacing).as_uvec3();

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
        self.build_fluid_sdf();
        self.mark_cell_types();
    }

    fn apply_gravity(&mut self, dt: f32) {
        for particle in self.particles.iter_mut() {
            if particle.particle_type == ParticleType::Solid {
                continue;
            }
            particle.velocity.y += GRAVITY * dt;
        }
    }

    fn transfer_velocity_component_to_grid<F>(
        particles: &[Particle],
        particle_grid: &ParticleGrid,
        velocity_grid: &mut Grid3D<f32>,
        valid_grid: &mut Grid3D<bool>,
        face_offset: Vec3,
        neighbor_extent: UVec3,
        max_dimension: f32,
        get_velocity: F,
    ) where
        F: Fn(&Particle) -> f32 + Copy,
    {
        const RADIUS: f32 = 1.4;
        const RADIUS_SQUARED: f32 = RADIUS * RADIUS;

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

                        let grid_position = particle.position * max_dimension;

                        let distance_squared = face_position.distance_squared(grid_position);

                        let weight = particle.mass
                            * (RADIUS_SQUARED / distance_squared.max(1.0e-5) - 1.0).max(0.0);

                        weighted_velocity_sum += weight * get_velocity(particle);

                        weight_sum += weight;
                    });

                    let velocity = if weight_sum > 0.0 {
                        valid_grid.set(x, y, z, true);
                        weighted_velocity_sum / weight_sum
                    } else {
                        valid_grid.set(x, y, z, false);
                        0.0
                    };
                    velocity_grid.set(x, y, z, velocity);
                }
            }
        }
    }

    fn transfer_particle_velocities_to_mac_grid(&mut self) {
        let particles = &self.particles;
        let particle_grid = &self.particle_grid;
        let max_dimension = self.dimensions.max_element() as f32;
        let mac_grid = &mut self.mac_grid;

        Self::transfer_velocity_component_to_grid(
            particles,
            particle_grid,
            &mut mac_grid.u_x,
            &mut mac_grid.valid_u_x,
            vec3(0.0, 0.5, 0.5),
            UVec3::new(1, 2, 2),
            max_dimension,
            |particle| particle.velocity.x,
        );

        Self::transfer_velocity_component_to_grid(
            particles,
            particle_grid,
            &mut mac_grid.u_y,
            &mut mac_grid.valid_u_y,
            vec3(0.5, 0.0, 0.5),
            UVec3::new(2, 1, 2),
            max_dimension,
            |particle| particle.velocity.y,
        );

        Self::transfer_velocity_component_to_grid(
            particles,
            particle_grid,
            &mut mac_grid.u_z,
            &mut mac_grid.valid_u_z,
            vec3(0.5, 0.5, 0.0),
            UVec3::new(2, 2, 1),
            max_dimension,
            |particle| particle.velocity.z,
        );
    }

    fn pressure_diagonal(
        position: UVec3,
        dimensions: UVec3,
        cell_types: &Grid3D<CellType>,
        sdf: &Grid3D<f32>,
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
                CellType::Fluid => diagonal += inverse_h_squared,
                CellType::Air => {
                    let theta = Self::interface_fraction(
                        sdf.get(position.x as u32, position.y as u32, position.z as u32),
                        sdf.get(neighbor.x, neighbor.y, neighbor.z),
                    );
                    diagonal += inverse_h_squared / theta;
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
                        &mac_grid.sdf,
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

                        let position = IVec3::new(x as i32, y as i32, z as i32);
                        let cell_index = index(x, y, z);
                        let diagonal = FlipSimulation::pressure_diagonal(
                            position.as_uvec3(),
                            dimensions,
                            cell_types,
                            &mac_grid.sdf,
                            inverse_h_squared,
                        );
                        let mut result = diagonal * input[cell_index];
                        let neighbor_offsets = [
                            -IVec3::X,
                            IVec3::X,
                            -IVec3::Y,
                            IVec3::Y,
                            -IVec3::Z,
                            IVec3::Z,
                        ];

                        for offset in neighbor_offsets {
                            let neighbor = position + offset;

                            if neighbor.cmplt(IVec3::ZERO).any()
                                || neighbor.cmpge(dimensions.as_ivec3()).any()
                            {
                                continue;
                            }

                            let neighbor = neighbor.as_uvec3();

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
            println!("in");
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
            &mac_grid.cell_type,
            &mac_grid.sdf,
            &mut mac_grid.u_x,
            dimensions,
            0,
            h,
        );

        Self::subtract_pressure_gradient_component(
            &mac_grid.pressure,
            &mac_grid.cell_type,
            &mac_grid.sdf,
            &mut mac_grid.u_y,
            dimensions,
            1,
            h,
        );

        Self::subtract_pressure_gradient_component(
            &mac_grid.pressure,
            &mac_grid.cell_type,
            &mac_grid.sdf,
            &mut mac_grid.u_z,
            dimensions,
            2,
            h,
        );
    }

    fn subtract_pressure_gradient_component(
        pressure: &Grid3D<f32>,
        cell_types: &Grid3D<CellType>,
        sdf: &Grid3D<f32>,
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

                    let forward_type = cell_types.get(forward.x, forward.y, forward.z);
                    let backward_type = cell_types.get(backward.x, backward.y, backward.z);
                    if forward_type == CellType::Solid
                        || backward_type == CellType::Solid
                        || (forward_type != CellType::Fluid && backward_type != CellType::Fluid)
                    {
                        continue;
                    }
                    let phi_forward = sdf.get(forward.x, forward.y, forward.z);
                    let phi_backward = sdf.get(backward.x, backward.y, backward.z);
                    let theta = match (forward_type, backward_type) {
                        (CellType::Fluid, CellType::Air) => {
                            Self::interface_fraction(phi_forward, phi_backward)
                        }
                        (CellType::Air, CellType::Fluid) => {
                            Self::interface_fraction(phi_backward, phi_forward)
                        }
                        _ => 1.0,
                    };
                    let forward_pressure = pressure.get(forward.x, forward.y, forward.z);
                    let backward_pressure = pressure.get(backward.x, backward.y, backward.z);
                    let pressure_gradient = (forward_pressure - backward_pressure) / (h * theta);
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
                    if self.mac_grid.check_wall(x, y, z) || self.mac_grid.check_wall(x, y, z - 1) {
                        self.mac_grid.u_z.set(x, y, z, 0.0);
                    }
                }
            }
        }
    }

    fn extrapolate_velocity_component<F>(
        velocity: &mut Grid3D<f32>,
        valid: &mut Grid3D<bool>,
        cell_types: &Grid3D<CellType>,
        face_dimensions: UVec3,
        adjacent_cells: F,
        iterations: usize,
    ) where
        F: Fn(UVec3) -> (Option<UVec3>, Option<UVec3>),
    {
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

                    let touches_solid = negative_type == Some(CellType::Solid)
                        || positive_type == Some(CellType::Solid);

                    let boundary = negative_cell.is_none() || positive_cell.is_none();

                    wall_mark.set(x, y, z, boundary || touches_solid);
                }
            }
        }

        for _ in 0..iterations {
            let old_velocity = velocity.clone();
            let old_valid = valid.clone();

            for x in 0..face_dimensions.x {
                for y in 0..face_dimensions.y {
                    for z in 0..face_dimensions.z {
                        if old_valid.get(x, y, z) || wall_mark.get(x, y, z) {
                            continue;
                        }

                        let position = UVec3::new(x, y, z);
                        let mut velocity_sum = 0.0;
                        let mut neighbor_count = 0;

                        let neighbor_positions = [
                            position.saturating_sub(UVec3::X),
                            (position + UVec3::X).min(face_dimensions - 1),
                            position.saturating_sub(UVec3::Y),
                            (position + UVec3::Y).min(face_dimensions - 1),
                            position.saturating_sub(UVec3::Z),
                            (position + UVec3::Z).min(face_dimensions - 1),
                        ];

                        for neighbor in neighbor_positions {
                            if !old_valid.get(neighbor.x, neighbor.y, neighbor.z) {
                                continue;
                            }

                            velocity_sum += old_velocity.get(neighbor.x, neighbor.y, neighbor.z);

                            neighbor_count += 1;

                            if neighbor_count > 0 {
                                velocity.set(x, y, z, velocity_sum / neighbor_count as f32);
                            }
                            valid.set(x, y, z, true);
                        }
                    }
                }
            }
        }
    }

    fn extrapolate_velocities(&mut self) {
        let mac_grid = &mut self.mac_grid;
        let dimensions = mac_grid.dimensions;
        let cell_type = &mac_grid.cell_type;
        let iterations = 3;

        Self::extrapolate_velocity_component(
            &mut mac_grid.u_x,
            &mut mac_grid.valid_u_x,
            cell_type,
            UVec3::new(dimensions.x + 1, dimensions.y, dimensions.z),
            |position| {
                (
                    (position.x > 0).then(|| position - UVec3::X),
                    (position.x < dimensions.x).then_some(position),
                )
            },
            iterations,
        );

        Self::extrapolate_velocity_component(
            &mut mac_grid.u_y,
            &mut mac_grid.valid_u_y,
            cell_type,
            UVec3::new(dimensions.x, dimensions.y + 1, dimensions.z),
            |position| {
                (
                    (position.y > 0).then(|| position - UVec3::Y),
                    (position.y < dimensions.y).then_some(position),
                )
            },
            iterations,
        );

        Self::extrapolate_velocity_component(
            &mut mac_grid.u_z,
            &mut mac_grid.valid_u_z,
            cell_type,
            UVec3::new(dimensions.x, dimensions.y, dimensions.z + 1),
            |position| {
                (
                    (position.z > 0).then(|| position - UVec3::Z),
                    (position.z < dimensions.z).then_some(position),
                )
            },
            iterations,
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

    fn advance_particles(&mut self, dt: f32) {
        for particle in self
            .particles
            .iter_mut()
            .filter(|particle| particle.particle_type == ParticleType::Fluid)
        {
            let postion = particle.position;
            let velocity_1 = Self::interpolate_velocities(&self.mac_grid, postion);
            let mid_position = postion + 0.5 * dt * velocity_1;
            let velocity_2 = Self::interpolate_velocities(&self.mac_grid, mid_position);
            particle.position += dt * velocity_2;
        }
        self.constrain_particles_to_domain();
    }

    fn constrain_particles_to_domain(&mut self) {
        let max_dimension = self.dimensions.max_element() as f32;
        let domain_size = self.dimensions.as_vec3() / max_dimension;

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

    fn overlap_direction(a: usize, b: usize) -> Vec3 {
        let hash = a.wrapping_mul(73_856_093) ^ b.wrapping_mul(19_349_663);

        match hash % 6 {
            0 => Vec3::X,
            1 => -Vec3::X,
            2 => Vec3::Y,
            3 => -Vec3::Y,
            4 => Vec3::Z,
            _ => -Vec3::Z,
        }
    }

    fn sperate_particles(&mut self, dt: f32) -> Vec<Vec3> {
        const SPRING_FORCE: f32 = 50.0;
        const MIN_DISTANCE_FACTOR: f32 = 0.1;
        const MAX_DISPLACEMENT_FACTOR: f32 = 0.1;

        let max_dimension = self.dimensions.max_element() as f32;
        let radius = self.density / max_dimension;
        let radius_squared = radius * radius;
        self.particle_grid.sort(&self.particles);

        let mut temporary_positions: Vec<Vec3> = self
            .particles
            .iter()
            .map(|particle| particle.position)
            .collect();

        for particle_index in 0..self.particles.len() {
            let particle = &self.particles[particle_index];
            if particle.particle_type != ParticleType::Fluid {
                continue;
            }

            let grid_position = (particle.position * max_dimension)
                .floor()
                .as_uvec3()
                .min(self.dimensions - UVec3::ONE);

            let mut spring = Vec3::ZERO;

            self.particle_grid
                .for_get_cell_neighbors(grid_position, UVec3::ONE, |other_index| {
                    if particle_index == other_index {
                        return;
                    }

                    let other = &self.particles[other_index];
                    let difference = particle.position - other.position;

                    let distance_squared = difference.length_squared();

                    if distance_squared >= radius_squared {
                        return;
                    }

                    let distance = distance_squared.sqrt();

                    if distance > MIN_DISTANCE_FACTOR * radius {
                        let direction = difference / distance;
                        let q = (1.0 - distance_squared / radius_squared).max(0.0);
                        let weight = SPRING_FORCE * other.mass * q * q * q;
                        spring += weight * direction * radius;
                    } else if other.particle_type == ParticleType::Fluid {
                        spring += Self::overlap_direction(particle_index, other_index)
                            * (0.01 * radius / dt);
                    }
                });
            let maximum_displacement = MAX_DISPLACEMENT_FACTOR * radius;

            let displacement = (dt * spring).clamp_length_max(maximum_displacement);

            temporary_positions[particle_index] = particle.position + displacement;
        }

        temporary_positions
    }

    fn resample_particle_velocity(
        particle_grid: &ParticleGrid,
        particles: &[Particle],
        sample_position: Vec3,
        fallback_velocity: Vec3,
        dimensions: UVec3,
        radius: f32,
    ) -> Vec3 {
        let max_dimension = dimensions.max_element() as f32;
        let maximum_cell = (dimensions - UVec3::ONE).as_vec3();
        let grid_position = (sample_position * max_dimension)
            .floor()
            .clamp(Vec3::ZERO, maximum_cell)
            .as_uvec3();
        let radius_squared = radius * radius;

        let mut weighted_velocity = Vec3::ZERO;
        let mut weight_sum = 0.0;

        particle_grid.for_get_cell_neighbors(grid_position, UVec3::ONE, |neighbor_index| {
            let neighbor = &particles[neighbor_index];
            if neighbor.particle_type != ParticleType::Fluid {
                return;
            }

            let distance_squared = neighbor.position.distance_squared(sample_position);
            if distance_squared >= radius_squared {
                return;
            }

            let q = (1.0 - distance_squared / radius_squared).max(0.0);
            let weight = neighbor.mass * q * q * q;
            weighted_velocity += weight * neighbor.velocity;
            weight_sum += weight;
        });

        if weight_sum > f32::EPSILON {
            weighted_velocity / weight_sum
        } else {
            fallback_velocity
        }
    }

    fn resample_particles(&mut self, dt: f32) {
        let temporary_positions = self.sperate_particles(dt);
        let max_dimension = self.dimensions.max_element() as f32;
        let radius = self.density / max_dimension;

        let mut temporary_velocities: Vec<Vec3> = self
            .particles
            .iter()
            .map(|particle| particle.velocity)
            .collect();

        for particle_index in 0..self.particles.len() {
            let particle = &self.particles[particle_index];
            if particle.particle_type != ParticleType::Fluid {
                continue;
            }

            temporary_velocities[particle_index] = Self::resample_particle_velocity(
                &self.particle_grid,
                &self.particles,
                temporary_positions[particle_index],
                particle.velocity,
                self.dimensions,
                radius,
            );
        }

        for particle_index in 0..self.particles.len() {
            let particle = &mut self.particles[particle_index];
            if particle.particle_type != ParticleType::Fluid {
                continue;
            }

            particle.position = temporary_positions[particle_index];
            particle.velocity = temporary_velocities[particle_index];
        }

        self.constrain_particles_to_domain();
        self.particle_grid.sort(&self.particles);
    }

    fn calculate_substeps(&self) -> usize {
        const CFL: f32 = 0.5;
        const MAX_SUBSTEPS: usize = 8;

        let max_dimension = self.dimensions.max_element() as f32;
        let cell_size = 1.0 / max_dimension;

        let maximum_velocity = self
            .particles
            .iter()
            .filter(|particle| particle.particle_type == ParticleType::Fluid)
            .map(|particle| particle.velocity.length())
            .fold(0.0, f32::max);

        let maximum_distance = CFL * cell_size;

        let required_substeps =
            (maximum_velocity * self.step_size / maximum_distance).ceil() as usize;
        required_substeps.clamp(1, MAX_SUBSTEPS)
    }

    fn build_fluid_sdf(&mut self) {
        let dimensions = self.dimensions;
        let max_dimension = dimensions.max_element() as f32;

        let particle_spacing = self.density / max_dimension;
        let particle_radius = 1.5 * particle_spacing;

        let search_extent = (particle_radius * max_dimension).ceil() as u32 + 1;
        let particles = &self.particles;
        let particle_grid = &self.particle_grid;

        let fluid_sdf = &mut self.mac_grid.sdf;
        // Truncate the field one cell outside the surface; pressure only needs
        // distances at adjacent fluid/air cell centers.
        let band_width = 1.0 / max_dimension;

        for x in 0..self.dimensions.x {
            for y in 0..self.dimensions.y {
                for z in 0..self.dimensions.z {
                    let mut minimum_distance = band_width;
                    let cell = uvec3(x, y, z);
                    let cell_center = (cell.as_vec3() + Vec3::splat(0.5)) / max_dimension;
                    particle_grid.for_get_cell_neighbors(
                        cell,
                        UVec3::splat(search_extent),
                        |particle_index| {
                            let particle = &particles[particle_index];
                            if particle.particle_type != ParticleType::Fluid {
                                return;
                            }

                            let phi = cell_center.distance(particle.position) - particle_radius;
                            minimum_distance = minimum_distance.min(phi);
                        },
                    );
                    fluid_sdf.set(x, y, z, minimum_distance);
                }
            }
        }
    }

    fn mark_cell_types(&mut self) {
        self.particle_grid
            .mark_cell_types(&self.particles, &mut self.mac_grid.cell_type);
        for x in 0..self.dimensions.x {
            for y in 0..self.dimensions.y {
                for z in 0..self.dimensions.z {
                    if self.mac_grid.cell_type.get(x, y, z) != CellType::Solid {
                        let kind = if self.mac_grid.sdf.get(x, y, z) < 0.0 {
                            CellType::Fluid
                        } else {
                            CellType::Air
                        };
                        self.mac_grid.cell_type.set(x, y, z, kind);
                    }
                }
            }
        }
    }

    fn interface_fraction(phi_fluid: f32, phi_air: f32) -> f32 {
        let denominator = phi_fluid - phi_air;

        if denominator.abs() <= f32::EPSILON {
            return 1.0;
        }

        (phi_fluid / denominator).clamp(0.01, 1.0)
    }
    fn substep(&mut self, dt: f32) {
        self.particle_grid.sort(&self.particles);
        self.compute_density();
        self.apply_gravity(dt);
        self.build_fluid_sdf();
        self.transfer_particle_velocities_to_mac_grid();
        self.mark_cell_types();
        self.enforce_boundary_velocities();
        self.previous_mac_grid = self.mac_grid.clone();
        self.project();
        self.enforce_boundary_velocities();
        self.extrapolate_velocities();
        self.subtract_previous_grid();
        self.solve_pic_flip();
        self.advance_particles(dt);
        self.constrain_particles_to_domain();
        self.particle_grid.sort(&self.particles);
        self.resample_particles(dt);
        self.constrain_particles_to_domain();
    }

    pub fn step(&mut self) {
        let substeps = self.calculate_substeps();
        let dt = self.step_size / substeps as f32;

        for _ in 0..substeps {
            self.substep(dt);
        }
    }

    pub fn particle_positions(&self) -> impl Iterator<Item = Vec3> + '_ {
        self.particles
            .iter()
            .filter(|particle| particle.particle_type == ParticleType::Fluid)
            .map(|particle| particle.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdf_matches_particle_distances_and_resets_after_removal() {
        let mut sim = FlipSimulation::new(uvec3(8, 6, 4), 0.5, 0.01);
        sim.particles = vec![Particle::new_fluid(vec3(1.5, 1.5, 1.5) / 8.0, Vec3::ZERO)];
        let mut solid = Particle::new_fluid(vec3(6.5, 4.5, 2.5) / 8.0, Vec3::ZERO);
        solid.particle_type = ParticleType::Solid;
        sim.particles.push(solid);
        sim.particle_grid.sort(&sim.particles);
        sim.build_fluid_sdf();
        sim.mark_cell_types();
        for x in 0..8 {
            for y in 0..6 {
                for z in 0..4 {
                    let center = (uvec3(x, y, z).as_vec3() + Vec3::splat(0.5)) / 8.0;
                    let expected =
                        (center.distance(sim.particles[0].position) - 0.75 / 8.0).min(1.0 / 8.0);
                    assert!((sim.mac_grid.sdf.get(x, y, z) - expected).abs() < 1e-6);
                }
            }
        }
        assert_eq!(sim.mac_grid.cell_type.get(1, 1, 1), CellType::Fluid);
        assert_eq!(sim.mac_grid.cell_type.get(6, 4, 2), CellType::Solid);
        sim.particles.clear();
        sim.particle_grid.sort(&sim.particles);
        sim.build_fluid_sdf();
        sim.mark_cell_types();
        assert_eq!(sim.mac_grid.sdf.get(1, 1, 1), 1.0 / 8.0);
        assert_eq!(sim.mac_grid.cell_type.get(1, 1, 1), CellType::Air);
    }

    #[test]
    fn surface_projection_removes_divergence_on_every_axis() {
        for axis in 0..3 {
            for reverse in [false, true] {
                let mut sim = FlipSimulation::new(UVec3::splat(3), 0.5, 0.01);
                sim.mac_grid = MacGrid3D::new(UVec3::splat(3));
                sim.mac_grid.cell_type.set(1, 1, 1, CellType::Fluid);
                sim.mac_grid.sdf.set(1, 1, 1, -0.1);
                let mut face = UVec3::ONE;
                if reverse {
                    face[axis] += 1;
                }
                let velocity = match axis {
                    0 => &mut sim.mac_grid.u_x,
                    1 => &mut sim.mac_grid.u_y,
                    _ => &mut sim.mac_grid.u_z,
                };
                velocity.set(face.x, face.y, face.z, 1.0);
                sim.project();
                let grid = &sim.mac_grid;
                let divergence = grid.u_x.get(2, 1, 1) - grid.u_x.get(1, 1, 1)
                    + grid.u_y.get(1, 2, 1)
                    - grid.u_y.get(1, 1, 1)
                    + grid.u_z.get(1, 1, 2)
                    - grid.u_z.get(1, 1, 1);
                assert!(divergence.abs() < 1e-5, "axis {axis}: {divergence}");
            }
        }
    }
}
