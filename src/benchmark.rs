use crate::simulation::FlipSimulation;
use glam::UVec3;

pub fn benchmark() {
    let mut simulation = FlipSimulation::new(UVec3::splat(32), 0.5, 1.0 / 120.0);

    for _ in 0..1000 {
        simulation.step();
    }
    println!("Simulation completed.");
}
