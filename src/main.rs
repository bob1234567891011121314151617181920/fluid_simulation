use fluid_simulation::run;
fn main() {
    match run() {
        Ok(_) => log::info!("Simulation completed successfully."),
        Err(error) => log::error!("Simulation failed: {:?}", error),
    }
}
