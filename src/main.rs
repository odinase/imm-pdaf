#![allow(non_snake_case)]
use imm_pdaf::simulator as sim;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    sim::run_imm()
}
