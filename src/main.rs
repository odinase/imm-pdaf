#![allow(non_snake_case)]
use imm_pdaf::simulator::run_ekf;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_ekf()
}
