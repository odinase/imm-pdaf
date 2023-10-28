#![allow(non_snake_case)]
use imm_pdaf::simulator as sim;

fn main() -> anyhow::Result<()> {
    sim::run_pdaf()
}
