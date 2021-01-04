#![allow(non_snake_case)]
use matfile::{NumericData, MatFile};
use nalgebra::{DMatrix, DVector};
use imm_pdaf::{
    simulator as sim,
    state_estimator::{
        StateEstimator,
        models::{
            dynamic::CV,
            measurement::CartesianPosition
        },
        ekf
    }
};
use itertools::izip;
use gnuplot::*;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    sim::run_ekf()
}
