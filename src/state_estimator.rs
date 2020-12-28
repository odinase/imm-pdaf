pub mod ekf;
pub mod models;

pub trait StateEstimator {
    type Params;
    type Measurement;

    fn predict(&self, eststate: Self::Params, ts: f64) -> Self::Params;

    fn update(&self, z: Self::Measurement, eststate: Self::Params) -> Self::Params;

    fn step(&self, z: Self::Measurement, eststate: Self::Params, ts: f64) -> Self::Params;

    fn estimate(&self, eststate: Self::Params) -> Self::Params;

    fn loglikelihood(&self, z: &Self::Measurement, eststate: &Self::Params) -> f64;

    fn gate(&self, z: &Self::Measurement, eststate: &Self::Params, gate_size_square: f64) -> bool;
}