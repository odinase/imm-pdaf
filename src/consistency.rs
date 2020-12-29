pub trait Consistency {
    type Params;
    type Measurement;
    type GroundTruth;

    fn NIS(&self, eststate: &Self::Params, z: &Self::Measurement) -> f64;
    fn NEES(&self,eststate: &Self::Params, x_gt: &Self::GroundTruth) -> f64;
}