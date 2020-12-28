pub mod dynamic;
pub mod measurement;

pub trait DynamicModel {
    type State;
    type Covariance;
    type Jacobian;
    
    fn f(&self, x: &Self::State, ts: f64) -> Self::State;
    fn F(&self, x: &Self::State, ts: f64) -> Self::Jacobian;
    fn Q(&self, x: &Self::State, ts: f64) -> Self::Covariance;
}

pub trait MeasurementModel {
    type State;
    type Measurement;
    type Jacobian;
    type Covariance;
    
    fn h(&self, x: &Self::State) -> Self::Measurement;
    fn H(&self, x: &Self::State) -> Self::Jacobian;
    fn R(&self, x: &Self::State, z: &Self::Measurement) -> Self::Covariance;
}