use super::MeasurementModel;
use nalgebra::{DVector, DMatrix};

pub struct CartesianPosition {
    sigma_p: f64,
}

impl CartesianPosition {
    pub fn new(sigma_p: f64) -> Self {
        CartesianPosition {
            sigma_p,
        }
    }
}

impl MeasurementModel for CartesianPosition {
    type State = DVector<f64>;
    type Measurement = DVector<f64>;
    type Jacobian = DMatrix<f64>;
    type Covariance = DMatrix<f64>;
    
    /// Assumes p is the first state
    fn h(&self, x: &Self::State) -> Self::Measurement {
        let mut p = DVector::zeros(2);
        p.copy_from(&x.rows(0, 2));
        p
    }
    fn H(&self, x: &Self::State) -> Self::Jacobian {
        let n = x.len();
        let H = DMatrix::identity(2, n);
        H
    }
    fn R(&self, x: &Self::State, z: &Self::Measurement) -> Self::Covariance {
        let n = z.len();
        let R = DMatrix::<f64>::identity(n,n)*self.sigma_p.powi(2);
        R
    }
}