use super::models::{DynamicModel, MeasurementModel};
use super::{StateEstimator, Consistency};
use nalgebra::{DVector, DMatrix};
use std::f64::consts::TAU as _2_PI;

pub struct GaussParams {
    pub x: DVector<f64>,
    pub P: DMatrix<f64>,
}

impl GaussParams {
    fn set(self, x: DVector<f64>, P: DMatrix<f64>) -> Self {
        self = GaussParams {x, P};
        self
    }
}

impl GaussParams {
    pub fn new(x: DVector<f64>, P: DMatrix<f64>) -> Self {
        GaussParams {
            x,
            P
        }
    }
}

pub struct EKF<D, M>
where
    D: DynamicModel,
    M: MeasurementModel,
{
    dynmod: D,
    measmod: M,
}

impl<D, K> StateEstimator for EKF<D, K>
where
    D: DynamicModel<State=DVector<f64>, Jacobian=DMatrix<f64>, Covariance=DMatrix<f64>>,
    K: MeasurementModel<State = DVector<f64>, Jacobian = DMatrix<f64>, Covariance = DMatrix<f64>, Measurement = DVector<f64>>,
{
    type Params = GaussParams;
    type Measurement = DVector<f64>;

    fn predict(&self, eststate: Self::Params, ts: f64) -> Self::Params {
        let x = eststate.x;
        let P = eststate.P;
        let F = self.dynmod.F(&x, ts);
        let Q = self.dynmod.Q(&x, ts);
        
        let x = self.dynmod.f(&x, ts);
        let P = F * P * F.transpose() + Q;

        GaussParams::new(x, P)
    }

    fn update(&self, z: Self::Measurement, eststate: Self::Params) -> Self::Params {

        let (v, S) = self.innovation(&eststate, &z);
        
        let x = eststate.x;
        let P = eststate.P;
        let n = x.len();
        let H = self.measmod.H(&x);
        let R = self.measmod.R(&x, &z);

        // Kalman gain. S should be PSD, or else we fucked up
        let W = S.cholesky().expect("S not PSD").solve(&(H*P)).transpose();

        let I = DMatrix::identity(n, n);
        let Jo = I - W*H;

        let P = Jo*P*Jo.transpose() + W*R*W.transpose();
        let x = x + W*v;

        GaussParams::new(x, P)
    }

    fn step(&self, z: Self::Measurement, eststate: Self::Params, ts: f64) -> Self::Params {
        let eststate_pred = self.predict(eststate, ts);
        let eststate_upd = self.update(z, eststate_pred);
        eststate_upd
    }

    fn estimate(&self, eststate: Self::Params) -> Self::Params {
        // Get the estimate from the state with its covariance. (Compatibility method)
        eststate
    }

    fn loglikelihood(&self, z: &Self::Measurement, eststate: &Self::Params) -> f64 {
        let nis = self.NIS(eststate);
        let S = self.innovation_cov(eststate, z);
        let c = (_2_PI*S).determinant();
        let llh = -0.5*(c + nis);
        llh
    }

    fn gate(&self, z: &Self::Measurement, eststate: &Self::Params, gate_size_square: f64) -> bool {
        let nis = self.NIS(eststate, z);
        nis <= gate_size_square
    }
}

impl<D, K> Consistency for EKF<D, K>
where
D: DynamicModel<State=DVector<f64>, Jacobian=DMatrix<f64>, Covariance=DMatrix<f64>>,
K: MeasurementModel<State = DVector<f64>, Jacobian = DMatrix<f64>, Covariance = DMatrix<f64>, Measurement = DVector<f64>>,
{
    type Params = GaussParams;
    type Measurement = DVector<f64>;
    type GroundTruth = DVector<f64>;

    fn NIS(&self, eststate: &GaussParams, z: &DVector<f64>) -> f64 {
        let (v, S) = self.innovation(&eststate, &z);
        let S_inv_v = S.cholesky().expect("S not PSD??").solve(&v);
        let nis = v.dot(&S_inv_v);
        nis
    }

    fn NEES(&self,eststate: &Self::Params, x_gt: &Self::GroundTruth) -> f64 {
        let x = &eststate.x;
        let P = &eststate.P;
        let x_err = x - x_gt;
        let P_inv_x_err = P.cholesky().expect("P not PSD??").solve(&x_err);
        let nees = x_err.dot(&P_inv_x_err);
        nees
    }
}


impl<D, M> EKF<D, M>
where
D: DynamicModel<State=DVector<f64>, Jacobian=DMatrix<f64>, Covariance=DMatrix<f64>>,
M: MeasurementModel<State = DVector<f64>, Jacobian = DMatrix<f64>, Covariance = DMatrix<f64>, Measurement = DVector<f64>>,
{
    pub fn innovation(&self, eststate: &GaussParams, z: &DVector<f64>) -> (DVector<f64>, DMatrix<f64>) {
        let v = self.innovation_mean(eststate, z);
        let S = self.innovation_cov(eststate, z);
        (v, S)
    }

    fn innovation_mean(&self, eststate: &GaussParams, z: &DVector<f64>) -> DVector<f64> {
        let x = &eststate.x;
        let zpred = self.measmod.h(x);
        let v = zpred - z;
        v
    }

    fn innovation_cov(&self, eststate: &GaussParams, z: &DVector<f64>) -> DMatrix<f64> {
        let x = &eststate.x;
        let H = self.measmod.H(x);
        let P = eststate.P;
        let R = self.measmod.R(x, z);
        let S = H * P * H.transpose() + R;
        S
    }
}