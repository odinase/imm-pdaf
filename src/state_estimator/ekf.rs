use super::models::{DynamicModel, MeasurementModel};
use super::StateEstimator;
use crate::consistency::Consistency;
use crate::mixture::{MixtureParameters, ReduceMixture};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::TAU as _2_PI;

#[derive(Debug, Clone)]
pub struct GaussParams {
    pub x: DVector<f64>,
    pub P: DMatrix<f64>,
}

impl std::fmt::Display for GaussParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "x: {}\nP: {}", self.x, self.P)
    }
}

impl GaussParams {
    fn set(self, x: DVector<f64>, P: DMatrix<f64>) -> Self {
        GaussParams { x, P }
    }

    pub fn new(x: DVector<f64>, P: DMatrix<f64>) -> Self {
        GaussParams { x, P }
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
    D: DynamicModel<State = DVector<f64>, Jacobian = DMatrix<f64>, Covariance = DMatrix<f64>>,
    K: MeasurementModel<
        State = DVector<f64>,
        Jacobian = DMatrix<f64>,
        Covariance = DMatrix<f64>,
        Measurement = DVector<f64>,
    >,
{
    type Params = GaussParams;
    type Measurement = DVector<f64>;

    fn predict(&self, eststate: Self::Params, ts: f64) -> Self::Params {
        let x = &eststate.x;
        let P = &eststate.P;
        let F = &self.dynmod.F(&x, ts);
        let Q = &self.dynmod.Q(&x, ts);
        let x = self.dynmod.f(&x, ts);
        let P = F * P * F.transpose() + Q;

        GaussParams::new(x, P)
    }

    fn update(&self, z: &Self::Measurement, eststate: Self::Params) -> Self::Params {
        let (v, S) = self.innovation(&eststate, &z);
        let x = &eststate.x;
        let P = &eststate.P;
        let n = x.len();
        let H = &self.measmod.H(&x);
        let R = &self.measmod.R(&x, &z);

        // Kalman gain. S should be PSD, or else we fucked up
        let W = &(S.cholesky().expect("S not PSD").solve(&(H * P)).transpose());

        let I = DMatrix::identity(n, n);
        let Jo = &(I - W * H);

        let P = Jo * P * Jo.transpose() + W * R * W.transpose();
        let x = x + W * v;

        GaussParams::new(x, P)
    }

    fn step(&self, z: &Self::Measurement, eststate: Self::Params, ts: f64) -> Self::Params {
        let eststate_pred = self.predict(eststate, ts);
        let eststate_upd = self.update(z, eststate_pred);
        eststate_upd
    }

    fn estimate(&self, eststate: Self::Params) -> Self::Params {
        // Get the estimate from the state with its covariance. (Compatibility method)
        eststate
    }

    fn loglikelihood(&self, z: &Self::Measurement, eststate: &Self::Params) -> f64 {
        let nis = self.NIS(eststate, z);
        let S = self.innovation_cov(eststate, z);
        let c = (_2_PI * S).determinant().ln();
        let llh = -0.5 * (c + nis);
        llh
    }

    fn gate(&self, z: &Self::Measurement, eststate: &Self::Params, gate_size_square: f64) -> bool {
        let nis = self.NIS(eststate, z);
        nis <= gate_size_square
    }
}

impl<D, K> Consistency for EKF<D, K>
where
    D: DynamicModel<State = DVector<f64>, Jacobian = DMatrix<f64>, Covariance = DMatrix<f64>>,
    K: MeasurementModel<
        State = DVector<f64>,
        Jacobian = DMatrix<f64>,
        Covariance = DMatrix<f64>,
        Measurement = DVector<f64>,
    >,
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

    fn NEES(&self, eststate: &Self::Params, x_gt: &Self::GroundTruth) -> f64 {
        let x = &eststate.x;
        let P = eststate.P.clone();
        let x_err = x - x_gt;
        let P_inv_x_err = &(P.cholesky().expect("P not PSD??").solve(&x_err));
        let nees = x_err.dot(P_inv_x_err);
        nees
    }
}

fn gaussian_reduce_mixture(mix_params: MixtureParameters<GaussParams>) -> (DVector<f64>, DMatrix<f64>) {
    let num_params = mix_params.components.len();
    // We assume all components have equal state length
    let nx = mix_params.components[0].x.len();

    let x_mean = mix_params.components.iter().map(|p| &p.x).zip(mix_params.weights.iter()).fold(
        DVector::zeros(nx),
        |mut xmean, (x, &w)| {
            xmean += x*w;
            xmean
        }
    );
    let P_mean = mix_params.components.iter().map(|p| (&p.x, &p.P)).zip(mix_params.weights.iter()).fold(
        DMatrix::zeros(nx, nx),
        |mut P_mean, ((x, P), &w)| {
            let xdiff = x - &x_mean;
            P_mean += P*w + w*&xdiff*&xdiff.transpose();
            P_mean
        }
    );
    (x_mean, P_mean)
}

impl<D, K> ReduceMixture<GaussParams> for EKF<D, K>
where
    D: DynamicModel<State = DVector<f64>, Jacobian = DMatrix<f64>, Covariance = DMatrix<f64>>,
    K: MeasurementModel<
        State = DVector<f64>,
        Jacobian = DMatrix<f64>,
        Covariance = DMatrix<f64>,
        Measurement = DVector<f64>,
    >,
{
    fn reduce_mixture(&self, estimator_mixture: MixtureParameters<GaussParams>) -> GaussParams {
        let (xmean, Pmean) = gaussian_reduce_mixture(estimator_mixture);
        GaussParams::new(
            xmean,
            Pmean
        )
    }
}

impl<D, M> EKF<D, M>
where
    D: DynamicModel<State = DVector<f64>, Jacobian = DMatrix<f64>, Covariance = DMatrix<f64>>,
    M: MeasurementModel<
        State = DVector<f64>,
        Jacobian = DMatrix<f64>,
        Covariance = DMatrix<f64>,
        Measurement = DVector<f64>,
    >,
{
    pub fn init(dynmod: D, measmod: M) -> Self {
        EKF { dynmod, measmod }
    }
    pub fn innovation(
        &self,
        eststate: &GaussParams,
        z: &DVector<f64>,
    ) -> (DVector<f64>, DMatrix<f64>) {
        let v = self.innovation_mean(eststate, z);
        let S = self.innovation_cov(eststate, z);
        (v, S)
    }

    fn innovation_mean(&self, eststate: &GaussParams, z: &DVector<f64>) -> DVector<f64> {
        let x = &eststate.x;
        let zpred = self.measmod.h(x);
        let v = z - zpred;
        v
    }

    fn innovation_cov(&self, eststate: &GaussParams, z: &DVector<f64>) -> DMatrix<f64> {
        let x = &eststate.x;
        let H = &self.measmod.H(x);
        let P = &eststate.P;
        let R = &self.measmod.R(x, z);
        let S = H * P * H.transpose() + R;
        S
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_estimator::models::dynamic::{CT, CV};
    use crate::state_estimator::models::measurement::CartesianPosition;
    use crate::state_estimator::models::{DynamicModel, MeasurementModel};

    static SIGMA_A_CV: f64 = 0.5;
    static SIGMA_A_CT: f64 = 0.5;
    static SIGMA_OMEGA: f64 = 0.3;
    static SIGMA_Z: f64 = 10.0;
    static TS: f64 = 0.1;

    #[test]
    fn test_ekf_ct_predict() {
        let x = DVector::from_row_slice(&[0., 0., 1., 1., 0.1]);
        let P = DMatrix::from_row_slice(
            5,
            5,
            &[
                1000000., 0., 0., 0., 0., 0., 1000000., 0., 0., 0., 0., 0., 900., 0., 0., 0., 0.,
                0., 900., 0., 0., 0., 0., 0., 0.01,
            ],
        );
        let z = DVector::from_row_slice(&[1.0, 0.0]);

        let measmod = CartesianPosition::new(SIGMA_Z);
        let dynmod = CT::new(SIGMA_A_CT, SIGMA_OMEGA);

        let ekf = EKF::init(dynmod, measmod);

        let ekfstate = GaussParams::new(x, P);

        let ekfstate = ekf.predict(ekfstate, TS);
        let x_correct =
            DVector::from_row_slice(&[0.09949834, 0.10049833, 0.98995017, 1.00994983, 0.1]);
        let P_correct = DMatrix::from_row_slice(
            5,
            5,
            &[
                1000009.00000859,
                -0.00000025,
                89.99975509,
                0.44999127,
                -0.00005033,
                -0.00000025,
                1000009.00000858,
                -0.45000127,
                89.99975492,
                0.00004967,
                89.99975509,
                -0.45000127,
                900.025102,
                -0.00009998,
                -0.00100995,
                0.44999127,
                89.99975492,
                -0.00009998,
                900.025098,
                0.00098995,
                -0.00005033,
                0.00004967,
                -0.00100995,
                0.00098995,
                0.019,
            ],
        );
        // println!("\nx_correct: {}\nP_correct: {}\n", x_correct, P_correct);
        // println!("\nx_pred: {}\nP_pred: {}\n", ekfstate.x, ekfstate.P);
        // println!("\nP_correct.shape() = {:#?}", P_correct.shape());
        // println!("\nekf.state.P.shape() = {:#?}", ekfstate.P.shape());
        assert!(ekfstate.x.len() == 5, "x.len() = {}", ekfstate.x.len());
        assert!(
            ekfstate.P.shape() == (5, 5),
            "P.shape() = {:#?}",
            ekfstate.P.shape()
        );
        assert!(x_correct.relative_eq(&ekfstate.x, 1e-5, 1e-5));
        assert!(P_correct.relative_eq(&ekfstate.P, 1e-5, 1e-5));
    }

    #[test]
    fn test_ekf_ct_update() {
        let x = DVector::from_row_slice(&[0.09949834, 0.10049833, 0.98995017, 1.00994983, 0.1]);
        let P = DMatrix::from_row_slice(
            5,
            5,
            &[
                1000009.00000859,
                -0.00000025,
                89.99975509,
                0.44999127,
                -0.00005033,
                -0.00000025,
                1000009.00000858,
                -0.45000127,
                89.99975492,
                0.00004967,
                89.99975509,
                -0.45000127,
                900.025102,
                -0.00009998,
                -0.00100995,
                0.44999127,
                89.99975492,
                -0.00009998,
                900.025098,
                0.00098995,
                -0.00005033,
                0.00004967,
                -0.00100995,
                0.00098995,
                0.019,
            ],
        );
        let z = DVector::from_row_slice(&[1.0, 0.0]);

        let measmod = CartesianPosition::new(SIGMA_Z);
        let dynmod = CT::new(SIGMA_A_CT, SIGMA_OMEGA);

        let ekf = EKF::init(dynmod, measmod);

        let ekfstate = GaussParams::new(x, P);
        let ekfstate = ekf.update(&z, ekfstate);

        let x_correct =
            DVector::from_row_slice(&[0.99990996, 0.00001005, 0.99003125, 1.0099412, 0.1]);
        let P_correct = DMatrix::from_row_slice(
            5,
            5,
            &[
                99.99000109,
                -0.,
                0.00899899,
                0.00004499,
                -0.00000001,
                -0.,
                99.99000109,
                -0.000045,
                0.00899899,
                0.,
                0.00899899,
                -0.000045,
                900.01700272,
                -0.00009998,
                -0.00100995,
                0.00004499,
                0.00899899,
                -0.00009998,
                900.01699872,
                0.00098995,
                -0.00000001,
                0.,
                -0.00100995,
                0.00098995,
                0.019,
            ],
        );

        assert!(ekfstate.x.len() == 5, "x.len() = {}", ekfstate.x.len());
        assert!(
            ekfstate.P.shape() == (5, 5),
            "P.shape() = {:#?}",
            ekfstate.P.shape()
        );
        assert!(x_correct.relative_eq(&ekfstate.x, 1e-5, 1e-5));
        assert!(P_correct.relative_eq(&ekfstate.P, 1e-5, 1e-5));
    }

    #[test]
    fn test_ekf_ct_step() {
        let x = DVector::from_row_slice(&[0., 0., 1., 1., 0.1]);
        let P = DMatrix::from_row_slice(
            5,
            5,
            &[
                1000000., 0., 0., 0., 0., 0., 1000000., 0., 0., 0., 0., 0., 900., 0., 0., 0., 0.,
                0., 900., 0., 0., 0., 0., 0., 0.01,
            ],
        );
        let z = DVector::from_row_slice(&[1.0, 0.0]);

        let measmod = CartesianPosition::new(SIGMA_Z);
        let dynmod = CT::new(SIGMA_A_CT, SIGMA_OMEGA);

        let ekf = EKF::init(dynmod, measmod);

        let ekfstate = GaussParams::new(x, P);

        let ekfstate = ekf.step(&z, ekfstate, TS);

        let x_correct =
            DVector::from_row_slice(&[0.99990996, 0.00001005, 0.99003125, 1.0099412, 0.1]);
        let P_correct = DMatrix::from_row_slice(
            5,
            5,
            &[
                99.99000109,
                -0.,
                0.00899899,
                0.00004499,
                -0.00000001,
                -0.,
                99.99000109,
                -0.000045,
                0.00899899,
                0.,
                0.00899899,
                -0.000045,
                900.01700272,
                -0.00009998,
                -0.00100995,
                0.00004499,
                0.00899899,
                -0.00009998,
                900.01699872,
                0.00098995,
                -0.00000001,
                0.,
                -0.00100995,
                0.00098995,
                0.019,
            ],
        );
        // println!("\nx_correct: {}\nP_correct: {}\n", x_correct, P_correct);
        // println!("\nx_pred: {}\nP_pred: {}\n", ekfstate.x, ekfstate.P);
        // println!("\nP_correct.shape() = {:#?}", P_correct.shape());
        // println!("\nekf.state.P.shape() = {:#?}", ekfstate.P.shape());
        assert!(ekfstate.x.len() == 5, "x.len() = {}", ekfstate.x.len());
        assert!(
            ekfstate.P.shape() == (5, 5),
            "P.shape() = {:#?}",
            ekfstate.P.shape()
        );
        assert!(x_correct.relative_eq(&ekfstate.x, 1e-5, 1e-5));
        assert!(P_correct.relative_eq(&ekfstate.P, 1e-5, 1e-5));
    }

    #[test]
    fn test_ekf_cv_predict() {
        let x = DVector::from_row_slice(&[0., 0., 1., 1., 0.]);
        let P = DMatrix::from_row_slice(
            5,
            5,
            &[
                1000000., 0., 0., 0., 0., 0., 1000000., 0., 0., 0., 0., 0., 900., 0., 0., 0., 0.,
                0., 900., 0., 0., 0., 0., 0., 0.,
            ],
        );
        let z = DVector::from_row_slice(&[1.0, 0.0]);

        let measmod = CartesianPosition::new(SIGMA_Z);
        let dynmod = CV::new(SIGMA_A_CV);

        let ekf = EKF::init(dynmod, measmod);

        let ekfstate = GaussParams::new(x, P);

        let ekfstate = ekf.predict(ekfstate, TS);
        let x_correct = DVector::from_row_slice(&[0.1, 0.1, 1., 1., 0.]);
        let P_correct = DMatrix::from_row_slice(
            5,
            5,
            &[
                1000009.00008333,
                0.,
                90.00125,
                0.,
                0.,
                0.,
                1000009.00008333,
                0.,
                90.00125,
                0.,
                90.00125,
                0.,
                900.025,
                0.,
                0.,
                0.,
                90.00125,
                0.,
                900.025,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
            ],
        );
        // println!("\nx_correct: {}\nP_correct: {}\n", x_correct, P_correct);
        // println!("\nx_pred: {}\nP_pred: {}\n", ekfstate.x, ekfstate.P);
        // println!("\nP_correct.shape() = {:#?}", P_correct.shape());
        // println!("\nekf.state.P.shape() = {:#?}", ekfstate.P.shape());
        assert!(ekfstate.x.len() == 5, "x.len() = {}", ekfstate.x.len());
        assert!(
            ekfstate.P.shape() == (5, 5),
            "P.shape() = {:#?}",
            ekfstate.P.shape()
        );
        assert!(x_correct.relative_eq(&ekfstate.x, 1e-5, 1e-5));
        assert!(P_correct.relative_eq(&ekfstate.P, 1e-5, 1e-5));
    }

    #[test]
    fn test_ekf_cv_update() {
        let x = DVector::from_row_slice(&[0.09949834, 0.10049833, 0.98995017, 1.00994983, 0.1]);
        let P = DMatrix::from_row_slice(
            5,
            5,
            &[
                1000009.00000859,
                -0.00000025,
                89.99975509,
                0.44999127,
                -0.00005033,
                -0.00000025,
                1000009.00000858,
                -0.45000127,
                89.99975492,
                0.00004967,
                89.99975509,
                -0.45000127,
                900.025102,
                -0.00009998,
                -0.00100995,
                0.44999127,
                89.99975492,
                -0.00009998,
                900.025098,
                0.00098995,
                -0.00005033,
                0.00004967,
                -0.00100995,
                0.00098995,
                0.019,
            ],
        );
        let z = DVector::from_row_slice(&[1.0, 0.0]);

        let measmod = CartesianPosition::new(SIGMA_Z);
        let dynmod = CV::new(SIGMA_A_CV);

        let ekf = EKF::init(dynmod, measmod);

        let ekfstate = GaussParams::new(x, P);
        let ekfstate = ekf.update(&z, ekfstate);

        let x_correct =
            DVector::from_row_slice(&[0.99990996, 0.00001005, 0.99003125, 1.00994119, 0.1]);
        let P_correct = DMatrix::from_row_slice(
            5,
            5,
            &[
                99.99000109,
                -0.,
                0.00899899,
                0.00004499,
                -0.00000001,
                -0.,
                99.99000109,
                -0.000045,
                0.00899899,
                0.,
                0.00899899,
                -0.000045,
                900.01700272,
                -0.00009998,
                -0.00100995,
                0.00004499,
                0.00899899,
                -0.00009998,
                900.01699872,
                0.00098995,
                -0.00000001,
                0.,
                -0.00100995,
                0.00098995,
                0.019,
            ],
        );
        println!("\nx_correct: {}\nP_correct: {}\n", x_correct, P_correct);
        println!("\nx_pred: {}\nP_pred: {}\n", ekfstate.x, ekfstate.P);
        println!("\nP_correct.shape() = {:#?}", P_correct.shape());
        println!("\nekf.state.P.shape() = {:#?}", ekfstate.P.shape());
        assert!(ekfstate.x.len() == 5, "x.len() = {}", ekfstate.x.len());
        assert!(ekfstate.x.len() == 5, "x.len() = {}", ekfstate.x.len());
        assert!(
            ekfstate.P.shape() == (5, 5),
            "P.shape() = {:#?}",
            ekfstate.P.shape()
        );
        assert!(x_correct.relative_eq(&ekfstate.x, 1e-5, 1e-5));
        assert!(P_correct.relative_eq(&ekfstate.P, 1e-5, 1e-5));
    }
    #[test]
    fn test_ekf_cv_step() {
        let x = DVector::from_row_slice(&[0., 0., 1., 1., 0.]);
        let P = DMatrix::from_row_slice(
            5,
            5,
            &[
                1000000., 0., 0., 0., 0., 0., 1000000., 0., 0., 0., 0., 0., 900., 0., 0., 0., 0.,
                0., 900., 0., 0., 0., 0., 0., 0.,
            ],
        );
        let z = DVector::from_row_slice(&[1.0, 0.0]);

        let measmod = CartesianPosition::new(SIGMA_Z);
        let dynmod = CV::new(SIGMA_A_CV);

        let ekf = EKF::init(dynmod, measmod);

        let ekfstate = GaussParams::new(x, P);

        let ekfstate = ekf.step(&z, ekfstate, TS);

        let x_correct = DVector::from_row_slice(&[0.99991001, 0.00001, 1.00008099, 0.999991, 0.]);
        let P_correct = DMatrix::from_row_slice(
            5,
            5,
            &[
                99.99000109,
                0.,
                0.00899914,
                0.,
                0.,
                0.,
                99.99000109,
                0.,
                0.00899914,
                0.,
                0.00899914,
                0.,
                900.01690066,
                0.,
                0.,
                0.,
                0.00899914,
                0.,
                900.01690066,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.,
            ],
        );
        assert!(
            ekfstate.P.shape() == (5, 5),
            "P.shape() = {:#?}",
            ekfstate.P.shape()
        );
        assert!(x_correct.relative_eq(&ekfstate.x, 1e-5, 1e-5));
        assert!(P_correct.relative_eq(&ekfstate.P, 1e-5, 1e-5));
    }
}
