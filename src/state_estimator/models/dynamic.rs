use super::DynamicModel;
use super::MeasurementModel;
use nalgebra::{DMatrix, DVector, Matrix4, Vector4, U2};

#[derive(Debug, Clone)]
pub struct CV {
    // Acceleration noise covariance
    sigma_a: f64,
}

impl CV {
    pub fn new(sigma_a: f64) -> Self {
        CV { sigma_a }
    }
}

impl DynamicModel for CV {
    type State = DVector<f64>;
    type Covariance = DMatrix<f64>;
    type Jacobian = DMatrix<f64>;
    fn f(&self, x: &Self::State, ts: f64) -> Self::State {
        let n = x.len();
        let mut x_next = DVector::zeros(n);
        let p = x.fixed_rows::<U2>(0);
        let u = x.fixed_rows::<U2>(2);
        x_next.fixed_rows_mut::<U2>(0).copy_from(&(p + ts * u));
        x_next.fixed_rows_mut::<U2>(2).copy_from(&(u));
        x_next
    }
    fn F(&self, x: &Self::State, ts: f64) -> Self::Jacobian {
        let n = x.len();
        let mut F = DMatrix::<f64>::identity(n, n);
        F.slice_mut((0, 2), (2, 2))
            .copy_from(&(DMatrix::identity(2, 2) * ts));
        F
    }
    fn Q(&self, x: &Self::State, ts: f64) -> Self::Covariance {
        let n = x.len();
        let mut Q = DMatrix::zeros(n, n);
        Q.slice_mut((0, 0), (2, 2))
            .copy_from(&DMatrix::<f64>::from_diagonal_element(
                2,
                2,
                ts.powi(3) / 3.0,
            ));
        Q.slice_mut((2, 2), (2, 2))
            .copy_from(&DMatrix::<f64>::from_diagonal_element(2, 2, ts));
        Q.slice_mut((0, 2), (2, 2))
            .copy_from(&DMatrix::<f64>::from_diagonal_element(
                2,
                2,
                ts.powi(2) / 2.0,
            ));
        Q.slice_mut((2, 0), (2, 2))
            .copy_from(&DMatrix::<f64>::from_diagonal_element(
                2,
                2,
                ts.powi(2) / 2.0,
            ));
        Q *= self.sigma_a.powi(2);
        Q
    }
}

#[derive(Debug, Clone)]
pub struct CT {
    sigma_a: f64,
    sigma_w: f64,
}

impl CT {
    pub fn new(sigma_a: f64, sigma_w: f64) -> Self {
        CT { sigma_a, sigma_w }
    }
}

// Computes sin(x)/x
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-3 {
        1.0 - x.powi(2) / 6.0
    } else {
        x.sin() / x
    }
}

// Computes (1 - cos(x))/x
fn cosc(x: f64) -> f64 {
    if x.abs() < 1e-3 {
        x / 2.0 - x.powi(3) / 24.0
    } else {
        (1.0 - x.cos()) / x
    }
}

fn dsinc(x: f64) -> f64 {
    if x.abs() < 1e-3 {
        -x / 3.0
    } else {
        (x.cos() - sinc(x)) / x
    }
}

fn dcosc(x: f64) -> f64 {
    if x.abs() < 1e-3 {
        0.5 - x.powi(2) / 6.0
    } else {
        (x.sin() - cosc(x)) / x
    }
}

impl DynamicModel for CT {
    type State = DVector<f64>;
    type Jacobian = DMatrix<f64>;
    type Covariance = DMatrix<f64>;
    fn f(&self, x: &Self::State, ts: f64) -> Self::State {
        let x0 = x[0];
        let y0 = x[1];
        let u0 = x[2];
        let v0 = x[3];
        let omega = x[4];

        let theta = omega * ts;

        let cth = theta.cos();
        let sth = theta.sin();

        let sincth = sinc(theta);
        let coscth = cosc(theta);

        DVector::from_row_slice(&[
            x0 + ts * u0 * sincth - ts * v0 * coscth,
            y0 + ts * u0 * coscth + ts * v0 * sincth,
            u0 * cth - v0 * sth,
            u0 * sth + v0 * cth,
            omega,
        ])
    }

    fn F(&self, x: &Self::State, ts: f64) -> Self::Jacobian {
        let x0 = x[0];
        let y0 = x[1];
        let u0 = x[2];
        let v0 = x[3];
        let omega = x[4];

        let theta = ts * omega;

        let sth = theta.sin();
        let cth = theta.cos();

        let sincth = sinc(theta);
        let coscth = cosc(theta);

        let dsincth = dsinc(theta);
        let dcoscth = dcosc(theta);

        DMatrix::<f64>::from_row_slice(
            5,
            5,
            &[
                1.,                0.,                ts * sincth,                -ts * coscth,                ts.powi(2) * (u0 * dsincth - v0 * dcoscth),
                0.,                1.,                ts * coscth,                ts * sincth,                ts.powi(2) * (u0 * dcoscth + v0 * dsincth),
                0.,                0.,                cth,                -sth,                -ts * (u0 * sth + v0 * cth),
                0.,                0.,                sth,                cth,                ts * (u0 * cth - v0 * sth),
                0.,                0.,                0.,                0.,                1.,
            ],
        )
    }

    fn Q(&self, _x: &Self::State, ts: f64) -> Self::Covariance {
        let mut Q = DMatrix::zeros(5, 5);
        Q.slice_mut((0, 0), (2, 2))
            .copy_from(&DMatrix::<f64>::from_diagonal_element(
                2,
                2,
                ts.powi(3) / 3.0,
            ));
        Q.slice_mut((2, 2), (2, 2))
            .copy_from(&DMatrix::<f64>::from_diagonal_element(2, 2, ts));
        Q.slice_mut((0, 2), (2, 2))
            .copy_from(&DMatrix::<f64>::from_diagonal_element(
                2,
                2,
                ts.powi(2) / 2.0,
            ));
        Q.slice_mut((2, 0), (2, 2))
            .copy_from(&DMatrix::<f64>::from_diagonal_element(
                2,
                2,
                ts.powi(2) / 2.0,
            ));
        Q *= self.sigma_a.powi(2);
        Q[(4, 4)] = ts * self.sigma_w.powi(2);
        Q
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_CV_f() {
        let sigma_a = 0.01;
        let ts = 1.0;
        let x = DVector::from_row_slice(&[1., 2., 3., 4.]);
        let cv = CV::new(sigma_a);
        let x_correct = DVector::from_row_slice(&[4., 6., 3., 4.]);
        let x_next = cv.f(&x, ts);
        println!("x_next: {}", x_next);
        assert!(x_correct.relative_eq(&x_next, 1e-5, 1e-5));
    }

    #[test]
    fn test_CV_F() {
        let sigma_a = 0.01;
        let ts = 1.0;
        let x = DVector::from_row_slice(&[1., 2., 3., 4.]);
        let cv = CV::new(sigma_a);
        let F_correct = DMatrix::from_row_slice(
            4,
            4,
            &[
                1., 0., ts, 0., 0., 1., 0., ts, 0., 0., 1., 0., 0., 0., 0., 1.,
            ],
        );
        let F_test = cv.F(&x, ts);
        assert!(F_correct.relative_eq(&F_test, 1e-5, 1e-5));
    }

    #[test]
    fn test_CV_Q() {
        let sigma_a = 0.01;
        let ts = 1.0;
        let x = DVector::from_row_slice(&[1., 2., 3., 4.]);
        let cv = CV::new(sigma_a);
        let Q_correct = DMatrix::<f64>::from_row_slice(
            4,
            4,
            &[
                1. / 3.,
                0.,
                0.5000,
                0.,
                0.,
                1. / 3.,
                0.,
                0.5000,
                0.5000,
                0.,
                1.0000,
                0.,
                0.,
                0.5000,
                0.,
                1.0000,
            ],
        ) * sigma_a
            * sigma_a;
        let Q_test = cv.Q(&x, ts);
        assert!(Q_correct.relative_eq(&Q_test, 1e-5, 1e-5));
    }

    #[test]
    fn test_CT_f() {
        let sigma_a = 0.01;
        let sigma_w = 0.005;
        let ts = 1.0;
        let x = DVector::from_row_slice(&[
            0.814723686393179,
            0.905791937075619,
            0.126986816293506,
            0.913375856139019,
            0.123
        ]);
        let ct = CT::new(sigma_a, sigma_w);
        let x_correct = DVector::from_row_slice(&[
            0.885288716322700,
            1.824666305629147,
            0.013965268961422,
            0.922055354820350,
            0.123000000000000
        ]);
        let x_next = ct.f(&x, ts);
        assert!(x_correct.relative_eq(&x_next, 1e-5, 1e-5));
    }

    #[test]
    fn test_CT_F() {
        let sigma_a = 0.01;
        let sigma_w = 0.005;
        let ts = 1.0;
        let x = DVector::from_row_slice(&[
            0.814723686393179,
            0.905791937075619,
            0.126986816293506,
            0.913375856139019,
            0.123
        ]);
        let ct = CT::new(sigma_a, sigma_w);
        let F_correct = DMatrix::from_row_slice(
            5,
            5,
            &[
                 1.        ,  0.        ,  0.99748041, -0.0614225 , -0.46016066,
        0.        ,  1.        ,  0.0614225 ,  0.99748041,  0.02586168,
        0.        ,  0.        ,  0.99244503, -0.12269009, -0.92205535,
        0.        ,  0.        ,  0.12269009,  0.99244503,  0.01396527,
        0.        ,  0.        ,  0.        ,  0.        ,  1.        
        ]);
        let F_test = ct.F(&x, ts);
        assert!(F_correct.relative_eq(&F_test, 1e-5, 1e-5));
    }

    #[test]
    fn test_CT_Q() {
        let sigma_a = 0.01;
        let sigma_w = 0.005;
        let ts = 1.0;
        let x = DVector::from_row_slice(&[
            0.814723686393179,
            0.905791937075619,
            0.126986816293506,
            0.913375856139019,
            0.123
        ]);
        let ct = CT::new(sigma_a, sigma_w);
        let Q_correct = DMatrix::<f64>::from_row_slice(
            5,
            5,
            &[
                0.00003333, 0.        , 0.00005   , 0.        , 0.,
       0.        , 0.00003333, 0.        , 0.00005   , 0.        ,
       0.00005   , 0.        , 0.0001    , 0.        , 0.        ,
       0.        , 0.00005   , 0.        , 0.0001    , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.000025  
        ]);
        let Q_test = ct.Q(&x, ts);
        assert!(Q_correct.relative_eq(&Q_test, 1e-5, 1e-5));
    }
}
