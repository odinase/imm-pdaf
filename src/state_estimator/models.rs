use nalgebra::{DMatrix, DVector, U2};

#[derive(Debug, Clone)]
pub enum DynamicModel {
    CV(f64),
    CT(f64, f64),
}

#[derive(Debug, Clone)]
pub enum MeasurementModel {
    CartesianPosition(f64),
}

impl MeasurementModel {
    pub fn cartesian_position(sigma_z: f64) -> Self {
        Self::CartesianPosition(sigma_z)
    }
    /// Assumes p is the first state
    pub fn h(&self, x: &DVector<f64>) -> DVector<f64> {
        match self {
            Self::CartesianPosition(_) => {
                let mut p = DVector::zeros(2);
                p.copy_from(&x.rows(0, 2));
                p
            }
        }
    }

    pub fn H(&self, x: &DVector<f64>) -> DMatrix<f64> {
        match self {
            Self::CartesianPosition(_) => {
                let n = x.len();
                let H = DMatrix::identity(2, n);
                H
            }
        }
    }
    pub fn R(&self, x: &DVector<f64>, z: &DVector<f64>) -> DMatrix<f64> {
        match self {
            Self::CartesianPosition(sigma_z) => {
                let n = z.len();
                let R = DMatrix::<f64>::identity(n, n) * sigma_z.powi(2);
                R
            }
        }
    }
}

impl DynamicModel {
    pub fn cv(sigma_a: f64) -> Self {
        Self::CV(sigma_a)
    }

    pub fn ct(sigma_a: f64, sigma_w: f64) -> Self {
        Self::CT(sigma_a, sigma_w)
    }
    
    pub fn f(&self, x: &DVector<f64>, ts: f64) -> DVector<f64> {
        match self {
            Self::CV(_) => {
                let n = x.len();
                let mut x_next = DVector::zeros(n);
                let p = x.fixed_rows::<U2>(0);
                let u = x.fixed_rows::<U2>(2);
                x_next.fixed_rows_mut::<U2>(0).copy_from(&(p + ts * u));
                x_next.fixed_rows_mut::<U2>(2).copy_from(&(u));
                x_next
            }
            Self::CT(_, _) => {
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
        }
    }
    pub fn F(&self, x: &DVector<f64>, ts: f64) -> DMatrix<f64> {
        match self {
            Self::CV(_) => {
                let n = x.len();
                let mut F = DMatrix::<f64>::identity(n, n);
                F.slice_mut((0, 2), (2, 2))
                    .copy_from(&(DMatrix::identity(2, 2) * ts));
                // In case of IMM, where F is 5 x 5
                if n == 5 {
                    F[(4,4)] = 0.0;
                }
                F
            }
            Self::CT(_, _) => {
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
                        1.,
                        0.,
                        ts * sincth,
                        -ts * coscth,
                        ts.powi(2) * (u0 * dsincth - v0 * dcoscth),
                        0.,
                        1.,
                        ts * coscth,
                        ts * sincth,
                        ts.powi(2) * (u0 * dcoscth + v0 * dsincth),
                        0.,
                        0.,
                        cth,
                        -sth,
                        -ts * (u0 * sth + v0 * cth),
                        0.,
                        0.,
                        sth,
                        cth,
                        ts * (u0 * cth - v0 * sth),
                        0.,
                        0.,
                        0.,
                        0.,
                        1.,
                    ],
                )
            }
        }
    }
    pub fn Q(&self, x: &DVector<f64>, ts: f64) -> DMatrix<f64> {
        match self {
            Self::CV(sigma_a) => {
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
                Q *= sigma_a.powi(2);
                Q
            }
            Self::CT(sigma_a, sigma_w) => {
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
                Q *= sigma_a.powi(2);
                Q[(4, 4)] = ts * sigma_w.powi(2);
                Q
            }
        }
    }
}


// Helpers for computing Q in CT

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
