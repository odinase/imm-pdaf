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
    let file = std::fs::File::open("data/data_for_pda.mat")?;
    let mat_file = matfile::MatFile::parse(file)?;
    let K = sim::read_var_from_mat(&mat_file, "K").unwrap()[0]; // Scalar
    let Ts = sim::read_var_from_mat(&mat_file, "Ts").unwrap()[0]; // Scalar
    let Xgt = sim::read_var_from_mat(&mat_file, "Xgt").unwrap(); // 5 x K
    let Z = sim::read_var_from_mat(&mat_file, "Z").unwrap(); // 2 x K


    let sigma_z = 3.1;
    let sigma_a = 2.6;

    let dynmod = CV::new(sigma_a);
    let measmod = CartesianPosition::new(sigma_z);
    
    let filter = ekf::EKF::init(dynmod, measmod);
    
    let x0 = DVector::from_row_slice(&[
        Z[(0,1)], Z[(1,1)], (Z[(0,1)] - Z[(0,0)]) / Ts, (Z[(1,1)] - Z[(1,0)]) / Ts 
    ]);
    let pn = 2;
    let vn = 2;
    let n = pn + vn;
    let cov11 = sigma_z.powi(2) * DMatrix::identity(pn, pn);
    let cov12 = sigma_z.powi(2) * DMatrix::identity(pn, pn) / Ts;
    let cov22 = (2.0 * (sigma_z / Ts).powi(2) + sigma_a.powi(2) * Ts / 3.0) * DMatrix::identity(vn, vn);
    
    let mut P0 = DMatrix::zeros(n, n);
    
    P0.index_mut((..2, ..2)).copy_from(&cov11);
    P0.index_mut((0..2, 2..)).copy_from(&cov12);
    P0.index_mut((2.., ..2)).copy_from(&cov12.transpose());
    P0.index_mut((2.., 2..)).copy_from(&cov22);
    
    let mut ekfupd = ekf::GaussParams::new(x0, P0);
    let mut state = Vec::with_capacity(K as usize);
    
    for (k, (xgt, z)) in izip!(
        Xgt.column_iter(),
        Z.column_iter().map(|z| z.clone_owned()) // Unfortunately, this is the only way of doing this
    ).enumerate() {
        let ekfpred = filter.predict(ekfupd, Ts);
        ekfupd = filter.update(&z, ekfpred);
        state.push(ekfupd.clone());
    }

    let mut fg = Figure::new();
    fg.axes2d()
        // .set_y_range(Fix, Fix(1.5))
        // .set_x_range(Fix(-1.5), Fix(1.5))
        .lines(
            state.iter().map(|s| s.x[0]),
            state.iter().map(|s| s.x[1]),
            &[Caption("Estimate")/*, PointSymbol('x')*/],
        )
        .lines(
            Xgt.column_iter().map(|xgt| xgt[0]),
            Xgt.column_iter().map(|xgt| xgt[1]),
            &[Caption("Ground truth")/*, PointSymbol('x')*/],
        );
    fg.show().unwrap();

    Ok(())
}
