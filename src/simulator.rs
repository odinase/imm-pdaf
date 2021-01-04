use matfile::{NumericData, MatFile};
use matfile;
use nalgebra::{DMatrix, DVector};
use crate::{
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


pub fn read_var_from_mat(mat_file: &MatFile, var_name: &str) -> Option<DMatrix<f64>> {
    if let Some(var_data) = mat_file.find_by_name(var_name) {
        let (r, c) = (var_data.size()[0], var_data.size()[1]);
        let arr = var_data.data();
        if let NumericData::Double{real: d, imag: _} = arr {
            Some(DMatrix::from_vec(r, c, d.to_vec()))
        } else  {
            None
        }
    } else {
        None
    }
}


pub fn run_ekf() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open("data/data_for_ekf.mat")?;
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
    
    
    for (k, (xgt, z)) in izip!(
        Xgt.column_iter(),
        Z.column_iter().map(|z| z.clone_owned()) // Unfortunately, this is the only way of doing this
    ).enumerate() {
        let ekfpred = filter.predict(ekfupd, Ts);
        ekfupd = filter.update(&z, ekfpred);
    }

    println!("Final state:\nx: {}\nP: {}", ekfupd.x, ekfupd.P);

    Ok(())
}