#![feature(proc_macro_hygiene)]
use inline_python::{python, Context};
use nalgebra_numpy::matrix_from_numpy;
use nalgebra_numpy::matrix_slice_mut_from_numpy;
use nalgebra_numpy::matrix_slice_from_numpy;
use nalgebra::{DMatrix, DVector};
use pyo3::GILGuard;
use matfile::{NumericData, MatFile};
use matfile;
use crate::{
    simulator as sim,
    state_estimator::{
        StateEstimator,
        models::{
            dynamic::CV,
            measurement::CartesianPosition
        },
        ekf
    },
    pdaf::PDAF,
};
use itertools::izip;
use gnuplot::*;

pub fn run_pdaf() -> Result<(), Box<dyn std::error::Error>> {
    let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
    let context = Context::new_with_gil(py).unwrap();
    
	python! {
		#![context = &context]
        from scipy.io import loadmat
        import numpy as np

        data_file_name = "data_for_pda.mat"
        loaded_data = loadmat(data_file_name)
        K = loaded_data["K"].item()
        Ts = loaded_data["Ts"].item()
        Xgt = loaded_data["Xgt"].T
        Z = [zk.T for zk in loaded_data["Z"].ravel()]
        true_association = loaded_data["a"].ravel()
    }

    let Xgt_numpy = context.globals(py).get_item("Xgt").unwrap();

    let Xgt: nalgebra::DMatrix<f64> = matrix_from_numpy(py, Xgt_numpy).unwrap();
    let K = context.globals(py).get_item("K").unwrap();
    let ts = context.globals(py).get_item("Ts").unwrap();

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
    
    let ekfupd = ekf::GaussParams::new(x0, P0);
    let clutter_intensity = 1e-3;
    let PD = 0.8;
    let gate_size = 5;

    let tracker = PDAF::init(filter, clutter_intensity, PD, gate_size);

    for k in 0..K {
        python! {
            #![context = &context]
                zk = Z['k]
            }
        
        let zk = context.globals(py).get_item("zk").unwrap();
        let zk: DMatrix<f64> = matrix_from_numpy(py, zk).unwrap();
        let Z = A.row_iter().map(|z| z.clone_owned()).collect::<Vec<_>>();

        let ekfpred = tracker.predict(ekfupd, Ts);
        ekfupd = tracker.update(&Z, filter_state);
    }
    
    Ok(())
}

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
