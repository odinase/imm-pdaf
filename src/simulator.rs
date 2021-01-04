#![feature(proc_macro_hygiene)]
use inline_python::{python, Context};
use nalgebra_numpy::matrix_from_numpy;
use nalgebra_numpy::matrix_slice_mut_from_numpy;
use nalgebra_numpy::matrix_slice_from_numpy;
use nalgebra::{DMatrix, DVector};
use pyo3::GILGuard;
use matfile::{NumericData, MatFile};

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

