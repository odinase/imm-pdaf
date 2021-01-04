#![feature(proc_macro_hygiene)]

use inline_python::{python, Context};
use nalgebra_numpy::matrix_from_numpy;
use nalgebra_numpy::matrix_slice_mut_from_numpy;
use nalgebra_numpy::matrix_slice_from_numpy;
use nalgebra::{DMatrix, Dynamic, U2, U3};


fn read_mat() {
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
    let mut k = 0;
    python! {
    #![context = &context]
        matrix = Z['k]   
    }
    k += 1;
    

    let matrix = context.globals(py).get_item("matrix").unwrap();

    let matrix:  nalgebra::DMatrix<f64> = matrix_from_numpy(py, matrix).unwrap();
    println!("{}", matrix);

    python! {
    #![context = &context]
        matrix = Z['k]   
    }

    let matrix = context.globals(py).get_item("matrix").unwrap();

    let matrix: nalgebra::DMatrix<f64> = matrix_from_numpy(py, matrix).unwrap();
    println!("{}", matrix);

    let K = context.globals(py).get_item("K").unwrap();

    println!("K: {}", K);
	// assert!(matrix == nalgebra::DMatrix::from_row_slice(3, 3, &[
	// 	1.0, 2.0, 3.0,
	// 	4.0, 5.0, 6.0,
	// 	7.0, 8.0, 9.0,
    // ]));
}