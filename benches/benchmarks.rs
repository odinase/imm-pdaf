use imm_pdaf::simulator::run_imm;
use inline_python::{python, Context};
use nalgebra_numpy::matrix_from_numpy;
use nalgebra::{DMatrix, DVector};
use pyo3::ObjectProtocol;
use matfile::{NumericData, MatFile};
use matfile;
use imm_pdaf::{
    simulator as sim,
    state_estimator::{
        StateEstimator,
        models::{DynamicModel, MeasurementModel},
        ekf,
        imm
    },
    pdaf::PDAF,
    plotting,
    mixture::{
        MixtureParameters
    }
};
use itertools::izip;
use std::time::{Duration, Instant};


// fn criterion_benchmark(c: &mut Criterion) {
    //     c.bench_function("run_imm", |b| b.iter(|| run_imm()));
    //     // c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
    // }
    
    
    use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
fn criterion_benchmark(c: &mut Criterion) {

    let gil = pyo3::Python::acquire_gil();
	let py = gil.python();
    let context = Context::new_with_gil(py).unwrap();
    
	python! {
        #![context = &context]
        from scipy.io import loadmat
        import numpy as np

        data_file_name = "data/data_for_imm_pda.mat"
        loaded_data = loadmat(data_file_name)
        K = loaded_data["K"].item()
        Ts = loaded_data["Ts"].item()
        Xgt = loaded_data["Xgt"]
        Z = [zk.T for zk in loaded_data["Z"].ravel()]
        true_association = loaded_data["a"].ravel()
    }

    let Xgt_numpy = context.globals(py).get_item("Xgt").unwrap();

    let Xgt: nalgebra::DMatrix<f64> = matrix_from_numpy(py, Xgt_numpy).unwrap();
    let K: i32 = context.globals(py).get_item("K").unwrap().extract::<f64>().unwrap() as i32;
    let Ts: f64 = context.globals(py).get_item("Ts").unwrap().extract().unwrap();

    let sigma_z: f64 = 2.84;
    let sigma_a_cv = 0.14;
    let sigma_a_ct = 0.04;
    let sigma_w = 0.01;

    let PI = DMatrix::from_row_slice(2, 2, &[
        0.92, 0.08,
        0.1, 0.9
    ]);

    let dynmod_cv = DynamicModel::CV(sigma_a_cv);
    let measmod_cv = MeasurementModel::CartesianPosition(sigma_z);
    
    let dynmod_ct = DynamicModel::CT(sigma_a_ct, sigma_w);
    let measmod_ct = measmod_cv.clone();

    let ekf_cv = ekf::EKF::init(dynmod_cv, measmod_cv);
    let ekf_ct = ekf::EKF::init(dynmod_ct, measmod_ct);

    let filters = vec![ekf_cv, ekf_ct];

    let imm_filter = imm::IMM::init(filters, PI);
    
    let x0 = DVector::from_row_slice(&[
        2.47, 24.7,   0.,   0., 0.
    ]);

    let P0 = DMatrix::from_row_slice(5, 5, &[
     sigma_w*sigma_w,   0.  ,   0.  ,   0.  , 0.,
          0.  ,  sigma_w*sigma_w,   0.  ,   0. , 0.,
          0.  ,   0.  , 5.6*5.6  ,   0. , 0.,
          0.  ,   0.  ,   0.  , 5.6*5.6, 0.,
          0., 0., 0., 0., 0.1*0.1 ]);
    
    let ekf_cv_init = ekf::GaussParams::new(x0, P0);
    let ekf_ct_init = ekf_cv_init.clone();

    let init_components = vec![ekf_cv_init, ekf_ct_init];
    let init_weights = vec![0.9, 0.1];

    let mut immstate_upd = MixtureParameters::new(init_weights, init_components);
    let immstate_upd_clone = immstate_upd.clone();

    let clutter_intensity = 1e-3;
    let PD = 0.92;
    let gate_size = 3.0;

    let tracker = PDAF::init(imm_filter, clutter_intensity, PD, gate_size);
    // let mut state = Vec::with_capacity(K as usize);

    python! {
        #![context = &context]
        zk = Z[0]
    }

    c.bench_function("read Z", |b| b.iter( || {   
        let zk = context.globals(py).get_item("zk").unwrap();
        let zk: DMatrix<f64> = matrix_from_numpy(py, zk).unwrap();
        let Z = zk.row_iter().map(|z| z.clone_owned().transpose()).collect::<Vec<_>>();
    }));

    let zk = context.globals(py).get_item("zk").unwrap();
    let zk: DMatrix<f64> = matrix_from_numpy(py, zk).unwrap();
    let Z = zk.row_iter().map(|z| z.clone_owned().transpose()).collect::<Vec<_>>();
    
    c.bench_with_input(
        BenchmarkId::new("predict", &immstate_upd),
        &immstate_upd,
        |b, immstate_upd| b.iter( || {
        tracker.predict(immstate_upd.clone(), Ts);
    }));

    let immstate_pred = tracker.predict(immstate_upd_clone, Ts);

    c.bench_with_input(
        BenchmarkId::new("predict", &immstate_pred),
        &immstate_pred,
        |b, immstate_pred| b.iter( || {
            tracker.update(Z.clone(), immstate_pred.clone());
    }));

}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);