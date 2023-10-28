use crate::{
    data_parsing::{read_dataset_from_json, Timestep},
    mixture::MixtureParameters,
    pdaf::PDAF,
    plotting, simulator as sim,
    state_estimator::{
        ekf, imm,
        models::{DynamicModel, MeasurementModel},
        StateEstimator,
    },
};
use itertools::Itertools as _;
use nalgebra::{dvector, DMatrix, DVector};
use std::time::Instant;
use tokio::sync::mpsc::Sender;

pub fn run_pdaf() -> anyhow::Result<()> {
    let pdaf_path = "json/data_for_pda.json";

    println!("Reading data from file...");
    let dataread_start = Instant::now();
    let timesteps = read_dataset_from_json(pdaf_path)?;
    println!("Done! Spent {} s", dataread_start.elapsed().as_secs_f64());
    // We know that all Ts are equal for this dataset, use first Ts
    let Ts = timesteps[0].Ts;

    let sigma_z: f64 = 3.2;
    let sigma_a = 2.2;

    let dynmod = DynamicModel::cv(sigma_a);
    let measmod = MeasurementModel::cartesian_position(sigma_z);

    let filter = ekf::EKF::init(dynmod, measmod);

    let x0 = DVector::from_row_slice(&[31.52499977, -76.76457537, 0., 0.]);
    let pn = 2;
    let vn = 2;
    let n = pn + vn;
    let cov11 = sigma_z.powi(2) * DMatrix::identity(pn, pn) * 500.0;
    let cov12: DMatrix<f64> = sigma_z.powi(2) * DMatrix::identity(pn, pn) / Ts;
    let cov22 = (2.0 * (sigma_z / Ts).powi(2) + sigma_a.powi(2) * Ts / 3.0)
        * DMatrix::identity(vn, vn)
        * 100.0;

    let mut P0 = DMatrix::zeros(n, n);

    P0.index_mut((..2, ..2)).copy_from(&cov11);
    P0.index_mut((0..2, 2..)).copy_from(&cov12);
    P0.index_mut((2.., ..2)).copy_from(&cov12.transpose());
    P0.index_mut((2.., 2..)).copy_from(&cov22);
    let P0 = DMatrix::from_row_slice(
        n,
        n,
        &[
            20.48, 0., 0., 0., 0., 20.48, 0., 0., 0., 0., 100., 0., 0., 0., 0., 100.,
        ],
    );

    let mut ekfupd = ekf::GaussParams::new(x0, P0);
    let clutter_intensity = 1e-3;
    let PD = 0.8;
    let gate_size = 5.0;

    let tracker = PDAF::init(filter, clutter_intensity, PD, gate_size);
    let K = timesteps.len();

    let mut states = Vec::with_capacity(K);
    let mut Xgts = Vec::with_capacity(K);

    println!("Starting simulation...");
    let start = Instant::now();
    for Timestep { Ts, Z, Xgt, .. } in timesteps {
        let Z: Vec<_> = Z
            .into_iter()
            .tuples()
            .map(|(zx, zy)| dvector![zx, zy])
            .collect();
        let ekfpred = tracker.predict(ekfupd, Ts);
        ekfupd = tracker.update(Z, ekfpred);
        states.push(ekfupd.clone());
        Xgts.push(DVector::from_vec(Xgt));
    }
    let duration = start.elapsed();
    println!("Done! Spent {} s in simulation", duration.as_secs_f64());

    plotting::plot_states("Simulated PDAF", states, Some(Xgts)).unwrap();

    Ok(())
}

pub fn run_imm_pdaf() -> anyhow::Result<()> {
    let imm_path = "json/data_for_imm_pda.json";

    println!("Reading data from file...");
    let dataread_start = Instant::now();
    let timesteps = read_dataset_from_json(imm_path)?;
    println!("Done! Spent {} s", dataread_start.elapsed().as_secs_f64());
    // We know that all Ts are equal for this dataset, use first Ts
    let Ts = timesteps[0].Ts;
    let K = timesteps.len();

    let sigma_z: f64 = 2.84;
    let sigma_a_cv = 0.14;
    let sigma_a_ct = 0.04;
    let sigma_w = 0.01;

    let PI = DMatrix::from_row_slice(2, 2, &[0.92, 0.08, 0.1, 0.9]);

    let dynmod_cv = DynamicModel::CV {
        sigma_a: sigma_a_cv,
    };
    let measmod_cv = MeasurementModel::CartesianPosition { sigma_z };

    let dynmod_ct = DynamicModel::CT {
        sigma_a: sigma_a_ct,
        sigma_w,
    };
    let measmod_ct = measmod_cv.clone();

    let ekf_cv = ekf::EKF::init(dynmod_cv, measmod_cv);
    let ekf_ct = ekf::EKF::init(dynmod_ct, measmod_ct);

    let filters = vec![ekf_cv, ekf_ct];

    let imm_filter = imm::IMM::init(filters, PI);

    let x0 = DVector::from_row_slice(&[2.47, 24.7, 0., 0., 0.]);

    let P0 = DMatrix::from_partial_diagonal(
        5,
        5,
        &[
            sigma_w * sigma_w,
            sigma_w * sigma_w,
            5.6 * 5.6,
            5.6 * 5.6,
            0.1 * 0.1,
        ],
    );

    let ekf_cv_init = ekf::GaussParams::new(x0, P0);
    let ekf_ct_init = ekf_cv_init.clone();

    let init_components = vec![ekf_cv_init, ekf_ct_init];
    let init_weights = vec![0.9, 0.1];

    let mut immstate_upd = MixtureParameters::new(init_weights, init_components);

    let clutter_intensity = 1e-3;
    let PD = 0.92;
    let gate_size = 3.0;

    let tracker = PDAF::init(imm_filter, clutter_intensity, PD, gate_size);

    let mut states = Vec::with_capacity(K);
    let mut Xgts = Vec::with_capacity(K);

    let start = Instant::now();
    for Timestep { Ts, Z, Xgt, .. } in timesteps {
        let Z: Vec<_> = Z
            .into_iter()
            .tuples()
            .map(|(zx, zy)| dvector![zx, zy])
            .collect();
        let immstate_pred = tracker.predict(immstate_upd, Ts);
        immstate_upd = tracker.update(Z, immstate_pred);
        let estimate = tracker.estimate(immstate_upd.clone());
        states.push(estimate);
        Xgts.push(DVector::from_vec(Xgt));
        
    }
    let duration = start.elapsed();
    println!("Time elapsed in sim is: {:?}", duration);

    plotting::plot_states("Simulated IMM-PDAF", states, Some(Xgts)).unwrap();

    Ok(())
}

pub fn run_joyride() -> anyhow::Result<()> {
    let imm_path = "json/data_joyride.json";

    println!("Reading data from file...");
    let dataread_start = Instant::now();
    let timesteps = read_dataset_from_json(imm_path)?;
    println!("Done! Spent {} s", dataread_start.elapsed().as_secs_f64());
    // We know that all Ts are equal for this dataset, use first Ts
    let Ts = timesteps[0].Ts;
    let K = timesteps.len();

    let sigma_z: f64 = 10.0;
    let sigma_a_cv = 4.0;
    let sigma_a_ct = 8.0;
    let sigma_w = 0.25;

    let pi_matrix = DMatrix::from_row_slice(2, 2, &[0.92, 0.08, 0.1, 0.9]);

    let dynmod_cv = DynamicModel::CV {
        sigma_a: sigma_a_cv,
    };
    let measmod_cv = MeasurementModel::CartesianPosition { sigma_z };

    let dynmod_ct = DynamicModel::CT {
        sigma_a: sigma_a_ct,
        sigma_w,
    };
    let measmod_ct = measmod_cv.clone();

    let ekf_cv = ekf::EKF::init(dynmod_cv, measmod_cv);
    let ekf_ct = ekf::EKF::init(dynmod_ct, measmod_ct);

    let filters = vec![ekf_cv, ekf_ct];

    let imm_filter = imm::IMM::init(filters, pi_matrix);

    let x0 = DVector::from_row_slice(&[7100.0, 3630.0, -6.0, -3.0, 0.0]);

    use std::f64::consts::PI;
    let P0 = 4.0*DMatrix::from_partial_diagonal(
        5,
        5,
        &[
            10.0 * 10.0,
            10.0 * 10.0,
            1.0,
            1.0,
            PI / 10.0 * PI / 10.0,
        ],
    );

    let ekf_cv_init = ekf::GaussParams::new(x0, P0);
    let ekf_ct_init = ekf_cv_init.clone();

    let init_components = vec![ekf_cv_init, ekf_ct_init];
    let init_weights = vec![0.9, 0.1];

    let mut immstate_upd = MixtureParameters::new(init_weights, init_components);

    let clutter_intensity = 5.006470e-06;
    let PD = 0.95;
    let gate_size = 3.0;

    let tracker = PDAF::init(imm_filter, clutter_intensity, PD, gate_size);

    let mut states = Vec::with_capacity(K);
    let mut Xgts = Vec::with_capacity(K);

    let start = Instant::now();
    for Timestep { Ts, Z, Xgt, .. } in timesteps {
        let Z: Vec<_> = Z
            .into_iter()
            .tuples()
            .map(|(zx, zy)| dvector![zx, zy])
            .collect();
        let immstate_pred = tracker.predict(immstate_upd, Ts);
        immstate_upd = tracker.update(Z, immstate_pred);
        let estimate = tracker.estimate(immstate_upd.clone());
        states.push(estimate);
        Xgts.push(DVector::from_vec(Xgt));
        
    }
    let duration = start.elapsed();
    println!("Time elapsed in sim is: {:?}", duration);

    plotting::plot_states("Joyride trajectory", states, Some(Xgts)).unwrap();

    Ok(())
}

// pub fn run_ekf() -> Result<(), Box<dyn std::error::Error>> {
//     let file = std::fs::File::open("data/data_for_ekf.mat")?;
//     let mat_file = matfile::MatFile::parse(file)?;
//     let K = sim::read_var_from_mat(&mat_file, "K").unwrap()[0]; // Scalar
//     let Ts = sim::read_var_from_mat(&mat_file, "Ts").unwrap()[0]; // Scalar
//     let Xgt = sim::read_var_from_mat(&mat_file, "Xgt").unwrap(); // 5 x K
//     let Z = sim::read_var_from_mat(&mat_file, "Z").unwrap(); // 2 x K

//     let sigma_z = 3.1;
//     let sigma_a = 2.6;

//     let dynmod = DynamicModel::cv(sigma_a);
//     let measmod = MeasurementModel::cartesian_position(sigma_z);

//     let filter = ekf::EKF::init(dynmod, measmod);

//     let x0 = DVector::from_row_slice(&[
//         Z[(0, 1)],
//         Z[(1, 1)],
//         (Z[(0, 1)] - Z[(0, 0)]) / Ts,
//         (Z[(1, 1)] - Z[(1, 0)]) / Ts,
//     ]);
//     let pn = 2;
//     let vn = 2;
//     let n = pn + vn;
//     let cov11 = sigma_z.powi(2) * DMatrix::identity(pn, pn);
//     let cov12 = sigma_z.powi(2) * DMatrix::identity(pn, pn) / Ts;
//     let cov22 =
//         (2.0 * (sigma_z / Ts).powi(2) + sigma_a.powi(2) * Ts / 3.0) * DMatrix::identity(vn, vn);

//     let mut P0 = DMatrix::zeros(n, n);

//     P0.index_mut((..2, ..2)).copy_from(&cov11);
//     P0.index_mut((0..2, 2..)).copy_from(&cov12);
//     P0.index_mut((2.., ..2)).copy_from(&cov12.transpose());
//     P0.index_mut((2.., 2..)).copy_from(&cov22);

//     let mut ekfupd = ekf::GaussParams::new(x0, P0);
//     let mut state = Vec::with_capacity(K as usize);

//     let start = Instant::now();
//     for (k, (xgt, z)) in izip!(
//         Xgt.column_iter(),
//         Z.column_iter().map(|z| z.clone_owned()) // Unfortunately, this is the only way of doing this
//     )
//     .enumerate()
//     {
//         let ekfpred = filter.predict(&ekfupd, Ts);
//         ekfupd = filter.update(&z, &ekfpred);
//         state.push(ekfupd.clone());
//     }
//     let duration = start.elapsed();
//     println!("Time elapsed in sim is: {:?}", duration);

//     plotting::plot_states(state.as_slice(), Some(&Xgt));

//     Ok(())
// }
