#![allow(non_snake_case)]
use matfile;
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


fn main() -> Result<(), Box<dyn std::error::Error>> {
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

/*
# %% a: tune by hand and comment

# set parameters
sigma_a = 2.6
sigma_z = 3.1

# create the model and estimator object
dynmod = dynamicmodels.WhitenoiseAccelleration(sigma_a)
measmod = measurementmodels.CartesianPosition(sigma_z)
ekf_filter = ekf.EKF(dynmod, measmod)
print(ekf_filter)

# Optimal init for model
mean = np.array([*Z[1], *(Z[1] - Z[0]) / Ts])
cov11 = sigma_z ** 2 * np.eye(2)
cov12 = sigma_z ** 2 * np.eye(2) / Ts
cov22 = (2 * sigma_z ** 2 / Ts ** 2 + sigma_a ** 2 * Ts / 3) * np.eye(2)
cov = np.block([[cov11, cov12], [cov12.T, cov22]])
init_ekfstate = GaussParams(mean, cov)

ekfpred_list = []
ekfupd_list = []
ekfupd = init_ekfstate
NIS = np.empty(K)
NEES_pred = np.empty(K)
NEES_upd = np.empty(K)
dists_pred = np.empty((K, 2))
dists_upd = np.empty((K, 2))
# estimate
for k, (zk, x_true_k) in enumerate(zip(Z[2:], Xgt[2:])):
    ekfpred = ekf_filter.predict(ekfupd, Ts)
    ekfupd = ekf_filter.update(zk, ekfpred)

    NIS[k] = ekf_filter.NIS(zk, ekfpred)
    NEES_pred[k] = estats.NEES(*ekfpred, x_true_k)
    NEES_upd[k] = estats.NEES(*ekfupd, x_true_k)

    diff_pred = ekfpred.mean - x_true_k[:4]
    diff_upd = ekfupd.mean - x_true_k[:4]
    dists_pred[k] = np.linalg.norm(diff_pred[:2]), np.linalg.norm(diff_pred[2:])
    dists_pred[k] = np.linalg.norm(diff_upd[:2]), np.linalg.norm(diff_upd[2:])

    ekfpred_list.append(ekfpred)
    ekfupd_list.append(ekfupd)


x_bar = np.array([pred.mean for pred in ekfpred_list])
P_bar = np.array([pred.cov for pred in ekfpred_list])

x_hat = np.array([upd.mean for upd in ekfupd_list])
P_hat = np.array([upd.cov for upd in ekfupd_list])

*/