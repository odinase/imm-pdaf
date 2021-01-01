#![allow(non_snake_case)]
use matfile;
use nalgebra::{DMatrix, DVector};
use imm_pdaf::simulator as sim;
use itertools::izip;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open("data/data_for_ekf.mat")?;
    let mat_file = matfile::MatFile::parse(file)?;
    let K = sim::read_var_from_mat(&mat_file, "K").unwrap()[0]; // Scalar
    let Ts = sim::read_var_from_mat(&mat_file, "Ts").unwrap()[0]; // Scalar
    let Xgt = sim::read_var_from_mat(&mat_file, "Xgt").unwrap(); // 5 x K
    let Z = sim::read_var_from_mat(&mat_file, "Z").unwrap(); // 2 x K
    println!("K: {}", K);
    println!("Ts: {}", Ts);
    println!("Xgt: {}\n, Xgt.shape: {:#?}", Xgt.slice((0,0), (5, 2)), Xgt.shape());
    println!("Z: {}\n, Z.shape: {:#?}", Z.slice((0,0), (2, 2)), Z.shape());
    for (k, (z, xgt)) in izip!(
        Z.slice((0,0), (2, 2)).column_iter(),
        Xgt.slice((0,0), (5, 2)).column_iter()
    ).enumerate() {
        println!("{}:\nz: {}\nxgt: {}", k, z, xgt);
    }
    Ok(())
}