#![allow(non_snake_case)]
use imm_pdaf::simulator as sim;
use matfile::{NumericData, MatFile};
use nalgebra::{DMatrix, DVector};
use matfile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open("data/data_for_pda.mat")?;
    let mat_file = matfile::MatFile::parse(file)?;
    println!("{:#?}", mat_file.arrays());
    let Z = if let Some(var_data) = mat_file.find_by_name("Z") {
        println!("Came here!!!");
        let (r, c) = (var_data.size()[0], var_data.size()[1]);
        let arr = var_data.data();
        if let NumericData::Double{real: d, imag: _} = arr {
            Some(DMatrix::from_vec(r, c, d.to_vec()))
        } else  {
            None
        }
    } else {
        None
    };
    println!("Z: {:#?}", Z);

    Ok(())
}
