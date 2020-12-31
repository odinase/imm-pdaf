use matfile::NumericData;
use nalgebra::{DMatrix, DVector};

fn main() {
    let file = std::fs::File::open("data/data_for_ekf.mat").unwrap();
    let mat_file = matfile::MatFile::parse(file).unwrap();
    let z_data = mat_file.find_by_name("Z").unwrap();
    let (r, c) = (z_data.size()[0], z_data.size()[1]);
    let z_arr = z_data.data(); 
    println!("{:#?}", (r, c));
    if let NumericData::Double{real: d, imag: _} = z_arr {
        println!("length vec: {}", d.len());
        let data = DMatrix::from_vec(r, c, d.to_vec());
        println!("shape data: {:#?}", data.shape());
        println!("data: {}", data.slice((0,997), (1,999)));
    }
}