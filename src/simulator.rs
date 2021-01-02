use matfile::{NumericData, MatFile};
use nalgebra::{DMatrix, DVector};


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

