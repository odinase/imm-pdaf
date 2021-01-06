use crate::state_estimator::ekf::GaussParams;
use gnuplot::*;
use nalgebra::{DMatrix, DVector};

pub fn plot_states(state: &[GaussParams], Xgt: Option<&DMatrix<f64>>) {
    let mut fg = Figure::new();
    let ax = fg.axes2d();
        // .set_y_range(Fix, Fix(1.5))
        // .set_x_range(Fix(-1.5), Fix(1.5))
        ax.lines(
            state.iter().map(|s| s.x[0]),
            state.iter().map(|s| s.x[1]),
            &[Caption("Estimate") /*, PointSymbol('x')*/],
        );
    if let Some(Xgt) = Xgt {
        ax.lines(
            Xgt.column_iter().map(|xgt| xgt[0]),
            Xgt.column_iter().map(|xgt| xgt[1]),
            &[Caption("Ground truth") /*, PointSymbol('x')*/],
        );
    }
    fg.show().unwrap();
}
