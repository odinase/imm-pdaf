use crate::state_estimator::ekf::GaussParams;
use gnuplot::*;
use nalgebra::DMatrix;

pub fn plot_states(state: &[GaussParams], Xgt: Option<&DMatrix<f64>>) {
    let mut fg = Figure::new();
    let ax = fg.axes2d();
        ax.lines(
            state.iter().map(|s| s.x[0]),
            state.iter().map(|s| s.x[1]),
            &[Caption("Estimate")],
        )
        .set_x_grid(true)
        .set_y_grid(true);
    if let Some(Xgt) = Xgt {
        ax.lines(
            Xgt.column_iter().map(|xgt| xgt[0]),
            Xgt.column_iter().map(|xgt| xgt[1]),
            &[Caption("Ground truth")],
        );
    }
    fg.show().unwrap();
}
