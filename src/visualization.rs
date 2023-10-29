use crate::state_estimator::ekf::GaussParams;
use eframe::{egui, epaint::Color32};
use egui_plot::{Legend, Line, MarkerShape, Plot, PlotPoints, Points};
use nalgebra::DVector;
use std::collections::VecDeque;

pub fn plot_states(
    fig_name: &str,
    estimate_data: Vec<GaussParams>,
    measurement_sets: Vec<Vec<DVector<f64>>>,
    gt_data: Option<Vec<DVector<f64>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(fig_name, native_options, Box::new(|cc| Box::new(TrackerVizApp::new(cc, estimate_data, measurement_sets, gt_data))))?;

    // eframe::run_simple_native(fig_name, native_options, move |ctx, _frame| {
    //     egui::CentralPanel::default().show(ctx, |ui| {
    //         let x_start = self.estimate_data.first().unwrap().x[0];
    //         let y_start = self.estimate_data.first().unwrap().x[1];
    //         let x_end = self.estimate_data.last().unwrap().x[0];
    //         let y_end = self.estimate_data.last().unwrap().x[1];
    //         let start_point = Points::new([x_start, y_start])
    //             .shape(MarkerShape::Diamond)
    //             .color(Color32::RED)
    //             .highlight(true)
    //             .radius(5.0)
    //             .name("Start");
    //         let end_point = Points::new([x_end, y_end])
    //             .shape(MarkerShape::Diamond)
    //             .color(Color32::GREEN)
    //             .highlight(true)
    //             .radius(5.0)
    //             .name("End");

    //         let estimate_line = Line::new(
    //             self.estimate_data
    //                 .iter()
    //                 .map(|state| [state.x[0], state.x[1]])
    //                 .collect::<PlotPoints>(),
    //         )
    //         .name("Estimate");
    //         let gt_line = Xgt.as_ref().map(|Xgt| {
    //             Line::new(
    //                 Xgt.as_slice()
    //                     .iter()
    //                     .map(|gt| [gt[0], gt[1]])
    //                     .collect::<PlotPoints>(),
    //             )
    //             .name("Ground truth")
    //         });

    //         Plot::new("state_trajectory")
    //             .legend(Legend::default())
    //             .x_axis_label("East")
    //             .y_axis_label("North")
    //             .show(ui, |plot_ui| {
    //                 plot_ui.line(estimate_line);
    //                 if let Some(gt_line) = gt_line {
    //                     plot_ui.line(gt_line);
    //                 }
    //                 plot_ui.points(start_point);
    //                 plot_ui.points(end_point);
    //             });
    //     });
    // })?;

    Ok(())
}

pub struct TrackerVizApp { 
    estimate_data: Vec<GaussParams>,
    gt_data: Option<Vec<DVector<f64>>>,
    measurement_sets: Vec<Vec<DVector<f64>>>,
}

impl TrackerVizApp {
    fn new(_cc: &eframe::CreationContext<'_>, estimate_data: Vec<GaussParams>, measurement_sets: Vec<Vec<DVector<f64>>>, gt_data: Option<Vec<DVector<f64>>>) -> Self {
        Self {
            estimate_data,
            gt_data,
            measurement_sets
        }
    }
}

impl eframe::App for TrackerVizApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let x_start = self.estimate_data.first().unwrap().x[0];
            let y_start = self.estimate_data.first().unwrap().x[1];
            let x_end = self.estimate_data.last().unwrap().x[0];
            let y_end = self.estimate_data.last().unwrap().x[1];
            let start_point = Points::new([x_start, y_start])
                .shape(MarkerShape::Diamond)
                .color(Color32::RED)
                .highlight(true)
                .radius(5.0)
                .name("Start");
            let end_point = Points::new([x_end, y_end])
                .shape(MarkerShape::Diamond)
                .color(Color32::GREEN)
                .highlight(true)
                .radius(5.0)
                .name("End");

            let estimate_line = Line::new(
                self.estimate_data
                    .iter()
                    .map(|state| [state.x[0], state.x[1]])
                    .collect::<PlotPoints>(),
            )
            .name("Estimate");
            let gt_line = self.gt_data.as_ref().map(|Xgt| {
                Line::new(
                    Xgt.as_slice()
                        .iter()
                        .map(|gt| [gt[0], gt[1]])
                        .collect::<PlotPoints>(),
                )
                .name("Ground truth")
            });

            Plot::new("state_trajectory")
                .legend(Legend::default())
                .x_axis_label("East")
                .y_axis_label("North")
                .show(ui, |plot_ui| {
                    plot_ui.line(estimate_line);
                    if let Some(gt_line) = gt_line {
                        plot_ui.line(gt_line);
                    }
                    plot_ui.points(start_point);
                    plot_ui.points(end_point);
                });
        });
        // ctx.request_repaint();
    }
}
