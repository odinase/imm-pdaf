use crate::state_estimator::ekf::GaussParams;
use nalgebra::DVector;
use eframe::{egui, epaint::Color32};
use egui_plot::{Legend, Line, Plot, PlotPoints};
use std::collections::VecDeque;


pub fn plot_states(
    states: Vec<GaussParams>,
    Xgt: Option<Vec<DVector<f64>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let native_options = eframe::NativeOptions::default();

    let states = std::rc::Rc::new(states);
    let Xgt = std::rc::Rc::new(Xgt);
    eframe::run_simple_native("My egui App", native_options, move |ctx, _frame| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let estimate_line = Line::new(
                states
                    .iter()
                    .map(|state| [state.x[0], state.x[1]])
                    .collect::<PlotPoints>(),
            )
            .name("Estimate");
            Plot::new("state_trajectory")
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    plot_ui.line(estimate_line);
                    if let Some(Xgt) = &*Xgt {
                        let gt_line =
                            Line::new(Xgt.as_slice().iter().map(|gt| [gt[0], gt[1]]).collect::<PlotPoints>())
                                .name("Ground truth");
                        plot_ui.line(gt_line);
                    }
                });
        });
    })?;

    Ok(())
}

// pub struct TrackerVizApp;

// impl TrackerVizApp {
//     fn new(_cc: &eframe::CreationContext<'_>) -> Self {
//         // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
//         // Restore app state using cc.storage (requires the "persistence" feature).
//         // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
//         // for e.g. egui::PaintCallback.
//         MyEguiApp {
//             roll_data: VecDeque::new(),
//             pitch_data: VecDeque::new(),
//             yaw_data: VecDeque::new(),
//             buffer_size: BufferSize::Limited(100),
//             rx,
//         }
//     }
// }

// impl eframe::App for MyEguiApp {
//     fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
//         egui::CentralPanel::default().show(ctx, |ui| {
//             ui.heading("Hello World!");

//             let my_plot = Plot::new("Euler angles").legend(Legend::default());

//             // let's create a dummy line in the plot
//             let _inner = my_plot.show(ui, |plot_ui| {
//                 plot_ui.line(
//                     Line::new(self.roll_data.iter().copied().collect::<PlotPoints>())
//                         .name("Roll")
//                         .color(Color32::GREEN),
//                 );
//                 plot_ui.line(
//                     Line::new(self.pitch_data.iter().copied().collect::<PlotPoints>())
//                         .name("Pitch")
//                         .color(Color32::RED),
//                 );
//                 plot_ui.line(
//                     Line::new(self.yaw_data.iter().copied().collect::<PlotPoints>())
//                         .name("Yaw")
//                         .color(Color32::BLUE),
//                 );
//             });
//         });
//         ctx.request_repaint();
//     }
// }
