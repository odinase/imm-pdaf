use crate::state_estimator::ekf::GaussParams;
use eframe::{egui, epaint::Color32};
use egui_plot::{Legend, Line, MarkerShape, Plot, PlotPoints, Points};
use nalgebra::DVector;
use std::collections::VecDeque;
use tokio::sync::mpsc::{Sender, Receiver, channel};

pub fn plot_states(
    fig_name: &str,
    states: Vec<GaussParams>,
    Xgt: Option<Vec<DVector<f64>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let native_options = eframe::NativeOptions::default();

    eframe::run_simple_native(fig_name, native_options, move |ctx, _frame| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let x_start = states.first().unwrap().x[0];
            let y_start = states.first().unwrap().x[1];
            let x_end = states.last().unwrap().x[0];
            let y_end = states.last().unwrap().x[1];
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
                states
                    .iter()
                    .map(|state| [state.x[0], state.x[1]])
                    .collect::<PlotPoints>(),
            )
            .name("Estimate");
            let gt_line = Xgt.as_ref().map(|Xgt| {
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
    })?;

    Ok(())
}

pub struct BiDirectionalChannel<T> {
    rx: Receiver<T>,
    tx: Sender<T>,
}

pub fn bidirectional_channel<T>(buffer: usize) -> (BiDirectionalChannel<T>, BiDirectionalChannel<T>) {
    let (tx1, rx1) = channel(buffer);
    let (tx2, rx2) = channel(buffer);
    (BiDirectionalChannel {rx: rx1, tx: tx2}, BiDirectionalChannel {rx: rx2, tx: tx1})
}

impl<T> BiDirectionalChannel<T> {
    pub async fn send(&self, value: T) -> Result<(), tokio::sync::mpsc::error::SendError<T>> {
        self.tx.send(value).await
    }

    pub async fn recv(&mut self) -> Option<T> {
        self.rx.recv().await
    }
}


// pub fn init_viz() -> 


pub struct TrackerVizApp;

impl TrackerVizApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.
        TrackerVizApp {
        }
    }
}

impl eframe::App for TrackerVizApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Hello World!");

            let my_plot = Plot::new("Euler angles").legend(Legend::default());

            // let's create a dummy line in the plot
            let _inner = my_plot.show(ui, |plot_ui| {
                plot_ui.line(
                    Line::new(self.roll_data.iter().copied().collect::<PlotPoints>())
                        .name("Roll")
                        .color(Color32::GREEN),
                );
                plot_ui.line(
                    Line::new(self.pitch_data.iter().copied().collect::<PlotPoints>())
                        .name("Pitch")
                        .color(Color32::RED),
                );
                plot_ui.line(
                    Line::new(self.yaw_data.iter().copied().collect::<PlotPoints>())
                        .name("Yaw")
                        .color(Color32::BLUE),
                );
            });
        });
        ctx.request_repaint();
    }
}
