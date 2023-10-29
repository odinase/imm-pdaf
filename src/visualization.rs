pub mod double_slide_bar;

use crate::state_estimator::ekf::GaussParams;
use eframe::{egui, epaint::Color32, glow::COMPUTE_WORK_GROUP_SIZE};
use egui_plot::{Legend, Line, MarkerShape, Plot, PlotPoints, Points};
use nalgebra::DVector;

pub fn run(
    fig_name: &str,
    estimate_data: Vec<GaussParams>,
    measurement_sets: Vec<Vec<DVector<f64>>>,
    gt_data: Option<Vec<DVector<f64>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        fig_name,
        native_options,
        Box::new(|cc| {
            Box::new(TrackerVizApp::new(
                cc,
                estimate_data,
                measurement_sets,
                gt_data,
            ))
        }),
    )?;

    Ok(())
}

#[derive(Copy, Clone, Debug)]
struct DataWindow {
    pub start: usize,
    pub curr: usize,
    pub end: usize,
    pub max: usize,
    pub start_f64: f64,
    pub end_f64: f64,
}


impl std::default::Default for DataWindow {
    fn default() -> Self {
        Self {
            start: 0,
            curr: 0,
            end: 1,
            max: 1,
            start_f64: 0.0,
            end_f64: 1.0,
        }
    }
}

impl DataWindow {
    fn new(max: usize) -> Self {
        Self {
            start: 0,
            curr: max,
            start_f64: 0.0,
            end: max,
            end_f64: max as f64,
            max,
        }
    }

    fn update(&mut self) {
        self.curr = ((self.curr - self.start) % (self.end - self.start)) + self.start + 1;
    }

    fn as_range(self) -> std::ops::Range<usize> {
        self.into()
    }

    fn empty(&self) -> bool {
        self.curr == self.start
    }

    fn reset(&mut self) {
        self.start = 0;
        self.end = self.max;
        self.curr = self.end;
        self.start_f64 = 0.0;
        self.end_f64 = self.max as f64;
    }
}

impl From<DataWindow> for std::ops::Range<usize> {
    fn from(value: DataWindow) -> Self {
        Self {
            start: value.start,
            end: value.curr,
        }
    }
}

pub struct TrackerVizApp {
    estimate_data: Vec<GaussParams>,
    data_window: DataWindow,
    is_playing_movie: bool,
    gt_data: Option<Vec<DVector<f64>>>,
    measurement_sets: Vec<Vec<DVector<f64>>>,
}

impl std::default::Default for TrackerVizApp {
    fn default() -> Self {
        Self {
            estimate_data: Vec::new(),
            data_window: DataWindow::default(),
            is_playing_movie: false,
            gt_data: None,
            measurement_sets: Vec::new(),
        }
    }
}

impl TrackerVizApp {
    fn new(
        _cc: &eframe::CreationContext<'_>,
        estimate_data: Vec<GaussParams>,
        measurement_sets: Vec<Vec<DVector<f64>>>,
        gt_data: Option<Vec<DVector<f64>>>,
    ) -> Self {
        let data_window = DataWindow::new(estimate_data.len());
        Self {
            estimate_data,
            data_window,
            gt_data,
            measurement_sets,
            ..Default::default()
        }
    }

    fn draw_play_buttons(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("play_buttons").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("▶").clicked() {
                    self.enable_movie();
                }
                if ui.button("⏸").clicked() {
                    self.disable_movie();
                }
                if ui.button("⏹").clicked() {
                    self.disable_movie();
                    self.data_window.reset();
                }
            });
        });
    }

    fn draw_data_length_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("data_length_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let width = (ui.available_width() * 0.85) as f64;
                ui.add(double_slide_bar::windowing(
                    &width,
                    &100.0,
                    &(self.data_window.max as f64),
                    &mut self.data_window.start_f64,
                    &mut self.data_window.end_f64,
                ));
                self.data_window.start =
                    self.data_window
                        .start_f64
                        .clamp(0.0, self.data_window.end_f64) as usize;
                self.data_window.end = self
                    .data_window
                    .end_f64
                    .clamp(self.data_window.start_f64, self.data_window.max as f64)
                    as usize;
                self.data_window.curr = self.data_window.end;
            });
        });
    }

    fn enable_movie(&mut self) {
        self.is_playing_movie = true;
    }

    fn disable_movie(&mut self) {
        self.is_playing_movie = false;
    }

    fn draw_main_plot(&mut self, ctx: &egui::Context) {
        if self.is_playing_movie {
            self.data_window.update();
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            let plot = Plot::new("state_trajectory")
                .legend(Legend::default())
                .x_axis_label("East")
                .y_axis_label("North");

            if !self.data_window.empty() {
                let (estimate_line, start_point, end_point) = {
                    let estimate_data: &[GaussParams] =
                        &self.estimate_data[self.data_window.as_range()];
                    let x_start = estimate_data.first().unwrap().x[0];
                    let y_start = estimate_data.first().unwrap().x[1];
                    let x_end = estimate_data.last().unwrap().x[0];
                    let y_end = estimate_data.last().unwrap().x[1];
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
                        estimate_data
                            .iter()
                            .map(|state| [state.x[0], state.x[1]])
                            .collect::<PlotPoints>(),
                    )
                    .name("Estimate");
                    (estimate_line, start_point, end_point)
                };
                let gt_line = self.gt_data.as_ref().map(|Xgt| {
                    Line::new(
                        (&Xgt[self.data_window.as_range()])
                            .iter()
                            .map(|gt| [gt[0], gt[1]])
                            .collect::<PlotPoints>(),
                    )
                    .name("Ground truth")
                });
                plot.show(ui, |plot_ui| {
                    plot_ui.line(estimate_line);
                    if let Some(gt_line) = gt_line {
                        plot_ui.line(gt_line);
                    }
                    plot_ui.points(start_point);
                    plot_ui.points(end_point);
                });
            }
        });
    }
}

impl eframe::App for TrackerVizApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.draw_play_buttons(ctx);
        self.draw_main_plot(ctx);
        // self.draw_data_length_bar(ctx);
        if self.is_playing_movie {
            ctx.request_repaint();
        }
    }
}
