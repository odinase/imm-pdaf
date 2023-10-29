use egui::Shape::Path;
use eframe::egui;
use eframe::egui::{Color32, DragValue, Pos2, Stroke, Visuals};
use eframe::epaint::PathShape;

fn make_triangle(pos: Pos2, size: f64) -> Vec<Pos2> {
    vec![
        Pos2 {
            x: pos.x - size as f32 / 2.0,
            y: pos.y + size as f32 / 2.0,
        },
        Pos2 {
            x: pos.x,
            y: pos.y - size as f32 / 2.0,
        },
        Pos2 {
            x: pos.x + size as f32 / 2.0,
            y: pos.y + size as f32 / 2.0,
        },
    ]
}

pub fn windowing_ui(
    ui: &mut egui::Ui,
    width: &f64,
    height: &f64,
    range: &f64,
    lb: &mut f64,
    ub: &mut f64,
) -> egui::Response {
    let desired_size = egui::vec2(*width as f32, *height as f32 / 10.0);
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click_and_drag());
    let color: Color32;
    if ui.visuals() == &Visuals::dark() {
        color = Color32::WHITE;
    } else {
        color = Color32::BLACK;
    }
    if ui.is_rect_visible(rect) {
        // let visuals = ui.style().interact(&response);
        let bar_line = vec![rect.left_center(), rect.right_center()];
        ui.painter().add(Path(PathShape::line(
            bar_line,
            Stroke::new(*height as f32 / 20.0, Color32::GRAY),
        )));
        let mut bounds_i = [0.0; 2];
        bounds_i[0] = *lb / range * width;
        bounds_i[1] = *ub / range * width;
        let mut lb_pos = Pos2 {
            x: rect.left_center().x + bounds_i[0] as f32,
            y: rect.left_center().y,
        };
        let mut ub_pos = Pos2 {
            x: rect.left_center().x + bounds_i[1] as f32,
            y: rect.right_center().y,
        };

        if response.dragged() {
            let pos = response.interact_pointer_pos();

            match pos {
                None => {}
                Some(p) => {
                    if lb_pos.x - *width as f32 / 10.0 < p.x
                        && p.x < lb_pos.x + *width as f32 / 10.0
                        && lb_pos.y - *height as f32 / 5.0 < p.y
                        && p.y < lb_pos.y + *height as f32 / 5.0
                    {
                        // dragging the lower one
                        lb_pos.x = p.x;
                    } else if ub_pos.x - *width as f32 / 10.0 < p.x
                        && p.x < ub_pos.x + *width as f32 / 10.0
                        && ub_pos.y - *height as f32 / 5.0 < p.y
                        && p.y < ub_pos.y + *height as f32 / 5.0
                    {
                        // dragging the upper one
                        ub_pos.x = p.x;
                    }
                }
            }
        }

        bounds_i[0] = lb_pos.x as f64 - rect.left_center().x as f64;
        *lb = bounds_i[0] / width * range;
        bounds_i[1] = ub_pos.x as f64 - rect.left_center().x as f64;
        *ub = bounds_i[1] / width * range;

        ui.painter().add(Path(PathShape::convex_polygon(
            make_triangle(lb_pos, *height / 10.0),
            color,
            Stroke::new(1.0, color),
        )));
        ui.painter().add(Path(PathShape::convex_polygon(
            make_triangle(ub_pos, *height / 10.0),
            color,
            Stroke::new(1.0, color),
        )));
        // ui.vertical_centered(|ui| {
        //     ui.horizontal(|ui| {
        //         ui.add(
        //             DragValue::new( lb)
        //                 .clamp_range(0.0..=*range),
        //         );
        //         ui.add_space(2.0 * ui.available_width() - *width as f32 / 0.85);
        //         ui.add(
        //             DragValue::new( ub)
        //                 .clamp_range(0.0..=*range),
        //         );
        //     });
        // });
    }
    response
}

// A wrapper that allows the more idiomatic usage pattern: `ui.add(gauge(&temperatue, "temperature"))`
pub fn windowing<'a>(
    width: &'a f64,
    height: &'a f64,
    range: &'a f64,
    lb: &'a mut f64,
    ub: &'a mut f64,
) -> impl egui::Widget + 'a {
    move |ui: &mut egui::Ui| windowing_ui(ui, width, height, range, lb, ub)
}