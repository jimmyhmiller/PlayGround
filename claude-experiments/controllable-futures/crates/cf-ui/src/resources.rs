//! Resources tab: lists every live runtime resource (channels, sockets,
//! sync primitives) with current state.

use cf_runtime::resource::ResourceMetaSnapshot;
use eframe::egui;

pub struct ResourcesView<'a> {
    pub items: &'a [ResourceMetaSnapshot],
}

impl<'a> ResourcesView<'a> {
    pub fn show(self, ui: &mut egui::Ui) {
        if self.items.is_empty() {
            ui.label("(no live resources)");
            return;
        }
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let header = |ui: &mut egui::Ui, txt: &str, w: f32| {
                    ui.allocate_ui(egui::vec2(w, 18.0), |ui| {
                        ui.label(
                            egui::RichText::new(txt).color(egui::Color32::GRAY).monospace(),
                        );
                    });
                };
                ui.horizontal(|ui| {
                    header(ui, "id", 40.0);
                    header(ui, "kind", 100.0);
                    header(ui, "label", 200.0);
                    header(ui, "depth/cap", 100.0);
                    header(ui, "sends", 70.0);
                    header(ui, "recvs", 70.0);
                    header(ui, "high-water", 100.0);
                    header(ui, "by", 50.0);
                    header(ui, "age", 80.0);
                });
                ui.separator();
                for r in self.items {
                    ui.horizontal(|ui| {
                        ui.allocate_ui(egui::vec2(40.0, 18.0), |ui| {
                            ui.monospace(format!("#{}", r.id.0));
                        });
                        ui.allocate_ui(egui::vec2(100.0, 18.0), |ui| {
                            ui.label(
                                egui::RichText::new(r.kind.label())
                                    .color(kind_color(r.kind))
                                    .monospace(),
                            );
                        });
                        ui.allocate_ui(egui::vec2(200.0, 18.0), |ui| {
                            ui.label(&r.label);
                        });
                        ui.allocate_ui(egui::vec2(100.0, 18.0), |ui| {
                            let txt = match (r.state.depth, r.state.capacity) {
                                (Some(d), Some(c)) => format!("{d}/{c}"),
                                (Some(d), None) => format!("{d}"),
                                _ => String::new(),
                            };
                            ui.monospace(txt);
                        });
                        ui.allocate_ui(egui::vec2(70.0, 18.0), |ui| {
                            ui.monospace(format!("{}", r.state.sends));
                        });
                        ui.allocate_ui(egui::vec2(70.0, 18.0), |ui| {
                            ui.monospace(format!("{}", r.state.recvs));
                        });
                        ui.allocate_ui(egui::vec2(100.0, 18.0), |ui| {
                            ui.monospace(format!("{}", r.state.high_water));
                        });
                        ui.allocate_ui(egui::vec2(50.0, 18.0), |ui| {
                            if let Some(t) = r.created_by {
                                ui.monospace(format!("#{}", t.0));
                            }
                        });
                        ui.allocate_ui(egui::vec2(80.0, 18.0), |ui| {
                            let ms = r.age_nanos / 1_000_000;
                            ui.monospace(format!("{ms}ms"));
                        });
                    });
                }
            });
    }
}

fn kind_color(k: cf_runtime::resource::ResourceKind) -> egui::Color32 {
    use cf_runtime::resource::ResourceKind::*;
    match k {
        MpscChannel | BroadcastChannel | OneshotChannel => {
            egui::Color32::from_rgb(180, 220, 255)
        }
        Notify | Semaphore => egui::Color32::from_rgb(255, 200, 120),
        TcpListener | TcpStream => egui::Color32::from_rgb(120, 220, 180),
    }
}
