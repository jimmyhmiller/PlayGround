//! Spawn tree panel: renders the parent/child task hierarchy as a
//! collapsible tree. Cheap; we rebuild the index from the registry snapshot
//! every frame.

use cf_runtime::scheduler::TaskMetaSnapshot;
use cf_runtime::task::{TaskId, TaskState};
use eframe::egui;
use std::collections::HashMap;

pub struct TreeView<'a> {
    pub tasks: &'a [TaskMetaSnapshot],
    pub selected: &'a mut Option<TaskId>,
}

impl<'a> TreeView<'a> {
    pub fn show(self, ui: &mut egui::Ui) {
        // Build child map.
        let mut children: HashMap<Option<TaskId>, Vec<&TaskMetaSnapshot>> = HashMap::new();
        for t in self.tasks {
            children.entry(t.parent).or_default().push(t);
        }
        for v in children.values_mut() {
            v.sort_by_key(|t| t.id.0);
        }
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let roots = children.get(&None).cloned().unwrap_or_default();
                for r in roots {
                    render_node(ui, r, &children, self.selected, 0);
                }
            });
    }
}

fn render_node<'a>(
    ui: &mut egui::Ui,
    node: &'a TaskMetaSnapshot,
    children: &HashMap<Option<TaskId>, Vec<&'a TaskMetaSnapshot>>,
    selected: &mut Option<TaskId>,
    depth: usize,
) {
    let kids = children.get(&Some(node.id)).cloned().unwrap_or_default();
    let id = ui.id().with(("tree-node", node.id.0));
    egui::CollapsingHeader::new(node_label(node))
        .id_salt(id)
        .default_open(depth < 2)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.small_button("select").clicked() {
                    *selected = Some(node.id);
                }
                ui.label(format!(
                    "{:?} polls={} busy={}µs",
                    node.state,
                    node.poll_count,
                    node.busy_nanos / 1000
                ));
            });
            for k in kids {
                render_node(ui, k, children, selected, depth + 1);
            }
        });
}

fn node_label(t: &TaskMetaSnapshot) -> egui::RichText {
    let color = match t.state {
        TaskState::Running => egui::Color32::from_rgb(80, 200, 120),
        TaskState::Runnable => egui::Color32::from_rgb(120, 180, 240),
        TaskState::Suspended => egui::Color32::from_rgb(180, 180, 180),
        TaskState::Fresh => egui::Color32::from_rgb(220, 220, 100),
        TaskState::PausedByUser => egui::Color32::from_rgb(240, 180, 60),
        TaskState::Completed => egui::Color32::from_rgb(120, 120, 120),
        TaskState::Aborted => egui::Color32::from_rgb(220, 80, 80),
    };
    egui::RichText::new(format!("#{} {}", t.id.0, t.name)).color(color)
}
