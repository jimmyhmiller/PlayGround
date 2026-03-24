use std::sync::{Arc, Mutex};
use eframe::egui;

use crate::database::*;
use crate::types::*;

pub struct App {
    pub db: Arc<Mutex<Database>>,
    current_branch: BranchId,
    current_table: Option<String>,
    compare_branch: Option<BranchId>,

    // UI input state
    new_branch_name: String,
    new_table_name: String,
    new_col_name: String,
    new_col_type: usize, // index into TYPE_OPTIONS
    status: String,

    // Popups
    show_new_table: bool,
    show_new_column: bool,
    new_table_cols: Vec<(String, usize)>,

    // Inline editing
    editing_cell: Option<(RowId, String)>,
    editing_value: String,
    rename_field: Option<String>,
    rename_value: String,

    // View mode
    show_overview: bool,
}

const TYPE_OPTIONS: &[(&str, AtomicType)] = &[
    ("Str", AtomicType::Str),
    ("Num", AtomicType::Num),
    ("Bool", AtomicType::Bool),
];

impl App {
    pub fn new(db: Arc<Mutex<Database>>) -> Self {
        App {
            db,
            current_branch: 0,
            current_table: None,
            compare_branch: None,
            new_branch_name: String::new(),
            new_table_name: String::new(),
            new_col_name: String::new(),
            new_col_type: 0,
            status: "Ready".into(),
            show_new_table: false,
            show_new_column: false,
            new_table_cols: vec![("".into(), 0)],
            editing_cell: None,
            editing_value: String::new(),
            rename_field: None,
            rename_value: String::new(),
            show_overview: true,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request repaint continuously so stdin changes show up
        ctx.request_repaint();

        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.selectable_label(self.show_overview, "Overview").clicked() {
                    self.show_overview = true;
                }
                if ui.selectable_label(!self.show_overview, "Detail").clicked() {
                    self.show_overview = false;
                }
                ui.separator();
                if !self.show_overview {
                    self.top_bar(ui);
                } else {
                    ui.label("All branches and tables at a glance");
                }
            });
        });

        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
            });
        });

        if self.show_overview {
            egui::CentralPanel::default().show(ctx, |ui| {
                self.overview_panel(ui);
            });
        } else {
            egui::SidePanel::right("diff_panel").min_width(280.0).show(ctx, |ui| {
                self.diff_panel(ui);
            });

            egui::CentralPanel::default().show(ctx, |ui| {
                self.schema_panel(ui);
                ui.separator();
                self.data_panel(ui);
            });
        }

        self.popups(ctx);
    }
}

impl App {
    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let branches: Vec<(BranchId, String)> = {
                let db = self.db.lock().unwrap();
                db.list_branches().into_iter().map(|(id, n)| (id, n.to_string())).collect()
            };

            // Branch selector
            ui.label("Branch:");
            let current_name = branches.iter()
                .find(|(id, _)| *id == self.current_branch)
                .map(|(_, n)| n.as_str())
                .unwrap_or("(none)");
            egui::ComboBox::from_id_salt("branch_selector")
                .selected_text(current_name)
                .show_ui(ui, |ui| {
                    for (id, name) in &branches {
                        ui.selectable_value(&mut self.current_branch, *id, name.as_str());
                    }
                });

            // Fork button
            ui.text_edit_singleline(&mut self.new_branch_name);
            if ui.button("Fork").clicked() && !self.new_branch_name.is_empty() {
                let mut db = self.db.lock().unwrap();
                match db.fork_branch(self.current_branch, &self.new_branch_name) {
                    Ok(id) => {
                        self.status = format!("Forked → branch {}", id);
                        self.new_branch_name.clear();
                    }
                    Err(e) => self.status = format!("Error: {}", e),
                }
            }

            ui.separator();

            // Table selector
            let db = self.db.lock().unwrap();
            let tables = db.list_tables(self.current_branch).unwrap_or_default();
            drop(db);

            ui.label("Table:");
            let table_text = self.current_table.clone().unwrap_or("(none)".into());
            egui::ComboBox::from_id_salt("table_selector")
                .selected_text(&table_text)
                .show_ui(ui, |ui| {
                    for t in &tables {
                        let selected = self.current_table.as_deref() == Some(t.as_str());
                        if ui.selectable_label(selected, t).clicked() {
                            self.current_table = Some(t.clone());
                        }
                    }
                });

            if ui.button("+ Table").clicked() {
                self.show_new_table = true;
            }
        });
    }

    fn schema_panel(&mut self, ui: &mut egui::Ui) {
        let table_name = match &self.current_table {
            Some(t) => t.clone(),
            None => { ui.label("Select a table"); return; }
        };

        ui.heading(format!("Schema: {}", table_name));

        let db = self.db.lock().unwrap();
        let view = match db.get_table_view(self.current_branch, &table_name) {
            Ok(v) => v,
            Err(_) => { ui.label("Table not found"); return; }
        };
        let columns = view.columns.clone();
        drop(db);

        egui::Grid::new("schema_grid").striped(true).show(ui, |ui| {
            ui.label("Column");
            ui.label("Type");
            ui.label("Actions");
            ui.end_row();

            for (name, ty) in &columns {
                // Rename inline
                if self.rename_field.as_deref() == Some(name.as_str()) {
                    let resp = ui.text_edit_singleline(&mut self.rename_value);
                    if resp.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        if !self.rename_value.is_empty() && self.rename_value != *name {
                            let mut db = self.db.lock().unwrap();
                            match db.rename_column(self.current_branch, &table_name, name, &self.rename_value) {
                                Ok(_) => self.status = format!("Renamed {} → {}", name, self.rename_value),
                                Err(e) => self.status = format!("Error: {}", e),
                            }
                        }
                        self.rename_field = None;
                    }
                } else {
                    if ui.link(name).clicked() {
                        self.rename_field = Some(name.clone());
                        self.rename_value = name.clone();
                    }
                }

                // Type with convert dropdown
                egui::ComboBox::from_id_salt(format!("type_{}", name))
                    .selected_text(ty)
                    .show_ui(ui, |ui| {
                        for (label, at) in TYPE_OPTIONS {
                            if ui.selectable_label(ty == &format!("{:?}", at), *label).clicked() {
                                let mut db = self.db.lock().unwrap();
                                match db.convert_column(self.current_branch, &table_name, name, *at) {
                                    Ok(_) => self.status = format!("Converted {} to {}", name, label),
                                    Err(e) => self.status = format!("Error: {}", e),
                                }
                            }
                        }
                    });

                // Delete button
                if ui.button("×").clicked() {
                    let mut db = self.db.lock().unwrap();
                    match db.remove_column(self.current_branch, &table_name, name) {
                        Ok(_) => self.status = format!("Removed column {}", name),
                        Err(e) => self.status = format!("Error: {}", e),
                    }
                }
                ui.end_row();
            }
        });

        if ui.button("+ Column").clicked() {
            self.show_new_column = true;
        }
    }

    fn data_panel(&mut self, ui: &mut egui::Ui) {
        let table_name = match &self.current_table {
            Some(t) => t.clone(),
            None => return,
        };

        ui.heading("Data");

        let db = self.db.lock().unwrap();
        let rows = match db.list_rows(self.current_branch, &table_name) {
            Ok(r) => r,
            Err(_) => return,
        };
        let schema = match db.get_table_view(self.current_branch, &table_name) {
            Ok(v) => v.columns.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
            Err(_) => return,
        };
        // Clone what we need before dropping the lock
        let row_data: Vec<(RowId, Vec<(String, serde_json::Value)>)> = rows.into_iter()
            .map(|r| (r.row_id, r.fields))
            .collect();
        drop(db);

        if row_data.is_empty() {
            ui.label("No rows");
        } else {
            egui::Grid::new("data_grid").striped(true).show(ui, |ui| {
                // Header
                for col in &schema {
                    ui.label(egui::RichText::new(col).strong());
                }
                ui.label("");
                ui.end_row();

                // Rows
                for (row_id, fields) in &row_data {
                    for (name, val) in fields {
                        let val_str = match val {
                            serde_json::Value::String(s) => s.clone(),
                            serde_json::Value::Number(n) => n.to_string(),
                            serde_json::Value::Bool(b) => b.to_string(),
                            serde_json::Value::Null => "null".into(),
                            _ => format!("{}", val),
                        };

                        let is_editing = self.editing_cell.as_ref()
                            .map(|(rid, fname)| *rid == *row_id && fname == name)
                            .unwrap_or(false);

                        if is_editing {
                            let resp = ui.text_edit_singleline(&mut self.editing_value);
                            if resp.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                let value = parse_json_value(&self.editing_value);
                                let mut db = self.db.lock().unwrap();
                                match db.set_field(self.current_branch, &table_name, *row_id, name, value) {
                                    Ok(()) => self.status = format!("Set {}.{} on row {}", table_name, name, row_id),
                                    Err(e) => self.status = format!("Error: {}", e),
                                }
                                self.editing_cell = None;
                            }
                        } else if ui.link(&val_str).clicked() {
                            self.editing_cell = Some((*row_id, name.clone()));
                            self.editing_value = val_str;
                        }
                    }

                    if ui.button("Del").clicked() {
                        let mut db = self.db.lock().unwrap();
                        match db.delete_row(self.current_branch, &table_name, *row_id) {
                            Ok(()) => self.status = format!("Deleted row {}", row_id),
                            Err(e) => self.status = format!("Error: {}", e),
                        }
                    }
                    ui.end_row();
                }
            });
        }

        if ui.button("+ Row").clicked() {
            let mut db = self.db.lock().unwrap();
            match db.insert_row(self.current_branch, &table_name, vec![]) {
                Ok(id) => self.status = format!("Inserted row {}", id),
                Err(e) => self.status = format!("Error: {}", e),
            }
        }
    }

    fn overview_panel(&mut self, ui: &mut egui::Ui) {
        let db = self.db.lock().unwrap();
        let branches: Vec<(BranchId, String)> = db.list_branches()
            .into_iter().map(|(id, n)| (id, n.to_string())).collect();

        // Collect all table names across all branches
        let mut all_tables: Vec<String> = Vec::new();
        for (bid, _) in &branches {
            if let Ok(tables) = db.list_tables(*bid) {
                for t in tables {
                    if !all_tables.contains(&t) {
                        all_tables.push(t);
                    }
                }
            }
        }
        all_tables.sort();

        if branches.is_empty() || all_tables.is_empty() {
            ui.label("No branches or tables yet. Create a table to get started.");
            return;
        }

        // Preload all data while holding the lock
        let mut grid_data: Vec<Vec<Option<(Vec<(String, String)>, Vec<Vec<(String, serde_json::Value)>>)>>> = Vec::new();
        for (bid, _) in &branches {
            let mut row = Vec::new();
            for table in &all_tables {
                if let Ok(view) = db.get_table_view(*bid, table) {
                    let cols = view.columns;
                    let rows_data: Vec<Vec<(String, serde_json::Value)>> =
                        db.list_rows(*bid, table).ok()
                            .map(|rows| rows.into_iter().map(|r| r.fields).collect())
                            .unwrap_or_default();
                    row.push(Some((cols, rows_data)));
                } else {
                    row.push(None);
                }
            }
            grid_data.push(row);
        }
        drop(db);

        egui::ScrollArea::both().show(ui, |ui| {
            for (bi, (bid, bname)) in branches.iter().enumerate() {
                ui.heading(format!("🔀 {}", bname));

                ui.horizontal_wrapped(|ui| {
                    for (ti, table) in all_tables.iter().enumerate() {
                        if let Some(Some((cols, rows))) = grid_data.get(bi).and_then(|r| r.get(ti)) {
                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                ui.set_min_width(200.0);
                                ui.strong(table);
                                ui.label(format!("{} cols, {} rows", cols.len(), rows.len()));

                                // Column names
                                ui.horizontal_wrapped(|ui| {
                                    for (name, ty) in cols {
                                        ui.label(egui::RichText::new(format!("{}:{}", name, ty))
                                            .small()
                                            .color(egui::Color32::LIGHT_BLUE));
                                    }
                                });

                                // Row data (compact)
                                for (ri, row) in rows.iter().enumerate().take(5) {
                                    ui.horizontal_wrapped(|ui| {
                                        ui.label(egui::RichText::new(format!("r{}", ri)).small().weak());
                                        for (_, val) in row {
                                            let s = match val {
                                                serde_json::Value::String(s) => {
                                                    if s.len() > 12 { format!("{}…", &s[..12]) } else { s.clone() }
                                                }
                                                serde_json::Value::Number(n) => n.to_string(),
                                                serde_json::Value::Bool(b) => b.to_string(),
                                                _ => "—".into(),
                                            };
                                            ui.label(egui::RichText::new(s).small());
                                        }
                                    });
                                }
                                if rows.len() > 5 {
                                    ui.label(egui::RichText::new(format!("...+{} more", rows.len() - 5)).small().weak());
                                }

                                // Click to jump to detail view
                                if ui.small_button("Open").clicked() {
                                    self.current_branch = *bid;
                                    self.current_table = Some(table.clone());
                                    self.show_overview = false;
                                }
                            });
                        }
                    }
                });

                ui.add_space(8.0);
                ui.separator();
            }
        });
    }

    fn diff_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Diff / Merge");

        let branches: Vec<(BranchId, String)> = {
            let db = self.db.lock().unwrap();
            db.list_branches().into_iter().map(|(id, n)| (id, n.to_string())).collect()
        };

        // Compare-with selector
        ui.horizontal(|ui| {
            ui.label("Compare with:");
            let compare_text = self.compare_branch
                .and_then(|id| branches.iter().find(|(bid, _)| *bid == id).map(|(_, n)| n.as_str()))
                .unwrap_or("(select)");
            egui::ComboBox::from_id_salt("compare_branch")
                .selected_text(compare_text)
                .show_ui(ui, |ui| {
                    for (id, name) in &branches {
                        if *id != self.current_branch {
                            if ui.selectable_label(self.compare_branch == Some(*id), name.as_str()).clicked() {
                                self.compare_branch = Some(*id);
                                // Start tracking
                                let mut db = self.db.lock().unwrap();
                                let _ = db.diff_branches(self.current_branch, *id);
                            }
                        }
                    }
                });
        });

        let compare = match self.compare_branch {
            Some(b) => b,
            None => { ui.label("Select a branch to compare"); return; }
        };

        // Show diffs
        let db = self.db.lock().unwrap();
        let diffs = db.get_diffs(self.current_branch, compare);
        let conflicts = db.get_conflicts(self.current_branch, compare);
        drop(db);

        ui.separator();

        if diffs.is_empty() || diffs.iter().all(|(_, a, b)| *a == 0 && *b == 0) {
            ui.label("No differences");
        } else {
            for (table, from_count, to_count) in &diffs {
                ui.label(egui::RichText::new(format!("📋 {}", table)).strong());
                if *from_count > 0 {
                    ui.label(format!("  ← {} edits from here", from_count));
                }
                if *to_count > 0 {
                    ui.label(format!("  → {} edits from other", to_count));
                }
            }
        }

        // Conflicts
        if !conflicts.is_empty() {
            ui.separator();
            ui.label(egui::RichText::new(format!("⚠ {} conflicts", conflicts.len()))
                .color(egui::Color32::YELLOW));
            for (location, conflict) in &conflicts {
                ui.label(format!("  {} : {:?} vs {:?}",
                    location,
                    conflict.from_edit,
                    conflict.to_edit,
                ));
            }
        }

        ui.separator();

        // Merge buttons
        let current_name = branches.iter()
            .find(|(id, _)| *id == self.current_branch)
            .map(|(_, n)| n.as_str()).unwrap_or("?");
        let compare_name = branches.iter()
            .find(|(id, _)| *id == compare)
            .map(|(_, n)| n.as_str()).unwrap_or("?");

        ui.horizontal(|ui| {
            if ui.button(format!("Merge {} → {}", current_name, compare_name)).clicked() {
                let mut db = self.db.lock().unwrap();
                match db.merge_all(self.current_branch, compare) {
                    Ok(applied) => self.status = format!("Merged {} edits → {}", applied.len(), compare_name),
                    Err(e) => self.status = format!("Error: {}", e),
                }
            }
            if ui.button(format!("Merge {} → {}", compare_name, current_name)).clicked() {
                let mut db = self.db.lock().unwrap();
                match db.merge_all(compare, self.current_branch) {
                    Ok(applied) => self.status = format!("Merged {} edits → {}", applied.len(), current_name),
                    Err(e) => self.status = format!("Error: {}", e),
                }
            }
        });
    }

    fn popups(&mut self, ctx: &egui::Context) {
        // New table popup
        if self.show_new_table {
            egui::Window::new("New Table")
                .collapsible(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        ui.text_edit_singleline(&mut self.new_table_name);
                    });

                    ui.label("Columns:");
                    let mut to_remove = None;
                    for (i, (name, ty_idx)) in self.new_table_cols.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.text_edit_singleline(name);
                            egui::ComboBox::from_id_salt(format!("new_col_type_{}", i))
                                .selected_text(TYPE_OPTIONS[*ty_idx].0)
                                .show_ui(ui, |ui| {
                                    for (j, (label, _)) in TYPE_OPTIONS.iter().enumerate() {
                                        ui.selectable_value(ty_idx, j, *label);
                                    }
                                });
                            if ui.button("×").clicked() {
                                to_remove = Some(i);
                            }
                        });
                    }
                    if let Some(i) = to_remove {
                        self.new_table_cols.remove(i);
                    }
                    if ui.button("+ Column").clicked() {
                        self.new_table_cols.push(("".into(), 0));
                    }

                    ui.horizontal(|ui| {
                        if ui.button("Create").clicked() && !self.new_table_name.is_empty() {
                            let cols: Vec<(&str, AtomicType)> = self.new_table_cols.iter()
                                .filter(|(n, _)| !n.is_empty())
                                .map(|(n, ti)| (n.as_str(), TYPE_OPTIONS[*ti].1))
                                .collect();
                            let mut db = self.db.lock().unwrap();
                            match db.create_table(self.current_branch, &self.new_table_name, cols) {
                                Ok(()) => {
                                    self.current_table = Some(self.new_table_name.clone());
                                    self.status = format!("Created table {}", self.new_table_name);
                                    self.show_new_table = false;
                                    self.new_table_name.clear();
                                    self.new_table_cols = vec![("".into(), 0)];
                                }
                                Err(e) => self.status = format!("Error: {}", e),
                            }
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_new_table = false;
                        }
                    });
                });
        }

        // New column popup
        if self.show_new_column {
            egui::Window::new("Add Column")
                .collapsible(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        ui.text_edit_singleline(&mut self.new_col_name);
                    });
                    ui.horizontal(|ui| {
                        ui.label("Type:");
                        egui::ComboBox::from_id_salt("new_col_type")
                            .selected_text(TYPE_OPTIONS[self.new_col_type].0)
                            .show_ui(ui, |ui| {
                                for (i, (label, _)) in TYPE_OPTIONS.iter().enumerate() {
                                    ui.selectable_value(&mut self.new_col_type, i, *label);
                                }
                            });
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Add").clicked() && !self.new_col_name.is_empty() {
                            let table = self.current_table.clone().unwrap_or_default();
                            let ty = TYPE_OPTIONS[self.new_col_type].1;
                            let mut db = self.db.lock().unwrap();
                            match db.add_column(self.current_branch, &table, &self.new_col_name, ty) {
                                Ok(_) => {
                                    self.status = format!("Added column {}", self.new_col_name);
                                    self.show_new_column = false;
                                    self.new_col_name.clear();
                                }
                                Err(e) => self.status = format!("Error: {}", e),
                            }
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_new_column = false;
                        }
                    });
                });
        }
    }
}

fn parse_json_value(s: &str) -> Value {
    if let Ok(n) = s.parse::<f64>() {
        return Value::Num(n);
    }
    if s == "true" { return Value::Bool(true); }
    if s == "false" { return Value::Bool(false); }
    Value::Str(s.to_string())
}
