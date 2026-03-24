use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::apply::conform;
use crate::schema::*;
use crate::schema_edit::*;
use crate::types::*;

pub type BranchId = u64;
pub type RowId = u64;

#[derive(Debug)]
pub enum DbError {
    BranchNotFound(BranchId),
    TableNotFound(String),
    RowNotFound(RowId),
    FieldNotFound(String),
    TableAlreadyExists(String),
    Schema(SchemaError),
    ApplyFailed,
    MigrationFailed,
    NoDiffTracking,
}

impl std::fmt::Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DbError::BranchNotFound(id) => write!(f, "branch not found: {}", id),
            DbError::TableNotFound(name) => write!(f, "table not found: {}", name),
            DbError::RowNotFound(id) => write!(f, "row not found: {}", id),
            DbError::FieldNotFound(name) => write!(f, "field not found: {}", name),
            DbError::TableAlreadyExists(name) => write!(f, "table already exists: {}", name),
            DbError::Schema(e) => write!(f, "{}", e),
            DbError::ApplyFailed => write!(f, "failed to apply edit"),
            DbError::MigrationFailed => write!(f, "migration failed"),
            DbError::NoDiffTracking => write!(f, "no diff tracking between these branches"),
        }
    }
}

impl std::error::Error for DbError {}

impl From<SchemaError> for DbError {
    fn from(e: SchemaError) -> Self {
        DbError::Schema(e)
    }
}

/// A table: schema (column definitions) + rows (data).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    pub schema: Schema,
    pub rows: HashMap<RowId, Document>,
    /// Ordered list of RowIds. Tombstoned rows have their id set to u64::MAX.
    pub row_order: Vec<RowId>,
}

/// A branch: a self-contained workspace with named tables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    pub name: String,
    pub tables: HashMap<String, Table>,
    /// Parent branch (None for root branches).
    pub parent: Option<BranchId>,
    /// Edit log: every SchemaEdit applied to this branch, per table.
    /// Used to seed diff trackers when comparing branches that weren't
    /// tracked from the start.
    pub edit_log: HashMap<String, Vec<SchemaEdit>>,
}

/// A named row view for display.
#[derive(Debug, Serialize)]
pub struct RowView {
    pub row_id: RowId,
    pub fields: Vec<(String, JsonValue)>,
}

/// A table schema view for display.
#[derive(Debug, Serialize)]
pub struct TableView {
    pub table: String,
    pub columns: Vec<(String, String)>,
    pub row_count: usize,
}

/// Global ID generator.
#[derive(Default, Serialize, Deserialize)]
struct IdGen {
    branch: u64,
    row: u64,
    ins: u64,
}

impl IdGen {
    fn branch(&mut self) -> BranchId {
        let id = self.branch;
        self.branch += 1;
        id
    }
    fn row(&mut self) -> RowId {
        let id = self.row;
        self.row += 1;
        id
    }
    #[allow(dead_code)]
    fn ins(&mut self) -> u64 {
        let id = self.ins;
        self.ins += 1;
        id
    }
}

/// Diff key: (min_branch, max_branch, table_name).
/// The "a" side of SchemaDifferences is always the min branch id.
type DiffKey = (BranchId, BranchId, String);

fn canonical_key(a: BranchId, b: BranchId, table: &str) -> DiffKey {
    if a <= b {
        (a, b, table.to_string())
    } else {
        (b, a, table.to_string())
    }
}

/// Whether a branch is the "a" side (min id) of the canonical pair.
fn is_a_side(branch: BranchId, key: &DiffKey) -> bool {
    branch == key.0
}

fn key_table(key: &DiffKey) -> &str {
    &key.2
}

fn key_has_branch(key: &DiffKey, branch: BranchId) -> bool {
    key.0 == branch || key.1 == branch
}

/// The database: branches, per-table diff tracking, ID generation.
#[derive(Serialize, Deserialize)]
pub struct Database {
    branches: HashMap<BranchId, Branch>,
    /// Schema diffs per (branch_pair, table). Uses name-based SchemaEdit OT.
    #[serde(
        serialize_with = "serialize_diffs",
        deserialize_with = "deserialize_diffs"
    )]
    diffs: HashMap<DiffKey, SchemaDifferences>,
    ids: IdGen,
}

fn serialize_diffs<S: serde::Serializer>(
    diffs: &HashMap<DiffKey, SchemaDifferences>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(diffs.len()))?;
    for (key, value) in diffs {
        seq.serialize_element(&(key, value))?;
    }
    seq.end()
}

fn deserialize_diffs<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<HashMap<DiffKey, SchemaDifferences>, D::Error> {
    let entries: Vec<(DiffKey, SchemaDifferences)> = Vec::deserialize(deserializer)?;
    Ok(entries.into_iter().collect())
}

impl Database {
    pub fn new() -> Self {
        Database {
            branches: HashMap::new(),
            diffs: HashMap::new(),
            ids: IdGen { ins: 1, ..Default::default() },
        }
    }

    // ── Branch operations ──────────────────────────────────────────────

    pub fn create_branch(&mut self, name: &str) -> BranchId {
        let id = self.ids.branch();
        self.branches.insert(id, Branch {
            name: name.to_string(),
            tables: HashMap::new(),
            parent: None,
            edit_log: HashMap::new(),
        });
        id
    }

    pub fn fork_branch(&mut self, source: BranchId, name: &str) -> Result<BranchId, DbError> {
        let src = self.branches.get(&source)
            .ok_or(DbError::BranchNotFound(source))?
            .clone();
        let id = self.ids.branch();

        let siblings: Vec<BranchId> = self.branches.iter()
            .filter(|(bid, b)| b.parent == Some(source) && **bid != id)
            .map(|(bid, _)| *bid)
            .collect();

        self.branches.insert(id, Branch {
            name: name.to_string(),
            tables: src.tables,
            parent: Some(source),
            edit_log: src.edit_log.clone(), // inherit parent's history
        });

        // Auto-track against parent
        self.diff_branches(source, id)?;

        // Auto-track against all siblings
        for sibling in siblings {
            self.diff_branches(sibling, id)?;
        }

        Ok(id)
    }

    pub fn list_branches(&self) -> Vec<(BranchId, &str)> {
        let mut v: Vec<_> = self.branches.iter()
            .map(|(id, b)| (*id, b.name.as_str()))
            .collect();
        v.sort_by_key(|(id, _)| *id);
        v
    }

    fn get_branch(&self, id: BranchId) -> Result<&Branch, DbError> {
        self.branches.get(&id).ok_or(DbError::BranchNotFound(id))
    }

    fn get_branch_mut(&mut self, id: BranchId) -> Result<&mut Branch, DbError> {
        self.branches.get_mut(&id).ok_or(DbError::BranchNotFound(id))
    }

    // ── Table operations ───────────────────────────────────────────────

    pub fn create_table(
        &mut self,
        branch: BranchId,
        table: &str,
        columns: Vec<(&str, AtomicType)>,
    ) -> Result<(), DbError> {
        let ins_start = self.ids.ins;
        self.ids.ins += columns.len() as u64;
        let b = self.get_branch_mut(branch)?;
        if b.tables.contains_key(table) {
            return Err(DbError::TableAlreadyExists(table.to_string()));
        }
        b.tables.insert(table.to_string(), Table {
            schema: Schema::new(columns, ins_start),
            rows: HashMap::new(),
            row_order: Vec::new(),
        });
        Ok(())
    }

    pub fn list_tables(&self, branch: BranchId) -> Result<Vec<String>, DbError> {
        let b = self.get_branch(branch)?;
        let mut names: Vec<String> = b.tables.keys().cloned().collect();
        names.sort();
        Ok(names)
    }

    pub fn get_table_view(
        &self,
        branch: BranchId,
        table: &str,
    ) -> Result<TableView, DbError> {
        let b = self.get_branch(branch)?;
        let t = b.tables.get(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
        let columns = t.schema.active_fields().iter()
            .map(|(_, nf)| (nf.name.clone(), format!("{:?}", nf.ty)))
            .collect();
        Ok(TableView {
            table: table.to_string(),
            columns,
            row_count: t.rows.len(),
        })
    }

    // ── Row operations ─────────────────────────────────────────────────

    pub fn insert_row(
        &mut self,
        branch: BranchId,
        table: &str,
        data: Vec<(&str, Value)>,
    ) -> Result<RowId, DbError> {
        let id = self.ids.row();
        {
            let b = self.get_branch_mut(branch)?;
            let t = b.tables.get_mut(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
            let mut doc = t.schema.empty_document();
            for (name, value) in data {
                if let Some(idx) = t.schema.index_of(name) {
                    doc.fields[idx].value = value;
                }
            }
            t.rows.insert(id, doc);
            t.row_order.push(id);
        }
        Ok(id)
    }

    pub fn delete_row(
        &mut self,
        branch: BranchId,
        table: &str,
        row: RowId,
    ) -> Result<(), DbError> {
        let b = self.get_branch_mut(branch)?;
        let t = b.tables.get_mut(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
        if t.rows.remove(&row).is_none() {
            return Err(DbError::RowNotFound(row));
        }
        let row_idx = t.row_order.iter().position(|&id| id == row);
        if let Some(pos) = row_idx {
            t.row_order[pos] = u64::MAX;
        }
        Ok(())
    }

    /// Set a value on a specific row. Records a SetField edit in the schema diff channel.
    pub fn set_field(
        &mut self,
        branch: BranchId,
        table: &str,
        row: RowId,
        field: &str,
        value: Value,
    ) -> Result<(), DbError> {
        {
            let b = self.get_branch_mut(branch)?;
            let t = b.tables.get_mut(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
            let idx = t.schema.index_of(field)
                .ok_or_else(|| DbError::FieldNotFound(field.into()))?;
            let doc = t.rows.get_mut(&row).ok_or(DbError::RowNotFound(row))?;
            doc.fields[idx].value = value.clone();
        }
        let edit = SchemaEdit::SetField { row, field: field.to_string(), value };
        self.record_schema_edit(branch, table, &edit);
        Ok(())
    }

    pub fn get_row(
        &self,
        branch: BranchId,
        table: &str,
        row: RowId,
    ) -> Result<RowView, DbError> {
        let b = self.get_branch(branch)?;
        let t = b.tables.get(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
        let doc = t.rows.get(&row).ok_or(DbError::RowNotFound(row))?;
        let conformed = conform(doc);
        let fields = t.schema.active_fields().iter()
            .filter(|(idx, _)| *idx < conformed.fields.len() && conformed.fields[*idx].ty != AtomicType::Del)
            .map(|(idx, nf)| (nf.name.clone(), value_to_json(&conformed.fields[*idx].value)))
            .collect();
        Ok(RowView { row_id: row, fields })
    }

    pub fn list_rows(
        &self,
        branch: BranchId,
        table: &str,
    ) -> Result<Vec<RowView>, DbError> {
        let b = self.get_branch(branch)?;
        let t = b.tables.get(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
        let mut views = Vec::new();
        let mut row_ids: Vec<RowId> = t.rows.keys().cloned().collect();
        row_ids.sort();
        for row_id in row_ids {
            let doc = &t.rows[&row_id];
            let conformed = conform(doc);
            let fields = t.schema.active_fields().iter()
                .filter(|(idx, _)| *idx < conformed.fields.len() && conformed.fields[*idx].ty != AtomicType::Del)
                .map(|(idx, nf)| (nf.name.clone(), value_to_json(&conformed.fields[*idx].value)))
                .collect();
            views.push(RowView { row_id, fields });
        }
        Ok(views)
    }

    // ── Schema edit operations (propagate to all rows) ─────────────────

    pub fn add_column(
        &mut self,
        branch: BranchId,
        table: &str,
        name: &str,
        ty: AtomicType,
    ) -> Result<(), DbError> {
        {
            let b = self.get_branch_mut(branch)?;
            let t = b.tables.get_mut(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
            // Check if field already exists
            if t.schema.index_of(name).is_some() {
                return Err(DbError::Schema(SchemaError::FieldAlreadyExists(name.to_string())));
            }
            // Add field to schema
            let idx = t.schema.fields.len();
            t.schema.fields.push(NamedField {
                name: name.to_string(),
                ty,
            });
            // Add field to all existing rows (with default value)
            let default_value = default_for_type(ty);
            for doc in t.rows.values_mut() {
                // Extend fields to match schema length
                while doc.fields.len() <= idx {
                    doc.fields.push(Field::null(AtomicType::Del));
                }
                doc.fields[idx] = Field { ty, value: default_value.clone() };
            }
        }
        let edit = SchemaEdit::AddField { name: name.to_string(), ty };
        self.record_schema_edit(branch, table, &edit);
        Ok(())
    }

    pub fn remove_column(
        &mut self,
        branch: BranchId,
        table: &str,
        name: &str,
    ) -> Result<(), DbError> {
        {
            let b = self.get_branch_mut(branch)?;
            let t = b.tables.get_mut(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
            let idx = t.schema.index_of(name)
                .ok_or_else(|| DbError::FieldNotFound(name.into()))?;
            // Tombstone the field in schema
            t.schema.fields[idx].ty = AtomicType::Del;
            // Tombstone in all rows
            for doc in t.rows.values_mut() {
                if idx < doc.fields.len() {
                    doc.fields[idx] = Field::null(AtomicType::Del);
                }
            }
        }
        let edit = SchemaEdit::RemoveField { name: name.to_string() };
        self.record_schema_edit(branch, table, &edit);
        Ok(())
    }

    pub fn convert_column(
        &mut self,
        branch: BranchId,
        table: &str,
        name: &str,
        to: AtomicType,
    ) -> Result<(), DbError> {
        {
            let b = self.get_branch_mut(branch)?;
            let t = b.tables.get_mut(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
            let idx = t.schema.index_of(name)
                .ok_or_else(|| DbError::FieldNotFound(name.into()))?;
            t.schema.fields[idx].ty = to;
            for doc in t.rows.values_mut() {
                if idx < doc.fields.len() {
                    doc.fields[idx].ty = to;
                }
            }
        }
        let edit = SchemaEdit::ConvertField { name: name.to_string(), ty: to };
        self.record_schema_edit(branch, table, &edit);
        Ok(())
    }

    pub fn rename_column(
        &mut self,
        branch: BranchId,
        table: &str,
        old: &str,
        new: &str,
    ) -> Result<(), DbError> {
        {
            let b = self.get_branch_mut(branch)?;
            let t = b.tables.get_mut(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
            let idx = t.schema.index_of(old)
                .ok_or_else(|| DbError::FieldNotFound(old.into()))?;
            if t.schema.index_of(new).is_some() {
                return Err(DbError::Schema(SchemaError::FieldAlreadyExists(new.to_string())));
            }
            t.schema.fields[idx].name = new.to_string();
        }
        let edit = SchemaEdit::RenameField { old_name: old.to_string(), new_name: new.to_string() };
        self.record_schema_edit(branch, table, &edit);
        Ok(())
    }

    // ── Diff & merge ───────────────────────────────────────────────────

    /// Start tracking diffs between two branches for all shared tables.
    /// Called automatically by `fork_branch`.
    ///
    /// If the branches have already diverged, the diffs are seeded from
    /// each branch's edit log — the permanent record of every SchemaEdit
    /// applied since the branch was created. This correctly captures the
    /// direction of changes (AddField vs RemoveField) without guessing.
    pub fn diff_branches(&mut self, a: BranchId, b: BranchId) -> Result<(), DbError> {
        let tables_a: Vec<String> = {
            let ba = self.get_branch(a)?;
            ba.tables.keys().cloned().collect()
        };
        let tables_b: Vec<String> = {
            let bb = self.get_branch(b)?;
            bb.tables.keys().cloned().collect()
        };

        for table in &tables_a {
            if !tables_b.contains(table) {
                continue;
            }
            let key = canonical_key(a, b, table);
            if self.diffs.contains_key(&key) {
                continue; // already tracking
            }

            // Seed from edit logs. Each branch's edit log records what it did
            // since creation. For parent-child pairs (fork), both branches
            // share history up to the fork point — only post-fork edits matter.
            // For arbitrary pairs, we use the full logs.
            let a_parent = self.branches.get(&a).and_then(|br| br.parent);
            let b_parent = self.branches.get(&b).and_then(|br| br.parent);

            let (a_log, b_log) = if a_parent == Some(b) || b_parent == Some(a) {
                // Parent-child: start with empty diffs (identical at fork)
                (vec![], vec![])
            } else {
                // Non-fork pair: seed from full edit logs
                let al: Vec<SchemaEdit> = self.branches.get(&a)
                    .and_then(|br| br.edit_log.get(table.as_str()))
                    .cloned()
                    .unwrap_or_default();
                let bl: Vec<SchemaEdit> = self.branches.get(&b)
                    .and_then(|br| br.edit_log.get(table.as_str()))
                    .cloned()
                    .unwrap_or_default();
                (al, bl)
            };

            let mut sd = SchemaDifferences::new();
            for edit in &a_log {
                sd.edit_a(edit);
            }
            for edit in &b_log {
                sd.edit_b(edit);
            }

            self.diffs.insert(key, sd);
        }
        Ok(())
    }

    /// Get diff summary between two branches.
    /// Returns (table_name, from_diffs_count, to_diffs_count).
    pub fn get_diffs(&self, from: BranchId, to: BranchId) -> Vec<(String, usize, usize)> {
        let mut result = Vec::new();
        for (key, diffs) in &self.diffs {
            if !key_has_branch(key, from) || !key_has_branch(key, to) {
                continue;
            }
            let (from_count, to_count) = if is_a_side(from, key) {
                (diffs.a_diffs.len(), diffs.b_diffs.len())
            } else {
                (diffs.b_diffs.len(), diffs.a_diffs.len())
            };
            if from_count == 0 && to_count == 0 {
                continue;
            }
            result.push((key_table(key).to_string(), from_count, to_count));
        }
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }

    /// Migrate the first diff from branch `from` to branch `to` for a specific table.
    pub fn migrate_table(
        &mut self,
        from: BranchId,
        to: BranchId,
        table: &str,
    ) -> Result<Option<SchemaEdit>, DbError> {
        let key = canonical_key(from, to, table);
        let from_is_a = is_a_side(from, &key);
        let diffs = self.diffs.get_mut(&key).ok_or(DbError::NoDiffTracking)?;

        let has_diffs = if from_is_a {
            !diffs.a_diffs.is_empty()
        } else {
            !diffs.b_diffs.is_empty()
        };
        if !has_diffs {
            return Ok(None);
        }

        let delta = if from_is_a {
            diffs.migrate_first_a_to_b().ok_or(DbError::MigrationFailed)?
        } else {
            diffs.migrate_first_b_to_a().ok_or(DbError::MigrationFailed)?
        };

        if !delta.is_id() {
            // Apply the schema edit to the target branch's table
            self.apply_schema_edit_to_branch(to, table, &delta)?;

            // Record in other diff channels for bilateral convergence.
            let other_keys: Vec<DiffKey> = self.diffs.keys()
                .filter(|k| key_table(k) == table
                    && key_has_branch(k, to)
                    && !key_has_branch(k, from))
                .cloned()
                .collect();
            for key in other_keys {
                if is_a_side(to, &key) {
                    self.diffs.get_mut(&key).unwrap().edit_a(&delta);
                } else {
                    self.diffs.get_mut(&key).unwrap().edit_b(&delta);
                }
            }
        }

        Ok(Some(delta))
    }

    /// Get all conflicts between two branches (across all tables).
    pub fn get_conflicts(
        &self,
        from: BranchId,
        to: BranchId,
    ) -> Vec<(String, SchemaConflict)> {
        let mut result = Vec::new();
        for (key, diffs) in &self.diffs {
            if !key_has_branch(key, from) || !key_has_branch(key, to) {
                continue;
            }
            let from_is_a = is_a_side(from, key);
            let conflicts = if from_is_a {
                diffs.all_conflicts()
            } else {
                // Swap perspective: b_diffs are "from", a_diffs are "to"
                let swapped = SchemaDifferences {
                    a_diffs: diffs.b_diffs.clone(),
                    b_diffs: diffs.a_diffs.clone(),
                };
                swapped.all_conflicts()
            };
            for c in conflicts {
                result.push((key_table(key).to_string(), c));
            }
        }
        result
    }

    /// Merge all diffs from branch `from` to branch `to` across all tracked tables.
    pub fn merge_all(
        &mut self,
        from: BranchId,
        to: BranchId,
    ) -> Result<Vec<(String, SchemaEdit)>, DbError> {
        // Ensure tracking exists between these branches
        self.diff_branches(from, to)?;

        let mut applied = Vec::new();

        // Collect tables that have any tracking between these branches
        let tables: Vec<String> = self.diffs.keys()
            .filter(|key| key_has_branch(key, from) && key_has_branch(key, to))
            .map(|key| key_table(key).to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for table in &tables {
            // Migrate all schema edits (which now include SetField)
            loop {
                match self.migrate_table(from, to, table)? {
                    Some(delta) if !delta.is_id() => applied.push((table.clone(), delta)),
                    Some(_) => continue, // Id delta (absorbed) — keep going
                    None => break,
                }
            }
        }

        // Post-merge: sort columns into canonical alphabetical order.
        // Records are unordered maps — column order is cosmetic. Using a
        // deterministic canonical order ensures convergence regardless of
        // merge direction. This is equivalent to a ReorderFields edit that
        // always produces the same result.
        for table in &tables {
            if let Some(b) = self.branches.get_mut(&to) {
                if let Some(t) = b.tables.get_mut(table.as_str()) {
                    canonicalize_column_order(t);
                }
            }
        }

        Ok(applied)
    }

    // ── Internal helpers ───────────────────────────────────────────────

    /// Record a schema edit in all active diff trackers AND the branch's edit log.
    fn record_schema_edit(&mut self, branch: BranchId, table: &str, edit: &SchemaEdit) {
        // Append to the branch's edit log (permanent history)
        if let Some(b) = self.branches.get_mut(&branch) {
            b.edit_log.entry(table.to_string()).or_default().push(edit.clone());
        }

        // Record in active diff trackers
        let keys: Vec<DiffKey> = self.diffs.keys().cloned().collect();
        for key in keys {
            if key_table(&key) != table || !key_has_branch(&key, branch) {
                continue;
            }
            if is_a_side(branch, &key) {
                self.diffs.get_mut(&key).unwrap().edit_a(edit);
            } else {
                self.diffs.get_mut(&key).unwrap().edit_b(edit);
            }
        }
    }

    /// Apply a SchemaEdit delta to a target branch's table (schema + all rows).
    fn apply_schema_edit_to_branch(
        &mut self,
        branch: BranchId,
        table: &str,
        edit: &SchemaEdit,
    ) -> Result<(), DbError> {
        let b = self.branches.get_mut(&branch).ok_or(DbError::BranchNotFound(branch))?;
        let t = b.tables.get_mut(table)
            .ok_or_else(|| DbError::TableNotFound(table.into()))?;

        match edit {
            SchemaEdit::AddField { name, ty } => {
                if let Some(idx) = t.schema.index_of(name) {
                    // Column already exists (same-name add from both sides).
                    // Update type and value to match the winning side.
                    t.schema.fields[idx].ty = *ty;
                    let default_value = default_for_type(*ty);
                    for doc in t.rows.values_mut() {
                        if idx < doc.fields.len() {
                            doc.fields[idx] = Field { ty: *ty, value: default_value.clone() };
                        }
                    }
                } else {
                    let idx = t.schema.fields.len();
                    t.schema.fields.push(NamedField {
                        name: name.clone(),
                        ty: *ty,
                    });
                    let default_value = default_for_type(*ty);
                    for doc in t.rows.values_mut() {
                        while doc.fields.len() <= idx {
                            doc.fields.push(Field::null(AtomicType::Del));
                        }
                        doc.fields[idx] = Field { ty: *ty, value: default_value.clone() };
                    }
                }
            }
            SchemaEdit::RemoveField { name } => {
                if let Some(idx) = t.schema.index_of(name) {
                    t.schema.fields[idx].ty = AtomicType::Del;
                    for doc in t.rows.values_mut() {
                        if idx < doc.fields.len() {
                            doc.fields[idx] = Field::null(AtomicType::Del);
                        }
                    }
                }
            }
            SchemaEdit::ConvertField { name, ty } => {
                if let Some(idx) = t.schema.index_of(name) {
                    t.schema.fields[idx].ty = *ty;
                    for doc in t.rows.values_mut() {
                        if idx < doc.fields.len() {
                            doc.fields[idx].ty = *ty;
                        }
                    }
                }
            }
            SchemaEdit::RenameField { old_name, new_name } => {
                if let Some(idx) = t.schema.fields.iter().position(|f| f.name == *old_name && f.ty != AtomicType::Del) {
                    // If new_name already exists, tombstone the old field
                    // (the OT's conflict rule should prevent this, but guard defensively)
                    if t.schema.index_of(new_name).is_some() {
                        t.schema.fields[idx].ty = AtomicType::Del;
                        for doc in t.rows.values_mut() {
                            if idx < doc.fields.len() {
                                doc.fields[idx] = Field::null(AtomicType::Del);
                            }
                        }
                    } else {
                        t.schema.fields[idx].name = new_name.clone();
                    }
                }
            }
            SchemaEdit::SetField { row: row_id, field, value } => {
                if let Some(idx) = t.schema.index_of(field) {
                    if let Some(doc) = t.rows.get_mut(row_id) {
                        if idx < doc.fields.len() {
                            doc.fields[idx].value = value.clone();
                        }
                    }
                }
            }
            SchemaEdit::Id => {}
        }
        Ok(())
    }
}

impl Database {
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

impl Default for Database {
    fn default() -> Self {
        Self::new()
    }
}

/// Reorder a table's columns (schema + row fields) to match the given name order.
/// Columns present in `order` come first in that order; any remaining columns
/// are appended at the end in their existing order.
/// Canonical column ordering: sort active fields alphabetically by name.
/// Tombstoned fields (Del) go to the end.
/// Records are unordered maps — this gives a deterministic order
/// that doesn't depend on merge direction.
fn canonicalize_column_order(table: &mut Table) {
    let n = table.schema.fields.len();
    if n == 0 {
        return;
    }

    // Build permutation: sort by (is_del, name)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        let fa = &table.schema.fields[a];
        let fb = &table.schema.fields[b];
        let a_del = fa.ty == AtomicType::Del;
        let b_del = fb.ty == AtomicType::Del;
        (a_del, &fa.name).cmp(&(b_del, &fb.name))
    });

    // Check if already canonical
    if indices.iter().enumerate().all(|(i, &p)| i == p) {
        return;
    }

    // Apply permutation to schema
    let old_fields = table.schema.fields.clone();
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        table.schema.fields[new_idx] = old_fields[old_idx].clone();
    }

    // Apply permutation to all rows
    for doc in table.rows.values_mut() {
        let old_row = doc.fields.clone();
        while doc.fields.len() < n {
            doc.fields.push(Field::null(AtomicType::Del));
        }
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            if old_idx < old_row.len() {
                doc.fields[new_idx] = old_row[old_idx].clone();
            } else {
                doc.fields[new_idx] = Field::null(AtomicType::Del);
            }
        }
    }
}

fn default_for_type(ty: AtomicType) -> Value {
    match ty {
        AtomicType::Num => Value::Num(0.0),
        AtomicType::Str => Value::Str(String::new()),
        AtomicType::Bool => Value::Bool(false),
        AtomicType::Del => Value::Null,
    }
}

fn json_to_value(j: &JsonValue) -> Value {
    match j {
        JsonValue::Number(n) => Value::Num(n.as_f64().unwrap_or(0.0)),
        JsonValue::String(s) => Value::Str(s.clone()),
        JsonValue::Bool(b) => Value::Bool(*b),
        JsonValue::Null => Value::Null,
        _ => Value::Error,
    }
}

fn value_to_json(value: &Value) -> JsonValue {
    match value {
        Value::Num(n) => JsonValue::Number(
            serde_json::Number::from_f64(*n).unwrap_or(serde_json::Number::from(0)),
        ),
        Value::Str(s) => JsonValue::String(s.clone()),
        Value::Bool(b) => JsonValue::Bool(*b),
        Value::Null => JsonValue::Null,
        Value::Error => JsonValue::String("<error>".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_branch_table_row() {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "users", vec![("name", AtomicType::Str), ("age", AtomicType::Num)]).unwrap();

        let r1 = db.insert_row(main, "users", vec![
            ("name", Value::Str("Alice".into())),
            ("age", Value::Num(30.0)),
        ]).unwrap();

        let view = db.get_row(main, "users", r1).unwrap();
        assert_eq!(view.fields[0], ("name".to_string(), serde_json::json!("Alice")));
        assert_eq!(view.fields[1], ("age".to_string(), serde_json::json!(30.0)));
    }

    #[test]
    fn test_schema_edit_propagates_to_rows() {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "users", vec![("name", AtomicType::Str)]).unwrap();

        let r1 = db.insert_row(main, "users", vec![("name", Value::Str("Alice".into()))]).unwrap();
        let r2 = db.insert_row(main, "users", vec![("name", Value::Str("Bob".into()))]).unwrap();

        db.add_column(main, "users", "age", AtomicType::Num).unwrap();

        let v1 = db.get_row(main, "users", r1).unwrap();
        assert_eq!(v1.fields.len(), 2);
        assert_eq!(v1.fields[1].0, "age");
        assert_eq!(v1.fields[1].1, serde_json::json!(0.0)); // default for Num

        let v2 = db.get_row(main, "users", r2).unwrap();
        assert_eq!(v2.fields.len(), 2);
    }

    #[test]
    fn test_fork_and_diverge() {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "users", vec![("name", AtomicType::Str)]).unwrap();
        db.insert_row(main, "users", vec![("name", Value::Str("Alice".into()))]).unwrap();

        let fork = db.fork_branch(main, "fork").unwrap();

        // Add column on main
        db.add_column(main, "users", "age", AtomicType::Num).unwrap();

        // Fork should NOT have the column (it was forked before the edit)
        let fork_view = db.get_table_view(fork, "users").unwrap();
        assert_eq!(fork_view.columns.len(), 1);

        let main_view = db.get_table_view(main, "users").unwrap();
        assert_eq!(main_view.columns.len(), 2);
    }

    #[test]
    fn test_fork_diff_merge() {
        let mut db = Database::new();
        let main = db.create_branch("main");
        db.create_table(main, "users", vec![("name", AtomicType::Str)]).unwrap();
        db.insert_row(main, "users", vec![("name", Value::Str("Alice".into()))]).unwrap();

        let alice = db.fork_branch(main, "alice").unwrap();
        let bob = db.fork_branch(main, "bob").unwrap();

        // Start tracking
        db.diff_branches(alice, bob).unwrap();

        // Alice adds email
        db.add_column(alice, "users", "email", AtomicType::Str).unwrap();

        // Bob adds age
        db.add_column(bob, "users", "age", AtomicType::Num).unwrap();

        // Check diffs — should have schema diffs for users table
        let diffs = db.get_diffs(alice, bob);
        assert!(!diffs.is_empty(), "should have diffs");
        let schema_diffs: Vec<_> = diffs.iter().filter(|(n, _, _)| n == "users").collect();
        assert!(!schema_diffs.is_empty(), "should have users diffs");

        // alice has edits, bob has edits
        let (_, alice_count, bob_count) = schema_diffs[0];
        assert!(*alice_count > 0, "alice should have diffs");
        assert!(*bob_count > 0, "bob should have diffs");

        // Merge alice→bob
        let applied = db.merge_all(alice, bob).unwrap();
        assert!(!applied.is_empty());

        // Bob should now have email column
        let bob_view = db.get_table_view(bob, "users").unwrap();
        let col_names: Vec<&str> = bob_view.columns.iter().map(|(n, _)| n.as_str()).collect();
        assert!(col_names.contains(&"email"), "bob should have email after merge: {:?}", col_names);
        assert!(col_names.contains(&"age"), "bob should still have age: {:?}", col_names);

        // Merge bob→alice
        let applied = db.merge_all(bob, alice).unwrap();
        assert!(!applied.is_empty());

        // Alice should now have age column
        let alice_view = db.get_table_view(alice, "users").unwrap();
        let col_names: Vec<&str> = alice_view.columns.iter().map(|(n, _)| n.as_str()).collect();
        assert!(col_names.contains(&"email"), "alice should have email: {:?}", col_names);
        assert!(col_names.contains(&"age"), "alice should have age after merge: {:?}", col_names);
    }
}
