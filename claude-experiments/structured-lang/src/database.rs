use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::apply::{apply, conform};
use crate::diff::{Conflict, ConflictResolver, Differences, FromWins};
use crate::schema::*;
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
/// Rows are stored by RowId in a HashMap for fast lookup, but also tracked
/// positionally in `row_order` for the OT algebra. InsertRow = Ins on the
/// row_order vec, DeleteRow = tombstone in row_order.
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
    fn ins(&mut self) -> u64 {
        let id = self.ins;
        self.ins += 1;
        id
    }
}

/// Diff channel: three levels of tracking per table.
/// - Schema: column add/remove/convert/rename (applies to all rows)
/// - Rows: row insert/delete (row manifest level)
/// - RowData(RowId): per-row value changes (Set edits)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum DiffChannel {
    Schema,
    Rows,
    RowData(RowId),
}

/// Canonical diff key: (min_branch, max_branch, table_name, channel).
/// The "a" side of Differences is always the min branch id.
type DiffKey = (BranchId, BranchId, String, DiffChannel);

fn canonical_key(a: BranchId, b: BranchId, table: &str, channel: DiffChannel) -> DiffKey {
    if a <= b {
        (a, b, table.to_string(), channel)
    } else {
        (b, a, table.to_string(), channel)
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
    /// Serialized as a list of (key, diffs) pairs since tuple keys can't be JSON map keys.
    #[serde(
        serialize_with = "serialize_diffs",
        deserialize_with = "deserialize_diffs"
    )]
    diffs: HashMap<DiffKey, Differences>,
    ids: IdGen,
}

fn serialize_diffs<S: serde::Serializer>(
    diffs: &HashMap<DiffKey, Differences>,
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
) -> Result<HashMap<DiffKey, Differences>, D::Error> {
    let entries: Vec<(DiffKey, Differences)> = Vec::deserialize(deserializer)?;
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
        });
        id
    }

    pub fn fork_branch(&mut self, source: BranchId, name: &str) -> Result<BranchId, DbError> {
        let src = self.branches.get(&source)
            .ok_or(DbError::BranchNotFound(source))?
            .clone();
        let id = self.ids.branch();
        self.branches.insert(id, Branch {
            name: name.to_string(),
            tables: src.tables,
        });
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
        let ins_id = self.ids.ins();
        let row_idx;
        {
            let b = self.get_branch_mut(branch)?;
            let t = b.tables.get_mut(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
            let mut doc = t.schema.empty_document();
            for (name, value) in data {
                if let Some(idx) = t.schema.index_of(name) {
                    doc.fields[idx].value = value;
                }
            }
            row_idx = t.row_order.len();
            t.rows.insert(id, doc);
            t.row_order.push(id);
        }
        // Record Ins on the row manifest
        let edit = Edit::Ins { idx: row_idx, ty: AtomicType::Num, id: ins_id };
        self.record_edit(branch, table, &edit, &DiffChannel::Rows);
        Ok(id)
    }

    pub fn delete_row(
        &mut self,
        branch: BranchId,
        table: &str,
        row: RowId,
    ) -> Result<(), DbError> {
        let row_idx;
        {
            let b = self.get_branch_mut(branch)?;
            let t = b.tables.get_mut(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
            if t.rows.remove(&row).is_none() {
                return Err(DbError::RowNotFound(row));
            }
            row_idx = t.row_order.iter().position(|&id| id == row);
            if let Some(pos) = row_idx {
                t.row_order[pos] = u64::MAX;
            }
        }
        // Record Conv{Del} on the row manifest
        if let Some(idx) = row_idx {
            let edit = Edit::Conv { idx, ty: AtomicType::Del };
            self.record_edit(branch, table, &edit, &DiffChannel::Rows);
        }
        Ok(())
    }

    /// Set a value on a specific row. Records a Set edit in the per-row
    /// diff channel (RowData). Each row has its own Differences tracker,
    /// so Set edits on different rows don't conflict.
    pub fn set_field(
        &mut self,
        branch: BranchId,
        table: &str,
        row: RowId,
        field: &str,
        value: Value,
    ) -> Result<(), DbError> {
        let idx;
        {
            let b = self.get_branch_mut(branch)?;
            let t = b.tables.get_mut(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
            idx = t.schema.index_of(field)
                .ok_or_else(|| DbError::FieldNotFound(field.into()))?;
            let doc = t.rows.get_mut(&row).ok_or(DbError::RowNotFound(row))?;
            doc.fields[idx].value = value.clone();
        }
        let edit = Edit::Set { idx, value };
        self.record_edit(branch, table, &edit, &DiffChannel::RowData(row));
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

    /// Apply a schema edit to a table. Returns the Edit produced (if any).
    /// The edit is applied to all rows and recorded in any active diff trackers.
    pub fn add_column(
        &mut self,
        branch: BranchId,
        table: &str,
        name: &str,
        ty: AtomicType,
    ) -> Result<Vec<Edit>, DbError> {
        let ins_id = self.ids.ins();
        let edits = {
            let b = self.get_branch_mut(branch)?;
            let t = b.tables.get_mut(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
            let mut edits = t.schema.add_field(name, ty)?;
            // Override ins id with global one
            if let Some(Edit::Ins { idx, ty, .. }) = edits.first() {
                edits[0] = Edit::Ins { idx: *idx, ty: *ty, id: ins_id };
            }
            // Apply structural edits (Ins) to all rows — Rename is schema-only
            for edit in &edits {
                if matches!(edit, Edit::Ins { .. }) {
                    for doc in t.rows.values_mut() {
                        *doc = apply(doc, edit).ok_or(DbError::ApplyFailed)?;
                    }
                }
            }
            edits
        };
        for edit in &edits {
            self.record_edit(branch, table, edit, &DiffChannel::Schema);
        }
        Ok(edits)
    }

    pub fn remove_column(
        &mut self,
        branch: BranchId,
        table: &str,
        name: &str,
    ) -> Result<Edit, DbError> {
        let edit = {
            let b = self.get_branch_mut(branch)?;
            let t = b.tables.get_mut(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
            let edit = t.schema.remove_field(name)?;
            for doc in t.rows.values_mut() {
                *doc = apply(doc, &edit).ok_or(DbError::ApplyFailed)?;
            }
            edit
        };
        self.record_edit(branch, table, &edit, &DiffChannel::Schema);
        Ok(edit)
    }

    pub fn convert_column(
        &mut self,
        branch: BranchId,
        table: &str,
        name: &str,
        to: AtomicType,
    ) -> Result<Edit, DbError> {
        let edit = {
            let b = self.get_branch_mut(branch)?;
            let t = b.tables.get_mut(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
            let edit = t.schema.convert_field(name, to)?;
            for doc in t.rows.values_mut() {
                *doc = apply(doc, &edit).ok_or(DbError::ApplyFailed)?;
            }
            edit
        };
        self.record_edit(branch, table, &edit, &DiffChannel::Schema);
        Ok(edit)
    }

    pub fn rename_column(
        &mut self,
        branch: BranchId,
        table: &str,
        old: &str,
        new: &str,
    ) -> Result<Edit, DbError> {
        let edit = {
            let b = self.get_branch_mut(branch)?;
            let t = b.tables.get_mut(table).ok_or_else(|| DbError::TableNotFound(table.into()))?;
            t.schema.rename_field(old, new)?
        };
        self.record_edit(branch, table, &edit, &DiffChannel::Schema);
        Ok(edit)
    }

    // ── Diff & merge ───────────────────────────────────────────────────

    /// Start tracking diffs between two branches for all shared tables.
    pub fn diff_branches(&mut self, a: BranchId, b: BranchId) -> Result<(), DbError> {
        let tables_a: Vec<String> = {
            let ba = self.get_branch(a)?;
            ba.tables.keys().cloned().collect()
        };
        let tables_b: Vec<String> = {
            let bb = self.get_branch(b)?;
            bb.tables.keys().cloned().collect()
        };
        // Collect shared row ids before creating trackers
        let mut shared_rows: HashMap<String, Vec<RowId>> = HashMap::new();
        for table in &tables_a {
            if tables_b.contains(table) {
                let ba = self.get_branch(a)?;
                let row_ids: Vec<RowId> = ba.tables.get(table.as_str())
                    .map(|t| t.rows.keys().cloned().collect())
                    .unwrap_or_default();
                shared_rows.insert(table.clone(), row_ids);
            }
        }

        for (table, row_ids) in &shared_rows {
            let schema_key = canonical_key(a, b, table, DiffChannel::Schema);
            self.diffs.entry(schema_key).or_insert_with(Differences::new);
            let rows_key = canonical_key(a, b, table, DiffChannel::Rows);
            self.diffs.entry(rows_key).or_insert_with(Differences::new);
            // Per-row data trackers for existing rows
            for &row_id in row_ids {
                let row_key = canonical_key(a, b, table, DiffChannel::RowData(row_id));
                self.diffs.entry(row_key).or_insert_with(Differences::new);
            }
        }
        Ok(())
    }

    /// Get diff summary between two branches.
    /// Returns (table, from_diffs_count, to_diffs_count) where "from" is the
    /// first argument and "to" is the second.
    pub fn get_diffs(&self, from: BranchId, to: BranchId) -> Vec<(String, usize, usize)> {
        let mut result = Vec::new();
        for (key, diffs) in &self.diffs {
            if key.3 != DiffChannel::Schema {
                continue;
            }
            if !key_has_branch(key, from) || !key_has_branch(key, to) {
                continue;
            }
            let (from_count, to_count) = if is_a_side(from, key) {
                (diffs.a_diffs.len(), diffs.b_diffs.len())
            } else {
                (diffs.b_diffs.len(), diffs.a_diffs.len())
            };
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
    ) -> Result<Option<Edit>, DbError> {
        let key = canonical_key(from, to, table, DiffChannel::Schema);
        let from_is_a = is_a_side(from, &key);
        let diffs = self.diffs.get_mut(&key).ok_or(DbError::NoDiffTracking)?;

        // Check if the "from" side has diffs to migrate
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
            // Get the field name from the source branch
            let field_name = if let Edit::Ins { idx, .. } = delta {
                self.branches.get(&from)
                    .and_then(|fb| fb.tables.get(table))
                    .and_then(|ft| ft.schema.name_at(idx).map(|s| s.to_string()))
                    .unwrap_or_else(|| format!("col_{}", idx))
            } else {
                String::new()
            };

            // Apply the edit to the target branch's table
            let b = self.branches.get_mut(&to).ok_or(DbError::BranchNotFound(to))?;
            let t = b.tables.get_mut(table)
                .ok_or_else(|| DbError::TableNotFound(table.into()))?;

            match &delta {
                Edit::Ins { idx, ty, .. } => {
                    t.schema.fields.insert(*idx, NamedField {
                        name: field_name,
                        ty: *ty,
                    });
                }
                Edit::Conv { idx, ty } => {
                    if *idx < t.schema.fields.len() {
                        t.schema.fields[*idx].ty = *ty;
                    }
                }
                Edit::Rename { idx, name } => {
                    if *idx < t.schema.fields.len() {
                        t.schema.fields[*idx].name = name.clone();
                    }
                }
                _ => {}
            }
            for doc in t.rows.values_mut() {
                if let Some(new_doc) = apply(doc, &delta) {
                    *doc = new_doc;
                }
            }
        }

        Ok(Some(delta))
    }

    /// Get all conflicts between two branches (across all tables and channels).
    pub fn get_conflicts(
        &self,
        from: BranchId,
        to: BranchId,
    ) -> Vec<(String, Conflict)> {
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
                let swapped = Differences {
                    a_diffs: diffs.b_diffs.clone(),
                    b_diffs: diffs.a_diffs.clone(),
                };
                swapped.all_conflicts()
            };
            for c in conflicts {
                let channel_label = match &key.3 {
                    DiffChannel::Schema => "schema".to_string(),
                    DiffChannel::Rows => "rows".to_string(),
                    DiffChannel::RowData(id) => format!("row:{}", id),
                };
                result.push((
                    format!("{}/{}", key_table(key), channel_label),
                    c,
                ));
            }
        }
        result
    }

    /// Merge all diffs from branch `from` to branch `to` across all tracked tables.
    /// Uses the provided ConflictResolver to decide which side wins each conflict.
    /// Processes three levels in order:
    /// 1. Schema edits (column add/remove/convert/rename)
    /// 2. Row manifest edits (insert/delete rows)
    /// 3. Per-row data edits (Set values)
    pub fn merge_all(
        &mut self,
        from: BranchId,
        to: BranchId,
    ) -> Result<Vec<(String, Edit)>, DbError> {
        self.merge_all_with(&mut FromWins, from, to)
    }

    /// Merge with a custom conflict resolver.
    pub fn merge_all_with(
        &mut self,
        _resolver: &mut dyn ConflictResolver,
        from: BranchId,
        to: BranchId,
    ) -> Result<Vec<(String, Edit)>, DbError> {
        let mut applied = Vec::new();

        // Collect tables that have any tracking between these branches
        let tables: Vec<String> = self.diffs.keys()
            .filter(|key| key_has_branch(key, from) && key_has_branch(key, to))
            .map(|key| key_table(key).to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for table in &tables {
            // Level 1: Schema edits
            let schema_deltas = self.migrate_channel(from, to, table, DiffChannel::Schema)?;
            for delta in &schema_deltas {
                applied.push((table.clone(), delta.clone()));
            }

            // Level 2: Row manifest edits (not yet wired to create/delete rows on target)
            // TODO: wire row Ins/Del through to actually create/remove rows

            // Level 3: Per-row data edits
            let row_keys: Vec<RowId> = self.diffs.keys()
                .filter(|key| key_has_branch(key, from) && key_has_branch(key, to)
                    && key_table(key) == table.as_str()
                    && matches!(key.3, DiffChannel::RowData(_)))
                .filter_map(|key| if let DiffChannel::RowData(rid) = key.3 { Some(rid) } else { None })
                .collect();

            for row_id in row_keys {
                let row_deltas = self.migrate_channel(from, to, table, DiffChannel::RowData(row_id))?;
                for delta in &row_deltas {
                    // Apply Set edit to the specific row on the target branch
                    if let Edit::Set { idx, value } = delta {
                        if let Some(branch) = self.branches.get_mut(&to) {
                            if let Some(t) = branch.tables.get_mut(table.as_str()) {
                                if let Some(doc) = t.rows.get_mut(&row_id) {
                                    if *idx < doc.fields.len() {
                                        doc.fields[*idx].value = value.clone();
                                    }
                                }
                            }
                        }
                    }
                    applied.push((table.clone(), delta.clone()));
                }
            }
        }
        Ok(applied)
    }

    /// Migrate all edits in a specific channel from one branch to another.
    /// Returns the list of deltas that were applied.
    fn migrate_channel(
        &mut self,
        from: BranchId,
        to: BranchId,
        table: &str,
        channel: DiffChannel,
    ) -> Result<Vec<Edit>, DbError> {
        let mut deltas = Vec::new();
        loop {
            let key = canonical_key(from, to, table, channel.clone());
            let from_is_a = is_a_side(from, &key);
            let has_diffs = self.diffs.get(&key)
                .map(|d| if from_is_a { !d.a_diffs.is_empty() } else { !d.b_diffs.is_empty() })
                .unwrap_or(false);
            if !has_diffs {
                break;
            }

            // For Schema channel, use migrate_table (handles schema metadata updates).
            // For other channels, do a raw diff migration.
            if matches!(channel, DiffChannel::Schema) {
                match self.migrate_table(from, to, table)? {
                    Some(delta) if !delta.is_id() => deltas.push(delta),
                    _ => break,
                }
            } else {
                let diffs = self.diffs.get_mut(&key).ok_or(DbError::NoDiffTracking)?;
                let delta = if from_is_a {
                    diffs.migrate_first_a_to_b()
                } else {
                    diffs.migrate_first_b_to_a()
                };
                match delta {
                    Some(d) if !d.is_id() => deltas.push(d),
                    _ => break,
                }
            }
        }
        Ok(deltas)
    }

    // ── Internal helpers ───────────────────────────────────────────────

    /// Record an edit in all active diff trackers for this branch+table.
    fn record_edit(&mut self, branch: BranchId, table: &str, edit: &Edit, channel: &DiffChannel) {
        let keys: Vec<DiffKey> = self.diffs.keys().cloned().collect();
        for key in keys {
            if key_table(&key) != table {
                continue;
            }
            if key.3 != *channel {
                continue;
            }
            if is_a_side(branch, &key) {
                self.diffs.get_mut(&key).unwrap().edit_a(&edit);
            } else if key_has_branch(&key, branch) {
                self.diffs.get_mut(&key).unwrap().edit_b(&edit);
            }
        }
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

        // Check diffs
        let diffs = db.get_diffs(alice, bob);
        assert_eq!(diffs.len(), 1); // one table tracked
        assert_eq!(diffs[0].0, "users");
        assert_eq!(diffs[0].1, 1); // alice has 1 diff
        assert_eq!(diffs[0].2, 1); // bob has 1 diff

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
