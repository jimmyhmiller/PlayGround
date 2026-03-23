use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};

use crate::database::*;
use crate::types::AtomicType;
use crate::types::Value;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Message {
    // Branch ops
    CreateBranch { name: String },
    ForkBranch { source: u64, name: String },
    ListBranches,

    // Table ops
    CreateTable { branch: u64, table: String, columns: Vec<(String, String)> },

    // Schema ops
    AddColumn { branch: u64, table: String, name: String, #[serde(rename = "type")] ty: String },
    RemoveColumn { branch: u64, table: String, name: String },
    RenameColumn { branch: u64, table: String, old: String, new: String },
    ConvertColumn { branch: u64, table: String, name: String, to: String },

    // Row ops
    InsertRow { branch: u64, table: String, data: Vec<(String, JsonValue)> },
    SetField { branch: u64, table: String, row: u64, field: String, value: JsonValue },
    GetRow { branch: u64, table: String, row: u64 },
    ListRows { branch: u64, table: String },

    // Row deletion
    DeleteRow { branch: u64, table: String, row: u64 },

    // Diff & merge
    DiffBranches { branch_a: u64, branch_b: u64 },
    GetConflicts { from: u64, to: u64 },
    Migrate { from: u64, to: u64, table: String },
    MergeAll { from: u64, to: u64 },

    // Inspect
    GetSchema { branch: u64, table: String },
    ListTables { branch: u64 },

    // Persistence
    Save { path: String },
    Load { path: String },
}

#[derive(Debug, Serialize)]
pub struct Response {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(flatten)]
    pub data: serde_json::Map<String, JsonValue>,
}

impl Response {
    fn ok(data: serde_json::Map<String, JsonValue>) -> Self {
        Response { ok: true, error: None, data }
    }
    fn err(msg: impl ToString) -> Self {
        Response { ok: false, error: Some(msg.to_string()), data: serde_json::Map::new() }
    }
}

fn parse_type(s: &str) -> Result<AtomicType, String> {
    match s {
        "Str" | "str" | "string" | "String" => Ok(AtomicType::Str),
        "Num" | "num" | "number" | "Number" => Ok(AtomicType::Num),
        "Bool" | "bool" | "boolean" | "Boolean" => Ok(AtomicType::Bool),
        other => Err(format!("unknown type: {}", other)),
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

pub fn dispatch(db: &mut Database, input: &str) -> Response {
    let msg: Message = match serde_json::from_str(input) {
        Ok(m) => m,
        Err(e) => return Response::err(format!("parse error: {}", e)),
    };

    match msg {
        Message::CreateBranch { name } => {
            let id = db.create_branch(&name);
            let mut m = serde_json::Map::new();
            m.insert("branch_id".into(), json!(id));
            Response::ok(m)
        }

        Message::ForkBranch { source, name } => match db.fork_branch(source, &name) {
            Ok(id) => {
                let mut m = serde_json::Map::new();
                m.insert("branch_id".into(), json!(id));
                Response::ok(m)
            }
            Err(e) => Response::err(e),
        },

        Message::ListBranches => {
            let branches: Vec<JsonValue> = db.list_branches().iter()
                .map(|(id, name)| json!({"id": id, "name": name}))
                .collect();
            let mut m = serde_json::Map::new();
            m.insert("branches".into(), JsonValue::Array(branches));
            Response::ok(m)
        }

        Message::CreateTable { branch, table, columns } => {
            let parsed: Result<Vec<(&str, AtomicType)>, String> = columns.iter()
                .map(|(n, t)| parse_type(t).map(|ty| (n.as_str(), ty)))
                .collect();
            match parsed {
                Ok(cols) => match db.create_table(branch, &table, cols) {
                    Ok(()) => Response::ok(serde_json::Map::new()),
                    Err(e) => Response::err(e),
                },
                Err(e) => Response::err(e),
            }
        }

        Message::AddColumn { branch, table, name, ty } => {
            match parse_type(&ty) {
                Ok(at) => match db.add_column(branch, &table, &name, at) {
                    Ok(edits) => {
                        let mut m = serde_json::Map::new();
                        m.insert("edits".into(), json!(format!("{:?}", edits)));
                        Response::ok(m)
                    }
                    Err(e) => Response::err(e),
                },
                Err(e) => Response::err(e),
            }
        }

        Message::RemoveColumn { branch, table, name } => {
            match db.remove_column(branch, &table, &name) {
                Ok(edit) => {
                    let mut m = serde_json::Map::new();
                    m.insert("edit".into(), json!(format!("{:?}", edit)));
                    Response::ok(m)
                }
                Err(e) => Response::err(e),
            }
        }

        Message::RenameColumn { branch, table, old, new } => {
            match db.rename_column(branch, &table, &old, &new) {
                Ok(edit) => {
                    let mut m = serde_json::Map::new();
                    m.insert("edit".into(), json!(format!("{:?}", edit)));
                    Response::ok(m)
                }
                Err(e) => Response::err(e),
            }
        }

        Message::ConvertColumn { branch, table, name, to } => {
            match parse_type(&to) {
                Ok(at) => match db.convert_column(branch, &table, &name, at) {
                    Ok(edit) => {
                        let mut m = serde_json::Map::new();
                        m.insert("edit".into(), json!(format!("{:?}", edit)));
                        Response::ok(m)
                    }
                    Err(e) => Response::err(e),
                },
                Err(e) => Response::err(e),
            }
        }

        Message::InsertRow { branch, table, data } => {
            let values: Vec<(&str, Value)> = data.iter()
                .map(|(k, v)| (k.as_str(), json_to_value(v)))
                .collect();
            match db.insert_row(branch, &table, values) {
                Ok(id) => {
                    let mut m = serde_json::Map::new();
                    m.insert("row_id".into(), json!(id));
                    Response::ok(m)
                }
                Err(e) => Response::err(e),
            }
        }

        Message::SetField { branch, table, row, field, value } => {
            let v = json_to_value(&value);
            match db.set_field(branch, &table, row, &field, v) {
                Ok(()) => Response::ok(serde_json::Map::new()),
                Err(e) => Response::err(e),
            }
        }

        Message::DeleteRow { branch, table, row } => {
            match db.delete_row(branch, &table, row) {
                Ok(()) => Response::ok(serde_json::Map::new()),
                Err(e) => Response::err(e),
            }
        }

        Message::GetRow { branch, table, row } => {
            match db.get_row(branch, &table, row) {
                Ok(view) => {
                    let mut m = serde_json::Map::new();
                    m.insert("row_id".into(), json!(view.row_id));
                    let fields: serde_json::Map<String, JsonValue> = view.fields.into_iter().collect();
                    m.insert("fields".into(), JsonValue::Object(fields));
                    Response::ok(m)
                }
                Err(e) => Response::err(e),
            }
        }

        Message::ListRows { branch, table } => {
            match db.list_rows(branch, &table) {
                Ok(views) => {
                    let rows: Vec<JsonValue> = views.into_iter().map(|v| {
                        let fields: serde_json::Map<String, JsonValue> = v.fields.into_iter().collect();
                        json!({"row_id": v.row_id, "fields": fields})
                    }).collect();
                    let mut m = serde_json::Map::new();
                    m.insert("rows".into(), JsonValue::Array(rows));
                    Response::ok(m)
                }
                Err(e) => Response::err(e),
            }
        }

        Message::DiffBranches { branch_a, branch_b } => {
            match db.diff_branches(branch_a, branch_b) {
                Ok(()) => {
                    let diffs = db.get_diffs(branch_a, branch_b);
                    let summary: Vec<JsonValue> = diffs.into_iter()
                        .map(|(table, a_count, b_count)| json!({
                            "table": table,
                            "a_diffs": a_count,
                            "b_diffs": b_count,
                        }))
                        .collect();
                    let mut m = serde_json::Map::new();
                    m.insert("diffs".into(), JsonValue::Array(summary));
                    Response::ok(m)
                }
                Err(e) => Response::err(e),
            }
        }

        Message::GetConflicts { from, to } => {
            let conflicts = db.get_conflicts(from, to);
            let items: Vec<JsonValue> = conflicts.into_iter()
                .map(|(location, c)| json!({
                    "location": location,
                    "from_edit": format!("{:?}", c.from_edit),
                    "to_edit": format!("{:?}", c.to_edit),
                }))
                .collect();
            let mut m = serde_json::Map::new();
            m.insert("conflicts".into(), JsonValue::Array(items));
            Response::ok(m)
        }

        Message::Migrate { from, to, table } => {
            match db.migrate_table(from, to, &table) {
                Ok(Some(delta)) => {
                    let mut m = serde_json::Map::new();
                    m.insert("edit".into(), json!(format!("{:?}", delta)));
                    m.insert("is_id".into(), json!(delta.is_id()));
                    Response::ok(m)
                }
                Ok(None) => {
                    let mut m = serde_json::Map::new();
                    m.insert("message".into(), json!("no diffs to migrate"));
                    Response::ok(m)
                }
                Err(e) => Response::err(e),
            }
        }

        Message::MergeAll { from, to } => {
            match db.merge_all(from, to) {
                Ok(applied) => {
                    let edits: Vec<JsonValue> = applied.into_iter()
                        .map(|(table, edit)| json!({"table": table, "edit": format!("{:?}", edit)}))
                        .collect();
                    let mut m = serde_json::Map::new();
                    m.insert("applied".into(), JsonValue::Array(edits));
                    Response::ok(m)
                }
                Err(e) => Response::err(e),
            }
        }

        Message::GetSchema { branch, table } => {
            match db.get_table_view(branch, &table) {
                Ok(view) => {
                    let mut m = serde_json::Map::new();
                    let cols: serde_json::Map<String, JsonValue> = view.columns.into_iter()
                        .map(|(k, v)| (k, JsonValue::String(v)))
                        .collect();
                    m.insert("table".into(), json!(view.table));
                    m.insert("columns".into(), JsonValue::Object(cols));
                    m.insert("row_count".into(), json!(view.row_count));
                    Response::ok(m)
                }
                Err(e) => Response::err(e),
            }
        }

        Message::ListTables { branch } => {
            match db.list_tables(branch) {
                Ok(tables) => {
                    let mut m = serde_json::Map::new();
                    m.insert("tables".into(), json!(tables));
                    Response::ok(m)
                }
                Err(e) => Response::err(e),
            }
        }

        Message::Save { path } => {
            match db.save(std::path::Path::new(&path)) {
                Ok(()) => {
                    let mut m = serde_json::Map::new();
                    m.insert("path".into(), json!(path));
                    Response::ok(m)
                }
                Err(e) => Response::err(e),
            }
        }

        Message::Load { path } => {
            match Database::load(std::path::Path::new(&path)) {
                Ok(loaded) => {
                    *db = loaded;
                    let mut m = serde_json::Map::new();
                    m.insert("path".into(), json!(path));
                    Response::ok(m)
                }
                Err(e) => Response::err(e),
            }
        }
    }
}
