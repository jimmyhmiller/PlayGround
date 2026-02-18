use std::collections::HashMap;

use crate::datom::{Datom, EntityId, TxId, Value};
use crate::index;
use crate::query::{Pattern, Query, WhereClause};
use crate::storage::StorageBackend;

/// A set of variable bindings produced during query execution.
type Bindings = HashMap<String, Value>;

/// Query result: rows of bound values in `find` order.
#[derive(Debug)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}

/// Execute a query against the storage backend.
pub async fn execute_query(
    storage: &dyn StorageBackend,
    query: &Query,
) -> Result<QueryResult, String> {
    let mut binding_sets: Vec<Bindings> = vec![HashMap::new()];

    for clause in &query.where_clauses {
        let mut next_binding_sets = Vec::new();

        for bindings in &binding_sets {
            let matches = evaluate_clause(storage, clause, bindings, query.as_of).await?;
            next_binding_sets.extend(matches);
        }

        binding_sets = next_binding_sets;
        if binding_sets.is_empty() {
            break;
        }
    }

    // Project find variables
    let columns = query.find.clone();
    let rows = binding_sets
        .into_iter()
        .map(|bindings| {
            columns
                .iter()
                .map(|var| {
                    bindings
                        .get(var)
                        .cloned()
                        .unwrap_or(Value::String("<unbound>".to_string()))
                })
                .collect()
        })
        .collect();

    Ok(QueryResult { columns, rows })
}

/// Evaluate a single where clause against current bindings.
/// Returns extended binding sets for each matching entity.
async fn evaluate_clause(
    storage: &dyn StorageBackend,
    clause: &WhereClause,
    bindings: &Bindings,
    as_of: Option<TxId>,
) -> Result<Vec<Bindings>, String> {
    // Check if the entity variable is already bound
    let bound_entity = bindings.get(&clause.bind).and_then(|v| {
        if let Value::Ref(id) = v {
            Some(*id)
        } else {
            None
        }
    });

    let entities = if let Some(eid) = bound_entity {
        // Entity already bound — just check this one
        vec![eid]
    } else {
        // Find all entities of this type via AVET index on __type
        find_entities_of_type(storage, &clause.entity_type, as_of).await?
    };

    let mut results = Vec::new();

    for eid in entities {
        // Load all current field values for this entity
        let fields = load_entity_fields(storage, eid, &clause.entity_type, as_of).await?;

        // Try to match all field patterns
        if let Some(new_bindings) =
            match_field_patterns(&clause.field_patterns, &fields, bindings)
        {
            let mut extended = bindings.clone();
            // Bind the entity variable as a Ref
            extended.insert(clause.bind.clone(), Value::Ref(eid));
            extended.extend(new_bindings);
            results.push(extended);
        }
    }

    Ok(results)
}

/// Find all entity IDs of a given type.
async fn find_entities_of_type(
    storage: &dyn StorageBackend,
    type_name: &str,
    as_of: Option<TxId>,
) -> Result<Vec<EntityId>, String> {
    let type_value = Value::String(type_name.to_string());
    let prefix = index::avet_attr_value_prefix("__type", &type_value);
    let end = index::prefix_end(&prefix);

    let entries = storage
        .scan(&prefix, &end)
        .await
        .map_err(|e| e.to_string())?;

    // Group datoms by entity, then resolve history properly
    let mut entity_datoms: HashMap<EntityId, Vec<Datom>> = HashMap::new();

    for (key, _) in &entries {
        if let Some(datom) = index::decode_datom_from_avet(key) {
            if let Some(max_tx) = as_of {
                if datom.tx > max_tx {
                    continue;
                }
            }
            entity_datoms.entry(datom.entity).or_default().push(datom);
        }
    }

    let mut entities = Vec::new();
    for (eid, mut datoms) in entity_datoms {
        // Sort by tx ascending, retracts before asserts within same tx
        datoms.sort_by(|a, b| a.tx.cmp(&b.tx).then_with(|| a.added.cmp(&b.added)));
        let mut exists = false;
        for d in datoms {
            exists = d.added;
        }
        if exists {
            entities.push(eid);
        }
    }

    Ok(entities)
}

/// Load all current field values for an entity.
/// Returns a map of field_name -> Value.
///
/// For enum fields stored as `EntityType/field.__tag` and `EntityType/field.Variant/subfield`,
/// the returned map contains entries like:
///   "field.__tag" => Value::String("Circle")
///   "field.Circle/radius" => Value::F64(5.0)
async fn load_entity_fields(
    storage: &dyn StorageBackend,
    entity: EntityId,
    entity_type: &str,
    as_of: Option<TxId>,
) -> Result<HashMap<String, Value>, String> {
    let prefix = index::eavt_entity_prefix(entity);
    let end = index::prefix_end(&prefix);

    let entries = storage
        .scan(&prefix, &end)
        .await
        .map_err(|e| e.to_string())?;

    // Group datoms by attribute, find latest value
    let mut attr_state: HashMap<String, Vec<Datom>> = HashMap::new();

    for (key, _) in &entries {
        if let Some(datom) = index::decode_datom_from_eavt(key) {
            if let Some(max_tx) = as_of {
                if datom.tx > max_tx {
                    continue;
                }
            }
            attr_state
                .entry(datom.attribute.clone())
                .or_default()
                .push(datom);
        }
    }

    let attr_prefix = format!("{}/", entity_type);

    // Use resolve_current_values to properly handle retract+assert history
    let resolved = crate::db::resolve_current_values(attr_state, as_of);

    let mut fields = HashMap::new();
    for (attr, value) in resolved {
        if !attr.starts_with(&attr_prefix) {
            continue;
        }
        let field_name = &attr[attr_prefix.len()..];
        fields.insert(field_name.to_string(), value);
    }

    Ok(fields)
}

/// Try to match field patterns against entity fields. Returns new bindings if successful.
fn match_field_patterns(
    patterns: &[(String, Pattern)],
    fields: &HashMap<String, Value>,
    existing_bindings: &Bindings,
) -> Option<Bindings> {
    let mut new_bindings = Bindings::new();

    for (field_name, pattern) in patterns {
        match pattern {
            Pattern::Variable(var) => {
                // For enum fields, a bare variable on an enum field binds the tag
                // Check if this is an enum field by looking for __tag
                let tag_key = format!("{}/__tag", field_name);
                if let Some(tag_val) = fields.get(&tag_key) {
                    // This is an enum field — bind the tag
                    if let Some(bound_value) = existing_bindings
                        .get(var)
                        .or_else(|| new_bindings.get(var))
                    {
                        if bound_value != tag_val {
                            return None;
                        }
                    } else {
                        new_bindings.insert(var.clone(), tag_val.clone());
                    }
                } else {
                    // Regular scalar field
                    if let Some(bound_value) = existing_bindings
                        .get(var)
                        .or_else(|| new_bindings.get(var))
                    {
                        let field_val = fields.get(field_name)?;
                        if bound_value != field_val {
                            return None;
                        }
                    } else {
                        let field_val = fields.get(field_name)?;
                        new_bindings.insert(var.clone(), field_val.clone());
                    }
                }
            }
            Pattern::Constant(expected) => {
                // For enum fields, a bare constant matches the tag
                let tag_key = format!("{}/__tag", field_name);
                if let Some(tag_val) = fields.get(&tag_key) {
                    if tag_val != expected {
                        return None;
                    }
                } else {
                    let field_val = fields.get(field_name)?;
                    if field_val != expected {
                        return None;
                    }
                }
            }
            Pattern::Predicate { op, value } => {
                let field_val = fields.get(field_name)?;
                if !op.evaluate(field_val, value) {
                    return None;
                }
            }
            Pattern::EnumMatch {
                variant,
                field_patterns,
            } => {
                // Match enum tag
                let tag_key = format!("{}/__tag", field_name);
                let tag_val = fields.get(&tag_key)?;

                match variant.as_ref() {
                    Pattern::Variable(var) => {
                        if let Some(bound_value) = existing_bindings
                            .get(var)
                            .or_else(|| new_bindings.get(var))
                        {
                            if bound_value != tag_val {
                                return None;
                            }
                        } else {
                            new_bindings.insert(var.clone(), tag_val.clone());
                        }
                    }
                    Pattern::Constant(expected) => {
                        if tag_val != expected {
                            return None;
                        }
                    }
                    _ => return None,
                }

                // Get the variant name to construct field keys
                let variant_name = match tag_val {
                    Value::String(s) => s.as_str(),
                    _ => return None,
                };

                // Match variant field patterns
                for (vf_name, vf_pattern) in field_patterns {
                    let vf_key = format!("{}.{}/{}", field_name, variant_name, vf_name);
                    match vf_pattern {
                        Pattern::Variable(var) => {
                            if let Some(bound_value) = existing_bindings
                                .get(var)
                                .or_else(|| new_bindings.get(var))
                            {
                                let vf_val = fields.get(&vf_key)?;
                                if bound_value != vf_val {
                                    return None;
                                }
                            } else {
                                let vf_val = fields.get(&vf_key)?;
                                new_bindings.insert(var.clone(), vf_val.clone());
                            }
                        }
                        Pattern::Constant(expected) => {
                            let vf_val = fields.get(&vf_key)?;
                            if vf_val != expected {
                                return None;
                            }
                        }
                        Pattern::Predicate { op, value } => {
                            let vf_val = fields.get(&vf_key)?;
                            if !op.evaluate(vf_val, value) {
                                return None;
                            }
                        }
                        Pattern::EnumMatch { .. } => {
                            // Nested enum matching not supported
                            return None;
                        }
                    }
                }
            }
        }
    }

    Some(new_bindings)
}
