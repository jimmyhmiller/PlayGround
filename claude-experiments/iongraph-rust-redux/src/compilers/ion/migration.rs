// Ion version migration logic
use super::schema::IonJSON;

/// Migrate Ion JSON from version 0 to version 1
pub fn migrate(mut ion_json: serde_json::Value) -> IonJSON {
    // Check if we need to migrate from version 0
    if ion_json.get("version").is_none() {
        // This is version 0, migrate to version 1
        ion_json["version"] = serde_json::Value::Number(1.into());

        // Migrate passes
        if let Some(passes) = ion_json.get_mut("passes") {
            if let Some(passes_array) = passes.as_array_mut() {
                for pass in passes_array {
                    // Migrate MIR blocks
                    if let Some(mir) = pass.get_mut("mir") {
                        if let Some(mir_array) = mir.as_array_mut() {
                            for block in mir_array {
                                migrate_block_v0_to_v1(block);
                            }
                        }
                    }

                    // Migrate LIR blocks
                    if let Some(lir) = pass.get_mut("lir") {
                        if let Some(lir_array) = lir.as_array_mut() {
                            for block in lir_array {
                                migrate_block_v0_to_v1(block);
                            }
                        }
                    }
                }
            }
        }
    }

    serde_json::from_value(ion_json).expect("Failed to deserialize migrated IonJSON")
}

fn migrate_block_v0_to_v1(block: &mut serde_json::Value) {
    // In version 0, block IDs were numbers
    // In version 1, we use block numbers directly as IDs
    // The main change is handling predecessors/successors as BlockPtrs

    if let Some(obj) = block.as_object_mut() {
        // Migrate predecessors
        if let Some(preds) = obj.get("predecessors") {
            if let Some(preds_array) = preds.as_array() {
                let migrated: Vec<_> = preds_array
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u32))
                    .collect();
                obj.insert("predecessors".to_string(), serde_json::json!(migrated));
            }
        }

        // Migrate successors
        if let Some(succs) = obj.get("successors") {
            if let Some(succs_array) = succs.as_array() {
                let migrated: Vec<_> = succs_array
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u32))
                    .collect();
                obj.insert("successors".to_string(), serde_json::json!(migrated));
            }
        }
    }
}
