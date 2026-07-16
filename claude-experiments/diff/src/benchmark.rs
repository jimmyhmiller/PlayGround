use std::collections::BTreeSet;

use crate::DeltaRevision;

pub fn synthetic_revisions(
    module_count: usize,
    fanout: usize,
    imports_per_module: usize,
) -> Vec<DeltaRevision> {
    assert!(module_count > 0, "module count must be positive");
    assert!(fanout > 0, "fanout must be positive");
    assert!(
        imports_per_module > 0,
        "imports per module must be positive"
    );

    let module_ids = (0..module_count)
        .map(|index| format!("module-{index:08}.js"))
        .collect::<Vec<_>>();
    let mut edges = (1..module_count)
        .map(|child| {
            let parent = (child - 1) / fanout;
            (module_ids[parent].clone(), module_ids[child].clone())
        })
        .collect::<BTreeSet<_>>();
    for source in 1..module_count {
        for import_index in 1..imports_per_module {
            let target = source
                .wrapping_mul(1_000_003)
                .wrapping_add(import_index.wrapping_mul(97))
                % source;
            edges.insert((module_ids[source].clone(), module_ids[target].clone()));
        }
    }

    let edited_index = module_count / 2;
    let edited_module = module_ids[edited_index].clone();
    let mut changed = BTreeSet::new();
    changed.insert(edited_module.clone());

    let leaf = module_count - 1;
    let removed_edges = if module_count > 1 {
        edges
            .iter()
            .filter(|(source, target)| source == &module_ids[leaf] || target == &module_ids[leaf])
            .cloned()
            .map(|edge| (edge, -1))
            .collect()
    } else {
        Vec::new()
    };

    vec![
        DeltaRevision {
            label: "initial-load".into(),
            entry_updates: vec![(module_ids[0].clone(), 1)],
            edge_updates: edges.into_iter().map(|edge| (edge, 1)).collect(),
            module_updates: module_ids
                .iter()
                .enumerate()
                .map(|(index, module)| ((module.clone(), index as u64 + 1), 1))
                .collect(),
            ..DeltaRevision::default()
        },
        DeltaRevision {
            label: "one-content-edit".into(),
            module_updates: vec![
                ((edited_module.clone(), edited_index as u64 + 1), -1),
                ((edited_module, u64::MAX), 1),
            ],
            changed,
            ..DeltaRevision::default()
        },
        DeltaRevision {
            label: "remove-one-leaf".into(),
            edge_updates: removed_edges,
            module_updates: (module_count > 1)
                .then(|| ((module_ids[leaf].clone(), leaf as u64 + 1), -1))
                .into_iter()
                .collect(),
            ..DeltaRevision::default()
        },
    ]
}

#[cfg(test)]
mod tests {
    use crate::run_delta_revisions;

    use super::*;

    #[test]
    fn creates_a_fully_reachable_tree_and_small_edits() {
        let results = run_delta_revisions(synthetic_revisions(31, 2, 4));
        assert_eq!(results[0].reachable_facts, 31);
        assert_eq!(results[1].added_facts, 0);
        assert_eq!(results[1].removed_facts, 0);
        assert_eq!(results[1].changed_facts, 1);
        assert_eq!(results[2].removed_facts, 1);
        assert_eq!(results[2].reachable_facts, 30);
    }
}
