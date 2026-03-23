use crate::types::*;

/// Result of project or retract. Contains the transformed edit and the adjusted difference.
#[derive(Debug, Clone, PartialEq)]
pub struct TransformResult {
    /// The transformed edit (post for project, pre for retract)
    pub edit: Edit,
    /// The adjusted difference
    pub adjust: Edit,
}

/// A "point edit" operates on a single index without changing the document's size.
/// Conv, Rename, and Set are all point edits. They follow identical OT rules:
/// shift with Ins, follow/override with Move, conflict at same index with same kind.
fn point_edit_idx(edit: &Edit) -> Option<usize> {
    match edit {
        Edit::Conv { idx, .. } | Edit::Rename { idx, .. } | Edit::Set { idx, .. } => Some(*idx),
        _ => None,
    }
}

/// Reconstruct a point edit with a new index.
fn with_idx(edit: &Edit, new_idx: usize) -> Edit {
    match edit {
        Edit::Conv { ty, .. } => Edit::Conv { idx: new_idx, ty: *ty },
        Edit::Rename { name, .. } => Edit::Rename { idx: new_idx, name: name.clone() },
        Edit::Set { value, .. } => Edit::Set { idx: new_idx, value: value.clone() },
        other => other.clone(),
    }
}

/// Check if two edits are the same structural kind (for conflict detection).
fn same_kind(a: &Edit, b: &Edit) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

/// Project: given `pre` (an edit applied to the left side) and `diff` (the difference
/// across the top), compute `post` (what `pre` becomes on the right) and `adjust`
/// (the new difference across the bottom).
///
/// Satisfies: post ∘ diff = adjust ∘ pre
pub fn project(pre: &Edit, diff: &Edit) -> TransformResult {
    // Equal edits cancel out
    if pre == diff && !pre.is_id() {
        return TransformResult {
            edit: Edit::Id,
            adjust: Edit::Id,
        };
    }

    // Id is a fixpoint
    if pre.is_id() {
        return TransformResult {
            edit: Edit::Id,
            adjust: diff.clone(),
        };
    }
    if diff.is_id() {
        return TransformResult {
            edit: pre.clone(),
            adjust: Edit::Id,
        };
    }

    // --- Point edit vs Point edit (Conv, Rename, Set) ---
    if let (Some(pi), Some(di)) = (point_edit_idx(pre), point_edit_idx(diff)) {
        if pi == di && same_kind(pre, diff) {
            // Conflict: same kind at same index, pre wins
            return TransformResult {
                edit: pre.clone(),
                adjust: Edit::Id,
            };
        }
        // Independent (same or different index, different kinds)
        return TransformResult {
            edit: pre.clone(),
            adjust: diff.clone(),
        };
    }

    // --- Point edit vs Ins ---
    if let (Some(pi), Edit::Ins { idx: j, ty: u, id }) = (point_edit_idx(pre), diff) {
        return if pi >= *j {
            TransformResult {
                edit: with_idx(pre, pi + 1),
                adjust: Edit::Ins { idx: *j, ty: *u, id: *id },
            }
        } else {
            TransformResult {
                edit: pre.clone(),
                adjust: diff.clone(),
            }
        };
    }

    // --- Ins vs Point edit ---
    if let (Edit::Ins { idx: i, ty: t, id }, Some(dj)) = (pre, point_edit_idx(diff)) {
        return if *i <= dj {
            TransformResult {
                edit: Edit::Ins { idx: *i, ty: *t, id: *id },
                adjust: with_idx(diff, dj + 1),
            }
        } else {
            TransformResult {
                edit: pre.clone(),
                adjust: diff.clone(),
            }
        };
    }

    // --- Point edit vs Move ---
    if let (Some(pi), Edit::Move { i: mi, j: mj }) = (point_edit_idx(pre), diff) {
        return if pi == *mi {
            // Point edit at Move target: overridden
            TransformResult {
                edit: Edit::Id,
                adjust: Edit::Move { i: *mi, j: *mj },
            }
        } else if pi == *mj {
            // Point edit at Move source: follows to target
            TransformResult {
                edit: with_idx(pre, *mi),
                adjust: Edit::Move { i: *mi, j: *mj },
            }
        } else {
            TransformResult {
                edit: pre.clone(),
                adjust: diff.clone(),
            }
        };
    }

    // --- Move vs Point edit ---
    if let (Edit::Move { i: mi, j: mj }, Some(dj)) = (pre, point_edit_idx(diff)) {
        return if dj == *mi {
            // Point edit at Move's target is overridden by Move
            TransformResult {
                edit: Edit::Move { i: *mi, j: *mj },
                adjust: Edit::Id,
            }
        } else if dj == *mj {
            // Point edit at Move's source: follows value to target
            TransformResult {
                edit: Edit::Move { i: *mi, j: *mj },
                adjust: with_idx(diff, *mi),
            }
        } else {
            TransformResult {
                edit: pre.clone(),
                adjust: diff.clone(),
            }
        };
    }

    match (pre, diff) {
        // --- Ins vs Ins ---
        (Edit::Ins { idx: i, ty: t, id: p }, Edit::Ins { idx: j, ty: u, id: q }) => {
            if p == q {
                return TransformResult {
                    edit: Edit::Id,
                    adjust: Edit::Id,
                };
            }
            if *i <= *j {
                TransformResult {
                    edit: Edit::Ins { idx: *i, ty: *t, id: *p },
                    adjust: Edit::Ins { idx: j + 1, ty: *u, id: *q },
                }
            } else {
                TransformResult {
                    edit: Edit::Ins { idx: i + 1, ty: *t, id: *p },
                    adjust: Edit::Ins { idx: *j, ty: *u, id: *q },
                }
            }
        }

        // --- Ins vs Move ---
        (Edit::Ins { idx: i, ty: t, id }, Edit::Move { i: mi, j: mj }) => {
            let new_mi = if *i <= *mi { mi + 1 } else { *mi };
            let new_mj = if *i <= *mj { mj + 1 } else { *mj };
            TransformResult {
                edit: Edit::Ins { idx: *i, ty: *t, id: *id },
                adjust: Edit::Move { i: new_mi, j: new_mj },
            }
        }

        // --- Move vs Ins ---
        (Edit::Move { i: mi, j: mj }, Edit::Ins { idx, ty, id }) => {
            let new_mi = if *idx <= *mi { mi + 1 } else { *mi };
            let new_mj = if *idx <= *mj { mj + 1 } else { *mj };
            TransformResult {
                edit: Edit::Move { i: new_mi, j: new_mj },
                adjust: Edit::Ins { idx: *idx, ty: *ty, id: *id },
            }
        }

        // --- Move vs Move ---
        // From the paper's Appendix A, extended with gap fills (see DEVIATIONS.md).
        (Edit::Move { i: pi, j: pj }, Edit::Move { i: di, j: dj }) => {
            if pi == di && pj == dj {
                unreachable!("equal case handled above")
            } else if pi == di {
                TransformResult {
                    edit: Edit::Move { i: *pi, j: *pj },
                    adjust: Edit::Conv { idx: *dj, ty: AtomicType::Del },
                }
            } else if pi == dj && pj == di {
                TransformResult {
                    edit: Edit::Move { i: *di, j: *dj },
                    adjust: Edit::Move { i: *dj, j: *di },
                }
            } else if pj == dj {
                TransformResult {
                    edit: Edit::Move { i: *pi, j: *di },
                    adjust: Edit::Move { i: *di, j: *dj },
                }
            } else if pi == dj {
                TransformResult {
                    edit: Edit::Move { i: *di, j: *pj },
                    adjust: Edit::Move { i: *di, j: *dj },
                }
            } else if pj == di {
                TransformResult {
                    edit: Edit::Move { i: *pi, j: *di },
                    adjust: Edit::Move { i: *pi, j: *dj },
                }
            } else {
                TransformResult {
                    edit: Edit::Move { i: *pi, j: *pj },
                    adjust: Edit::Move { i: *di, j: *dj },
                }
            }
        }

        // Fallback: edits are independent (shouldn't reach here for valid edit pairs)
        _ => TransformResult {
            edit: pre.clone(),
            adjust: diff.clone(),
        },
    }
}

/// Retract: given `post` (an edit applied to the right side) and `diff` (the difference
/// across the top), compute `pre` (what `post` was before `diff`) and `adjust`
/// (the new difference across the bottom).
///
/// Returns None if retraction is not possible (dependency).
///
/// **DEVIATION 1** (see DEVIATIONS.md): The paper's "equal edits cancel out"
/// retract rule `retract(x, x) = (Id, x)` requires x² = x (idempotence).
/// This holds for Conv but FAILS for Move and Ins. We handle equal edits
/// case-by-case instead of using a generic rule.
pub fn retract(post: &Edit, diff: &Edit) -> Option<TransformResult> {
    // Id fixpoints
    if post.is_id() {
        return Some(TransformResult {
            edit: Edit::Id,
            adjust: diff.clone(),
        });
    }
    if diff.is_id() {
        return Some(TransformResult {
            edit: post.clone(),
            adjust: Edit::Id,
        });
    }

    // --- Point edit vs Point edit ---
    if let (Some(pi), Some(di)) = (point_edit_idx(post), point_edit_idx(diff)) {
        if pi == di && same_kind(post, diff) {
            // Same kind at same index: post wins (idempotent for Conv/Rename/Set)
            return Some(TransformResult {
                edit: post.clone(),
                adjust: Edit::Id,
            });
        }
        return Some(TransformResult {
            edit: post.clone(),
            adjust: diff.clone(),
        });
    }

    // --- Point edit vs Ins ---
    if let (Some(pi), Edit::Ins { idx: j, .. }) = (point_edit_idx(post), diff) {
        if pi == *j {
            return None; // dependency
        }
        return if pi > *j {
            Some(TransformResult {
                edit: with_idx(post, pi - 1),
                adjust: diff.clone(),
            })
        } else {
            Some(TransformResult {
                edit: post.clone(),
                adjust: diff.clone(),
            })
        };
    }

    // --- Ins vs Point edit ---
    if let (Edit::Ins { .. }, Some(_)) = (post, point_edit_idx(diff)) {
        // Ins through point edit: same as project (Ins not shifted by point edits)
        let proj = project(post, diff);
        return Some(proj);
    }

    // --- Point edit vs Move ---
    if let (Some(pi), Edit::Move { i: mi, j: mj }) = (point_edit_idx(post), diff) {
        if pi == *mj {
            return None; // dependency: point edit at source, can't retract
        }
        if pi == *mi {
            // Point edit at target: retract to source
            return Some(TransformResult {
                edit: with_idx(post, *mj),
                adjust: Edit::Move { i: *mi, j: *mj },
            });
        }
        return Some(TransformResult {
            edit: post.clone(),
            adjust: diff.clone(),
        });
    }

    // --- Move vs Point edit ---
    if let (Edit::Move { .. }, Some(_)) = (post, point_edit_idx(diff)) {
        // Move through point edit: same as project (Move not shifted by point edits)
        let proj = project(post, diff);
        return Some(proj);
    }

    match (post, diff) {
        // === Ins vs Ins ===
        (Edit::Ins { idx: i, ty: t, id: p }, Edit::Ins { idx: j, ty: u, id: q }) => {
            if p == q {
                return Some(TransformResult {
                    edit: Edit::Id,
                    adjust: Edit::Id,
                });
            }
            if *i <= *j {
                Some(TransformResult {
                    edit: Edit::Ins { idx: *i, ty: *t, id: *p },
                    adjust: Edit::Ins { idx: j + 1, ty: *u, id: *q },
                })
            } else {
                Some(TransformResult {
                    edit: Edit::Ins { idx: i - 1, ty: *t, id: *p },
                    adjust: Edit::Ins { idx: *j, ty: *u, id: *q },
                })
            }
        }

        // === Ins vs Move ===
        (Edit::Ins { idx: i, ty: t, id }, Edit::Move { i: mi, j: mj }) => {
            let new_mi = if *i <= *mi { mi + 1 } else { *mi };
            let new_mj = if *i <= *mj { mj + 1 } else { *mj };
            Some(TransformResult {
                edit: Edit::Ins { idx: *i, ty: *t, id: *id },
                adjust: Edit::Move { i: new_mi, j: new_mj },
            })
        }

        // === Move vs Ins ===
        (Edit::Move { i: mi, j: mj }, Edit::Ins { idx, ty, id }) => {
            if idx == mi || idx == mj {
                return None;
            }
            let new_mi = if *idx < *mi { mi - 1 } else { *mi };
            let new_mj = if *idx < *mj { mj - 1 } else { *mj };
            Some(TransformResult {
                edit: Edit::Move { i: new_mi, j: new_mj },
                adjust: Edit::Ins { idx: *idx, ty: *ty, id: *id },
            })
        }

        // === Move vs Move ===
        (Edit::Move { i: pi, j: pj }, Edit::Move { i: di, j: dj }) => {
            if pi == di && pj == dj {
                // Equal Moves: see DEVIATIONS.md
                Some(TransformResult {
                    edit: Edit::Move { i: *pi, j: *pj },
                    adjust: Edit::Conv { idx: *pi, ty: AtomicType::Del },
                })
            } else if pi == di {
                Some(TransformResult {
                    edit: Edit::Move { i: *pi, j: *pj },
                    adjust: Edit::Conv { idx: *dj, ty: AtomicType::Del },
                })
            } else if pj == di {
                Some(TransformResult {
                    edit: Edit::Move { i: *pi, j: *di },
                    adjust: Edit::Move { i: *pi, j: *dj },
                })
            } else if pj == dj {
                Some(TransformResult {
                    edit: Edit::Move { i: *pi, j: *dj },
                    adjust: Edit::Move { i: *di, j: *pi },
                })
            } else if pi == dj {
                Some(TransformResult {
                    edit: Edit::Move { i: *di, j: *dj },
                    adjust: Edit::Move { i: *pi, j: *pj },
                })
            } else {
                Some(TransformResult {
                    edit: Edit::Move { i: *pi, j: *pj },
                    adjust: Edit::Move { i: *di, j: *dj },
                })
            }
        }

        _ => Some(TransformResult {
            edit: post.clone(),
            adjust: diff.clone(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_edits_cancel_project() {
        let edit = Edit::Conv { idx: 1, ty: AtomicType::Num };
        let result = project(&edit, &edit);
        assert_eq!(result.edit, Edit::Id);
        assert_eq!(result.adjust, Edit::Id);
    }

    #[test]
    fn test_id_fixpoint_project() {
        let diff = Edit::Conv { idx: 1, ty: AtomicType::Str };
        let result = project(&Edit::Id, &diff);
        assert_eq!(result.edit, Edit::Id);
        assert_eq!(result.adjust, diff);
    }

    #[test]
    fn test_project_pre_id_diff() {
        let pre = Edit::Conv { idx: 0, ty: AtomicType::Num };
        let result = project(&pre, &Edit::Id);
        assert_eq!(result.edit, pre);
        assert_eq!(result.adjust, Edit::Id);
    }

    #[test]
    fn test_conv_conflict_project() {
        let pre = Edit::Conv { idx: 1, ty: AtomicType::Num };
        let diff = Edit::Conv { idx: 1, ty: AtomicType::Str };
        let result = project(&pre, &diff);
        assert_eq!(result.edit, Edit::Conv { idx: 1, ty: AtomicType::Num });
        assert_eq!(result.adjust, Edit::Id);
    }

    #[test]
    fn test_ins_shifts_conv() {
        let pre = Edit::Ins { idx: 1, ty: AtomicType::Bool, id: 1 };
        let diff = Edit::Conv { idx: 1, ty: AtomicType::Str };
        let result = project(&pre, &diff);
        assert_eq!(result.edit, Edit::Ins { idx: 1, ty: AtomicType::Bool, id: 1 });
        assert_eq!(result.adjust, Edit::Conv { idx: 2, ty: AtomicType::Str });
    }

    #[test]
    fn test_retract_impossible_conv_through_ins() {
        let post = Edit::Conv { idx: 1, ty: AtomicType::Num };
        let diff = Edit::Ins { idx: 1, ty: AtomicType::Str, id: 1 };
        assert!(retract(&post, &diff).is_none());
    }

    #[test]
    fn test_retract_conv_through_move_at_target() {
        let post = Edit::Conv { idx: 0, ty: AtomicType::Str };
        let diff = Edit::Move { i: 0, j: 2 };
        let result = retract(&post, &diff).unwrap();
        assert_eq!(result.edit, Edit::Conv { idx: 2, ty: AtomicType::Str });
        assert_eq!(result.adjust, Edit::Move { i: 0, j: 2 });
    }

    #[test]
    fn test_retract_conv_through_move_at_source_fails() {
        let post = Edit::Conv { idx: 2, ty: AtomicType::Str };
        let diff = Edit::Move { i: 0, j: 2 };
        assert!(retract(&post, &diff).is_none());
    }

    #[test]
    fn test_project_retract_roundtrip_conv_ins() {
        let pre = Edit::Conv { idx: 2, ty: AtomicType::Str };
        let diff = Edit::Ins { idx: 1, ty: AtomicType::Bool, id: 1 };
        let proj = project(&pre, &diff);
        assert_eq!(proj.edit, Edit::Conv { idx: 3, ty: AtomicType::Str });
        let retr = retract(&proj.edit, &diff).unwrap();
        assert_eq!(retr.edit, pre);
        assert_eq!(retr.adjust, proj.adjust);
    }

    #[test]
    fn test_rename_shifts_with_ins() {
        let pre = Edit::Rename { idx: 2, name: "email".into() };
        let diff = Edit::Ins { idx: 1, ty: AtomicType::Bool, id: 1 };
        let result = project(&pre, &diff);
        assert_eq!(result.edit, Edit::Rename { idx: 3, name: "email".into() });
    }

    #[test]
    fn test_rename_conflict() {
        let pre = Edit::Rename { idx: 1, name: "full_name".into() };
        let diff = Edit::Rename { idx: 1, name: "display_name".into() };
        let result = project(&pre, &diff);
        assert_eq!(result.edit, Edit::Rename { idx: 1, name: "full_name".into() });
        assert_eq!(result.adjust, Edit::Id);
    }

    #[test]
    fn test_rename_independent_of_conv() {
        let pre = Edit::Rename { idx: 1, name: "age".into() };
        let diff = Edit::Conv { idx: 1, ty: AtomicType::Str };
        let result = project(&pre, &diff);
        // Different kinds at same index: independent
        assert_eq!(result.edit, pre);
        assert_eq!(result.adjust, diff);
    }

    #[test]
    fn test_set_shifts_with_ins() {
        let pre = Edit::Set { idx: 2, value: Value::Str("hello".into()) };
        let diff = Edit::Ins { idx: 1, ty: AtomicType::Bool, id: 1 };
        let result = project(&pre, &diff);
        assert_eq!(result.edit, Edit::Set { idx: 3, value: Value::Str("hello".into()) });
    }

    #[test]
    fn test_set_follows_move() {
        let pre = Edit::Set { idx: 1, value: Value::Num(42.0) };
        let diff = Edit::Move { i: 0, j: 1 };
        // Set at Move source → follows to target
        let result = project(&pre, &diff);
        assert_eq!(result.edit, Edit::Set { idx: 0, value: Value::Num(42.0) });
    }
}
