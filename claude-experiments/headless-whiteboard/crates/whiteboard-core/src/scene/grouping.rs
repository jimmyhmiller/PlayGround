//! High-level group / ungroup operations and group-aware selection helpers.
//!
//! Reimplemented in Rust from Excalidraw's grouping semantics. Upstream
//! derivation:
//! - `packages/excalidraw/groups.ts` (Excalidraw, MIT) — the group/ungroup
//!   `actionGroup` / `actionUngroup` flows, `addToGroup`,
//!   `removeFromSelectedGroups`, and the outermost-group resolution used when a
//!   plain click on a grouped element selects the whole group.
//!
//! These build on the lower-level membership primitives in `scene/groups.rs`
//! (`add_to_group` / `remove_from_group` / `outermost_group_id` /
//! `selected_group_ids` / `expand_selection_to_groups`).
//!
//! Group ids are **created by the caller** (deterministically — e.g. derived
//! from a seed/counter) and passed in. This module never invents one.
//!
//! Group-id ordering matches Excalidraw: innermost-first, outermost-last. A new
//! group from `group` therefore appends and becomes the new outermost group.

use crate::element::{ElementId, GroupId};
use crate::scene::Scene;
use std::collections::HashSet;

/// Group the given elements under `new_group`, which becomes their new
/// *outermost* group (appended to the end of each element's `group_ids`).
///
/// Excalidraw requires at least two distinct elements for a group to be
/// meaningful; we enforce that here and return `false` (no-op) for fewer. We
/// also skip ids that are not live elements when counting distinctness, so a
/// selection of one real element plus stale ids cannot form a group.
///
/// Returns whether anything changed.
pub fn group(scene: &mut Scene, ids: &[ElementId], new_group: GroupId) -> bool {
    // Distinct, live element ids only.
    let mut seen: HashSet<&ElementId> = HashSet::new();
    let mut distinct: Vec<ElementId> = Vec::new();
    for id in ids {
        if scene.get(id).is_some_and(|e| !e.is_deleted) && seen.insert(id) {
            distinct.push(id.clone());
        }
    }

    if distinct.len() < 2 {
        return false;
    }

    let changed = scene.add_to_group(&distinct, &new_group);
    !changed.is_empty()
}

/// Ungroup the selection: remove the *outermost* shared group id (the one
/// `selected_group_ids` would resolve to) from every element of that group,
/// leaving any inner groups intact.
///
/// Returns the removed group id(s). If the selection is not fully grouped at any
/// outermost level, nothing is removed and the result is empty.
pub fn ungroup(scene: &mut Scene, ids: &[ElementId]) -> Vec<GroupId> {
    // The outermost group(s) the selection fully covers. Empty if the selection
    // is not a complete group (matches Excalidraw, which only ungroups groups
    // whose every member is selected).
    let groups = scene.selected_group_ids(ids);
    if groups.is_empty() {
        return Vec::new();
    }

    let mut removed: Vec<GroupId> = Vec::new();
    for g in groups {
        // Strip this outermost group from *all* its members (in any nesting
        // position), not just the literal selection — every member shares it as
        // their outermost group by construction.
        let members = scene.elements_in_group(&g);
        if !scene.remove_from_group(&members, &g).is_empty() {
            removed.push(g);
        }
    }
    removed
}

/// The full set of element ids the selection expands to once group membership is
/// honored: clicking any member of a group selects the whole outermost group.
///
/// Convenience name over [`Scene::expand_selection_to_groups`]; emitted in paint
/// order, de-duplicated.
pub fn group_members(scene: &Scene, ids: &[ElementId]) -> Vec<ElementId> {
    scene.expand_selection_to_groups(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{Element, ElementKind};

    fn el(id: &str, groups: &[&str]) -> Element {
        let mut e = Element::new(
            ElementId::from(id),
            1,
            0.0,
            0.0,
            10.0,
            10.0,
            ElementKind::Rectangle,
        );
        e.group_ids = groups.iter().map(|g| GroupId::from(*g)).collect();
        e
    }

    fn scene(elements: Vec<Element>) -> Scene {
        let mut s = Scene::new();
        for e in elements {
            s.insert(e);
        }
        s
    }

    fn gids(s: &Scene, id: &str) -> Vec<String> {
        s.get(&ElementId::from(id))
            .unwrap()
            .group_ids
            .iter()
            .map(|g| g.as_str().to_string())
            .collect()
    }

    #[test]
    fn group_two_elements_share_new_outermost() {
        let mut s = scene(vec![el("a", &[]), el("b", &[])]);
        let g = GroupId::from("g1");
        assert!(group(
            &mut s,
            &[ElementId::from("a"), ElementId::from("b")],
            g.clone()
        ));
        assert_eq!(gids(&s, "a"), ["g1"]);
        assert_eq!(gids(&s, "b"), ["g1"]);
        // Both resolve to the new group as outermost.
        assert_eq!(s.outermost_group_id(&ElementId::from("a")), Some(g.clone()));
        assert_eq!(s.outermost_group_id(&ElementId::from("b")), Some(g));
    }

    #[test]
    fn group_appends_outermost_over_existing_inner() {
        let mut s = scene(vec![el("a", &["inner"]), el("b", &["inner"])]);
        let g = GroupId::from("outer");
        assert!(group(
            &mut s,
            &[ElementId::from("a"), ElementId::from("b")],
            g.clone()
        ));
        // Inner preserved, new group is now outermost (last).
        assert_eq!(gids(&s, "a"), ["inner", "outer"]);
        assert_eq!(s.outermost_group_id(&ElementId::from("a")), Some(g));
    }

    #[test]
    fn grouping_fewer_than_two_is_noop() {
        let mut s = scene(vec![el("a", &[])]);
        // Single element.
        assert!(!group(&mut s, &[ElementId::from("a")], GroupId::from("g")));
        assert!(gids(&s, "a").is_empty());
        // Empty selection.
        assert!(!group(&mut s, &[], GroupId::from("g")));
        // Two ids but the same element repeated -> only one distinct element.
        assert!(!group(
            &mut s,
            &[ElementId::from("a"), ElementId::from("a")],
            GroupId::from("g")
        ));
        assert!(gids(&s, "a").is_empty());
    }

    #[test]
    fn grouping_ignores_nonexistent_and_deleted_ids() {
        let mut d = el("d", &[]);
        d.is_deleted = true;
        let mut s = scene(vec![el("a", &[]), d]);
        // a (live) + d (deleted) + ghost -> only one live distinct element.
        assert!(!group(
            &mut s,
            &[
                ElementId::from("a"),
                ElementId::from("d"),
                ElementId::from("ghost")
            ],
            GroupId::from("g")
        ));
        assert!(gids(&s, "a").is_empty());
    }

    #[test]
    fn ungroup_removes_outermost_leaves_inner() {
        let mut s = scene(vec![
            el("a", &["inner", "outer"]),
            el("b", &["inner", "outer"]),
        ]);
        let removed = ungroup(&mut s, &[ElementId::from("a"), ElementId::from("b")]);
        assert_eq!(removed, vec![GroupId::from("outer")]);
        // Inner group intact, outer stripped.
        assert_eq!(gids(&s, "a"), ["inner"]);
        assert_eq!(gids(&s, "b"), ["inner"]);
    }

    #[test]
    fn ungroup_from_single_clicked_member_strips_whole_group() {
        // After a click is expanded via group_members, ungroup the whole group
        // even though only the expanded set is passed.
        let mut s = scene(vec![el("a", &["g"]), el("b", &["g"])]);
        let sel = group_members(&s, &[ElementId::from("a")]);
        let removed = ungroup(&mut s, &sel);
        assert_eq!(removed, vec![GroupId::from("g")]);
        assert!(gids(&s, "a").is_empty());
        assert!(gids(&s, "b").is_empty());
    }

    #[test]
    fn ungroup_partial_selection_is_noop() {
        // Only one of two members selected: group not fully selected, no change.
        let mut s = scene(vec![el("a", &["g"]), el("b", &["g"])]);
        let removed = ungroup(&mut s, &[ElementId::from("a")]);
        assert!(removed.is_empty());
        assert_eq!(gids(&s, "a"), ["g"]);
        assert_eq!(gids(&s, "b"), ["g"]);
    }

    #[test]
    fn ungroup_nothing_grouped_returns_empty() {
        let mut s = scene(vec![el("a", &[]), el("b", &[])]);
        let removed = ungroup(&mut s, &[ElementId::from("a"), ElementId::from("b")]);
        assert!(removed.is_empty());
    }

    #[test]
    fn group_members_expands_single_click_to_whole_group() {
        let s = scene(vec![el("a", &["g1"]), el("b", &["g1"]), el("c", &[])]);
        // Clicking b selects all of g1, in paint order.
        let ids: Vec<String> = group_members(&s, &[ElementId::from("b")])
            .iter()
            .map(|i| i.as_str().to_string())
            .collect();
        assert_eq!(ids, ["a", "b"]);
        // Clicking an ungrouped element only selects itself.
        let ids2: Vec<String> = group_members(&s, &[ElementId::from("c")])
            .iter()
            .map(|i| i.as_str().to_string())
            .collect();
        assert_eq!(ids2, ["c"]);
    }

    #[test]
    fn group_then_ungroup_round_trips() {
        let mut s = scene(vec![el("a", &[]), el("b", &[])]);
        let g = GroupId::from("rt");
        assert!(group(
            &mut s,
            &[ElementId::from("a"), ElementId::from("b")],
            g.clone()
        ));
        let removed = ungroup(&mut s, &[ElementId::from("a"), ElementId::from("b")]);
        assert_eq!(removed, vec![g]);
        assert!(gids(&s, "a").is_empty());
        assert!(gids(&s, "b").is_empty());
    }
}
