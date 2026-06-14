//! Group membership and resolution.
//!
//! Reimplemented in Rust from Excalidraw's grouping logic. Upstream derivation:
//! - `packages/excalidraw/groups.ts` (Excalidraw, MIT) — `getSelectedGroupIds`,
//!   `getElementsInGroup`, `addToGroup` / `removeFromSelectedGroups`, and the
//!   outermost-group resolution used when clicking a grouped element.
//!
//! Each element carries an ordered `group_ids: Vec<GroupId>`, innermost first and
//! outermost last (matching Excalidraw). The *outermost* group is therefore the
//! last entry — that is the group a plain click selects.

use crate::element::{ElementId, GroupId};
use crate::scene::Scene;

impl Scene {
    /// All live elements that belong to `group` (in any nesting position), in
    /// current paint order.
    pub fn elements_in_group(&self, group: &GroupId) -> Vec<ElementId> {
        self.iter_live()
            .filter(|e| e.group_ids.iter().any(|g| g == group))
            .map(|e| e.id.clone())
            .collect()
    }

    /// The outermost group id of an element, if it belongs to any group. This is
    /// the *last* entry of `group_ids` (Excalidraw orders innermost-first).
    pub fn outermost_group_id(&self, id: &ElementId) -> Option<GroupId> {
        self.get(id).and_then(|e| e.group_ids.last().cloned())
    }

    /// Given a set of selected element ids, the set of group ids that are
    /// *fully* selected at their outermost level — i.e. the groups whose every
    /// member is in the selection. Mirrors `getSelectedGroupIds` semantics used
    /// to decide which group-drag handles to show.
    pub fn selected_group_ids(&self, selected: &[ElementId]) -> Vec<GroupId> {
        use std::collections::HashSet;
        let selected_set: HashSet<&ElementId> = selected.iter().collect();

        // Candidate outermost groups from the selection.
        let mut candidates: Vec<GroupId> = Vec::new();
        for id in selected {
            if let Some(g) = self.outermost_group_id(id) {
                if !candidates.contains(&g) {
                    candidates.push(g);
                }
            }
        }

        candidates
            .into_iter()
            .filter(|g| {
                // Every live member of the group must be selected.
                self.iter_live()
                    .filter(|e| e.group_ids.iter().any(|x| x == g))
                    .all(|e| selected_set.contains(&e.id))
            })
            .collect()
    }

    /// Expand a selection so that selecting any member of a group selects the
    /// whole outermost group. Returns ids in current paint order, de-duplicated.
    pub fn expand_selection_to_groups(&self, selected: &[ElementId]) -> Vec<ElementId> {
        use std::collections::HashSet;
        let mut groups: Vec<GroupId> = Vec::new();
        let mut singles: HashSet<ElementId> = HashSet::new();
        for id in selected {
            match self.outermost_group_id(id) {
                Some(g) => {
                    if !groups.contains(&g) {
                        groups.push(g);
                    }
                }
                None => {
                    singles.insert(id.clone());
                }
            }
        }
        let mut result_set: HashSet<ElementId> = singles;
        for g in &groups {
            for member in self.elements_in_group(g) {
                result_set.insert(member);
            }
        }
        // Emit in paint order for determinism.
        self.order()
            .iter()
            .filter(|id| result_set.contains(id))
            .cloned()
            .collect()
    }

    /// Add `group` as the new outermost group of every given element (appended,
    /// so it becomes the last / outermost entry). Already-present membership is
    /// not duplicated. Returns the ids actually modified.
    pub fn add_to_group(&mut self, ids: &[ElementId], group: &GroupId) -> Vec<ElementId> {
        let mut changed = Vec::new();
        for id in ids {
            if let Some(el) = self.get_mut(id) {
                if !el.group_ids.iter().any(|g| g == group) {
                    el.group_ids.push(group.clone());
                    changed.push(id.clone());
                }
            }
        }
        changed
    }

    /// Remove `group` from every given element's membership (at any nesting
    /// level). Returns the ids actually modified.
    pub fn remove_from_group(&mut self, ids: &[ElementId], group: &GroupId) -> Vec<ElementId> {
        let mut changed = Vec::new();
        for id in ids {
            if let Some(el) = self.get_mut(id) {
                let before = el.group_ids.len();
                el.group_ids.retain(|g| g != group);
                if el.group_ids.len() != before {
                    changed.push(id.clone());
                }
            }
        }
        changed
    }
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

    #[test]
    fn elements_in_group_in_paint_order() {
        let s = scene(vec![el("a", &["g1"]), el("b", &[]), el("c", &["g1"])]);
        let g1 = GroupId::from("g1");
        let ids: Vec<String> = s
            .elements_in_group(&g1)
            .iter()
            .map(|i| i.as_str().to_string())
            .collect();
        assert_eq!(ids, ["a", "c"]);
    }

    #[test]
    fn outermost_is_last_group_id() {
        // innermost-first ordering: g_inner then g_outer.
        let s = scene(vec![el("a", &["g_inner", "g_outer"])]);
        assert_eq!(
            s.outermost_group_id(&ElementId::from("a")),
            Some(GroupId::from("g_outer"))
        );
        // Ungrouped element.
        let s2 = scene(vec![el("b", &[])]);
        assert_eq!(s2.outermost_group_id(&ElementId::from("b")), None);
    }

    #[test]
    fn selected_group_ids_requires_full_membership() {
        let s = scene(vec![el("a", &["g1"]), el("b", &["g1"]), el("c", &[])]);
        let g1 = GroupId::from("g1");
        // Only a selected: group not fully selected.
        assert!(s.selected_group_ids(&[ElementId::from("a")]).is_empty());
        // Both a and b selected: group fully selected.
        assert_eq!(
            s.selected_group_ids(&[ElementId::from("a"), ElementId::from("b")]),
            vec![g1]
        );
    }

    #[test]
    fn expand_selection_pulls_in_whole_group() {
        let s = scene(vec![el("a", &["g1"]), el("b", &["g1"]), el("c", &[])]);
        // Selecting just b expands to the whole g1 group, in paint order.
        let ids: Vec<String> = s
            .expand_selection_to_groups(&[ElementId::from("b")])
            .iter()
            .map(|i| i.as_str().to_string())
            .collect();
        assert_eq!(ids, ["a", "b"]);
        // Selecting an ungrouped element leaves it alone.
        let ids2: Vec<String> = s
            .expand_selection_to_groups(&[ElementId::from("c")])
            .iter()
            .map(|i| i.as_str().to_string())
            .collect();
        assert_eq!(ids2, ["c"]);
    }

    #[test]
    fn add_appends_outermost_no_duplicate() {
        let mut s = scene(vec![el("a", &["g1"])]);
        let g2 = GroupId::from("g2");
        let changed = s.add_to_group(&[ElementId::from("a")], &g2);
        assert_eq!(changed, vec![ElementId::from("a")]);
        assert_eq!(
            s.get(&ElementId::from("a")).unwrap().group_ids,
            vec![GroupId::from("g1"), GroupId::from("g2")]
        );
        // Re-adding g2 is a no-op.
        let changed2 = s.add_to_group(&[ElementId::from("a")], &g2);
        assert!(changed2.is_empty());
        assert_eq!(
            s.outermost_group_id(&ElementId::from("a")),
            Some(GroupId::from("g2"))
        );
    }

    #[test]
    fn remove_strips_group() {
        let mut s = scene(vec![el("a", &["g1", "g2"])]);
        let changed = s.remove_from_group(&[ElementId::from("a")], &GroupId::from("g1"));
        assert_eq!(changed, vec![ElementId::from("a")]);
        assert_eq!(
            s.get(&ElementId::from("a")).unwrap().group_ids,
            vec![GroupId::from("g2")]
        );
        // Removing a group the element doesn't have is a no-op.
        let changed2 = s.remove_from_group(&[ElementId::from("a")], &GroupId::from("nope"));
        assert!(changed2.is_empty());
    }

    #[test]
    fn deleted_elements_excluded_from_group_queries() {
        let mut a = el("a", &["g1"]);
        a.is_deleted = true;
        let s = scene(vec![a, el("b", &["g1"])]);
        let ids: Vec<String> = s
            .elements_in_group(&GroupId::from("g1"))
            .iter()
            .map(|i| i.as_str().to_string())
            .collect();
        assert_eq!(ids, ["b"]);
    }
}
