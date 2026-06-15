//! Clipboard + duplicate as pure operations over elements / a [`Scene`].
//!
//! Reimplemented from Excalidraw's copy/paste/duplicate semantics
//! (`packages/excalidraw/clipboard.ts`, `actions/actionDuplicate.tsx`,
//! `duplicate.ts`): pasting/duplicating mints fresh element ids and a fresh set
//! of group ids while **remapping** every intra-selection reference so the new
//! elements form a self-consistent island, and **dropping** any reference whose
//! referent was not part of the copied set (so we never dangle a pointer to a
//! non-pasted element).
//!
//! Everything here is pure: ids and group ids come from caller-injected
//! generators, so the operations are deterministic and free of hidden global
//! state (no OS RNG, no clocks). The editor layer wires these to its own
//! id/group-id counters and undo recording — see the module-level seam docs.

use super::Scene;
use crate::element::{Element, ElementId, GroupId};
use crate::geometry::Vec2;
use std::collections::HashMap;

/// A snapshot of copied elements. Holds owned clones with their *original* ids
/// and references intact; remapping happens at paste time so one copy can be
/// pasted many times, each yielding a fresh, independent island.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Clipboard {
    elements: Vec<Element>,
}

impl Clipboard {
    /// An empty clipboard.
    pub fn new() -> Self {
        Clipboard::default()
    }

    /// Build a clipboard directly from owned elements (e.g. cross-document
    /// paste). Prefer [`copy`] when the source is a live [`Scene`].
    pub fn from_elements(elements: Vec<Element>) -> Self {
        Clipboard { elements }
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// The copied elements, in copy order.
    pub fn elements(&self) -> &[Element] {
        &self.elements
    }
}

/// Copy the given live elements into a [`Clipboard`].
///
/// Clones exactly the ids requested (preserving every field) — *no* implicit
/// group expansion. Ids that are missing or refer to deleted elements are
/// skipped. Duplicate ids in `ids` are copied once, in first-seen order, so the
/// clipboard never holds two elements with the same id.
pub fn copy(scene: &Scene, ids: &[ElementId]) -> Clipboard {
    let mut seen: HashMap<&ElementId, ()> = HashMap::new();
    let mut elements = Vec::new();
    for id in ids {
        if seen.insert(id, ()).is_some() {
            continue;
        }
        if let Some(e) = scene.get(id) {
            if !e.is_deleted {
                elements.push(e.clone());
            }
        }
    }
    Clipboard { elements }
}

/// Paste a clipboard's contents as fresh elements.
///
/// Each pasted element gets:
/// - a new [`ElementId`] from `id_gen` (called once per element, in order);
/// - all of its [`GroupId`]s remapped, where every *distinct* old group id maps
///   to a single fresh group id from `group_id_gen` (so two elements that shared
///   a group still share the new group, and the paste stays a coherent group
///   hierarchy);
/// - its position shifted by `offset`;
/// - every intra-selection reference remapped to the corresponding new id when
///   the referent is also in the pasted set, and **cleared/dropped** otherwise.
///
/// Remapped references: `frame_id`, `bound_elements[*].id`, `TextData.container_id`,
/// and the `element_id` inside arrow/line `start_binding` / `end_binding`.
///
/// `group_id_gen` is invoked lazily and exactly once per distinct old group id,
/// in the order group ids are first encountered while walking the elements in
/// paste order; this keeps the new group ids deterministic given the generators.
pub fn paste(
    clipboard: &Clipboard,
    id_gen: &mut dyn FnMut() -> ElementId,
    group_id_gen: &mut dyn FnMut() -> GroupId,
    offset: Vec2,
) -> Vec<Element> {
    paste_elements(clipboard.elements(), id_gen, group_id_gen, offset)
}

/// Duplicate live scene elements in one step: [`copy`] then [`paste`].
pub fn duplicate(
    scene: &Scene,
    ids: &[ElementId],
    id_gen: &mut dyn FnMut() -> ElementId,
    group_id_gen: &mut dyn FnMut() -> GroupId,
    offset: Vec2,
) -> Vec<Element> {
    let clip = copy(scene, ids);
    paste(&clip, id_gen, group_id_gen, offset)
}

/// Core paste logic, shared by [`paste`] and [`duplicate`], operating on a raw
/// element slice.
fn paste_elements(
    sources: &[Element],
    id_gen: &mut dyn FnMut() -> ElementId,
    group_id_gen: &mut dyn FnMut() -> GroupId,
    offset: Vec2,
) -> Vec<Element> {
    // 1. Mint a fresh id for each source, building the old->new id map. This map
    //    is also our membership test: a reference is "in the pasted set" iff its
    //    target id is a key here.
    let mut id_map: HashMap<ElementId, ElementId> = HashMap::with_capacity(sources.len());
    let mut new_ids: Vec<ElementId> = Vec::with_capacity(sources.len());
    for src in sources {
        let new_id = id_gen();
        new_ids.push(new_id.clone());
        // If the source list somehow contains a duplicate id, the last mapping
        // wins; new_ids still tracks one fresh id per element positionally.
        id_map.insert(src.id.clone(), new_id);
    }

    // 2. Remap group ids: one fresh group id per distinct old group id, assigned
    //    in first-seen order across the elements in paste order.
    let mut group_map: HashMap<GroupId, GroupId> = HashMap::new();
    for src in sources {
        for gid in &src.group_ids {
            if !group_map.contains_key(gid) {
                group_map.insert(gid.clone(), group_id_gen());
            }
        }
    }

    // 3. Build the new elements.
    let mut out = Vec::with_capacity(sources.len());
    for (src, new_id) in sources.iter().zip(new_ids.into_iter()) {
        let mut e = src.clone();
        e.id = new_id;
        e.x += offset.x;
        e.y += offset.y;

        // Group ids: every membership remaps (all groups are minted above).
        e.group_ids = src
            .group_ids
            .iter()
            .map(|gid| group_map[gid].clone())
            .collect();

        // frame_id: remap if the frame was pasted too, else clear.
        e.frame_id = src
            .frame_id
            .as_ref()
            .and_then(|fid| id_map.get(fid).cloned());

        // bound_elements: keep only those whose target was pasted, remapping ids.
        e.bound_elements = src
            .bound_elements
            .iter()
            .filter_map(|be| {
                id_map.get(&be.id).map(|nid| {
                    let mut nb = be.clone();
                    nb.id = nid.clone();
                    nb
                })
            })
            .collect();

        // Kind-specific references.
        match &mut e.kind {
            crate::element::ElementKind::Text(t) => {
                // container_id: remap if container pasted, else clear (orphaned label).
                t.container_id = t
                    .container_id
                    .as_ref()
                    .and_then(|cid| id_map.get(cid).cloned());
            }
            crate::element::ElementKind::Arrow(l) | crate::element::ElementKind::Line(l) => {
                remap_binding(&mut l.start_binding, &id_map);
                remap_binding(&mut l.end_binding, &id_map);
            }
            _ => {}
        }

        out.push(e);
    }

    out
}

/// Remap (or clear) an arrow/line endpoint binding: keep the binding only if its
/// bound element was also pasted, rewriting it to the new id; otherwise drop it.
fn remap_binding(
    binding: &mut Option<crate::element::PointBinding>,
    id_map: &HashMap<ElementId, ElementId>,
) {
    if let Some(b) = binding {
        match id_map.get(&b.element_id) {
            Some(new_id) => b.element_id = new_id.clone(),
            None => *binding = None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{
        BoundElement, BoundElementKind, ElementKind, LinearData, PointBinding, TextData,
    };
    use crate::geometry::Point;

    /// Deterministic id generator: prefix + monotonically increasing counter.
    fn id_gen(prefix: &'static str) -> impl FnMut() -> ElementId {
        let mut n = 0u32;
        move || {
            let id = ElementId::new(format!("{prefix}{n}"));
            n += 1;
            id
        }
    }

    fn group_gen(prefix: &'static str) -> impl FnMut() -> GroupId {
        let mut n = 0u32;
        move || {
            let g = GroupId::new(format!("{prefix}{n}"));
            n += 1;
            g
        }
    }

    fn rect(id: &str, x: f64, y: f64) -> Element {
        Element::new(
            ElementId::from(id),
            7,
            x,
            y,
            10.0,
            10.0,
            ElementKind::Rectangle,
        )
    }

    fn arrow(id: &str) -> Element {
        Element::new(
            ElementId::from(id),
            7,
            0.0,
            0.0,
            10.0,
            0.0,
            ElementKind::Arrow(LinearData::arrow(vec![
                Point::new(0.0, 0.0),
                Point::new(10.0, 0.0),
            ])),
        )
    }

    #[test]
    fn copy_skips_missing_deleted_and_dedups() {
        let mut s = Scene::new();
        s.insert(rect("a", 0.0, 0.0));
        let mut del = rect("b", 0.0, 0.0);
        del.is_deleted = true;
        s.insert(del);
        let clip = copy(
            &s,
            &[
                ElementId::from("a"),
                ElementId::from("a"), // dup
                ElementId::from("b"), // deleted
                ElementId::from("z"), // missing
            ],
        );
        assert_eq!(clip.len(), 1);
        assert_eq!(clip.elements()[0].id, ElementId::from("a"));
    }

    #[test]
    fn paste_yields_fresh_ids_offset_correctly() {
        let mut s = Scene::new();
        s.insert(rect("a", 5.0, 5.0));
        s.insert(rect("b", 20.0, 30.0));
        let clip = copy(&s, &[ElementId::from("a"), ElementId::from("b")]);
        assert_eq!(clip.len(), 2);

        let mut ig = id_gen("new");
        let mut gg = group_gen("g");
        let pasted = paste(&clip, &mut ig, &mut gg, Vec2::new(100.0, 200.0));

        assert_eq!(pasted.len(), 2);
        // Fresh, distinct ids, none equal to the originals.
        assert_eq!(pasted[0].id, ElementId::from("new0"));
        assert_eq!(pasted[1].id, ElementId::from("new1"));
        // Offset applied.
        assert_eq!((pasted[0].x, pasted[0].y), (105.0, 205.0));
        assert_eq!((pasted[1].x, pasted[1].y), (120.0, 230.0));
        // Other fields preserved (size, seed).
        assert_eq!(pasted[0].width, 10.0);
        assert_eq!(pasted[0].seed, 7);
    }

    #[test]
    fn pasting_grouped_pair_keeps_shared_new_group() {
        let mut s = Scene::new();
        let mut a = rect("a", 0.0, 0.0);
        let mut b = rect("b", 0.0, 0.0);
        a.group_ids = vec![GroupId::from("oldg")];
        b.group_ids = vec![GroupId::from("oldg")];
        s.insert(a);
        s.insert(b);

        let clip = copy(&s, &[ElementId::from("a"), ElementId::from("b")]);
        let mut ig = id_gen("n");
        let mut gg = group_gen("g");
        let pasted = paste(&clip, &mut ig, &mut gg, Vec2::ZERO);

        assert_eq!(pasted[0].group_ids.len(), 1);
        assert_eq!(pasted[1].group_ids.len(), 1);
        // Both share the SAME new group...
        assert_eq!(pasted[0].group_ids[0], pasted[1].group_ids[0]);
        // ...which is fresh, not the old one.
        assert_eq!(pasted[0].group_ids[0], GroupId::from("g0"));
        assert_ne!(pasted[0].group_ids[0], GroupId::from("oldg"));
    }

    #[test]
    fn distinct_groups_get_distinct_new_groups() {
        let mut s = Scene::new();
        let mut a = rect("a", 0.0, 0.0);
        let mut b = rect("b", 0.0, 0.0);
        a.group_ids = vec![GroupId::from("g1")];
        b.group_ids = vec![GroupId::from("g2")];
        s.insert(a);
        s.insert(b);

        let clip = copy(&s, &[ElementId::from("a"), ElementId::from("b")]);
        let mut ig = id_gen("n");
        let mut gg = group_gen("g");
        let pasted = paste(&clip, &mut ig, &mut gg, Vec2::ZERO);
        assert_ne!(pasted[0].group_ids[0], pasted[1].group_ids[0]);
    }

    #[test]
    fn nested_group_membership_remaps_consistently() {
        // a in [inner, outer], b in [outer] only.
        let mut s = Scene::new();
        let mut a = rect("a", 0.0, 0.0);
        let mut b = rect("b", 0.0, 0.0);
        a.group_ids = vec![GroupId::from("inner"), GroupId::from("outer")];
        b.group_ids = vec![GroupId::from("outer")];
        s.insert(a);
        s.insert(b);

        let clip = copy(&s, &[ElementId::from("a"), ElementId::from("b")]);
        let mut ig = id_gen("n");
        let mut gg = group_gen("g");
        let pasted = paste(&clip, &mut ig, &mut gg, Vec2::ZERO);

        assert_eq!(pasted[0].group_ids.len(), 2);
        // a's outer == b's outer (shared), and both fresh.
        assert_eq!(pasted[0].group_ids[1], pasted[1].group_ids[0]);
        // a's inner != a's outer.
        assert_ne!(pasted[0].group_ids[0], pasted[0].group_ids[1]);
    }

    #[test]
    fn arrow_bound_to_copied_rect_remaps_binding() {
        let mut s = Scene::new();
        let mut r = rect("rect", 0.0, 0.0);
        let mut ar = arrow("arr");
        // Wire arrow.end_binding -> rect, and rect.bound_elements -> arrow.
        if let ElementKind::Arrow(l) = &mut ar.kind {
            l.end_binding = Some(PointBinding {
                element_id: ElementId::from("rect"),
                focus: 0.0,
                gap: 4.0,
            });
        }
        r.bound_elements = vec![BoundElement {
            id: ElementId::from("arr"),
            kind: BoundElementKind::Arrow,
        }];
        s.insert(r);
        s.insert(ar);

        let clip = copy(&s, &[ElementId::from("rect"), ElementId::from("arr")]);
        let mut ig = id_gen("n");
        let mut gg = group_gen("g");
        let pasted = paste(&clip, &mut ig, &mut gg, Vec2::ZERO);

        // pasted[0] is the new rect (n0), pasted[1] is the new arrow (n1).
        let new_rect = &pasted[0];
        let new_arrow = &pasted[1];
        assert_eq!(new_rect.id, ElementId::from("n0"));
        assert_eq!(new_arrow.id, ElementId::from("n1"));

        // Arrow binding now points at the NEW rect id, not the old one.
        if let ElementKind::Arrow(l) = &new_arrow.kind {
            let b = l.end_binding.as_ref().expect("binding kept");
            assert_eq!(b.element_id, new_rect.id);
            assert_eq!(b.gap, 4.0);
        } else {
            panic!("expected arrow");
        }
        // rect.bound_elements remapped to the new arrow id.
        assert_eq!(new_rect.bound_elements.len(), 1);
        assert_eq!(new_rect.bound_elements[0].id, new_arrow.id);
    }

    #[test]
    fn arrow_bound_to_noncopied_element_clears_binding() {
        let mut s = Scene::new();
        let r = rect("rect", 0.0, 0.0);
        let mut ar = arrow("arr");
        if let ElementKind::Arrow(l) = &mut ar.kind {
            l.start_binding = Some(PointBinding {
                element_id: ElementId::from("rect"),
                focus: 0.0,
                gap: 4.0,
            });
        }
        s.insert(r);
        s.insert(ar);

        // Copy ONLY the arrow, not the rect it is bound to.
        let clip = copy(&s, &[ElementId::from("arr")]);
        let mut ig = id_gen("n");
        let mut gg = group_gen("g");
        let pasted = paste(&clip, &mut ig, &mut gg, Vec2::ZERO);

        assert_eq!(pasted.len(), 1);
        if let ElementKind::Arrow(l) = &pasted[0].kind {
            assert!(
                l.start_binding.is_none(),
                "dangling binding must be cleared"
            );
        } else {
            panic!("expected arrow");
        }
    }

    #[test]
    fn bound_elements_to_noncopied_target_dropped() {
        let mut s = Scene::new();
        let mut r = rect("rect", 0.0, 0.0);
        r.bound_elements = vec![BoundElement {
            id: ElementId::from("arr"), // not copied
            kind: BoundElementKind::Arrow,
        }];
        s.insert(r);

        let clip = copy(&s, &[ElementId::from("rect")]);
        let mut ig = id_gen("n");
        let mut gg = group_gen("g");
        let pasted = paste(&clip, &mut ig, &mut gg, Vec2::ZERO);
        assert!(
            pasted[0].bound_elements.is_empty(),
            "dangling bound element must be dropped"
        );
    }

    #[test]
    fn text_container_remaps_or_clears() {
        // Container rect + bound text label.
        let mut s = Scene::new();
        let container = rect("c", 0.0, 0.0);
        let label = Element::new(
            ElementId::from("t"),
            7,
            1.0,
            1.0,
            8.0,
            8.0,
            ElementKind::Text({
                let mut td = TextData::new("hi");
                td.container_id = Some(ElementId::from("c"));
                td
            }),
        );
        s.insert(container);
        s.insert(label);

        // Both copied: container_id remaps.
        {
            let clip = copy(&s, &[ElementId::from("c"), ElementId::from("t")]);
            let mut ig = id_gen("n");
            let mut gg = group_gen("g");
            let pasted = paste(&clip, &mut ig, &mut gg, Vec2::ZERO);
            let new_container = &pasted[0];
            if let ElementKind::Text(td) = &pasted[1].kind {
                assert_eq!(td.container_id.as_ref(), Some(&new_container.id));
            } else {
                panic!("expected text");
            }
        }
        // Only the label copied: container_id cleared.
        {
            let clip = copy(&s, &[ElementId::from("t")]);
            let mut ig = id_gen("m");
            let mut gg = group_gen("h");
            let pasted = paste(&clip, &mut ig, &mut gg, Vec2::ZERO);
            if let ElementKind::Text(td) = &pasted[0].kind {
                assert!(td.container_id.is_none(), "orphaned label cleared");
            } else {
                panic!("expected text");
            }
        }
    }

    #[test]
    fn frame_id_remaps_when_frame_copied_else_clears() {
        // `frame_id` is just an ElementId reference; the remap logic is
        // kind-agnostic, so a plain rect stands in for the frame referent here
        // (FrameData is not re-exported from the element module, and this test
        // exercises the reference plumbing, not frame geometry).
        let mut s = Scene::new();
        let frame = rect("f", 0.0, 0.0);
        let mut child = rect("ch", 5.0, 5.0);
        child.frame_id = Some(ElementId::from("f"));
        s.insert(frame);
        s.insert(child);

        // Both copied: frame_id remaps.
        {
            let clip = copy(&s, &[ElementId::from("f"), ElementId::from("ch")]);
            let mut ig = id_gen("n");
            let mut gg = group_gen("g");
            let pasted = paste(&clip, &mut ig, &mut gg, Vec2::ZERO);
            let new_frame_id = &pasted[0].id;
            assert_eq!(pasted[1].frame_id.as_ref(), Some(new_frame_id));
        }
        // Only child copied: frame_id cleared.
        {
            let clip = copy(&s, &[ElementId::from("ch")]);
            let mut ig = id_gen("m");
            let mut gg = group_gen("h");
            let pasted = paste(&clip, &mut ig, &mut gg, Vec2::ZERO);
            assert!(pasted[0].frame_id.is_none());
        }
    }

    #[test]
    fn duplicate_is_copy_then_paste() {
        let mut s = Scene::new();
        let mut a = rect("a", 0.0, 0.0);
        let mut b = rect("b", 0.0, 0.0);
        a.group_ids = vec![GroupId::from("g")];
        b.group_ids = vec![GroupId::from("g")];
        s.insert(a);
        s.insert(b);

        let mut ig = id_gen("d");
        let mut gg = group_gen("dg");
        let dup = duplicate(
            &s,
            &[ElementId::from("a"), ElementId::from("b")],
            &mut ig,
            &mut gg,
            Vec2::new(10.0, 10.0),
        );
        assert_eq!(dup.len(), 2);
        assert_eq!(dup[0].id, ElementId::from("d0"));
        assert_eq!((dup[0].x, dup[0].y), (10.0, 10.0));
        // Shared fresh group preserved through duplicate.
        assert_eq!(dup[0].group_ids[0], dup[1].group_ids[0]);
    }

    #[test]
    fn empty_clipboard_pastes_nothing() {
        let clip = Clipboard::new();
        assert!(clip.is_empty());
        let mut ig = id_gen("n");
        let mut gg = group_gen("g");
        let pasted = paste(&clip, &mut ig, &mut gg, Vec2::new(1.0, 1.0));
        assert!(pasted.is_empty());
    }
}
