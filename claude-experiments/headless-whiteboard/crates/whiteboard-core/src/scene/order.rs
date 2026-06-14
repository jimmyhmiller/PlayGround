//! Fractional-index z-ordering.
//!
//! Reimplemented in Rust from the algorithm Excalidraw uses for stable element
//! ordering. Upstream derivation:
//! - `packages/excalidraw/fractionalIndex.ts` (Excalidraw, MIT) — how fractional
//!   indices drive paint order and the bring/send/raise/lower operations.
//! - David Greenspan's `fractional-indexing` JS library (MIT), which Excalidraw
//!   depends on, for the base-62 `generate_key_between` / midpoint algorithm.
//!
//! A fractional index is an opaque, lexicographically-ordered string. Between any
//! two existing indices a new one can always be generated, so inserting an
//! element "between" two others is O(1) and never requires renumbering its
//! neighbours. Paint order is simply the elements sorted by `(index, id)`.
//!
//! We use the same alphabet and integer/fraction split as the upstream library:
//! a leading magnitude character encodes how many digits the integer part has,
//! followed by the integer digits, optionally followed by a fractional tail.

use crate::element::ElementId;
use crate::scene::Scene;

/// The digit alphabet, ascending in byte order. Matches `fractional-indexing`.
const DIGITS: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const BASE: usize = 62;

/// The "smallest" integer magnitude head (`a0`) — the start of the integer line.
/// Heads `a`..=`z` cover positive magnitudes, `Z`..=`A` cover negative ones,
/// exactly as in the reference implementation.
const SMALLEST_INTEGER: &str = "A00000000000000000000000000";

/// Errors from fractional-index generation. We never silently fall back to a
/// bogus key: an invalid ordering request is a programming error and surfaces.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IndexError {
    #[error("fractional index key is empty")]
    Empty,
    #[error("invalid fractional index head byte: {0:?}")]
    InvalidHead(u8),
    #[error("invalid order key digit: {0:?}")]
    InvalidDigit(u8),
    #[error("`a` must be < `b` to generate a key between them: {a:?} !< {b:?}")]
    NotAscending { a: String, b: String },
    #[error("ran out of integer range generating a fractional index")]
    IntegerOverflow,
}

#[inline]
fn digit_value(b: u8) -> Result<usize, IndexError> {
    DIGITS
        .iter()
        .position(|&d| d == b)
        .ok_or(IndexError::InvalidDigit(b))
}

/// How many digits the integer part has, given its head byte. Positive heads
/// `a..z` => 2..=27 digits; negative heads `Z..A` => mirror. Returns the total
/// length (head + digits) of the integer portion.
fn integer_len(head: u8) -> Result<usize, IndexError> {
    if head.is_ascii_lowercase() {
        // 'a' => 2, ..., 'z' => 27
        Ok((head - b'a') as usize + 2)
    } else if head.is_ascii_uppercase() {
        // 'Z' => 2, ..., 'A' => 27
        Ok((b'Z' - head) as usize + 2)
    } else {
        Err(IndexError::InvalidHead(head))
    }
}

/// Split a key into its integer portion and its (possibly empty) fractional tail.
fn integer_part(key: &str) -> Result<&str, IndexError> {
    let bytes = key.as_bytes();
    let head = *bytes.first().ok_or(IndexError::Empty)?;
    let len = integer_len(head)?;
    if len > key.len() {
        return Err(IndexError::InvalidHead(head));
    }
    Ok(&key[..len])
}

/// Increment an integer key (head + digits) to the next integer, growing or
/// shrinking the magnitude head when the digits roll over. Returns `None` on
/// overflow of the representable range (`head == 'z'`). Faithful port of
/// `incrementInteger` from the `fractional-indexing` library.
fn increment_integer(int: &str) -> Result<Option<String>, IndexError> {
    let bytes = int.as_bytes();
    let head = bytes[0];
    let mut digits: Vec<usize> = bytes[1..]
        .iter()
        .map(|&b| digit_value(b))
        .collect::<Result<_, _>>()?;

    let mut carry = true;
    for d in digits.iter_mut().rev() {
        if !carry {
            break;
        }
        let v = *d + 1;
        if v == BASE {
            *d = 0;
            carry = true;
        } else {
            *d = v;
            carry = false;
        }
    }

    if carry {
        // Crossed the negative→positive boundary, or overflowed entirely.
        if head == b'Z' {
            return Ok(Some("a0".to_string()));
        }
        if head == b'z' {
            return Ok(None);
        }
        let h = head + 1;
        if h > b'a' {
            // Positive line: grew a magnitude, append a zero digit.
            digits.push(0);
        } else {
            // Negative line: shrank toward zero, drop a digit.
            digits.pop();
        }
        let mut s = String::new();
        s.push(h as char);
        for &d in &digits {
            s.push(DIGITS[d] as char);
        }
        return Ok(Some(s));
    }

    let mut s = String::new();
    s.push(head as char);
    for &d in &digits {
        s.push(DIGITS[d] as char);
    }
    Ok(Some(s))
}

/// The lexicographic midpoint between two fractional digit strings (no heads),
/// where `a < b` or `b` is empty (meaning "+infinity"). Port of the reference
/// `midpoint`.
fn midpoint(a: &str, b: &str) -> Result<String, IndexError> {
    if !b.is_empty() {
        // Strip the common prefix, recursing past it.
        let common: usize = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
        if common > 0 {
            let prefix = &b[..common];
            let rest = midpoint(&a[common..], &b[common..])?;
            return Ok(format!("{prefix}{rest}"));
        }
    }

    // No common prefix. Work on the first digits.
    let digit_a = if a.is_empty() {
        0
    } else {
        digit_value(a.as_bytes()[0])?
    };
    let digit_b = if b.is_empty() {
        BASE
    } else {
        digit_value(b.as_bytes()[0])?
    };

    if digit_b - digit_a > 1 {
        let mid = (digit_a + digit_b) / 2;
        return Ok((DIGITS[mid] as char).to_string());
    }

    // Adjacent first digits: keep b's first digit, recurse into a's tail vs "".
    if !b.is_empty() && b.len() > 1 {
        return Ok(b[..1].to_string());
    }
    // b exhausted (or single digit): descend into a's fractional tail.
    let head = (DIGITS[digit_a] as char).to_string();
    let tail = midpoint(if a.is_empty() { "" } else { &a[1..] }, "")?;
    Ok(format!("{head}{tail}"))
}

/// Validate that a key parses (head + valid digits + valid fractional tail) and
/// has no trailing zero in its fractional part (the canonical form).
fn validate_key(key: &str) -> Result<(), IndexError> {
    let int = integer_part(key)?;
    let frac = &key[int.len()..];
    // integer digits valid
    for &b in &int.as_bytes()[1..] {
        digit_value(b)?;
    }
    // fractional digits valid
    for &b in frac.as_bytes() {
        digit_value(b)?;
    }
    Ok(())
}

/// Generate a fractional index strictly between `a` and `b`.
///
/// `a == None` means "before everything"; `b == None` means "after everything".
/// At least the produced key `k` satisfies `a < k < b` lexicographically. Port
/// of `generateKeyBetween`.
pub fn generate_key_between(a: Option<&str>, b: Option<&str>) -> Result<String, IndexError> {
    if let Some(a) = a {
        validate_key(a)?;
    }
    if let Some(b) = b {
        validate_key(b)?;
    }
    if let (Some(a), Some(b)) = (a, b) {
        if a >= b {
            return Err(IndexError::NotAscending {
                a: a.to_string(),
                b: b.to_string(),
            });
        }
    }

    match (a, b) {
        (None, None) => Ok("a0".to_string()),
        (None, Some(b)) => {
            // Key before b.
            let int_b = integer_part(b)?;
            let frac_b = &b[int_b.len()..];
            if int_b == SMALLEST_INTEGER {
                // Must go into the fractional space below b.
                let m = midpoint("", frac_b)?;
                return Ok(format!("{int_b}{m}"));
            }
            if int_b < b {
                // b has a fractional part — the integer alone is a valid key < b.
                return Ok(int_b.to_string());
            }
            // Decrement the integer.
            let dec = decrement_integer(int_b)?;
            Ok(dec)
        }
        (Some(a), None) => {
            // Key after a.
            let int_a = integer_part(a)?;
            let frac_a = &a[int_a.len()..];
            match increment_integer(int_a)? {
                Some(inc) => Ok(inc),
                None => {
                    // Integer maxed out: extend fractionally above a.
                    let m = midpoint(frac_a, "")?;
                    Ok(format!("{int_a}{m}"))
                }
            }
        }
        (Some(a), Some(b)) => {
            let int_a = integer_part(a)?;
            let int_b = integer_part(b)?;
            let frac_a = &a[int_a.len()..];
            let frac_b = &b[int_b.len()..];
            if int_a == int_b {
                // Same integer: midpoint of the fractional tails.
                let m = midpoint(frac_a, frac_b)?;
                return Ok(format!("{int_a}{m}"));
            }
            // Different integers: try the integer just above a.
            let ia = increment_integer(int_a)?.ok_or(IndexError::IntegerOverflow)?;
            if ia.as_str() < b {
                return Ok(ia);
            }
            // Otherwise extend a fractionally.
            let m = midpoint(frac_a, "")?;
            Ok(format!("{int_a}{m}"))
        }
    }
}

/// Decrement an integer key to the previous integer, growing or shrinking the
/// magnitude head when digits underflow. Returns an error on underflow of the
/// representable range (`head == 'A'`). Faithful port of `decrementInteger`.
fn decrement_integer(int: &str) -> Result<String, IndexError> {
    let bytes = int.as_bytes();
    let head = bytes[0];
    let mut digits: Vec<usize> = bytes[1..]
        .iter()
        .map(|&b| digit_value(b))
        .collect::<Result<_, _>>()?;

    let mut borrow = true;
    for d in digits.iter_mut().rev() {
        if !borrow {
            break;
        }
        if *d == 0 {
            *d = BASE - 1;
            borrow = true;
        } else {
            *d -= 1;
            borrow = false;
        }
    }

    if borrow {
        if head == b'a' {
            // Cross the positive→negative boundary to "Z" + max digit.
            let mut s = String::new();
            s.push('Z');
            s.push(DIGITS[BASE - 1] as char);
            return Ok(s);
        }
        if head == b'A' {
            return Err(IndexError::IntegerOverflow);
        }
        let h = head - 1;
        if h < b'Z' {
            // Negative line: grew a magnitude, append a max digit.
            digits.push(BASE - 1);
        } else {
            // Positive line: shrank, drop a digit.
            digits.pop();
        }
        let mut s = String::new();
        s.push(h as char);
        for &d in &digits {
            s.push(DIGITS[d] as char);
        }
        return Ok(s);
    }

    let mut s = String::new();
    s.push(head as char);
    for &d in &digits {
        s.push(DIGITS[d] as char);
    }
    Ok(s)
}

/// Generate `n` evenly spread keys strictly between `a` and `b`, all distinct
/// and ascending. Port of `generateNKeysBetween` (simple bisection form).
pub fn generate_n_keys_between(
    a: Option<&str>,
    b: Option<&str>,
    n: usize,
) -> Result<Vec<String>, IndexError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![generate_key_between(a, b)?]);
    }
    let mid = n / 2;
    let mid_key = generate_key_between(a, b)?;
    let mut left = generate_n_keys_between(a, Some(&mid_key), mid)?;
    let mut right = generate_n_keys_between(Some(&mid_key), b, n - mid - 1)?;
    let mut out = Vec::with_capacity(n);
    out.append(&mut left);
    out.push(mid_key);
    out.append(&mut right);
    Ok(out)
}

/// Compare two elements for paint order: by fractional `index`, then by `id` as
/// the deterministic tiebreaker (mirrors Excalidraw, which also falls back to id
/// so concurrent inserts with equal indices still order deterministically).
///
/// Elements with no index sort *before* those with one, matching "legacy"
/// elements that predate fractional indexing.
pub fn compare_order(
    a_index: Option<&str>,
    a_id: &ElementId,
    b_index: Option<&str>,
    b_id: &ElementId,
) -> std::cmp::Ordering {
    match (a_index, b_index) {
        (Some(x), Some(y)) => x.cmp(y).then_with(|| a_id.cmp(b_id)),
        (None, None) => a_id.cmp(b_id),
        (None, Some(_)) => std::cmp::Ordering::Less,
        (Some(_), None) => std::cmp::Ordering::Greater,
    }
}

impl Scene {
    /// (Re)generate the paint order from each element's fractional `index`,
    /// breaking ties by id. Elements lacking an index are ordered before indexed
    /// ones (legacy elements), then by id. This is the canonical way to derive
    /// `order()` once indices are authoritative.
    pub fn reorder_by_index(&mut self) {
        let mut ids: Vec<ElementId> = self.order_vec_clone();
        ids.sort_by(|a, b| {
            let ai = self.index_of(a);
            let bi = self.index_of(b);
            compare_order(ai.as_deref(), a, bi.as_deref(), b)
        });
        self.set_order(ids);
    }

    /// Assign a fresh fractional index to every element in the current paint
    /// order so that sorting by index reproduces that order. Spreads `n` keys
    /// across the whole line. Bumps each touched element's version via the
    /// provided nonce source (so the change is observable), passing
    /// `version_nonce = 0` and no timestamp — callers that need sync metadata can
    /// re-touch.
    pub fn assign_initial_indices(&mut self) -> Result<(), IndexError> {
        let ids = self.order_vec_clone();
        let keys = generate_n_keys_between(None, None, ids.len())?;
        for (id, key) in ids.iter().zip(keys.into_iter()) {
            if let Some(el) = self.get_mut(id) {
                el.index = Some(key);
            }
        }
        Ok(())
    }

    /// Compute a fractional index that would place an element between the
    /// elements currently at paint positions `lower` and `upper` (by current
    /// paint order). `None` bounds mean "below everything" / "above everything".
    pub fn index_between(
        &self,
        lower: Option<&ElementId>,
        upper: Option<&ElementId>,
    ) -> Result<String, IndexError> {
        let a = lower.and_then(|id| self.index_of(id));
        let b = upper.and_then(|id| self.index_of(id));
        generate_key_between(a.as_deref(), b.as_deref())
    }

    /// Bring the given ids to the front (top of paint order) while preserving
    /// their relative order. Reassigns their indices to be above every other
    /// element, then re-derives paint order.
    pub fn bring_to_front(&mut self, ids: &[ElementId]) -> Result<(), IndexError> {
        self.move_block(ids, MoveTarget::Front)
    }

    /// Send the given ids to the back (bottom of paint order), preserving their
    /// relative order.
    pub fn send_to_back(&mut self, ids: &[ElementId]) -> Result<(), IndexError> {
        self.move_block(ids, MoveTarget::Back)
    }

    /// Raise the given ids one step: move them just above the lowest element
    /// that is currently above the highest of the moved set.
    pub fn raise(&mut self, ids: &[ElementId]) -> Result<(), IndexError> {
        self.move_step(ids, true)
    }

    /// Lower the given ids one step: move them just below the highest element
    /// currently below the lowest of the moved set.
    pub fn lower(&mut self, ids: &[ElementId]) -> Result<(), IndexError> {
        self.move_step(ids, false)
    }

    // ---- internals -------------------------------------------------------

    fn move_block(&mut self, ids: &[ElementId], target: MoveTarget) -> Result<(), IndexError> {
        let moved = self.ordered_subset(ids);
        if moved.is_empty() {
            return Ok(());
        }
        // The remaining elements keep their order; we generate `moved.len()` keys
        // either entirely above or entirely below the current extremes.
        let rest: Vec<ElementId> = self
            .order_vec_clone()
            .into_iter()
            .filter(|id| !moved.contains(id))
            .collect();

        let keys = match target {
            MoveTarget::Front => {
                let lower = rest.last().and_then(|id| self.index_of(id));
                generate_n_keys_between(lower.as_deref(), None, moved.len())?
            }
            MoveTarget::Back => {
                let upper = rest.first().and_then(|id| self.index_of(id));
                generate_n_keys_between(None, upper.as_deref(), moved.len())?
            }
        };
        self.apply_keys(&moved, keys);
        self.reorder_by_index();
        Ok(())
    }

    fn move_step(&mut self, ids: &[ElementId], up: bool) -> Result<(), IndexError> {
        let moved = self.ordered_subset(ids);
        if moved.is_empty() {
            return Ok(());
        }
        let order = self.order_vec_clone();
        let moved_set: std::collections::HashSet<&ElementId> = moved.iter().collect();
        // Positions of moved elements in current order.
        let positions: Vec<usize> = order
            .iter()
            .enumerate()
            .filter(|(_, id)| moved_set.contains(id))
            .map(|(i, _)| i)
            .collect();

        if up {
            // The neighbour just above the topmost moved element.
            let top = *positions.last().unwrap();
            // Find first index above `top` not in the moved set.
            let above = (top + 1..order.len()).find(|&i| !moved_set.contains(&order[i]));
            let Some(above) = above else {
                return Ok(()); // already at front
            };
            // Place moved block between `above` and the element after it.
            let lower = self.index_of(&order[above]);
            let upper = order.get(above + 1).and_then(|id| self.index_of(id));
            let keys = generate_n_keys_between(lower.as_deref(), upper.as_deref(), moved.len())?;
            self.apply_keys(&moved, keys);
        } else {
            let bottom = positions[0];
            let below = (0..bottom).rev().find(|&i| !moved_set.contains(&order[i]));
            let Some(below) = below else {
                return Ok(()); // already at back
            };
            let upper = self.index_of(&order[below]);
            let lower = if below == 0 {
                None
            } else {
                self.index_of(&order[below - 1])
            };
            let keys = generate_n_keys_between(lower.as_deref(), upper.as_deref(), moved.len())?;
            self.apply_keys(&moved, keys);
        }
        self.reorder_by_index();
        Ok(())
    }

    /// The subset of `ids` that exist, in current paint order.
    fn ordered_subset(&self, ids: &[ElementId]) -> Vec<ElementId> {
        let want: std::collections::HashSet<&ElementId> = ids.iter().collect();
        self.order_vec_clone()
            .into_iter()
            .filter(|id| want.contains(id))
            .collect()
    }

    fn apply_keys(&mut self, ids: &[ElementId], keys: Vec<String>) {
        for (id, key) in ids.iter().zip(keys.into_iter()) {
            if let Some(el) = self.get_mut(id) {
                el.index = Some(key);
            }
        }
    }

    fn index_of(&self, id: &ElementId) -> Option<String> {
        self.get(id).and_then(|e| e.index.clone())
    }
}

#[derive(Clone, Copy)]
enum MoveTarget {
    Front,
    Back,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{Element, ElementKind};

    fn el(id: &str) -> Element {
        Element::new(
            ElementId::from(id),
            1,
            0.0,
            0.0,
            10.0,
            10.0,
            ElementKind::Rectangle,
        )
    }

    // ---- generate_key_between --------------------------------------------

    #[test]
    fn first_key_is_a0() {
        assert_eq!(generate_key_between(None, None).unwrap(), "a0");
    }

    #[test]
    fn append_increments_integer() {
        let mut prev = generate_key_between(None, None).unwrap();
        for _ in 0..200 {
            let next = generate_key_between(Some(&prev), None).unwrap();
            assert!(prev < next, "expected {prev} < {next}");
            prev = next;
        }
    }

    #[test]
    fn prepend_decrements_below_a0() {
        let a0 = generate_key_between(None, None).unwrap();
        let before = generate_key_between(None, Some(&a0)).unwrap();
        assert!(before < a0, "{before} < {a0}");
        let before2 = generate_key_between(None, Some(&before)).unwrap();
        assert!(before2 < before, "{before2} < {before}");
    }

    #[test]
    fn between_is_strictly_ordered() {
        let a = generate_key_between(None, None).unwrap();
        let b = generate_key_between(Some(&a), None).unwrap();
        let mid = generate_key_between(Some(&a), Some(&b)).unwrap();
        assert!(a < mid && mid < b, "{a} < {mid} < {b}");
    }

    #[test]
    fn repeated_between_never_collides() {
        // Insert 500 keys all between a0 and a1 by always bisecting the lower gap.
        let lo = generate_key_between(None, None).unwrap();
        let hi = generate_key_between(Some(&lo), None).unwrap();
        let mut left = lo.clone();
        let mut seen = std::collections::HashSet::new();
        seen.insert(lo.clone());
        seen.insert(hi.clone());
        for _ in 0..500 {
            let k = generate_key_between(Some(&left), Some(&hi)).unwrap();
            assert!(left < k && k < hi, "{left} < {k} < {hi}");
            assert!(seen.insert(k.clone()), "duplicate key generated: {k}");
            left = k;
        }
    }

    #[test]
    fn rejects_non_ascending_bounds() {
        let a = generate_key_between(None, None).unwrap();
        let b = generate_key_between(Some(&a), None).unwrap();
        assert!(matches!(
            generate_key_between(Some(&b), Some(&a)),
            Err(IndexError::NotAscending { .. })
        ));
        assert!(matches!(
            generate_key_between(Some(&a), Some(&a)),
            Err(IndexError::NotAscending { .. })
        ));
    }

    #[test]
    fn rejects_invalid_key() {
        // '/' is below '0' in the alphabet and not a valid digit/head.
        assert!(generate_key_between(Some("/"), None).is_err());
        assert!(generate_key_between(None, Some("")).is_err());
    }

    // ---- generate_n_keys_between -----------------------------------------

    #[test]
    fn n_keys_are_ascending_and_distinct() {
        for n in [0usize, 1, 2, 3, 7, 16, 50] {
            let keys = generate_n_keys_between(None, None, n).unwrap();
            assert_eq!(keys.len(), n);
            for w in keys.windows(2) {
                assert!(w[0] < w[1], "not ascending: {:?}", w);
            }
            let set: std::collections::HashSet<_> = keys.iter().collect();
            assert_eq!(set.len(), keys.len(), "duplicates in {keys:?}");
        }
    }

    #[test]
    fn n_keys_within_bounds() {
        let lo = "a0".to_string();
        let hi = generate_key_between(Some(&lo), None).unwrap();
        let keys = generate_n_keys_between(Some(&lo), Some(&hi), 10).unwrap();
        for k in &keys {
            assert!(lo < *k && *k < hi, "{lo} < {k} < {hi}");
        }
    }

    // ---- compare_order ----------------------------------------------------

    #[test]
    fn compare_indexed_before_unindexed() {
        let a = ElementId::from("a");
        let b = ElementId::from("b");
        // None sorts before Some.
        assert_eq!(
            compare_order(None, &a, Some("a0"), &b),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_order(Some("a0"), &a, None, &b),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn compare_breaks_ties_by_id() {
        let a = ElementId::from("a");
        let b = ElementId::from("b");
        assert_eq!(
            compare_order(Some("a0"), &a, Some("a0"), &b),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_order(None, &b, None, &a),
            std::cmp::Ordering::Greater
        );
    }

    // ---- Scene integration -----------------------------------------------

    #[test]
    fn assign_initial_then_reorder_is_identity() {
        let mut s = Scene::new();
        for id in ["a", "b", "c", "d"] {
            s.insert(el(id));
        }
        s.assign_initial_indices().unwrap();
        s.reorder_by_index();
        let ids: Vec<_> = s.order().iter().map(|i| i.as_str().to_string()).collect();
        assert_eq!(ids, ["a", "b", "c", "d"]);
        // Indices are strictly ascending in paint order.
        let idx: Vec<String> = s
            .order()
            .iter()
            .map(|i| s.get(i).unwrap().index.clone().unwrap())
            .collect();
        for w in idx.windows(2) {
            assert!(w[0] < w[1], "{:?} not ascending", w);
        }
    }

    #[test]
    fn index_between_inserts_in_the_middle() {
        let mut s = Scene::new();
        for id in ["a", "b", "c"] {
            s.insert(el(id));
        }
        s.assign_initial_indices().unwrap();
        // Insert "x" between a and b.
        let a = ElementId::from("a");
        let b = ElementId::from("b");
        let key = s.index_between(Some(&a), Some(&b)).unwrap();
        let mut x = el("x");
        x.index = Some(key);
        s.insert(x);
        s.reorder_by_index();
        let ids: Vec<_> = s.order().iter().map(|i| i.as_str().to_string()).collect();
        assert_eq!(ids, ["a", "x", "b", "c"]);
    }

    #[test]
    fn bring_to_front_and_send_to_back() {
        let mut s = Scene::new();
        for id in ["a", "b", "c", "d"] {
            s.insert(el(id));
        }
        s.assign_initial_indices().unwrap();
        s.bring_to_front(&[ElementId::from("a")]).unwrap();
        let ids: Vec<_> = s.order().iter().map(|i| i.as_str().to_string()).collect();
        assert_eq!(ids, ["b", "c", "d", "a"]);

        s.send_to_back(&[ElementId::from("d")]).unwrap();
        let ids: Vec<_> = s.order().iter().map(|i| i.as_str().to_string()).collect();
        assert_eq!(ids, ["d", "b", "c", "a"]);
    }

    #[test]
    fn bring_to_front_preserves_relative_order_of_set() {
        let mut s = Scene::new();
        for id in ["a", "b", "c", "d", "e"] {
            s.insert(el(id));
        }
        s.assign_initial_indices().unwrap();
        // Bring a and c to front; they should keep order a then c, on top.
        s.bring_to_front(&[ElementId::from("c"), ElementId::from("a")])
            .unwrap();
        let ids: Vec<_> = s.order().iter().map(|i| i.as_str().to_string()).collect();
        assert_eq!(ids, ["b", "d", "e", "a", "c"]);
    }

    #[test]
    fn raise_and_lower_one_step() {
        let mut s = Scene::new();
        for id in ["a", "b", "c", "d"] {
            s.insert(el(id));
        }
        s.assign_initial_indices().unwrap();
        // Raise b one step: b moves above c.
        s.raise(&[ElementId::from("b")]).unwrap();
        let ids: Vec<_> = s.order().iter().map(|i| i.as_str().to_string()).collect();
        assert_eq!(ids, ["a", "c", "b", "d"]);

        // Lower b one step: back to a,b,c,d.
        s.lower(&[ElementId::from("b")]).unwrap();
        let ids: Vec<_> = s.order().iter().map(|i| i.as_str().to_string()).collect();
        assert_eq!(ids, ["a", "b", "c", "d"]);
    }

    #[test]
    fn raise_at_front_is_noop() {
        let mut s = Scene::new();
        for id in ["a", "b"] {
            s.insert(el(id));
        }
        s.assign_initial_indices().unwrap();
        s.raise(&[ElementId::from("b")]).unwrap();
        let ids: Vec<_> = s.order().iter().map(|i| i.as_str().to_string()).collect();
        assert_eq!(ids, ["a", "b"]);
    }

    #[test]
    fn lower_at_back_is_noop() {
        let mut s = Scene::new();
        for id in ["a", "b"] {
            s.insert(el(id));
        }
        s.assign_initial_indices().unwrap();
        s.lower(&[ElementId::from("a")]).unwrap();
        let ids: Vec<_> = s.order().iter().map(|i| i.as_str().to_string()).collect();
        assert_eq!(ids, ["a", "b"]);
    }

    #[test]
    fn reorder_places_unindexed_before_indexed() {
        let mut s = Scene::new();
        let mut a = el("a"); // no index
        a.index = None;
        let mut b = el("b");
        b.index = Some("a0".to_string());
        s.insert(b);
        s.insert(a);
        s.reorder_by_index();
        let ids: Vec<_> = s.order().iter().map(|i| i.as_str().to_string()).collect();
        // unindexed "a" sorts before indexed "b".
        assert_eq!(ids, ["a", "b"]);
    }
}
