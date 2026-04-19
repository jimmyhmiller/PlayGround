//! Ports of CodeMirror 6 `state/test/test-selection.ts` and
//! `state/test/test-charcategory.ts`.

use editor_core::selection::{Range, Selection};

fn rngs(s: &Selection) -> String {
    s.ranges
        .iter()
        .map(|r| format!("{}/{}", r.anchor, r.head))
        .collect::<Vec<_>>()
        .join(",")
}

mod editor_selection {
    use super::*;

    #[test]
    fn stores_ranges_with_a_primary_range() {
        let sel = Selection::new(
            vec![Range::new(0, 1), Range::new(3, 2), Range::new(4, 5)],
            1,
        );
        assert_eq!(sel.main().from(), 2);
        assert_eq!(sel.main().to(), 3);
        assert_eq!(sel.main().anchor, 3);
        assert_eq!(sel.main().head, 2);
        assert_eq!(rngs(&sel), "0/1,3/2,4/5");
    }

    #[test]
    fn merges_and_sorts_ranges_when_normalizing() {
        let sel = Selection::new(
            vec![
                Range::new(10, 12),
                Range::new(6, 7),
                Range::new(4, 5),
                Range::new(3, 4),
                Range::new(0, 6),
                Range::new(7, 8),
                Range::new(9, 13),
                Range::new(13, 14),
            ],
            0,
        );
        assert_eq!(rngs(&sel), "0/6,6/7,7/8,9/13,13/14");
    }

    #[test]
    fn merges_adjacent_point_ranges_when_normalizing() {
        let sel = Selection::new(
            vec![
                Range::new(10, 12),
                Range::new(12, 12),
                Range::new(12, 12),
                Range::new(10, 10),
                Range::new(8, 10),
            ],
            0,
        );
        assert_eq!(rngs(&sel), "8/10,10/12");
    }

    #[test]
    fn preserves_direction_of_last_range_when_merging() {
        let sel = Selection::new(vec![Range::new(0, 2), Range::new(10, 1)], 0);
        assert_eq!(rngs(&sel), "10/0");
    }
}

mod char_categorizer {
    use editor_core::char_class::{categorize, CharCategory};

    #[test]
    fn categorises_into_alphanumeric() {
        assert_eq!(categorize('1'), CharCategory::Word);
        assert_eq!(categorize('a'), CharCategory::Word);
    }

    #[test]
    fn categorises_into_whitespace() {
        assert_eq!(categorize(' '), CharCategory::Space);
    }

    #[test]
    fn categorises_into_other() {
        assert_eq!(categorize('/'), CharCategory::Other);
        assert_eq!(categorize('<'), CharCategory::Other);
    }
}
