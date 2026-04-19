use ropey::Rope;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range {
    pub anchor: usize,
    pub head: usize,
}

impl Range {
    pub fn cursor(pos: usize) -> Self {
        Self { anchor: pos, head: pos }
    }

    pub fn new(anchor: usize, head: usize) -> Self {
        Self { anchor, head }
    }

    pub fn from(&self) -> usize {
        self.anchor.min(self.head)
    }

    pub fn to(&self) -> usize {
        self.anchor.max(self.head)
    }

    pub fn is_empty(&self) -> bool {
        self.anchor == self.head
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Selection {
    pub ranges: Vec<Range>,
    pub primary: usize,
}

impl Selection {
    pub fn single(range: Range) -> Self {
        Self { ranges: vec![range], primary: 0 }
    }

    pub fn cursor(pos: usize) -> Self {
        Self::single(Range::cursor(pos))
    }

    /// Construct from a list of ranges, sorting and merging as `EditorSelection.create`
    /// does in CM6:
    ///   * stable sort by `from`
    ///   * merge overlapping ranges; merge touching ranges only when at least
    ///     one of them is a point range
    ///   * the merged range adopts the direction of the *latest in input order*
    ///     range absorbed
    ///   * `primary` is remapped to the index of the merged range that absorbed
    ///     the original primary
    pub fn new(ranges: Vec<Range>, primary: usize) -> Self {
        assert!(!ranges.is_empty());
        assert!(primary < ranges.len());

        let n = ranges.len();
        let mut indexed: Vec<(usize, Range)> = ranges.into_iter().enumerate().collect();
        indexed.sort_by_key(|(_, r)| r.from());

        struct Group {
            from: usize,
            to: usize,
            last_orig_idx: usize,
            last_forward: bool,
        }

        let mut groups: Vec<Group> = Vec::with_capacity(n);
        let mut input_to_group: Vec<usize> = vec![0; n];

        for (orig_idx, r) in indexed {
            let r_forward = r.head >= r.anchor;
            let merge = match groups.last() {
                Some(g) => {
                    r.from() < g.to
                        || (r.from() == g.to && (g.from == g.to || r.is_empty()))
                }
                None => false,
            };
            if merge {
                let g = groups.last_mut().unwrap();
                g.from = g.from.min(r.from());
                g.to = g.to.max(r.to());
                if orig_idx > g.last_orig_idx {
                    g.last_orig_idx = orig_idx;
                    g.last_forward = r_forward;
                }
                input_to_group[orig_idx] = groups.len() - 1;
            } else {
                groups.push(Group {
                    from: r.from(),
                    to: r.to(),
                    last_orig_idx: orig_idx,
                    last_forward: r_forward,
                });
                input_to_group[orig_idx] = groups.len() - 1;
            }
        }

        let new_primary = input_to_group[primary];
        let new_ranges = groups
            .into_iter()
            .map(|g| {
                if g.last_forward {
                    Range { anchor: g.from, head: g.to }
                } else {
                    Range { anchor: g.to, head: g.from }
                }
            })
            .collect();
        Self { ranges: new_ranges, primary: new_primary }
    }

    pub fn primary_range(&self) -> Range {
        self.ranges[self.primary]
    }

    /// Same as `primary_range`, mirrors CM6's `selection.main` accessor.
    pub fn main(&self) -> Range {
        self.ranges[self.primary]
    }

    pub fn map(&self, doc: &Rope) -> Self {
        let len = doc.len_chars();
        let ranges = self
            .ranges
            .iter()
            .map(|r| Range::new(r.anchor.min(len), r.head.min(len)))
            .collect();
        Self { ranges, primary: self.primary }
    }
}
