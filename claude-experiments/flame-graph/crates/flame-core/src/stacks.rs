use crate::strings::StringId;
use ahash::AHashMap;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct FrameId(pub u32);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct StackId(pub u32);

#[derive(Clone, Debug)]
pub struct Frame {
    pub name: StringId,
    pub file: StringId,
    pub line: u32,
    pub addr: u64,
}

#[derive(Default, Clone)]
pub struct StackTable {
    pub frames: Vec<Frame>,
    pub frame_dedup: AHashMap<(StringId, StringId, u32, u64), FrameId>,
    pub nodes: Vec<StackNode>,
    pub node_dedup: AHashMap<(FrameId, Option<StackId>), StackId>,
}

#[derive(Copy, Clone, Debug)]
pub struct StackNode {
    pub frame: FrameId,
    pub parent: Option<StackId>,
    pub depth: u16,
}

impl StackTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern_frame(
        &mut self,
        name: StringId,
        file: StringId,
        line: u32,
        addr: u64,
    ) -> FrameId {
        let key = (name, file, line, addr);
        if let Some(&id) = self.frame_dedup.get(&key) {
            return id;
        }
        let id = FrameId(self.frames.len() as u32);
        self.frames.push(Frame { name, file, line, addr });
        self.frame_dedup.insert(key, id);
        id
    }

    pub fn intern_stack(&mut self, frame: FrameId, parent: Option<StackId>) -> StackId {
        let key = (frame, parent);
        if let Some(&id) = self.node_dedup.get(&key) {
            return id;
        }
        let depth = match parent {
            Some(p) => self.nodes[p.0 as usize].depth + 1,
            None => 0,
        };
        let id = StackId(self.nodes.len() as u32);
        self.nodes.push(StackNode { frame, parent, depth });
        self.node_dedup.insert(key, id);
        id
    }

    pub fn frame(&self, id: FrameId) -> &Frame {
        &self.frames[id.0 as usize]
    }

    pub fn node(&self, id: StackId) -> &StackNode {
        &self.nodes[id.0 as usize]
    }

    /// Walk a stack from leaf to root, calling `f` with each frame in leaf-first order.
    pub fn walk(&self, mut id: Option<StackId>, mut f: impl FnMut(FrameId, u16)) {
        while let Some(s) = id {
            let n = self.nodes[s.0 as usize];
            f(n.frame, n.depth);
            id = n.parent;
        }
    }

    /// Collect a stack root-first (depth 0 → leaf).
    pub fn frames_root_first(&self, id: StackId) -> Vec<FrameId> {
        let mut out = Vec::new();
        let mut cur = Some(id);
        while let Some(s) = cur {
            let n = self.nodes[s.0 as usize];
            out.push(n.frame);
            cur = n.parent;
        }
        out.reverse();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strings::StringInterner;

    #[test]
    fn stack_dedup_and_walk() {
        let mut s = StringInterner::new();
        let mut t = StackTable::new();
        let a = t.intern_frame(s.intern("a"), StringId::EMPTY, 0, 0);
        let b = t.intern_frame(s.intern("b"), StringId::EMPTY, 0, 0);
        let s_a = t.intern_stack(a, None);
        let s_ab = t.intern_stack(b, Some(s_a));
        let s_a2 = t.intern_stack(a, None);
        let s_ab2 = t.intern_stack(b, Some(s_a2));
        assert_eq!(s_a, s_a2);
        assert_eq!(s_ab, s_ab2);
        assert_eq!(t.node(s_ab).depth, 1);
        assert_eq!(t.frames_root_first(s_ab), vec![a, b]);
    }
}
