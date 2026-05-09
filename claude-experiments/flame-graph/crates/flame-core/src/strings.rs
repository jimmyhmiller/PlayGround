use ahash::AHashMap;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct StringId(pub u32);

impl StringId {
    pub const EMPTY: StringId = StringId(0);
}

pub struct StringInterner {
    values: Vec<String>,
    by_value: AHashMap<String, StringId>,
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

impl StringInterner {
    pub fn new() -> Self {
        let mut s = Self {
            values: Vec::new(),
            by_value: AHashMap::new(),
        };
        let id = s.intern("");
        debug_assert_eq!(id, StringId::EMPTY);
        s
    }

    pub fn intern(&mut self, value: &str) -> StringId {
        if let Some(&id) = self.by_value.get(value) {
            return id;
        }
        let id = StringId(self.values.len() as u32);
        self.values.push(value.to_owned());
        self.by_value.insert(value.to_owned(), id);
        id
    }

    pub fn get(&self, id: StringId) -> &str {
        &self.values[id.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_roundtrip() {
        let mut s = StringInterner::new();
        let a = s.intern("hello");
        let b = s.intern("world");
        let a2 = s.intern("hello");
        assert_eq!(a, a2);
        assert_ne!(a, b);
        assert_eq!(s.get(a), "hello");
        assert_eq!(s.get(b), "world");
        assert_eq!(s.get(StringId::EMPTY), "");
    }
}
