/// Card table for tracking old→young pointer writes.
///
/// Each "card" covers a fixed-size region (512 bytes) of tenured heap space.
/// When a mutator stores a nursery pointer into a tenured object, the
/// corresponding card is marked dirty. During minor GC, only dirty cards
/// need scanning for nursery references.

const CARD_SHIFT: usize = 9; // 512 bytes per card
const CARD_CLEAN: u8 = 0;
const CARD_DIRTY: u8 = 1;

pub struct CardTable {
    cards: Vec<u8>,
    base_addr: usize,
}

// Safety: CardTable is only mutated during STW or by the owning mutator
// thread (plain byte store for dirtying is idempotent and race-safe).
unsafe impl Send for CardTable {}
unsafe impl Sync for CardTable {}

impl CardTable {
    /// Create a new card table covering `region_size` bytes starting at `base`.
    pub fn new(base: *const u8, region_size: usize) -> Self {
        let num_cards = (region_size + (1 << CARD_SHIFT) - 1) >> CARD_SHIFT;
        CardTable {
            cards: vec![CARD_CLEAN; num_cards],
            base_addr: base as usize,
        }
    }

    /// Mark the card containing `addr` as dirty.
    ///
    /// This is a plain byte store — redundant dirtying is harmless, and
    /// races between mutators marking the same card are benign (both
    /// store CARD_DIRTY = 1).
    #[inline(always)]
    pub fn mark_dirty(&self, addr: *const u8) {
        let offset = addr as usize - self.base_addr;
        let card_idx = offset >> CARD_SHIFT;
        if card_idx < self.cards.len() {
            // Safety: racing byte stores of the same value are benign.
            // We use a raw pointer write to avoid needing &mut self.
            unsafe {
                let card_ptr = self.cards.as_ptr().add(card_idx) as *mut u8;
                *card_ptr = CARD_DIRTY;
            }
        }
    }

    /// Check if a specific card is dirty.
    pub fn is_dirty(&self, card_idx: usize) -> bool {
        card_idx < self.cards.len() && self.cards[card_idx] == CARD_DIRTY
    }

    /// Clear all cards to clean state.
    ///
    /// Takes `&self` rather than `&mut self` because this is called during
    /// STW when we only have shared references. Safety: no mutators are
    /// running, so there are no concurrent accesses.
    pub fn clear_all(&self) {
        unsafe {
            core::ptr::write_bytes(self.cards.as_ptr() as *mut u8, CARD_CLEAN, self.cards.len());
        }
    }

    /// Iterate over dirty cards, yielding (card_index, card_start_address).
    pub fn iter_dirty(&self) -> impl Iterator<Item = (usize, *const u8)> + '_ {
        self.cards.iter().enumerate().filter_map(move |(idx, &val)| {
            if val == CARD_DIRTY {
                let addr = self.base_addr + (idx << CARD_SHIFT);
                Some((idx, addr as *const u8))
            } else {
                None
            }
        })
    }

    /// Number of cards in the table.
    pub fn card_count(&self) -> usize {
        self.cards.len()
    }

    /// Size in bytes of the region covered by one card.
    pub fn card_size(&self) -> usize {
        1 << CARD_SHIFT
    }

    /// Base address this card table covers.
    pub fn base_addr(&self) -> usize {
        self.base_addr
    }
}
