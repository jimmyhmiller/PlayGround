use mmap_rs::{Mmap, MmapOptions, Reserved, ReservedMut};

pub struct CodeAllocator {
    unused_pages: Reserved,
    current_page: Option<ReservedMut>,
    current_offset: usize,
    pending_pages: Option<Reserved>,
    #[allow(dead_code)]
    used_pages: Option<Mmap>,
}

impl Default for CodeAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeAllocator {
    pub fn new() -> Self {
        let unused_pages = MmapOptions::new(MmapOptions::page_size() * 100000)
            .unwrap()
            .reserve()
            .unwrap();
        Self {
            unused_pages,
            used_pages: None,
            pending_pages: None,
            current_page: None,
            current_offset: 0,
        }
    }

    pub fn take_page(&mut self) -> ReservedMut {
        let page = self
            .unused_pages
            .split_to(MmapOptions::page_size())
            .unwrap();
        page.make_mut().unwrap()
    }

    pub fn mark_page_as_pending(&mut self) {
        if self.current_offset == 0 {
            return;
        }
        if let Some(page) = self.current_page.take() {
            let page = page.make_read_only().unwrap();
            if let Some(mut pending) = self.pending_pages.take() {
                pending.merge(page).unwrap();
                self.pending_pages = Some(pending);
            } else {
                self.pending_pages = Some(page);
            }
        }

        self.current_page = None;
        self.current_offset = 0;
    }

    pub fn write_bytes(&mut self, bytes: &[u8]) -> *const u8 {
        if self.current_page.is_none() {
            self.current_page = Some(self.take_page());
        }
        unsafe {
            let bytes_remaining = MmapOptions::page_size() - self.current_offset;
            if bytes.len() < bytes_remaining {
                let start = self
                    .current_page
                    .as_mut()
                    .unwrap()
                    .as_mut_ptr()
                    .add(self.current_offset);
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), start, bytes.len());
                self.current_offset += bytes.len();

                // I think this should align 16 bytes
                // but I might be wrong and I need to fix something going wrong
                // I know that if I had any functions to my lldb libs
                // I get some weird error. I thought it might be alignment
                // but it could be that compilation is crossing a page
                // boundary and this code isn't dealing with that
                // correctly
                if self.current_offset % 2 != 0 {
                    self.current_offset += 1;
                }

                start
            } else {
                let (first, second) = bytes.split_at(bytes_remaining);
                let start = self
                    .current_page
                    .as_mut()
                    .unwrap()
                    .as_mut_ptr()
                    .add(self.current_offset);

                std::ptr::copy_nonoverlapping(first.as_ptr(), start, first.len());
                self.current_offset += first.len();
                self.mark_page_as_pending();
                self.write_bytes(second);
                start
            }
        }
    }

    pub fn make_executable(&mut self) {
        self.mark_page_as_pending();

        let pending = self.pending_pages.take().unwrap();
        let pending = pending.make_exec().unwrap();
        let pending: Mmap = pending.try_into().unwrap();
        if let Some(mut used) = self.used_pages.take() {
            used.merge(pending).unwrap();
            self.used_pages = Some(used);
        } else {
            self.used_pages = Some(pending);
        }
        // let used = self.used_pages.take();
        // let used = used.unwrap().make_exec().unwrap();
        // self.used_pages = Some(used);

        assert!(self.current_page.is_none());
        assert!(self.pending_pages.is_none());
    }
}
