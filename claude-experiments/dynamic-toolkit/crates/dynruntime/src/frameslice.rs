use dynexec::{FrameSliceError, FrameSliceSnapshot, FrameSliceStore};
use dynobj::RootSource;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameSliceHandle(usize);

impl FrameSliceHandle {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Default)]
pub struct OwnedFrameSliceStore {
    slices: Vec<FrameSliceSnapshot>,
}

impl OwnedFrameSliceStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn root_source<'a>(
        &'a self,
        handle: &'a FrameSliceHandle,
    ) -> Result<FrameSliceRootSource<'a>, FrameSliceError> {
        Ok(FrameSliceRootSource {
            slice: self.slice(handle)?,
        })
    }
}

impl FrameSliceStore for OwnedFrameSliceStore {
    type Handle = FrameSliceHandle;

    fn insert_slice(&mut self, slice: FrameSliceSnapshot) -> Result<Self::Handle, FrameSliceError> {
        slice.validate()?;
        let handle = FrameSliceHandle(self.slices.len());
        self.slices.push(slice);
        Ok(handle)
    }

    fn clone_slice(&mut self, handle: &Self::Handle) -> Result<Self::Handle, FrameSliceError> {
        let mut cloned = self.slice(handle)?.clone();
        cloned.consumed = false;
        self.insert_slice(cloned)
    }

    fn slice(&self, handle: &Self::Handle) -> Result<&FrameSliceSnapshot, FrameSliceError> {
        self.slices.get(handle.0).ok_or(FrameSliceError::MissingSlice)
    }

    fn slice_mut(
        &mut self,
        handle: &Self::Handle,
    ) -> Result<&mut FrameSliceSnapshot, FrameSliceError> {
        self.slices
            .get_mut(handle.0)
            .ok_or(FrameSliceError::MissingSlice)
    }

    fn encode_handle(handle: &Self::Handle) -> u64 {
        handle.0 as u64
    }

    fn decode_handle(bits: u64) -> Result<Self::Handle, FrameSliceError> {
        usize::try_from(bits)
            .map(FrameSliceHandle)
            .map_err(|_| FrameSliceError::MissingSlice)
    }
}

pub struct FrameSliceRootSource<'a> {
    slice: &'a FrameSliceSnapshot,
}

impl RootSource for FrameSliceRootSource<'_> {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        for frame in &self.slice.frames {
            for &root_idx in &frame.root_value_indices {
                let slot = (&frame.values[root_idx] as *const u64).cast_mut();
                visitor(slot);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynexec::{CapturedCallerResume, CapturedFrame, FrameResumePoint, FrameSliceMode};

    fn sample_slice() -> FrameSliceSnapshot {
        FrameSliceSnapshot {
            prompt_id: 7,
            mode: FrameSliceMode::OneShot,
            frames: vec![
                CapturedFrame {
                    resume: FrameResumePoint {
                        func_idx: 1,
                        block_idx: 2,
                        inst_idx: 3,
                    },
                    values: vec![10, 20, 30],
                    root_value_indices: vec![1],
                    resume_arg_value_indices: vec![0],
                    active_prompts: vec![7],
                    caller_resume: CapturedCallerResume::TopLevel,
                },
                CapturedFrame {
                    resume: FrameResumePoint {
                        func_idx: 4,
                        block_idx: 5,
                        inst_idx: 6,
                    },
                    values: vec![40, 50, 60, 70],
                    root_value_indices: vec![0, 3],
                    resume_arg_value_indices: vec![],
                    active_prompts: vec![],
                    caller_resume: CapturedCallerResume::FromCall { return_dest: Some(2) },
                },
            ],
            consumed: false,
        }
    }

    #[test]
    fn owned_store_clones_and_marks_consumed() {
        let mut store = OwnedFrameSliceStore::new();
        let handle = store.insert_slice(sample_slice()).unwrap();
        store.mark_consumed(&handle).unwrap();
        assert!(store.slice(&handle).unwrap().consumed);

        let cloned = store.clone_slice(&handle).unwrap();
        assert!(!store.slice(&cloned).unwrap().consumed);
        assert_eq!(store.slice(&cloned).unwrap().prompt_id, 7);
    }

    #[test]
    fn frame_slice_root_source_scans_only_marked_root_values() {
        let mut store = OwnedFrameSliceStore::new();
        let handle = store.insert_slice(sample_slice()).unwrap();
        let source = store.root_source(&handle).unwrap();

        let mut seen = Vec::new();
        source.scan_roots(&mut |slot| unsafe {
            seen.push(*slot);
        });

        assert_eq!(seen, vec![20, 40, 70]);
    }

    #[test]
    fn invalid_root_index_is_rejected() {
        let mut store = OwnedFrameSliceStore::new();
        let mut bad = sample_slice();
        bad.frames[0].root_value_indices.push(9);
        let err = store.insert_slice(bad).unwrap_err();
        assert_eq!(
            err,
            FrameSliceError::InvalidRootIndex {
                frame_idx: 0,
                root_idx: 9,
            }
        );
    }
}
