use std::cell::RefCell;
use std::marker::PhantomData;

use dynalloc::{Heap, PtrPolicy};
use dynlower::{SafepointHandlerPayloadKind, SafepointRecord};
use dynobj::RootSource;

pub trait JitRootTransportRuntime {
    fn payload_kind(&self) -> SafepointHandlerPayloadKind;

    unsafe fn scan_roots(
        &self,
        frame_ptr: *mut u8,
        payload: usize,
        safepoints: &[SafepointRecord],
        visitor: &mut dyn FnMut(*mut u64),
    );
}

pub struct FrameScanJitTransport;
pub struct StackMapJitTransport;
pub struct ShadowStackJitTransport;

impl JitRootTransportRuntime for FrameScanJitTransport {
    fn payload_kind(&self) -> SafepointHandlerPayloadKind {
        SafepointHandlerPayloadKind::FrameSize
    }

    unsafe fn scan_roots(
        &self,
        frame_ptr: *mut u8,
        payload: usize,
        _safepoints: &[SafepointRecord],
        visitor: &mut dyn FnMut(*mut u64),
    ) {
        let root_source = FrameWordRootSource {
            frame_ptr,
            frame_size: payload,
        };
        root_source.scan_roots(visitor);
    }
}

impl JitRootTransportRuntime for StackMapJitTransport {
    fn payload_kind(&self) -> SafepointHandlerPayloadKind {
        SafepointHandlerPayloadKind::SafepointIndex
    }

    unsafe fn scan_roots(
        &self,
        frame_ptr: *mut u8,
        payload: usize,
        safepoints: &[SafepointRecord],
        visitor: &mut dyn FnMut(*mut u64),
    ) {
        let record = safepoints
            .get(payload)
            .unwrap_or_else(|| panic!("missing stack-map safepoint record {payload}"));
        let root_source = SlotOffsetRootSource {
            frame_ptr,
            slot_offsets: &record.root_slots,
        };
        root_source.scan_roots(visitor);
    }
}

impl JitRootTransportRuntime for ShadowStackJitTransport {
    fn payload_kind(&self) -> SafepointHandlerPayloadKind {
        SafepointHandlerPayloadKind::SafepointIndex
    }

    unsafe fn scan_roots(
        &self,
        frame_ptr: *mut u8,
        payload: usize,
        safepoints: &[SafepointRecord],
        visitor: &mut dyn FnMut(*mut u64),
    ) {
        let record = safepoints
            .get(payload)
            .unwrap_or_else(|| panic!("missing shadow-stack safepoint record {payload}"));
        let root_source = SlotOffsetRootSource {
            frame_ptr,
            slot_offsets: &record.root_slots,
        };
        root_source.scan_roots(visitor);
    }
}

struct FrameWordRootSource {
    frame_ptr: *mut u8,
    frame_size: usize,
}

impl RootSource for FrameWordRootSource {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        let word_count = self.frame_size / 8;
        for idx in 0..word_count {
            let slot = unsafe { self.frame_ptr.add(idx * 8).cast::<u64>() };
            visitor(slot);
        }
    }
}

struct SlotOffsetRootSource<'a> {
    frame_ptr: *mut u8,
    slot_offsets: &'a [i32],
}

impl RootSource for SlotOffsetRootSource<'_> {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        for &offset in self.slot_offsets {
            let slot = unsafe { self.frame_ptr.add(offset as usize).cast::<u64>() };
            visitor(slot);
        }
    }
}

struct RuntimeRootSource<'a, T: JitRootTransportRuntime> {
    transport: &'a T,
    frame_ptr: *mut u8,
    payload: usize,
    safepoints: &'a [SafepointRecord],
}

impl<T: JitRootTransportRuntime> RootSource for RuntimeRootSource<'_, T> {
    fn scan_roots(&self, visitor: &mut dyn FnMut(*mut u64)) {
        unsafe {
            self.transport
                .scan_roots(self.frame_ptr, self.payload, self.safepoints, visitor);
        }
    }
}

pub struct JitSafepointSession<'a, P: PtrPolicy, T: JitRootTransportRuntime> {
    heap: &'a Heap,
    transport: T,
    safepoints: &'a [SafepointRecord],
    _policy: PhantomData<P>,
}

#[derive(Clone, Copy)]
struct ActiveJitSafepointSession {
    ptr: *const (),
    handle: unsafe fn(*const (), *mut u8, usize),
}

impl<'a, P: PtrPolicy, T: JitRootTransportRuntime> JitSafepointSession<'a, P, T> {
    pub fn new(heap: &'a Heap, transport: T, safepoints: &'a [SafepointRecord]) -> Self {
        Self {
            heap,
            transport,
            safepoints,
            _policy: PhantomData,
        }
    }

    pub fn payload_kind(&self) -> SafepointHandlerPayloadKind {
        self.transport.payload_kind()
    }

    pub fn with_installed<R>(&self, f: impl FnOnce() -> R) -> R {
        ACTIVE_JIT_SAFEPOINT_SESSION.with(|cell| {
            let previous = {
                let mut slot = cell.borrow_mut();
                slot.replace(ActiveJitSafepointSession {
                    ptr: self as *const Self as *const (),
                    handle: Self::handle_erased,
                })
            };
            let result = f();
            let mut slot = cell.borrow_mut();
            *slot = previous;
            result
        })
    }

    unsafe fn handle_erased(ptr: *const (), frame_ptr: *mut u8, payload: usize) {
        let session = unsafe { &*(ptr as *const Self) };
        unsafe {
            session.handle(frame_ptr, payload);
        }
    }

    unsafe fn handle(&self, frame_ptr: *mut u8, payload: usize) {
        let root_source = RuntimeRootSource {
            transport: &self.transport,
            frame_ptr,
            payload,
            safepoints: self.safepoints,
        };
        unsafe {
            self.heap.collect::<P>(&[&root_source]);
        }
    }
}

thread_local! {
    static ACTIVE_JIT_SAFEPOINT_SESSION: RefCell<Option<ActiveJitSafepointSession>> =
        RefCell::new(None);
}

pub extern "C" fn active_jit_safepoint_handler(frame_ptr: *mut u8, payload: usize) {
    ACTIVE_JIT_SAFEPOINT_SESSION.with(|cell| {
        let session = (*cell.borrow()).expect("no active JIT safepoint session installed");
        unsafe {
            (session.handle)(session.ptr, frame_ptr, payload);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    use dynalloc::Heap;
    use dynobj::Compact;

    #[test]
    fn frame_scan_transport_scans_full_frame_words() {
        let mut frame = [0u64; 4];
        frame[1] = 11;
        frame[2] = 22;

        let mut seen = Vec::new();
        unsafe {
            FrameScanJitTransport.scan_roots(
                frame.as_mut_ptr().cast::<u8>(),
                frame.len() * 8,
                &[],
                &mut |slot| seen.push(*slot),
            );
        }
        assert_eq!(seen, vec![0, 11, 22, 0]);
    }

    #[test]
    fn metadata_transport_scans_only_recorded_slots() {
        let mut frame = [0u64; 6];
        frame[2] = 33;
        frame[4] = 55;
        let safepoints = [SafepointRecord {
            code_offset: 0,
            root_slots: vec![16, 32],
        }];

        let mut seen = Vec::new();
        unsafe {
            StackMapJitTransport.scan_roots(
                frame.as_mut_ptr().cast::<u8>(),
                0,
                &safepoints,
                &mut |slot| seen.push(*slot),
            );
        }
        assert_eq!(seen, vec![33, 55]);
    }

    struct CountingTransport<'a> {
        last_payload: &'a Cell<usize>,
    }

    impl JitRootTransportRuntime for CountingTransport<'_> {
        fn payload_kind(&self) -> SafepointHandlerPayloadKind {
            SafepointHandlerPayloadKind::SafepointIndex
        }

        unsafe fn scan_roots(
            &self,
            _frame_ptr: *mut u8,
            payload: usize,
            _safepoints: &[SafepointRecord],
            _visitor: &mut dyn FnMut(*mut u64),
        ) {
            self.last_payload.set(payload);
        }
    }

    #[test]
    fn installed_session_routes_handler_calls_through_transport() {
        let heap = Heap::new::<Compact>(256);
        let last_payload = Cell::new(usize::MAX);
        let transport = CountingTransport {
            last_payload: &last_payload,
        };
        let session =
            JitSafepointSession::<crate::ptr_policy::LowBitPtrPolicy<3>, _>::new(&heap, transport, &[]);
        let mut frame = [0u64; 2];

        session.with_installed(|| {
            active_jit_safepoint_handler(frame.as_mut_ptr().cast::<u8>(), 7);
        });

        assert_eq!(last_payload.get(), 7);
    }
}
