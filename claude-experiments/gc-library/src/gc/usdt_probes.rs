//! USDT (User Statically-Defined Tracing) probes for DTrace integration
//!
//! These probes allow fine-grained tracing of GC and thread states using DTrace.
//! Enable with the `dtrace` feature.
//!
//! Usage with DTrace:
//! ```bash
//! sudo dtrace -s scripts/thread_states_usdt.d -c './target/release/main program.bg'
//! ```

#![allow(dead_code)]

// When dtrace feature is enabled, use real USDT probes
#[cfg(feature = "dtrace")]
mod inner {
    /// Get the number of threads in the current process using Mach APIs
    #[cfg(target_os = "macos")]
    pub fn get_thread_count() -> u32 {
        use std::ptr;

        unsafe extern "C" {
            fn mach_task_self() -> u32;
            fn task_threads(target_task: u32, act_list: *mut *mut u32, act_list_cnt: *mut u32) -> i32;
            fn vm_deallocate(target_task: u32, address: usize, size: usize) -> i32;
        }

        unsafe {
            let mut threads: *mut u32 = ptr::null_mut();
            let mut count: u32 = 0;

            let result = task_threads(mach_task_self(), &mut threads, &mut count);

            if result == 0 {
                if !threads.is_null() {
                    vm_deallocate(
                        mach_task_self(),
                        threads as usize,
                        (count as usize) * std::mem::size_of::<u32>(),
                    );
                }
                count
            } else {
                0
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn get_thread_count() -> u32 {
        0
    }

    #[usdt::provider(provider = "gc_lib")]
    mod probes {
        fn gc__start(gc_number: u64, thread_count: u32) {}
        fn gc__end(gc_number: u64, objects_copied: u64, thread_count: u32) {}
        fn gc__minor__start(gc_number: u64, thread_count: u32) {}
        fn gc__minor__end(gc_number: u64, thread_count: u32) {}
        fn gc__full__start(gc_number: u64, thread_count: u32) {}
        fn gc__full__end(gc_number: u64, thread_count: u32) {}
        fn thread__pause__enter(thread_id: u64, thread_count: u32) {}
        fn thread__pause__exit(thread_id: u64, pause_duration_ns: u64, thread_count: u32) {}
        fn thread__state(thread_id: u64, state: u64, thread_count: u32) {}
        fn thread__start(thread_id: u64, thread_count: u32) {}
        fn thread__exit(thread_id: u64, thread_count: u32) {}
        fn thread__register(thread_id: u64, registered_count: u64, thread_count: u32) {}
        fn thread__unregister(thread_id: u64, registered_count: u64, thread_count: u32) {}
        fn thread__spawn(thread_count: u32) {}
        fn stw__begin(requesting_thread: u64, thread_count: u32) {}
        fn stw__all__paused(num_paused: u64, total_registered: u64, thread_count: u32) {}
        fn stw__end(requesting_thread: u64, thread_count: u32) {}
    }

    #[inline]
    pub fn thread_id_u64() -> u64 {
        let id = std::thread::current().id();
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        id.hash(&mut hasher);
        hasher.finish()
    }

    #[inline]
    pub fn fire_gc_start(gc_number: usize) {
        probes::gc__start!(|| (gc_number as u64, get_thread_count()));
    }

    #[inline]
    pub fn fire_gc_end(gc_number: usize, objects_copied: usize) {
        probes::gc__end!(|| (gc_number as u64, objects_copied as u64, get_thread_count()));
    }

    #[inline]
    pub fn fire_gc_minor_start(gc_number: usize) {
        probes::gc__minor__start!(|| (gc_number as u64, get_thread_count()));
    }

    #[inline]
    pub fn fire_gc_minor_end(gc_number: usize) {
        probes::gc__minor__end!(|| (gc_number as u64, get_thread_count()));
    }

    #[inline]
    pub fn fire_gc_full_start(gc_number: usize) {
        probes::gc__full__start!(|| (gc_number as u64, get_thread_count()));
    }

    #[inline]
    pub fn fire_gc_full_end(gc_number: usize) {
        probes::gc__full__end!(|| (gc_number as u64, get_thread_count()));
    }

    #[inline]
    pub fn fire_thread_pause_enter() {
        probes::thread__pause__enter!(|| (thread_id_u64(), get_thread_count()));
    }

    #[inline]
    pub fn fire_thread_pause_exit(pause_duration_ns: u64) {
        probes::thread__pause__exit!(|| (thread_id_u64(), pause_duration_ns, get_thread_count()));
    }

    #[inline]
    pub fn fire_thread_state(state: ThreadStateCode) {
        probes::thread__state!(|| (thread_id_u64(), state as u64, get_thread_count()));
    }

    #[inline]
    pub fn fire_thread_start() {
        probes::thread__start!(|| (thread_id_u64(), get_thread_count()));
    }

    #[inline]
    pub fn fire_thread_exit() {
        probes::thread__exit!(|| (thread_id_u64(), get_thread_count()));
    }

    #[inline]
    pub fn fire_thread_spawn() {
        probes::thread__spawn!(|| get_thread_count());
    }

    #[inline]
    pub fn fire_thread_register(registered_count: usize) {
        probes::thread__register!(|| (thread_id_u64(), registered_count as u64, get_thread_count()));
    }

    #[inline]
    pub fn fire_thread_unregister(registered_count: usize) {
        probes::thread__unregister!(|| (thread_id_u64(), registered_count as u64, get_thread_count()));
    }

    #[inline]
    pub fn fire_stw_begin() {
        probes::stw__begin!(|| (thread_id_u64(), get_thread_count()));
    }

    #[inline]
    pub fn fire_stw_all_paused(num_paused: usize, total_registered: usize) {
        probes::stw__all__paused!(|| (
            num_paused as u64,
            total_registered as u64,
            get_thread_count()
        ));
    }

    #[inline]
    pub fn fire_stw_end() {
        probes::stw__end!(|| (thread_id_u64(), get_thread_count()));
    }

    pub fn register() -> Result<(), usdt::Error> {
        usdt::register_probes()
    }

    /// Thread state codes for USDT probes
    #[repr(u64)]
    #[derive(Debug, Clone, Copy)]
    pub enum ThreadStateCode {
        Running = 0,
        PausedForGc = 1,
        WaitingOnLock = 2,
        InCCall = 3,
        Starting = 4,
        Exiting = 5,
    }
}

// When dtrace feature is disabled, provide no-op stubs
#[cfg(not(feature = "dtrace"))]
mod inner {
    #[inline(always)]
    pub fn fire_gc_start(_gc_number: usize) {}
    #[inline(always)]
    pub fn fire_gc_end(_gc_number: usize, _objects_copied: usize) {}
    #[inline(always)]
    pub fn fire_gc_minor_start(_gc_number: usize) {}
    #[inline(always)]
    pub fn fire_gc_minor_end(_gc_number: usize) {}
    #[inline(always)]
    pub fn fire_gc_full_start(_gc_number: usize) {}
    #[inline(always)]
    pub fn fire_gc_full_end(_gc_number: usize) {}
    #[inline(always)]
    pub fn fire_thread_pause_enter() {}
    #[inline(always)]
    pub fn fire_thread_pause_exit(_pause_duration_ns: u64) {}
    #[inline(always)]
    pub fn fire_thread_state(_state: ThreadStateCode) {}
    #[inline(always)]
    pub fn fire_thread_start() {}
    #[inline(always)]
    pub fn fire_thread_exit() {}
    #[inline(always)]
    pub fn fire_thread_spawn() {}
    #[inline(always)]
    pub fn fire_thread_register(_registered_count: usize) {}
    #[inline(always)]
    pub fn fire_thread_unregister(_registered_count: usize) {}
    #[inline(always)]
    pub fn fire_stw_begin() {}
    #[inline(always)]
    pub fn fire_stw_all_paused(_num_paused: usize, _total_registered: usize) {}
    #[inline(always)]
    pub fn fire_stw_end() {}

    pub fn register() -> Result<(), ()> {
        Ok(())
    }

    /// Thread state codes for USDT probes
    #[repr(u64)]
    #[derive(Debug, Clone, Copy)]
    pub enum ThreadStateCode {
        Running = 0,
        PausedForGc = 1,
        WaitingOnLock = 2,
        InCCall = 3,
        Starting = 4,
        Exiting = 5,
    }
}

pub use inner::*;
