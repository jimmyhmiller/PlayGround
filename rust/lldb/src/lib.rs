use cpp::{cpp, cpp_class};
use std::{os::raw::{c_char, c_int, c_uint}, ffi::CStr};

cpp! {{
    #include <lldb/API/SBDebugger.h>
    using namespace lldb;
}}




//     pub fn terminate() {
//         cpp!(unsafe [] {
//             SBDebugger::Terminate();
//         })
//     }
//     pub fn create(source_init_files: bool) -> SBDebugger {
//         cpp!(unsafe [source_init_files as "bool"] -> SBDebugger as "SBDebugger" {
//             return SBDebugger::Create(source_init_files);
//         })
//     }
// }

cpp_class!(pub unsafe struct SBDebugger as "SBDebugger");

pub(crate) unsafe fn get_str<'a>(ptr: *const c_char) -> &'a str {
    if ptr.is_null() {
        ""
    } else {
        let cstr = CStr::from_ptr(ptr);
        match cstr.to_str() {
            Ok(val) => val,
            Err(err) => std::str::from_utf8(&cstr.to_bytes()[..err.valid_up_to()]).unwrap(),
        }
    }
}


impl SBDebugger {

    pub fn create(source_init_files: bool) -> SBDebugger {
        cpp!(unsafe [source_init_files as "bool"] -> SBDebugger as "SBDebugger" {
            return SBDebugger::Create(source_init_files);
        })
    }

    pub fn instance_name(&self) -> &str {
        let ptr = cpp!(unsafe [self as "SBDebugger*"] ->  *const c_char as "const char*" {
            return self->GetInstanceName();
        });
        unsafe { get_str(ptr) }
    }
}
