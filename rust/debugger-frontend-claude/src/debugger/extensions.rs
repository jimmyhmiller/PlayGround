use lldb::{SBFrame, SBValue, SBTarget, SBBreakpoint, SBProcess, SBInstructionList};
use lldb_sys::{SBValueGetValueAsUnsigned2, SBTargetBreakpointCreateByName, SBProcessReadMemory, SBTargetGetInstructionsWithFlavor};
use std::ffi::{c_void, CString};
use crate::Result;

/// Extensions to LLDB SBFrame for easier register access
pub trait FrameExtensions {
    fn get_register(&self, name: &str) -> Option<SBValue>;
}

impl FrameExtensions for SBFrame {
    fn get_register(&self, name: &str) -> Option<SBValue> {
        for register in self.registers().into_iter() {
            if matches!(register.name(), Some(n) if n == name) {
                return Some(register);
            }
            for child in register.children().into_iter() {
                if matches!(child.name(), Some(n) if n == name) {
                    return Some(child);
                }
            }
        }
        None
    }
}

/// Extensions to LLDB SBValue for easier value extraction
pub trait ValueExtensions {
    fn to_usize(&self) -> usize;
    fn to_u64(&self) -> u64;
}

impl ValueExtensions for SBValue {
    fn to_usize(&self) -> usize {
        unsafe { SBValueGetValueAsUnsigned2(self.raw, 0) as usize }
    }

    fn to_u64(&self) -> u64 {
        unsafe { SBValueGetValueAsUnsigned2(self.raw, 0) }
    }
}

/// Extensions to LLDB SBTarget for easier breakpoint creation
pub trait TargetExtensions {
    fn create_breakpoint_by_name(&self, name: &str, module_name: &str) -> Option<SBBreakpoint>;
}

impl TargetExtensions for SBTarget {
    fn create_breakpoint_by_name(&self, name: &str, module_name: &str) -> Option<SBBreakpoint> {
        unsafe {
            let name = CString::new(name).ok()?;
            let module_name = CString::new(module_name).ok()?;
            let pointer = SBTargetBreakpointCreateByName(self.raw, name.as_ptr(), module_name.as_ptr());
            if pointer.is_null() {
                None
            } else {
                Some(SBBreakpoint { raw: pointer })
            }
        }
    }
}

/// Extensions to LLDB SBProcess for memory operations
pub trait ProcessExtensions {
    fn read_memory(&self, address: u64, buffer: &mut [u8]) -> Result<usize>;
    fn get_instructions(&self, frame: &SBFrame, target: &SBTarget) -> Result<SBInstructionList>;
    fn read_memory_as_u64_array(&self, address: u64, size: usize) -> Result<Vec<u64>>;
}

impl ProcessExtensions for SBProcess {
    fn read_memory(&self, address: u64, buffer: &mut [u8]) -> Result<usize> {
        let ptr: *mut c_void = buffer.as_mut_ptr() as *mut c_void;
        let count = buffer.len();
        let error = lldb::SBError::default();
        let bytes_read = unsafe { 
            SBProcessReadMemory(self.raw, address, ptr, count, error.raw) 
        };
        
        if error.is_success() {
            Ok(bytes_read)
        } else {
            Err(anyhow::anyhow!("Failed to read memory: {}", error.to_string()))
        }
    }

    fn get_instructions(&self, frame: &SBFrame, target: &SBTarget) -> Result<SBInstructionList> {
        let mut buffer = [0u8; 1024];
        self.read_memory(frame.pc_address().load_address(target), &mut buffer)?;
        
        let base_addr = frame.pc_address();
        let ptr: *mut c_void = buffer.as_mut_ptr() as *mut c_void;
        let count = buffer.len();
        let flavor_string = CString::new("intel").unwrap();
        let flavor = flavor_string.as_ptr();
        
        unsafe {
            Ok(SBInstructionList {
                raw: SBTargetGetInstructionsWithFlavor(
                    target.raw,
                    base_addr.raw,
                    flavor,
                    ptr,
                    count,
                ),
            })
        }
    }

    fn read_memory_as_u64_array(&self, address: u64, size: usize) -> Result<Vec<u64>> {
        let mut buffer = vec![0u8; size];
        self.read_memory(address, &mut buffer)?;
        
        Ok(buffer
            .chunks(8)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .fold(0, |acc, (i, &byte)| acc | (byte as u64) << (8 * i))
            })
            .collect())
    }
}