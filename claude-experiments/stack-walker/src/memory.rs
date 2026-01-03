use crate::error::{Result, StackWalkError};

/// Trait for reading memory from a target context
///
/// Implementations can read from the current process, external processes,
/// or captured memory dumps.
pub trait MemoryReader {
    /// Read a u64 value from the given address
    fn read_u64(&mut self, address: u64) -> Result<u64>;

    /// Read raw bytes from the given address into the buffer
    fn read_bytes(&mut self, address: u64, buffer: &mut [u8]) -> Result<()>;

    /// Check if an address range is likely readable (optional optimization)
    ///
    /// Returns true if the address might be readable. A false positive is safe
    /// (will fail on actual read), but a false negative could skip valid memory.
    fn is_readable(&self, _address: u64, _len: usize) -> bool {
        true // Conservative default: assume readable
    }
}

/// Memory reader using a closure for reads
///
/// This is the most flexible option, matching the pattern used by framehop.
///
/// # Example
///
/// ```ignore
/// let mut reader = ClosureMemoryReader::new(|addr| {
///     // Read from target process
///     read_process_memory(addr)
/// });
/// ```
pub struct ClosureMemoryReader<F>
where
    F: FnMut(u64) -> std::result::Result<u64, ()>,
{
    read_fn: F,
}

impl<F> ClosureMemoryReader<F>
where
    F: FnMut(u64) -> std::result::Result<u64, ()>,
{
    /// Create a new closure-based memory reader
    pub fn new(read_fn: F) -> Self {
        Self { read_fn }
    }
}

impl<F> MemoryReader for ClosureMemoryReader<F>
where
    F: FnMut(u64) -> std::result::Result<u64, ()>,
{
    fn read_u64(&mut self, address: u64) -> Result<u64> {
        (self.read_fn)(address).map_err(|_| StackWalkError::MemoryReadFailed { address })
    }

    fn read_bytes(&mut self, address: u64, buffer: &mut [u8]) -> Result<()> {
        // Read u64 values and copy into buffer
        let mut offset = 0usize;
        while offset < buffer.len() {
            let val = self.read_u64(address + offset as u64)?;
            let bytes = val.to_le_bytes();
            let remaining = buffer.len() - offset;
            let copy_len = remaining.min(8);
            buffer[offset..offset + copy_len].copy_from_slice(&bytes[..copy_len]);
            offset += 8;
        }
        Ok(())
    }
}

/// Memory reader backed by a byte slice (for testing and offline analysis)
///
/// # Example
///
/// ```
/// use stack_walker::SliceMemoryReader;
/// use stack_walker::MemoryReader;
///
/// let stack_data = vec![0u8; 1024];
/// let base_address = 0x7fff0000u64;
/// let mut reader = SliceMemoryReader::new(base_address, &stack_data);
///
/// // Read from simulated stack
/// let value = reader.read_u64(base_address);
/// ```
pub struct SliceMemoryReader<'a> {
    base_address: u64,
    data: &'a [u8],
}

impl<'a> SliceMemoryReader<'a> {
    /// Create a new slice-based memory reader
    pub fn new(base_address: u64, data: &'a [u8]) -> Self {
        Self { base_address, data }
    }

    /// Get the base address
    #[inline]
    pub fn base_address(&self) -> u64 {
        self.base_address
    }

    /// Get the length of the backing data
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the reader has no data
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Check if an address falls within the slice
    #[inline]
    fn contains(&self, address: u64, len: usize) -> bool {
        if let Some(offset) = address.checked_sub(self.base_address) {
            let offset = offset as usize;
            offset.checked_add(len).map_or(false, |end| end <= self.data.len())
        } else {
            false
        }
    }
}

impl<'a> MemoryReader for SliceMemoryReader<'a> {
    fn read_u64(&mut self, address: u64) -> Result<u64> {
        let offset = address
            .checked_sub(self.base_address)
            .ok_or(StackWalkError::MemoryReadFailed { address })?;

        let offset = offset as usize;
        if offset + 8 > self.data.len() {
            return Err(StackWalkError::MemoryReadFailed { address });
        }

        let bytes: [u8; 8] = self.data[offset..offset + 8]
            .try_into()
            .map_err(|_| StackWalkError::MemoryReadFailed { address })?;

        Ok(u64::from_le_bytes(bytes))
    }

    fn read_bytes(&mut self, address: u64, buffer: &mut [u8]) -> Result<()> {
        let offset = address
            .checked_sub(self.base_address)
            .ok_or(StackWalkError::MemoryReadFailed { address })?;

        let offset = offset as usize;
        if offset + buffer.len() > self.data.len() {
            return Err(StackWalkError::MemoryReadFailed { address });
        }

        buffer.copy_from_slice(&self.data[offset..offset + buffer.len()]);
        Ok(())
    }

    fn is_readable(&self, address: u64, len: usize) -> bool {
        self.contains(address, len)
    }
}

/// Direct memory reader for the current process
///
/// **WARNING**: This uses unsafe pointer dereferences. Only use for reading
/// the current process's own memory.
///
/// # Safety
///
/// The caller must ensure that:
/// - All addresses passed to read methods are valid and readable
/// - The memory is not being concurrently modified
/// - This is only used for the current process
#[derive(Debug, Clone, Copy, Default)]
pub struct UnsafeDirectReader;

impl UnsafeDirectReader {
    /// Create a new direct memory reader
    pub fn new() -> Self {
        Self
    }
}

impl MemoryReader for UnsafeDirectReader {
    #[inline]
    fn read_u64(&mut self, address: u64) -> Result<u64> {
        // Safety: The caller guarantees the address is valid for the current process
        unsafe {
            let ptr = address as *const u64;
            if ptr.is_null() {
                return Err(StackWalkError::MemoryReadFailed { address });
            }
            Ok(ptr.read_unaligned())
        }
    }

    #[inline]
    fn read_bytes(&mut self, address: u64, buffer: &mut [u8]) -> Result<()> {
        unsafe {
            let ptr = address as *const u8;
            if ptr.is_null() {
                return Err(StackWalkError::MemoryReadFailed { address });
            }
            std::ptr::copy_nonoverlapping(ptr, buffer.as_mut_ptr(), buffer.len());
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_reader_basic() {
        let data = 0x123456789abcdef0u64.to_le_bytes();
        let mut reader = SliceMemoryReader::new(0x1000, &data);

        assert_eq!(reader.read_u64(0x1000).unwrap(), 0x123456789abcdef0);
    }

    #[test]
    fn test_slice_reader_out_of_bounds() {
        let data = [0u8; 8];
        let mut reader = SliceMemoryReader::new(0x1000, &data);

        assert!(reader.read_u64(0x0fff).is_err()); // Before range
        assert!(reader.read_u64(0x1001).is_err()); // Partial overlap
        assert!(reader.read_u64(0x2000).is_err()); // After range
    }

    #[test]
    fn test_closure_reader() {
        let mut reader = ClosureMemoryReader::new(|addr| {
            if addr == 0x1000 {
                Ok(0xdeadbeef)
            } else {
                Err(())
            }
        });

        assert_eq!(reader.read_u64(0x1000).unwrap(), 0xdeadbeef);
        assert!(reader.read_u64(0x2000).is_err());
    }
}
