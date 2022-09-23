use std::mem;
use memmap2::{MmapMut, Mmap};
use std::io::Result;

// Notes:
// Can't make things RWX
// This is big endian


struct Page {
    mem_write: Option<MmapMut>,
    mem_exec: Option<Mmap>,
}

impl Page {
    fn new() -> Result<Self> {
        let mem = MmapMut::map_anon(4096)?;
        Ok(Self {
            mem_write: Some(mem),
            mem_exec: None,
        })
    }

    fn write_u32(&mut self, offset: usize, data: u32) -> Result<()> {
        self.writeable()?;
        let memory = &mut self.mem_write.as_mut().unwrap()[..];
        for (i, byte) in data.to_be_bytes().iter().enumerate() {
            memory[offset as usize + i] = *byte;
        }
        Ok(())
    }


    fn executable(&mut self) -> Result<()> {
        if let Some(m) = self.mem_write.take() {
            let m = m.make_exec()?;
            self.mem_exec = Some(m);
        }
        Ok(())
    }

    fn writeable(&mut self) -> Result<()> {
        if let Some(m) = self.mem_exec.take() {
            let m = m.make_mut()?;
            self.mem_write = Some(m);
        }
        Ok(())
    }

    fn get_function(&mut self) -> Result<extern "C" fn() -> i32> {
        self.writeable()?;
        self.mem_write.as_mut().unwrap().flush()?;
        self.executable()?;
        Ok(unsafe {
            mem::transmute(self.mem_exec.as_ref().unwrap().as_ptr())
        })
    }

}



fn main() -> Result<()> {


    let mut page = Page::new()?;
    let move_2_to_w0 = 0xE00280D2;
    let ret: u32 = 0xC0035FD6;
    page.write_u32(0, move_2_to_w0)?;
    page.write_u32(4, ret)?;

    let main_fn = page.get_function()?;
    println!("Hello, world! {}", main_fn());

    return Ok(())
}
