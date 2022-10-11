use std::mem;
// use memmap2::{MmapMut, Mmap};
use mmap_rs::{Error, MmapOptions, MmapMut, Mmap};

// Notes:
// Can't make things RWX
// This is big endian


struct Page {
    mem_write: Option<MmapMut>,
    mem_exec: Option<Mmap>,
}

impl Page {
    fn new() -> Result<Self, Error> {
        let mem = MmapOptions::new(MmapOptions::page_size().0)
        .map_mut()?;

        Ok(Self {
            mem_write: Some(mem),
            mem_exec: None,
        })
    }

    fn write_u32_be(&mut self, offset: usize, data: u32) -> Result<(), Error> {
        self.writeable()?;
        let memory = &mut self.mem_write.as_mut().unwrap()[..];
        for (i, byte) in data.to_be_bytes().iter().enumerate() {
            memory[offset as usize + i] = *byte;
        }
        Ok(())
    }

    fn write_u32_le(&mut self, offset: usize, data: u32) -> Result<(), Error> {
        self.writeable()?;
        let memory = &mut self.mem_write.as_mut().unwrap()[..];
        for (i, byte) in data.to_le_bytes().iter().enumerate() {
            memory[offset as usize + i] = *byte;
        }
        Ok(())
    }


    fn executable(&mut self) -> Result<(), Error> {
        if let Some(m) = self.mem_write.take() {
            let m = m.make_exec().unwrap_or_else(|(_map, e)| {
                panic!("Failed to make mmap executable: {}", e);
            });
            self.mem_exec = Some(m);
        }
        Ok(())
    }

    fn writeable(&mut self) -> Result<(), Error> {
        if let Some(m) = self.mem_exec.take() {
            let m = m.make_mut().unwrap_or_else(|(_map, e)| {
                panic!("Failed to make mmap writeable: {}", e);
            });
            self.mem_write = Some(m);
        }
        Ok(())
    }

    fn get_function(&mut self) -> Result<extern "C" fn() -> u64, Error> {
        self.writeable()?;
        let size = self.mem_write.as_ref().unwrap().size();
        self.mem_write.as_mut().unwrap().flush(0..size)?;
        self.executable()?;
        Ok(unsafe {
            mem::transmute(self.mem_exec.as_ref().unwrap().as_ptr())
        })
    }

}


enum BitSize {
    B32,
    B64,
}

impl BitSize {
    fn sf(&self) -> u32  {
        match self {
            BitSize::B32 => 0,
            BitSize::B64 => 1,
        }
    }
}


struct Register {
    index: u8,
    size: BitSize,
}


// LSL
enum Shift {
    S0 = 0b00,
    S16 = 0b01,
    S32 = 0b10,
    S48 = 0b11,
}



fn encode_movz_16(destination: &Register, value: u16) -> u32 {
    0
    | destination.size.sf() << 31
    | 0b10 << 29
    | 0b1000 << 25
    | 0b101 << 23
    | (Shift::S0 as u32) << 21
    | (value as u32) << 5
    | destination.index as u32
}
fn encode_movk(destination: &Register, shift: Shift, value: u16) -> u32 {
    0
    | destination.size.sf() << 31
    | 0b11 << 29
    | 0b1000 << 25
    | 0b101 << 23
    | (shift as u32) << 21
    | (value as u32) << 5
    | destination.index as u32
}


// TODO:
// This is the longest possible encoding and we don't
// even short circuit here when we could.
fn encode_u64(destination: &Register, value: u64) -> [u32; 4] {
    let mut value = value;
    let mut result = [0; 4];
    result[0] = encode_movz_16(destination, value as u16 & 0xffff);
    value >>= 16;
    result[1] = encode_movk(destination, Shift::S16, value as u16 & 0xffff);
    value >>= 16;
    result[2] = encode_movk(destination, Shift::S32, value as u16 & 0xffff);
    value >>= 16;
    result[3] = encode_movk(destination, Shift::S48, value as u16 & 0xffff);
    result
}

fn main() -> Result<(), Error> {

    let mut page = Page::new()?;
    // let move_2_to_w0 = 0xE00280D2;
    // let movz2 = encode_movz_16(&Register { index: 0, size: BitSize::B64 }, 1);
    // let movk2 = encode_movk(&Register { index: 0, size: BitSize::B64 }, Shift::S48, 1);

    let mov = encode_u64(&Register { index: 0, size: BitSize::B64 }, 0x1234567890abcdef);
    page.write_u32_le(0, mov[0])?;
    page.write_u32_le(4, mov[1])?;
    page.write_u32_le(8, mov[2])?;
    page.write_u32_le(12, mov[3])?;

    // println!("{:#x}", movk2);
    let ret: u32 = 0xC0035FD6;
    // page.write_u32_le(0, movk2)?;
    page.write_u32_be(16, ret)?;

    let main_fn = page.get_function()?;
    println!("Hello, world! {:#x}", main_fn());

    return Ok(())
}
