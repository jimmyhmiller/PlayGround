use std::fs::File;
use std::io::{BufReader};
use std::io::prelude::*;
use std::io;

// https://docs.oracle.com/javase/specs/jvms/se7/html/jvms-4.html
// Need to implement the other structs now.

#[derive(Debug)]
struct ClassFile {
    minor_version: u16,
    major_version: u16,
    constant_pool_count: u16,
    constant_pool: Vec<u8>,
    access_flags: u16,
    this_class: u16,
    super_class: u16,
    interfaces_count: u16,
    interfaces: Vec<u8>,
    fields_count: u16,
    fields: Vec<u8>,
    methods_count: u16,
    methods: Vec<u8>,
    attributes_count: u16,
    attributes: Vec<u8>,
}


struct BinaryReader<'a> {
    reader: &'a mut BufReader<File>,
    buffer: Vec<u8>,
}

impl<'a> BinaryReader<'a> {
    fn read_u16(&mut self) -> io::Result<u16> {
        // No idea about efficiency here.
        let mut limited_reader = self.reader.take(2);
        limited_reader.read_to_end(&mut self.buffer)?;
        let bytes = [self.buffer[0], self.buffer[1]];
        self.buffer.clear();
        Ok(u16::from_be_bytes(bytes))
    }
    fn read_n(&mut self, n : u64) -> io::Result<Vec<u8>> {
        // No idea about efficiency here.
        let mut buffer = Vec::with_capacity(n as usize);
        let mut limited_reader = self.reader.take(n);
        limited_reader.read_to_end(&mut buffer)?;
        Ok(buffer)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.reader.read_exact(buf)
    }
}

// Feel like this making multiple buffers thing is a bad idea.
// Need to abstract this with a single buffer
fn read_class_file() -> io::Result<ClassFile> {
    let f = File::open("./Hello.class").unwrap();
    let mut reader = BufReader::new(f);
    let mut binary_reader = BinaryReader{
        reader: &mut reader,
        buffer: vec![],
    };



    let mut buffer = [0; 4];
    binary_reader.read_exact(&mut buffer)?;
    assert!(buffer == [0xCA, 0xFE, 0xBA, 0xBE]);

    let minor_version = binary_reader.read_u16()?;
    let major_version = binary_reader.read_u16()?;

    let constant_pool_count = binary_reader.read_u16()?;
    let constant_pool = binary_reader.read_n((constant_pool_count - 1) as u64)?;

    let access_flags = binary_reader.read_u16()?;
    let this_class = binary_reader.read_u16()?;
    let super_class = binary_reader.read_u16()?;

    let interfaces_count = binary_reader.read_u16()?;
    let interfaces = binary_reader.read_n(interfaces_count as u64)?;

    let fields_count = binary_reader.read_u16()?;
    let fields = binary_reader.read_n(fields_count as u64)?;

    let methods_count = binary_reader.read_u16()?;
    let methods = binary_reader.read_n(methods_count as u64)?;

    let attributes_count = binary_reader.read_u16()?;
    let attributes = binary_reader.read_n(attributes_count as u64)?;

    Ok(ClassFile {
        minor_version,
        major_version,
        constant_pool_count,
        constant_pool,
        access_flags,
        this_class,
        super_class,
        interfaces_count,
        interfaces,
        fields_count,
        fields,
        methods_count,
        methods,
        attributes_count,
        attributes,
    })
}

fn main() {
    println!("{:?}", read_class_file().expect("failed reading class"));
}
