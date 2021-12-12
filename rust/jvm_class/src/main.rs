use std::fs::File;
use std::io::{BufReader};
use std::io::prelude::*;
use std::io;
use std::str;

// https://docs.oracle.com/javase/specs/jvms/se7/html/jvms-4.html
// Need to implement the other structs now.

#[derive(Debug)]
struct ClassFile {
    minor_version: u16,
    major_version: u16,
    constant_pool_count: u16,
    constant_pool: Vec<Constant>,
    access_flags: u16,
    this_class: u16,
    super_class: u16,
    interfaces_count: u16,
    interfaces: Vec<Constant>,
    fields_count: u16,
    fields: Vec<Constant>,
    methods_count: u16,
    methods: Vec<Constant>,
    attributes_count: u16,
    attributes: Vec<Constant>,
}

#[derive(Debug)]
enum Constant {
    MethodRef{tag: u8, class_index: u16, name_and_type_index: u16},
    FieldRef{tag: u8, class_index: u16, name_and_type_index: u16},
    InterfaceMethodRef{tag: u8, class_index: u16, name_and_type_index: u16},
    String{tag: u8, string_index: u16},
    Class{tag: u8, name_index: u16},
    UTF8{tag: u8, length: u16, string: String},
    NameAndType{tag:u8, name_index: u16, descriptor_index: u16},
    Noop,
}

fn parse_single_constant(reader: &mut BinaryReader) -> io::Result<Constant> {
    let tag = reader.read_u8()?;
    Ok(match tag {
        1 => {
            let length = reader.read_u16()?;
            let mut bytes = vec![0; length as usize];
            reader.read_exact(&mut bytes)?;
            let string = str::from_utf8(bytes.as_slice()).unwrap().to_string();
            Constant::UTF8{tag, length, string}
        }
        7 => {
            let name_index = reader.read_u16()?;
            Constant::Class{tag, name_index}
        }
        8 => {
            let string_index = reader.read_u16()?;
            Constant::String{tag, string_index}
        }
        9 => {
            let class_index = reader.read_u16()?;
            let name_and_type_index = reader.read_u16()?;
            Constant::FieldRef{tag, class_index, name_and_type_index}
        },
        10 => {
            let class_index = reader.read_u16()?;
            let name_and_type_index = reader.read_u16()?;
            Constant::MethodRef{tag, class_index, name_and_type_index}
        },
        11 => {
            let class_index = reader.read_u16()?;
            let name_and_type_index = reader.read_u16()?;
            Constant::InterfaceMethodRef{tag, class_index, name_and_type_index}
        }
        12 => {
            let name_index = reader.read_u16()?;
            let descriptor_index = reader.read_u16()?;
            Constant::NameAndType{tag, name_index, descriptor_index}
        }
        _ => {
            Constant::Noop
            // panic!("Didn't handle {:?}", tag)
        }
    })
}

fn parse_constants(entries: u16, reader: &mut BinaryReader) -> io::Result<Vec<Constant>> {
    let mut constants = vec![];
    for _ in 0..entries {
        constants.push(parse_single_constant(reader)?)
    }
    Ok(constants)
}


struct BinaryReader<'a> {
    reader: &'a mut dyn Read,
    buffer: Vec<u8>,
}

impl<'a> BinaryReader<'a> {

    fn read_u8(&mut self) -> io::Result<u8> {
        let mut limited_reader = self.reader.take(1);
        limited_reader.read_to_end(&mut self.buffer)?;
        let bytes = [self.buffer[0]];
        self.buffer.clear();
        Ok(u8::from_be_bytes(bytes))
    }

    fn read_u16(&mut self) -> io::Result<u16> {
        // No idea about efficiency here.
        let mut limited_reader = self.reader.take(2);
        limited_reader.read_to_end(&mut self.buffer)?;
        let bytes = [self.buffer[0], self.buffer[1]];
        self.buffer.clear();
        Ok(u16::from_be_bytes(bytes))
    }

    // I'm sure there is a better way to do these methods.
    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.reader.read_exact(buf)
    }
}

// Feel like this making multiple buffers thing is a bad idea.
// Need to abstract this with a single buffer
fn read_class_file() -> io::Result<ClassFile> {
    let f = File::open("/Users/jimmyhmiller/Desktop/test-java-stuff/lang/makeFn.class").unwrap();
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
    let constant_pool = parse_constants(constant_pool_count - 1, &mut binary_reader)?;

    println!("Finished constants pool");

    let access_flags = binary_reader.read_u16()?;
    let this_class = binary_reader.read_u16()?;
    let super_class = binary_reader.read_u16()?;

    // It looks like interfaces are Constant::Class but not 100% sure. Not interfaces in hello world
    // Or these might be numbers? 
    let interfaces_count = binary_reader.read_u16()?;
    let interfaces = parse_constants(interfaces_count, &mut binary_reader)?;
    println!("Finished interfaces");


    // need to parse out fields, methods and attributes next.

    let fields_count = binary_reader.read_u16()?;
    let fields = parse_constants(fields_count, &mut binary_reader)?;
    println!("Finished fields");

    
    let methods_count = binary_reader.read_u16()?;
    let methods = parse_constants(methods_count, &mut binary_reader)?;
    println!("Finished methods");

    // let attributes_count = binary_reader.read_u16()?;
    // let attributes =  parse_constants(attributes_count, &mut binary_reader)?;
    // println!("Finished attributes");
    // let mut buffer = vec![];
    // binary_reader.reader.read_to_end(&mut buffer)?;


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
        attributes_count: 0,
        attributes: vec![],
    })
}

fn main() {
    println!("{:#?}", read_class_file().expect("failed reading class"));
}
