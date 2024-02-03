use core::fmt;
use std::error::Error;

use mmap_rs::{Mmap, MmapOptions};

use crate::{ast::Ast, debugger, ir::{BuiltInTypes, StringValue, Value}, parser::Parser, Data, Message};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    name: String,
    offset: usize,
    jump_table_offset: usize,
    is_foreign: bool,
    pub is_builtin: bool,
}

#[derive(Debug)]
struct HeapObject<'a> {
    size: usize,
    data: &'a [u8],
}

pub struct Compiler {
    code_offset: usize,
    code_memory: Option<Mmap>,
    // TODO: Think about the jump table more
    jump_table: Option<Mmap>,
    // DO I need this offset?
    jump_table_offset: usize,
    functions: Vec<Function>,
    #[allow(dead_code)]
    // Need much better system obviously
    heap: Option<Mmap>,
    heap_offset: usize,
    string_constants: Vec<StringValue>,
}

impl fmt::Debug for Compiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: Make this better
        f.debug_struct("Compiler")
            .field("code_offset", &self.code_offset)
            .field("jump_table_offset", &self.jump_table_offset)
            .field("functions", &self.functions)
            .finish()
    }
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    pub fn new() -> Self {
        Self {
            code_memory: Some(
                MmapOptions::new(MmapOptions::page_size())
                    .unwrap()
                    .map()
                    .unwrap(),
            ),
            code_offset: 0,
            jump_table: Some(
                MmapOptions::new(MmapOptions::page_size())
                    .unwrap()
                    .map()
                    .unwrap(),
            ),
            jump_table_offset: 0,
            functions: Vec::new(),
            heap: Some(
                MmapOptions::new(MmapOptions::page_size() * 100)
                    .unwrap()
                    .map()
                    .unwrap(),
            ),
            heap_offset: 0,
            string_constants: vec![],
        }
    }

    pub fn get_heap_pointer(&self) -> usize {
        self.heap.as_ref().unwrap().as_ptr() as usize
    }

    pub fn allocate(&mut self, size: usize) -> Result<usize, Box<dyn Error>> {
        let size = size * 8;
        let memory = self.heap.take();
        let mut memory = memory.unwrap().make_mut().map_err(|(_, e)| e)?;
        let buffer = &mut memory[self.heap_offset..];
        // write the size of the object to the first 8 bytes
        for (index, byte) in size.to_le_bytes().iter().enumerate() {
            buffer[index] = *byte;
        }
        self.heap_offset += size + 8;
        let pointer = buffer.as_ptr() as usize;
        let pointer = BuiltInTypes::Array.tag(pointer as isize) as usize;
        self.heap = Some(memory.make_read_only().unwrap_or_else(|(_map, e)| {
            panic!("Failed to make mmap executable: {}", e);
        }));
        Ok(pointer)
    }

    pub fn add_foreign_function(
        &mut self,
        name: &str,
        function: *const u8,
    ) -> Result<usize, Box<dyn Error>> {
        let index = self.functions.len();
        let offset = function as usize;
        let jump_table_offset = self.add_jump_table_entry(index, offset)?;
        self.functions.push(Function {
            name: name.to_string(),
            offset,
            jump_table_offset,
            is_foreign: true,
            is_builtin: false,
        });
        debugger(Message {
            kind: "foreign_function".to_string(),
            data: Data::ForeignFunction {
                name: name.to_string(),
                pointer: Self::get_function_pointer(self, self.functions.last().unwrap().clone()).unwrap()
            }
        });
        Ok(self.functions.len() - 1)
    }
    pub fn add_builtin_function(
        &mut self,
        name: &str,
        function: *const u8,
    ) -> Result<usize, Box<dyn Error>> {
        let index = self.functions.len();
        let offset = function as usize;

        debugger(Message {
            kind: "builtin_function".to_string(),
            data: Data::BuiltinFunction {
                name: name.to_string(),
                pointer: Self::get_function_pointer(self, self.functions.last().unwrap().clone()).unwrap()
            }
        });
        let jump_table_offset = self.add_jump_table_entry(index, offset)?;
        self.functions.push(Function {
            name: name.to_string(),
            offset,
            jump_table_offset,
            is_foreign: true,
            is_builtin: true,
        });
        Ok(self.functions.len() - 1)
    }

    pub fn upsert_function(&mut self, name: &str, code: &[u8]) -> Result<usize, Box<dyn Error>> {
        for (index, function) in self.functions.iter_mut().enumerate() {
            if function.name == name {
                self.overwrite_function(index, code)?;
                return Ok(index);
            }
        }
        self.add_function(name, code)
    }

    pub fn reserve_function(&mut self, name: &str) -> Result<Function, Box<dyn Error>> {
        for function in self.functions.iter_mut() {
            if function.name == name {
                return Ok(function.clone());
            }
        }
        let index = self.functions.len();
        let jump_table_offset = self.add_jump_table_entry(index, 0)?;
        let function = Function {
            name: name.to_string(),
            offset: 0,
            jump_table_offset,
            is_foreign: false,
            is_builtin: true,
        };
        self.functions.push(function.clone());
        Ok(function)
    }

    pub fn add_function(&mut self, name: &str, code: &[u8]) -> Result<usize, Box<dyn Error>> {
        let offset = self.add_code(code)?;
        let index = self.functions.len();
        self.functions.push(Function {
            name: name.to_string(),
            offset,
            jump_table_offset: 0,
            is_foreign: false,
            is_builtin: false,
        });
        debugger(Message {
            kind: "user_function".to_string(),
            data: Data::UserFunction {
                name: name.to_string(),
                pointer: Self::get_function_pointer(self, self.functions.last().unwrap().clone()).unwrap(),
                len: code.len(),
            }
        });
        let function_pointer = Self::get_function_pointer(self, self.functions.last().unwrap().clone()).unwrap();
        let jump_table_offset = self.add_jump_table_entry(index, function_pointer)?;

        self.functions[index].jump_table_offset = jump_table_offset;
        Ok(jump_table_offset)
    }
    pub fn overwrite_function(&mut self, index: usize, code: &[u8]) -> Result<(), Box<dyn Error>> {
        let offset = self.add_code(code)?;
        let function = &mut self.functions[index];
        function.offset = offset;
        self.add_jump_table_entry(index, offset)?;
        Ok(())
    }

    pub fn get_function_pointer(&self, function: Function) -> Result<usize, Box<dyn Error>> {
        // Gets the absolute pointer to a function
        // if it is a foreign function, return the offset
        // if it is a local function, return the offset + the start of code_memory
        if function.is_foreign {
            Ok(function.offset)
        } else {
            Ok(function.offset + self.code_memory.as_ref().unwrap().as_ptr() as usize)
        }
    }

    pub fn add_jump_table_entry(
        &mut self,
        index: usize,
        offset: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let jump_table_offset = self.jump_table_offset;
        self.jump_table_offset += 1;
        let memory = self.jump_table.take();
        let mut memory = memory.unwrap().make_mut().map_err(|(_, e)| e)?;
        let buffer = &mut memory[jump_table_offset * 8..];
        // Write full usize to buffer
        for (index, byte) in offset.to_le_bytes().iter().enumerate() {
            buffer[index] = *byte;
        }
        let mem = memory.make_read_only().unwrap_or_else(|(_map, e)| {
            panic!("Failed to make mmap executable: {}", e);
        });
        self.jump_table_offset += 1;
        self.jump_table = Some(mem);
        Ok(jump_table_offset)
    }

    pub fn add_code(&mut self, code: &[u8]) -> Result<usize, Box<dyn Error>> {
        let start = self.code_offset;
        let memory = self.code_memory.take();
        let mut memory = memory.unwrap().make_mut().map_err(|(_, e)| e)?;
        let buffer = &mut memory[self.code_offset..];

        for (index, byte) in code.iter().enumerate() {
            buffer[index] = *byte;
        }

        let size: usize = memory.size();
        memory.flush(0..size)?;
        memory.flush_icache()?;
        self.code_offset += code.len();

        let exec = memory.make_exec().unwrap_or_else(|(_map, e)| {
            panic!("Failed to make mmap executable: {}", e);
        });

        self.code_memory = Some(exec);

        Ok(start)
    }

    pub fn get_compiler_ptr(&self) -> *const Compiler {
        self as *const Compiler
    }

    // TODO: Make this good
    pub fn run(&self, jump_table_offset: usize) -> Result<u64, Box<dyn Error>> {
        // get offset stored in jump table as a usize
        let offset =
            &self.jump_table.as_ref().unwrap()[jump_table_offset * 8..jump_table_offset * 8 + 8];
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(offset);
        let start = usize::from_le_bytes(bytes);
        let memory = &self.code_memory.as_ref().unwrap()[start..];
        let f: fn() -> u64 = unsafe { std::mem::transmute(memory.as_ref().as_ptr()) };
        Ok(f())
    }

    pub fn run1(&self, jump_table_offset: usize, arg: u64) -> Result<u64, Box<dyn Error>> {
        // get offset stored in jump table as a usize
        let offset =
            &self.jump_table.as_ref().unwrap()[jump_table_offset * 8..jump_table_offset * 8 + 8];
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(offset);
        let start = usize::from_le_bytes(bytes) as *const u8;
        let f: fn(u64) -> u64 = unsafe { std::mem::transmute(start) };
        let arg = BuiltInTypes::Int.tag(arg as isize) as u64;
        let result = f(arg);
        // TODO: When running in release mode, this fails here.
        // I'm guessing I'm not setting up the stack correctly
        Ok(result)
    }

    pub fn run2(
        &self,
        jump_table_offset: usize,
        arg1: u64,
        arg2: u64,
    ) -> Result<u64, Box<dyn Error>> {
        // get offset stored in jump table as a usize
        let offset =
            &self.jump_table.as_ref().unwrap()[jump_table_offset * 8..jump_table_offset * 8 + 8];
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(offset);
        let start = usize::from_le_bytes(bytes);
        let memory = &self.code_memory.as_ref().unwrap()[start..];
        let f: fn(u64, u64) -> u64 = unsafe { std::mem::transmute(memory.as_ref().as_ptr()) };
        Ok(f(arg1, arg2))
    }

    pub fn print(&self, value: usize) {
        let tag = BuiltInTypes::get_kind(value);
        match tag {
            BuiltInTypes::Int => {
                let value = BuiltInTypes::untag(value);
                println!("{}", value);
            }
            BuiltInTypes::Float => todo!(),
            BuiltInTypes::String => {
                let value = BuiltInTypes::untag(value);
                let string = unsafe { &*(value as *const StringValue) };
                println!("{}", string.str);
            }
            BuiltInTypes::Bool => {
                let value = BuiltInTypes::untag(value);
                if value == 0 {
                    println!("false");
                } else {
                    println!("true");
                }
            }
            BuiltInTypes::Function => todo!(),
            BuiltInTypes::Struct => todo!(),
            BuiltInTypes::Array => {
                unsafe {
                    let value = BuiltInTypes::untag(value);
                    let pointer = value as *const u8;
                    // get first 8 bytes as size
                    let size = *(pointer as *const usize);
                    let pointer = pointer.add(8);
                    let data = std::slice::from_raw_parts(pointer, size);
                    let heap_object = HeapObject { size, data };

                    println!("{:?}", heap_object);
                }
                // print!("[");
                // for i in 0..array.size {
                //     let value = array.data[i];
                //     self.print(value as usize);
                //     if i != array.size - 1 {
                //         print!(", ");
                //     }
                // }
                // println!("]");
            }
        }
    }

    pub fn array_store(
        &mut self,
        array: usize,
        index: usize,
        value: usize,
    ) -> Result<usize, Box<dyn Error>> {
        unsafe {
            let tag = BuiltInTypes::get_kind(array);
            match tag {
                BuiltInTypes::Array => {
                    // TODO: Bounds check
                    let index = BuiltInTypes::untag(index);
                    let index = index * 8;
                    let heap = self.heap.take();
                    let heap = heap.unwrap().make_mut().map_err(|(_, e)| e)?;
                    let array = BuiltInTypes::untag(array);
                    let pointer = array as *mut u8;
                    // get first 8 bytes as size
                    let size = *(pointer as *const usize);
                    let pointer = pointer.add(8);
                    let data = std::slice::from_raw_parts_mut(pointer, size);
                    // store all 8 bytes of value in data
                    for (offset, byte) in value.to_le_bytes().iter().enumerate() {
                        data[index + offset] = *byte;
                    }
                    let mem = heap.make_read_only().unwrap_or_else(|(_map, e)| {
                        panic!("Failed to make mmap executable: {}", e);
                    });

                    self.heap = Some(mem);
                    Ok(BuiltInTypes::Array.tag(array as isize) as usize)
                }
                _ => panic!("Not an array"),
            }
        }
    }

    pub(crate) fn array_get(
        &mut self,
        array: usize,
        index: usize,
    ) -> Result<usize, Box<dyn Error>> {
        unsafe {
            let tag = BuiltInTypes::get_kind(array);
            match tag {
                BuiltInTypes::Array => {
                    let index = BuiltInTypes::untag(index);
                    let array = BuiltInTypes::untag(array);
                    let pointer = array as *mut u8;
                    // get first 8 bytes as size
                    let size = *(pointer as *const usize);
                    let pointer = pointer.add(8);
                    let data = std::slice::from_raw_parts_mut(pointer, size);
                    // get next 8 bytes as usize
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&data[index * 8..index * 8 + 8]);
                    let data = usize::from_le_bytes(bytes);
                    Ok(data)
                }
                _ => panic!("Not an array"),
            }
        }
    }

    fn get_top_level_functions(&self, ast: crate::ast::Ast) -> Vec<Ast> {
        let mut functions = Vec::new();
        for node in ast.nodes() {
            match node {
                Ast::Function {..} => {
                    // TODO: Get rid of this clone
                    functions.push(node.clone());
                }
                _ => {}
            }
        }
        functions
    }

    pub fn compile(&mut self, code: String) -> Result<(), Box<dyn Error>> {
        let mut parser = Parser::new(code);
        let ast = parser.parse();
        self.compile_ast(ast)
    }

    // TODO: Strings disappear because I am holding on to them only in IR

    pub fn compile_ast(&mut self, ast: crate::ast::Ast) -> Result<(), Box<dyn Error>> {
        let functions = self.get_top_level_functions(ast);
        for function in functions {
            let mut ir = function.compile(self);
            let mut code = ir.compile();
            self.add_function(&function.name(), &code.compile_to_bytes())?;
        }
        Ok(())
    }

    // TODO: Make less ugly
    pub(crate) fn run_function(&self, arg: &str, vec: Vec<i32>) -> u64 {
        match vec.len() {
            0 => {
                let function = self.functions.iter().find(|f| f.name == arg).unwrap();
                self.run(function.jump_table_offset).unwrap()
            }
            1 => {
                let function = self.functions.iter().find(|f| f.name == arg).unwrap();
                self.run1(function.jump_table_offset, vec[0] as u64).unwrap()
            }
            2 => {
                let function = self.functions.iter().find(|f| f.name == arg).unwrap();
                self.run2(function.jump_table_offset, vec[0] as u64, vec[1] as u64).unwrap()
            }
            _ => panic!("Too many arguments"),
        }
    }

    pub fn add_string(&mut self, string_value: StringValue) -> Value {
        self.string_constants.push(string_value);
        let last = self.string_constants.last().unwrap();
        return Value::StringConstantPtr(last as *const StringValue as usize)
    }
}
