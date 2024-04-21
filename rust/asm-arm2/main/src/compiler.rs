use core::fmt;
use std::{collections::HashMap, error::Error, slice::{from_raw_parts, from_raw_parts_mut}};

use mmap_rs::{Mmap, MmapMut, MmapOptions};

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

#[derive(Debug, Clone)]
pub struct Struct {
    pub name: String,
    pub fields: Vec<String>,
}

impl Struct {
    pub fn size(&self) -> usize {
        self.fields.len()
    }
}

struct StructManager {
    name_to_id: HashMap<String, usize>,
    structs: Vec<Struct>,
}

impl StructManager {
    fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
            structs: Vec::new(),
        }
    }

    fn insert(&mut self, name: String, s: Struct) {
        let id = self.structs.len();
        self.name_to_id.insert(name.clone(), id);
        self.structs.push(s);
    }

    fn get(&self, name: &str) -> Option<(usize, &Struct)> {
        let id = self.name_to_id.get(name)?;
        self.structs.get(*id).map(|x| (*id, x))
    }
    
    fn get_by_id(&self, type_id: usize) -> Option<&Struct> {
        self.structs.get(type_id)
    }
}


pub struct Compiler {
    code_offset: usize,
    code_memory: Option<Mmap>,
    // TODO: Think about the jump table more
    jump_table: Option<Mmap>,
    // DO I need this offset?
    jump_table_offset: usize,
    structs: StructManager,
    functions: Vec<Function>,
    #[allow(dead_code)]
    // Need much better system obviously
    heap: Option<MmapMut>,
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
            structs: StructManager::new(),
            functions: Vec::new(),
            heap: Some(
                MmapOptions::new(MmapOptions::page_size() * 100)
                    .unwrap()
                    .map()
                    .unwrap()
                    .make_mut()
                    .unwrap_or_else(|(_map, e)| {
                        panic!("Failed to make mmap executable: {}", e);
                    }),
            ),
            heap_offset: 0,
            string_constants: vec![],
        }
    }

    pub fn get_heap_pointer(&self) -> usize {
        self.heap.as_ref().unwrap().as_ptr() as usize
    }

    pub fn allocate(&mut self, bytes: usize) -> Result<usize, Box<dyn Error>> {
        let size = bytes * 8;
        let memory = self.heap.take();
        let mut memory = memory.unwrap();
        let buffer = &mut memory[self.heap_offset..];
        // write the size of the object to the first 8 bytes
        for (index, byte) in size.to_le_bytes().iter().enumerate() {
            buffer[index] = *byte;
        }
        self.heap_offset += size + 8;
        let pointer = buffer.as_ptr() as usize;
        self.heap = Some(memory);
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

        let jump_table_offset = self.add_jump_table_entry(index, offset)?;
        self.functions.push(Function {
            name: name.to_string(),
            offset,
            jump_table_offset,
            is_foreign: true,
            is_builtin: true,
        });
        debugger(Message {
            kind: "builtin_function".to_string(),
            data: Data::BuiltinFunction {
                name: name.to_string(),
                pointer: Self::get_function_pointer(self, self.functions.last().unwrap().clone()).unwrap()
            }
        });
        Ok(self.functions.len() - 1)
    }

    pub fn upsert_function(&mut self, name: &str, code: &[u8]) -> Result<usize, Box<dyn Error>> {
        for (index, function) in self.functions.iter_mut().enumerate() {
            if function.name == name {
                self.overwrite_function(index, code)?;
                break;
            }
        }
        self.add_function(name, code)?;

        let function = self.find_function(name).unwrap();
        let function_pointer = Self::get_function_pointer(self, function.clone()).unwrap();
        debugger(Message {
            kind: "user_function".to_string(),
            data: Data::UserFunction {
                name: name.to_string(),
                pointer: function_pointer,
                len: code.len(),
            }
        });
        Ok(function_pointer)
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
        let function_pointer = Self::get_function_pointer(self, self.functions.last().unwrap().clone()).unwrap();
        let jump_table_offset = self.add_jump_table_entry(index, function_pointer)?;

        self.functions[index].jump_table_offset = jump_table_offset;
        Ok(jump_table_offset)
    }

    pub fn overwrite_function(&mut self, index: usize, code: &[u8]) -> Result<usize, Box<dyn Error>> {
        let offset = self.add_code(code)?;
        let function = &mut self.functions[index];
        function.offset = offset;
        let function_pointer = Self::get_function_pointer(self, self.functions.last().unwrap().clone()).unwrap();
        let jump_table_offset = self.add_jump_table_entry(index, function_pointer)?;
        let function = &mut self.functions[index];
        function.jump_table_offset = jump_table_offset;
        Ok(jump_table_offset)
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
        pointer: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let jump_table_offset = self.jump_table_offset;
        self.jump_table_offset += 1;
        let memory = self.jump_table.take();
        let mut memory = memory.unwrap().make_mut().map_err(|(_, e)| e)?;
        let buffer = &mut memory[jump_table_offset * 8..];
        // Write full usize to buffer
        for (index, byte) in pointer.to_le_bytes().iter().enumerate() {
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
        let start = usize::from_le_bytes(bytes) as *const u8;
        let f: fn() -> u64 = unsafe { std::mem::transmute(start) };
        let result = f();
        // TODO: When running in release mode, this fails here.
        // I'm guessing I'm not setting up the stack correctly
        Ok(result)
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
        let arg1 = BuiltInTypes::Int.tag(arg1 as isize) as u64;
        let arg2 = BuiltInTypes::Int.tag(arg2 as isize) as u64;
        let f: fn(u64, u64) -> u64 = unsafe { std::mem::transmute(start) };
        Ok(f(arg1, arg2))
    }

    pub fn get_repr(&self, value: usize) -> String {
        let tag = BuiltInTypes::get_kind(value);
        match tag {
            BuiltInTypes::Int => {
                let value = BuiltInTypes::untag(value);
                value.to_string()
            }
            BuiltInTypes::Float => todo!(),
            BuiltInTypes::String => {
                let value = BuiltInTypes::untag(value);
                let string = unsafe { &*(value as *const StringValue) };
                string.str.clone()
            }
            BuiltInTypes::Bool => {
                let value = BuiltInTypes::untag(value);
                if value == 0 {
                    "false".to_string()
                } else {
                    "true".to_string()
                }
            }
            BuiltInTypes::Function => todo!(),
            BuiltInTypes::Closure => todo!(),
            BuiltInTypes::Struct => {
                unsafe {
                    let value = BuiltInTypes::untag(value);
                    let pointer = value as *const u8;
                    // get first 8 bytes as size le encoded
                    let size = *(pointer as *const usize);
                    let pointer = pointer.add(8);
                    let data = std::slice::from_raw_parts(pointer, size);
                    // type id is the first 8 bytes of data
                    let type_id = usize::from_le_bytes(data[0..8].try_into().unwrap());
                    let type_id = BuiltInTypes::untag(type_id);
                    let struct_value = self.structs.get_by_id(type_id as usize);
                    self.get_struct_repr(struct_value.unwrap(), data[8..].to_vec())
                }
            }
            BuiltInTypes::Array => {
                unsafe {
                    let value = BuiltInTypes::untag(value);
                    let pointer = value as *const u8;
                    // get first 8 bytes as size le encoded
                    let size = *(pointer as *const usize);
                    let pointer = pointer.add(8);
                    let data = std::slice::from_raw_parts(pointer, size);
                    let heap_object = HeapObject { size, data };

                    let mut repr = "[".to_string();
                    for i in 0..heap_object.size {
                        let value = &heap_object.data[i * 8..i * 8 + 8];
                        let mut bytes = [0u8; 8];
                        bytes.copy_from_slice(value);
                        let value = usize::from_le_bytes(bytes);
                        repr.push_str(&self.get_repr(value));
                        if i != heap_object.size - 1 {
                            repr.push_str(", ");
                        }
                    }
                    repr.push_str("]");
                    repr
                }
            }
        }
    }

    pub fn print(&self, value: usize) {
        let tag = BuiltInTypes::get_kind(value);
        println!("{}", self.get_repr(value));
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
        ast.compile(self);
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

    pub(crate) fn find_function(&self, name: &str) -> Option<Function> {
        self.functions.iter().find(|f| f.name == name).cloned()
    }

    pub fn get_function_by_pointer(&self, value: usize) -> Option<&Function> {
        let offset = value - self.code_memory.as_ref().unwrap().as_ptr() as usize;
        self.functions.iter().find(|f| f.offset == offset)
    }

    // TODO: Call this
    // After I make a function, I need to check if there are free variables
    // If so I need to grab those variables from the evironment
    // pop them on the stack
    // call this funciton with a pointer to the base of the stack
    // and the number of variables
    // then I need to pop the stack
    // and get the closure
    // When it comes to function calls
    // if it is a function, do direct call.
    // if it is a closure, grab the closure data
    // and put the free variables on the stack before the locals

    pub fn make_closure(&mut self, function: usize, free_variables: &[usize]) -> Result<usize, Box<dyn Error>> {
        let len = 8 + 8 + free_variables.len() * 8;
        let heap_pointer = self.allocate(len)?;
        let pointer = heap_pointer as *mut u8;
        let num_free = free_variables.len();
        let function = function.to_le_bytes();
        let free_variables = free_variables.iter().map(|v| v.to_le_bytes()).flatten();
        let buffer = unsafe { from_raw_parts_mut(pointer, len) };
        // write function pointer
        for (index, byte) in function.iter().enumerate() {
            buffer[index] = *byte;
        }
        let num_free = num_free.to_le_bytes();
        // Write number of free variables
        for (index, byte) in num_free.iter().enumerate() {
            buffer[8 + index] = *byte;
        }
        // write free variables
        for (index, byte) in free_variables.enumerate() {
            buffer[16 + index] = byte;
        }
        Ok(BuiltInTypes::Closure.tag(heap_pointer as isize) as usize)
    }
    
    pub fn add_struct(&mut self, s: Struct) {
        self.structs.insert(s.name.clone(), s);
    }
    
    pub fn get_struct(&self, name: &str) -> Option<(usize, &Struct)> {
        self.structs.get(name)
    }
    
    fn get_struct_repr(&self, struct_value: &Struct, to_vec: Vec<u8>) -> String {
        // It should look like this
        // struct_name { field1: value1, field2: value2 }
        let mut repr = struct_value.name.clone();
        repr.push_str(" { ");
        for (index, field) in struct_value.fields.iter().enumerate() {
            repr.push_str(field);
            repr.push_str(": ");
            let value = &to_vec[index * 8..index * 8 + 8];
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(value);
            let value = usize::from_le_bytes(bytes);
            repr.push_str(&self.get_repr(value));
            if index != struct_value.fields.len() - 1 {
                repr.push_str(", ");
            }
        }
        repr.push_str(" }");
        repr

    }
}
