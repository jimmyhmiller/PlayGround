use core::fmt;
use std::{collections::HashMap, error::Error, slice::from_raw_parts_mut};

use bincode::{Decode, Encode};
use mmap_rs::{Mmap, MmapMut, MmapOptions};

use crate::{
    arm::LowLevelArm,
    debugger,
    ir::{BuiltInTypes, StringValue, Value},
    parser::Parser,
    Data, Message,
};

// TODO: Probably need a better way if I want to be able to 
// abstract over different allocators
pub const GC_ALWAYS: bool = false;
pub const GC_NEVER: bool = false;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    name: String,
    offset_or_pointer: usize,
    jump_table_offset: usize,
    is_foreign: bool,
    pub is_builtin: bool,
    is_defined: bool,
    number_of_locals: usize,
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

    pub fn get_by_id(&self, type_id: usize) -> Option<&Struct> {
        self.structs.get(type_id)
    }
}

#[derive(Debug, Encode, Decode, Clone)]
pub struct StackMapDetails {
    pub number_of_locals: usize,
    pub current_stack_size: usize,
    pub max_stack_size: usize,
}

pub const STACK_SIZE: usize = 1024 * 1024 * 32;

pub struct StackMap {
    details: Vec<(usize, StackMapDetails)>,
}

impl StackMap {
    fn new() -> Self {
        Self { details: vec![] }
    }

    pub fn find_stack_data(&self, pointer: usize) -> Option<&StackMapDetails> {
        for (key, value) in self.details.iter() {
            if *key == pointer.saturating_sub(4) {
                return Some(value);
            }
        }
        None
    }
    
    fn extend(&mut self, translated_stack_map: Vec<(usize, StackMapDetails)>) {
        self.details.extend(translated_stack_map);
    }
}


pub trait Printer {
    fn print(&mut self, value: String);
    fn println(&mut self, value: String);
    // Gross just for testing. I'll need to do better;
    fn get_output(&self) -> Vec<String>;
}

pub struct DefaultPrinter;

impl Printer for DefaultPrinter {
    fn print(&mut self, value: String) {
        print!("{}", value);
    }

    fn println(&mut self, value: String) {
        println!("{}", value);
    }

    fn get_output(&self) -> Vec<String> {
        unimplemented!("We don't store this in the default")
    }
}


pub struct TestPrinter {
    pub output: Vec<String>,
    pub other_printer: Box<dyn Printer>,
}

impl TestPrinter {
    pub fn new(other_printer: Box<dyn Printer>) -> Self {
        Self { 
            output: vec![],
            other_printer: other_printer
        }
    }
}

impl Printer for TestPrinter {
    fn print(&mut self, value: String) {
        self.output.push(value.clone());
        self.other_printer.print(value);
    }

    fn println(&mut self, value: String) {
        self.output.push(value.clone());
        self.other_printer.println(value);
    }

    fn get_output(&self) -> Vec<String> {
        self.output.clone()
    }
}

pub trait Allocator {
    fn allocate(&mut self, stack: &mut MmapMut, stack_map: &StackMap, stack_pointer: usize, bytes: usize,  kind: BuiltInTypes) -> Result<usize, Box<dyn Error>>;
    fn gc(&mut self, stack: &mut MmapMut, stack_map: &StackMap, stack_pointer: usize);
}

pub struct Compiler<Alloc: Allocator> {
    code_offset: usize,
    code_memory: Option<Mmap>,
    // TODO: Think about the jump table more
    jump_table: Option<Mmap>,
    // Do I need this offset?
    jump_table_offset: usize,
    structs: StructManager,
    functions: Vec<Function>,
    heap: Alloc,
    stack: MmapMut,
    string_constants: Vec<StringValue>,
    stack_map: StackMap,
    pub printer: Box<dyn Printer>,
}

impl<Alloc : Allocator> fmt::Debug for Compiler<Alloc> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: Make this better
        f.debug_struct("Compiler")
            .field("code_offset", &self.code_offset)
            .field("jump_table_offset", &self.jump_table_offset)
            .field("functions", &self.functions)
            .finish()
    }
}

impl<Alloc: Allocator> Compiler<Alloc> {
    pub fn new(allocator: Alloc, printer: Box<dyn Printer>) -> Self {
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
            heap: allocator, 
            stack: MmapOptions::new(STACK_SIZE).unwrap().map_mut().unwrap(),
            string_constants: vec![],
            stack_map: StackMap::new(),
            printer,
        }
    }

    pub fn allocate(
        &mut self,
        bytes: usize,
        stack_pointer: usize,
        kind: BuiltInTypes,
    ) -> Result<usize, Box<dyn Error>> {
       self.heap.allocate(&mut self.stack, &self.stack_map, stack_pointer, bytes, kind)
    }

    fn get_stack_pointer(&self) -> usize {
        // I think I want the end of the stack
        (self.stack.as_ptr() as usize) + (STACK_SIZE)
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
            offset_or_pointer: offset,
            jump_table_offset,
            is_foreign: true,
            is_builtin: false,
            is_defined: true,
            number_of_locals: 0,
        });
        debugger(Message {
            kind: "foreign_function".to_string(),
            data: Data::ForeignFunction {
                name: name.to_string(),
                pointer: Self::get_function_pointer(self, self.functions.last().unwrap().clone())
                    .unwrap(),
            },
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
            offset_or_pointer: offset,
            jump_table_offset,
            is_foreign: true,
            is_builtin: true,
            is_defined: true,
            number_of_locals: 0,
        });
        debugger(Message {
            kind: "builtin_function".to_string(),
            data: Data::BuiltinFunction {
                name: name.to_string(),
                pointer: Self::get_function_pointer(self, self.functions.last().unwrap().clone())
                    .unwrap(),
            },
        });
        Ok(self.functions.len() - 1)
    }

    pub fn upsert_function(
        &mut self,
        name: &str,
        code: &mut LowLevelArm,
        number_of_locals: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let bytes = &(code.compile_to_bytes());
        for (index, function) in self.functions.iter_mut().enumerate() {
            if function.name == name {
                self.overwrite_function(index, bytes)?;
                break;
            }
        }
        self.add_function(name, bytes, number_of_locals)?;

        // TODO: Make this better
        let function = self.find_function(name).unwrap();
        let function_pointer = Self::get_function_pointer(self, function.clone()).unwrap();

        let translated_stack_map = code.translate_stack_map(function_pointer);
        let translated_stack_map: Vec<(usize, StackMapDetails)> = translated_stack_map
            .iter()
            .map(|(key, value)| {
                (
                    *key,
                    StackMapDetails {
                        current_stack_size: *value,
                        number_of_locals: code.max_locals as usize,
                        max_stack_size: code.max_stack_size as usize,
                    },
                )
            })
            .collect();

        debugger(Message {
            kind: "stack_map".to_string(),
            data: Data::StackMap {
                pc: function_pointer,
                name: name.to_string(),
                stack_map: translated_stack_map.clone(),
            },
        });
        self.stack_map.extend(translated_stack_map);

        debugger(Message {
            kind: "user_function".to_string(),
            data: Data::UserFunction {
                name: name.to_string(),
                pointer: function_pointer,
                len: bytes.len(),
            },
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
            offset_or_pointer: 0,
            jump_table_offset,
            is_foreign: false,
            is_builtin: false,
            is_defined: false,
            number_of_locals: 0,
        };
        self.functions.push(function.clone());
        Ok(function)
    }

    pub fn add_function(&mut self, name: &str, code: &[u8], number_of_locals: usize) -> Result<usize, Box<dyn Error>> {
        let offset = self.add_code(code)?;
        let index = self.functions.len();
        self.functions.push(Function {
            name: name.to_string(),
            offset_or_pointer: offset,
            jump_table_offset: 0,
            is_foreign: false,
            is_builtin: false,
            is_defined: true,
            number_of_locals,
        });
        let function_pointer =
            Self::get_function_pointer(self, self.functions.last().unwrap().clone()).unwrap();
        let jump_table_offset = self.add_jump_table_entry(index, function_pointer)?;

        self.functions[index].jump_table_offset = jump_table_offset;
        Ok(jump_table_offset)
    }

    pub fn overwrite_function(
        &mut self,
        index: usize,
        code: &[u8],
    ) -> Result<usize, Box<dyn Error>> {
        let offset = self.add_code(code)?;
        let function = &mut self.functions[index];
        function.offset_or_pointer = offset;
        let jump_table_offset = function.jump_table_offset;
        let function_clone = function.clone();
        let function_pointer = self.get_function_pointer(function_clone).unwrap();
        self.modify_jump_table_entry(jump_table_offset, function_pointer)?;
        let function = &mut self.functions[index];
        function.is_defined = true;
        Ok(function.jump_table_offset)
    }

    pub fn get_function_pointer(&self, function: Function) -> Result<usize, Box<dyn Error>> {
        // Gets the absolute pointer to a function
        // if it is a foreign function, return the offset
        // if it is a local function, return the offset + the start of code_memory
        if function.is_foreign {
            Ok(function.offset_or_pointer)
        } else {
            Ok(function.offset_or_pointer + self.code_memory.as_ref().unwrap().as_ptr() as usize)
        }
    }

    pub fn get_jump_table_pointer(&self, function: Function) -> Result<usize, Box<dyn Error>> {
        Ok(function.jump_table_offset * 8 + self.jump_table.as_ref().unwrap().as_ptr() as usize)
    }

    pub fn add_jump_table_entry(
        &mut self,
        _index: usize,
        pointer: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let jump_table_offset = self.jump_table_offset;
        let memory = self.jump_table.take();
        let mut memory = memory.unwrap().make_mut().map_err(|(_, e)| e)?;
        let buffer = &mut memory[jump_table_offset * 8..];
        let pointer = BuiltInTypes::Function.tag(pointer as isize) as usize;
        // Write full usize to buffer
        for (index, byte) in pointer.to_le_bytes().iter().enumerate() {
            buffer[index] = *byte;
        }
        let mem = memory.make_read_only().unwrap_or_else(|(_map, e)| {
            panic!("Failed to make mmap read_only: {}", e);
        });
        self.jump_table_offset += 1;
        self.jump_table = Some(mem);
        Ok(jump_table_offset)
    }

    fn modify_jump_table_entry(
        &mut self,
        jump_table_offset: usize,
        function_pointer: usize,
    ) -> Result<usize, Box<dyn Error>> {
        let memory = self.jump_table.take();
        let mut memory = memory.unwrap().make_mut().map_err(|(_, e)| e)?;
        let buffer = &mut memory[jump_table_offset * 8..];

        let function_pointer = BuiltInTypes::Function.tag(function_pointer as isize) as usize;
        // Write full usize to buffer
        for (index, byte) in function_pointer.to_le_bytes().iter().enumerate() {
            buffer[index] = *byte;
        }
        let mem = memory.make_read_only().unwrap_or_else(|(_map, e)| {
            panic!("Failed to make mmap read_only: {}", e);
        });
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

    pub fn get_compiler_ptr(&self) -> *const Compiler<Alloc> {
        self as *const Compiler<Alloc>
    }

    // TODO: Make this good
    pub fn run(&self, jump_table_offset: usize) -> Result<u64, Box<dyn Error>> {
        // get offset stored in jump table as a usize
        let offset =
            &self.jump_table.as_ref().unwrap()[jump_table_offset * 8..jump_table_offset * 8 + 8];
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(offset);
        let start = BuiltInTypes::untag(usize::from_le_bytes(bytes)) as *const u8;

        let trampoline = self
            .functions
            .iter()
            .find(|f| f.name == "trampoline")
            .unwrap();
        let trampoline_jump_table_offset = trampoline.jump_table_offset;
        let trampoline_offset = &self.jump_table.as_ref().unwrap()
            [trampoline_jump_table_offset * 8..trampoline_jump_table_offset * 8 + 8];

        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(trampoline_offset);
        let trampoline_start = BuiltInTypes::untag(usize::from_le_bytes(bytes)) as *const u8;

        let f: fn(u64, u64) -> u64 = unsafe { std::mem::transmute(trampoline_start) };
        let stack_pointer = self.get_stack_pointer();
        let result = f(stack_pointer as u64, start as u64);
        Ok(result)
    }

    pub fn run1(&self, jump_table_offset: usize, arg: u64) -> Result<u64, Box<dyn Error>> {
        // get offset stored in jump table as a usize
        let offset =
            &self.jump_table.as_ref().unwrap()[jump_table_offset * 8..jump_table_offset * 8 + 8];
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(offset);
        let start = BuiltInTypes::untag(usize::from_le_bytes(bytes)) as *const u8;

        let trampoline = self
            .functions
            .iter()
            .find(|f| f.name == "trampoline")
            .unwrap();
        let trampoline_jump_table_offset = trampoline.jump_table_offset;
        let trampoline_offset = &self.jump_table.as_ref().unwrap()
            [trampoline_jump_table_offset * 8..trampoline_jump_table_offset * 8 + 8];

        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(trampoline_offset);
        let trampoline_start = BuiltInTypes::untag(usize::from_le_bytes(bytes)) as *const u8;

        let f: fn(u64, u64, u64) -> u64 = unsafe { std::mem::transmute(trampoline_start) };
        let arg = BuiltInTypes::Int.tag(arg as isize) as u64;
        let stack_pointer = self.get_stack_pointer();
        let result = f(stack_pointer as u64, start as u64, arg);
        Ok(result)
    }

    pub fn run2(
        &self,
        _jump_table_offset: usize,
        _arg1: u64,
        _arg2: u64,
    ) -> Result<u64, Box<dyn Error>> {
        unimplemented!("Add trampoline");
    }

    pub fn get_repr(&self, value: usize, depth: usize) -> Option<String> {
        if depth > 1000 {
            return Some("...".to_string());
        }
        let tag = BuiltInTypes::get_kind(value);
        match tag {
            BuiltInTypes::Null => Some("null".to_string()),
            BuiltInTypes::Int => {
                let value = BuiltInTypes::untag(value);
                Some(value.to_string())
            }
            BuiltInTypes::Float => todo!(),
            BuiltInTypes::String => {
                let value = BuiltInTypes::untag(value);
                let string = &self.string_constants[value];
                Some(string.str.clone())
            }
            BuiltInTypes::Bool => {
                let value = BuiltInTypes::untag(value);
                if value == 0 {
                    Some("false".to_string())
                } else {
                    Some("true".to_string())
                }
            }
            BuiltInTypes::Function => Some("function".to_string()),
            BuiltInTypes::Closure => todo!(),
            BuiltInTypes::Struct => {
                unsafe {
                    let value = BuiltInTypes::untag(value);
                    let pointer = value as *const u8;

                    if pointer as usize % 8 != 0 {
                        panic!("Not aligned");
                    }
                    // get first 8 bytes as size le encoded
                    let size = *(pointer as *const usize) >> 1;
                    let pointer = pointer.add(8);
                    let data = std::slice::from_raw_parts(pointer, size);
                    // type id is the first 8 bytes of data
                    let type_id = usize::from_le_bytes(data[0..8].try_into().unwrap());
                    let type_id = BuiltInTypes::untag(type_id);
                    let struct_value = self.structs.get_by_id(type_id as usize);
                    Some(self.get_struct_repr(struct_value?, data[8..].to_vec(), depth + 1)?)
                }
            }
            BuiltInTypes::Array => {
                unsafe {
                    let value = BuiltInTypes::untag(value);
                    let pointer = value as *const u8;
                    // get first 8 bytes as size le encoded
                    let size = *(pointer as *const usize) >> 1;
                    let pointer = pointer.add(8);
                    let data = std::slice::from_raw_parts(pointer, size);
                    let heap_object = HeapObject { size, data };

                    let mut repr = "[".to_string();
                    for i in 0..heap_object.size {
                        let value = &heap_object.data[i * 8..i * 8 + 8];
                        let mut bytes = [0u8; 8];
                        bytes.copy_from_slice(value);
                        let value = usize::from_le_bytes(bytes);
                        repr.push_str(&self.get_repr(value, depth + 1)?);
                        if i != heap_object.size - 1 {
                            repr.push_str(", ");
                        }
                    }
                    repr.push(']');
                    Some(repr)
                }
            }
        }
    }

    pub fn compile(&mut self, code: String) -> Result<(), Box<dyn Error>> {
        let mut parser = Parser::new(code);
        let ast = parser.parse();
        self.compile_ast(ast)
    }

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

                self.run1(function.jump_table_offset, vec[0] as u64)
                    .unwrap()
            }
            2 => {
                let function = self.functions.iter().find(|f| f.name == arg).unwrap();
                self.run2(function.jump_table_offset, vec[0] as u64, vec[1] as u64)
                    .unwrap()
            }
            _ => panic!("Too many arguments"),
        }
    }

    pub fn add_string(&mut self, string_value: StringValue) -> Value {
        self.string_constants.push(string_value);
        let offset = self.string_constants.len() - 1;
        Value::StringConstantPtr(offset)
    }

    pub(crate) fn find_function(&self, name: &str) -> Option<Function> {
        self.functions.iter().find(|f| f.name == name).cloned()
    }

    pub fn get_function_by_pointer(&self, value: usize) -> Option<&Function> {
        let offset = value.saturating_sub(self.code_memory.as_ref().unwrap().as_ptr() as usize);
        self.functions
            .iter()
            .find(|f| f.offset_or_pointer == offset)
    }

    pub fn get_function_by_pointer_mut(&mut self, value: usize) -> Option<&mut Function> {
        let offset = value - self.code_memory.as_ref().unwrap().as_ptr() as usize;
        self.functions
            .iter_mut()
            .find(|f| f.offset_or_pointer == offset)
    }

    pub fn make_closure(
        &mut self,
        function: usize,
        free_variables: &[usize],
    ) -> Result<usize, Box<dyn Error>> {
        let len = 8 + 8 + 8 + free_variables.len() * 8;
        // TODO: Stack pointer should be passed in
        let heap_pointer = self.allocate(len, 0, BuiltInTypes::Closure)?;
        let pointer = heap_pointer as *mut u8;
        let num_free = free_variables.len();
        let function_definition =  self.get_function_by_pointer(BuiltInTypes::untag(function));
        if function_definition.is_none() {
            panic!("Function not found");
        }
        let function_definition = function_definition.unwrap();
        let num_locals = function_definition.number_of_locals;

        let function = function.to_le_bytes();

        let free_variables = free_variables.iter().flat_map(|v| v.to_le_bytes());
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

        let num_locals = num_locals.to_le_bytes();
        // Write number of locals
        for (index, byte) in num_locals.iter().enumerate() {
            buffer[16 + index] = *byte;
        }
        // write free variables
        for (index, byte) in free_variables.enumerate() {
            buffer[24 + index] = byte;
        }
        Ok(BuiltInTypes::Closure.tag(heap_pointer as isize) as usize)
    }

    pub fn property_access(&self, struct_pointer: usize, str_constant_ptr: usize) -> usize {
        unsafe {
            if BuiltInTypes::untag(struct_pointer) as usize % 8 != 0 {
                panic!("Not aligned");
            }
            let struct_pointer = BuiltInTypes::untag(struct_pointer);
            let struct_pointer = struct_pointer as *const u8;
            let size = *(struct_pointer as *const usize) >> 1;
            let str_constant_ptr: usize = BuiltInTypes::untag(str_constant_ptr);
            let string_value = &self.string_constants[str_constant_ptr];
            let string = &string_value.str;
            let struct_pointer = struct_pointer.add(8);
            let data = std::slice::from_raw_parts(struct_pointer, size);
            // type id is the first 8 bytes of data
            let type_id = usize::from_le_bytes(data[0..8].try_into().unwrap());
            let type_id = BuiltInTypes::untag(type_id);
            let struct_value = self.structs.get_by_id(type_id as usize).unwrap();
            let field_index = struct_value
                .fields
                .iter()
                .position(|f| f == string)
                .unwrap();
            let field_index = (field_index + 1) * 8;
            let field = &data[field_index..field_index + 8];
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(field);

            usize::from_le_bytes(bytes)
        }
    }

    pub fn add_struct(&mut self, s: Struct) {
        self.structs.insert(s.name.clone(), s);
    }

    pub fn get_struct(&self, name: &str) -> Option<(usize, &Struct)> {
        self.structs.get(name)
    }

    fn get_struct_repr(
        &self,
        struct_value: &Struct,
        to_vec: Vec<u8>,
        depth: usize,
    ) -> Option<String> {
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
            repr.push_str(&self.get_repr(value, depth + 1)?);
            if index != struct_value.fields.len() - 1 {
                repr.push_str(", ");
            }
        }
        repr.push_str(" }");
        Some(repr)
    }

    pub fn check_functions(&self) {
        let undefined_functions: Vec<&Function> =
            self.functions.iter().filter(|f| !f.is_defined).collect();
        if !undefined_functions.is_empty() {
            panic!(
                "Undefined functions: {:?}",
                undefined_functions
                    .iter()
                    .map(|f| f.name.clone())
                    .collect::<Vec<String>>()
            );
        }
    }

    pub fn get_function_by_name_mut(&mut self, name: &str) -> Option<&mut Function> {
        self.functions.iter_mut().find(|f| f.name == name)
    }

    pub fn print(&mut self, result: usize) {
        let result = self.get_repr(result, 0).unwrap();
        self.printer.println(result);
    }
    
    pub fn println(&mut self, result: usize) {
        let result = self.get_repr(result, 0).unwrap();
        self.printer.println(result);
    }
}
