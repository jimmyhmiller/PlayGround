use core::fmt;
use std::{
    collections::HashMap,
    error::Error,
    mem,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use bincode::{Decode, Encode};
use mmap_rs::{Mmap, MmapMut, MmapOptions};

use crate::{
    arm::LowLevelArm,
    ast::Ast,
    debugger,
    ir::{println_value, BuiltInTypes, StringValue, Value},
    parser::Parser,
    Data, Message,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    name: String,
    offset_or_pointer: usize,
    jump_table_offset: usize,
    is_foreign: bool,
    pub is_builtin: bool,
    is_defined: bool,
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

struct Heap {
    segments: Vec<Option<MmapMut>>,
    heap_offset: usize,
    segment_offset: usize,
    segment_size: usize,
}

impl Heap {
    fn new() -> Self {
        let segment_size = MmapOptions::page_size() * 10;
        Self {
            segments: vec![Some(
                MmapOptions::new(segment_size)
                    .unwrap()
                    .map()
                    .unwrap()
                    .make_mut()
                    .unwrap_or_else(|(_map, e)| {
                        panic!("Failed to make mmap executable: {}", e);
                    }),
            )],
            heap_offset: 0,
            segment_offset: 0,
            segment_size,
        }
    }

    fn segment_pointer(&self, arg: usize) -> usize {
        let segment = self.segments.get(arg as usize).unwrap();
        segment.as_ref().unwrap().as_ptr() as usize
    }

    // TODO: I need garbage collection now
    // In order to do that I need to know what is on the stack
    // and I need to grab things from registers
    // Once I have those, I can walk references and mark things.
    // Then I could try moving things
    // Or I could keep a list of free pages or something.

    fn allocate(&mut self, bytes: usize, stack_pointer: usize) -> Result<usize, Box<dyn Error>> {
        if self.heap_offset + bytes + 8 > self.segment_size {
            self.segment_offset += 1;
            self.segments.push(Some(
                MmapOptions::new(self.segment_size)
                    .unwrap()
                    .map()
                    .unwrap()
                    .make_mut()
                    .unwrap_or_else(|(_map, e)| {
                        panic!("Failed to make mmap executable: {}", e);
                    }),
            ));
            let segment_pointer = self.segment_pointer(self.segment_offset);
            debugger(Message {
                kind: "HeapSegmentPointer".to_string(),
                data: Data::HeapSegmentPointer {
                    pointer: segment_pointer,
                },
            });
            self.heap_offset = 0;
        }

        let size = (bytes * 8) << 1;
        let memory = mem::replace(&mut self.segments[self.segment_offset], None);
        let mut memory = memory.unwrap();
        let buffer = &mut memory[self.heap_offset..];
        // write the size of the object to the first 8 bytes
        for (index, byte) in size.to_le_bytes().iter().enumerate() {
            buffer[index] = *byte;
        }
        self.heap_offset += size + 8;
        // need to make sure my offset is aligned
        // going to round up to the nearest 8 bytes
        let remainder = self.heap_offset % 8;
        if remainder != 0 {
            self.heap_offset += 8 - remainder;
        }
        assert!(self.heap_offset % 8 == 0, "Heap offset is not aligned");

        let pointer = buffer.as_ptr() as usize;
        self.segments[self.segment_offset] = Some(memory);
        Ok(pointer)
    }
}

#[derive(Debug, Encode, Decode, Clone)]
pub struct StackMapDetails {
    pub number_of_locals: usize,
    pub current_stack_size: usize,
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
    heap: Heap,
    stack: Option<MmapMut>,
    string_constants: Vec<StringValue>,
    stack_map: Vec<(usize, StackMapDetails)>,
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
            heap: Heap::new(),
            stack: Some(
                MmapOptions::new(8 * 1024 * 1024)
                    .unwrap()
                    .map_mut()
                    .unwrap(),
            ),
            string_constants: vec![],
            stack_map: vec![],
        }
    }

    pub fn mark(&mut self, latest_root: usize, current_stack_pointer: usize) -> Vec<usize> {
        let mut marked = 0;

        // TODO: Only go up to stack pointer

        // Get the stack as a [usize]
        let stack = self.stack.as_ref().unwrap();
        // I'm adding to the end of the stack I've allocated so I only need to go from the end
        // til the current stack

        let stack_end = stack.as_ptr() as usize + (8 * 1024 * 1024);
        // let current_stack_pointer = current_stack_pointer & !0b111;
        let distance_till_end = stack_end - current_stack_pointer as usize;
        let num_64_till_end = distance_till_end / 8;
        let stack = unsafe {
            std::slice::from_raw_parts(stack.as_ptr() as *const usize, 8 * 1024 * 1024 / 8)
        };
        let stack = &stack[stack.len() - num_64_till_end..];
        let mut roots = Vec::new();
        // Walk the stack

        let mut to_mark: Vec<usize> = vec![];

        if BuiltInTypes::is_heap_pointer(latest_root) {
            unsafe {
                let untagged = BuiltInTypes::untag(latest_root);
                let pointer = untagged as *mut u8;
                if pointer as usize % 8 != 0 {
                    panic!("Not aligned");
                }

                marked += 1;

                let mut data: usize = *pointer.cast::<usize>();
                data |= 1;
                *pointer.cast::<usize>() = data;

                let size = (*(pointer as *const usize) >> 1);
                // Why would it be 16?
                let data = std::slice::from_raw_parts(pointer.add(8) as *const usize, size / 8);
                for datum in data.iter() {
                    if BuiltInTypes::is_heap_pointer(*datum) {
                        to_mark.push(*datum)
                    }
                }
            }
            roots.push(latest_root);
        }

        // TODO: It seems I might be putting untagged pointers on the stack

        for value in stack {
            if BuiltInTypes::is_heap_pointer(*value) {
                unsafe {
                    let untagged = BuiltInTypes::untag(*value);
                    println!(
                        "builtintypes: {:?} {} {}",
                        BuiltInTypes::get_kind(*value),
                        value,
                        untagged
                    );
                    let pointer = untagged as *mut u8;
                    if pointer as usize % 8 != 0 {
                        panic!("Not aligned");
                    }

                    marked += 1;

                    let mut data: usize = *pointer.cast::<usize>();
                    data |= 1;
                    *pointer.cast::<usize>() = data;

                    let size = (*(pointer as *const usize) >> 1);
                    // Why would it be 16?
                    let data = std::slice::from_raw_parts(pointer.add(8) as *const usize, size / 8);
                    for datum in data.iter() {
                        if BuiltInTypes::is_heap_pointer(*datum) {
                            to_mark.push(*datum)
                        }
                    }
                }
                roots.push(*value);
            }
        }

        while !to_mark.is_empty() {
            let value = to_mark.pop().unwrap();
            let untagged = BuiltInTypes::untag(value);
            let pointer = untagged as *mut u8;
            if pointer as usize % 8 != 0 {
                panic!("Not aligned");
            }
            unsafe {
                let mut data: usize = *pointer.cast::<usize>();
                // check right most bit
                if (data & 1) == 1 {
                    continue;
                }
                data |= 1;
                *pointer.cast::<usize>() = data;

                marked += 1;

                let size = (*(pointer as *const usize) >> 1);
                // Why would it be 16?
                let data = std::slice::from_raw_parts(pointer.add(8) as *const usize, (size / 8));
                for datum in data.iter() {
                    if BuiltInTypes::is_heap_pointer(*datum) {
                        to_mark.push(*datum)
                    }
                }
            }
        }

        self.sweep();

        println!("marked {}", marked);

        roots
    }

    fn sweep(&mut self) {
        for segment in self.heap.segments.iter_mut() {
            if let Some(segment) = segment {
                let mut offset = 0;
                // TODO: I'm scanning whole segment even if unused
                while offset < self.heap.segment_size {
                    unsafe {
                        let mut pointer = segment.as_ptr();

                        let mut data: usize = *pointer.cast::<usize>();

                        // check right most bit
                        if (data & 1) == 1 {
                            // println!("marked!");
                            data &= !1;
                        }
                        let size = (data >> 1) - 8;
                        // println!("size: {}", size);
                        offset += size;
                        let remainder = offset % 8;
                        if remainder != 0 {
                            offset += 8 - remainder;
                        }
                    }
                }
            }
        }
    }

    pub fn get_heap_pointer(&self) -> usize {
        // TODO: I need to tell my debugger about all the heap pointers
        self.heap.segment_pointer(0)
    }

    pub fn allocate(
        &mut self,
        bytes: usize,
        stack_pointer: usize,
        kind: BuiltInTypes,
    ) -> Result<usize, Box<dyn Error>> {
        let segment = self.heap.segment_offset;
        let result = self.heap.allocate(bytes, stack_pointer).unwrap(); // TODO: do better
                                                                        // if segment != self.heap.segment_offset {
        // self.mark(kind.tag(result.clone() as isize) as usize, stack_pointer);
        // }
        Ok(result)
    }

    fn get_stack_pointer(&self) -> usize {
        // I think I want the end of the stack
        (self.stack.as_ref().unwrap().as_ptr() as usize) + (8 * 1024 * 1024)
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
    ) -> Result<usize, Box<dyn Error>> {
        let bytes = &(code.compile_to_bytes());
        for (index, function) in self.functions.iter_mut().enumerate() {
            if function.name == name {
                self.overwrite_function(index, bytes)?;
                break;
            }
        }
        self.add_function(name, bytes)?;

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
        };
        self.functions.push(function.clone());
        Ok(function)
    }

    pub fn add_function(&mut self, name: &str, code: &[u8]) -> Result<usize, Box<dyn Error>> {
        let offset = self.add_code(code)?;
        let index = self.functions.len();
        self.functions.push(Function {
            name: name.to_string(),
            offset_or_pointer: offset,
            jump_table_offset: 0,
            is_foreign: false,
            is_builtin: false,
            is_defined: true,
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
        index: usize,
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
        jump_table_offset: usize,
        arg1: u64,
        arg2: u64,
    ) -> Result<u64, Box<dyn Error>> {
        unimplemented!("Add trampoline");
        // get offset stored in jump table as a usize
        let offset =
            &self.jump_table.as_ref().unwrap()[jump_table_offset * 8..jump_table_offset * 8 + 8];
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(offset);
        let start = BuiltInTypes::untag(usize::from_le_bytes(bytes));
        let arg1 = BuiltInTypes::Int.tag(arg1 as isize) as u64;
        let arg2 = BuiltInTypes::Int.tag(arg2 as isize) as u64;
        let f: fn(u64, u64) -> u64 = unsafe { std::mem::transmute(start) };
        Ok(f(arg1, arg2))
    }

    pub fn get_repr(&self, value: usize) -> String {
        let tag = BuiltInTypes::get_kind(value);
        match tag {
            BuiltInTypes::Null => "null".to_string(),
            BuiltInTypes::Int => {
                let value = BuiltInTypes::untag(value);
                value.to_string()
            }
            BuiltInTypes::Float => todo!(),
            BuiltInTypes::String => {
                let value = BuiltInTypes::untag(value);
                let string = &self.string_constants[value];
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

    pub fn println(&self, value: usize) {
        println!("{}", self.get_repr(value));
    }

    pub fn print(&self, value: usize) {
        print!("{}", self.get_repr(value));
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
                Ast::Function { .. } => {
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
                let result = self
                    .run1(function.jump_table_offset, vec[0] as u64)
                    .unwrap();
                result
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
        return Value::StringConstantPtr(offset);
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

    pub fn make_closure(
        &mut self,
        function: usize,
        free_variables: &[usize],
    ) -> Result<usize, Box<dyn Error>> {
        let len = 8 + 8 + free_variables.len() * 8;
        // TODO: Stack pointer should be passed in
        let heap_pointer = self.allocate(len, 0, BuiltInTypes::Closure)?;
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

    pub fn property_access(&self, struct_pointer: usize, str_constant_ptr: usize) -> usize {
        unsafe {
            let struct_pointer = BuiltInTypes::untag(struct_pointer);
            let struct_pointer = struct_pointer as *const u8;
            let size = *(struct_pointer as *const usize);
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
            let value = usize::from_le_bytes(bytes);
            value
        }
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
}
