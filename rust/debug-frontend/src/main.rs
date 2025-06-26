use std::{
    collections::{HashMap, HashSet},
    env,
    ffi::{c_void, CString},
};

use bincode::{config::standard, Decode, Encode};
use lldb::{
    SBBreakpoint, SBDebugger, SBError, SBFrame, SBInstructionList, SBLaunchInfo, SBProcess,
    SBTarget, SBValue,
};
use lldb_sys::{RunMode, SBTargetBreakpointCreateByName, SBValueGetValueAsUnsigned2};
use mapping::BreakpointMapper;
use regex::Regex;
use skia_safe::{
    paint,
    textlayout::{FontCollection, ParagraphBuilder, ParagraphStyle, TextStyle},
    FontMgr,
};
use skia_window::{App, Options};
use winit::{
    event::{ElementState, Event, WindowEvent::KeyboardInput},
    event_loop::EventLoopProxy,
    keyboard::{KeyCode, PhysicalKey},
    window::CursorIcon,
};

mod mapping;
mod websocket;

#[derive(Debug, Clone, Encode, Decode)]
pub struct Message {
    kind: String,
    data: Data,
}

#[allow(unused)]
impl Message {
    fn to_string(&self) -> String {
        format!("{} {}", self.kind, self.data.to_string())
    }
}

#[derive(Debug, Encode, Decode, Clone)]
pub struct StackMapDetails {
    pub function_name: Option<String>,
    pub number_of_locals: usize,
    pub current_stack_size: usize,
    pub max_stack_size: usize,
}

// TODO: This should really live on the debugger side of things
#[derive(Debug, Encode, Decode, Clone)]
enum Data {
    ForeignFunction {
        name: String,
        pointer: usize,
    },
    BuiltinFunction {
        name: String,
        pointer: usize,
    },
    HeapSegmentPointer {
        pointer: usize,
    },
    UserFunction {
        name: String,
        pointer: usize,
        len: usize,
        number_of_arguments: usize,
    },
    Label {
        label: String,
        function_pointer: usize,
        label_index: usize,
        label_location: usize,
    },
    StackMap {
        pc: usize,
        name: String,
        stack_map: Vec<(usize, StackMapDetails)>,
    },
    Allocate {
        bytes: usize,
        stack_pointer: usize,
        kind: String,
    },
    Tokens {
        file_name: String,
        tokens: Vec<String>,
        token_line_column_map: Vec<(usize, usize)>,
    },
    Ir {
        function_pointer: usize,
        file_name: String,
        instructions: Vec<String>,
        token_range_to_ir_range: Vec<((usize, usize), (usize, usize))>,
    },
    Arm {
        function_pointer: usize,
        file_name: String,
        instructions: Vec<String>,
        ir_to_machine_code_range: Vec<(usize, (usize, usize))>,
    },
}

#[derive(Debug, Encode, Decode, Clone)]
struct Label {
    label: String,
    function_pointer: usize,
    label_index: usize,
    label_location: usize,
}

impl Data {
    fn to_string(&self) -> String {
        match self {
            Data::ForeignFunction { name, pointer } => {
                format!("{}: 0x{:x}", name, pointer)
            }
            Data::BuiltinFunction { name, pointer } => {
                format!("{}: 0x{:x}", name, pointer)
            }
            Data::HeapSegmentPointer { pointer } => {
                format!("0x{:x}", pointer)
            }
            Data::UserFunction {
                name,
                pointer,
                len,
                number_of_arguments: _,
            } => {
                format!("{}: 0x{:x} 0x{:x}", name, pointer, (pointer + len))
            }
            Data::Label {
                label,
                function_pointer,
                label_index,
                label_location,
            } => {
                format!(
                    "{}: 0x{:x} 0x{:x} 0x{:x}",
                    label, function_pointer, label_index, label_location
                )
            }
            Data::StackMap {
                pc,
                name,
                stack_map,
            } => {
                let stack_map_details_string = stack_map
                    .iter()
                    .map(|(key, details)| {
                        format!(
                            "0x{:x}: size: {}, locals: {}",
                            key, details.current_stack_size, details.number_of_locals
                        )
                    })
                    .collect::<Vec<String>>()
                    .join(" | ");
                format!("{}, 0x{:x}, {}", name, pc, stack_map_details_string)
            }
            Data::Allocate {
                bytes,
                stack_pointer,
                kind,
            } => {
                format!("{}: {} 0x{:x}", kind, bytes, stack_pointer)
            }
            Data::Tokens {
                file_name,
                tokens,
                token_line_column_map,
            } => {
                let tokens = tokens.join(" ");
                let token_line_column_map = token_line_column_map
                    .iter()
                    .map(|(line, column)| format!("{}:{}", line, column))
                    .collect::<Vec<String>>()
                    .join(" ");
                format!("{}: {} {}", file_name, tokens, token_line_column_map)
            }
            Data::Ir {
                function_pointer: _,
                file_name,
                instructions,
                token_range_to_ir_range,
            } => {
                let instructions = instructions.join(" ");
                let token_range_to_ir_range = token_range_to_ir_range
                    .iter()
                    .map(|((start, end), (start_ir, end_ir))| {
                        format!("{}-{}:{}-{}", start, end, start_ir, end_ir)
                    })
                    .collect::<Vec<String>>()
                    .join(" ");
                format!(
                    "{}: {} {}",
                    file_name, instructions, token_range_to_ir_range
                )
            }
            Data::Arm {
                function_pointer: _,
                file_name,
                instructions,
                ir_to_machine_code_range,
            } => {
                let instructions = instructions.join(" ");
                let ir_to_machine_code_range = ir_to_machine_code_range
                    .iter()
                    .map(|(ir, (start, end))| format!("{}:{}-{}", ir, start, end))
                    .collect::<Vec<String>>()
                    .join(" ");
                format!(
                    "{}: {} {}",
                    file_name, instructions, ir_to_machine_code_range
                )
            }
        }
    }
}

trait Serialize {
    fn to_binary(&self) -> Vec<u8>;
    fn from_binary(data: &[u8]) -> Self;
}

impl<T: Encode + Decode> Serialize for T {
    fn to_binary(&self) -> Vec<u8> {
        bincode::encode_to_vec(self, standard()).unwrap()
    }
    fn from_binary(data: &[u8]) -> T {
        let (data, _) = bincode::decode_from_slice(data, standard()).unwrap();
        data
    }
}

struct Frontend {
    state: State,
    should_step: bool,
    should_step_hyper: bool,
    is_continuing: bool,
}

#[allow(unused_variables)]
impl App for Frontend {
    fn on_window_create(&mut self, event_loop_proxy: EventLoopProxy<()>, size: skia_window::Size) {
        if let Some((target, process)) = start_process(&self.state.beagle_program_path) {
            self.state.process.process = Some(process);
            self.state.process.target = Some(target);
        }
    }

    fn add_event(&mut self, event: &Event<()>) -> bool {
        // TODO: This is super gross
        match event {
            Event::WindowEvent { window_id, event } => match event {
                KeyboardInput { event, .. } => match event.state {
                    ElementState::Pressed => match event.physical_key {
                        PhysicalKey::Code(KeyCode::ArrowDown) => {
                            self.is_continuing = false;
                            self.step_over();
                        }
                        PhysicalKey::Code(KeyCode::KeyI) => {
                            self.is_continuing = false;
                            self.step_in();
                        }
                        PhysicalKey::Code(KeyCode::ArrowRight) => {
                            self.is_continuing = false;
                            if self.should_step {
                                self.should_step_hyper = true;
                            } else {
                                self.keep_stepping();
                            }
                        }
                        PhysicalKey::Code(KeyCode::ArrowLeft) => {
                            self.is_continuing = false;
                            if self.should_step_hyper {
                                self.should_step_hyper = false;
                            } else {
                                self.stop_stepping();
                            }
                        }
                        PhysicalKey::Code(KeyCode::ArrowUp) => {
                            self.is_continuing = true;
                            self.should_step = false;
                            self.should_step_hyper = false;
                            self.step_over();
                            self.state
                                .process
                                .process
                                .as_mut()
                                .unwrap()
                                .continue_execution()
                                .unwrap();
                        }
                        PhysicalKey::Code(KeyCode::KeyR) => {
                            if let Some(process) = &self.state.process.process {
                                process.kill().unwrap();
                                let beagle_program_path = self.state.beagle_program_path.clone();
                                let function_regex = self.state.function_regex.clone();
                                self.state = get_initial_state(self.state.websocket_sender.clone(), beagle_program_path.clone(), function_regex.clone());
                                if let Some((target, process)) = start_process(&beagle_program_path) {
                                    self.state.process.process = Some(process);
                                    self.state.process.target = Some(target);
                                }
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                },
                _ => {}
            },
            _ => {}
        }
        true
    }

    fn exit(&mut self) {}

    fn draw(&mut self, canvas: &skia_safe::Canvas) {
        canvas.clear(skia_safe::Color::WHITE);
        canvas.translate((100.0, 100.0));
        let debug = debug(&self.state);
        debug.draw(canvas);
    }

    fn end_frame(&mut self) {}

    fn tick(&mut self) {
        let process = &self.state.process.process;
        if process.is_some() {
            let process = process.as_ref().unwrap();
            if let Some(out) = process.get_stdout_all() {
                if !out.is_empty() {
                    println!("{:?}", out);
                }
            }
        }

        let start = std::time::Instant::now();
        let ms_threshold = 10;
        if self.should_step {
            self.step_over();
        }
        if self.should_step_hyper {
            let mut i = 0;
            loop {
                let elapsed = start.elapsed().as_millis();
                if elapsed > ms_threshold {
                    break;
                }
                self.step_over();
                i += 1;
            }
        }
        if self.is_continuing {
            self.state.update_process_state(false);
        }
        if self.state.process.instructions.is_some() {
            return;
        }
        self.state.update_process_state(false);
    }

    fn should_redraw(&mut self) -> bool {
        true
    }

    fn cursor_icon(&mut self) -> CursorIcon {
        CursorIcon::Default
    }

    fn set_window_size(&mut self, size: skia_window::Size) {}
}

impl Frontend {
    fn step(&mut self, force_in: bool) {
        if let Some(process) = self.state.process.process.clone() {
            if let Some(thread) = process.thread_by_index_id(1) {
                let _was_debugger_info = self.state.check_debugger_info(&thread, &process, true);
                let pc_before = thread.selected_frame().pc();

                let current_instruction = self
                    .state
                    .disasm
                    .disasm_values
                    .iter()
                    .find(|x| x.address == pc_before);
                let step_over = if current_instruction
                    .map(|x| format!("{} {}", x.instruction, x.arguments.join(" ")))
                    == Some("mov x29 sp".to_string())
                {
                    false
                } else {
                    true
                };
                let step_over = if force_in { false } else { step_over };
                thread.step_instruction(step_over).unwrap();
                let pc_after = thread.selected_frame().pc();
                if pc_before == pc_after {
                    thread.selected_frame().set_pc(pc_after + 4);
                }
                self.state.update_process_state(true);
            }
        }
    }

    fn step_over(&mut self) {
        self.step(false)
    }

    fn step_in(&mut self) {
        self.step(true);
    }

    fn keep_stepping(&mut self) {
        self.should_step = true;
    }

    fn stop_stepping(&mut self) {
        self.should_step = false;
    }
}

fn get_initial_state(websocket_sender: std::sync::mpsc::Sender<Message>, beagle_program_path: String, function_regex: Regex) -> State {
    State {
        disasm: Disasm {
            disasm_values: vec![],
            show_address: true,
            show_hex: false,
        },
        registers: Registers {
            registers: vec![
                Register {
                    name: "x0".to_string(),
                    value: Value::Integer(0),
                    kind: BuiltInTypes::Int,
                },
                Register {
                    name: "x1".to_string(),
                    value: Value::Integer(1),
                    kind: BuiltInTypes::Int,
                },
            ],
        },
        process: Process {
            process: None,
            target: None,
            instructions: None,
        },
        pc: 0,
        sp: 0,
        fp: 0,
        stack: Stack { stack: vec![] },
        messages: vec![],
        functions: HashMap::new(),
        labels: HashMap::new(),
        heap: Heap {
            memory: vec![],
            heap_pointers: vec![],
        },
        websocket_sender,
        breakpoint_mapper: BreakpointMapper::new(),
        beagle_program_path,
        function_regex,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <beagle_program_path> [function_regex]", args[0]);
        std::process::exit(1);
    }

    let beagle_program_path = if args[1].starts_with('/') {
        args[1].clone()
    } else {
        format!("/Users/jimmyhmiller/Documents/Code/beagle/resources/{}", args[1])
    };

    // Optional function name regex (default: match all)
    let function_pattern = if args.len() > 2 {
        args[2].clone()
    } else {
        ".*".to_string()
    };
    let function_regex = Regex::new(&function_pattern).unwrap();

    let (websocket_sender, _wait_for_client) = websocket::start_websocket_thread().unwrap();
    
    let mut frontend = Frontend {
        should_step: false,
        should_step_hyper: false,
        is_continuing: false,
        state: get_initial_state(websocket_sender, beagle_program_path, function_regex.clone()),
    };
    frontend.create_window(
        "Debug",
        Options {
            vsync: false,
            width: 2400,
            height: 1800,
            title: "Debugger".to_string(),
            position: (0, 0),
        },
    );
}

// Stolen from compiler
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BuiltInTypes {
    Int,
    Float,
    String,
    Bool,
    Function,
    Closure,
    Struct,
    Array,
    Null,
    None,
}

impl BuiltInTypes {
    pub fn null_value() -> isize {
        0b111
    }

    pub fn tag(&self, value: isize) -> isize {
        let value = value << 3;
        let tag = self.get_tag();
        value | tag
    }

    // TODO: Given this scheme how do I represent null?
    pub fn get_tag(&self) -> isize {
        match self {
            BuiltInTypes::Int => 0b000,
            BuiltInTypes::Float => 0b001,
            BuiltInTypes::String => 0b010,
            BuiltInTypes::Bool => 0b011,
            BuiltInTypes::Function => 0b100,
            BuiltInTypes::Closure => 0b101,
            BuiltInTypes::Struct => 0b110,
            BuiltInTypes::Array => 0b111,
            BuiltInTypes::Null => 0b111,
            BuiltInTypes::None => 0,
        }
    }

    pub fn untag(pointer: usize) -> usize {
        pointer >> 3
    }

    pub fn get_kind(pointer: usize) -> Self {
        if pointer == Self::null_value() as usize {
            return BuiltInTypes::Null;
        }
        match pointer & 0b111 {
            0b000 => BuiltInTypes::Int,
            0b001 => BuiltInTypes::Float,
            0b010 => BuiltInTypes::String,
            0b011 => BuiltInTypes::Bool,
            0b100 => BuiltInTypes::Function,
            0b101 => BuiltInTypes::Closure,
            0b110 => BuiltInTypes::Struct,
            0b111 => BuiltInTypes::Array,
            _ => panic!("Invalid tag"),
        }
    }

    pub fn is_embedded(&self) -> bool {
        match self {
            BuiltInTypes::Int => true,
            BuiltInTypes::Float => true,
            BuiltInTypes::String => false,
            BuiltInTypes::Bool => true,
            BuiltInTypes::Function => false,
            BuiltInTypes::Struct => false,
            BuiltInTypes::Array => false,
            BuiltInTypes::Closure => false,
            BuiltInTypes::Null => true,
            BuiltInTypes::None => false,
        }
    }

    pub fn construct_int(value: isize) -> isize {
        if value > isize::MAX >> 3 {
            panic!("Integer overflow")
        }
        BuiltInTypes::Int.tag(value)
    }

    pub fn construct_boolean(value: bool) -> isize {
        let bool = BuiltInTypes::Bool;
        if value {
            bool.tag(1)
        } else {
            bool.tag(0)
        }
    }

    pub fn tag_size() -> i32 {
        3
    }

    pub fn is_heap_pointer(value: usize) -> bool {
        match BuiltInTypes::get_kind(value) {
            BuiltInTypes::Int => false,
            BuiltInTypes::Float => false,
            BuiltInTypes::String => false,
            BuiltInTypes::Bool => false,
            BuiltInTypes::Function => false,
            BuiltInTypes::Closure => true,
            BuiltInTypes::Struct => true,
            BuiltInTypes::Array => true,
            BuiltInTypes::Null => false,
            BuiltInTypes::None => false,
        }
    }
    pub fn to_string(&self) -> String {
        match self {
            BuiltInTypes::Int => "Int".to_string(),
            BuiltInTypes::Float => "Float".to_string(),
            BuiltInTypes::String => "String".to_string(),
            BuiltInTypes::Bool => "Bool".to_string(),
            BuiltInTypes::Function => "Function".to_string(),
            BuiltInTypes::Closure => "Closure".to_string(),
            BuiltInTypes::Struct => "Struct".to_string(),
            BuiltInTypes::Array => "Array".to_string(),
            BuiltInTypes::Null => "Null".to_string(),
            BuiltInTypes::None => "None".to_string(),
        }
    }
}

struct Disasm {
    disasm_values: Vec<ParsedDisasm>,
    show_address: bool,
    show_hex: bool,
}

#[derive(Debug)]
enum Value {
    Integer(u64),
    String(String),
}

impl Value {
    fn to_string(&self) -> String {
        match self {
            Value::Integer(i) => format!("{}", i),
            Value::String(s) => s.to_string(),
        }
    }
}

#[derive(Debug)]
struct Register {
    name: String,
    value: Value,
    kind: BuiltInTypes,
}

impl Register {
    fn to_string(&self) -> String {
        format!(
            "{}: {} - {}",
            self.name,
            self.value.to_string(),
            self.kind.to_string()
        )
    }
}

struct Registers {
    registers: Vec<Register>,
}

struct Process {
    process: Option<SBProcess>,
    target: Option<SBTarget>,
    instructions: Option<SBInstructionList>,
}

fn convert_to_u64_array(input: &[u8; 512]) -> Vec<u64> {
    input
        .chunks(8)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0, |acc, (i, &byte)| acc | (byte as u64) << (8 * i))
        })
        .collect()
}

impl State {
    fn update_process_state(&mut self, is_stepping: bool) {
        // TODO: remove this and let the websocket set the breakpoints
        if let (Some(process), Some(target)) =
            (self.process.process.clone(), self.process.target.clone())
        {
            if !process.is_stopped() {
                return;
            }

            loop {
                let mut all_false = true;
                for thread in process.threads() {
                    // println!("{:?}", thread.stop_reason());
                    let result = self.check_debugger_info(&thread, &process, is_stepping);
                    all_false = all_false && !result;
                }
                if all_false {
                    break;
                }
            }

            if let Some(thread) = process.thread_by_index_id(1) {
                let was_debugger_info = self.check_debugger_info(&thread, &process, is_stepping);

                if was_debugger_info {
                    return;
                }

                let frame = thread.selected_frame();

                self.pc = frame.pc();
                let pc_in_instructions = self
                    .disasm
                    .disasm_values
                    .iter()
                    .map(|v| v.address)
                    .any(|v| v == self.pc);

                // make sure the address 8 instructions ahead exists in disasm
                let n = 30;
                let n_ahead = self
                    .disasm
                    .disasm_values
                    .iter()
                    .find(|x| x.address == self.pc + n * 4)
                    .is_some();

                self.sp = frame.sp();
                self.fp = frame.fp();
                let stack_root = frame.sp() - 256;
                let stack = &mut [0u8; 512];
                process.read_memory(stack_root, stack);
                let stack = convert_to_u64_array(stack);
                self.stack.stack = stack
                    .iter()
                    .enumerate()
                    .map(|(i, value)| Memory {
                        address: stack_root + (i as u64 * 8),
                        value: *value,
                        kind: BuiltInTypes::get_kind(*value as usize),
                    })
                    .collect();

                // TODO: Handle more complex heap
                if let Some(heap_pointer) = self.heap.heap_pointers.get(0) {
                    let heap_root = *heap_pointer as u64;
                    let heap = &mut [0u8; 512];
                    process.read_memory(heap_root, heap);
                    let heap = convert_to_u64_array(heap);
                    self.heap.memory = heap
                        .iter()
                        .enumerate()
                        .map(|(i, value)| Memory {
                            address: heap_root + (i as u64 * 8),
                            value: *value,
                            kind: BuiltInTypes::get_kind(*value as usize),
                        })
                        .collect();
                }

                if !pc_in_instructions || !n_ahead {
                    let instructions = process.get_instructions(&frame, &target);
                    self.process_instructions(instructions, target);
                }
                self.disasm
                    .disasm_values
                    .sort_by(|a, b| a.address.cmp(&b.address));
                self.disasm
                    .disasm_values
                    .dedup_by(|a, b| a.address == b.address);
                self.registers.registers.clear();
                frame.registers().iter().for_each(|register| {
                    if let Some(name) = register.name() {
                        if name.contains("General") {
                            register.children().for_each(|child| {
                                if let Some(child_name) = child.name() {
                                    if child_name.contains("x")
                                        || child_name == "pc"
                                        || child_name == "sp"
                                        || child_name == "fp"
                                        || child_name == "lr"
                                    {
                                        self.registers.registers.push(Register {
                                            name: child_name.to_string(),
                                            value: Value::String(
                                                child.value().unwrap_or("no-value").to_string(),
                                            ),
                                            kind: BuiltInTypes::get_kind(
                                                usize::from_str_radix(
                                                    child
                                                        .value()
                                                        .unwrap_or("no-value")
                                                        .trim_start_matches("0x"),
                                                    16,
                                                )
                                                .unwrap(),
                                            ),
                                        });
                                    }
                                }
                            })
                        };
                    }
                });
            }
        }
    }

    fn process_instructions(&mut self, instructions: SBInstructionList, target: SBTarget) {
        self.process.instructions = Some(instructions.clone());
        for instruction in &instructions {
            let instruction = ParsedDisasm {
                address: instruction.address().load_address(&target),
                // I think this is data, but not sure
                hex: "".to_string(),
                instruction: instruction.mnemonic(&target).to_string(),
                arguments: instruction
                    .operands(&target)
                    .to_string()
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect(),
                comment: instruction.comment(&target).to_string(),
            };
            if instruction.instruction == "udf" {
                continue;
            }
            self.disasm.disasm_values.push(instruction);
            self.disasm
                .disasm_values
                .sort_by(|a, b| a.address.cmp(&b.address));
        }
    }

    fn check_debugger_info(
        &mut self,
        thread: &lldb::SBThread,
        process: &SBProcess,
        is_stepping: bool,
    ) -> bool {
        let function_name = thread
            .selected_frame()
            .function_name()
            .unwrap_or("")
            .to_string();
        if function_name.contains("black_box") {
            thread.step_instruction(false).unwrap();
            thread.step_instruction(false).unwrap();
        }

        let function_name = thread
            .selected_frame()
            .function_name()
            .unwrap_or("")
            .to_string();
        if function_name == "debugger_info" {
            let x0 = thread
                .selected_frame()
                .get_register("x0")
                .unwrap()
                .to_usize();
            let x1 = thread
                .selected_frame()
                .get_register("x1")
                .unwrap()
                .to_usize();

            let mut buffer: Vec<u8> = vec![0; x1];
            process.read_memory(x0 as u64, &mut buffer);
            let message = Message::from_binary(&buffer);
            let _ = self.websocket_sender.send(message.clone());
            self.breakpoint_mapper.process_message(&message);
            // TODO: Do better
            for ((_, _), address) in self
                .breakpoint_mapper
                .file_line_to_address
                .iter()
                .filter(|x| {
                    x.0.0 == "/Users/jimmyhmiller/Documents/Code/beagle/target/debug/resources/register_allocation_test.bg"
                })
                {
                    // print address as hex
                    self.labels.insert(
                        *address as usize,
                        Label {
                            label: "breakpoint".to_string(),
                            function_pointer: 0,
                            label_index: 0,
                            label_location: 0,
                        },
                    );
                    let breakpoint = self
                        .process
                        .target
                        .as_mut()
                        .unwrap()
                        .breakpoint_create_by_address(*address);
                    breakpoint.set_enabled(true);
                }
            match message.data.clone() {
                Data::ForeignFunction {
                    name: _,
                    pointer: _,
                } => {}
                Data::BuiltinFunction {
                    name: _,
                    pointer: _,
                } => {}
                Data::Label {
                    label,
                    function_pointer,
                    label_index,
                    label_location,
                } => {
                    self.labels.insert(
                        function_pointer + label_location,
                        Label {
                            label,
                            function_pointer,
                            label_index,
                            label_location,
                        },
                    );
                }
                Data::HeapSegmentPointer { pointer } => {
                    self.heap.heap_pointers.push(pointer);
                }
                Data::UserFunction {
                    name,
                    pointer,
                    len,
                    number_of_arguments,
                } => {
                    // Only set breakpoints for functions matching the provided regex
                    if self.function_regex.is_match(&name) {
                        let mut bp = self
                            .process
                            .target
                            .as_mut()
                            .unwrap()
                            .breakpoint_create_by_address(pointer as u64);
                        bp.set_enabled(true);
                    }

                    self.functions.insert(
                        pointer,
                        Function::User {
                            name,
                            address_range: (pointer, pointer + len),
                            number_of_arguments,
                        },
                    );
                }
                Data::StackMap {
                    pc: _,
                    name: _,
                    stack_map: _,
                } => {
                    // TODO: do something here
                    // println!("{}", message.data.to_string());
                }
                Data::Allocate {
                    bytes: _,
                    stack_pointer: _,
                    kind: _,
                } => {
                    // println!("{}", message.data.to_string());
                }
                Data::Tokens {
                    file_name: _,
                    tokens: _,
                    token_line_column_map: _,
                } => {
                    // println!("{}", message.data.to_string());
                }
                Data::Ir {
                    function_pointer: _,
                    file_name: _,
                    instructions: _,
                    token_range_to_ir_range: _,
                } => {
                    // println!("{}", message.data.to_string());
                }
                Data::Arm {
                    function_pointer: _,
                    file_name: _,
                    instructions: _,
                    ir_to_machine_code_range: _,
                } => {
                    // println!("{}", message.data.to_string());
                }
            }
            self.messages.push(message);

            if is_stepping {
                thread.step_over(RunMode::OnlyThisThread).unwrap();
            } else {
                process.continue_execution().unwrap();
            }
            true
        } else {
            false
        }
    }
}

#[derive(Debug)]
struct Memory {
    address: u64,
    value: u64,
    kind: BuiltInTypes,
}

impl Memory {
    fn to_string(&self) -> String {
        format!("0x{:x}: 0x{:x} {:?}", self.address, self.value, self.kind)
    }
}

struct Stack {
    stack: Vec<Memory>,
}

#[allow(unused)]
enum Function {
    Foreign {
        name: String,
        pointer: usize,
    },
    Builtin {
        name: String,
        pointer: usize,
    },
    User {
        name: String,
        address_range: (usize, usize),
        number_of_arguments: usize,
    },
}

impl Function {
    fn get_name(&self) -> String {
        match self {
            Function::Foreign { name, pointer: _ } => name.to_string(),
            Function::Builtin { name, pointer: _ } => name.to_string(),
            Function::User {
                name,
                address_range: _,
                number_of_arguments: _,
            } => name.to_string(),
        }
    }
}

type Address = usize;

struct Heap {
    memory: Vec<Memory>,
    heap_pointers: Vec<usize>,
}

struct State {
    pc: u64,
    disasm: Disasm,
    registers: Registers,
    process: Process,
    stack: Stack,
    sp: u64,
    fp: u64,
    messages: Vec<Message>,
    functions: HashMap<Address, Function>,
    labels: HashMap<Address, Label>,
    heap: Heap,
    websocket_sender: std::sync::mpsc::Sender<Message>,
    breakpoint_mapper: BreakpointMapper,
    beagle_program_path: String,
    function_regex: Regex,
}

#[derive(Debug)]
struct ParsedDisasm {
    address: u64,
    hex: String,
    instruction: String,
    arguments: Vec<String>,
    comment: String,
}

impl ParsedDisasm {
    // TODO: I should probably make it so I don't just make a big string
    // but instead things I can then chunk up and display with different styles
    fn to_string(&self, show_address: bool, show_hex: bool) -> String {
        let mut result = String::new();
        if show_address {
            result.push_str(&format!("0x{:x} ", self.address));
        }
        if show_hex {
            result.push_str(&format!("{} ", self.hex));
        }
        result.push_str(&format!("{:8} ", self.instruction));
        for argument in &self.arguments {
            result.push_str(&format!("{} ", argument));
        }
        if !self.comment.is_empty() {
            // left pad with spaces
            result.push_str(&format!("  ; {}", self.comment));
            // result.push_str(&format!(";{} ", self.comment));
        }
        result
    }
}

trait Node {
    fn draw(&self, canvas: &skia_safe::Canvas);
    fn width(&self) -> f32;
    fn height(&self) -> f32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    Horizontal,
    Vertical,
}

struct Layout {
    direction: Direction,
    margin: f32,
    child_spacing: f32,
}

#[allow(unused)]
impl Layout {
    fn with_margin(self, margin: f32) -> Self {
        Self { margin, ..self }
    }

    fn with_child_spacing(self, child_spacing: f32) -> Self {
        Self {
            child_spacing,
            ..self
        }
    }
}

struct Root {
    layout: Layout,
    children: Vec<Box<dyn Node>>,
}

impl Node for Root {
    fn draw(&self, canvas: &skia_safe::Canvas) {
        canvas.save();
        canvas.translate((self.layout.margin, self.layout.margin));
        for child in &self.children {
            canvas.save();
            child.draw(canvas);
            canvas.restore();
            if self.layout.direction == Direction::Horizontal {
                canvas.translate((child.width() + self.layout.child_spacing, 0.0));
            } else {
                canvas.translate((0.0, child.height() + self.layout.child_spacing));
            }
        }
        canvas.restore();
    }
    fn height(&self) -> f32 {
        if self.layout.direction == Direction::Vertical {
            self.children
                .iter()
                .map(|child| child.height())
                .sum::<f32>()
                + self.layout.margin
        } else {
            self.children
                .iter()
                .map(|child| child.height())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or_default()
                + self.layout.margin
        }
    }

    fn width(&self) -> f32 {
        if self.layout.direction == Direction::Horizontal {
            self.children.iter().map(|child| child.width()).sum::<f32>() + self.layout.margin
        } else {
            self.children
                .iter()
                .map(|child| child.width())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or_default()
                + self.layout.margin
        }
    }
}

struct Paragraph {
    text: String,
}

impl Paragraph {
    fn build_paragraph(&self) -> skia_safe::textlayout::Paragraph {
        let mut font_collection = FontCollection::new();
        font_collection.set_default_font_manager(FontMgr::default(), "Ubuntu Mono");
        let paragraph_style = ParagraphStyle::new();
        // change color to black

        let mut text_style = TextStyle::new();
        let mut paint = paint::Paint::default();
        paint.set_color(skia_safe::Color::BLACK);
        // canvas.draw_rect(Rect::from_xywh(0.0, 0.0, 1000.0, 1000.0), &paint);
        text_style.set_foreground_paint(&paint);
        text_style.set_font_size(32.0);

        let mut builder = ParagraphBuilder::new(&paragraph_style, &font_collection);
        builder.push_style(&text_style);
        builder.add_text(&self.text);
        let paragraph = builder.build();
        paragraph
    }
}

impl Node for Paragraph {
    fn draw(&self, canvas: &skia_safe::Canvas) {
        let mut paragraph = self.build_paragraph();
        // TODO: get correct width
        paragraph.layout(1000.0);
        paragraph.paint(canvas, (100.0, 100.0));
    }
    fn height(&self) -> f32 {
        let mut paragraph = self.build_paragraph();
        paragraph.layout(1000.0);
        paragraph.height()
    }

    fn width(&self) -> f32 {
        // TODO: Fix the width
        let mut paragraph = self.build_paragraph();
        paragraph.layout(1000.0);
        1000.0
    }
}

impl Paragraph {
    fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
        }
    }
}

impl Root {
    fn new(layout: Layout) -> Self {
        Self {
            layout,
            children: Vec::new(),
        }
    }

    fn add(&mut self, node: impl Node + 'static) {
        self.children.push(Box::new(node));
    }
}

fn horizontal() -> Layout {
    Layout {
        direction: Direction::Horizontal,
        margin: 0.0,
        child_spacing: 0.0,
    }
}

fn vertical() -> Layout {
    Layout {
        direction: Direction::Vertical,
        margin: 0.0,
        child_spacing: 0.0,
    }
}

// takes an iter<String>
fn lines(lines: &Vec<String>) -> impl Node {
    let mut root = Root::new(vertical());
    for line in lines {
        root.add(Paragraph::new(line));
    }
    root
}

#[allow(unused)]
struct Empty {}

impl Node for Empty {
    fn draw(&self, _canvas: &skia_safe::Canvas) {}
    fn height(&self) -> f32 {
        0.0
    }
    fn width(&self) -> f32 {
        0.0
    }
}

#[allow(unused)]
fn empty() -> impl Node {
    Empty {}
}

fn disasm(state: &State) -> impl Node {
    let window_around_pc = 25;
    let mut start = 0;
    for (i, disasm) in state.disasm.disasm_values.iter().enumerate() {
        if disasm.address == state.pc {
            start = i.saturating_sub(window_around_pc);
            break;
        }
    }
    let disasm = &state.disasm.disasm_values[start..];
    let mut contiguous_disasm = vec![];
    let mut last_address = 0;
    for disasm in disasm.iter() {
        if disasm.address == last_address + 4 || last_address == 0 {
            contiguous_disasm.push(disasm);
        } else {
            if disasm.address > state.pc {
                break;
            }
        }
        last_address = disasm.address;
    }

    let disasm = contiguous_disasm;

    lines(
        &disasm
            .iter()
            .take(50)
            .map(|disasm| {
                let prefix = if disasm.address == state.pc {
                    "> "
                } else {
                    "  "
                };
                if let Some(function) = state.functions.get(&(disasm.address as usize)) {
                    format!(
                        "{}{: <11} {}",
                        prefix,
                        function.get_name(),
                        disasm.to_string(false, state.disasm.show_hex)
                    )
                } else if let Some(label) = state.labels.get(&(disasm.address as usize)) {
                    format!(
                        "{}{: <11} {}",
                        prefix,
                        label.label,
                        disasm.to_string(false, state.disasm.show_hex)
                    )
                } else {
                    format!(
                        "{}{}",
                        prefix,
                        disasm.to_string(state.disasm.show_address, state.disasm.show_hex)
                    )
                }
            })
            .collect::<Vec<String>>(),
    )
}

fn registers(state: &State) -> impl Node {
    let mut mentioned_registers = state
        .disasm
        .disasm_values
        .iter()
        .flat_map(|disasm| disasm.arguments.iter())
        .filter(|argument| argument.starts_with("x"))
        .map(|argument| argument.trim_end_matches(','))
        .map(|argument| argument.to_string())
        .collect::<HashSet<String>>();

    mentioned_registers.insert("pc".to_string());
    mentioned_registers.insert("sp".to_string());
    mentioned_registers.insert("fp".to_string());
    mentioned_registers.insert("lr".to_string());

    lines(
        &state
            .registers
            .registers
            .iter()
            .filter(|register| mentioned_registers.contains(&register.name))
            .map(|register| register.to_string())
            .collect::<Vec<String>>(),
    )
}

fn memory(state: &State) -> impl Node {
    lines(
        &state
            .heap
            .memory
            .iter()
            .map(|memory| memory.to_string())
            .collect::<Vec<String>>(),
    )
}

fn stack(state: &State) -> impl Node {
    lines(
        &state
            .stack
            .stack
            .iter()
            .map(|value| {
                if value.address == state.sp && value.address == state.fp {
                    format!("fsp> {}", value.to_string())
                } else if value.address == state.fp {
                    format!("fp>  {}", value.to_string())
                } else if value.address == state.sp {
                    format!("sp>  {}", value.to_string())
                } else {
                    format!("     {}", value.to_string())
                }
            })
            .collect::<Vec<String>>(),
    )
}

fn current_function(state: &State) -> impl Node {
    let mut root = Root::new(horizontal());
    let mut function_name = "No Function".to_string();
    let function_look_up = state.breakpoint_mapper.file_line_by_address(state.pc);
    let mut file_line = None;
    if let Some((file, line)) = function_look_up {
        // file is a path and we want the last segment
        let file = file.split('/').last().unwrap();
        file_line = Some((file.to_string(), line));
    }
    for function in state.functions.values() {
        if let Function::User {
            name,
            address_range,
            number_of_arguments: _,
        } = function
        {
            if state.pc as usize >= address_range.0 && state.pc as usize <= address_range.1 {
                function_name = name.clone();
            }
        }
    }
    let result = if let Some((file, file_line)) = file_line {
        format!("{}:{} {}", file, file_line, function_name)
    } else {
        format!("{}", function_name)
    };
    root.add(Paragraph::new(&result));
    root
}

fn debug(state: &State) -> impl Node {
    let mut outer = Root::new(vertical().with_child_spacing(30.0));
    let mut row1 = Root::new(horizontal());
    let mut row2 = Root::new(horizontal());
    row1.add(disasm(state));
    row1.add(registers(state));
    // root.add(memory(state));
    row1.add(stack(state));

    row2.add(current_function(state));
    // row2.add(meta(state));
    row2.add(memory(state));

    outer.add(row1);
    outer.add(row2);
    outer
}

#[allow(unused)]
fn meta(state: &State) -> impl Node {
    lines(
        &state
            .messages
            .iter()
            .map(|message| format!("{}", message.to_string()))
            .collect::<Vec<String>>(),
    )
}

trait FrameExentions {
    fn get_register(&self, name: &str) -> Option<SBValue>;
}

impl FrameExentions for SBFrame {
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

trait ValueExtensions {
    fn to_usize(&self) -> usize;
}

impl ValueExtensions for SBValue {
    fn to_usize(&self) -> usize {
        unsafe { SBValueGetValueAsUnsigned2(self.raw, 0) as usize }
    }
}

trait TargetExtensions {
    fn create_breakpoint_by_name(&self, name: &str, module_name: &str) -> Option<SBBreakpoint>;
}

impl TargetExtensions for SBTarget {
    fn create_breakpoint_by_name(&self, name: &str, module_name: &str) -> Option<SBBreakpoint> {
        unsafe {
            let name = CString::new(name).unwrap();
            let module_name = CString::new(module_name).unwrap();
            let pointer =
                SBTargetBreakpointCreateByName(self.raw, name.as_ptr(), module_name.as_ptr());
            if pointer.is_null() {
                None
            } else {
                Some(SBBreakpoint { raw: pointer })
            }
        }
    }
}

trait ProcessExtensions {
    fn read_memory(&self, address: u64, buffer: &mut [u8]) -> usize;
    fn get_instructions(&self, frame: &SBFrame, target: &SBTarget) -> SBInstructionList;
}

impl ProcessExtensions for SBProcess {
    fn read_memory(&self, address: u64, buffer: &mut [u8]) -> usize {
        let ptr: *mut c_void = buffer.as_mut_ptr() as *mut c_void;
        let count = buffer.len();
        let error = SBError::default();
        unsafe { lldb_sys::SBProcessReadMemory(self.raw, address, ptr, count, error.raw) }
    }
    fn get_instructions(&self, frame: &SBFrame, target: &SBTarget) -> SBInstructionList {
        let mut buffer = [0u8; 1024];
        self.read_memory(frame.pc_address().load_address(&target), &mut buffer);
        let base_addr = frame.pc_address();
        let ptr: *mut c_void = buffer.as_mut_ptr() as *mut c_void;
        let count = buffer.len();
        let flavor_string = CString::new("intel").unwrap();
        let flavor = flavor_string.as_ptr();
        unsafe {
            SBInstructionList {
                raw: lldb_sys::SBTargetGetInstructionsWithFlavor(
                    target.raw,
                    base_addr.raw,
                    flavor,
                    ptr,
                    count,
                ),
            }
        }
    }
}

fn start_process(beagle_program_path: &str) -> Option<(SBTarget, SBProcess)> {
    SBDebugger::initialize();

    let debugger = SBDebugger::create(false);
    debugger.set_asynchronous(false);

    if let Some(target) =
        debugger.create_target_simple("/Users/jimmyhmiller/Documents/Code/beagle/target/debug/main")
    {
        // let symbol_list = target.find_functions("debugger_info", 2);
        // let _ = symbol_list.into_iter().next().unwrap();
        let breakpoint = target
            .create_breakpoint_by_name("debugger_info", "main")
            .unwrap();
        breakpoint.set_enabled(true);
        target.enable_all_breakpoints();
        target.breakpoints().for_each(|breakpoint| {
            println!("{:?}", breakpoint);
        });

        // TODO: Make all of this better
        // and configurable at runtime
        let launchinfo = SBLaunchInfo::new();
        launchinfo.set_arguments(
            vec![beagle_program_path],
            false,
        );
        // launchinfo.set_launch_flags(LaunchFlags::STOP_AT_ENTRY);
        match target.launch(launchinfo) {
            Ok(process) => Some((target, process)),
            Err(e) => {
                println!("Uhoh: {e:?}");
                None
            }
        }
    } else {
        None
    }
}
