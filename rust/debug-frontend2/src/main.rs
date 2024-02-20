use std::{collections::HashMap, ffi::{c_void, CString}};

use bincode::{config::standard, Decode, Encode};
use gpui::{
    div, px, rgb, App, AppContext, Context, Div, InteractiveElement, IntoElement, KeyDownEvent, Model, ParentElement, Render, StatefulInteractiveElement, Styled, TitlebarOptions, ViewContext, VisualContext, WindowOptions
};
use lldb::{SBBreakpoint, SBDebugger, SBError, SBFrame, SBInstructionList, SBLaunchInfo, SBProcess, SBTarget, SBValue};
use lldb_sys::{RunMode, SBTargetBreakpointCreateByName, SBValueGetValueAsUnsigned2};

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
    HeapPointer {
        pointer: usize,
    },
    UserFunction {
        name: String,
        pointer: usize,
        len: usize,
    },
    Label {
        label: String,
        function_pointer: usize,
        label_index: usize,
        label_location: usize,
    },
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
            Data::HeapPointer { pointer } => {
                format!("0x{:x}", pointer)
            }
            Data::UserFunction { name, pointer, len } => {
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
        }
    }
}

#[derive(Debug, Encode, Decode, Clone)]
struct Label {
    label: String,
    function_pointer: usize,
    label_index: usize,
    label_location: usize,
}

struct Process {
    process: Option<SBProcess>,
    target: Option<SBTarget>,
    instructions: Option<SBInstructionList>,
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

// Stolen from compiler
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltInTypes {
    Int,
    Float,
    String,
    Bool,
    Function,
    Closure,
    Struct,
    Array,
    None,
}

impl BuiltInTypes {
    pub fn tag(&self, value: isize) -> isize {
        let value = value << 3;
        let tag = self.get_tag();
        value | tag
    }

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
            BuiltInTypes::None => panic!("None has no tag"),
        }
    }

    pub fn untag(pointer: usize) -> usize {
        pointer >> 3
    }

    pub fn get_kind(pointer: usize) -> Self {
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
            BuiltInTypes::None => "None".to_string(),
        }
    }
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

struct Disasm {
    disasm_values: Vec<ParsedDisasm>,
    show_address: bool,
    show_hex: bool,
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
            } => name.to_string(),
        }
    }
}

type Address = usize;

struct Heap {
    memory: Vec<Memory>,
    heap_pointers: Vec<usize>,
}

fn convert_to_u64_array(input: &[u8; 256]) -> Vec<u64> {
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
}


impl State {
    fn update_process_state(&mut self, is_stepping: bool) {
        if let (Some(process), Some(target)) =
            (self.process.process.clone(), self.process.target.clone())
        {
            if !process.is_stopped() {
                return;
            }
            if let Some(thread) = process.thread_by_index_id(1) {
                let was_debugger_info = self.check_debugger_info(&thread, &process, is_stepping);

                if was_debugger_info {
                    self.update_process_state(is_stepping);
                    return
                }

                let frame = thread.selected_frame();

                self.pc = frame.pc();
                let pc_in_instructions = self
                    .disasm
                    .disasm_values
                    .iter()
                    .map(|v| v.address)
                    .any(|v| v == self.pc);

                // TODO: Make this actually correct
                // Consider when we have later addresses, but gaps
                let last = self
                    .disasm
                    .disasm_values
                    .last()
                    .map(|v| v.address)
                    .unwrap_or(0);
                let remaining = last.saturating_sub(self.pc) / 4;

                self.sp = frame.sp();
                self.fp = frame.fp();
                let stack_root = frame.sp() - 128;
                let stack = &mut [0u8; 256];
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
                    let heap = &mut [0u8; 256];
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

                if !pc_in_instructions || remaining < 30 {
                    let instructions = process.get_instructions(&frame, &target);
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
                    }
                }
                self.disasm
                    .disasm_values
                    .sort_by(|a, b| a.address.cmp(&b.address));
                self.disasm
                    .disasm_values
                    .dedup_by(|a, b| a.address == b.address);
                self.registers.registers.clear();
                frame.registers().iter().for_each(|register| {
                    if register.name().contains("General") {
                        register.children().for_each(|child| {
                            if child.name().contains("x")
                                || child.name() == "pc"
                                || child.name() == "sp"
                                || child.name() == "fp"
                            {
                                self.registers.registers.push(Register {
                                    name: child.name().to_string(),
                                    value: Value::String(child.value().to_string()),
                                    kind: BuiltInTypes::get_kind(
                                        usize::from_str_radix(
                                            child.value().trim_start_matches("0x"),
                                            16,
                                        )
                                        .unwrap(),
                                    ),
                                });
                            }
                        });
                    }
                });
            }
        }
    }

    fn check_debugger_info(
        &mut self,
        thread: &lldb::SBThread,
        process: &SBProcess,
        is_stepping: bool,
    ) -> bool {
        if thread.selected_frame().function_name() == Some("debugger_info") {
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
            match message.data.clone() {
                Data::ForeignFunction { name, pointer } => {}
                Data::BuiltinFunction { name, pointer } => {}
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
                Data::HeapPointer { pointer } => {
                    self.heap.heap_pointers.push(pointer);
                }
                Data::UserFunction { name, pointer, len } => {
                    self.process
                        .target
                        .as_mut()
                        .unwrap()
                        .breakpoint_create_by_address(pointer as u64);
                    self.functions.insert(
                        pointer,
                        Function::User {
                            name,
                            address_range: (pointer, pointer + len),
                        },
                    );
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

    fn step_over(&mut self) {
        if let Some(process) = self.process.process.clone() {
            if let Some(thread) = process.thread_by_index_id(1) {
                let _was_debugger_info = self.check_debugger_info(&thread, &process, true);
                let pc_before = thread.selected_frame().pc();
                thread.step_instruction(true).unwrap();
                let pc_after = thread.selected_frame().pc();
                if pc_before == pc_after {
                    thread.selected_frame().set_pc(pc_after + 4);
                }
                self.update_process_state(true);
            }
        }
    }
}

#[derive(Debug, Encode, Decode)]
pub struct Message {
    kind: String,
    data: Data,
}

impl Message {
    fn to_string(&self) -> String {
        format!("{} {}", self.kind, self.data.to_string())
    }
}

impl State {
    fn new() -> Self {
        let mut this = Self {
            disasm: Disasm {
                disasm_values: vec![],
                show_address: true,
                show_hex: false,
            },
            registers: Registers { registers: vec![] },
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
        };

        if let Some((target, process)) = start_process() {
            this.process.process = Some(process);
            this.process.target = Some(target);
        }
        this
    }
}

struct Debugger {
    state: Model<State>,
    subscription: gpui::Subscription,
}

impl Debugger {
    fn new(state: Model<State>, cx: &mut ViewContext<Self>) -> Self {
        let state_cloned = state.clone();
        cx.observe_keystrokes(move |event, cx| {
            if event.keystroke.key == "down" {
                cx.update_model(&state_cloned, |state, cx| {
                    state.step_over();
                    // Be sure to notify
                    cx.notify();
                });
            }
        }).detach();
        let subscription = cx.observe(&state, |debugger, state, cx| {
            cx.notify()
        });
        cx.update_model(&state, |state, cx| {
            if state.process.instructions.is_none() {
                state.update_process_state(false);
            }
        });

        Self { state, subscription }
    }
}

impl Render for Debugger {
    fn render(&mut self, cx: &mut ViewContext<Self>) -> impl IntoElement {

        let state = self.state.read(cx);


        div()
            .id("instructions")
            .overflow_y_scroll()
            .flex()
            .size_full()
            .text_sm()
            .p(px(120.0))
            .bg(rgb(0xffffff))
            .text_color(rgb(0x000000))
            .child(disasm(&state))
    }
}


fn disasm(state: &State) -> Div {
    let window_around_pc = 10;
    let mut start = 0;
    for (i, disasm) in state.disasm.disasm_values.iter().enumerate() {
        if disasm.address == state.pc {
            start = i.saturating_sub(window_around_pc);
            break;
        }
    }
    let disasm = &state.disasm.disasm_values[start..];

    div()
    .children(
        disasm
            .iter()
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
            .collect::<Vec<String>>()
    )
}

trait FrameExentions {
    fn get_register(&self, name: &str) -> Option<SBValue>;
}

impl FrameExentions for SBFrame {
    fn get_register(&self, name: &str) -> Option<SBValue> {
        for register in self.registers().into_iter() {
            if register.name() == name {
                return Some(register);
            }
            for child in register.children().into_iter() {
                if child.name() == name {
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
        let mut error = SBError::default();
        unsafe { lldb_sys::SBProcessReadMemory(self.raw, address, ptr, count, error.raw) }
    }
    fn get_instructions(&self, frame: &SBFrame, target: &SBTarget) -> SBInstructionList {
        let mut buffer = [0u8; 128];
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

fn start_process() -> Option<(SBTarget, SBProcess)> {
    SBDebugger::initialize();

    let debugger = SBDebugger::create(false);
    debugger.set_asynchronous(false);

    if let Some(target) = debugger.create_target_simple(
        "/Users/jimmyhmiller/Documents/Code/PlayGround/rust/asm-arm2/target/debug/main",
    ) {
        let symbol_list = target.find_functions("debugger_info", 2);
        let symbol = symbol_list.into_iter().next().unwrap();
        let breakpoint = target
            .create_breakpoint_by_name("debugger_info", "main")
            .unwrap();
        breakpoint.set_enabled(true);
        target.enable_all_breakpoints();

        let launchinfo = SBLaunchInfo::new();
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

fn main() {
    App::new().run(|cx: &mut AppContext| {
        let model = cx.new_model(|cx| State::new());

        let mut window_options = WindowOptions::default();
        window_options.titlebar = Some(TitlebarOptions {
            title: Some("Debugger".to_string().into()),
            appears_transparent: true,
            ..Default::default()
        });


        cx.open_window(window_options, |cx| {

            cx.new_view(|cx| Debugger::new(model, cx))
        });
    });
}
