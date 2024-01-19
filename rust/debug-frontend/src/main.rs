use std::{ffi::{c_void, CString}, collections::HashSet, str::from_utf8};

use lldb::{SBDebugger, SBError, SBFrame, SBInstructionList, SBLaunchInfo, SBProcess, SBTarget, SBBreakpoint, SBValue};
use lldb_sys::{RunMode, SBTargetBreakpointCreateByName, SBValueGetValueAsUnsigned2};
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


#[derive(Debug)]
struct Message {
    kind: String,
    data: Vec<u8>,
}

impl Message {
    fn to_binary(&self) -> Vec<u8> {
        let kind_length = self.kind.len();
        let data_length = self.data.len();
        let mut result = vec![0; 8 + 8 + kind_length + data_length];
        result[0..8].copy_from_slice(&u64::to_le_bytes(kind_length as u64));
        result[8..16].copy_from_slice(&u64::to_le_bytes(data_length as u64));
        result[16..16 + kind_length].copy_from_slice(self.kind.as_bytes());
        result[16 + kind_length..].copy_from_slice(&self.data);
        result
    }

    fn from_binary(data: &[u8]) -> Message {
        let kind_length = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        let data_length = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        let kind = String::from_utf8(data[16..16 + kind_length].to_vec()).unwrap();
        let data = data[16 + kind_length..16 + kind_length + data_length].to_vec();
        Message { kind, data }
    }
}

struct Frontend {
    state: State,
    should_step: bool,
}

#[allow(unused_variables)]
impl App for Frontend {
    fn on_window_create(&mut self, event_loop_proxy: EventLoopProxy<()>, size: skia_window::Size) {
        if let Some((target, process)) = start_process() {
            self.state.process.process = Some(process);
            self.state.process.target = Some(target);
        }
    }

    fn add_event(&mut self, event: &Event<()>) -> bool {
        // TODO: This is super gross
        match event {
            Event::WindowEvent { window_id, event } => match event {
                KeyboardInput { event, .. } => {
                    match event.state {
                        ElementState::Pressed => match event.physical_key {
                            PhysicalKey::Code(KeyCode::ArrowDown) => {
                                self.step_over();
                            }
                            PhysicalKey::Code(KeyCode::ArrowRight) => {
                                self.keep_stepping();
                            }
                            PhysicalKey::Code(KeyCode::ArrowLeft) => {
                                self.stop_stepping();
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
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

        if self.should_step {
            self.step_over();
        }
        if self.state.process.instructions.is_some() {
            return;
        }
        self.state.update_process_state();
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
    fn step_over(&mut self) {
        if let Some(process) = &self.state.process.process {
            if let Some(thread) = process.thread_by_index_id(1) {
                if thread.selected_frame().function_name() == Some("debugger_info") {
                    println!("Got info!");
                    thread.resume().unwrap();
                }
                let pc_before = thread.selected_frame().pc();
                thread.step_over(RunMode::OnlyDuringStepping).unwrap();
                let pc_after = thread.selected_frame().pc();
                if pc_before == pc_after {
                    thread.selected_frame().set_pc(pc_after + 4);
                }
                self.state.update_process_state();
            }
        }
    }

    fn keep_stepping(&mut self) {
        self.should_step = true;
    }

    fn stop_stepping(&mut self) {
        self.should_step = false;
    }
}

fn main() {
    let mut frontend = Frontend {
        should_step: false,
        state: State {
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
                    },
                    Register {
                        name: "x1".to_string(),
                        value: Value::Integer(1),
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
            stack: Stack {
                stack: vec![],
            },
        },
    };
    frontend.create_window("Debug", 1600, 1600, Options { vsync: false });
}


// Stolen from compiler
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltInTypes {
    Int,
    Float,
    String,
    Bool,
    Function,
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
            BuiltInTypes::Struct => 0b101,
            BuiltInTypes::Array => 0b110,
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
            0b101 => BuiltInTypes::Struct,
            0b110 => BuiltInTypes::Array,
            _ => BuiltInTypes::None
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
            BuiltInTypes::None => false,
        }
    }

    pub fn construct_int(value: isize) -> isize {
        if value > 0b1111111 {
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
}

impl Register {
    fn to_string(&self) -> String {
        format!("{}: {}", self.name, self.value.to_string())
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

fn convert_to_u64_array(input: &[u8; 256]) -> Vec<u64> {
    input
        .chunks(8)
        .map(|chunk| {
            chunk.iter().enumerate().fold(0, |acc, (i, &byte)| {
                acc | (byte as u64) << (8 * i)
            })
        })
        .collect()
}



impl State {
    fn update_process_state(&mut self) {
        if let (Some(process), Some(target)) = (&self.process.process, &self.process.target) {
            if let Some(thread) = process.thread_by_index_id(1) {

                if thread.selected_frame().function_name() == Some("debugger_info") {
                    let x0 = thread.selected_frame().get_register("x0").unwrap().to_usize();
                    let x1 = thread.selected_frame().get_register("x1").unwrap().to_usize();
    
                    let mut buffer: Vec<u8> = vec![0; x1];
                    process.read_memory(x0 as u64, &mut buffer);
                    let message = Message::from_binary(&buffer);
                    println!("Message: {:?}", message);
                    

                    process.continue_execution().unwrap();
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

                // TODO: Make this actually correct
                // Consider when we have later addresses, but gaps
                let last = self.disasm.disasm_values.last().map(|v| v.address).unwrap_or(0);
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
                        kind: BuiltInTypes::get_kind(*value as usize)
                    })
                    .collect();



                if !pc_in_instructions || remaining < 30 {
                    let instructions = process.get_instructions(&frame, &target);
                    self.process.instructions = Some(instructions.clone());
                    for instruction in &instructions {
                        let instruction = ParsedDisasm {
                            address: instruction.address().load_address(&target),
                            // I think this is data, but not sure
                            hex: "".to_string(),
                            instruction: instruction.mnemonic(target).to_string(),
                            arguments: instruction
                                .operands(target)
                                .to_string()
                                .split(',')
                                .map(|s| s.trim().to_string())
                                .collect(),
                            comment: instruction.comment(target).to_string(),
                        };
                        if instruction.instruction == "udf" {
                            continue;
                        }
                        if instruction.address <= last {
                            continue;
                        }
                        self.disasm.disasm_values.push(instruction);
                    }
                }
                self.disasm.disasm_values.sort_by(|a, b| a.address.cmp(&b.address));
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
                                });
                            }
                        });
                    }
                });
            }
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

struct State {
    pc: u64,
    disasm: Disasm,
    registers: Registers,
    process: Process,
    stack: Stack,
    sp: u64,
    fp: u64,
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
            result.push_str(&format!("{:x} ", self.address));
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
}

struct Root {
    layout: Layout,
    children: Vec<Box<dyn Node>>,
}

impl Node for Root {
    fn draw(&self, canvas: &skia_safe::Canvas) {
        canvas.save();
        for child in &self.children {
            canvas.save();
            child.draw(canvas);
            canvas.restore();
            if self.layout.direction == Direction::Horizontal {
                canvas.translate((child.width(), 0.0));
            } else {
                canvas.translate((0.0, child.height()));
            }
        }
        canvas.restore();
    }
    fn height(&self) -> f32 {
        if self.layout.direction == Direction::Vertical {
            self.children.iter().map(|child| child.height()).sum()
        } else {
            self.children
                .iter()
                .map(|child| child.height())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or_default()
        }
    }

    fn width(&self) -> f32 {
        if self.layout.direction == Direction::Horizontal {
            self.children.iter().map(|child| child.width()).sum()
        } else {
            self.children
                .iter()
                .map(|child| child.width())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or_default()
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
        text_style.set_font_size(36.0);

        let mut builder = ParagraphBuilder::new(&paragraph_style, &font_collection);
        builder.push_style(&text_style);
        builder.add_text(&self.text);
        let mut paragraph = builder.build();
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
    }
}

fn vertical() -> Layout {
    Layout {
        direction: Direction::Vertical,
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

struct Empty {}

impl Node for Empty {
    fn draw(&self, canvas: &skia_safe::Canvas) {}
    fn height(&self) -> f32 {
        0.0
    }
    fn width(&self) -> f32 {
        0.0
    }
}

fn empty() -> impl Node {
    Empty {}
}

fn disasm(state: &State) -> impl Node {
    let window_around_pc = 10;
    let mut start = 0;
    for (i, disasm) in state.disasm.disasm_values.iter().enumerate() {
        if disasm.address == state.pc {
            start = i.saturating_sub(window_around_pc);
            break;
        }
    }
    let disasm = &state.disasm.disasm_values[start..];

    lines(
        &disasm
            .iter()
            .map(|disasm| {
                if disasm.address == state.pc {
                    format!(
                        "> {}",
                        disasm.to_string(state.disasm.show_address, state.disasm.show_hex)
                    )
                } else {
                    format!(
                        "  {}",
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
    empty()
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

fn debug(state: &State) -> impl Node {
    let mut root = Root::new(horizontal());
    root.add(disasm(state));
    root.add(registers(state));
    // root.add(memory(state));
    root.add(stack(state));
    root
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
        unsafe { 
            SBValueGetValueAsUnsigned2(self.raw, 0) as usize
        }
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
            let pointer = SBTargetBreakpointCreateByName(self.raw, name.as_ptr(), module_name.as_ptr());
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
        let breakpoint = target.create_breakpoint_by_name("debugger_info", "main").unwrap();
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
