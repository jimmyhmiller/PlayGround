use std::{
    error::Error,
    ffi::{c_void, CString},
    time::Duration,
};

use bincode::{config::standard, Decode, Encode};
use gpui::{
    div, rgb, uniform_list, AbsoluteLength, App, AppContext, ClickEvent, InteractiveElement,
    IntoElement, MouseUpEvent, ParentElement, Pixels, Render, StatefulInteractiveElement, Styled,
    Task, Timer, ViewContext, VisualContext, WindowOptions,
};
use itertools::Itertools;
use lldb::{
    SBBreakpoint, SBDebugger, SBError, SBFrame, SBInstructionList, SBLaunchInfo, SBProcess,
    SBTarget, SBValue,
};
use lldb_sys::{SBTargetBreakpointCreateByName, SBValueGetValueAsUnsigned2};

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

trait ValueExtensions {
    fn to_usize(&self) -> usize;
}

impl ValueExtensions for SBValue {
    fn to_usize(&self) -> usize {
        unsafe { SBValueGetValueAsUnsigned2(self.raw, 0) as usize }
    }
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

#[derive(Debug, Encode, Decode)]
pub struct Message {
    kind: String,
    data: Data,
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
        }
    }
}

impl Message {
    fn to_string(&self) -> String {
        format!("{} {}", self.kind, self.data.to_string())
    }
}

#[derive(Debug, Encode, Decode, Clone)]
pub struct StackMapDetails {
    pub number_of_locals: usize,
    pub current_stack_size: usize,
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
}

#[derive(Debug, Encode, Decode, Clone)]
struct Label {
    label: String,
    function_pointer: usize,
    label_index: usize,
    label_location: usize,
}

struct Process {
    process: SBProcess,
    target: SBTarget,
    data: Vec<Data>,
}

impl Process {
    fn start() -> Result<Process, Box<dyn Error>> {
        SBDebugger::initialize();

        let debugger = SBDebugger::create(false);
        debugger.set_asynchronous(false);

        if let Some(target) = debugger.create_target_simple(
            "/Users/jimmyhmiller/Documents/Code/beagle/target/release-with-debug/main",
        ) {
            // TODO: Clean up
            let symbol_list = target.find_functions("debugger_info", 2);
            let _symbol = symbol_list.into_iter().next().unwrap();
            let breakpoint = target
                .create_breakpoint_by_name("debugger_info", "main")
                .unwrap();
            breakpoint.set_enabled(true);
            target.enable_all_breakpoints();

            // TODO: Make all of this better
            // and configurable at runtime
            let launchinfo = SBLaunchInfo::new();
            launchinfo.set_arguments(
                vec!["/Users/jimmyhmiller/Documents/Code/beagle/resources/atom.bg"],
                false,
            );
            // launchinfo.set_launch_flags(LaunchFlags::STOP_AT_ENTRY);
            let process = target.launch(launchinfo)?;
            Ok(Process {
                process,
                target,
                data: vec![],
            })
        } else {
            Err("Could not create target".into())
        }
    }

    fn is_stopped(&self) -> bool {
        self.process.is_stopped()
    }

    async fn update_process(&mut self) {
        loop {
            Timer::after(Duration::from_millis(1)).await;
            if !self.process.is_stopped() {
                return;
            }
            if let Some(thread) = self.process.thread_by_index_id(1) {
                let mut function_name = thread
                    .selected_frame()
                    .function_name()
                    .unwrap_or("")
                    .to_string();
                while function_name.contains("black_box") {
                    thread.step_instruction(false).unwrap();
                    function_name = thread
                        .selected_frame()
                        .function_name()
                        .unwrap_or("")
                        .to_string();
                }
                if function_name != "debugger_info" {
                    self.process.continue_execution().unwrap();
                    break;
                }
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
                self.process.read_memory(x0 as u64, &mut buffer);
                let message = Message::from_binary(&buffer);
                if !matches!(message.data, Data::Label { .. } | Data::StackMap { .. }) {
                    self.data.push(message.data);
                }

                self.process.continue_execution().unwrap();
            }
        }
    }

    fn drain_data(&mut self) -> Vec<Data> {
        std::mem::take(&mut self.data)
    }
}

struct ProcessState {
    started: bool,
    stopped: bool,
    data: Vec<Data>,
}

struct Debugger {
    process_state: ProcessState,
    spawn_task: Option<Task<()>>,
}

impl Debugger {
    fn spawn_process(&mut self, _: &ClickEvent, cx: &mut ViewContext<Self>) {
        self.spawn_task = Some(cx.spawn(|view, mut cx| async move {
            let mut process = Process::start().unwrap();
            let _ = cx.update(|cx| {
                view.update(cx, |view, cx| {
                    view.process_state.started = true;
                    cx.notify();
                })
            });
            loop {
                process.update_process().await;
                let _ = cx.update(|cx| {
                    view.update(cx, |view, cx| {
                        view.process_state.stopped = process.is_stopped();
                        view.process_state.data.extend(process.drain_data());
                        cx.notify();
                    })
                });
                Timer::after(Duration::from_millis(16)).await;
            }
        }));
    }
}

fn card_stack(data: Vec<String>) -> impl IntoElement {
    // The goal here is to make it appear like cards are being stacked.
    // If we hit the cap, we just have the same thickness of cards.
    // Otherwise, we make it clear how many cards are in the stack

    let cap = 30;
    if data.len() >= cap {
        // We are going to draw a series of lines on the header
        // that alternate colors to make it look like cards are stacked
        // then we will draw the last card beneath it

        let mut header = div().rounded_md().relative().h_5().w_48();
        for i in 0..20 {
            header = header.child(
                div()
                    .h(Pixels(40.0))
                    .w_48()
                    .absolute()
                    .top(Pixels(i as f32))
                    .rounded_md()
                    .border(Pixels(1.0))
                    .border_color(if i % 2 == 0 {
                        rgb(0x000000)
                    } else {
                        rgb(0xffffff)
                    }),
            );
        }
        div()
            .rounded_md()
            .bg(rgb(0xffffff))
            .text_color(rgb(0x000000))
            .child(div().child(header))
            .child(
                div()
                    .p_2()
                    .top(Pixels(20.0))
                    .h_48()
                    .rounded_md()
                    .w_48()
                    .bg(rgb(0xffffff))
                    .text_color(rgb(0x000000))
                    .child(data.last().unwrap().clone()),
            )
    } else {
        div()
            .h_48()
            .w_48()
            .bg(rgb(0xffffff))
            .text_color(rgb(0x000000))
            .children(data)
    }
}

impl Render for Debugger {
    fn render(&mut self, cx: &mut ViewContext<Self>) -> impl IntoElement {
        div()
            .size_full()
            .text_color(rgb(0xffffff))
            .child("Debugger")
            .child(if self.process_state.started {
                div()
                    .m_4()
                    .size_full()
                    .id("test2")
                    .child("Process started")
                    .child(format!("Stopped: {}", self.process_state.stopped))
                    .child(
                        div()
                            .size_full()
                            .flex()
                            .gap_8()
                            .flex_row()
                            .flex_wrap()
                            .children(
                                self.process_state
                                    .data
                                    .iter()
                                    .filter(|data| matches!(data, Data::UserFunction { .. }))
                                    .chunks(30)
                                    .into_iter()
                                    .map(|data| {
                                        div().relative().w_48().h_48().child(card_stack(
                                            data.map(|x| x.to_string()).collect(),
                                        ))
                                    }),
                            ),
                    )
            } else {
                div()
                    .id("test")
                    .child("Start")
                    .on_click(cx.listener(Self::spawn_process))
            })
    }
}

fn main() {
    App::new().run(|cx: &mut AppContext| {
        let window = cx
            .open_window(
                WindowOptions {
                    ..Default::default()
                },
                |cx| {
                    cx.new_view(|cx| Debugger {
                        process_state: ProcessState {
                            started: false,
                            stopped: false,
                            data: vec![],
                        },
                        spawn_task: None,
                    })
                },
            )
            .unwrap();
        cx.observe_keystrokes(move |ev, cx| {
            window
                .update(cx, |view, cx| {
                    // view.recent_keystrokes.push(ev.keystroke.clone());
                    cx.notify();
                })
                .unwrap();
        })
        .detach();

        window
            .update(cx, |view, cx| {
                // cx.focus_view(&view.text_input);
                cx.activate(true);
            })
            .unwrap();
    });
}
