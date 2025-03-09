use std::collections::HashMap;

use iced::widget::container::Style;
use iced::widget::shader::wgpu::Color;
use iced::widget::{button, mouse_area, row, text_input, column, scrollable, text, Column, Container, Row};
use iced::{window, Background, Element, Size, Subscription};
use iced::window::settings::PlatformSpecific;

mod websocket;

pub fn main() -> iced::Result {
    iced::application("Code Explorer", State::update, State::view)
        .window(window::Settings {
            size: Size {
                width: 1000.0,
                height: 800.0,
            },
            platform_specific: PlatformSpecific {
                title_hidden: true,
                titlebar_transparent: true,
                fullsize_content_view: true,
                ..Default::default()
            },
            ..window::Settings::default()
        })
        .subscription(State::subscription)
        .run()
}

struct File {
    name: String,
    tokens: Vec<String>,
    token_line_column_map: Vec<(usize, usize)>,
    instructions: HashMap<usize, Vec<String>>,
    arm_instructions: HashMap<usize, Vec<String>>,
    token_range_to_ir_range: HashMap<usize, Vec<((usize, usize), (usize, usize))>>,
    ir_to_machine_code_range: HashMap<usize, Vec<(usize, (usize, usize))>>,
}

struct FunctionInfo {
    name: String,
    pointer: usize,
    len: usize,
}

#[derive(Default)]
struct State {
    connection: Option<websocket::Connection>,
    files: Vec<File>,
    hovered_token: Option<usize>,
    hovered_ir: Option<(usize, usize)>,
    selected_file: usize,
    functions: HashMap<String, FunctionInfo>,
    searching_address: String,
}

#[derive(Debug, Clone)]
enum Message {
    Hover(usize),
    Leave(usize),
    IrHover(usize, usize),
    IrLeave(usize, usize),
    ToggleSelectedFile,
    Websocket(websocket::Event),
    AddressSearch(String),
}

impl State {

    fn add_function(&mut self, function: FunctionInfo) {
        self.functions.insert(function.name.clone(), function);
    }

    fn upsert_file(&mut self, file_name: String, data: websocket::Data) {
        let file = self.files.iter_mut().find(|f| f.name == file_name);

        match file {
            Some(file) => match data {
                websocket::Data::Tokens {
                    tokens,
                    token_line_column_map,
                    ..
                } => {
                    file.tokens = tokens;
                    file.token_line_column_map = token_line_column_map;
                }
                websocket::Data::Ir {
                    instructions,
                    token_range_to_ir_range,
                    function_pointer,
                    file_name: _,
                } => {
                    file.instructions.insert(function_pointer, instructions);
                    file.token_range_to_ir_range
                        .insert(function_pointer, token_range_to_ir_range);
                }
                websocket::Data::Arm {
                    function_pointer,
                    instructions,
                    ir_to_machine_code_range,
                    file_name: _,
                } => {
                    file.arm_instructions.insert(function_pointer, instructions);
                    file.ir_to_machine_code_range
                        .insert(function_pointer, ir_to_machine_code_range);
                }
                _ => {}
            },
            None => match data {
                websocket::Data::Tokens {
                    tokens,
                    token_line_column_map,
                    ..
                } => {
                    self.files.push(File {
                        name: file_name,
                        tokens,
                        token_line_column_map,
                        instructions: HashMap::new(),
                        token_range_to_ir_range: HashMap::new(),
                        arm_instructions: HashMap::new(),
                        ir_to_machine_code_range: HashMap::new(),
                    });
                }
                websocket::Data::Ir {
                    instructions,
                    token_range_to_ir_range,
                    function_pointer,
                    file_name,
                } => {
                    let mut instructions_map = HashMap::new();
                    instructions_map.insert(function_pointer, instructions);
                    let mut token_range_to_ir_range_map = HashMap::new();
                    token_range_to_ir_range_map.insert(function_pointer, token_range_to_ir_range);
                    self.files.push(File {
                        name: file_name,
                        tokens: Vec::new(),
                        token_line_column_map: Vec::new(),
                        instructions: instructions_map,
                        token_range_to_ir_range: token_range_to_ir_range_map,
                        arm_instructions: HashMap::new(),
                        ir_to_machine_code_range: HashMap::new(),
                    });
                }
                websocket::Data::Arm {
                    function_pointer,
                    instructions,
                    ir_to_machine_code_range,
                    file_name,
                } => {
                    let mut instructions_map = HashMap::new();
                    instructions_map.insert(function_pointer, instructions);
                    let mut ir_to_machine_code_range_map = HashMap::new();
                    ir_to_machine_code_range_map.insert(function_pointer, ir_to_machine_code_range);
                    self.files.push(File {
                        name: file_name,
                        tokens: Vec::new(),
                        token_line_column_map: Vec::new(),
                        instructions: instructions_map,
                        token_range_to_ir_range: HashMap::new(),
                        arm_instructions: HashMap::new(),
                        ir_to_machine_code_range: ir_to_machine_code_range_map,
                    });
                }
                _ => {}
            },
        }
    }

    fn update(&mut self, message: Message) {
        match message {
            Message::AddressSearch(address) => {
                self.searching_address = address;
            }
            Message::Hover(i) => {
                self.hovered_token = Some(i);
            }
            Message::Leave(i) => {
                if self.hovered_token == Some(i) {
                    self.hovered_token = None;
                }
            }
            Message::IrHover(function_pointer, i) => {
                self.hovered_ir = Some((function_pointer, i));
            }
            Message::IrLeave(function_pointer, i) => {
                if self.hovered_ir == Some((function_pointer, i)) {
                    self.hovered_ir = None;
                }
            }
            Message::ToggleSelectedFile => {
                self.selected_file = (self.selected_file + 1) % self.files.len();
            }
            Message::Websocket(event) => match event {
                websocket::Event::Connected(connection) => {
                    self.connection = Some(connection);
                    println!("Connected to websocket");
                }
                websocket::Event::Disconnected => {
                    println!("Disconnected from websocket");
                }
                websocket::Event::MessageReceived(message) => match message {
                    websocket::Message::CompilerInfo(info) => match info.data {
                        websocket::Data::UserFunction {
                            name,
                            pointer,
                            len,
                            number_of_arguments,
                        } => {
                            self.add_function(FunctionInfo {
                                name,
                                pointer,
                                len,
                            });
                        }

                        websocket::Data::Tokens {
                            file_name,
                            tokens,
                            token_line_column_map,
                        } => {
                            self.upsert_file(
                                file_name.clone(),
                                websocket::Data::Tokens {
                                    file_name,
                                    tokens,
                                    token_line_column_map,
                                },
                            );
                        }
                        websocket::Data::Ir {
                            file_name,
                            instructions,
                            token_range_to_ir_range,
                            function_pointer,
                        } => {
                            self.upsert_file(
                                file_name.clone(),
                                websocket::Data::Ir {
                                    file_name,
                                    instructions,
                                    token_range_to_ir_range,
                                    function_pointer,
                                },
                            );
                        }
                        websocket::Data::Arm {
                            function_pointer,
                            file_name,
                            instructions,
                            ir_to_machine_code_range,
                        } => {
                            self.upsert_file(
                                file_name.clone(),
                                websocket::Data::Arm {
                                    function_pointer,
                                    file_name,
                                    instructions,
                                    ir_to_machine_code_range,
                                },
                            );
                        }
                        _ => {}
                    },
                    _ => {}
                },
            },
        }
    }

    fn subscription(&self) -> Subscription<Message> {
        Subscription::run(websocket::connect).map(Message::Websocket)
    }

    fn view(&self) -> Element<Message> {
        if self.files.len() == 0 {
            return Column::new().into();
        }

        // 30002001C
        let found_file = if self.searching_address.len() == 9 {
            let searching_address_as_number_from_hex = usize::from_str_radix(&self.searching_address, 16).unwrap();
            let mut function_pointer = None;
            for function in self.functions.values() {
                if function.pointer <= searching_address_as_number_from_hex && function.pointer + function.len >= searching_address_as_number_from_hex {
                    function_pointer = Some(function.pointer);
                    break;
                }
            }
            let mut found_file = None;
            if let Some(function_pointer) = function_pointer {
                for file in self.files.iter() {
                    if file.ir_to_machine_code_range.contains_key(&function_pointer) {
                        found_file = Some(file);
                        break;
                    }
                }
            }

            found_file
        } else {
            None
        };

        let file = if let Some(file) = found_file {
            file
        }  else {
            &self.files[self.selected_file]
        };
        let tokens_with_index: Vec<(usize, &String)> =
            file.tokens.iter().enumerate().collect::<Vec<_>>();
        let lines = tokens_with_index
            .split(|(_, token)| token.as_str() == "\n")
            .map(|line| line.to_vec());

        let mut token_depth = vec![0; file.tokens.len()];

        let mut max_depth = 0;
        for (_, ranges) in file.token_range_to_ir_range.iter() {
            for ((start, end), _) in ranges.iter() {
                for i in *start..*end {
                    token_depth[i] += 1;
                    max_depth = max_depth.max(token_depth[i]);
                }
            }
        }

        let mut highlight_ir_range: Option<(usize, std::ops::Range<usize>)> = None;
        if let Some(hovered_token) = self.hovered_token {
            let mut chosen_token_range: Option<std::ops::Range<usize>> = None;
            for (function_pointer, ranges) in file.token_range_to_ir_range.iter() {
                for ((start, end), ir_range) in ranges.iter() {
                    if hovered_token >= *start && hovered_token < *end {
                        if let Some(range) = chosen_token_range.clone() {
                            // I want to choose the tightest range that contains the hovered token
                            // so if the length of current range is bigger than the new range
                            // I will replace it

                            if range.len() > (end - start)
                            {
                                highlight_ir_range =
                                    Some((*function_pointer, (ir_range.0)..(ir_range.1)));
                                chosen_token_range = Some(*start..*end);
                            }
                        } else {
                            highlight_ir_range =
                                Some((*function_pointer, (ir_range.0)..(ir_range.1)));
                            chosen_token_range = Some(*start..*end);
                        }
                    }
                }
            }
        }

        let mut highlight_machine_code_range: Option<(usize, std::ops::Range<usize>)> = None;
        if let Some(highlight_ir_range) = &highlight_ir_range {
            let (function_pointer, ir_range) = highlight_ir_range;
            if let Some(machine_code_ranges) = file.ir_to_machine_code_range.get(function_pointer) {
                for i in ir_range.clone().into_iter() {
                    for ((ir_index, (start, end))) in machine_code_ranges.iter() {
                        if i == *ir_index {
                            if highlight_machine_code_range.is_none() {
                                highlight_machine_code_range =
                                    Some((*function_pointer, *start..*end));
                            } else {
                                let (_, current_range) =
                                    highlight_machine_code_range.clone().unwrap();
                                if *start < current_range.start {
                                    highlight_machine_code_range =
                                        Some((*function_pointer, *start..current_range.end));
                                }
                                if *end > current_range.end {
                                    highlight_machine_code_range =
                                        Some((*function_pointer, current_range.start..*end));
                                }
                            }
                        }
                    }
                }
            }
        }

        let red_value_depth_increment = 1.0
            / (if max_depth == 0 {
                1.0
            } else {
                max_depth as f32
            });

        let code = Element::from(
            Column::with_children(lines.map(|line| {
                Element::from(Row::with_children(line.iter().map(|(i, token)| {
                    let mut color = [1.0, 1.0, 1.0];
                    if let Some(hovered_token) = self.hovered_token {
                        if hovered_token == *i {
                            color = [0.0, 1.0, 0.0];
                        }
                    }
                    Element::from(
                        Container::new(
                            mouse_area(text(token.to_string()).color(color))
                            .on_enter(Message::Hover(*i))
                            .on_exit(Message::Leave(*i)),
                        )
                        .style(move |_| Style {
                            ..Default::default()
                        }),
                    )
                })))
            }))
            .padding(20),
        );

        let ir = Column::with_children(file.instructions.iter().map(
            |(function_pointer, instructions)| {
                Element::from(
                    Column::with_children(instructions.iter().enumerate().map(
                        |(i, instruction)| {
                            let mut color = [1.0, 1.0, 1.0];
                            if let Some((hovered_function_pointer, hovered_range)) =
                                &highlight_ir_range
                            {
                                if *function_pointer == *hovered_function_pointer {
                                    if hovered_range.contains(&i) {
                                        color = [1.0, 0.0, 0.0];
                                    }
                                }
                            }
                            Element::from(
                                mouse_area(text(instruction.to_string()).color(color))
                                    .on_enter(Message::IrHover(*function_pointer, i))
                                    .on_exit(Message::IrLeave(*function_pointer, i)),
                            )
                        },
                    ))
                    .padding(20),
                )
            },
        ))
        .padding(20);

        let arm = Column::with_children(file.arm_instructions.iter().map(
            |(function_pointer, instructions)| {
                Element::from(
                    Column::with_children(instructions.iter().enumerate().map(
                        |(i, instruction)| {

                            let address = function_pointer + i * 4;
                            let mut color = [1.0, 1.0, 1.0];
                            if let Some((hovered_function_pointer, hovered_range)) =
                                &highlight_machine_code_range
                            {
                                if *function_pointer == *hovered_function_pointer {
                                    if hovered_range.contains(&i) {
                                        color = [1.0, 0.0, 0.0];
                                    }
                                }
                            }
                            if let Some(searching_address_as_number_from_hex) = usize::from_str_radix(&self.searching_address, 16).ok() {
                                if searching_address_as_number_from_hex == address {
                                    color = [0.0, 1.0, 0.0];
                                }
                            }

                            Element::from(mouse_area(text(format!("{:X}:     {}", address, instruction.to_string())).color(color)))
                        },
                    ))
                    .padding(20),
                )
            },
        ))
        .padding(20);
        row![
            column![
                button("Toggle selected file").on_press(Message::ToggleSelectedFile),
                text_input("", &self.searching_address).on_input(Message::AddressSearch),
            ].width(300),
            scrollable(code),
            scrollable(ir),
            scrollable(arm),
        ].padding(40)
        .into()
    }
}
