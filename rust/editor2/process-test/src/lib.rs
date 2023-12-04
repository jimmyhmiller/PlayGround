use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Read,
    str::{from_utf8, FromStr},
};

use framework::{
    app,
    serde_json::{self, json},
    App, Canvas, Size, Ui, WidgetData,
};
use lsp_types::{
    notification::{
        DidChangeTextDocument, DidOpenTextDocument, DidSaveTextDocument, Initialized, Notification,
    },
    request::{Initialize, Request, SemanticTokensFullRequest, WorkspaceSymbolRequest},
    ClientCapabilities, DidChangeTextDocumentParams, DidOpenTextDocumentParams,
    DidSaveTextDocumentParams, InitializeParams, InitializeResult, InitializedParams,
    MessageActionItemCapabilities, PartialResultParams, Position,
    PublishDiagnosticsClientCapabilities, Range, SemanticTokensParams,
    ShowDocumentClientCapabilities, ShowMessageRequestClientCapabilities, SymbolKind,
    SymbolKindCapability, TagSupport, TextDocumentClientCapabilities,
    TextDocumentContentChangeEvent, TextDocumentIdentifier, TextDocumentItem, Url,
    VersionedTextDocumentIdentifier, WindowClientCapabilities, WorkDoneProgressParams,
    WorkspaceClientCapabilities, WorkspaceFolder, WorkspaceSymbolClientCapabilities,
    WorkspaceSymbolParams, WorkspaceSymbolResolveSupportCapability, SymbolTag,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize, Debug)]
struct TokenRequestMeta {
    path: String,
    document_version: usize,
}

#[derive(Clone, Deserialize, Serialize, Debug)]
struct Data {
    state: State,
    pending_tokens: Vec<(String, usize)>,
    message_type: HashMap<String, String>,
    last_request_id: usize,
    messages_by_type: HashMap<String, Vec<String>>,
    token_request_metadata: HashMap<String, TokenRequestMeta>,
    open_files: HashSet<String>,
    files_to_open: HashSet<OpenFileInfo>,
    widget_data: WidgetData,
    y_scroll_offset: f32,
}

#[derive(Copy, Clone, Deserialize, Serialize, PartialEq, Eq, Debug)]
enum State {
    Initializing,
    Initialized,
}

struct ProcessSpawner {
    state: Data,
    process_id: i32,
    root_path: String,
    remaining_message: String,
}

#[derive(Clone, Deserialize, Serialize)]
pub enum Edit {
    Insert(usize, usize, Vec<u8>),
    Delete(usize, usize),
}

#[derive(Serialize, Deserialize, Clone)]
struct EditWithPath {
    edit: Edit,
    path: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct MultiEditWithPath {
    version: usize,
    edits: Vec<Edit>,
    path: String,
}

#[derive(Clone, Deserialize, Serialize)]
struct TokensWithVersion {
    tokens: Vec<u64>,
    version: usize,
    path: String,
}

enum ParseError {
    NoClosingBrace(String),
    InvalidJson,
}

// TODO:
// I need to properly handle versions of tokens and make sure I always use the latest.
// I need to actually update my tokens myself and then get the refresh.

impl ProcessSpawner {
    fn parse_message(&self, message: &str) -> Result<Vec<serde_json::Value>, ParseError> {
        let mut results = vec![];
        if let Some(start_json_object) = message.find('{') {
            // TODO: This can crash
            // I need to investigate this closer
            // Probably need to hold onto the message until we get this?
            let last_close_brace = message
                .rfind('}')
                .ok_or(ParseError::NoClosingBrace(message.to_string()))?;
            let message = &message[start_json_object..last_close_brace + 1];
            let message = message.trim();
            let deserializer = serde_json::Deserializer::from_str(message);
            let iterator: serde_json::StreamDeserializer<
                '_,
                serde_json::de::StrRead<'_>,
                serde_json::Value,
            > = deserializer.into_iter::<serde_json::Value>();
            for item in iterator {
                let item = item.map_err(|_| ParseError::InvalidJson)?;
                results.push(item);
            }
        }

        Ok(results)
    }

    fn next_request_id(&mut self) -> String {
        self.state.last_request_id += 1;
        format!("client/{}", self.state.last_request_id)
    }

    fn request(&mut self, id: String, method: &str, params: &str) -> String {
        // construct the request and add the headers including content-length
        let body = format!(
            "{{\"jsonrpc\":\"{}\",\"id\":\"{}\",\"method\":\"{}\",\"params\":{}}}",
            "2.0", id, method, params
        );
        let content_length = body.len();
        let headers = format!("Content-Length: {}\r\n\r\n", content_length);
        format!("{}{}", headers, body)
    }

    fn send_request(&mut self, method: &str, params: &str) -> String {
        let id = self.next_request_id();
        let request = self.request(id.clone(), method, params);
        self.state
            .message_type
            .insert(id.clone(), method.to_string());
        self.send_message(self.process_id, request);
        id
    }

    fn notification(&self, method: &str, params: &str) -> String {
        let body = format!(
            "{{\"jsonrpc\":\"{}\",\"method\":\"{}\",\"params\":{}}}",
            "2.0", method, params
        );
        let content_length = body.len();
        let headers = format!("Content-Length: {}\r\n\r\n", content_length);
        format!("{}{}", headers, body)
    }

    fn initialize_rust_analyzer(&mut self) {
        // println!("Initializing rust analyzer");
        let process_id = self.start_process(find_rust_analyzer());
        self.process_id = process_id;

        #[allow(deprecated)]
        // Root path is deprecated, but I also need to specify it
        // TODO: set initialization options like rust-analyzer.semanticHighlighting.punctuation.enable
        let mut initialize_params = InitializeParams {
            process_id: Some(self.process_id as u32),
            root_path: Some(self.root_path.to_string()),
            root_uri: Some(Url::from_str(&format!("file://{}", self.root_path)).unwrap()),
            initialization_options: Some(json!({
                "workspace":{
                    "symbol": {
                        "search": {
                            "limit": 10000,
                        }
                    }
                }
            })),
            capabilities: ClientCapabilities::default(),
            trace: None,
            workspace_folders: Some(vec![WorkspaceFolder {
                uri: Url::from_str(&format!("file://{}", self.root_path)).unwrap(),
                name: "editor2".to_string(),
            }]),
            client_info: None,
            locale: None,
        };
        initialize_params.capabilities.window = Some(WindowClientCapabilities {
            work_done_progress: Some(true),
            show_message: Some(ShowMessageRequestClientCapabilities {
                message_action_item: Some(MessageActionItemCapabilities {
                    additional_properties_support: Some(true),
                }),
            }),
            show_document: Some(ShowDocumentClientCapabilities { support: true }),
        });

        // TODO: Workspace symbols don't give me enough hierarchy
        // I need to get the hierarchy from textDocument.documentSymbol.
        // Hopefully that will give me enough information to show the kinds
        // of things I'm looking for.
        let mut workspace_capabilities = WorkspaceClientCapabilities::default();
        workspace_capabilities.symbol = Some(WorkspaceSymbolClientCapabilities {
            dynamic_registration: None,
            symbol_kind: Some(SymbolKindCapability {
                value_set: Some(vec![
                    SymbolKind::FILE,
                    SymbolKind::MODULE,
                    SymbolKind::NAMESPACE,
                    SymbolKind::PACKAGE,
                    SymbolKind::CLASS,
                    SymbolKind::METHOD,
                    SymbolKind::PROPERTY,
                    SymbolKind::FIELD,
                    SymbolKind::CONSTRUCTOR,
                    SymbolKind::ENUM,
                    SymbolKind::INTERFACE,
                    SymbolKind::FUNCTION,
                    SymbolKind::VARIABLE,
                    SymbolKind::CONSTANT,
                    SymbolKind::STRING,
                    SymbolKind::NUMBER,
                    SymbolKind::BOOLEAN,
                    SymbolKind::ARRAY,
                    SymbolKind::OBJECT,
                    SymbolKind::KEY,
                    SymbolKind::NULL,
                    SymbolKind::ENUM_MEMBER,
                    SymbolKind::STRUCT,
                    SymbolKind::EVENT,
                    SymbolKind::OPERATOR,
                    SymbolKind::TYPE_PARAMETER,
                ]),
            }),
            tag_support: Some(TagSupport { value_set: vec![
                SymbolTag::DEPRECATED,
            ] }),
            resolve_support: Some(WorkspaceSymbolResolveSupportCapability{
                properties: vec!["location.range".to_string()]
            }),
        });

        initialize_params.capabilities.workspace = Some(workspace_capabilities);

        let text_document = TextDocumentClientCapabilities {
            publish_diagnostics: Some(PublishDiagnosticsClientCapabilities {
                version_support: Some(true),
                ..PublishDiagnosticsClientCapabilities::default()
            }),
            ..TextDocumentClientCapabilities::default()
        };

        initialize_params.capabilities.text_document = Some(text_document);

        self.send_request(
            Initialize::METHOD,
            &serde_json::to_string(&initialize_params).unwrap(),
        );

        let params: <Initialized as Notification>::Params = InitializedParams {};
        let request = self.notification(
            Initialized::METHOD,
            &serde_json::to_string(&params).unwrap(),
        );
        self.send_message(self.process_id, request);
    }

    fn resolve_workspace_symbols(&mut self) {
        let params: <WorkspaceSymbolRequest as Request>::Params = WorkspaceSymbolParams {
            partial_result_params: PartialResultParams {
                partial_result_token: None,
            },
            work_done_progress_params: WorkDoneProgressParams {
                work_done_token: None,
            },
            query: "#".to_string(),
        };
        self.send_request(
            WorkspaceSymbolRequest::METHOD,
            &serde_json::to_string(&params).unwrap(),
        );
    }

    // TODO: I should probably ask the editor what files are open

    fn open_file(&mut self, file_info: &OpenFileInfo) {
        if !file_info.path.ends_with(".rs") {
            return;
        }

        let mut file = File::open(file_info.path.clone()).unwrap();
        let mut contents = String::new();
        let file_results = file.read_to_string(&mut contents);
        if let Err(err) = file_results {
            println!("Error reading file!: {} {}", file_info.path, err);
            return;
        }

        let params: <DidOpenTextDocument as Notification>::Params = DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri: Url::from_str(&format!("file://{}", &file_info.path)).unwrap(),
                language_id: "rust".to_string(),
                version: 0,
                text: contents,
            },
        };

        let notify = self.notification(
            DidOpenTextDocument::METHOD,
            &serde_json::to_string(&params).unwrap(),
        );
        self.send_message(self.process_id, notify);
        self.state.open_files.insert(file_info.path.clone());
        self.state.files_to_open.remove(file_info);
    }

    fn request_tokens(&mut self, path: &str, document_version: usize) {
        if self.state.state != State::Initialized {
            self.state
                .pending_tokens
                .push((path.to_string(), document_version));
            return;
        }
        let params: <SemanticTokensFullRequest as Request>::Params = SemanticTokensParams {
            text_document: TextDocumentIdentifier {
                uri: Url::from_str(&format!("file://{}", path)).unwrap(),
            },
            partial_result_params: PartialResultParams::default(),
            work_done_progress_params: WorkDoneProgressParams::default(),
        };

        let token_request = self.send_request(
            SemanticTokensFullRequest::METHOD,
            &serde_json::to_string(&params).unwrap(),
        );

        self.state.token_request_metadata.insert(
            token_request,
            TokenRequestMeta {
                path: path.to_string(),
                document_version,
            },
        );
    }

    fn update_document(&mut self, edits: &MultiEditWithPath) {
        let path = edits.path.clone();
        let version = edits.version;
        let mut content_changes = vec![];
        for edit in edits.edits.iter() {
            match edit {
                Edit::Insert(line, column, bytes) => {
                    // NOTE: for inserts we just always do the starting position
                    // I think we'd have do something else for replacements.
                    content_changes.push(TextDocumentContentChangeEvent {
                        range: Some(Range {
                            start: Position {
                                line: *line as u32,
                                character: *column as u32,
                            },
                            end: Position {
                                line: *line as u32,
                                character: *column as u32,
                            },
                        }),
                        range_length: None,
                        text: from_utf8(bytes).unwrap().to_string(),
                    })
                }
                Edit::Delete(line, column) => {
                    // TODO: Should we do a +1 here?
                    content_changes.push(TextDocumentContentChangeEvent {
                        range: Some(Range {
                            start: Position {
                                line: *line as u32,
                                character: *column as u32,
                            },
                            end: Position {
                                line: *line as u32,
                                character: *column as u32 + 1,
                            },
                        }),
                        range_length: None,
                        text: "".to_string(),
                    })
                }
            }
        }

        let params: <DidChangeTextDocument as Notification>::Params = DidChangeTextDocumentParams {
            text_document: VersionedTextDocumentIdentifier {
                uri: Url::from_str(&format!("file://{}", path)).unwrap(),
                version: version as i32,
            },
            content_changes,
        };
        let request = self.notification(
            DidChangeTextDocument::METHOD,
            &serde_json::to_string(&params).unwrap(),
        );
        self.send_message(self.process_id, request);
    }

    fn initialized(&mut self) {
        // TODO: Get list of initial open files
        self.state.state = State::Initialized;
        self.resolve_workspace_symbols();
        // println!("Opening files: {:?}", self.state.files_to_open);
        for info in self.state.files_to_open.clone().iter() {
            self.open_file(info);
            self.request_tokens(&info.path, info.version);
        }
    }
}

#[derive(Clone, Deserialize, Serialize, Debug, PartialEq, Eq, Hash)]
struct OpenFileInfo {
    path: String,
    version: usize,
}

impl App for ProcessSpawner {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn start(&mut self) {
        self.subscribe("text_change_multi");
        self.subscribe("lith/open-file");
        self.subscribe("lith/save_file");
        self.initialize_rust_analyzer();
    }

    fn draw(&mut self) {
        let mut canvas = Canvas::new();

        let ui = Ui::new();
        let ui = ui.pane(
            self.state.widget_data.size,
            (0.0, self.state.y_scroll_offset),
            ui.list(
                (0.0, self.state.y_scroll_offset),
                self.state.messages_by_type.iter(),
                |ui, item| ui.container(ui.text(&format!("{}: {}", item.0, item.1.len()))),
            ),
        );
        ui.draw(&mut canvas);

        canvas.translate(0.0, 100.0);
        let ui = Ui::new();
        let ui = ui.text(&format!("{:#?}", self.state.open_files));
        ui.draw(&mut canvas);
    }

    fn on_click(&mut self, _x: f32, _y: f32) {
        self.resolve_workspace_symbols();
    }

    fn on_key(&mut self, _input: framework::KeyboardInput) {}

    fn on_scroll(&mut self, _x: f64, _y: f64) {
        self.state.y_scroll_offset += _y as f32;
        if self.state.y_scroll_offset > 0.0 {
            self.state.y_scroll_offset = 0.0;
        }
    }

    fn on_event(&mut self, kind: String, event: String) {
        match kind.as_str() {
            "text_change_multi" => {
                let edits: MultiEditWithPath = serde_json::from_str(&event).unwrap();

                // TODO: I need a list of file types for which
                // I have an lsp
                if !edits.path.ends_with(".rs") {
                    return;
                }
                if !self.state.open_files.contains(&edits.path) {
                    self.open_file(&OpenFileInfo {
                        path: edits.path.clone(),
                        version: 0,
                    });
                }
                self.update_document(&edits);
                self.request_tokens(&edits.path, edits.version);
            }
            "lith/open-file" => {
                let info: OpenFileInfo = serde_json::from_str(&event).unwrap();
                if self.state.open_files.contains(&info.path) {
                    return;
                }
                if !info.path.ends_with(".rs") {
                    return;
                }
                self.state.files_to_open.insert(info.clone());
                self.request_tokens(&info.path, info.version);
                if self.state.state == State::Initialized {
                    self.open_file(&info);
                }
            }
            "lith/save_file" => {
                let path = event;
                let params: <DidSaveTextDocument as Notification>::Params =
                    DidSaveTextDocumentParams {
                        text_document: TextDocumentIdentifier {
                            uri: Url::from_str(&format!("file://{}", path)).unwrap(),
                        },
                        // TODO: Send the text to be safe
                        text: None,
                    };

                let notify = self.notification(
                    DidSaveTextDocument::METHOD,
                    &serde_json::to_string(&params).unwrap(),
                );
                self.send_message(self.process_id, notify);
            }
            _ => {
                println!("Unknown event: {}", kind);
            }
        }
    }

    fn get_state(&self) -> String {
        serde_json::to_string(&self.state).unwrap()
    }

    fn on_process_message(&mut self, _process_id: i32, message: String) {
        let message = format!("{}{}", self.remaining_message, message);
        self.remaining_message = String::new();
        let messages = message.split("Content-Length");
        for message in messages {
            match self.parse_message(message) {
                Ok(messages) => {
                    for message in messages {
                        // println!("parsed message: {:?}", message);
                        // let method = message["method"].as_str();
                        if let Some(id) = message["id"].as_str() {
                            if let Some(method) = self.state.message_type.get(id) {
                                self.state
                                    .messages_by_type
                                    .entry(method.to_string())
                                    .or_default()
                                    .push(message.to_string());

                                if method == "textDocument/semanticTokens/full" {
                                    let meta = self.state.token_request_metadata.get(id).unwrap();
                                    self.send_event(
                                        "tokens_with_version",
                                        serde_json::to_string(&TokensWithVersion {
                                            tokens: get_token_data(message.clone()),
                                            version: meta.document_version,
                                            path: meta.path.clone(),
                                        })
                                        .unwrap(),
                                    );
                                }
                                if method == "workspace/symbol" {
                                    self.send_event(
                                        "workspace/symbols",
                                        message.get("result").unwrap().to_string(),
                                    );
                                }
                                if method == "initialize" {
                                    let result = message.get("result").unwrap();
                                    let parsed_message =
                                        serde_json::from_value::<InitializeResult>(result.clone())
                                            .unwrap();

                                    if let Some(token_provider) =
                                        parsed_message.capabilities.semantic_tokens_provider
                                    {
                                        match token_provider {
                                            lsp_types::SemanticTokensServerCapabilities::SemanticTokensOptions(options) => {
                                                self.send_event("token_options", serde_json::to_string(&options.legend).unwrap());
                                            }
                                            lsp_types::SemanticTokensServerCapabilities::SemanticTokensRegistrationOptions(options) => {
                                                self.send_event("token_options", serde_json::to_string(&options.semantic_tokens_options.legend).unwrap());
                                            },
                                        }
                                    }
                                }
                            }
                        } else {
                            // This isn't in response to a message we sent
                            let method = message["method"].as_str();
                            if let Some(method) = method {
                                match method {
                                    "$/progress" => {
                                        if let Some(100) = message
                                            .get("params")
                                            .and_then(|x| x.get("value"))
                                            .and_then(|x| x.get("percentage"))
                                            .and_then(|x| x.as_u64())
                                        {
                                            self.initialized();
                                        }
                                    }
                                    "textDocument/publishDiagnostics" => {
                                        let diagnostic =
                                            serde_json::to_string(&message.get("params")).unwrap();
                                        self.state
                                            .messages_by_type
                                            .entry("diagnostics".to_string())
                                            .or_default()
                                            .push(diagnostic.clone());
                                        self.send_event("diagnostics", diagnostic);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                Err(err) => match err {
                    ParseError::NoClosingBrace(message) => {
                        self.remaining_message = message.to_string();
                    }
                    ParseError::InvalidJson => {
                        println!("Invalid json: {}", message);
                    }
                },
            }
        }
        // println!("Process {} sent message {}", process_id, message);
    }

    fn set_state(&mut self, state: String) {
        let value: Data = serde_json::from_str(&state).unwrap();
        self.state.widget_data = value.widget_data;
    }

    fn on_size_change(&mut self, width: f32, height: f32) {
        self.state.widget_data.size.width = width;
        self.state.widget_data.size.height = height;
    }

    fn on_move(&mut self, x: f32, y: f32) {
        self.state.widget_data.position = framework::Position { x, y };
    }

    fn get_position(&self) -> framework::Position {
        self.state.widget_data.position
    }

    fn get_size(&self) -> Size {
        self.state.widget_data.size
    }

    fn get_initial_state(&self) -> String {
        let init = Self::init();
        serde_json::to_string(&init.state).unwrap()
    }
}

impl ProcessSpawner {
    fn init() -> Self {
        ProcessSpawner {
            state: Data {
                pending_tokens: Vec::new(),
                state: State::Initializing,
                message_type: HashMap::new(),
                token_request_metadata: HashMap::new(),
                open_files: HashSet::new(),
                files_to_open: HashSet::new(),
                last_request_id: 0,
                messages_by_type: HashMap::new(),
                widget_data: WidgetData::default(),
                y_scroll_offset: 0.0,
            },
            process_id: 0,
            root_path: "/Users/jimmyhmiller/Documents/Code/PlayGround/rust/editor2".to_string(),
            remaining_message: String::new(),
        }
    }
}

fn _extract_tokens(path: String, message: &serde_json::Value) -> Vec<u8> {
    let result = &message["result"];
    let data = &result["data"];
    serde_json::to_string(&json!({
        "path": path,
        "tokens": data,
    }))
    .unwrap()
    .into_bytes()
}

fn get_token_data(message: serde_json::Value) -> Vec<u64> {
    // TODO: This can error
    let result = &message["result"];
    let result = serde_json::from_value(result["data"].clone());
    if let Ok(result) = result {
        result
    } else {
        panic!("Failed to get tokens from {}", message);
    }
}

fn find_rust_analyzer() -> String {
    // let root = "/Users/jimmyhmiller/.vscode/extensions/";
    // let folder = fs::read_dir(root)
    //     .unwrap()
    //     .map(|res| res.map(|e| e.path()))
    //     .find(|path| {
    //         path.as_ref()
    //             .unwrap()
    //             .file_name()
    //             .unwrap()
    //             .to_str()
    //             .unwrap()
    //             .starts_with("rust-lang.rust-analyzer")
    //     })
    //     .unwrap()
    //     .unwrap();

    // format!("{}/server/rust-analyzer", folder.to_str().unwrap())
    "/Users/jimmyhmiller/Documents/Code/open-source/rust-analyzer/target/release/rust-analyzer".to_string()
}

app!(ProcessSpawner);
