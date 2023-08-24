use std::{
    collections::HashMap,
    fs::{self, File},
    io::Read,
    str::{from_utf8, FromStr},
};

use framework::{app, encode_base64, macros::serde_json::{self, json}, App, Canvas, Ui, Size};
use lsp_types::{
    notification::{DidChangeTextDocument, DidOpenTextDocument, Initialized, Notification},
    request::{Initialize, Request, SemanticTokensFullRequest, WorkspaceSymbolRequest},
    ClientCapabilities, DidChangeTextDocumentParams, DidOpenTextDocumentParams, InitializeParams,
    InitializeResult, InitializedParams, MessageActionItemCapabilities, PartialResultParams,
    Position, Range, SemanticTokensParams, ShowDocumentClientCapabilities,
    ShowMessageRequestClientCapabilities, TextDocumentContentChangeEvent, TextDocumentIdentifier,
    TextDocumentItem, Url, VersionedTextDocumentIdentifier, WindowClientCapabilities,
    WorkDoneProgressParams, WorkspaceFolder, WorkspaceSymbolParams,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
struct Data {
    state: State,
    message_type: HashMap<String, String>,
    last_request_id: usize,
    messages_by_type: HashMap<String, Vec<String>>,
    token_request_to_file: HashMap<String, String>,
    open_files: Vec<String>,
    size: Size,
    y_scroll_offset: f32,
}

#[derive(Copy, Clone, Deserialize, Serialize)]
enum State {
    Initializing,
    Initialized,
}

struct ProcessSpawner {
    state: Data,
    process_id: i32,
    root_path: String,
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
// TODO:
// I need to properly handle versions of tokens and make sure I always use the latest.
// I need to actually update my tokens myself and then get the refresh.

impl ProcessSpawner {
    fn parse_message(
        &self,
        message: &str,
    ) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
        let mut results = vec![];
        if let Some(start_json_object) = message.find('{') {
            let message = &message[start_json_object..];
            let deserializer = serde_json::Deserializer::from_str(message);
            let iterator: serde_json::StreamDeserializer<
                '_,
                serde_json::de::StrRead<'_>,
                serde_json::Value,
            > = deserializer.into_iter::<serde_json::Value>();
            for item in iterator {
                results.push(item?);
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
        self.state.message_type.insert(id.clone(), method.to_string());
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
        let process_id = self.start_process(find_rust_analyzer());
        self.process_id = process_id;

        #[allow(deprecated)]
        // Root path is deprecated, but I also need to specify it
        let mut initialize_params = InitializeParams {
            process_id: Some(self.process_id as u32),
            root_path: Some(self.root_path.to_string()),
            root_uri: Some(Url::from_str(&format!("file://{}", self.root_path)).unwrap()),
            initialization_options: None,
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
            partial_result_params: PartialResultParams { partial_result_token: None },
            work_done_progress_params: WorkDoneProgressParams { work_done_token: None },
            query: "".to_string(),
        };
        self.send_request(
            WorkspaceSymbolRequest::METHOD,
            &serde_json::to_string(&params).unwrap(),
        );
    }

    fn open_file(&mut self, path: &str) {

        // read entire contents
        let mut file = File::open(path).unwrap();
        let mut contents = String::new();
        let file_results = file.read_to_string(&mut contents);
        if let Err(err) = file_results {
            println!("Error reading file!: {} {}", path, err);
            return;
        }

        let params: <DidOpenTextDocument as Notification>::Params = DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri: Url::from_str(&format!("file://{}", &path))
                .unwrap(),
                language_id: "rust".to_string(),
                version: 1,
                text: contents,
            },
        };

        let notify = self.notification(
            DidOpenTextDocument::METHOD,
            &serde_json::to_string(&params).unwrap(),
        );
        self.send_message(self.process_id, notify);
    }

    fn request_tokens(&mut self, path: &str) {
        let params: <SemanticTokensFullRequest as Request>::Params = SemanticTokensParams {
            text_document: TextDocumentIdentifier {
                uri: Url::from_str(&format!("file://{}", path))
                .unwrap(),
            },
            partial_result_params: PartialResultParams::default(),
            work_done_progress_params: WorkDoneProgressParams::default(),
        };

        let token_request = self.send_request(
            SemanticTokensFullRequest::METHOD,
            &serde_json::to_string(&params).unwrap(),
        );

        self.state.token_request_to_file.insert(token_request, path.to_string());
    }

    fn update_document_insert(
        &mut self,
        path: &str,
        line: usize,
        column: usize,
        bytes: Vec<u8>,
    ) {

        // TODO: This assumes there are no new lines
        assert!(if bytes.len() > 1 {
            bytes.contains(&b'\n') == false
        } else {
            true
        });
        let params: <DidChangeTextDocument as Notification>::Params = DidChangeTextDocumentParams {
            text_document: VersionedTextDocumentIdentifier {
                uri: Url::from_str(&format!("file://{}", path))
                    .unwrap(),
                version: 0,
            },
            content_changes: vec![TextDocumentContentChangeEvent {
                range: Some(Range {
                    start: Position {
                        line: line as u32,
                        character: column as u32,
                    },
                    end: Position {
                        line: line as u32,
                        character: (column + bytes.len()) as u32,
                    },
                }),
                range_length: None,
                text: from_utf8(&bytes).unwrap().to_string(),
            }],
        };
        let request = self.notification(
            DidChangeTextDocument::METHOD,
            &serde_json::to_string(&params).unwrap(),
        );
        self.send_message(self.process_id, request);
    }

    fn update_document_delete(&mut self, path: &str, line: usize, column: usize) {
        let params: <DidChangeTextDocument as Notification>::Params = DidChangeTextDocumentParams {
            text_document: VersionedTextDocumentIdentifier {
                uri: Url::from_str(&format!("file://{}", path))
                    .unwrap(),
                version: 0,
            },
            content_changes: vec![TextDocumentContentChangeEvent {
                range: Some(Range {
                    start: Position {
                        line: line as u32,
                        character: column as u32,
                    },
                    end: Position {
                        line: line as u32,
                        character: column as u32 + 1,
                    },
                }),
                range_length: None,
                text: "".to_string(),
            }],
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
        for file in self.state.open_files.clone().iter() {
            self.request_tokens(file);
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
struct OpenFileInfo {
    path: String,
}

impl App for ProcessSpawner {
    type State = Data;

    fn init() -> Self {
        let mut me = ProcessSpawner {
            state: Data {
                state: State::Initializing,
                message_type: HashMap::new(),
                token_request_to_file: HashMap::new(),
                open_files: Vec::new(),
                last_request_id: 0,
                messages_by_type: HashMap::new(),
                size: Size::default(),
                y_scroll_offset: 0.0,
            },
            process_id: 0,
            root_path: "/Users/jimmyhmiller/Documents/Code/PlayGround/rust/editor2".to_string(),
        };
        me.subscribe("text_change");
        me.subscribe("lith/open-file");
        me.initialize_rust_analyzer();
        me
    }

    fn draw(&mut self) {

        let mut canvas = Canvas::new();

        let ui = Ui::new();
        let ui = ui.pane(
            self.state.size,
            (0.0, self.state.y_scroll_offset),
            ui.list(self.state.messages_by_type.iter(), |ui, item|
                ui.container(ui.text(&format!("{}: {}", item.0, item.1.len())))
            ),
        );
        ui.draw(&mut canvas);
    }

    fn on_click(&mut self, _x: f32, _y: f32) {
        // self.request_tokens();
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
            "text_change" => {
                let edit: EditWithPath = serde_json::from_str(&event).unwrap();
                match edit.edit {
                    Edit::Insert(line, column, bytes) => {
                        self.update_document_insert(&edit.path, line, column, bytes);
                        self.request_tokens(&edit.path);
                    }
                    Edit::Delete(line, column) => {
                        self.update_document_delete(&edit.path, line, column);
                        self.request_tokens(&edit.path);
                    }
                }
            }
            "lith/open-file" => {
                let info: OpenFileInfo = serde_json::from_str(&event).unwrap();
                self.state.open_files.push(info.path.clone());
                self.open_file(&info.path);
                self.request_tokens(&info.path);
            }
            _ => {
                println!("Unknown event: {}", kind);
            }
        }
    }

    fn get_state(&self) -> Self::State {
        self.state.clone()
    }

    fn on_process_message(&mut self, _process_id: i32, message: String) {
        let messages = message.split("Content-Length");
        for message in messages {
            match self.parse_message(message) {
                Ok(messages) => {
                    for message in messages {
                        // let method = message["method"].as_str();
                        if let Some(id) = message["id"].as_str() {
                            if let Some(method) = self.state.message_type.get(id) {

                                if let Some(messages) = self.state.messages_by_type.get_mut(method) {
                                    messages.push(message.to_string());
                                } else {
                                    self.state.messages_by_type.insert(method.to_string(), vec![message.to_string()]);
                                }


                                // TODO: Need to correlate this with file
                                if method == "textDocument/semanticTokens/full" {
                                    let path = self.state.token_request_to_file.get(id).unwrap();
                                    self.send_event(
                                        "tokens",
                                        encode_base64(&extract_tokens(path.clone(), &message)),
                                    );
                                }
                                if method == "workspace/symbol" {
                                    self.send_event("workspace/symbols", message.to_string());
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
                                if method == "$/progress" {
                                    if let Some(100) = message.get("params")
                                        .and_then(|x| x.get("value"))
                                        .and_then(|x| x.get("percentage"))
                                        .and_then(|x| x.as_u64()) {
                                            self.initialized();
                                        }
                                }
                            }

                        }
                    } 
                }
                Err(err) => {
                    println!("Error: {}", err);
                    println!("Message: {}", message);
                }
            }
        }
        // println!("Process {} sent message {}", process_id, message);
    }

    fn set_state(&mut self, state: Self::State) {
        println!("Setting state, {:?}", state.size);
        self.state.size = state.size;
    }

    fn on_size_change(&mut self, width: f32, height: f32) {
        self.state.size.width = width;
        self.state.size.height = height;
    }
}

fn extract_tokens(path: String, message: &serde_json::Value) -> Vec<u8> {
    let result = &message["result"];
    let data = &result["data"];
    serde_json::to_string(&json!({
        "path": path,
        "tokens": data,
    })).unwrap().into_bytes()
}

fn find_rust_analyzer() -> String {
    let root = "/Users/jimmyhmiller/.vscode/extensions/";
    let folder = fs::read_dir(root)
        .unwrap()
        .map(|res| res.map(|e| e.path()))
        .find(|path| {
            path.as_ref()
                .unwrap()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("rust-lang.rust-analyzer")
        })
        .unwrap()
        .unwrap();

    format!("{}/server/rust-analyzer", folder.to_str().unwrap())
}

app!(ProcessSpawner);
