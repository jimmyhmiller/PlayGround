use std::{
    collections::HashMap,
    fs::{self, File},
    io::Read,
    str::{from_utf8, FromStr},
};

use framework::{app, macros::serde_json, App, Canvas};
use lsp_types::{
    notification::{DidChangeTextDocument, DidOpenTextDocument, Initialized, Notification},
    request::{Initialize, Request, SemanticTokensFullRequest},
    ClientCapabilities, DidChangeTextDocumentParams, DidOpenTextDocumentParams, InitializeParams,
    InitializedParams, MessageActionItemCapabilities, PartialResultParams, Position, Range,
    SemanticTokensParams, ShowDocumentClientCapabilities, ShowMessageRequestClientCapabilities,
    TextDocumentContentChangeEvent, TextDocumentIdentifier, TextDocumentItem, Url,
    VersionedTextDocumentIdentifier, WindowClientCapabilities, WorkDoneProgressParams,
    WorkspaceFolder,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
struct Data {
    state: State,
    message_type: HashMap<usize, String>,
    last_request_id: usize,
}

#[derive(Copy, Clone, Deserialize, Serialize)]
enum State {
    Init,
    Message,
    Recieve,
    Initialized,
    Progress,
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

    fn update_state(&mut self, state: State) {
        self.state.state = state;
    }

    fn next_request_id(&mut self) -> usize {
        self.state.last_request_id += 1;
        self.state.last_request_id
    }

    fn request(&mut self, id: usize, method: &str, params: &str) -> String {
        // construct the request and add the headers including content-length
        let body = format!(
            "{{\"jsonrpc\":\"{}\",\"id\":{},\"method\":\"{}\",\"params\":{}}}",
            "2.0", id, method, params
        );
        let content_length = body.len();
        let headers = format!("Content-Length: {}\r\n\r\n", content_length);
        format!("{}{}", headers, body)
    }

    fn send_request(&mut self, method: &str, params: &str) {
        let id = self.next_request_id();
        let request = self.request(id, method, params);
        self.state.message_type.insert(id, method.to_string());
        self.send_message(self.process_id, request);
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

    fn open_file(&mut self) {
        let file = "/code/process-test/src/lib.rs";

        // read entire contents
        let mut file = File::open(file).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();

        let params: <DidOpenTextDocument as Notification>::Params =
            DidOpenTextDocumentParams {
                text_document: TextDocumentItem {
                    uri: Url::from_str(&format!(
                        "file://{}/process-test/src/lib.rs",
                        self.root_path
                    ))
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

    fn request_tokens(&mut self) {
        let params: <SemanticTokensFullRequest as Request>::Params = SemanticTokensParams {
            text_document: TextDocumentIdentifier {
                uri: Url::from_str(&format!(
                    "file://{}/process-test/src/lib.rs",
                    self.root_path
                ))
                .unwrap(),
            },
            partial_result_params: PartialResultParams::default(),
            work_done_progress_params: WorkDoneProgressParams::default(),
        };

        self.send_request(
            SemanticTokensFullRequest::METHOD,
            &serde_json::to_string(&params).unwrap(),
        );
    }

    fn update_document_insert(&mut self, root_path: &str, line: usize, column: usize, bytes: Vec<u8>) {
        let params: <DidChangeTextDocument as Notification>::Params =
            DidChangeTextDocumentParams {
                text_document: VersionedTextDocumentIdentifier {
                    uri: Url::from_str(&format!(
                        "file://{}/process-test/src/lib.rs",
                        root_path
                    ))
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
                            character: column as u32,
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

    fn update_document_delete(&mut self, root_path: &str, line: usize, column: usize) {
        let params: <DidChangeTextDocument as Notification>::Params =
            DidChangeTextDocumentParams {
                text_document: VersionedTextDocumentIdentifier {
                    uri: Url::from_str(&format!(
                        "file://{}/process-test/src/lib.rs",
                        root_path
                    ))
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
}

impl App for ProcessSpawner {
    type State = Data;

    fn init() -> Self {
        let mut me = ProcessSpawner {
            state: Data {
                state: State::Init,
                message_type: HashMap::new(),
                last_request_id: 0,

            },
            process_id: 0,
            root_path: "/Users/jimmyhmiller/Documents/Code/PlayGround/rust/editor2".to_string(),
        };
        me.subscribe("text_change".to_string());
        me.initialize_rust_analyzer();
        me.open_file();
        me
    }

    fn draw(&mut self) {
        let canvas = Canvas::new();
        canvas.draw_rect(0.0, 0.0, 100.0, 100.0);
    }

    fn on_click(&mut self, _x: f32, _y: f32) {
        self.request_tokens();
    }

    fn on_key(&mut self, _input: framework::KeyboardInput) {}

    fn on_scroll(&mut self, _x: f64, _y: f64) {}

    fn on_event(&mut self, kind: String, event: String) {
        let root_path = "/Users/jimmyhmiller/Documents/Code/PlayGround/rust/editor2";
        match kind.as_str() {
            "text_change" => {
                let edit: Edit = serde_json::from_str(&event).unwrap();
                match edit {
                    Edit::Insert(line, column, bytes) => {
                        self.update_document_insert(root_path, line, column, bytes);
                        self.request_tokens();
                    }
                    Edit::Delete(line, column) => {
                        self.update_document_delete(root_path, line, column);
                        self.request_tokens();
                    }
                }
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
            match self.parse_message(&message) {
                Ok(messages) => {
                    for message in messages {
                        // let method = message["method"].as_str();
                        if let Some(id) = message["id"].as_u64() {
                            if let Some(method) = self.state.message_type.get(&(id as usize)) {
                                if method == "textDocument/semanticTokens/full" {
                                    self.provide_bytes("tokens", &extract_tokens(&message));
                                }
                                println!("Method: {:?}", method);
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

    fn set_state(&mut self, _state: Self::State) {}

    fn on_size_change(&mut self, width: f32, height: f32) {}
}

fn extract_tokens(message: &serde_json::Value) -> Vec<u8> {
    let result = &message["result"];
    let data = &result["data"];
    serde_json::to_string(&data).unwrap().into_bytes()
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
