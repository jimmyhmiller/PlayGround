use std::str::FromStr;

use framework::{app, macros::serde_json, App, Canvas};
use lsp_types::{
    notification::{Initialized, Notification, ShowMessage},
    request::{Initialize, Request, ShowMessageRequest, WorkDoneProgressCreate},
    ClientCapabilities, InitializeParams, InitializedParams, MessageActionItemCapabilities,
    ShowDocumentClientCapabilities, ShowMessageRequestClientCapabilities, Url,
    WindowClientCapabilities, WorkDoneProgressCreateParams, WorkspaceFolder,
};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Deserialize, Serialize)]
enum State {
    Init,
    Message,
    Recieve,
    Initialized,
    Progress,
}

struct JsonRpcRequest {
    jsonrpc: String,
    id: i32,
    method: String,
    params: String,
}

impl JsonRpcRequest {
    fn request(&self) -> String {
        // construct the request and add the headers including content-length
        let body = format!(
            "{{\"jsonrpc\":\"{}\",\"id\":{},\"method\":\"{}\",\"params\":{}}}",
            self.jsonrpc, self.id, self.method, self.params
        );
        let content_length = body.len();
        let headers = format!("Content-Length: {}\r\n\r\n", content_length);
        format!("{}{}", headers, body)
    }
    fn notification(&self) -> String {
        let body = format!(
            "{{\"jsonrpc\":\"{}\",\"method\":\"{}\",\"params\":{}}}",
            self.jsonrpc, self.method, self.params
        );
        let content_length = body.len();
        let headers = format!("Content-Length: {}\r\n\r\n", content_length);
        format!("{}{}", headers, body)
    }
}

struct ProcessSpawner {
    state: State,
    process_id: i32,
}


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
}

impl App for ProcessSpawner {
    type State = State;

    fn init() -> Self {
        ProcessSpawner {
            state: State::Init,
            process_id: 0,
        }
    }

    fn draw(&mut self) {
        let canvas = Canvas::new();
        canvas.draw_rect(0.0, 0.0, 100.0, 100.0);
    }

    fn on_click(&mut self, _x: f32, _y: f32) {
        println!("About to get value");
        let x = self.get_async_thing();
        println!("{:?}", x);

        let root_path = "/Users/jimmyhmiller/Documents/Code/PlayGround/rust/editor2";

        #[allow(deprecated)]
        // Root path is deprecated, but I also need to specify it
        let mut initialize_params = InitializeParams {
            process_id: Some(self.process_id as u32),
            root_path: Some(root_path.to_string()),
            root_uri: Some(Url::from_str(&format!("file://{}", root_path)).unwrap()),
            initialization_options: None,
            capabilities: ClientCapabilities::default(),
            trace: None,
            workspace_folders: Some(vec![WorkspaceFolder {
                uri: Url::from_str(&format!("file://{}", root_path)).unwrap(),
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
        let request = Initialize::METHOD;

        let json_rpc_request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: 1,
            method: request.to_string(),
            params: serde_json::to_string(&initialize_params).unwrap(),
        };

        let request = json_rpc_request.request();

        match self.state {
            State::Init => {
                let process_id = self.start_process("/Users/jimmyhmiller/.vscode/extensions/rust-lang.rust-analyzer-0.3.1541-darwin-arm64/server/rust-analyzer".to_string());
                self.process_id = process_id;
                self.state = State::Message;
            }
            State::Message => {
                self.state = State::Initialized;
                self.send_message(self.process_id, request)
            }
            State::Initialized => {
                let params: <Initialized as Notification>::Params = InitializedParams {};
                let json_rpc_request = JsonRpcRequest {
                    jsonrpc: "2.0".to_string(),
                    id: 1,
                    method: Initialized::METHOD.to_string(),
                    params: serde_json::to_string(&params).unwrap(),
                };
                let request = json_rpc_request.notification();
                self.send_message(self.process_id, request);
                self.state = State::Progress;
            }
            State::Progress => {
                self.state = State::Recieve;
            }
            State::Recieve => {
                println!("Noop")
            }
        }
    }

    fn on_key(&mut self, _input: framework::KeyboardInput) {}

    fn on_scroll(&mut self, _x: f64, _y: f64) {}

    fn get_state(&self) -> Self::State {
        self.state
    }

    fn on_process_message(&mut self, process_id: i32, message: String) {
        let messages = message.split("Content-Length");
        for message in messages {
            match self.parse_message(&message) {
                Ok(messages) => {
                    for message in messages {
                        let method = message["method"].as_str();
                        // let params = &message["params"];
                        println!("Method: {:?}", method);
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
}

app!(ProcessSpawner);
