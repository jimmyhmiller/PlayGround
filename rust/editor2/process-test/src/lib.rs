use framework::{app, App, Canvas, macros::serde_json};
use lsp_types::{InitializeParams, ClientCapabilities, request::{Initialize, Request}};
use serde::{Deserialize, Serialize};


#[derive(Copy, Clone, Deserialize, Serialize)]
enum State {
    Init,
    Message,
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
        let body = format!("{{\"jsonrpc\":\"{}\",\"id\":{},\"method\":\"{}\",\"params\":{}}}", self.jsonrpc, self.id, self.method, self.params);
        let content_length = body.len();
        let headers = format!("Content-Length: {}\r\n\r\n", content_length);
        format!("{}{}", headers, body)
    }
}

struct ProcessSpawner {
    state: State,
    process_id: i32,
}

impl App for ProcessSpawner {
    type State = State;

    fn init() -> Self {
        ProcessSpawner { 
            state: State::Init,
            process_id: 0
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

        #[allow(deprecated)]
        // Root path is deprecated, but I also need to specify it
        let initialize_params = InitializeParams {
            process_id: Some(self.process_id as u32),
            root_path: None,
            root_uri: None,
            initialization_options: None,
            capabilities: ClientCapabilities::default(),
            trace: None,
            workspace_folders: None,
            client_info: None,
            locale: None,
        };
        let request = Initialize::METHOD;
        let id = 1;
        
        let json_rpc_request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id,
            method: request.to_string(),
            params: serde_json::to_string(&initialize_params).unwrap(),
        };

        let request = json_rpc_request.request();


        match self.state {
            State::Init => {
                let process_id = self.start_process("/Users/jimmyhmiller/.vscode/extensions/rust-lang.rust-analyzer-0.3.1533-darwin-arm64/server/rust-analyzer".to_string());
                self.process_id = process_id;
                self.state = State::Message;
            },
            State::Message => {
                self.send_message(self.process_id, request)
            },
        }
        
        
    }

    fn on_key(&mut self, _input: framework::KeyboardInput) {
        
    }

    fn on_scroll(&mut self, _x: f64, _y: f64) {
    }

    fn get_state(&self) -> Self::State {
        self.state
    }

    fn set_state(&mut self, _state: Self::State) {
        
    }
    
}

app!(ProcessSpawner);
