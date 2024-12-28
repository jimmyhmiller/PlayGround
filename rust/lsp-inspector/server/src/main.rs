use matchit::Router;
use nonblock::NonBlockingReader;
use serde_json::Value;
use tiny_http::{Server, Response};
use std::{collections::HashMap, io::Write};


fn parse_message(data: String) -> Result<(String, String), String> {
    let split = data.split_once("\r\n\r\n");
    if split.is_none() {
        return Err(data);
    }
    let (headers, content) = split.unwrap();
    let split_headers = headers.split("\r\n").collect::<Vec<&str>>();
    let header_map = split_headers.iter().map(|header| {
        let split = header.split(":").collect::<Vec<&str>>();
        (split[0].trim().to_ascii_lowercase(), split[1].trim())
    }).collect::<HashMap<String, &str>>();
    if let Some(content_length) = header_map.get("content-length") {
        let content_length = content_length.parse::<usize>().unwrap();
        if content.len() < content_length {
            return Err(data);
        }
        let rest = content[content_length..].to_string();
        let content = content[..content_length].to_string();
        return Ok((content, rest));
    } else {
        return Err(data);
    }

}

fn parse_messages(data: String) -> Result<(Vec<String>, String), String> {
    let mut messages = Vec::new();
    let mut rest = data;
    loop {
        match parse_message(rest) {
            Ok((message, r)) => {
                messages.push(message);
                rest = r;
            }
            Err(data) => {
                if messages.is_empty() {
                    return Err(data);
                } else {
                    return Ok((messages, data));
                }
            }
        }
    }
}

enum Routes {
    SendRequest,
    DrainResponses,
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server = Server::http("0.0.0.0:8000").unwrap();
    
    let mut requests = Vec::new();
    let mut responses = Vec::new();

    let mut available_message_data = Vec::new();

    let mut process = std::process::Command::new("/Users/jimmyhmiller/.vscode/extensions/rust-lang.rust-analyzer-0.3.2228-darwin-arm64/server/rust-analyzer")
        .stdout(std::process::Stdio::piped())
        .stdin(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to execute process");

    let stderr = process.stderr.take().unwrap();
    let mut stderr = NonBlockingReader::from_fd(stderr).unwrap();

    let stdout = process.stdout.take().unwrap();
    let mut stdout = NonBlockingReader::from_fd(stdout).unwrap();

    let mut stdin = process.stdin.take().unwrap();
    
    
    let mut matcher = Router::new();
    matcher.insert("/request", Routes::SendRequest).ok();
    matcher.insert("/drain_responses", Routes::DrainResponses).ok();

    
    println!("Listening on port 8000");
    loop {

        let mut available_stderr_data = Vec::new();
        stderr.read_available(&mut available_stderr_data).unwrap();
        if !available_stderr_data.is_empty() {
            println!("stderr: {:?}", String::from_utf8_lossy(&available_stderr_data));
        }


        stdout.read_available(&mut available_message_data).unwrap();
        let cloned_data = available_message_data.clone();
        let string_data = String::from_utf8(cloned_data).unwrap();

        match parse_messages(string_data) {
            Ok((message, rest)) => {
                if !rest.is_empty() {
                    available_message_data.clear();
                    available_message_data.extend(rest.as_bytes());
                }
                for m in message {
                    let response = format!("Content-Length: {}\r\n\r\n{}", m.len(), m);
                    responses.push(response);
                }
            }
            Err(data) => {
                if !data.is_empty() {
                    available_message_data.clear();
                    available_message_data.extend(data.as_bytes());
                }
            }
        }
    

        if let Ok(Some(mut request)) = server.try_recv() {
            let route = matcher.at(request.url()).ok();
            match route.map(|r| r.value) {
                Some(Routes::SendRequest) => {
                    let body_reader = request.as_reader();
                    let mut body = String::new();
                    body_reader.read_to_string(&mut body).unwrap();
                    requests.push(body.clone());
                    stdin.write_all(format!("Content-Length: {}\r\n\r\n{}", body.len(), body.clone()).as_bytes()).unwrap();
                    // println!("received request! method: {:?}, url: {:?}, headers: {:?} body: {:?}",
                    //     request.method(),
                    //     request.url(),
                    //     request.headers(),
                    //     body,
                    // );
            
                    let response = Response::empty(204);
                    let _ = request.respond(response);
                }
                Some(Routes::DrainResponses) => {
                    if responses.is_empty() {
                        let response = Response::empty(204);
                        let _ = request.respond(response);
                        continue;
                    }
                    let response = Response::from_string(responses.join("\n"));
                    let _ = request.respond(response);
                    responses.clear();
                }
                None => {
                    let response = Response::empty(404);
                    let _ = request.respond(response);
                }
            }
        }
        
    }

}
