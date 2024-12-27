use matchit::Router;
use nonblock::NonBlockingReader;
use serde_json::Value;
use tiny_http::{Server, Response};
use std::io::{BufRead, BufReader, Read, Write};


fn parse_message<T: Read>(reader: &mut BufReader<T>) -> Result<Value, Box<dyn std::error::Error>> {
    loop {
        let mut headers = String::new();

        // Read headers until an empty line (\r\n)
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap() == 0 {
                // EOF reached
                return Err("EOF reached".into());
            }

            if line == "\r\n" {
                break;
            }
            headers.push_str(&line);
        }

        // Parse Content-Length from headers
        let content_length = headers
            .lines()
            .find_map(|line| {
                if line.to_lowercase().starts_with("content-length:") {
                    line.split_once(':').map(|(_, v)| v.trim().parse::<usize>().ok()).flatten()
                } else {
                    None
                }
            })
            .expect("Content-Length header is missing");

        // Read the content part of the message
        let mut content = vec![0; content_length];
        reader.read_exact(&mut content)?;
        let content = String::from_utf8(content)?;

        // Parse the JSON content
        match serde_json::from_str::<Value>(&content) {
            Ok(json) => {
                return Ok(json);
            }
            Err(e) => eprintln!("Invalid JSON content: {}", e),
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

    let mut process = std::process::Command::new("/Users/jimmyhmiller/.vscode/extensions/rust-lang.rust-analyzer-0.3.2196-darwin-arm64/server/rust-analyzer")
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
        if !available_message_data.is_empty() {
            println!("stderr: {:?}", String::from_utf8_lossy(&available_stderr_data));
        }


        stdout.read_available(&mut available_message_data).unwrap();
        let cloned_data = available_message_data.clone();
        let mut reader = BufReader::new(cloned_data.as_slice());

        if let Ok(message) = parse_message(&mut reader) {
            println!("message: {:?}", message);
            available_message_data.clear();
            reader.read_to_end(&mut available_message_data).unwrap();
            let message = message.to_string();
            let message = format!("Content-Length: {}\r\n\r\n{}", message.len(), message);
            responses.push(message.clone());
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
