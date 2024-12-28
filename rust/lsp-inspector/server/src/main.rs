use matchit::Router;
use nonblock::NonBlockingReader;
use tiny_http::{Server, Response};
use std::io::Write;

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
        if !available_message_data.is_empty() {
            let data = String::from_utf8(available_message_data.clone()).unwrap();
            println!("stdout: {:?}", data);
            responses.push(data);
            available_message_data.clear();
        }

        if let Ok(Some(mut request)) = server.try_recv() {
            let route = matcher.at(request.url()).ok();
            match route.map(|r| r.value) {
                Some(Routes::SendRequest) => {
                    let body_reader = request.as_reader();
                    let mut body = String::new();
                    body_reader.read_to_string(&mut body).unwrap();
                    requests.push(body.clone());
                    eprintln!("received message: {}", body);
                    stdin.write_all(body.as_bytes()).unwrap();
                    let response = Response::empty(204);
                    let _ = request.respond(response);
                }
                Some(Routes::DrainResponses) => {
                    if responses.is_empty() {
                        let response = Response::empty(204);
                        let _ = request.respond(response);
                        continue;
                    }
                    let response = responses.join("");
                    eprintln!("responding with: {}", response);
                    let response = Response::from_string(response);
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
