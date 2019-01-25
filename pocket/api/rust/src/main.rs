use std::io::Cursor;
use tiny_http::StatusCode;
use std::collections::HashMap;
use tiny_http::{Server, Response, Header};
use readability::extractor;
use url::Url;

fn html_response<S>(data: S) -> Response<Cursor<Vec<u8>>> where S: Into<String> {
    let data = data.into();
    let data_len = data.len();

    Response::new(
        StatusCode(200),
        vec![
            Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=UTF-8"[..]).unwrap()
        ],
        Cursor::new(data.into_bytes()),
        Some(data_len),
        None,
    )
}

fn main() {
    let base_url = Url::parse("http://localhost/").unwrap();
    let server = Server::http("0.0.0.0:8000").unwrap();

    for request in server.incoming_requests() {

        println!("{}", request.url());
        let parsed_url = base_url.join(request.url()).unwrap();
            
        let hash_query: HashMap<_, _> = parsed_url.query_pairs().into_owned().collect();
        match extractor::scrape(hash_query.get("url").unwrap()) {
            Ok(product) => {
                request.respond(html_response(product.content)).unwrap();
            },
            Err(_) => {
                request.respond(html_response("error")).unwrap();           
            }
        }
    }
}