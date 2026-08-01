//! Small framework-neutral HTTP/1.1 parsing helpers for development servers.

use std::io::{BufRead, Read};

pub fn read_head(reader: &mut impl BufRead) -> Result<(String, Vec<(String, String)>), String> {
    let mut start_line = String::new();
    loop {
        start_line.clear();
        let read = reader
            .read_line(&mut start_line)
            .map_err(|error| format!("cannot read request line: {error}"))?;
        if read == 0 {
            return Err("client closed before sending a request".to_string());
        }
        if !start_line.trim().is_empty() {
            break;
        }
    }
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        let read = reader
            .read_line(&mut line)
            .map_err(|error| format!("cannot read header line: {error}"))?;
        if read == 0 {
            break;
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            headers.push((name.trim().to_string(), value.trim().to_string()));
        }
    }
    Ok((
        start_line.trim_end_matches(['\r', '\n']).to_string(),
        headers,
    ))
}

pub fn read_body(reader: &mut impl Read, headers: &[(String, String)]) -> Result<Vec<u8>, String> {
    let length = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = vec![0; length];
    if length > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|error| format!("cannot read request body: {error}"))?;
    }
    Ok(body)
}

pub fn parse_request_line(line: &str) -> Result<(String, String), String> {
    let mut parts = line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| "empty request line".to_string())?
        .to_string();
    let target = parts
        .next()
        .ok_or_else(|| "request line has no target".to_string())?
        .to_string();
    Ok((method, target))
}

pub fn decode_chunked(mut input: &[u8]) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    loop {
        let line_end = find_subsequence(input, b"\r\n")
            .ok_or_else(|| "truncated chunk size line".to_string())?;
        let size_line = std::str::from_utf8(&input[..line_end])
            .map_err(|_| "non-utf8 chunk size".to_string())?;
        let size_hex = size_line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_hex, 16)
            .map_err(|error| format!("bad chunk size {size_hex:?}: {error}"))?;
        input = &input[line_end + 2..];
        if size == 0 {
            break;
        }
        if input.len() < size {
            return Err("truncated chunk data".to_string());
        }
        output.extend_from_slice(&input[..size]);
        input = &input[size..];
        if input.len() >= 2 {
            input = &input[2..];
        }
    }
    Ok(output)
}

pub fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_a_request_and_body() {
        let mut request = Cursor::new(b"POST /x HTTP/1.1\r\nContent-Length: 2\r\n\r\nhi");
        let (line, headers) = read_head(&mut request).unwrap();
        assert_eq!(
            parse_request_line(&line).unwrap(),
            ("POST".into(), "/x".into())
        );
        assert_eq!(read_body(&mut request, &headers).unwrap(), b"hi");
    }

    #[test]
    fn decodes_chunk_extensions() {
        assert_eq!(
            decode_chunked(b"4;name=value\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n").unwrap(),
            b"Wikipedia"
        );
    }
}
