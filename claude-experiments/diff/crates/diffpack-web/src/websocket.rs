//! Framework-neutral WebSocket framing and handshake support.

use std::io::Write;
use std::net::TcpStream;
use std::sync::{Arc, Mutex};

/// Broadcast fan-out for upgraded HMR WebSocket connections.
#[derive(Clone, Default)]
pub struct HmrHub {
    clients: Arc<Mutex<Vec<TcpStream>>>,
}

impl HmrHub {
    pub fn register(&self, stream: TcpStream) {
        self.clients.lock().unwrap().push(stream);
    }

    pub fn send(&self, json: &str) {
        let frame = text_frame(json.as_bytes());
        self.clients.lock().unwrap().retain_mut(|stream| {
            stream
                .write_all(&frame)
                .and_then(|()| stream.flush())
                .is_ok()
        });
    }

    pub fn send_to(&self, mut stream: &TcpStream, json: &str) {
        let frame = text_frame(json.as_bytes());
        let _ = stream.write_all(&frame).and_then(|()| stream.flush());
    }

    pub fn broadcast_reload(&self) {
        self.send(r#"{"type":"reload"}"#);
    }

    pub fn broadcast_rsc_refresh(&self) {
        self.send(r#"{"type":"rsc-refresh"}"#);
    }

    pub fn broadcast_build_error(&self, message: &str) {
        let message = serde_json::to_string(message).unwrap_or_else(|_| "null".to_string());
        self.send(&format!(
            "{{\"type\":\"build-error\",\"message\":{message}}}"
        ));
    }

    pub fn broadcast_build_ok(&self) {
        self.send(r#"{"type":"build-ok"}"#);
    }

    pub fn client_count(&self) -> usize {
        self.clients.lock().unwrap().len()
    }
}

pub fn text_frame(payload: &[u8]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(payload.len() + 10);
    frame.push(0x81);
    match payload.len() {
        length @ 0..=125 => frame.push(length as u8),
        length @ 126..=65535 => {
            frame.push(126);
            frame.extend_from_slice(&(length as u16).to_be_bytes());
        }
        length => {
            frame.push(127);
            frame.extend_from_slice(&(length as u64).to_be_bytes());
        }
    }
    frame.extend_from_slice(payload);
    frame
}

pub fn accept(key: &str) -> String {
    let mut input = key.to_string();
    input.push_str("258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    base64_encode(&sha1(input.as_bytes()))
}

fn sha1(message: &[u8]) -> [u8; 20] {
    let mut hash = [
        0x6745_2301u32,
        0xEFCD_AB89,
        0x98BA_DCFE,
        0x1032_5476,
        0xC3D2_E1F0,
    ];
    let bit_length = (message.len() as u64).wrapping_mul(8);
    let mut data = message.to_vec();
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_length.to_be_bytes());
    for block in data.chunks_exact(64) {
        let mut words = [0u32; 80];
        for (index, word) in block.chunks_exact(4).enumerate() {
            words[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..80 {
            words[index] =
                (words[index - 3] ^ words[index - 8] ^ words[index - 14] ^ words[index - 16])
                    .rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (hash[0], hash[1], hash[2], hash[3], hash[4]);
        for (index, &word) in words.iter().enumerate() {
            let (function, constant) = match index {
                0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let next = a
                .rotate_left(5)
                .wrapping_add(function)
                .wrapping_add(e)
                .wrapping_add(constant)
                .wrapping_add(word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = next;
        }
        for (slot, value) in hash.iter_mut().zip([a, b, c, d, e]) {
            *slot = slot.wrapping_add(value);
        }
    }
    let mut output = [0; 20];
    for (index, word) in hash.iter().enumerate() {
        output[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    output
}

fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let triple = ((chunk[0] as u32) << 16)
            | ((*chunk.get(1).unwrap_or(&0) as u32) << 8)
            | (*chunk.get(2).unwrap_or(&0) as u32);
        output.push(ALPHABET[((triple >> 18) & 0x3f) as usize] as char);
        output.push(ALPHABET[((triple >> 12) & 0x3f) as usize] as char);
        output.push(if chunk.len() > 1 {
            ALPHABET[((triple >> 6) & 0x3f) as usize] as char
        } else {
            '='
        });
        output.push(if chunk.len() > 2 {
            ALPHABET[(triple & 0x3f) as usize] as char
        } else {
            '='
        });
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handshake_matches_the_rfc_example() {
        assert_eq!(
            accept("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    #[test]
    fn frame_uses_the_smallest_length_encoding() {
        assert_eq!(text_frame(b"hi"), [0x81, 2, b'h', b'i']);
        assert_eq!(text_frame(&vec![0; 126])[1..4], [126, 0, 126]);
    }
}
