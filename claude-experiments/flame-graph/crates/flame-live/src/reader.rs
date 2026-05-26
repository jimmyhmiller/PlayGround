//! Decoders for the two on-wire formats. Both are pull-based: caller passes a
//! `BufRead` and the callback fires per event. Errors short-circuit; EOF is
//! reported as `Ok(())`.

use std::io::{self, BufRead, Read};

use crate::event::LiveEvent;

#[derive(Debug)]
pub enum ReadError {
    Io(io::Error),
    Decode(String),
}

impl From<io::Error> for ReadError {
    fn from(e: io::Error) -> Self {
        ReadError::Io(e)
    }
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::Io(e) => write!(f, "io: {e}"),
            ReadError::Decode(s) => write!(f, "decode: {s}"),
        }
    }
}

impl std::error::Error for ReadError {}

/// Read length-prefixed postcard events. Loops until EOF or callback signals
/// stop (returns `false`).
pub fn read_binary_stream<R: Read>(
    mut r: R,
    mut on_event: impl FnMut(LiveEvent) -> bool,
) -> Result<(), ReadError> {
    let mut len_buf = [0u8; 4];
    let mut payload = Vec::with_capacity(256);
    loop {
        match r.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        }
        let len = u32::from_le_bytes(len_buf) as usize;
        // 64 MiB sanity cap — a single LiveEvent in practice is < 1 KiB. A
        // length this big means the stream is desynced and we should bail
        // rather than allocate.
        if len > 64 * 1024 * 1024 {
            return Err(ReadError::Decode(format!(
                "implausible event length {len} (stream desync?)"
            )));
        }
        payload.clear();
        payload.resize(len, 0);
        r.read_exact(&mut payload)?;
        let event: LiveEvent = postcard::from_bytes(&payload)
            .map_err(|e| ReadError::Decode(format!("postcard: {e}")))?;
        if !on_event(event) {
            return Ok(());
        }
    }
}

/// Read newline-delimited JSON events.
pub fn read_ndjson_stream<R: BufRead>(
    r: R,
    mut on_event: impl FnMut(LiveEvent) -> bool,
) -> Result<(), ReadError> {
    for line in r.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let event: LiveEvent = serde_json::from_str(&line)
            .map_err(|e| ReadError::Decode(format!("json: {e}")))?;
        if !on_event(event) {
            return Ok(());
        }
    }
    Ok(())
}
