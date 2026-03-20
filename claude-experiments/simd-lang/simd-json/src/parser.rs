//! Stage 2 tape data structures and access API.

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapeType {
    Root = 0,
    OpenObject = 1,
    CloseObject = 2,
    OpenArray = 3,
    CloseArray = 4,
    String = 5,
    Int64 = 6,
    Double = 7,
    True = 8,
    False = 9,
    Null = 10,
}

/// A parsed JSON document in tape format.
/// Strings are stored as (input_offset, length) references — zero copy during parse.
/// Call `get_string(index, input)` to read/unescape strings on demand.
#[derive(Debug)]
pub struct Document {
    pub tape: Vec<u64>,
    /// String references: pairs of (input_offset: u32, length: u32)
    pub strings: Vec<u8>,
    pub numbers_i64: Vec<i64>,
    pub numbers_f64: Vec<f64>,
}

impl Document {
    #[inline(always)]
    pub fn tape_type(&self, index: usize) -> TapeType {
        let tag = (self.tape[index] >> 56) as u8;
        unsafe { std::mem::transmute(tag) }
    }

    #[inline(always)]
    pub fn tape_payload(&self, index: usize) -> u64 {
        self.tape[index] & 0x00FFFFFFFFFFFFFF
    }

    /// Get the string at tape index, reading from the original input.
    /// Handles unescaping on demand.
    pub fn get_string<'a>(&self, tape_index: usize, input: &'a [u8]) -> std::borrow::Cow<'a, str> {
        let offset = self.tape_payload(tape_index) as usize;
        let input_start = u32::from_le_bytes([
            self.strings[offset], self.strings[offset+1],
            self.strings[offset+2], self.strings[offset+3],
        ]) as usize;
        let len = u32::from_le_bytes([
            self.strings[offset+4], self.strings[offset+5],
            self.strings[offset+6], self.strings[offset+7],
        ]) as usize;

        let slice = &input[input_start..input_start + len];

        // Fast path: no escapes
        if !slice.contains(&b'\\') {
            std::borrow::Cow::Borrowed(
                std::str::from_utf8(slice).unwrap_or("")
            )
        } else {
            std::borrow::Cow::Owned(unescape(slice))
        }
    }

    /// Get string as raw bytes (no unescape, no copy).
    #[inline(always)]
    pub fn get_string_raw<'a>(&self, tape_index: usize, input: &'a [u8]) -> &'a [u8] {
        let offset = self.tape_payload(tape_index) as usize;
        let input_start = u32::from_le_bytes([
            self.strings[offset], self.strings[offset+1],
            self.strings[offset+2], self.strings[offset+3],
        ]) as usize;
        let len = u32::from_le_bytes([
            self.strings[offset+4], self.strings[offset+5],
            self.strings[offset+6], self.strings[offset+7],
        ]) as usize;
        &input[input_start..input_start + len]
    }

    #[inline(always)]
    pub fn get_i64(&self, tape_index: usize) -> i64 {
        let idx = self.tape_payload(tape_index) as usize;
        self.numbers_i64[idx]
    }

    #[inline(always)]
    pub fn get_f64(&self, tape_index: usize) -> f64 {
        let idx = self.tape_payload(tape_index) as usize;
        self.numbers_f64[idx]
    }

    #[inline(always)]
    pub fn get_partner(&self, tape_index: usize) -> usize {
        self.tape_payload(tape_index) as usize
    }

    pub fn dump(&self, input: &[u8]) {
        for i in 0..self.tape.len() {
            let typ = self.tape_type(i);
            let payload = self.tape_payload(i);
            match typ {
                TapeType::Root => println!("[{:>3}] Root → {}", i, payload),
                TapeType::OpenObject => println!("[{:>3}] {{ → close at {}", i, payload),
                TapeType::CloseObject => println!("[{:>3}] }} → open at {}", i, payload),
                TapeType::OpenArray => println!("[{:>3}] [ → close at {}", i, payload),
                TapeType::CloseArray => println!("[{:>3}] ] → close at {}", i, payload),
                TapeType::String => println!("[{:>3}] str: {:?}", i, self.get_string(i, input)),
                TapeType::Int64 => println!("[{:>3}] i64: {}", i, self.get_i64(i)),
                TapeType::Double => println!("[{:>3}] f64: {}", i, self.get_f64(i)),
                TapeType::True => println!("[{:>3}] true", i),
                TapeType::False => println!("[{:>3}] false", i),
                TapeType::Null => println!("[{:>3}] null", i),
            }
        }
    }
}

/// Borrowed view of a parsed document — zero allocation.
pub struct DocumentRef<'a> {
    pub tape: &'a [u64],
    pub strings: &'a [u8],
    pub numbers_i64: &'a [i64],
    pub numbers_f64: &'a [f64],
}

impl<'a> DocumentRef<'a> {
    #[inline(always)]
    pub fn tape_type(&self, index: usize) -> TapeType {
        unsafe { std::mem::transmute((self.tape[index] >> 56) as u8) }
    }

    #[inline(always)]
    pub fn tape_payload(&self, index: usize) -> u64 {
        self.tape[index] & 0x00FFFFFFFFFFFFFF
    }

    #[inline(always)]
    pub fn get_i64(&self, tape_index: usize) -> i64 {
        self.numbers_i64[self.tape_payload(tape_index) as usize]
    }

    #[inline(always)]
    pub fn get_f64(&self, tape_index: usize) -> f64 {
        self.numbers_f64[self.tape_payload(tape_index) as usize]
    }

    #[inline(always)]
    pub fn get_partner(&self, tape_index: usize) -> usize {
        self.tape_payload(tape_index) as usize
    }

    pub fn tape_len(&self) -> usize {
        self.tape.len()
    }
}

fn unescape(slice: &[u8]) -> String {
    let mut buf = Vec::with_capacity(slice.len());
    let mut i = 0;
    while i < slice.len() {
        if slice[i] == b'\\' && i + 1 < slice.len() {
            i += 1;
            match slice[i] {
                b'"' => buf.push(b'"'),
                b'\\' => buf.push(b'\\'),
                b'/' => buf.push(b'/'),
                b'n' => buf.push(b'\n'),
                b'r' => buf.push(b'\r'),
                b't' => buf.push(b'\t'),
                b'b' => buf.push(0x08),
                b'f' => buf.push(0x0C),
                b'u' => {
                    if i + 4 <= slice.len() {
                        if let Ok(hex) = std::str::from_utf8(&slice[i+1..i+5]) {
                            if let Ok(cp) = u32::from_str_radix(hex, 16) {
                                if let Some(ch) = char::from_u32(cp) {
                                    let mut tmp = [0u8; 4];
                                    buf.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
                                }
                            }
                        }
                        i += 4;
                    }
                }
                other => { buf.push(b'\\'); buf.push(other); }
            }
        } else {
            buf.push(slice[i]);
        }
        i += 1;
    }
    String::from_utf8(buf).unwrap_or_default()
}

#[derive(Debug)]
pub struct ParseError {
    pub message: String,
    pub position: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JSON error at byte {}: {}", self.position, self.message)
    }
}
