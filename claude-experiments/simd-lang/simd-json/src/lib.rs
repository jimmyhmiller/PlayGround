pub mod simd_stage1;
pub mod parser;

pub fn pad_input(input: &[u8]) -> Vec<u8> {
    let padded_len = ((input.len() + 63) / 64) * 64;
    let mut buf = vec![0u8; padded_len];
    buf[..input.len()].copy_from_slice(input);
    buf
}

pub fn parse(input: &[u8]) -> Result<parser::Document, parser::ParseError> {
    let padded = pad_input(input);
    parse_padded(&padded, input.len())
}

pub fn parse_padded(padded_input: &[u8], original_len: usize) -> Result<parser::Document, parser::ParseError> {
    let mut positions = vec![0i32; original_len];
    let num = simd_stage1::find_structural(padded_input, &mut positions);
    build_tape(padded_input, &positions[..num])
}

pub struct Parser {
    positions: Vec<i32>,
    tape: Vec<u64>,
    strings: Vec<u8>,
    numbers_i64: Vec<i64>,
    numbers_f64: Vec<f64>,
    stack: Vec<u32>,
}

impl Parser {
    pub fn new(max_input_size: usize) -> Self {
        Parser {
            positions: vec![0i32; max_input_size],
            tape: Vec::with_capacity(max_input_size / 4),
            strings: Vec::with_capacity(max_input_size),
            numbers_i64: Vec::with_capacity(max_input_size / 16),
            numbers_f64: Vec::new(),
            stack: Vec::with_capacity(64),
        }
    }

    pub fn parse(&mut self, padded_input: &[u8]) -> Result<parser::DocumentRef<'_>, parser::ParseError> {
        let num = simd_stage1::find_structural(padded_input, &mut self.positions);
        self.tape.clear();
        self.strings.clear();
        self.numbers_i64.clear();
        self.numbers_f64.clear();
        self.stack.clear();
        build_tape_into(padded_input, &self.positions[..num],
            &mut self.tape, &mut self.strings, &mut self.numbers_i64,
            &mut self.numbers_f64, &mut self.stack)?;
        Ok(parser::DocumentRef {
            tape: &self.tape, strings: &self.strings,
            numbers_i64: &self.numbers_i64, numbers_f64: &self.numbers_f64,
        })
    }
}

fn build_tape(input: &[u8], positions: &[i32]) -> Result<parser::Document, parser::ParseError> {
    let n = positions.len();
    let mut tape = Vec::with_capacity(n + 2);
    let mut strings = Vec::with_capacity(n * 8);
    let mut numbers_i64 = Vec::with_capacity(n / 8);
    let mut numbers_f64 = Vec::new();
    let mut stack = Vec::with_capacity(64);
    build_tape_into(input, positions, &mut tape, &mut strings, &mut numbers_i64, &mut numbers_f64, &mut stack)?;
    Ok(parser::Document { tape, strings, numbers_i64, numbers_f64 })
}

fn build_tape_into(
    input: &[u8], positions: &[i32],
    tape: &mut Vec<u64>, strings: &mut Vec<u8>,
    numbers_i64: &mut Vec<i64>, numbers_f64: &mut Vec<f64>,
    stack: &mut Vec<u32>,
) -> Result<(), parser::ParseError> {
    let n = positions.len();
    let inp = input.as_ptr();
    let inp_len = input.len();
    tape.reserve(n + 2);
    strings.reserve(n * 8);
    tape.push(0);
    stack.push(0);

    let mut si = 0usize;
    while si < n {
        let pos = unsafe { *positions.get_unchecked(si) } as usize;
        let ch = unsafe { *inp.add(pos) };
        si += 1;

        match ch {
            b'{' => { let ti = tape.len() as u32; tape.push(0); stack.push(ti); }
            b'}' => {
                let open = stack.pop().unwrap() as usize;
                let close = tape.len();
                tape.push(entry(2, open as u64));
                unsafe { *tape.get_unchecked_mut(open) = entry(1, close as u64); }
            }
            b'[' => {
                let ti = tape.len() as u32; tape.push(0); stack.push(ti);
                if si < n && unsafe { *inp.add(*positions.get_unchecked(si) as usize) } != b']' {
                    let vs = skip_ws(inp, inp_len, pos + 1);
                    write_atom(inp, inp_len, vs, tape, strings, numbers_i64, numbers_f64);
                }
            }
            b']' => {
                let open = stack.pop().unwrap() as usize;
                let close = tape.len();
                tape.push(entry(4, close as u64));
                unsafe { *tape.get_unchecked_mut(open) = entry(3, close as u64); }
            }
            b':' => {
                // Key: fast forward+backward for quotes
                let prev_end = if si >= 2 { (unsafe { *positions.get_unchecked(si - 2) }) as usize + 1 } else { 0 };
                write_key(inp, prev_end, pos, tape, strings);
                // Value
                let vs = skip_ws(inp, inp_len, pos + 1);
                write_atom(inp, inp_len, vs, tape, strings, numbers_i64, numbers_f64);
            }
            b',' => {
                if si < n && unsafe { *inp.add(*positions.get_unchecked(si) as usize) } != b':' {
                    let vs = skip_ws(inp, inp_len, pos + 1);
                    write_atom(inp, inp_len, vs, tape, strings, numbers_i64, numbers_f64);
                }
            }
            _ => {}
        }
    }
    let rc = tape.len();
    tape.push(entry(0, 0));
    tape[0] = entry(0, rc as u64);
    Ok(())
}

#[inline(always)] fn entry(t: u64, p: u64) -> u64 { (t << 56) | (p & 0x00FFFFFFFFFFFFFF) }

#[inline(always)]
fn skip_ws(inp: *const u8, len: usize, mut p: usize) -> usize {
    // Fast path: first byte is not whitespace (most common)
    if p < len {
        let b = unsafe { *inp.add(p) };
        if b != b' ' && b != b'\n' && b != b'\t' && b != b'\r' { return p; }
        p += 1;
        if p < len {
            let b = unsafe { *inp.add(p) };
            if b != b' ' && b != b'\n' && b != b'\t' && b != b'\r' { return p; }
            p += 1;
            while p < len {
                let b = unsafe { *inp.add(p) };
                if b != b' ' && b != b'\n' && b != b'\t' && b != b'\r' { return p; }
                p += 1;
            }
        }
    }
    p
}

#[inline(always)]
fn write_key(inp: *const u8, start: usize, colon: usize, tape: &mut Vec<u64>, strings: &mut Vec<u8>) {
    // Fast forward to opening quote
    let mut p = start;
    let b = unsafe { *inp.add(p) };
    if b != b'"' {
        p += 1;
        if unsafe { *inp.add(p) } != b'"' {
            while unsafe { *inp.add(p) } != b'"' { p += 1; }
        }
    }
    let key_start = p + 1;
    // Backward from colon to closing quote
    let mut end = colon - 1;
    while unsafe { *inp.add(end) } != b'"' { end -= 1; }
    write_str(tape, strings, key_start, end);
}

#[inline(always)]
fn write_atom(
    inp: *const u8, len: usize, vs: usize,
    tape: &mut Vec<u64>, strings: &mut Vec<u8>,
    ni: &mut Vec<i64>, nf: &mut Vec<f64>,
) {
    if vs >= len { return; }
    match unsafe { *inp.add(vs) } {
        b'"' => {
            let s = vs + 1;
            let e = find_q(inp, len, s);
            write_str(tape, strings, s, e);
        }
        b't' => tape.push(entry(8, 0)),
        b'f' => tape.push(entry(9, 0)),
        b'n' => tape.push(entry(10, 0)),
        b'0'..=b'9' | b'-' => write_num(inp, len, vs, tape, ni, nf),
        _ => {}
    }
}

#[inline(always)]
fn find_q(inp: *const u8, len: usize, s: usize) -> usize {
    let mut p = s;
    loop {
        if p >= len { return p; }
        let b = unsafe { *inp.add(p) };
        if b == b'"' { return p; }
        if b == b'\\' { p += 1; }
        p += 1;
    }
}

#[inline(always)]
fn write_str(tape: &mut Vec<u64>, strings: &mut Vec<u8>, s: usize, e: usize) {
    let off = strings.len();
    strings.extend_from_slice(&(s as u32).to_le_bytes());
    strings.extend_from_slice(&((e - s) as u32).to_le_bytes());
    tape.push(entry(5, off as u64));
}

#[inline(always)]
fn write_num(inp: *const u8, len: usize, s: usize, tape: &mut Vec<u64>, ni: &mut Vec<i64>, nf: &mut Vec<f64>) {
    let mut e = s;
    let mut fl = false;
    if e < len && unsafe { *inp.add(e) } == b'-' { e += 1; }
    while e < len && unsafe { *inp.add(e) }.is_ascii_digit() { e += 1; }
    if e < len && unsafe { *inp.add(e) } == b'.' { fl = true; e += 1; while e < len && unsafe { *inp.add(e) }.is_ascii_digit() { e += 1; } }
    if e < len && matches!(unsafe { *inp.add(e) }, b'e' | b'E') { fl = true; e += 1; if e < len && matches!(unsafe { *inp.add(e) }, b'+' | b'-') { e += 1; } while e < len && unsafe { *inp.add(e) }.is_ascii_digit() { e += 1; } }
    if fl {
        let st = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(inp.add(s), e - s)) };
        let v: f64 = st.parse().unwrap_or(0.0);
        let i = nf.len(); nf.push(v); tape.push(entry(7, i as u64));
    } else {
        let mut p = s; let neg = unsafe { *inp.add(p) } == b'-'; if neg { p += 1; }
        let mut v: i64 = 0;
        while p < e { v = v * 10 + (unsafe { *inp.add(p) } - b'0') as i64; p += 1; }
        if neg { v = -v; }
        let i = ni.len(); ni.push(v); tape.push(entry(6, i as u64));
    }
}
