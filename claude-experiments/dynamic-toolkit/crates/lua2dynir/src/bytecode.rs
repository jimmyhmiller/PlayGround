/// Lua 5.1 binary chunk parser.
///
/// Format reference: Lua 5.1 source `lundump.c`.
/// Header: 12 bytes, then recursive function prototypes.

/// A parsed Lua 5.1 binary chunk.
#[derive(Debug)]
pub struct Chunk {
    pub name: String,
    pub main: Proto,
}

/// A function prototype (the fundamental compilation unit in Lua).
#[derive(Debug, Clone)]
pub struct Proto {
    pub source: String,
    pub line_defined: u32,
    pub last_line_defined: u32,
    pub num_upvalues: u8,
    pub num_params: u8,
    pub is_vararg: u8,
    pub max_stack_size: u8,
    pub code: Vec<u32>,
    pub constants: Vec<Constant>,
    pub protos: Vec<Proto>,
    // debug info
    pub source_lines: Vec<u32>,
    pub locals: Vec<Local>,
    pub upvalue_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum Constant {
    Nil,
    Bool(bool),
    Number(f64),
    String(String),
}

#[derive(Debug, Clone)]
pub struct Local {
    pub name: String,
    pub start_pc: u32,
    pub end_pc: u32,
}

// Lua 5.1 opcodes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OpCode {
    Move = 0,
    LoadK = 1,
    LoadBool = 2,
    LoadNil = 3,
    GetUpval = 4,
    GetGlobal = 5,
    GetTable = 6,
    SetGlobal = 7,
    SetUpval = 8,
    SetTable = 9,
    NewTable = 10,
    Self_ = 11,
    Add = 12,
    Sub = 13,
    Mul = 14,
    Div = 15,
    Mod = 16,
    Pow = 17,
    Unm = 18,
    Not = 19,
    Len = 20,
    Concat = 21,
    Jmp = 22,
    Eq = 23,
    Lt = 24,
    Le = 25,
    Test = 26,
    TestSet = 27,
    Call = 28,
    TailCall = 29,
    Return = 30,
    ForLoop = 31,
    ForPrep = 32,
    TForLoop = 33,
    SetList = 34,
    Close = 35,
    Closure = 36,
    VarArg = 37,
}

impl OpCode {
    pub fn from_u8(v: u8) -> Option<OpCode> {
        if v <= 37 {
            Some(unsafe { std::mem::transmute(v) })
        } else {
            None
        }
    }
}

/// Instruction field extraction.
#[inline]
pub fn opcode(inst: u32) -> u8 {
    (inst & 0x3F) as u8
}

#[inline]
pub fn field_a(inst: u32) -> u16 {
    ((inst >> 6) & 0xFF) as u16
}

#[inline]
pub fn field_b(inst: u32) -> u16 {
    ((inst >> 23) & 0x1FF) as u16
}

#[inline]
pub fn field_c(inst: u32) -> u16 {
    ((inst >> 14) & 0x1FF) as u16
}

#[inline]
pub fn field_bx(inst: u32) -> u32 {
    inst >> 14
}

#[inline]
pub fn field_sbx(inst: u32) -> i32 {
    (inst >> 14) as i32 - 131071
}

/// Returns true if a B or C field refers to a constant (bit 8 set).
#[inline]
pub fn is_constant(field: u16) -> bool {
    field & 0x100 != 0
}

/// Extract the constant index from an RK field.
#[inline]
pub fn constant_index(field: u16) -> usize {
    (field & 0xFF) as usize
}

/// Binary chunk reader.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Reader { data, pos: 0 }
    }

    fn u8(&mut self) -> u8 {
        let v = self.data[self.pos];
        self.pos += 1;
        v
    }

    fn u32(&mut self) -> u32 {
        let b = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    fn int(&mut self, size: usize) -> u64 {
        let mut val = 0u64;
        for i in 0..size {
            val |= (self.data[self.pos + i] as u64) << (i * 8);
        }
        self.pos += size;
        val
    }

    fn f64(&mut self) -> f64 {
        let b = &self.data[self.pos..self.pos + 8];
        self.pos += 8;
        f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    }

    fn string(&mut self, size_t_size: usize) -> String {
        let len = self.int(size_t_size) as usize;
        if len == 0 {
            return String::new();
        }
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + len - 1])
            .unwrap_or("<invalid utf8>")
            .to_string();
        self.pos += len;
        s
    }
}

/// Parse a Lua 5.1 binary chunk from bytes.
pub fn parse(data: &[u8]) -> Result<Chunk, String> {
    let mut r = Reader::new(data);

    // Header (12 bytes)
    let sig = r.u32();
    if sig != 0x61754C1B {
        return Err(format!("bad signature: {:#010x}", sig));
    }
    let version = r.u8();
    if version != 0x51 {
        return Err(format!(
            "unsupported Lua version: {:#04x} (expected 0x51)",
            version
        ));
    }
    let format = r.u8(); // 0 = official
    if format != 0 {
        return Err(format!("unsupported format: {}", format));
    }
    let endian = r.u8(); // 1 = little-endian
    if endian != 1 {
        return Err("big-endian not supported".to_string());
    }
    let int_size = r.u8() as usize;
    let size_t_size = r.u8() as usize;
    let instruction_size = r.u8() as usize;
    if instruction_size != 4 {
        return Err(format!("unexpected instruction size: {}", instruction_size));
    }
    let number_size = r.u8() as usize;
    if number_size != 8 {
        return Err(format!("unexpected number size: {}", number_size));
    }
    let integral_flag = r.u8(); // 0 = floating point numbers
    let _ = (int_size, integral_flag);

    let main = parse_proto(&mut r, size_t_size)?;
    let name = main.source.clone();

    Ok(Chunk { name, main })
}

fn parse_proto(r: &mut Reader, size_t_size: usize) -> Result<Proto, String> {
    let source = r.string(size_t_size);
    let line_defined = r.u32();
    let last_line_defined = r.u32();
    let num_upvalues = r.u8();
    let num_params = r.u8();
    let is_vararg = r.u8();
    let max_stack_size = r.u8();

    // Instructions
    let code_len = r.u32() as usize;
    let mut code = Vec::with_capacity(code_len);
    for _ in 0..code_len {
        code.push(r.u32());
    }

    // Constants
    let const_len = r.u32() as usize;
    let mut constants = Vec::with_capacity(const_len);
    for _ in 0..const_len {
        let ty = r.u8();
        let c = match ty {
            0 => Constant::Nil,
            1 => Constant::Bool(r.u8() != 0),
            3 => Constant::Number(r.f64()),
            4 => Constant::String(r.string(size_t_size)),
            _ => return Err(format!("unknown constant type: {}", ty)),
        };
        constants.push(c);
    }

    // Nested protos
    let proto_len = r.u32() as usize;
    let mut protos = Vec::with_capacity(proto_len);
    for _ in 0..proto_len {
        protos.push(parse_proto(r, size_t_size)?);
    }

    // Debug info: source lines
    let lines_len = r.u32() as usize;
    let mut source_lines = Vec::with_capacity(lines_len);
    for _ in 0..lines_len {
        source_lines.push(r.u32());
    }

    // Debug info: locals
    let locals_len = r.u32() as usize;
    let mut locals = Vec::with_capacity(locals_len);
    for _ in 0..locals_len {
        let name = r.string(size_t_size);
        let start_pc = r.u32();
        let end_pc = r.u32();
        locals.push(Local {
            name,
            start_pc,
            end_pc,
        });
    }

    // Debug info: upvalue names
    let upval_len = r.u32() as usize;
    let mut upvalue_names = Vec::with_capacity(upval_len);
    for _ in 0..upval_len {
        upvalue_names.push(r.string(size_t_size));
    }

    Ok(Proto {
        source,
        line_defined,
        last_line_defined,
        num_upvalues,
        num_params,
        is_vararg,
        max_stack_size,
        code,
        constants,
        protos,
        source_lines,
        locals,
        upvalue_names,
    })
}
