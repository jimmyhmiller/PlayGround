// GC Types - Core type definitions for garbage collection

/// Built-in type tags for tagged pointers
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BuiltInTypes {
    Int,
    Float,
    String,
    Bool,
    Function,
    Closure,
    HeapObject,
    Null,
}

impl BuiltInTypes {
    pub fn null_value() -> isize {
        0b111
    }

    pub fn true_value() -> isize {
        Self::construct_boolean(true)
    }

    pub fn false_value() -> isize {
        Self::construct_boolean(false)
    }

    pub fn is_true(value: usize) -> bool {
        value == Self::true_value() as usize
    }

    pub fn tag(&self, value: isize) -> isize {
        let value = value << 3;
        let tag = self.get_tag();
        value | tag
    }

    pub fn get_tag(&self) -> isize {
        match self {
            BuiltInTypes::Int => 0b000,
            BuiltInTypes::Float => 0b001,
            BuiltInTypes::String => 0b010,
            BuiltInTypes::Bool => 0b011,
            BuiltInTypes::Function => 0b100,
            BuiltInTypes::Closure => 0b101,
            BuiltInTypes::HeapObject => 0b110,
            BuiltInTypes::Null => 0b111,
        }
    }

    pub fn untag(value: usize) -> usize {
        value >> 3
    }

    pub fn untag_isize(value: isize) -> isize {
        value >> 3
    }

    pub fn get_kind(pointer: usize) -> Self {
        if pointer == Self::null_value() as usize {
            return BuiltInTypes::Null;
        }
        match pointer & 0b111 {
            0b000 => BuiltInTypes::Int,
            0b001 => BuiltInTypes::Float,
            0b010 => BuiltInTypes::String,
            0b011 => BuiltInTypes::Bool,
            0b100 => BuiltInTypes::Function,
            0b101 => BuiltInTypes::Closure,
            0b110 => BuiltInTypes::HeapObject,
            0b111 => BuiltInTypes::Null,
            _ => panic!("Invalid tag {}", pointer & 0b111),
        }
    }

    pub fn is_embedded(&self) -> bool {
        match self {
            BuiltInTypes::Int => true,
            BuiltInTypes::Float => true,
            BuiltInTypes::String => false,
            BuiltInTypes::Bool => true,
            BuiltInTypes::Function => false,
            BuiltInTypes::HeapObject => false,
            BuiltInTypes::Closure => false,
            BuiltInTypes::Null => true,
        }
    }

    pub fn construct_int(value: isize) -> isize {
        if value > isize::MAX >> 3 {
            panic!("Integer overflow")
        }
        BuiltInTypes::Int.tag(value)
    }

    pub fn construct_boolean(value: bool) -> isize {
        let bool = BuiltInTypes::Bool;
        if value { bool.tag(1) } else { bool.tag(0) }
    }

    pub fn construct_float(x: f64) -> isize {
        let value = x.to_bits() as isize;
        BuiltInTypes::Float.tag(value)
    }

    pub fn tag_size() -> i32 {
        3
    }

    pub fn is_heap_pointer(value: usize) -> bool {
        // First check: must have a heap pointer tag
        let is_heap_tagged = match BuiltInTypes::get_kind(value) {
            BuiltInTypes::Int => false,
            BuiltInTypes::Float => true,
            BuiltInTypes::String => true,  // Strings ARE heap objects that need tracing
            BuiltInTypes::Bool => false,
            BuiltInTypes::Function => false,
            BuiltInTypes::Closure => true,
            BuiltInTypes::HeapObject => true,
            BuiltInTypes::Null => false,
        };
        if !is_heap_tagged {
            return false;
        }
        // Second check: untagged pointer must be 8-byte aligned
        // This filters out garbage values that happen to have the heap pointer tag
        let untagged = Self::untag(value);
        untagged % 8 == 0
    }

    /// Check if a tagged value is a closure (tag 0b101)
    pub fn is_closure(value: usize) -> bool {
        (value & 0b111) == 0b101
    }

    /// Check if a tagged value is a raw function (tag 0b100)
    pub fn is_function(value: usize) -> bool {
        (value & 0b111) == 0b100
    }
}

/// Header for heap-allocated objects
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Header {
    pub type_id: u8,
    pub type_data: u32,
    pub size: u16,  // Changed from u8 to u16 to support objects > 255 words
    pub opaque: bool,
    pub marked: bool,
}

impl Header {
    // | Byte 7  | Bytes 3-6     | Bytes 1-2  | Byte 0               |
    // |---------|---------------|------------|----------------------|
    // | Type    | Type Metadata | Size (u16) | Flag bits            |
    // |         | (4 bytes)     |            | Opaque object (bit 1) |
    // |         |               |            | Marked (bit 0)       |

    /// Position of the marked bit in the header.
    /// IMPORTANT: This MUST be in the 3 least significant bits (0, 1, or 2) for
    /// GC forwarding to work with 8-byte aligned pointers.
    const MARKED_BIT_POSITION: u32 = 0;

    /// Position of the opaque bit in the header.
    const OPAQUE_BIT_POSITION: u32 = 1;

    pub fn to_usize(self) -> usize {
        let mut data: usize = 0;
        data |= (self.type_id as usize) << 56;
        data |= (self.type_data as usize) << 24;
        // Size now uses bits 8-23 (u16)
        data |= (self.size as usize) << 8;
        if self.opaque {
            data |= 1 << Self::OPAQUE_BIT_POSITION;
        }
        if self.marked {
            data |= 1 << Self::MARKED_BIT_POSITION;
        }
        data
    }

    pub fn from_usize(data: usize) -> Self {
        let _type = (data >> 56) as u8;
        let type_data = (data >> 24) as u32;
        // Size now uses bits 8-23 (u16)
        let size = ((data >> 8) & 0xFFFF) as u16;
        let opaque = (data & (1 << Self::OPAQUE_BIT_POSITION)) != 0;
        let marked = (data & (1 << Self::MARKED_BIT_POSITION)) != 0;
        Header {
            type_id: _type,
            type_data,
            size,
            opaque,
            marked,
        }
    }

    pub fn type_id_offset() -> usize {
        7
    }

    pub fn type_data_offset() -> usize {
        3
    }

    pub fn size_offset() -> usize {
        1  // Size now starts at byte 1 (bits 8-23)
    }

    /// Get the bit mask for the marked bit
    pub const fn marked_bit_mask() -> usize {
        1 << Self::MARKED_BIT_POSITION
    }

    /// Set the marked bit in a raw header value, preserving other bits
    pub const fn set_marked_bit(header_value: usize) -> usize {
        header_value | Self::marked_bit_mask()
    }

    /// Clear the marked bit in a raw header value, preserving other bits
    pub const fn clear_marked_bit(header_value: usize) -> usize {
        header_value & !Self::marked_bit_mask()
    }

    /// Check if the marked bit is set in a raw header value
    pub const fn is_marked_bit_set(header_value: usize) -> bool {
        (header_value & Self::marked_bit_mask()) != 0
    }
}

/// Wrapper for heap objects
pub struct HeapObject {
    pointer: usize,
    tagged: bool,
}

impl HeapObject {
    pub fn from_tagged(pointer: usize) -> Self {
        assert!(BuiltInTypes::is_heap_pointer(pointer));
        assert!(BuiltInTypes::untag(pointer) % 8 == 0);
        HeapObject {
            pointer,
            tagged: true,
        }
    }

    pub fn from_untagged(pointer: *const u8) -> Self {
        assert!(pointer as usize % 8 == 0);
        HeapObject {
            pointer: pointer as usize,
            tagged: false,
        }
    }

    pub fn try_from_tagged(pointer: usize) -> Option<Self> {
        if BuiltInTypes::is_heap_pointer(pointer) {
            Some(HeapObject {
                pointer,
                tagged: true,
            })
        } else {
            None
        }
    }

    pub fn untagged(&self) -> usize {
        if self.tagged {
            BuiltInTypes::untag(self.pointer)
        } else {
            self.pointer
        }
    }

    pub fn get_object_type(&self) -> Option<BuiltInTypes> {
        if !self.tagged {
            return None;
        }
        Some(BuiltInTypes::get_kind(self.pointer))
    }

    pub fn mark(&self) {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let data: usize = unsafe { *pointer.cast::<usize>() };
        let marked_data = Header::set_marked_bit(data);
        unsafe { *pointer.cast::<usize>() = marked_data };
    }

    pub fn marked(&self) -> bool {
        self.get_header().marked
    }

    pub fn fields_size(&self) -> usize {
        let untagged = self.untagged();
        let pointer = untagged as *mut isize;
        let data: usize = unsafe { *pointer.cast::<usize>() };
        let header = Header::from_usize(data);
        header.size as usize * 8
    }

    pub fn get_fields(&self) -> &[usize] {
        if self.is_opaque_object() {
            return &[];
        }
        let size = self.fields_size();
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let pointer = unsafe { pointer.add(Self::header_size() / 8) };
        unsafe { std::slice::from_raw_parts(pointer, size / 8) }
    }

    pub fn get_string_bytes(&self) -> &[u8] {
        let header = self.get_header();
        let bytes = header.type_data as usize;
        let untagged = self.untagged();
        let pointer = untagged as *mut u8;
        let pointer = unsafe { pointer.add(Self::header_size()) };
        unsafe { std::slice::from_raw_parts(pointer, bytes) }
    }

    pub fn get_str_unchecked(&self) -> &str {
        let bytes = self.get_string_bytes();
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }

    pub fn get_full_object_data(&self) -> &[u8] {
        let size = self.full_size();
        let untagged = self.untagged();
        let pointer = untagged as *mut u8;
        assert!(pointer.is_aligned());
        unsafe { std::slice::from_raw_parts(pointer, size) }
    }

    pub fn get_heap_references(&self) -> impl Iterator<Item = HeapObject> + '_ {
        let fields = self.get_fields();
        fields
            .iter()
            .filter(|_x| !self.is_opaque_object())
            .filter(|x| BuiltInTypes::is_heap_pointer(**x))
            .map(|&pointer| HeapObject::from_tagged(pointer))
    }

    pub fn get_fields_mut(&mut self) -> &mut [usize] {
        if self.is_opaque_object() {
            return &mut [];
        }
        let size = self.fields_size();
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let pointer = unsafe { pointer.add(1) };
        unsafe { std::slice::from_raw_parts_mut(pointer, size / 8) }
    }

    pub fn unmark(&self) {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let data: usize = unsafe { *pointer.cast::<usize>() };
        let unmarked_data = Header::clear_marked_bit(data);
        unsafe { *pointer.cast::<usize>() = unmarked_data };
    }

    pub fn full_size(&self) -> usize {
        self.fields_size() + Self::header_size()
    }

    pub const fn header_size() -> usize {
        8
    }

    pub fn write_header_direct(&mut self, header: Header) {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        unsafe { *pointer.cast::<usize>() = header.to_usize() };
    }

    pub fn write_header(&mut self, field_size: Word) {
        assert!(field_size.to_bytes() % 8 == 0);
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;

        if field_size.to_bytes() % 8 != 0 {
            panic!("Size is not aligned");
        }

        let header = Header {
            type_id: 0,
            type_data: 0,
            size: field_size.to_words() as u16,
            opaque: false,
            marked: false,
        };

        unsafe { *pointer.cast::<usize>() = header.to_usize() };
    }

    pub fn write_full_object(&mut self, data: &[u8]) {
        unsafe {
            let untagged = self.untagged();
            let pointer = untagged as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), pointer, data.len());
        }
    }

    pub fn copy_full_object(&self, to_object: &mut HeapObject) {
        let data = self.get_full_object_data();
        to_object.write_full_object(data);
    }

    pub fn get_pointer(&self) -> *const u8 {
        let untagged = self.untagged();
        untagged as *const u8
    }

    pub fn write_field(&self, index: usize, value: usize) {
        debug_assert!(index < self.fields_size());
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let pointer = unsafe { pointer.add(index + Self::header_size() / 8) };
        unsafe { *pointer = value };
    }

    pub fn get_field(&self, arg: usize) -> usize {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let pointer = unsafe { pointer.add(arg + Self::header_size() / 8) };
        unsafe { *pointer }
    }

    #[allow(unused)]
    pub fn get_type_id(&self) -> usize {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let header = unsafe { *pointer };
        let header = Header::from_usize(header);
        header.type_id as usize
    }

    pub fn write_type_id(&mut self, type_id: usize) {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let header = unsafe { *pointer };
        let header = Header::from_usize(header);
        let new_header = Header {
            type_id: type_id as u8,
            ..header
        };
        unsafe { *pointer = new_header.to_usize() };
    }

    pub fn get_struct_id(&self) -> usize {
        self.get_type_data()
    }

    pub fn get_type_data(&self) -> usize {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let header = unsafe { *pointer };
        let header = Header::from_usize(header);
        header.type_data as usize
    }

    pub fn is_opaque_object(&self) -> bool {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let data: usize = unsafe { *pointer.cast::<usize>() };
        let header = Header::from_usize(data);
        header.opaque
    }

    pub fn get_header(&self) -> Header {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        assert!(pointer.is_aligned());
        let data: usize = unsafe { *pointer.cast::<usize>() };
        Header::from_usize(data)
    }

    pub fn tagged_pointer(&self) -> usize {
        if self.tagged {
            self.pointer
        } else {
            panic!("Not tagged");
        }
    }

    pub fn write_fields(&mut self, fields: &[u8]) {
        let untagged = self.untagged();
        let pointer = untagged as *mut u8;
        let pointer = unsafe { pointer.add(Self::header_size()) };
        unsafe { std::ptr::copy_nonoverlapping(fields.as_ptr(), pointer, fields.len()) };
    }

    pub fn is_zero_size(&self) -> bool {
        let untagged = self.untagged();
        let pointer = untagged as *mut usize;
        let data: usize = unsafe { *pointer.cast::<usize>() };
        let header = Header::from_usize(data);
        header.size == 0
    }
}

/// Word - represents size in words (8 bytes each)
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Word(usize);

impl Word {
    pub fn to_bytes(self) -> usize {
        self.0 * 8
    }

    pub fn from_word(size: usize) -> Word {
        Word(size)
    }

    pub fn from_bytes(len: usize) -> Word {
        Word(len / 8)
    }

    pub fn to_words(self) -> usize {
        self.0
    }
}
