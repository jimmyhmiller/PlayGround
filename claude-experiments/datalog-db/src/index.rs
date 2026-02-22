use std::io::Cursor;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};

use crate::datom::{Datom, EntityId, TxId, Value};

pub const EAVT_PREFIX: u8 = 0x01;
pub const AEVT_PREFIX: u8 = 0x02;
pub const AVET_PREFIX: u8 = 0x03;
pub const VAET_PREFIX: u8 = 0x04;
pub const META_PREFIX: u8 = 0x00;

// Current-state indexes (no tx/added, latest value only)
pub const CURRENT_AEVT_PREFIX: u8 = 0x11;
pub const CURRENT_AVET_PREFIX: u8 = 0x12;

// --- Encoding helpers ---

fn encode_value(buf: &mut Vec<u8>, value: &Value) {
    buf.push(value.type_tag()); // panics for Value::Enum — it should never reach storage
    match value {
        Value::String(s) => {
            let bytes = s.as_bytes();
            buf.write_u32::<BigEndian>(bytes.len() as u32).unwrap();
            buf.extend_from_slice(bytes);
        }
        Value::I64(n) => {
            // Flip sign bit for lexicographic ordering of signed integers
            buf.write_u64::<BigEndian>((*n as u64) ^ (1 << 63))
                .unwrap();
        }
        Value::F64(f) => {
            let bits = f.to_bits();
            let encoded = if bits & (1 << 63) != 0 {
                !bits // negative: flip all bits
            } else {
                bits ^ (1 << 63) // positive: flip sign bit
            };
            buf.write_u64::<BigEndian>(encoded).unwrap();
        }
        Value::Bool(b) => {
            buf.push(if *b { 1 } else { 0 });
        }
        Value::Ref(id) => {
            buf.write_u64::<BigEndian>(*id).unwrap();
        }
        Value::Bytes(bytes) => {
            buf.write_u32::<BigEndian>(bytes.len() as u32).unwrap();
            buf.extend_from_slice(bytes);
        }
        Value::Enum { .. } => panic!("Enum values cannot be encoded in index keys"),
        Value::Null => panic!("Null values cannot be encoded in index keys"),
    }
}

fn encode_attr(buf: &mut Vec<u8>, attr: &str) {
    let bytes = attr.as_bytes();
    buf.write_u16::<BigEndian>(bytes.len() as u16).unwrap();
    buf.extend_from_slice(bytes);
}

fn encode_entity(buf: &mut Vec<u8>, entity: EntityId) {
    buf.write_u64::<BigEndian>(entity).unwrap();
}

fn encode_tx(buf: &mut Vec<u8>, tx: TxId) {
    buf.write_u64::<BigEndian>(tx).unwrap();
}

fn encode_added(buf: &mut Vec<u8>, added: bool) {
    buf.push(if added { 1 } else { 0 });
}

// --- Index key encoding ---

pub fn encode_eavt(datom: &Datom) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64);
    buf.push(EAVT_PREFIX);
    encode_entity(&mut buf, datom.entity);
    encode_attr(&mut buf, &datom.attribute);
    encode_value(&mut buf, &datom.value);
    encode_tx(&mut buf, datom.tx);
    encode_added(&mut buf, datom.added);
    buf
}

pub fn encode_aevt(datom: &Datom) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64);
    buf.push(AEVT_PREFIX);
    encode_attr(&mut buf, &datom.attribute);
    encode_entity(&mut buf, datom.entity);
    encode_value(&mut buf, &datom.value);
    encode_tx(&mut buf, datom.tx);
    encode_added(&mut buf, datom.added);
    buf
}

pub fn encode_avet(datom: &Datom) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64);
    buf.push(AVET_PREFIX);
    encode_attr(&mut buf, &datom.attribute);
    encode_value(&mut buf, &datom.value);
    encode_entity(&mut buf, datom.entity);
    encode_tx(&mut buf, datom.tx);
    encode_added(&mut buf, datom.added);
    buf
}

pub fn encode_vaet(datom: &Datom) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64);
    buf.push(VAET_PREFIX);
    encode_value(&mut buf, &datom.value);
    encode_attr(&mut buf, &datom.attribute);
    encode_entity(&mut buf, datom.entity);
    encode_tx(&mut buf, datom.tx);
    encode_added(&mut buf, datom.added);
    buf
}

/// Encode a datom into all applicable index keys. Returns (key, value) pairs.
pub fn encode_datom(datom: &Datom) -> Vec<(Vec<u8>, Vec<u8>)> {
    let empty = vec![];
    let mut pairs = vec![
        (encode_eavt(datom), empty.clone()),
        (encode_aevt(datom), empty.clone()),
        (encode_avet(datom), empty.clone()),
    ];
    // VAET only for Ref values
    if matches!(datom.value, Value::Ref(_)) {
        pairs.push((encode_vaet(datom), empty));
    }
    pairs
}

// --- Scan prefix builders ---

pub fn eavt_entity_prefix(entity: EntityId) -> Vec<u8> {
    let mut buf = Vec::with_capacity(9);
    buf.push(EAVT_PREFIX);
    encode_entity(&mut buf, entity);
    buf
}

pub fn eavt_entity_attr_prefix(entity: EntityId, attr: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(11 + attr.len());
    buf.push(EAVT_PREFIX);
    encode_entity(&mut buf, entity);
    encode_attr(&mut buf, attr);
    buf
}

pub fn aevt_attr_prefix(attr: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(3 + attr.len());
    buf.push(AEVT_PREFIX);
    encode_attr(&mut buf, attr);
    buf
}

pub fn avet_attr_prefix(attr: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(3 + attr.len());
    buf.push(AVET_PREFIX);
    encode_attr(&mut buf, attr);
    buf
}

pub fn avet_attr_value_prefix(attr: &str, value: &Value) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16 + attr.len());
    buf.push(AVET_PREFIX);
    encode_attr(&mut buf, attr);
    encode_value(&mut buf, value);
    buf
}

/// Build AVET prefix for scanning all values of a given type under an attribute.
/// Key layout: [AVET_PREFIX][attr_len(u16)][attr_bytes][type_tag]
pub fn avet_attr_type_prefix(attr: &str, type_tag: u8) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + attr.len());
    buf.push(AVET_PREFIX);
    encode_attr(&mut buf, attr);
    buf.push(type_tag);
    buf
}

pub fn meta_key(name: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1 + name.len());
    buf.push(META_PREFIX);
    buf.extend_from_slice(name.as_bytes());
    buf
}

// --- Decoding helpers ---

fn decode_attr(cursor: &mut Cursor<&[u8]>) -> Option<String> {
    let len = cursor.read_u16::<BigEndian>().ok()? as usize;
    let pos = cursor.position() as usize;
    let data = cursor.get_ref();
    if pos + len > data.len() {
        return None;
    }
    let s = String::from_utf8(data[pos..pos + len].to_vec()).ok()?;
    cursor.set_position((pos + len) as u64);
    Some(s)
}

fn decode_value(cursor: &mut Cursor<&[u8]>) -> Option<Value> {
    let tag = cursor.read_u8().ok()?;
    match tag {
        0x01 => {
            // String
            let len = cursor.read_u32::<BigEndian>().ok()? as usize;
            let pos = cursor.position() as usize;
            let data = cursor.get_ref();
            if pos + len > data.len() {
                return None;
            }
            let s = String::from_utf8(data[pos..pos + len].to_vec()).ok()?;
            cursor.set_position((pos + len) as u64);
            Some(Value::String(s))
        }
        0x02 => {
            // I64
            let encoded = cursor.read_u64::<BigEndian>().ok()?;
            let n = (encoded ^ (1 << 63)) as i64;
            Some(Value::I64(n))
        }
        0x03 => {
            // F64
            let encoded = cursor.read_u64::<BigEndian>().ok()?;
            let bits = if encoded & (1 << 63) != 0 {
                encoded ^ (1 << 63) // positive: flip sign bit back
            } else {
                !encoded // negative: flip all bits back
            };
            Some(Value::F64(f64::from_bits(bits)))
        }
        0x04 => {
            // Bool
            let b = cursor.read_u8().ok()?;
            Some(Value::Bool(b != 0))
        }
        0x05 => {
            // Ref
            let id = cursor.read_u64::<BigEndian>().ok()?;
            Some(Value::Ref(id))
        }
        0x06 => {
            // Bytes
            let len = cursor.read_u32::<BigEndian>().ok()? as usize;
            let pos = cursor.position() as usize;
            let data = cursor.get_ref();
            if pos + len > data.len() {
                return None;
            }
            let bytes = data[pos..pos + len].to_vec();
            cursor.set_position((pos + len) as u64);
            Some(Value::Bytes(bytes))
        }
        _ => None,
    }
}

pub fn decode_datom_from_eavt(key: &[u8]) -> Option<Datom> {
    if key.is_empty() || key[0] != EAVT_PREFIX {
        return None;
    }
    let mut cursor = Cursor::new(&key[1..]);
    let entity = cursor.read_u64::<BigEndian>().ok()?;
    let attribute = decode_attr(&mut cursor)?;
    let value = decode_value(&mut cursor)?;
    let tx = cursor.read_u64::<BigEndian>().ok()?;
    let added = cursor.read_u8().ok()? != 0;
    Some(Datom {
        entity,
        attribute,
        value,
        tx,
        added,
    })
}

pub fn decode_datom_from_aevt(key: &[u8]) -> Option<Datom> {
    if key.is_empty() || key[0] != AEVT_PREFIX {
        return None;
    }
    let mut cursor = Cursor::new(&key[1..]);
    let attribute = decode_attr(&mut cursor)?;
    let entity = cursor.read_u64::<BigEndian>().ok()?;
    let value = decode_value(&mut cursor)?;
    let tx = cursor.read_u64::<BigEndian>().ok()?;
    let added = cursor.read_u8().ok()? != 0;
    Some(Datom {
        entity,
        attribute,
        value,
        tx,
        added,
    })
}

pub fn decode_datom_from_avet(key: &[u8]) -> Option<Datom> {
    if key.is_empty() || key[0] != AVET_PREFIX {
        return None;
    }
    let mut cursor = Cursor::new(&key[1..]);
    let attribute = decode_attr(&mut cursor)?;
    let value = decode_value(&mut cursor)?;
    let entity = cursor.read_u64::<BigEndian>().ok()?;
    let tx = cursor.read_u64::<BigEndian>().ok()?;
    let added = cursor.read_u8().ok()? != 0;
    Some(Datom {
        entity,
        attribute,
        value,
        tx,
        added,
    })
}

// --- Current-state index encoding ---
// CURRENT_AEVT: Key = [0x11][attr_len(u16)][attr_bytes][entity_id(u64)]
//               Value = encoded Value bytes
// CURRENT_AVET: Key = [0x12][attr_len(u16)][attr_bytes][type_tag][value_data][entity_id(u64)]
//               Value = empty

/// Encode a current-state AEVT key + value.
/// Returns (key, value_bytes).
pub fn encode_current_aevt(attr: &str, entity: EntityId, value: &Value) -> (Vec<u8>, Vec<u8>) {
    let mut key = Vec::with_capacity(3 + attr.len() + 8);
    key.push(CURRENT_AEVT_PREFIX);
    encode_attr(&mut key, attr);
    encode_entity(&mut key, entity);

    let mut val_buf = Vec::with_capacity(16);
    encode_value(&mut val_buf, value);

    (key, val_buf)
}

/// Encode a current-state AVET key. Value is empty.
pub fn encode_current_avet(attr: &str, value: &Value, entity: EntityId) -> Vec<u8> {
    let mut key = Vec::with_capacity(16 + attr.len());
    key.push(CURRENT_AVET_PREFIX);
    encode_attr(&mut key, attr);
    encode_value(&mut key, value);
    encode_entity(&mut key, entity);
    key
}

/// Build prefix for scanning CURRENT_AEVT by attribute.
pub fn current_aevt_attr_prefix(attr: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(3 + attr.len());
    buf.push(CURRENT_AEVT_PREFIX);
    encode_attr(&mut buf, attr);
    buf
}

/// Build prefix for scanning CURRENT_AVET by attribute.
pub fn current_avet_attr_prefix(attr: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(3 + attr.len());
    buf.push(CURRENT_AVET_PREFIX);
    encode_attr(&mut buf, attr);
    buf
}

/// Build prefix for scanning CURRENT_AVET by attribute + value.
pub fn current_avet_attr_value_prefix(attr: &str, value: &Value) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16 + attr.len());
    buf.push(CURRENT_AVET_PREFIX);
    encode_attr(&mut buf, attr);
    encode_value(&mut buf, value);
    buf
}

/// Build prefix for scanning CURRENT_AVET by attribute + type tag.
pub fn current_avet_attr_type_prefix(attr: &str, type_tag: u8) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + attr.len());
    buf.push(CURRENT_AVET_PREFIX);
    encode_attr(&mut buf, attr);
    buf.push(type_tag);
    buf
}

/// Extract entity ID from a CURRENT_AEVT key, given the known attr byte length.
/// Key layout: [1 prefix][2 attr_len][attr_bytes][8 entity_id]
pub fn current_aevt_entity_at(key: &[u8], attr_byte_len: usize) -> EntityId {
    let offset = 1 + 2 + attr_byte_len;
    u64::from_be_bytes(key[offset..offset + 8].try_into().unwrap())
}

/// Extract entity ID from a CURRENT_AVET key (entity is always the last 8 bytes).
pub fn current_avet_entity_at(key: &[u8]) -> EntityId {
    let offset = key.len() - 8;
    u64::from_be_bytes(key[offset..offset + 8].try_into().unwrap())
}

/// Decode a Value from raw bytes (as stored in CURRENT_AEVT RocksDB value).
pub fn decode_current_value(data: &[u8]) -> Option<Value> {
    let mut cursor = Cursor::new(data);
    decode_value(&mut cursor)
}

/// Compute the exclusive upper bound for a prefix scan.
/// Increments the last byte; if all 0xFF, extends.
pub fn prefix_end(prefix: &[u8]) -> Vec<u8> {
    let mut end = prefix.to_vec();
    while let Some(last) = end.last_mut() {
        if *last < 0xFF {
            *last += 1;
            return end;
        }
        end.pop();
    }
    vec![0xFF; prefix.len() + 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eavt_roundtrip() {
        let datom = Datom {
            entity: 42,
            attribute: "User/name".to_string(),
            value: Value::String("Alice".to_string()),
            tx: 1,
            added: true,
        };
        let key = encode_eavt(&datom);
        let decoded = decode_datom_from_eavt(&key).unwrap();
        assert_eq!(decoded.entity, 42);
        assert_eq!(decoded.attribute, "User/name");
        assert_eq!(decoded.value, Value::String("Alice".to_string()));
        assert_eq!(decoded.tx, 1);
        assert!(decoded.added);
    }

    #[test]
    fn test_aevt_roundtrip() {
        let datom = Datom {
            entity: 100,
            attribute: "Post/title".to_string(),
            value: Value::String("Hello".to_string()),
            tx: 5,
            added: true,
        };
        let key = encode_aevt(&datom);
        let decoded = decode_datom_from_aevt(&key).unwrap();
        assert_eq!(decoded.entity, 100);
        assert_eq!(decoded.attribute, "Post/title");
        assert_eq!(decoded.value, Value::String("Hello".to_string()));
        assert_eq!(decoded.tx, 5);
    }

    #[test]
    fn test_avet_roundtrip() {
        let datom = Datom {
            entity: 7,
            attribute: "User/age".to_string(),
            value: Value::I64(30),
            tx: 2,
            added: true,
        };
        let key = encode_avet(&datom);
        let decoded = decode_datom_from_avet(&key).unwrap();
        assert_eq!(decoded.entity, 7);
        assert_eq!(decoded.attribute, "User/age");
        assert_eq!(decoded.value, Value::I64(30));
    }

    #[test]
    fn test_i64_sort_order() {
        let neg = Datom {
            entity: 1,
            attribute: "a".to_string(),
            value: Value::I64(-10),
            tx: 1,
            added: true,
        };
        let zero = Datom {
            entity: 1,
            attribute: "a".to_string(),
            value: Value::I64(0),
            tx: 1,
            added: true,
        };
        let pos = Datom {
            entity: 1,
            attribute: "a".to_string(),
            value: Value::I64(10),
            tx: 1,
            added: true,
        };
        let k_neg = encode_avet(&neg);
        let k_zero = encode_avet(&zero);
        let k_pos = encode_avet(&pos);
        assert!(k_neg < k_zero);
        assert!(k_zero < k_pos);
    }

    #[test]
    fn test_f64_roundtrip() {
        for val in [0.0, 1.0, -1.0, f64::MAX, f64::MIN, 3.14, -2.718] {
            let datom = Datom {
                entity: 1,
                attribute: "x".to_string(),
                value: Value::F64(val),
                tx: 1,
                added: true,
            };
            let key = encode_eavt(&datom);
            let decoded = decode_datom_from_eavt(&key).unwrap();
            assert_eq!(decoded.value, Value::F64(val));
        }
    }

    #[test]
    fn test_prefix_end() {
        assert_eq!(prefix_end(&[0x01, 0x00]), vec![0x01, 0x01]);
        assert_eq!(prefix_end(&[0x01, 0xFF]), vec![0x02]);
        assert_eq!(prefix_end(&[0x01]), vec![0x02]);
    }

    #[test]
    fn test_bool_roundtrip() {
        for b in [true, false] {
            let datom = Datom {
                entity: 1,
                attribute: "x".to_string(),
                value: Value::Bool(b),
                tx: 1,
                added: true,
            };
            let key = encode_eavt(&datom);
            let decoded = decode_datom_from_eavt(&key).unwrap();
            assert_eq!(decoded.value, Value::Bool(b));
        }
    }

    #[test]
    fn test_ref_roundtrip() {
        let datom = Datom {
            entity: 1,
            attribute: "Post/author".to_string(),
            value: Value::Ref(42),
            tx: 1,
            added: true,
        };
        let key = encode_eavt(&datom);
        let decoded = decode_datom_from_eavt(&key).unwrap();
        assert_eq!(decoded.value, Value::Ref(42));
    }

    #[test]
    fn test_encode_datom_vaet_only_for_ref() {
        let string_datom = Datom {
            entity: 1,
            attribute: "User/name".to_string(),
            value: Value::String("Alice".to_string()),
            tx: 1,
            added: true,
        };
        assert_eq!(encode_datom(&string_datom).len(), 3); // EAVT, AEVT, AVET only

        let ref_datom = Datom {
            entity: 1,
            attribute: "Post/author".to_string(),
            value: Value::Ref(42),
            tx: 1,
            added: true,
        };
        assert_eq!(encode_datom(&ref_datom).len(), 4); // EAVT, AEVT, AVET, VAET
    }
}
