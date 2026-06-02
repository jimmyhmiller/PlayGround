use std::io::Cursor;
use std::sync::Arc;

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};

use crate::datom::{EntityId, TxId, Value};
use crate::intern::AttrId;

pub const EAVT_PREFIX: u8 = 0x01;
pub const AEVT_PREFIX: u8 = 0x02;
pub const AVET_PREFIX: u8 = 0x03;
pub const VAET_PREFIX: u8 = 0x04;
pub const META_PREFIX: u8 = 0x00;

// Current-state indexes (no tx/added, latest value only)
pub const CURRENT_AEVT_PREFIX: u8 = 0x11;
pub const CURRENT_AVET_PREFIX: u8 = 0x12;

/// Width of an interned attribute identifier in an index key.
const ATTR_ID_BYTES: usize = 4;

/// A datom decoded from a storage key. Carries the interned
/// `attr_id` rather than the attribute name; callers that need the
/// name resolve via the `AttrInterner`.
#[derive(Debug, Clone)]
pub struct DecodedDatom {
    pub entity: EntityId,
    pub attr_id: AttrId,
    pub value: Value,
    pub tx: TxId,
    pub added: bool,
}

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
        Value::Enum(_) => panic!("Enum values cannot be encoded in index keys"),
        Value::Null => panic!("Null values cannot be encoded in index keys"),
    }
}

fn encode_attr_id(buf: &mut Vec<u8>, attr_id: AttrId) {
    buf.write_u32::<BigEndian>(attr_id).unwrap();
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
// All keys now hold the attribute as a 4-byte interned id rather than
// the variable-length attribute name. Layouts:
//
//   EAVT: [0x01][entity(8)][attr_id(4)][value(var)][tx(8)][added(1)]
//   AEVT: [0x02][attr_id(4)][entity(8)][value(var)][tx(8)][added(1)]
//   AVET: [0x03][attr_id(4)][value(var)][entity(8)][tx(8)][added(1)]
//   VAET: [0x04][value(var)][attr_id(4)][entity(8)][tx(8)][added(1)]
//   CURRENT_AEVT: [0x11][attr_id(4)][entity(8)] → value bytes
//   CURRENT_AVET: [0x12][attr_id(4)][value(var)][entity(8)]

pub fn encode_eavt(entity: EntityId, attr_id: AttrId, value: &Value, tx: TxId, added: bool) -> Vec<u8> {
    let mut buf = Vec::with_capacity(48);
    buf.push(EAVT_PREFIX);
    encode_entity(&mut buf, entity);
    encode_attr_id(&mut buf, attr_id);
    encode_value(&mut buf, value);
    encode_tx(&mut buf, tx);
    encode_added(&mut buf, added);
    buf
}

pub fn encode_aevt(entity: EntityId, attr_id: AttrId, value: &Value, tx: TxId, added: bool) -> Vec<u8> {
    let mut buf = Vec::with_capacity(48);
    buf.push(AEVT_PREFIX);
    encode_attr_id(&mut buf, attr_id);
    encode_entity(&mut buf, entity);
    encode_value(&mut buf, value);
    encode_tx(&mut buf, tx);
    encode_added(&mut buf, added);
    buf
}

pub fn encode_avet(entity: EntityId, attr_id: AttrId, value: &Value, tx: TxId, added: bool) -> Vec<u8> {
    let mut buf = Vec::with_capacity(48);
    buf.push(AVET_PREFIX);
    encode_attr_id(&mut buf, attr_id);
    encode_value(&mut buf, value);
    encode_entity(&mut buf, entity);
    encode_tx(&mut buf, tx);
    encode_added(&mut buf, added);
    buf
}

pub fn encode_vaet(entity: EntityId, attr_id: AttrId, value: &Value, tx: TxId, added: bool) -> Vec<u8> {
    let mut buf = Vec::with_capacity(48);
    buf.push(VAET_PREFIX);
    encode_value(&mut buf, value);
    encode_attr_id(&mut buf, attr_id);
    encode_entity(&mut buf, entity);
    encode_tx(&mut buf, tx);
    encode_added(&mut buf, added);
    buf
}

/// Encode a datom into all applicable history-index keys, given its
/// interned attribute id. The caller is responsible for the
/// `attr_id` lookup (which must have been allocated before encoding).
pub fn encode_datom(
    entity: EntityId,
    attr_id: AttrId,
    value: &Value,
    tx: TxId,
    added: bool,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let empty = vec![];
    let mut pairs = vec![
        (encode_eavt(entity, attr_id, value, tx, added), empty.clone()),
        (encode_aevt(entity, attr_id, value, tx, added), empty.clone()),
        (encode_avet(entity, attr_id, value, tx, added), empty.clone()),
    ];
    // VAET only for Ref values
    if matches!(value, Value::Ref(_)) {
        pairs.push((encode_vaet(entity, attr_id, value, tx, added), empty));
    }
    pairs
}

// --- Scan prefix builders (history indexes) ---

pub fn eavt_entity_prefix(entity: EntityId) -> Vec<u8> {
    let mut buf = Vec::with_capacity(9);
    buf.push(EAVT_PREFIX);
    encode_entity(&mut buf, entity);
    buf
}

pub fn eavt_entity_attr_prefix(entity: EntityId, attr_id: AttrId) -> Vec<u8> {
    let mut buf = Vec::with_capacity(13);
    buf.push(EAVT_PREFIX);
    encode_entity(&mut buf, entity);
    encode_attr_id(&mut buf, attr_id);
    buf
}

pub fn aevt_attr_prefix(attr_id: AttrId) -> Vec<u8> {
    let mut buf = Vec::with_capacity(5);
    buf.push(AEVT_PREFIX);
    encode_attr_id(&mut buf, attr_id);
    buf
}

pub fn avet_attr_prefix(attr_id: AttrId) -> Vec<u8> {
    let mut buf = Vec::with_capacity(5);
    buf.push(AVET_PREFIX);
    encode_attr_id(&mut buf, attr_id);
    buf
}

/// Build a VAET prefix to scan every datom whose value equals `value`
/// (only Ref values land in VAET). Used to find inbound references to an
/// entity, e.g. when reporting refs left dangling by a hard purge.
pub fn vaet_value_prefix(value: &Value) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16);
    buf.push(VAET_PREFIX);
    encode_value(&mut buf, value);
    buf
}

pub fn avet_attr_value_prefix(attr_id: AttrId, value: &Value) -> Vec<u8> {
    let mut buf = Vec::with_capacity(20);
    buf.push(AVET_PREFIX);
    encode_attr_id(&mut buf, attr_id);
    encode_value(&mut buf, value);
    buf
}

/// Build AVET prefix for scanning all values of a given type under an attribute.
/// Key layout: [AVET_PREFIX][attr_id(4)][type_tag]
pub fn avet_attr_type_prefix(attr_id: AttrId, type_tag: u8) -> Vec<u8> {
    let mut buf = Vec::with_capacity(6);
    buf.push(AVET_PREFIX);
    encode_attr_id(&mut buf, attr_id);
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

fn decode_attr_id(cursor: &mut Cursor<&[u8]>) -> Option<AttrId> {
    cursor.read_u32::<BigEndian>().ok()
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
            let s = std::str::from_utf8(&data[pos..pos + len]).ok()?;
            cursor.set_position((pos + len) as u64);
            Some(Value::String(Arc::from(s)))
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

pub fn decode_datom_from_eavt(key: &[u8]) -> Option<DecodedDatom> {
    if key.is_empty() || key[0] != EAVT_PREFIX {
        return None;
    }
    let mut cursor = Cursor::new(&key[1..]);
    let entity = cursor.read_u64::<BigEndian>().ok()?;
    let attr_id = decode_attr_id(&mut cursor)?;
    let value = decode_value(&mut cursor)?;
    let tx = cursor.read_u64::<BigEndian>().ok()?;
    let added = cursor.read_u8().ok()? != 0;
    Some(DecodedDatom {
        entity,
        attr_id,
        value,
        tx,
        added,
    })
}

pub fn decode_datom_from_aevt(key: &[u8]) -> Option<DecodedDatom> {
    if key.is_empty() || key[0] != AEVT_PREFIX {
        return None;
    }
    let mut cursor = Cursor::new(&key[1..]);
    let attr_id = decode_attr_id(&mut cursor)?;
    let entity = cursor.read_u64::<BigEndian>().ok()?;
    let value = decode_value(&mut cursor)?;
    let tx = cursor.read_u64::<BigEndian>().ok()?;
    let added = cursor.read_u8().ok()? != 0;
    Some(DecodedDatom {
        entity,
        attr_id,
        value,
        tx,
        added,
    })
}

pub fn decode_datom_from_avet(key: &[u8]) -> Option<DecodedDatom> {
    if key.is_empty() || key[0] != AVET_PREFIX {
        return None;
    }
    let mut cursor = Cursor::new(&key[1..]);
    let attr_id = decode_attr_id(&mut cursor)?;
    let value = decode_value(&mut cursor)?;
    let entity = cursor.read_u64::<BigEndian>().ok()?;
    let tx = cursor.read_u64::<BigEndian>().ok()?;
    let added = cursor.read_u8().ok()? != 0;
    Some(DecodedDatom {
        entity,
        attr_id,
        value,
        tx,
        added,
    })
}

pub fn decode_datom_from_vaet(key: &[u8]) -> Option<DecodedDatom> {
    if key.is_empty() || key[0] != VAET_PREFIX {
        return None;
    }
    let mut cursor = Cursor::new(&key[1..]);
    let value = decode_value(&mut cursor)?;
    let attr_id = decode_attr_id(&mut cursor)?;
    let entity = cursor.read_u64::<BigEndian>().ok()?;
    let tx = cursor.read_u64::<BigEndian>().ok()?;
    let added = cursor.read_u8().ok()? != 0;
    Some(DecodedDatom {
        entity,
        attr_id,
        value,
        tx,
        added,
    })
}

// --- Current-state index encoding ---
// CURRENT_AEVT: Key = [0x11][attr_id(4)][entity_id(8)]
//               Value = encoded Value bytes
// CURRENT_AVET: Key = [0x12][attr_id(4)][type_tag][value_data][entity_id(8)]
//               Value = empty

/// Encode a current-state AEVT key + value.
pub fn encode_current_aevt(
    attr_id: AttrId,
    entity: EntityId,
    value: &Value,
) -> (Vec<u8>, Vec<u8>) {
    let mut key = Vec::with_capacity(1 + ATTR_ID_BYTES + 8);
    key.push(CURRENT_AEVT_PREFIX);
    encode_attr_id(&mut key, attr_id);
    encode_entity(&mut key, entity);

    let mut val_buf = Vec::with_capacity(16);
    encode_value(&mut val_buf, value);

    (key, val_buf)
}

/// Encode a current-state AVET key. Value bytes are empty.
pub fn encode_current_avet(attr_id: AttrId, value: &Value, entity: EntityId) -> Vec<u8> {
    let mut key = Vec::with_capacity(1 + ATTR_ID_BYTES + 16 + 8);
    key.push(CURRENT_AVET_PREFIX);
    encode_attr_id(&mut key, attr_id);
    encode_value(&mut key, value);
    encode_entity(&mut key, entity);
    key
}

pub fn current_aevt_attr_prefix(attr_id: AttrId) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1 + ATTR_ID_BYTES);
    buf.push(CURRENT_AEVT_PREFIX);
    encode_attr_id(&mut buf, attr_id);
    buf
}

pub fn current_avet_attr_prefix(attr_id: AttrId) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1 + ATTR_ID_BYTES);
    buf.push(CURRENT_AVET_PREFIX);
    encode_attr_id(&mut buf, attr_id);
    buf
}

pub fn current_avet_attr_value_prefix(attr_id: AttrId, value: &Value) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1 + ATTR_ID_BYTES + 16);
    buf.push(CURRENT_AVET_PREFIX);
    encode_attr_id(&mut buf, attr_id);
    encode_value(&mut buf, value);
    buf
}

pub fn current_avet_attr_type_prefix(attr_id: AttrId, type_tag: u8) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1 + ATTR_ID_BYTES + 1);
    buf.push(CURRENT_AVET_PREFIX);
    encode_attr_id(&mut buf, attr_id);
    buf.push(type_tag);
    buf
}

/// Extract entity ID from a CURRENT_AEVT key.
/// Key layout: [1 prefix][4 attr_id][8 entity_id]
pub fn current_aevt_entity_at(key: &[u8]) -> EntityId {
    let offset = 1 + ATTR_ID_BYTES;
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

    const ATTR_FOO: AttrId = 1;
    const ATTR_BAR: AttrId = 2;

    #[test]
    fn test_eavt_roundtrip() {
        let key = encode_eavt(42, ATTR_FOO, &Value::String("Alice".into()), 1, true);
        let decoded = decode_datom_from_eavt(&key).unwrap();
        assert_eq!(decoded.entity, 42);
        assert_eq!(decoded.attr_id, ATTR_FOO);
        assert_eq!(decoded.value, Value::String("Alice".into()));
        assert_eq!(decoded.tx, 1);
        assert!(decoded.added);
    }

    #[test]
    fn test_aevt_roundtrip() {
        let key = encode_aevt(100, ATTR_BAR, &Value::String("Hello".into()), 5, true);
        let decoded = decode_datom_from_aevt(&key).unwrap();
        assert_eq!(decoded.entity, 100);
        assert_eq!(decoded.attr_id, ATTR_BAR);
        assert_eq!(decoded.value, Value::String("Hello".into()));
        assert_eq!(decoded.tx, 5);
    }

    #[test]
    fn test_avet_roundtrip() {
        let key = encode_avet(7, ATTR_FOO, &Value::I64(30), 2, true);
        let decoded = decode_datom_from_avet(&key).unwrap();
        assert_eq!(decoded.entity, 7);
        assert_eq!(decoded.attr_id, ATTR_FOO);
        assert_eq!(decoded.value, Value::I64(30));
    }

    #[test]
    fn test_i64_sort_order() {
        let k_neg = encode_avet(1, ATTR_FOO, &Value::I64(-10), 1, true);
        let k_zero = encode_avet(1, ATTR_FOO, &Value::I64(0), 1, true);
        let k_pos = encode_avet(1, ATTR_FOO, &Value::I64(10), 1, true);
        assert!(k_neg < k_zero);
        assert!(k_zero < k_pos);
    }

    #[test]
    fn test_f64_roundtrip() {
        for val in [0.0, 1.0, -1.0, f64::MAX, f64::MIN, 3.14, -2.718] {
            let key = encode_eavt(1, ATTR_FOO, &Value::F64(val), 1, true);
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
            let key = encode_eavt(1, ATTR_FOO, &Value::Bool(b), 1, true);
            let decoded = decode_datom_from_eavt(&key).unwrap();
            assert_eq!(decoded.value, Value::Bool(b));
        }
    }

    #[test]
    fn test_ref_roundtrip() {
        let key = encode_eavt(1, ATTR_FOO, &Value::Ref(42), 1, true);
        let decoded = decode_datom_from_eavt(&key).unwrap();
        assert_eq!(decoded.value, Value::Ref(42));
    }

    #[test]
    fn test_encode_datom_vaet_only_for_ref() {
        let string_pairs = encode_datom(1, ATTR_FOO, &Value::String("Alice".into()), 1, true);
        assert_eq!(string_pairs.len(), 3); // EAVT, AEVT, AVET only

        let ref_pairs = encode_datom(1, ATTR_FOO, &Value::Ref(42), 1, true);
        assert_eq!(ref_pairs.len(), 4); // EAVT, AEVT, AVET, VAET
    }

    #[test]
    fn test_vaet_roundtrip_and_value_prefix() {
        let key = encode_vaet(7, ATTR_FOO, &Value::Ref(42), 3, true);
        let decoded = decode_datom_from_vaet(&key).unwrap();
        assert_eq!(decoded.entity, 7);
        assert_eq!(decoded.attr_id, ATTR_FOO);
        assert_eq!(decoded.value, Value::Ref(42));
        assert_eq!(decoded.tx, 3);
        assert!(decoded.added);

        // The value-prefix must bound exactly the keys for that ref value.
        let prefix = vaet_value_prefix(&Value::Ref(42));
        assert!(key.starts_with(&prefix));
        let other = encode_vaet(7, ATTR_FOO, &Value::Ref(43), 3, true);
        assert!(!other.starts_with(&prefix));
    }

    #[test]
    fn test_current_aevt_entity_extraction() {
        let (key, _) = encode_current_aevt(ATTR_FOO, 12345, &Value::I64(7));
        assert_eq!(current_aevt_entity_at(&key), 12345);
    }
}
