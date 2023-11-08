pub fn encode_base64(data: &str) -> String {
    use base64::Engine;

    base64::engine::general_purpose::STANDARD.encode(data)
}

pub fn decode_base64(data: &Vec<u8>) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use base64::Engine;
    let data = base64::engine::general_purpose::STANDARD.decode(data)?;
    Ok(data)
}
