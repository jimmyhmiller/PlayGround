use crate::types::*;
use serde_json;
use std::fs;
use std::io;

pub fn parse_ion_json(content: &str) -> Result<IonJSON, ParseError> {
    serde_json::from_str(content).map_err(ParseError::Json)
}

pub fn load_ion_json_from_file(path: &str) -> Result<IonJSON, ParseError> {
    let content = fs::read_to_string(path).map_err(ParseError::Io)?;
    parse_ion_json(&content)
}

pub fn load_ion_json_from_stdin() -> Result<IonJSON, ParseError> {
    let mut input = String::new();
    std::io::Read::read_to_string(&mut io::stdin(), &mut input).map_err(ParseError::Io)?;
    parse_ion_json(&input)
}

// Extract a specific pass from IonJSON based on function name and pass name
pub fn extract_pass(ion_json: &IonJSON, function_name: Option<&str>, pass_name: Option<&str>) -> Result<Pass, ParseError> {
    let func = if let Some(name) = function_name {
        ion_json.functions
            .iter()
            .find(|f| f.name == name)
            .ok_or_else(|| ParseError::NotFound(format!("Function '{}' not found", name)))?
    } else {
        ion_json.functions
            .first()
            .ok_or_else(|| ParseError::NotFound("No functions found in input".to_string()))?
    };

    let pass = if let Some(name) = pass_name {
        func.passes
            .iter()
            .find(|p| p.name == name)
            .ok_or_else(|| ParseError::NotFound(format!("Pass '{}' not found in function '{}'", name, func.name)))?
    } else {
        func.passes
            .last()
            .ok_or_else(|| ParseError::NotFound(format!("No passes found in function '{}'", func.name)))?
    };

    Ok(pass.clone())
}

// Auto-detect if input is ion format by checking for expected structure
pub fn is_ion_json(content: &str) -> bool {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(content) {
        if let Some(obj) = value.as_object() {
            return obj.contains_key("functions") || 
                   (obj.contains_key("mir") && obj.contains_key("lir"));
        }
    }
    false
}

// Parse pass directly if input is already a pass (not full IonJSON)
pub fn parse_pass_direct(content: &str) -> Result<Pass, ParseError> {
    serde_json::from_str(content).map_err(ParseError::Json)
}

#[derive(Debug)]
pub enum ParseError {
    Json(serde_json::Error),
    Io(io::Error),
    NotFound(String),
    InvalidFormat(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::Json(e) => write!(f, "JSON parse error: {}", e),
            ParseError::Io(e) => write!(f, "IO error: {}", e),
            ParseError::NotFound(msg) => write!(f, "Not found: {}", msg),
            ParseError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
        }
    }
}

impl std::error::Error for ParseError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::*;

    #[test]
    fn test_parse_ion_json() {
        let ion_json = create_simple_ion_json();
        let json_str = serde_json::to_string(&ion_json).unwrap();
        
        let parsed = parse_ion_json(&json_str).unwrap();
        assert_eq!(parsed.functions.len(), 1);
        assert_eq!(parsed.functions[0].passes.len(), 1);
    }

    #[test]
    fn test_extract_pass() {
        let ion_json = create_complex_ion_json();
        
        // Extract specific pass
        let pass = extract_pass(&ion_json, Some("test_function"), Some("test_pass")).unwrap();
        assert_eq!(pass.name, "test_pass");
        
        // Extract default (last) pass
        let pass = extract_pass(&ion_json, None, None).unwrap();
        assert!(!pass.name.is_empty());
    }

    #[test]
    fn test_is_ion_json() {
        let ion_json = create_simple_ion_json();
        let json_str = serde_json::to_string(&ion_json).unwrap();
        assert!(is_ion_json(&json_str));
        
        // Test direct pass format
        let pass = create_simple_pass();
        let pass_str = serde_json::to_string(&pass).unwrap();
        assert!(!is_ion_json(&pass_str)); // Should return false for direct pass
    }

    #[test]
    fn test_parse_pass_direct() {
        let pass = create_simple_pass();
        let json_str = serde_json::to_string(&pass).unwrap();
        
        let parsed = parse_pass_direct(&json_str).unwrap();
        assert_eq!(parsed.name, pass.name);
        assert_eq!(parsed.mir.blocks.len(), pass.mir.blocks.len());
    }
}