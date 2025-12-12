//! Parser for x86reference.xml

use roxmltree::{Document, Node};
use std::error::Error;
use std::fs;

use crate::{AddressingMode, Instruction, Operand, OperandType};

/// Parse the x86reference.xml file
pub fn parse_xml(xml_path: &str) -> Result<Vec<Instruction>, Box<dyn Error>> {
    let xml_content = fs::read_to_string(xml_path)?;
    let doc = Document::parse(&xml_content)?;

    let mut instructions = Vec::new();

    // Parse one-byte opcodes
    if let Some(one_byte) = doc.descendants().find(|n| n.has_tag_name("one-byte")) {
        parse_opcode_section(&one_byte, &mut instructions, false);
    }

    // Parse two-byte opcodes (0F xx)
    if let Some(two_byte) = doc.descendants().find(|n| n.has_tag_name("two-byte")) {
        parse_opcode_section(&two_byte, &mut instructions, true);
    }

    Ok(instructions)
}

fn parse_opcode_section(section: &Node, instructions: &mut Vec<Instruction>, is_two_byte: bool) {
    for pri_opcd in section.children().filter(|n| n.has_tag_name("pri_opcd")) {
        let opcode_value = pri_opcd.attribute("value").unwrap_or("00");

        for entry in pri_opcd.children().filter(|n| n.has_tag_name("entry")) {
            if let Some(instr) = parse_entry(&entry, opcode_value, is_two_byte) {
                instructions.push(instr);
            }
        }
    }
}

fn parse_entry(entry: &Node, opcode_value: &str, is_two_byte: bool) -> Option<Instruction> {
    // Skip invalid entries in 64-bit mode
    let mode = entry.attribute("mode");
    let attr = entry.attribute("attr");

    // Check if invalid in 64-bit mode
    if attr == Some("invd") && mode == Some("e") {
        return None;
    }

    // Skip entries with ref="two-byte" (these are escape codes, not instructions)
    if entry.attribute("ref").is_some() {
        return None;
    }

    // Get syntax element
    let syntax = entry.children().find(|n| n.has_tag_name("syntax"))?;

    // Get mnemonic
    let mnemonic = syntax
        .children()
        .find(|n| n.has_tag_name("mnem"))?
        .text()?
        .to_string();

    // Get opcode extension if present
    let opcode_ext = entry
        .children()
        .find(|n| n.has_tag_name("opcd_ext"))
        .and_then(|n| n.text())
        .and_then(|s| s.parse::<u8>().ok());

    // Get attributes
    let has_modrm = entry.attribute("r") == Some("yes");
    let direction = entry
        .attribute("direction")
        .and_then(|s| s.parse::<u8>().ok());
    let op_size = entry
        .attribute("op_size")
        .and_then(|s| s.parse::<u8>().ok());
    let lockable = entry.attribute("lock") == Some("yes");

    // Check 64-bit validity
    let valid_64bit = attr != Some("invd");
    let only_64bit = mode == Some("e") && attr != Some("invd");

    // Parse operands
    let mut operands = Vec::new();

    // Destination operand
    if let Some(dst) = syntax.children().find(|n| n.has_tag_name("dst")) {
        if let Some(op) = parse_operand(&dst, true) {
            operands.push(op);
        }
    }

    // Source operand(s)
    for src in syntax.children().filter(|n| n.has_tag_name("src")) {
        if let Some(op) = parse_operand(&src, false) {
            operands.push(op);
        }
    }

    // Get brief description
    let brief = entry
        .descendants()
        .find(|n| n.has_tag_name("brief"))
        .and_then(|n| n.text())
        .unwrap_or("")
        .to_string();

    // Format opcode string
    let opcode = if is_two_byte {
        format!("0F {}", opcode_value)
    } else {
        opcode_value.to_string()
    };

    Some(Instruction {
        mnemonic,
        opcode,
        opcode_ext,
        has_modrm,
        operands,
        brief,
        valid_64bit,
        only_64bit,
        direction,
        op_size,
        lockable,
        is_two_byte,
    })
}

fn parse_operand(node: &Node, is_dest: bool) -> Option<Operand> {
    // Check for displayed="no" (implicit operands we don't need)
    if node.attribute("displayed") == Some("no") {
        return None;
    }

    // Get addressing mode from <a> child
    let addressing = if let Some(a_node) = node.children().find(|n| n.has_tag_name("a")) {
        match a_node.text() {
            Some("E") => AddressingMode::ModRmRm,
            Some("G") => AddressingMode::ModRmReg,
            Some("I") => AddressingMode::Immediate,
            Some("J") => AddressingMode::RelativeOffset,
            Some("M") => AddressingMode::MemoryOnly,
            Some("O") => AddressingMode::DirectOffset,
            Some("Z") => AddressingMode::OpcodeReg,
            Some("S") | Some("S2") => AddressingMode::SegmentReg,
            _ => AddressingMode::None,
        }
    } else if let Some(nr) = node.attribute("nr") {
        // Fixed register specified by number
        AddressingMode::Fixed(node.text().unwrap_or("").to_string())
    } else if node.text().is_some() {
        // Fixed register or literal
        AddressingMode::Fixed(node.text().unwrap().to_string())
    } else {
        AddressingMode::None
    };

    // Get operand type from <t> child
    let op_type = if let Some(t_node) = node.children().find(|n| n.has_tag_name("t")) {
        match t_node.text() {
            Some("b") => OperandType::Byte,
            Some("w") => OperandType::Word,
            Some("d") => OperandType::Dword,
            Some("q") => OperandType::Qword,
            Some("v") => OperandType::WordDwordQword,
            Some("vqp") => OperandType::WordDwordQwordPromoted,
            Some("vds") => OperandType::WordDwordSignExtended,
            Some("bs") => OperandType::ByteSignExtended,
            Some(s) => OperandType::Other(s.to_string()),
            None => OperandType::Other("".to_string()),
        }
    } else if let Some(type_attr) = node.attribute("type") {
        match type_attr {
            "b" => OperandType::Byte,
            "w" => OperandType::Word,
            "d" => OperandType::Dword,
            "q" => OperandType::Qword,
            "v" => OperandType::WordDwordQword,
            "vqp" => OperandType::WordDwordQwordPromoted,
            s => OperandType::Other(s.to_string()),
        }
    } else {
        OperandType::Other("".to_string())
    };

    // Get fixed register info
    let fixed_reg = if matches!(addressing, AddressingMode::Fixed(_)) {
        node.text().map(|s| s.to_string())
    } else {
        None
    };

    let reg_nr = node.attribute("nr").and_then(|s| s.parse::<u8>().ok());

    Some(Operand {
        addressing,
        op_type,
        is_dest,
        fixed_reg,
        reg_nr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_xml() {
        // This test requires the actual XML file
        let result = parse_xml("x86reference.xml");
        assert!(result.is_ok());
        let instructions = result.unwrap();
        assert!(!instructions.is_empty());

        // Check that we have some known instructions
        let add_instructions: Vec<_> = instructions
            .iter()
            .filter(|i| i.mnemonic == "ADD")
            .collect();
        assert!(!add_instructions.is_empty());

        let mov_instructions: Vec<_> = instructions
            .iter()
            .filter(|i| i.mnemonic == "MOV")
            .collect();
        assert!(!mov_instructions.is_empty());
    }
}
