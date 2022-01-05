use sdl2::pixels::Color;

use crate::tokenizer::{RustSpecific, Token};




pub const BACKGROUND_COLOR: Color = Color::RGB(0x29, 0x2D, 0x3E);
pub const STANDARD_TEXT_COLOR: Color = Color::RGB(0x95, 0x9D, 0xCB);
pub const PURPLE_TEXT_COLOR: Color = Color::RGB(0xC7, 0x92, 0xEA);
pub const BLUE_TEXT_COLOR: Color = Color::RGB(0x82,0xAA,0xFF);
pub const LIGHT_BLUE_TEXT_COLOR: Color = Color::RGB(0x89,0xDD,0xFF);
pub const GREEN_TEXT_COLOR: Color = Color::RGB(0xC3,0xE8, 0x8D);
pub const COMMENT_TEXT_COLOR: Color = Color::RGB(0x67, 0x6E, 0x95);
pub const ORANGE_TEXT_COLOR: Color = Color::RGB(0xF7, 0x8C, 0x6C);
pub const CURSOR_COLOR: Color = Color::RGBA(255, 204, 0, 255);


pub fn color_for_token(token: &RustSpecific, input_bytes: &[u8]) -> Color {
    match token {
        RustSpecific::Keyword(_) => PURPLE_TEXT_COLOR,
        RustSpecific::Token(Token::Comment(_)) =>  COMMENT_TEXT_COLOR,
        RustSpecific::Token(t) => {
            match t {
                Token::Comment(_) => {
                    COMMENT_TEXT_COLOR
                },
                Token::OpenBracket |
                Token::CloseBracket |
                Token::OpenParen |
                Token::CloseParen |
                Token::OpenCurly |
                Token::CloseCurly |
                Token::Comma => {
                    LIGHT_BLUE_TEXT_COLOR
                },
                Token::Atom((s,_e)) => {
                    if input_bytes[*s].is_ascii_uppercase() {
                        BLUE_TEXT_COLOR
                    } else {
                        STANDARD_TEXT_COLOR
                    }
                },
                Token::String(_) => {
                    GREEN_TEXT_COLOR
                },
                Token::Integer(_) | Token::Float(_) => {
                    ORANGE_TEXT_COLOR
                }
                _ => {
                    STANDARD_TEXT_COLOR
                }
            }
        }
    }
}
