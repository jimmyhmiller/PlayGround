use serde::{Deserialize, Serialize};
use winit::event::VirtualKeyCode;

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum KeyState {
    Pressed,
    Released,
}

impl KeyState {
    fn to_u32(&self) -> u32 {
        match self {
            KeyState::Pressed => 0,
            KeyState::Released => 1,
        }
    }

    #[allow(dead_code)]
    fn from_u32(value: u32) -> Self {
        match value {
            0 => KeyState::Pressed,
            1 => KeyState::Released,
            _ => panic!("Invalid value for KeyState"),
        }
    }
}

// Not the most efficient representation
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default,
)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub option: bool,
    pub cmd: bool,
}

impl Modifiers {
    fn from_u32(modifiers: u32) -> Self {
        Self {
            shift: (modifiers & 1) != 0,
            ctrl: (modifiers & 2) != 0,
            option: (modifiers & 4) != 0,
            cmd: (modifiers & 8) != 0,
        }
    }
    fn to_u32(&self) -> u32 {
        let mut result = 0;
        if self.shift {
            result |= 1;
        }
        if self.ctrl {
            result |= 2;
        }
        if self.option {
            result |= 4;
        }
        if self.cmd {
            result |= 8;
        }
        result
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum KeyCode {
    Key0,
    Key1,
    Key2,
    Key3,
    Key4,
    Key5,
    Key6,
    Key7,
    Key8,
    Key9,
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Equals,
    Minus,
    LeftBracket,
    RightBracket,
    Backslash,
    Semicolon,
    Apostrophe,
    Comma,
    Period,
    Slash,
    Grave,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Escape,
    Space,
    Return,
    LeftArrow,
    RightArrow,
    UpArrow,
    DownArrow,
    BackSpace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyboardInput {
    pub state: KeyState,
    pub key_code: KeyCode,
    pub modifiers: Modifiers,
}

impl KeyCode {
    pub fn map_winit_vk_to_keycode(v: VirtualKeyCode) -> Option<KeyCode> {
        // TODO: VirtualKeyCode is serializable. Should I just use that?
        use VirtualKeyCode::*;
        match v {
            Key0 => Some(KeyCode::Key0),
            Key1 => Some(KeyCode::Key1),
            Key2 => Some(KeyCode::Key2),
            Key3 => Some(KeyCode::Key3),
            Key4 => Some(KeyCode::Key4),
            Key5 => Some(KeyCode::Key5),
            Key6 => Some(KeyCode::Key6),
            Key7 => Some(KeyCode::Key7),
            Key8 => Some(KeyCode::Key8),
            Key9 => Some(KeyCode::Key9),
            A => Some(KeyCode::A),
            B => Some(KeyCode::B),
            C => Some(KeyCode::C),
            D => Some(KeyCode::D),
            E => Some(KeyCode::E),
            F => Some(KeyCode::F),
            G => Some(KeyCode::G),
            H => Some(KeyCode::H),
            I => Some(KeyCode::I),
            J => Some(KeyCode::J),
            K => Some(KeyCode::K),
            L => Some(KeyCode::L),
            M => Some(KeyCode::M),
            N => Some(KeyCode::N),
            O => Some(KeyCode::O),
            P => Some(KeyCode::P),
            Q => Some(KeyCode::Q),
            R => Some(KeyCode::R),
            S => Some(KeyCode::S),
            T => Some(KeyCode::T),
            U => Some(KeyCode::U),
            V => Some(KeyCode::V),
            W => Some(KeyCode::W),
            X => Some(KeyCode::X),
            Y => Some(KeyCode::Y),
            Z => Some(KeyCode::Z),
            Equals => Some(KeyCode::Equals),
            Minus => Some(KeyCode::Minus),
            LBracket => Some(KeyCode::LeftBracket),
            RBracket => Some(KeyCode::RightBracket),
            Backslash => Some(KeyCode::Backslash),
            Semicolon => Some(KeyCode::Semicolon),
            Apostrophe => Some(KeyCode::Apostrophe),
            Comma => Some(KeyCode::Comma),
            Period => Some(KeyCode::Period),
            Slash => Some(KeyCode::Slash),
            Grave => Some(KeyCode::Grave),
            F1 => Some(KeyCode::F1),
            F2 => Some(KeyCode::F2),
            F3 => Some(KeyCode::F3),
            F4 => Some(KeyCode::F4),
            F5 => Some(KeyCode::F5),
            F6 => Some(KeyCode::F6),
            F7 => Some(KeyCode::F7),
            F8 => Some(KeyCode::F8),
            F9 => Some(KeyCode::F9),
            F10 => Some(KeyCode::F10),
            F11 => Some(KeyCode::F11),
            F12 => Some(KeyCode::F12),
            Escape => Some(KeyCode::Escape),
            Space => Some(KeyCode::Space),
            Return => Some(KeyCode::Return),
            Left => Some(KeyCode::LeftArrow),
            Right => Some(KeyCode::RightArrow),
            Up => Some(KeyCode::UpArrow),
            Down => Some(KeyCode::DownArrow),
            Back => Some(KeyCode::BackSpace),
            _ => None,
        }
    }

    fn from_u32(key: u32) -> Option<Self> {
        match key {
            0 => Some(KeyCode::Key0),
            1 => Some(KeyCode::Key1),
            2 => Some(KeyCode::Key2),
            3 => Some(KeyCode::Key3),
            4 => Some(KeyCode::Key4),
            5 => Some(KeyCode::Key5),
            6 => Some(KeyCode::Key6),
            7 => Some(KeyCode::Key7),
            8 => Some(KeyCode::Key8),
            9 => Some(KeyCode::Key9),
            10 => Some(KeyCode::A),
            11 => Some(KeyCode::B),
            12 => Some(KeyCode::C),
            13 => Some(KeyCode::D),
            14 => Some(KeyCode::E),
            15 => Some(KeyCode::F),
            16 => Some(KeyCode::G),
            17 => Some(KeyCode::H),
            18 => Some(KeyCode::I),
            19 => Some(KeyCode::J),
            20 => Some(KeyCode::K),
            21 => Some(KeyCode::L),
            22 => Some(KeyCode::M),
            23 => Some(KeyCode::N),
            24 => Some(KeyCode::O),
            25 => Some(KeyCode::P),
            26 => Some(KeyCode::Q),
            27 => Some(KeyCode::R),
            28 => Some(KeyCode::S),
            29 => Some(KeyCode::T),
            30 => Some(KeyCode::U),
            31 => Some(KeyCode::V),
            32 => Some(KeyCode::W),
            33 => Some(KeyCode::X),
            34 => Some(KeyCode::Y),
            35 => Some(KeyCode::Z),
            36 => Some(KeyCode::Equals),
            37 => Some(KeyCode::Minus),
            38 => Some(KeyCode::LeftBracket),
            39 => Some(KeyCode::RightBracket),
            40 => Some(KeyCode::Backslash),
            41 => Some(KeyCode::Semicolon),
            42 => Some(KeyCode::Apostrophe),
            43 => Some(KeyCode::Comma),
            44 => Some(KeyCode::Period),
            45 => Some(KeyCode::Slash),
            46 => Some(KeyCode::Grave),
            47 => Some(KeyCode::F1),
            48 => Some(KeyCode::F2),
            49 => Some(KeyCode::F3),
            50 => Some(KeyCode::F4),
            51 => Some(KeyCode::F5),
            52 => Some(KeyCode::F6),
            53 => Some(KeyCode::F7),
            54 => Some(KeyCode::F8),
            55 => Some(KeyCode::F9),
            56 => Some(KeyCode::F10),
            57 => Some(KeyCode::F11),
            58 => Some(KeyCode::F12),
            59 => Some(KeyCode::Escape),
            60 => Some(KeyCode::Space),
            61 => Some(KeyCode::Return),
            62 => Some(KeyCode::LeftArrow),
            63 => Some(KeyCode::RightArrow),
            64 => Some(KeyCode::UpArrow),
            65 => Some(KeyCode::DownArrow),
            66 => Some(KeyCode::BackSpace),
            _ => None,
        }
    }

    fn to_u32(&self) -> u32 {
        *self as u32
    }
}

impl KeyboardInput {
    #[allow(dead_code)]
    pub fn from_u32(key: u32, state: u32, modifiers: u32) -> Self {
        Self {
            state: KeyState::from_u32(state),
            key_code: KeyCode::from_u32(key).unwrap(),
            modifiers: Modifiers::from_u32(modifiers),
        }
    }

    pub fn to_u32_tuple(&self) -> (u32, u32, u32) {
        (
            self.key_code.to_u32(),
            self.state.to_u32(),
            self.modifiers.to_u32(),
        )
    }
}
