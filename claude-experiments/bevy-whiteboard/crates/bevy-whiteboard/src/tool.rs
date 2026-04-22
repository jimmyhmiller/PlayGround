use bevy::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    Select,
    Generator,
    Client,
    Worker,
    Sink,
    Router,
    Queue,
    Steps,
    Edge,
    Probe,
    /// User-defined preset from `PresetLibrary.user[i]`. Behaves
    /// like Client/Worker — clicking inside a Steps container
    /// appends the preset's Sequence as a new row. Clicking on
    /// empty canvas does nothing (presets are steps, not nodes).
    UserPreset(usize),
    /// Drop a primitive instruction into an opened node. Used by
    /// the step-library palette section. Clicking an opened node
    /// appends the primitive to its program; clicking empty canvas
    /// does nothing.
    Primitive(PrimitiveKind),
}

/// The subset of [`crate::sim::Instruction`] variants exposed as
/// clickable tiles in the step-library palette. Each variant knows
/// how to produce a default instruction and its display label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveKind {
    Filter,
    Sort,
    Take,
    Send,
    Require,
    Match,
    Process,
    Hold,
    Buffer,
    Respond,
    Consume,
    Emit,
    Accept,
    AwaitResponse,
    Sequence,
}

impl PrimitiveKind {
    pub fn label(self) -> &'static str {
        match self {
            PrimitiveKind::Filter => "Filter",
            PrimitiveKind::Sort => "Sort",
            PrimitiveKind::Take => "Take",
            PrimitiveKind::Send => "Send",
            PrimitiveKind::Require => "Require",
            PrimitiveKind::Match => "Match",
            PrimitiveKind::Process => "Process",
            PrimitiveKind::Hold => "Hold",
            PrimitiveKind::Buffer => "Buffer",
            PrimitiveKind::Respond => "Respond",
            PrimitiveKind::Consume => "Consume",
            PrimitiveKind::Emit => "Emit",
            PrimitiveKind::Accept => "Accept",
            PrimitiveKind::AwaitResponse => "Await",
            PrimitiveKind::Sequence => "Sequence",
        }
    }

    pub fn all() -> [PrimitiveKind; 15] {
        [
            PrimitiveKind::Filter,
            PrimitiveKind::Sort,
            PrimitiveKind::Take,
            PrimitiveKind::Send,
            PrimitiveKind::Require,
            PrimitiveKind::Match,
            PrimitiveKind::Process,
            PrimitiveKind::Hold,
            PrimitiveKind::Buffer,
            PrimitiveKind::Respond,
            PrimitiveKind::Consume,
            PrimitiveKind::Emit,
            PrimitiveKind::Accept,
            PrimitiveKind::AwaitResponse,
            PrimitiveKind::Sequence,
        ]
    }

    /// Build a default `Instruction` for this primitive. Used when
    /// the user clicks a palette tile to insert into a program.
    pub fn default_instruction(self, color: crate::sim::Color) -> crate::sim::Instruction {
        use crate::sim::{Instruction, LostReason, NS_PER_MS, PortKey, PortPredicate};
        match self {
            PrimitiveKind::Filter => Instruction::Filter { pred: PortPredicate::Ready },
            PrimitiveKind::Sort => Instruction::Sort { key: PortKey::LastSentAt },
            PrimitiveKind::Take => Instruction::Take { n: 1 },
            PrimitiveKind::Send => Instruction::Send,
            PrimitiveKind::Require => Instruction::Require { reason: LostReason::NoReadyOutbound },
            PrimitiveKind::Match => Instruction::MatchColor { color },
            PrimitiveKind::Process => Instruction::Process { duration_ns: 500 * NS_PER_MS },
            PrimitiveKind::Hold => Instruction::Hold { duration_ns: 500 * NS_PER_MS },
            PrimitiveKind::Buffer => Instruction::Buffer { capacity: usize::MAX },
            PrimitiveKind::Respond => Instruction::Respond,
            PrimitiveKind::Consume => Instruction::Consume,
            PrimitiveKind::Emit => Instruction::Emit { color, one_way: false },
            PrimitiveKind::Accept => Instruction::AcceptInbound,
            PrimitiveKind::AwaitResponse => Instruction::AwaitResponse,
            PrimitiveKind::Sequence => Instruction::Sequence {
                label: "Group".into(),
                body: Vec::new(),
            },
        }
    }
}

impl Tool {
    pub fn label(self) -> &'static str {
        match self {
            Tool::Select => "Select",
            Tool::Generator => "Generator",
            Tool::Client => "Client",
            Tool::Worker => "Worker",
            Tool::Sink => "Sink",
            Tool::Router => "Router",
            Tool::Queue => "Queue",
            Tool::Steps => "Steps",
            Tool::Edge => "Edge",
            Tool::Probe => "Probe",
            Tool::UserPreset(_) => "Preset",
            Tool::Primitive(_) => "Primitive",
        }
    }
}

#[derive(Resource)]
pub struct ActiveTool(pub Tool);

impl Default for ActiveTool {
    fn default() -> Self {
        ActiveTool(Tool::Select)
    }
}

/// The currently selected data-color (palette entry).
/// Generators emit packets of this color. Used as a crude type system.
#[derive(Resource)]
pub struct ActiveColor(pub Color);

impl Default for ActiveColor {
    fn default() -> Self {
        ActiveColor(PALETTE_COLORS[0])
    }
}

pub const PALETTE_COLORS: [Color; 6] = [
    Color::srgb(0.90, 0.30, 0.30), // red
    Color::srgb(0.95, 0.70, 0.20), // amber
    Color::srgb(0.40, 0.75, 0.35), // green
    Color::srgb(0.25, 0.55, 0.90), // blue
    Color::srgb(0.65, 0.40, 0.85), // purple
    Color::srgb(0.25, 0.25, 0.30), // slate
];
