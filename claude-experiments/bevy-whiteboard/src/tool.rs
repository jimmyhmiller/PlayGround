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
    Edge,
    Probe,
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
            Tool::Edge => "Edge",
            Tool::Probe => "Probe",
        }
    }

    pub fn all() -> [Tool; 9] {
        [
            Tool::Select,
            Tool::Generator,
            Tool::Client,
            Tool::Worker,
            Tool::Sink,
            Tool::Router,
            Tool::Queue,
            Tool::Edge,
            Tool::Probe,
        ]
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
