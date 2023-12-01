use std::path::PathBuf;

use crate::{
    keyboard::{KeyCode, KeyState, KeyboardInput, Modifiers},
    widget::WidgetId,
};

use framework::{CursorIcon, Position, Value};
use serde::{Deserialize, Serialize};
use winit::{
    event::{Event as WinitEvent, WindowEvent as WinitWindowEvent},
    keyboard::{ModifiersKeyState, PhysicalKey},
};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum TouchPhase {
    Started,
    Moved,
    Ended,
    Cancelled,
}

impl From<&winit::event::TouchPhase> for TouchPhase {
    fn from(phase: &winit::event::TouchPhase) -> Self {
        match phase {
            winit::event::TouchPhase::Started => TouchPhase::Started,
            winit::event::TouchPhase::Moved => TouchPhase::Moved,
            winit::event::TouchPhase::Ended => TouchPhase::Ended,
            winit::event::TouchPhase::Cancelled => TouchPhase::Cancelled,
        }
    }
}


#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Event {
    Noop,
    KeyEvent {
        // I maybe shouldn't use this. But I'm lazy right now.
        input: KeyboardInput,
    },
    ModifiersChanged(Modifiers),
    MouseMove {
        x: f32,
        y: f32,
        x_diff: f32,
        y_diff: f32,
    },
    LeftMouseDown {
        x: f32,
        y: f32,
    },
    LeftMouseUp {
        x: f32,
        y: f32,
    },
    RightMouseDown {
        x: f32,
        y: f32,
    },
    RightMouseUp {
        x: f32,
        y: f32,
    },
    Scroll {
        x: f64,
        y: f64,
    },
    HoveredFile {
        path: PathBuf,
        x: f32,
        y: f32,
    },
    DroppedFile {
        path: PathBuf,
        x: f32,
        y: f32,
    },
    HoveredFileCancelled,
    WidgetMouseDown {
        widget_id: WidgetId,
    },
    WidgetMouseUp {
        widget_id: WidgetId,
    },
    ReloadWidgets,
    ReloadWasm(String),
    SaveWidgets,
    StartProcess(usize, usize, String),
    SendProcessMessage(usize, String),
    Event(String, String),
    Subscribe(usize, String),
    Unsubscribe(usize, String),
    OpenFile(String),
    SetCursor(CursorIcon),
    Redraw(usize),
    CreateWidget(usize, f32, f32, f32, f32, u32),
    ValueNeeded(String, usize),
    ProvideValue(String, Value),
    PinchZoom { delta: f64, phase: TouchPhase},
    TouchPadPressure { pressure: f32, stage: i64 },
    SmartMagnify { x: f32, y: f32 },
    MarkDirty(u32),
}

impl Event {
    pub fn patch_mouse_event(&mut self, mouse_pos: &Position, canvas_offset: &Position, canvas_scale: f32) {
        match self {
            Event::LeftMouseDown { x, y } |
            Event::LeftMouseUp { x, y } |
            Event::RightMouseDown { x, y } |
            Event::RightMouseUp { x, y } |
            Event::HoveredFile { x, y, .. } |
            Event::DroppedFile { x, y, .. } |
            Event::SmartMagnify { x, y} 
             => {
                *x = (mouse_pos.x + canvas_offset.x) * canvas_scale;
                *y = (mouse_pos.y - canvas_offset.y) * canvas_scale;
            }
            _ => {}
        }
    }

    pub fn from_winit_event(event: &WinitEvent<()>, modifiers: Modifiers) -> Option<Self> {
        match event {
            WinitEvent::WindowEvent { event, .. } => {
                use WinitWindowEvent::*;
                match event {
                    CloseRequested => Some(Event::Noop),
                    TouchpadPressure {
                        device_id: _,
                        pressure,
                        stage,
                    } => Some(Event::TouchPadPressure { pressure: *pressure, stage: *stage }),
                    MouseWheel { delta, .. } => match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, _) => {
                            panic!("What is line delta?")
                        }
                        winit::event::MouseScrollDelta::PixelDelta(delta) => Some(Event::Scroll {
                            x: -delta.x,
                            y: delta.y,
                        }),
                    },
                    MouseInput { state, button, .. } => {
                        use winit::event::ElementState::*;
                        use winit::event::MouseButton::*;
                        match (state, button) {
                            // Silly hack for the fact that I don't know positions here.
                            (Pressed, Left) => Some(Event::LeftMouseDown { x: -0.0, y: -0.0 }),
                            (Released, Left) => Some(Event::LeftMouseUp { x: -0.0, y: -0.0 }),
                            (Pressed, Right) => Some(Event::RightMouseDown { x: -0.0, y: -0.0 }),
                            (Released, Right) => Some(Event::RightMouseUp { x: -0.0, y: -0.0 }),
                            _ => None,
                        }
                    }
                    CursorMoved { position, .. } => Some(Event::MouseMove {
                        x: position.x as f32,
                        y: position.y as f32,
                        x_diff: 0.0,
                        y_diff: 0.0,
                    }),
                    TouchpadMagnify { device_id: _, delta, phase } => {
                        Some(Event::PinchZoom { delta: *delta, phase: phase.into() })
                    }
                    SmartMagnify { device_id: _ } => {
                        Some(Event::SmartMagnify { x: 0.0, y: 0.0})
                    }
                    HoveredFile(path) => Some(Event::HoveredFile {
                        path: path.to_path_buf(),
                        x: -0.0,
                        y: -0.0,
                    }),
                    DroppedFile(path) => Some(Event::DroppedFile {
                        path: path.to_path_buf(),
                        x: -0.0,
                        y: -0.0,
                    }),
                    HoveredFileCancelled => Some(Event::HoveredFileCancelled),

                    ModifiersChanged(modifiers) => {
                        let ctrl = matches!(modifiers.lcontrol_state(), ModifiersKeyState::Pressed)
                            || matches!(modifiers.rcontrol_state(), ModifiersKeyState::Pressed);
                        let option = matches!(modifiers.lalt_state(), ModifiersKeyState::Pressed)
                            || matches!(modifiers.ralt_state(), ModifiersKeyState::Pressed);
                        let shift = matches!(modifiers.lshift_state(), ModifiersKeyState::Pressed)
                            || matches!(modifiers.rshift_state(), ModifiersKeyState::Pressed);
                        let cmd = matches!(modifiers.lsuper_state(), ModifiersKeyState::Pressed)
                            || matches!(modifiers.rsuper_state(), ModifiersKeyState::Pressed);
                        Some(Event::ModifiersChanged(Modifiers {
                            shift,
                            ctrl,
                            cmd,
                            option,
                        }))
                    }

                    KeyboardInput { event, .. } => {
                        match event.physical_key {
                            PhysicalKey::Code(key_code) => {
                                let state = match event.state {
                                    winit::event::ElementState::Pressed => KeyState::Pressed,
                                    winit::event::ElementState::Released => KeyState::Released,
                                };

                                let key_code = KeyCode::map_winit_vk_to_keycode(key_code)?;

                                Some(Event::KeyEvent {
                                    input: crate::keyboard::KeyboardInput {
                                        state,
                                        key_code,
                                        modifiers,
                                    },
                                })
                            }
                            PhysicalKey::Unidentified(_) => {
                                println!("Unidentified key: {:?}", event);
                                None
                            }
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

// Events
// ShowScene { widgets: [WidgetSelector]}
// ShowWidget { widget: One<WidgetSelector> }
// HideWidget { widget: One<WidgetSelector> }
// AddClick { action: [Action] }
// RemoveClick { action: [Action] }
// SendEvent { process: One<ProcessId> }

// I need to think about my abstractions here
// A scene is a list of widgets being displayed
// But I might also want to have cards or boards or whatever
// Persisted scenes more or less
// Then you can show them by identifier
// rather than as a collection
// I also want you to be able to do this adhoc
// by searching for widgets to show.

// Should a scene be a widget? I think that's not quite right
// A board can be a widget
// but a scene is really just what is currently shown
// Maybe that is the abstraction?

// I might want a temp message widget that appears and disappears
// Would be useful for demos, but also just in general
// I also need a command pallete. Might consider enso style as an option.
// I do that, I need some notion of timers
// Which I definitely want regardless

// If timers just push an event into the queue
// I can handle them fairly generically.
// That plus widget selector on position could make for
// a game of life demo.
