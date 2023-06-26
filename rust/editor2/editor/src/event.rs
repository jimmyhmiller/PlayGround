use std::path::PathBuf;

use crate::{
    keyboard::{KeyCode, KeyState, KeyboardInput, Modifiers},
    widget::{Position, WidgetId, WidgetSelector},
};

use serde::{Deserialize, Serialize};
use winit::event::{Event as WinitEvent, WindowEvent as WinitWindowEvent};

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
    MoveWidgetRelative {
        selector: WidgetSelector,
        x: f32,
        y: f32,
    },
    ReloadWidgets,
    ReloadWasm(String),
    SaveWidgets,
    StartProcess(usize, usize, String),
    SendProcessMessage(usize, String),
}

impl Event {
    pub fn patch_mouse_event(&mut self, mouse_pos: &Position) {
        match self {
            Event::LeftMouseDown { x, y } => {
                *x = mouse_pos.x;
                *y = mouse_pos.y;
            }
            Event::LeftMouseUp { x, y } => {
                *x = mouse_pos.x;
                *y = mouse_pos.y;
            }
            Event::RightMouseDown { x, y } => {
                *x = mouse_pos.x;
                *y = mouse_pos.y;
            }
            Event::RightMouseUp { x, y } => {
                *x = mouse_pos.x;
                *y = mouse_pos.y;
            }
            Event::HoveredFile { x, y, .. } => {
                *x = mouse_pos.x;
                *y = mouse_pos.y;
            }
            Event::DroppedFile { x, y, .. } => {
                *x = mouse_pos.x;
                *y = mouse_pos.y;
            }
            _ => {}
        }
    }

    pub fn from_winit_event(event: &WinitEvent<'_, ()>, modifiers: Modifiers) -> Option<Self> {
        match event {
            WinitEvent::WindowEvent { event, .. } => {
                use WinitWindowEvent::*;
                match event {
                    CloseRequested => Some(Event::Noop),
                    TouchpadPressure {
                        device_id: _,
                        pressure: _,
                        stage: _,
                    } => Some(Event::Noop),
                    MouseWheel { delta, .. } => match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, _) => {
                            panic!("What is line delta?")
                        }
                        winit::event::MouseScrollDelta::PixelDelta(delta) => Some(Event::Scroll {
                            x: delta.x,
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

                    ModifiersChanged(state) => {
                        let ctrl = state.ctrl();
                        let option = state.alt();
                        let shift = state.shift();
                        let cmd = state.logo();
                        Some(Event::ModifiersChanged(Modifiers {
                            shift,
                            ctrl,
                            cmd,
                            option,
                        }))
                    }

                    KeyboardInput { input, .. } => {
                        let key_code = input
                            .virtual_keycode
                            .and_then(KeyCode::map_winit_vk_to_keycode)?;

                        let state = match input.state {
                            winit::event::ElementState::Pressed => KeyState::Pressed,
                            winit::event::ElementState::Released => KeyState::Released,
                        };

                        Some(Event::KeyEvent {
                            input: crate::keyboard::KeyboardInput {
                                state,
                                key_code,
                                modifiers,
                            },
                        })
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
