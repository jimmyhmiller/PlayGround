use std::{process::{Child, ChildStdout, Command, Stdio}, ops::Neg, str::from_utf8};

use nonblock::NonBlockingReader;
use sdl2::{clipboard::ClipboardUtil, keyboard::{Scancode, Keycode, Mod}, event::{Event, WindowEvent}};

use crate::{PaneManager, renderer::EditorBounds, cursor::Cursor, transaction::EditAction, native};


#[derive(Debug, Clone)]
pub enum Action {
    MoveCursorUp(usize),
    MoveCursorDown(usize),
    MoveCursorLeft(usize),
    MoveCursorRight(usize),
    Delete(usize),
    RunPane(usize),
    InsertNewline(usize),
    Undo(usize),
    Redo(usize),
    MoveCursorToLineStart(usize),
    MoveCursorToLineEnd(usize),
    Copy(usize),
    Paste(usize),
    OpenFile(usize),
    SelectAll(usize),
    TextInput(usize, String),
    DeletePane(usize),
    StartEditPaneName(usize),
    EndEditPaneName(usize),
    StartResizePane(usize, (i32, i32)),
    EndResizePane((i32, i32)),
    StartCreatePane((i32, i32)),
    EndCreatePane((i32, i32)),
    StartMovePane(usize, (i32, i32)),
    EndMovePane((i32, i32)),
    DuplicatePane(usize),
    SetPaneActive(usize),
    SetScrollPane(usize),
    MouseDown(usize, (i32, i32)),
    MoveMouse((i32, i32)),
    ResizeWindow(i32, i32),
    Scroll(usize),
    Quit,

    // TODO:
    // These depend on the state of the system
    // For example, a click on a pane might more might not
    // set a cursor. We should generate these,
    // but they would be on a second pass.

    // SetCursor(usize, (i32, i32)),
    // // Set cursor line/col?
    // StartSelection(usize, (i32, i32)),
    // EndSelection(usize, (i32, i32)),

    
}

impl Action {
    pub fn pane_id(&self) -> Option<usize> {
        match self {
            Action::MoveCursorUp(id) |
            Action::MoveCursorDown(id) |
            Action::MoveCursorLeft(id) |
            Action::MoveCursorRight(id) |
            Action::Delete(id) |
            Action::RunPane(id) |
            Action::InsertNewline(id) |
            Action::Undo(id) |
            Action::Redo(id) |
            Action::MoveCursorToLineStart(id) |
            Action::MoveCursorToLineEnd(id) |
            Action::Copy(id) |
            Action::Paste(id) |
            Action::OpenFile(id) |
            Action::SelectAll(id) |
            Action::TextInput(id, _) |
            Action::DeletePane(id) |
            Action::StartEditPaneName(id) |
            Action::EndEditPaneName(id) |
            Action::StartResizePane(id, _) |
            Action::StartMovePane(id, _) |
            Action::DuplicatePane(id) |
            Action::SetPaneActive(id) |
            Action::SetScrollPane(id) |
            Action::MouseDown(id, _) |
            Action::Scroll(id) => Some(*id),
            _ => None,
        }
    }
}


pub enum PerFrameAction {
    ReadCommand(String, Child, NonBlockingReader<ChildStdout>),
    DisplayError(String, String),
}


pub enum PerFrameActionResult {
    RemoveAction(usize),
    Noop,
}




pub fn handle_events(
        event_pump: &mut sdl2::EventPump,
        pane_manager: &mut PaneManager,
        bounds: &EditorBounds,
        clipboard: &ClipboardUtil) -> Vec<Action> {
    let mut is_text_input = false;

    let mut actions: Vec<Action> = vec![];

    // let text_buffer = &mut pane.text_buffer;
    // let cursor_context = &mut pane.cursor_context;
    // let scroller = &mut pane.scroller;

    // This whole way of handling things is wrong.
    // We probably need a pane manager.
    // Maybe good chance for a pun?

    let ctrl_is_pressed = event_pump.keyboard_state().is_scancode_pressed(Scancode::LCtrl);
    let alt_is_pressed = event_pump.keyboard_state().is_scancode_pressed(Scancode::LAlt);
    let cmd_is_pressed = event_pump.keyboard_state().is_scancode_pressed(Scancode::LGui);

    for event in event_pump.poll_iter() {
        // println!("frame: {}, event {:?}", frame_counter, event);
        match event {
            Event::Quit { .. } => {
                actions.push(Action::Quit);
                ::std::process::exit(0);
            },
            // Note: These work I can do enso style quasimodal input
            // Event::KeyUp {keycode, ..} => {
            //     println!("{:?}", keycode);
            // }
            // Event::KeyDown{keycode: Some(Keycode::Escape), ..} => {
            //     println!("{:?}", "yep");
            // }

            Event::KeyDown { keycode, keymod, .. } => {
                matches!(keycode, Some(_));
                match (keycode.unwrap(), keymod) {
                    (Keycode::Up, _) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            pane.cursor_context.move_up(&pane.text_buffer);
                            actions.push(Action::MoveCursorUp(pane.id));
                        }
                    },
                    (Keycode::Down, _) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            pane.cursor_context.move_down(&pane.text_buffer);
                            actions.push(Action::MoveCursorDown(pane.id));
                        }
                    },
                    (Keycode::Left, _) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            pane.cursor_context.move_left(&pane.text_buffer);
                            actions.push(Action::MoveCursorLeft(pane.id));
                        }
                    },
                    (Keycode::Right, _) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            pane.cursor_context.move_right(&pane.text_buffer);
                            actions.push(Action::MoveCursorRight(pane.id));
                        }
                    },
                    (Keycode::Backspace, _) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            actions.push(Action::Delete(pane.id));

                            if pane.editing_name {
                                pane.name.pop();
                                continue;
                            }
                            // Need to deal with this in a nicer way

                            if let Some(current_selection) = pane.cursor_context.selection {
                                let (start, end) = current_selection;
                                let (start_line, start_column) = start;
                                let (end_line, end_column) = end;
                                if let Some((line_start, _line_end)) = pane.text_buffer.get_line(start_line as usize) {
                                    let char_start_pos = line_start + start_column as usize ;
                                    if let Some((end_line_start, _line_end)) = pane.text_buffer.get_line(end_line as usize) {
                                        let char_end_pos = end_line_start + end_column as usize;
                                        pane.text_buffer.chars.drain(char_start_pos as usize..char_end_pos as usize);
                                        // Probably shouldn't reparse the whole file.

                                        pane.text_buffer.parse_lines();
                                        pane.cursor_context.clear_selection();
                                        pane.cursor_context.fix_cursor(&pane.text_buffer);
                                        continue;
                                    }

                                }
                            }


                            // Is there a better way to do this other than clone?
                            // Maybe a non-mutating method?
                            // How to deal with optional aspect here?
                            if let Some(current_cursor) = pane.cursor_context.cursor {
                                let mut old_cursor = current_cursor;
                                // We do this move_left first, because otherwise we might end up at the end
                                // of the new line we formed from the deletion, rather than the old end of the line.
                                let cursor_action = old_cursor.move_left(&pane.text_buffer);
                                let action = pane.text_buffer.remove_char(current_cursor);

                                pane.transaction_manager.add_action(action);
                                pane.transaction_manager.add_action(cursor_action);


                                pane.cursor_context.set_cursor(old_cursor);
                            }
                        }
                    }
                    (Keycode::Return, _) if cmd_is_pressed => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            actions.push(Action::RunPane(pane.id));
                        }
                    }
                    (Keycode::Return, _) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {

                            if pane.editing_name {
                                pane.editing_name = false;
                                actions.push(Action::EndEditPaneName(pane.id));
                                continue;
                            }
                            // refactor to be better
                            let action = pane.cursor_context.handle_insert(&[b'\n'], &mut pane.text_buffer);
                            pane.transaction_manager.add_action(action);
                            pane.cursor_context.start_of_line();
                            actions.push(Action::InsertNewline(pane.id));
                        }
                    },


                     (Keycode::Z, key_mod) if key_mod == Mod::LGUIMOD || keymod == Mod::RGUIMOD => {
                        // I should think about the fact that I might not have an active editor
                        // Should undo undo pane movement?
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            pane.transaction_manager.undo(&mut pane.cursor_context, &mut pane.text_buffer);
                            actions.push(Action::Undo(pane.id));
                        }
                     }

                     (Keycode::Z, key_mod) if key_mod == (Mod::LSHIFTMOD | Mod::LGUIMOD) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            pane.transaction_manager.redo(&mut pane.cursor_context, &mut pane.text_buffer);
                            actions.push(Action::Redo(pane.id));
                        }
                     }


                    (Keycode::Z, _key_mod) => {
                        is_text_input = true
                    },

                    (Keycode::A, Mod::LCTRLMOD | Mod::RCTRLMOD) => {
                        
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            pane.cursor_context.start_of_line();
                            actions.push(Action::MoveCursorToLineStart(pane.id));
                        }
                    },
                    (Keycode::E, Mod::LCTRLMOD | Mod::RCTRLMOD) => {
                        
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            pane.cursor_context.end_of_line(&pane.text_buffer);
                            actions.push(Action::MoveCursorToLineEnd(pane.id));
                        }
                    },

                    (Keycode::C, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            pane.cursor_context.copy_selection(&clipboard, &pane.text_buffer);
                            actions.push(Action::Copy(pane.id));
                        }
                    },
                    (Keycode::V, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            // TODO: I NEED UNDO!
                            actions.push(Action::Paste(pane.id));
                            if let Some(inserted_string) = pane.cursor_context.paste(&clipboard, &mut pane.text_buffer) {
                                if let Some(Cursor(cursor_line, cursor_column)) = pane.cursor_context.cursor {
                                    pane.transaction_manager.add_action(EditAction::Insert((cursor_line, cursor_column), inserted_string));
                                    pane.transaction_manager.next_transaction();
                                }
                            }
                        }
                    },

                    (Keycode::O, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            actions.push(Action::OpenFile(pane.id));
                            let text = native::open_file_dialog();
                            if let Some(text) = text {
                                pane.text_buffer.set_contents(text.as_bytes());
                                pane.scroller.move_to_the_top();
                            }
                        }
                    }
                    (Keycode::A, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        if let Some(pane) = pane_manager.get_active_pane_mut() {
                            // This is super ugly, fix.
                            pane.cursor_context.set_selection(((0,0), (pane.text_buffer.line_count()-1, pane.text_buffer.line_length(pane.text_buffer.line_count()-1))));
                            actions.push(Action::SelectAll(pane.id));
                        }
                    }

                    _ => is_text_input = true
                }
            }
            Event::TextInput{text, ..} => {
                if let Some(pane) = pane_manager.get_active_pane_mut() {
                    actions.push(Action::TextInput(pane.id, text.to_string()));
                    if is_text_input && pane.editing_name {
                        pane.name.push_str(&text);
                    } else if is_text_input {
                        // TODO: Replace with actually deleting the selection.
                        pane.cursor_context.clear_selection();

                        let action = pane.cursor_context.handle_insert(text.as_bytes(), &mut pane.text_buffer);
                        pane.transaction_manager.add_action(action);
                    }
                }
            }

            // Need to make selection work
            // Which probably means changing cursor representation


            Event::MouseButtonDown { x, y, .. } if ctrl_is_pressed && alt_is_pressed && cmd_is_pressed => {
                if let Some(pane) = pane_manager.get_pane_at_mouse((x,y), bounds) {
                    actions.push(Action::DeletePane(pane.id));
                }
                pane_manager.delete_pane_at_mouse((x, y), bounds);
            }

            Event::MouseButtonDown { x, y, .. }  if ctrl_is_pressed && cmd_is_pressed  => {
                if let Some(pane) = pane_manager.get_pane_at_mouse_mut((x,y), bounds) {
                    pane.editing_name = true;
                    actions.push(Action::StartEditPaneName(pane.id));
                }
            }

            Event::MouseButtonDown { x, y, .. } if ctrl_is_pressed && alt_is_pressed => {
                if let Some(pane) = pane_manager.get_pane_at_mouse((x,y), bounds) {
                    actions.push(Action::StartResizePane(pane.id, (x, y)));
                } else {
                    actions.push(Action::StartCreatePane((x, y)));
                }
                let found = pane_manager.set_resize_start((x,y), bounds);
                if !found {
                    pane_manager.set_create_start((x,y));
                }
            }

            Event::MouseButtonDown { x, y, .. } if cmd_is_pressed && alt_is_pressed => {
                if let Some(i) = pane_manager.get_pane_index_at_mouse((x, y), bounds) {
                    let mut pane = pane_manager.panes[i].clone();
                    actions.push(Action::DuplicatePane(pane.id));
                    pane.position = (pane.position.0 + 20, pane.position.1 + 20);
                    pane_manager.panes.push(pane);

                }
            }

            Event::MouseButtonDown { x, y, .. } if ctrl_is_pressed => {
                if let Some(pane) = pane_manager.get_pane_at_mouse((x,y), bounds) {
                    actions.push(Action::StartMovePane(pane.id, (x, y)));
                } else {
                    actions.push(Action::StartCreatePane((x, y)));
                }
                let found = pane_manager.set_dragging_start((x, y), bounds);
                if !found {
                    pane_manager.set_create_start((x,y));
                }
            }

            Event::MouseButtonDown { x, y, .. } => {
                if let Some(pane) = pane_manager.get_pane_at_mouse((x,y), bounds) {
                    actions.push(Action::SetPaneActive(pane.id));
                }
                // These x,y coords are in global space, maybe want to make them pane relative
                pane_manager.set_active_from_click_coords((x, y), bounds);
                if let Some(pane) = pane_manager.get_active_pane_mut() {
                    actions.push(Action::MouseDown(pane.id, (x, y)));
                    if let Some(action) = pane.on_click((x, y), bounds) {
                        actions.push(action);
                    }
                    pane.transaction_manager.next_transaction();
                }

                if !(ctrl_is_pressed && cmd_is_pressed) {
                    // Really I want this to work even if this is pressed, just for all panes
                    // that are not at the mouse point
                    // But this is a temporary binding anyways.
                    for pane in pane_manager.panes.iter_mut() {
                        if pane.editing_name == true {
                            actions.push(Action::EndEditPaneName(pane.id));
                        }
                        pane.editing_name = false;
                    }
                }

            }

            Event::MouseMotion{x, y, .. } => {

                actions.push(Action::MoveMouse((x, y)));
                pane_manager.update_dragging_position((x, y));
                pane_manager.update_resize_size((x, y));
                pane_manager.update_create_pane((x, y));

                // TODO:
                // distinguish between active and scrolling.
                // Mouse over is enough for scrolling, but not for active.
                pane_manager.set_scroll_active_if_mouse_over((x, y), bounds);
                if let Some(pane) = pane_manager.get_pane_at_mouse((x,y), bounds) {
                    actions.push(Action::SetScrollPane(pane.id));
                }


                // Make a selection
                if let Some(pane) = pane_manager.get_active_pane_mut() {
                    let (x2, y2) = pane.adjust_position(x, y, bounds);


                    if let Some(Cursor(start_line, mut start_column)) = pane.cursor_context.mouse_down {
                        pane.cursor_context.move_cursor_from_screen_position(&pane.scroller, x2, y2, &pane.text_buffer, bounds);
                        // TODO: Get my int types correct!
                        if let Some(Cursor(line, mut column)) = pane.cursor_context.cursor {
                            let new_start_line = start_line.min(line);
                            let line = line.max(start_line);
                            if new_start_line != start_line || start_line == line && start_column > column {
                                let temp = start_column;
                                start_column = column;
                                column = temp as usize;
                            }

                            // ugly refactor
                            pane.cursor_context.set_selection(((new_start_line, start_column), (line, column)));

                        }
                    }
                }
            }

            Event::MouseButtonUp{x, y, ..} => {
                pane_manager.stop_dragging();
                actions.push(Action::EndMovePane((x, y)));
                pane_manager.stop_resizing();
                actions.push(Action::EndResizePane((x, y)));
                pane_manager.create_pane();
                actions.push(Action::EndCreatePane((x, y)));
                
                // Setting a selection
                if let Some(pane) = pane_manager.get_active_pane_mut() {
                    let (x, y) = pane.adjust_position(x, y, bounds);
                    if let Some(Cursor(start_line, mut start_column)) = pane.cursor_context.mouse_down {
                        pane.cursor_context.move_cursor_from_screen_position(&pane.scroller, x, y, &pane.text_buffer, bounds);
                        if pane.cursor_context.selection.is_some() {
                            if let Some(Cursor(line, mut column)) = pane.cursor_context.cursor {
                                let new_start_line = start_line.min(line);
                                let line = line.max(start_line);
                                if new_start_line != start_line || start_line == line && start_column > column {
                                    let temp = start_column;
                                    start_column = column;
                                    column = temp as usize;
                                }

                                pane.cursor_context.set_selection(((new_start_line, start_column), (line, column)));
                                // TODO: Set Cursor
                            }
                        }

                    }

                    pane.cursor_context.clear_mouse_down();
                }
            }
            // Continuous resize in sdl2 is a bit weird
            // Would need to watch events or something
            Event::Window {win_event: WindowEvent::Resized(width, height), ..} => {

                // for pane in pane_manager.panes.iter_mut() {
                //     pane.width = (width / 2) as usize;
                //     pane.height = height as usize;
                //     if pane.position.0 != 0 {
                //         pane.position.0 = pane.width;
                //     }
                // }
                pane_manager.window.resize(width, height);
                actions.push(Action::ResizeWindow(width, height));

            }

            Event::MouseWheel {x, y, direction , timestamp: _, .. } => {
                // mouse state does not update when not focused.
                // Need to fix that some how.
                // So that I can scroll both panes unfocused.
                // pane_manager.set_scroll_active_if_mouse_over((mouse_state.x(), mouse_state.y()));
                let direction_multiplier = match direction {
                    sdl2::mouse::MouseWheelDirection::Normal => 1,
                    sdl2::mouse::MouseWheelDirection::Flipped => -1,
                    sdl2::mouse::MouseWheelDirection::Unknown(x) => x as i32
                };
                if let Some(pane) = pane_manager.get_scroll_active_pane_mut() {
                    pane.scroller.scroll_x(pane.width, x * direction_multiplier.neg(), &mut pane.text_buffer, bounds);
                    pane.scroller.scroll_y(pane.height, y * direction_multiplier, &pane.text_buffer, bounds);
                    actions.push(Action::Scroll(pane.id));
                }

            }
            _ => {}
        }
        if let Some(pane) = pane_manager.get_active_pane_mut() {
            pane.cursor_context.fix_cursor(&pane.text_buffer);
        }
    }
    actions
}


pub fn handle_per_frame_actions(per_frame_actions: &mut Vec<PerFrameAction>, pane_manager: &mut PaneManager) {
    let mut per_frame_action_results = vec![];
    for (i, per_frame_action) in per_frame_actions.iter_mut().enumerate() {
        per_frame_action_results.push(
            handle_per_frame_action(i, pane_manager, per_frame_action)
        );
    }
    for per_frame_action_result in per_frame_action_results {
        match per_frame_action_result {
            PerFrameActionResult::RemoveAction(i) => {
                per_frame_actions.swap_remove(i);
            },
            PerFrameActionResult::Noop => {}
        }
    }
}



pub fn handle_per_frame_action(index: usize, pane_manager: &mut PaneManager, per_frame_action: &mut PerFrameAction) -> PerFrameActionResult {
    match per_frame_action {
        PerFrameAction::ReadCommand(output_pane_name, _child, non_blocking_reader) => {
            let output_pane = pane_manager.get_pane_by_name_mut(output_pane_name.clone());
            if output_pane.is_none() {
                return PerFrameActionResult::Noop
            }
            let output_pane = output_pane.unwrap();

            let max_attempts = 100;
            let mut i = 0;
            let mut buf = String::new();
            while !non_blocking_reader.is_eof() {
                if i > max_attempts {
                    break;
                }
                let length = non_blocking_reader.read_available_to_string(&mut buf).unwrap();
                if length == 0 {
                    break;
                }
                i += 1;
            }
            if buf.contains('\x0c') {
                output_pane.text_buffer.chars.clear();
                buf = buf[0..buf.chars().position(|x| x == '\x0c').unwrap()].to_string();
            }

            if !buf.is_empty(){
                // TODO: Have mode for if it should clear or have history
                // Or maybe just don't clear and always have scroll history
                // output_pane.text_buffer.chars.clear();
                output_pane.text_buffer.chars.extend(buf.as_bytes().to_vec());
                output_pane.text_buffer.parse_lines();

            }
            if non_blocking_reader.is_eof() {
                PerFrameActionResult::RemoveAction(index)
            } else {
                PerFrameActionResult::Noop
            }
        }
        PerFrameAction::DisplayError(_pane_name, _error_text) => {
            PerFrameActionResult::Noop
        }
    }
}

pub fn handle_side_effects(pane_manager: &mut PaneManager, side_effects: Vec<Action>, per_frame_actions: &mut Vec<PerFrameAction>) {
    for side_effect in side_effects.iter() {
        match side_effect {
            Action::RunPane(pane_id) => {
                // TODO: think about multiple panes of same name
                let pane_contents = {
                    if let Some(pane) = pane_manager.get_pane_by_id_mut(*pane_id) {
                        Some((from_utf8(&pane.text_buffer.chars).unwrap().to_string(), pane.name.to_string()))
                    } else {
                        None
                    }
                };

                if let Some((pane_contents, pane_name)) = pane_contents {
                    let pane_name = &pane_name;
                    let output_pane_name = format!("{}-output", pane_name);
                    let output_pane = pane_manager.get_pane_by_name_mut(output_pane_name.clone());
                    match output_pane {

                        None => {
                            let existing_pane = pane_manager.get_pane_by_name(pane_name);
                            let position = {
                                match existing_pane {
                                    Some(pane) =>  {
                                        (pane.position.0 + pane.width as i32 + 10, pane.position.1)
                                    }
                                    None => (0, 0)
                                }
                            };
                            pane_manager.create_pane_raw(output_pane_name.to_string(), position, 300, 300);
                        }
                        Some(output_pane) => {
                            // This causes a flash to happen
                            // Which is actually useful from a user experience perspective
                            // but it was unintentional.
                            // Makes me think something is taking longer to render
                            // than I thought.
                            // I guess it makes sense in some ways though.
                            // ls for example takes some amount of time,
                            // and then I have to fetch that data and render.
                            output_pane.text_buffer.chars.clear();
                            output_pane.text_buffer.parse_lines();
                        }
                    }

                    let current_running_action = per_frame_actions.iter().enumerate().find(|(_i, x)|
                        if let PerFrameAction::ReadCommand(name, _, _) = x {
                            *name == output_pane_name
                        } else {
                            false
                        }
                    );
                    if let Some((i, _)) = current_running_action {
                        let action = per_frame_actions.swap_remove(i);
                        if let PerFrameAction::ReadCommand(_name, mut child, _) = action {
                            // TODO: Get rid of this unwrap!

                            child.kill().unwrap();
                        }
                    }

                    let command = pane_contents;
                    // need to handle error
                    let child = Command::new("bash")
                        .arg("-c")
                        .arg(command)
                        .stdout(Stdio::piped())
                        .spawn();

                    match child {
                        Ok(mut child) => {
                            let stdout = child.stdout.take().unwrap();
                            let noblock_stdout = NonBlockingReader::from_fd(stdout).unwrap();
                            per_frame_actions.push(PerFrameAction::ReadCommand(output_pane_name, child, noblock_stdout))
                        }
                        Err(e) => {
                            per_frame_actions.push(PerFrameAction::DisplayError(output_pane_name, format!("error {:?}", e)))
                        }
                    }
                }
            },
            _ => {}
        }
    }
}
