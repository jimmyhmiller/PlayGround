use std::{process::{Child, ChildStdout, Command, Stdio}, ops::Neg, str::from_utf8, io::Write};

use nonblock::NonBlockingReader;
use sdl2::{clipboard::ClipboardUtil, keyboard::{Scancode, Keycode, Mod}, event::{Event, WindowEvent}};

use crate::{pane_manager::PaneManager, PaneSelector, renderer::EditorBounds, cursor::{Cursor, CursorContext}, transaction::EditAction, native};



#[derive(Debug, Clone)]
pub enum Action {
    MoveCursorUp(PaneSelector),
    MoveCursorDown(PaneSelector),
    MoveCursorLeft(PaneSelector),
    MoveCursorRight(PaneSelector),
    Delete(PaneSelector),
    DeletePaneName(PaneSelector),
    DeleteSelection(PaneSelector),
    DeleteChar(PaneSelector),
    RunPane(PaneSelector),
    Enter(PaneSelector),
    InsertNewline(PaneSelector),
    TextInput(PaneSelector, String),
    Indent(PaneSelector),
    DeIndent(PaneSelector),
    Undo(PaneSelector),
    Redo(PaneSelector),
    MoveCursorToLineStart(PaneSelector),
    MoveCursorToLineEnd(PaneSelector),
    Copy(PaneSelector),
    Paste(PaneSelector),
    OpenFile(PaneSelector),
    SelectAll(PaneSelector),
    DeletePane(PaneSelector),
    StartEditPaneName(PaneSelector),
    EndEditPaneName(PaneSelector),
    StartResizePane(PaneSelector, (i32, i32)),
    EndResizePane((i32, i32)),
    StartCreatePane((i32, i32)),
    EndCreatePane((i32, i32)),
    CreatePane(String, (i32, i32), usize, usize),
    StartMovePane(PaneSelector, (i32, i32)),
    EndMovePane((i32, i32)),
    DuplicatePane(PaneSelector),
    SetPaneActive(PaneSelector),
    SetScrollPane(PaneSelector),
    MouseDown(PaneSelector, (i32, i32)),
    CtrlMouseDown(PaneSelector, (i32, i32)),
    CtrlAltMouseDown(PaneSelector, (i32, i32)),
    MouseUp(PaneSelector, (i32, i32)),
    MoveMouse((i32, i32)),
    ResizeWindow(i32, i32),
    Scroll(PaneSelector, (i32, i32)),
    ClearPane(PaneSelector),
    PaneContentChanged(PaneSelector),
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

fn stop_pane_name_edits(pane_manager: &mut PaneManager, actions: &mut Vec<Action>) {
    // This is really not great. I shouldn't
    // have to do this all the time.
    for pane in pane_manager.panes.iter_mut() {
        if pane.editing_name() {
            actions.push(Action::EndEditPaneName(PaneSelector::Id(pane.id())));
        }
    }
}



impl Action {
    pub fn pane_id(&self, pane_manager: &PaneManager, editor_bounds: &EditorBounds) -> Option<usize> {

        match self.pane_selector() {
            None => None,
            Some(pane_selector) => {
                match pane_selector {
                    PaneSelector::Active => pane_manager.get_active_pane().map(|pane| pane.id()),
                    PaneSelector::Id(id) => Some(id),
                    PaneSelector::AtMouse(mouse_pos) => pane_manager.get_pane_at_mouse(mouse_pos, editor_bounds).map(|pane| pane.id()),
                    PaneSelector::Scroll => pane_manager.get_scroll_active_pane().map(|pane| pane.id()),
                }
            }
           
        }
    }

    pub fn pane_selector(&self) -> Option<PaneSelector> {
        match self {
            Action::MoveCursorUp(pane_selector) |
            Action::MoveCursorDown(pane_selector) |
            Action::MoveCursorLeft(pane_selector) |
            Action::MoveCursorRight(pane_selector) |
            Action::Delete(pane_selector) |
            Action::RunPane(pane_selector) |
            Action::InsertNewline(pane_selector) |
            Action::Undo(pane_selector) |
            Action::Redo(pane_selector) |
            Action::MoveCursorToLineStart(pane_selector) |
            Action::MoveCursorToLineEnd(pane_selector) |
            Action::Copy(pane_selector) |
            Action::Paste(pane_selector) |
            Action::OpenFile(pane_selector) |
            Action::SelectAll(pane_selector) |
            Action::TextInput(pane_selector, _) |
            Action::DeletePane(pane_selector) |
            Action::StartEditPaneName(pane_selector) |
            Action::EndEditPaneName(pane_selector) |
            Action::StartResizePane(pane_selector, _) |
            Action::StartMovePane(pane_selector, _) |
            Action::DuplicatePane(pane_selector) |
            Action::SetPaneActive(pane_selector) |
            Action::SetScrollPane(pane_selector) |
            Action::MouseDown(pane_selector, _) |
            Action::Scroll(pane_selector, _) |
            Action::DeletePaneName(pane_selector) |
            Action::DeleteSelection(pane_selector) |
            Action::DeleteChar(pane_selector) |
            Action::Enter(pane_selector) |
            Action::CtrlMouseDown(pane_selector, _) |
            Action::CtrlAltMouseDown(pane_selector, _) |
            Action::Indent(pane_selector) |
            Action::DeIndent(pane_selector) |
            Action::ClearPane(pane_selector) |
            Action::PaneContentChanged(pane_selector) => {
                Some(*pane_selector)
            }

            Action::EndResizePane(_) |
            Action::StartCreatePane(_) |
            Action::EndCreatePane(_) |
            Action::EndMovePane(_) |
            Action::MouseUp(_, _) |
            Action::MoveMouse(_) |
            Action::ResizeWindow(_, _) |
            Action::CreatePane(_, _, _, _) |
            Action::Quit => {
                None
            }
            
        }
    }

    pub fn process<'a>(&mut self, pane_manager: &mut PaneManager, bounds: &EditorBounds, clipboard: &ClipboardUtil, actions: &'a mut Vec<Action>) -> Option<()> {


        // TODO:
        // If the pane does not resolve,
        // should I change the selector?
        // or just leave it as is?

        match self {

            Action::MoveCursorUp(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                // TODO:
                // Work with other pane types
                let pane = pane.get_text_pane_mut()?;
                pane.cursor_context.move_up(&pane.text_buffer);
                                // ? is fine here only because there is nothing 
                // below this that I want to do that doesn't rely on cursor
                // If that changes, remove the ?
                pane.scroller.move_up(pane.cursor_context.cursor?, bounds)
            }
            Action::MoveCursorDown(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let pane = pane.get_text_pane_mut()?;
                pane.cursor_context.move_down(&pane.text_buffer);
                // ? is fine here only because there is nothing 
                // below this that I want to do that doesn't rely on cursor
                // If that changes, remove the ?
                pane.scroller.move_down(pane.cursor_context.cursor?, pane.height, bounds)
            },
            Action::MoveCursorLeft(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let pane = pane.get_text_pane_mut()?;
                pane.cursor_context.move_left(&pane.text_buffer);
            },
            Action::MoveCursorRight(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let pane = pane.get_text_pane_mut()?;
                pane.cursor_context.move_right(&pane.text_buffer);
            },
            Action::Delete(pane_selector) => {
                // TODO:
                // Do better
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                // TODO: editing name works without text pane
                // but I'm just shortcutting for now.

                let text_pane = pane.get_text_pane()?;
                
                if pane.editing_name() {
                    actions.push(Action::DeletePaneName(pane_selector.clone()));
                }
                else if text_pane.cursor_context.selection.is_some() {
                    actions.push(Action::DeleteSelection(pane_selector.clone()));
                }
                else {
                    actions.push(Action::DeleteChar(pane_selector.clone()));
                }
            },
            Action::DeletePaneName(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                pane.name().pop();
            }
            Action::DeleteSelection(pane_selector) => {
                // TODO: Make this better
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                
                let pane = pane.get_text_pane_mut()?;

                let current_selection = pane.cursor_context.selection?;
                let (start, end) = current_selection;
                let (start_line, start_column) = start;
                let (end_line, end_column) = end;
                
                let (line_start, _line_end) = pane.text_buffer.get_line(start_line as usize)?;
                let char_start_pos = line_start + start_column as usize;
                let (end_line_start, _line_end) = pane.text_buffer.get_line(end_line as usize)?;
                let char_end_pos = end_line_start + end_column as usize;
                let text: Vec<u8> = pane.text_buffer.chars.drain(char_start_pos as usize..char_end_pos as usize).collect();
                pane.transaction_manager.add_action(EditAction::Delete((end_line, end_column), from_utf8(&text).unwrap().to_string()));
                pane.transaction_manager.add_action(EditAction::CursorPosition(Cursor(start_line, start_column)));

                pane.transaction_manager.next_transaction();
                pane.text_buffer.parse_lines();


                // TODO:
                // Thinking about how selections fit into transactions
                // I guess I should restore them?
                pane.cursor_context.clear_selection();
                pane.cursor_context.set_cursor(Cursor(start_line, start_column));
                // pane.cursor_context.fix_cursor(&pane.text_buffer);

            }
            Action::DeleteChar(pane_selector) => {
                // I probably want to deal with transactions orthogonally.
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());


                let pane = pane.get_text_pane_mut()?;
                let current_cursor = pane.cursor_context.cursor?;
                let mut old_cursor = current_cursor;
                // Character is to the right of the cursor
                let cursor_before = current_cursor.to_the_left(&pane.text_buffer);

                if let Some(left_char) = pane.text_buffer.byte_at_pos(cursor_before) {
                    if let Some(right_char) = pane.text_buffer.byte_at_pos(current_cursor) {
                        if CursorContext::is_open_bracket(&[*left_char]) && CursorContext::is_close_bracket(&[*right_char])  {
                            let next_char_position = current_cursor.to_the_right(&pane.text_buffer);
                            let action = pane.text_buffer.remove_char(next_char_position);

                            pane.transaction_manager.next_transaction();
                            pane.transaction_manager.add_action(EditAction::CursorPosition(current_cursor));
                            pane.transaction_manager.add_action(action);
                        }
                    }
                }

                // We do this move_left first, because otherwise we might end up at the end
                // of the new line we formed from the deletion, rather than the old end of the line.
                let cursor_action = old_cursor.move_left(&pane.text_buffer);
                let action = pane.text_buffer.remove_char(current_cursor);


                pane.transaction_manager.add_action(action);
                pane.transaction_manager.add_action(cursor_action);
                

                pane.cursor_context.set_cursor(old_cursor);
            }
            Action::RunPane(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                
                
                // Should I do this here rather than side-effects?
                // How do I communicate that things need to keep
                // being done? Do I pass in per frame actions?
                // I mean that might make sense
                // Honestly not sure.

                // Also should make an action for updating the output pane
                // Really should capture that in detail, but even just a placeholder
                // would be a good idea.

            },
            Action::ClearPane(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let pane = pane.get_text_pane_mut()?;
                pane.text_buffer.chars.clear();
                pane.text_buffer.parse_lines();
            }
            Action::PaneContentChanged(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
            }
            Action::Enter(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let pane = pane.get_text_pane()?;
                if pane.editing_name {
                    actions.push(Action::EndEditPaneName(PaneSelector::Active));
                } else {
                    actions.push(Action::InsertNewline(PaneSelector::Active));
                }
            }
            Action::InsertNewline(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let pane = pane.get_text_pane_mut()?;
                let action = pane.cursor_context.handle_insert(&[b'\n'], &mut pane.text_buffer);
                pane.transaction_manager.add_action(action);
                pane.cursor_context.start_of_line();
            },
            Action::Indent(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());


                let pane = pane.get_text_pane_mut()?;
                // TODO: Deal with tab when selection exists
                // should add spaces to each line.
                if let Some(((start_line, _), (end_line, _))) =  pane.cursor_context.selection {
                    let current_cursor = pane.cursor_context.cursor;
                    for line in start_line..end_line + 1 {
                        pane.cursor_context.cursor = Some(Cursor(line, 0));
                        let action = pane.cursor_context.handle_insert("    ".as_bytes(), &mut pane.text_buffer);
                        pane.transaction_manager.add_action(action);
                    }
                    pane.cursor_context.cursor = current_cursor;
                } else {
                    let action = pane.cursor_context.handle_insert("    ".as_bytes(), &mut pane.text_buffer);
                    pane.transaction_manager.add_action(action);
                }
            }
            Action::DeIndent(_) => {
                // TODO: Implement
            }
            Action::TextInput(pane_selector, text) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());

                let pane = pane.get_text_pane_mut()?;
 
                if pane.editing_name {
                    pane.name.push_str(&text);
                } else {
                    // TODO: Replace with actually deleting the selection.
                    pane.cursor_context.clear_selection();
                    let action = pane.cursor_context.handle_insert(text.as_bytes(), &mut pane.text_buffer);
                    pane.transaction_manager.add_action(action);
                }
            },
            Action::Undo(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let pane = pane.get_text_pane_mut()?;
                pane.transaction_manager.undo(&mut pane.cursor_context, &mut pane.text_buffer);
            },
            Action::Redo(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let pane = pane.get_text_pane_mut()?;
                pane.transaction_manager.redo(&mut pane.cursor_context, &mut pane.text_buffer);
            },
            Action::MoveCursorToLineStart(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                 let pane = pane.get_text_pane_mut()?;
                pane.cursor_context.start_of_line();
            },
            Action::MoveCursorToLineEnd(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let pane = pane.get_text_pane_mut()?;
                pane.cursor_context.end_of_line(&pane.text_buffer);
            },
            Action::Copy(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let pane = pane.get_text_pane()?;
                pane.cursor_context.copy_selection(&clipboard, &pane.text_buffer);
            },
            Action::Paste(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let pane = pane.get_text_pane_mut()?;
                let Cursor(cursor_line, cursor_column) = pane.cursor_context.cursor?;
                let inserted_string = pane.cursor_context.paste(&clipboard, &mut pane.text_buffer)?;
                pane.transaction_manager.add_action(EditAction::Insert((cursor_line, cursor_column), inserted_string));
                let cursor = pane.cursor_context.cursor?;
                pane.transaction_manager.add_action(EditAction::CursorPosition(cursor));
                pane.transaction_manager.next_transaction();
            },
            Action::OpenFile(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let pane = pane.get_text_pane_mut()?;
                let text = native::open_file_dialog()?;
                pane.text_buffer.set_contents(text.as_bytes());
                pane.scroller.move_to_the_top();
            },
            Action::SelectAll(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let pane = pane.get_text_pane_mut()?;
                // This is super ugly, fix.
                pane.cursor_context.set_selection(((0,0), (pane.text_buffer.line_count()-1, pane.text_buffer.line_length(pane.text_buffer.line_count()-1))));
            },
            Action::DeletePane(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let id = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?.id();
                pane_manager.delete_pane(id);
            },
            Action::StartEditPaneName(pane_selector) => {
                stop_pane_name_edits(pane_manager, actions);
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let pane = pane.get_text_pane_mut()?;
                pane.editing_name = true;
                actions.push(Action::SetPaneActive(pane_selector.clone()));
            },
            Action::EndEditPaneName(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let pane = pane.get_text_pane_mut()?;
                pane.editing_name = false;
            },
            Action::CtrlMouseDown(pane_selector, mouse_pos) => {
                if let Some(pane) = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds) {
                    *pane_selector = PaneSelector::Id(pane.id());
                    actions.push(Action::StartMovePane(PaneSelector::AtMouse(*mouse_pos), *mouse_pos));
                } else {
                    actions.push(Action::StartCreatePane(*mouse_pos));
                }
            }
            Action::StartResizePane(pane_selector, mouse_pos) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let id = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?.id();
                pane_manager.set_resize_start(*mouse_pos, id);
            },
            Action::EndResizePane(_mouse_pos) => {
                pane_manager.stop_resizing();
            },
            Action::StartCreatePane(mouse_pos) => {
                pane_manager.set_create_start(*mouse_pos);
            },
            Action::EndCreatePane(_mouse_pos) => {
                pane_manager.create_pane();
                // Should I somehow signal that the pane was created?
            },
            Action::CreatePane(pane_name, position, width, height) => {
                pane_manager.create_pane_raw(pane_name.to_string(), *position, *width, *height);
            }
            Action::CtrlAltMouseDown(pane_selector, mouse_pos) => {
                if let Some(pane) = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds) {
                    *pane_selector = PaneSelector::Id(pane.id());
                    actions.push(Action::StartResizePane(PaneSelector::AtMouse(*mouse_pos), *mouse_pos));
                } else {
                    actions.push(Action::StartCreatePane(*mouse_pos));
                }
            },
            Action::StartMovePane(pane_selector, mouse_pos) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                pane_manager.set_dragging_start(*mouse_pos, bounds);
            },
            Action::EndMovePane(_mouse_pos) => {
                // TODO:
                // Do I need to capture the pane here?
                pane_manager.stop_dragging();
            },
            Action::DuplicatePane(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let i = pane_manager.get_pane_index_by_id(pane.id())?;
                let mut pane = pane_manager.panes[i].clone();
                pane.set_position(pane.position().0 + 20, pane.position().1 + 20);
                pane.set_id(pane_manager.new_pane_id());
                pane_manager.panes.push(pane);
            },
            Action::SetPaneActive(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let id = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?.id();
                pane_manager.set_active_by_id(id);    
            },
            Action::SetScrollPane(pane_selector) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let id = pane_manager.get_pane_by_selector(&pane_selector, bounds)?.id();
                pane_manager.set_scroll_active_by_id(id);
            },
            Action::MouseDown(pane_selector, mouse_pos) => {
                stop_pane_name_edits(pane_manager, actions);
                let mouse_pane_id = pane_manager.get_pane_at_mouse(*mouse_pos, bounds).map(|p| p.id());
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                // I kind of want to capture these things as actions. But how to do that?
                // Maybe this function actually just returns more actions
                // that are queued up immediately?
                let pane = pane.get_text_pane_mut()?;
                if Some(pane.id) == mouse_pane_id {
                    if pane.mouse_over_play_button(*mouse_pos, bounds) {
                        // run pane
                    } else {
                        // We could generate actions here instead of doing things.
                        let (x, y) = pane.adjust_position(mouse_pos.0, mouse_pos.1, bounds);
                        
                        pane.cursor_context.move_cursor_from_screen_position(&pane.scroller, x, y, &pane.text_buffer, bounds);
                        // If I move the cursor, I should make a new transaction
                        pane.transaction_manager.next_transaction();
                        pane.cursor_context.mouse_down();
                        pane.cursor_context.clear_selection();
                    }
                }


                if mouse_pane_id.is_none() {
                    pane_manager.clear_active();
                }
            },
            Action::MouseUp(pane_selector, mouse_pos) => {
                if let Some(pane) = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds) {
                    *pane_selector = PaneSelector::Id(pane.id());
                }
                

                if pane_manager.dragging_pane.is_some() {
                    actions.push(Action::EndMovePane(*mouse_pos));
                }
                
                if pane_manager.resize_pane.is_some() {
                    actions.push(Action::EndResizePane(*mouse_pos));
                }

                if pane_manager.create_pane_activated {
                    actions.push(Action::EndCreatePane(*mouse_pos));
                }

                // I have to do this because otherwise
                // it returns none for the whole function
                // I hate this hack and really wish
                // rust would implemnent options in blocks
                // The other option would be to pass in
                // actions as a mutable vector
                // Which might be better anyways
                (|| {
                    let pane = pane_manager.get_active_pane_mut()?;
                    let (x, y) = *mouse_pos;
                    let (x, y) = pane.adjust_position(x, y, bounds);
                    let pane = pane.get_text_pane_mut()?;
                    let Cursor(start_line, mut start_column) = pane.cursor_context.mouse_down?;
                    pane.cursor_context.move_cursor_from_screen_position(&pane.scroller, x, y, &pane.text_buffer, bounds);
                    if pane.cursor_context.selection.is_some() {
                        let Cursor(line, mut column) = pane.cursor_context.cursor?;
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

                    pane.cursor_context.clear_mouse_down();
                    Some(())
                })();
            }
            Action::MoveMouse(mouse_pos) => {
                pane_manager.update_dragging_position(*mouse_pos);
                pane_manager.update_resize_size(*mouse_pos);
                pane_manager.update_create_pane(*mouse_pos);
                let (x, y) = *mouse_pos;

                let pane = pane_manager.get_active_pane_mut()?;

                let (x2, y2) = pane.adjust_position(x, y, bounds);
                let pane = pane.get_text_pane_mut()?;
                let Cursor(start_line, mut start_column) = pane.cursor_context.mouse_down?;
                pane.cursor_context.move_cursor_from_screen_position(&pane.scroller, x2, y2, &pane.text_buffer, bounds);
                // TODO: Get my int types correct!
                let Cursor(line, mut column) = pane.cursor_context.cursor?;
                let new_start_line = start_line.min(line);
                let line = line.max(start_line);
                if new_start_line != start_line || start_line == line && start_column > column {
                    let temp = start_column;
                    start_column = column;
                    column = temp as usize;
                }

                // TODO:
                // This should almost certainly be a separate action
                if new_start_line != line || start_column != column {
                    pane.cursor_context.set_selection(((new_start_line, start_column), (line, column)));
                }


            },
            Action::ResizeWindow(width, height) => {
                pane_manager.window.resize(*width, *height);
            },
            Action::Scroll(pane_selector, (x_scroll, y_scroll)) => {
                let pane = pane_manager.get_pane_by_selector_mut(&pane_selector, bounds)?;
                *pane_selector = PaneSelector::Id(pane.id());
                let pane = pane.get_text_pane_mut()?;
                pane.scroller.scroll_x(pane.width, *x_scroll, &mut pane.text_buffer, bounds);
                pane.scroller.scroll_y(pane.height, *y_scroll, &pane.text_buffer, bounds);

            },
            Action::Quit => {
                ::std::process::exit(0);
            },
        }

        Some(())
    }

    pub fn handle_side_effect(&self, pane_manager: &mut PaneManager, bounds: &EditorBounds, per_frame_actions: &mut Vec<PerFrameAction>, actions: &mut Vec<Action>) -> Option<()> {
        match self {
            Action::RunPane(pane_selector) => {
                // TODO: think about multiple panes of same name
                let (pane_contents, pane_name) = {
                    let pane_id = pane_manager.get_pane_by_selector_mut(pane_selector, bounds).map(|p| p.id())?;
                    let pane = pane_manager.get_pane_by_id_mut(pane_id)?;
                    let pane = pane.get_text_pane()?;
                    (from_utf8(&pane.text_buffer.chars).unwrap().to_string(), &pane.name.to_string()) 
                };
    
                let output_pane_name = format!("{}-output", pane_name);
                let output_pane = pane_manager.get_pane_by_name_mut(output_pane_name.clone());
                match output_pane {
    
                    None => {
                        let existing_pane = pane_manager.get_pane_by_name(pane_name);
                        let position = {
                            match existing_pane {
                                Some(pane) =>  {
                                    let pane = pane.get_text_pane()?;
                                    (pane.position.0 + pane.width as i32 + 10, pane.position.1)
                                }
                                None => (0, 0)
                            }
                        };
                        actions.push(Action::CreatePane(output_pane_name.to_string(), position, 300, 300));
                        // pane_manager.create_pane_raw(output_pane_name.to_string(), position, 300, 300);
                    }

                    // I need to not edit these, but instead make new actions
                    Some(output_pane) => {
                        let output_pane_id = output_pane.id();
                        actions.push(Action::ClearPane(PaneSelector::Id(output_pane_id)));
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
                    if let PerFrameAction::ReadCommand(_pane_name, mut child, _) = action {
                        // TODO: Get rid of this unwrap!
                        child.kill().unwrap();
                    }
                }
                
    
                let command = pane_contents;
    
                let mut has_stdin = false;
    
                let child = if command.starts_with("#!") {
                    has_stdin = true;
                    let mut lines = command.lines();
                    let mut command_name = lines.nth(0).unwrap().trim_start_matches("#!").trim();
                    let mut args = vec![];
                    if command_name.starts_with("/usr/bin/env") {
                        args = command_name.split_whitespace().skip(1).map(|x| x.to_string()).collect();
                        command_name = "/usr/bin/env";
                    }
                    let rest_lines = lines.collect::<Vec<&str>>().join("\n");
    
                    let child = Command::new(command_name)
                        .args(args)
                        .stdout(Stdio::piped())
                        .stdin(Stdio::piped())
                        .spawn();
    
                    if let Ok(mut child) = child {
                        let child_stdin = child.stdin.as_mut().unwrap();
                        child_stdin.write_all(rest_lines.as_bytes()).unwrap();
                        child_stdin.flush().unwrap();
                        Ok(child)
                    } else {
                        child
                    }
    
                } else {
                    Command::new("bash")
                    .arg("-c")
                    .arg(command)
                    .stdout(Stdio::piped())
                    .spawn()
                };
    
    
                match child {
                // need to handle error
                    Ok(mut child) => {
                        let stdout = child.stdout.take().unwrap();
                        // got crash here when trying to run nothing
                        if has_stdin {
                            let stdin = child.stdin.take().unwrap();
                            drop(stdin);
                        }
                        let noblock_stdout = NonBlockingReader::from_fd(stdout).unwrap();
                        per_frame_actions.push(PerFrameAction::ReadCommand(output_pane_name, child, noblock_stdout))
                    }
                    Err(e) => {
                        per_frame_actions.push(PerFrameAction::DisplayError(output_pane_name, format!("error {:?}", e)))
                    }
                }
            },
            _ => {}
        }
        Some(())
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



// This should return two things
// 1. A resolution of all panes
// 2. A list of actions that need to be performed

pub fn handle_events(event_pump: &mut sdl2::EventPump) -> Vec<Action> {
    let mut is_text_input = false;

    // Consider moving this allocation
    let mut actions: Vec<Action> = vec![];

    let ctrl_is_pressed = event_pump.keyboard_state().is_scancode_pressed(Scancode::LCtrl);
    let alt_is_pressed = event_pump.keyboard_state().is_scancode_pressed(Scancode::LAlt);
    let cmd_is_pressed = event_pump.keyboard_state().is_scancode_pressed(Scancode::LGui);

    for event in event_pump.poll_iter() {
        // println!("frame: {}, event {:?}", frame_counter, event);
        match event {
            Event::Quit { .. } => {
                actions.push(Action::Quit);
            },

            Event::KeyDown { keycode, keymod, .. } => {
                // I am assuming the keycode is some
                // I had this matches, but that doesn't do anything.
                // matches!(keycode, Some(pane_id));
                match (keycode.unwrap(), keymod) {
                    (Keycode::Up, _) => {
                        actions.push(Action::MoveCursorUp(PaneSelector::Active));
                    },
                    (Keycode::Down, _) => {
                        actions.push(Action::MoveCursorDown(PaneSelector::Active));                    
                    },
                    (Keycode::Left, _) => {
                        actions.push(Action::MoveCursorLeft(PaneSelector::Active));                   
                    },
                    (Keycode::Right, _) => {
                        actions.push(Action::MoveCursorRight(PaneSelector::Active));                    
                    },
                    (Keycode::Backspace, _) => {
                        actions.push(Action::Delete(PaneSelector::Active));
                    }
                    (Keycode::Return, _) if cmd_is_pressed => {
                        actions.push(Action::RunPane(PaneSelector::Active));                   
                    }
                    (Keycode::Return, _) => {
                        actions.push(Action::Enter(PaneSelector::Active));
                    },
                    (Keycode::Tab, Mod::LSHIFTMOD) => {
                        actions.push(Action::DeIndent(PaneSelector::Active));
                    },
                    (Keycode::Tab, _) => {
                        actions.push(Action::Indent(PaneSelector::Active));
                    }
                    (Keycode::Z, key_mod) if key_mod == Mod::LGUIMOD || keymod == Mod::RGUIMOD => {
                        actions.push(Action::Undo(PaneSelector::Active));                   
                    }
                    (Keycode::Z, key_mod) if key_mod == (Mod::LSHIFTMOD | Mod::LGUIMOD) => {
                        actions.push(Action::Redo(PaneSelector::Active));                     
                    }
                    (Keycode::Z, _key_mod) => {
                        is_text_input = true
                    },
                    (Keycode::A, Mod::LCTRLMOD | Mod::RCTRLMOD) => {
                        actions.push(Action::MoveCursorToLineStart(PaneSelector::Active));                    
                    },
                    (Keycode::E, Mod::LCTRLMOD | Mod::RCTRLMOD) => {
                        actions.push(Action::MoveCursorToLineEnd(PaneSelector::Active));                    
                    },
                    (Keycode::C, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        actions.push(Action::Copy(PaneSelector::Active));                    
                    },
                    (Keycode::V, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        actions.push(Action::Paste(PaneSelector::Active));                    
                    },
                    (Keycode::O, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        actions.push(Action::OpenFile(PaneSelector::Active));                    
                    },
                    (Keycode::A, Mod::LGUIMOD | Mod::RGUIMOD) => {
                        actions.push(Action::SelectAll(PaneSelector::Active));                    
                    },
                    _ => is_text_input = true
                }
            }
            Event::TextInput{text, ..} => {
                if is_text_input {
                    actions.push(Action::TextInput(PaneSelector::Active, text.to_string()));
                }
            }
            Event::MouseButtonDown { x, y, .. } if ctrl_is_pressed && alt_is_pressed && cmd_is_pressed => {
                actions.push(Action::DeletePane(PaneSelector::AtMouse((x, y))));
            }
            Event::MouseButtonDown { x, y, .. }  if ctrl_is_pressed && cmd_is_pressed => {
                actions.push(Action::StartEditPaneName(PaneSelector::AtMouse((x, y))));
            }
            Event::MouseButtonDown { x, y, .. } if ctrl_is_pressed && alt_is_pressed => {
                actions.push(Action::CtrlAltMouseDown(PaneSelector::AtMouse((x, y)), (x, y)));
            }
            Event::MouseButtonDown { x, y, .. } if ctrl_is_pressed => {
                actions.push(Action::CtrlMouseDown(PaneSelector::AtMouse((x, y)), (x, y)));
            }
            Event::MouseButtonDown { x, y, .. } if cmd_is_pressed && alt_is_pressed => {
                actions.push(Action::DuplicatePane(PaneSelector::AtMouse((x, y))));
            }
            Event::MouseButtonDown { x, y, .. } => {
                actions.push(Action::SetPaneActive(PaneSelector::AtMouse((x, y))));
                actions.push(Action::MouseDown(PaneSelector::Active, (x, y)));
            }
            Event::MouseMotion{x, y, .. } => {
                actions.push(Action::MoveMouse((x, y)));
                actions.push(Action::SetScrollPane(PaneSelector::AtMouse((x, y))));
            }
            Event::MouseButtonUp{x, y, ..} => {
                actions.push(Action::MouseUp(PaneSelector::Active, (x, y)));
                
            }
            // Continuous resize in sdl2 is a bit weird
            // Would need to watch events or something
            Event::Window {win_event: WindowEvent::Resized(width, height), ..} => {
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
                let x_scroll = x * direction_multiplier.neg();
                let y_scroll = y * direction_multiplier;
                actions.push(Action::Scroll(PaneSelector::Scroll, (x_scroll, y_scroll)));

            }
            _ => {}
        }
    }
    actions
}


pub fn handle_per_frame_actions(per_frame_actions: &mut Vec<PerFrameAction>, pane_manager: &mut PaneManager, actions: &mut Vec<Action>) {
    let mut per_frame_action_results = vec![];
    for (i, per_frame_action) in per_frame_actions.iter_mut().enumerate() {
        per_frame_action_results.push(
            handle_per_frame_action(i, pane_manager, per_frame_action, actions)
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



pub fn handle_per_frame_action(index: usize, pane_manager: &mut PaneManager, per_frame_action: &mut PerFrameAction, actions: &mut Vec<Action>) -> PerFrameActionResult {
    match per_frame_action {
        PerFrameAction::ReadCommand(output_pane_name, _child, non_blocking_reader) => {
            let output_pane = pane_manager.get_pane_by_name_mut(output_pane_name.clone())
                .and_then(|pane| pane.get_text_pane_mut());
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

            actions.push(Action::PaneContentChanged(PaneSelector::Id(output_pane.id)));
            if buf.contains('\x0c') {
                buf = buf[buf.chars().position(|x| x == '\x0c').unwrap() + 1..].to_string();
                output_pane.text_buffer.chars = buf.as_bytes().to_vec();
                output_pane.text_buffer.parse_lines();
                return PerFrameActionResult::Noop
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