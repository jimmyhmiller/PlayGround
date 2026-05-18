wit_bindgen::generate!({
    world: "on-llm-returned",
    path: "wit",
});

struct Component;

impl Guest for Component {
    fn handle(event: LlmReturnedPayload) {
        let emit_id = event.emit_id;

        // Find which goal this LLM call belonged to.
        let goal_id = get_pending_llm(emit_id).expect("emit_id registered in pending_llm");
        delete_pending_llm(emit_id);

        // Pull the goal record out of the goals map. We can only `list-goals`
        // (footprint declares `goals[*]`, not a specific key).
        let mut goal = list_goals()
            .into_iter()
            .find_map(|(k, v)| if k == goal_id { Some(v) } else { None })
            .expect("goal exists for this goal_id");

        match event.outcome {
            LlmOutcome::Failed(_reason) => {
                goal.status = "failed".to_string();
                put_goals(&goal_id, &goal);
            }
            LlmOutcome::Ok(response) => {
                goal.messages.push(Message {
                    role: "assistant".to_string(),
                    text: response.text.clone(),
                });

                match response.kind.as_str() {
                    "final" => {
                        goal.status = "complete".to_string();
                        let final_text = response.text.clone();
                        put_goals(&goal_id, &goal);
                        emit_notify_user(&Notification {
                            goal_id: goal_id.clone(),
                            text: final_text,
                        });
                    }
                    "needs_tool" => {
                        goal.status = "tool_calling".to_string();
                        put_goals(&goal_id, &goal);
                        let req_id = emit_execute_tool(&ToolReq {
                            goal_id: goal_id.clone(),
                            tool: response.tool,
                            args: response.args,
                        });
                        put_pending_tool(req_id, &goal_id);
                    }
                    _ => panic!("LLM returned unknown kind: {}", response.kind),
                }
            }
        }
    }
}

export!(Component);
