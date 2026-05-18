wit_bindgen::generate!({
    world: "on-tool-returned",
    path: "wit",
});

struct Component;

impl Guest for Component {
    fn handle(event: ToolReturnedPayload) {
        let emit_id = event.emit_id;
        let goal_id = get_pending_tool(emit_id).expect("emit_id registered in pending_tool");
        delete_pending_tool(emit_id);

        let mut goal = list_goals()
            .into_iter()
            .find_map(|(k, v)| if k == goal_id { Some(v) } else { None })
            .expect("goal exists for this goal_id");

        let result_text = match event.outcome {
            ToolOutcome::Ok(resp) => resp.result,
            ToolOutcome::Failed(reason) => format!("(tool error: {reason})"),
        };

        goal.messages.push(Message {
            role: "tool".to_string(),
            text: result_text,
        });
        goal.status = "thinking".to_string();
        let messages = goal.messages.clone();
        put_goals(&goal_id, &goal);

        let req_id = emit_call_llm(&LlmReq {
            goal_id: goal_id.clone(),
            messages,
        });
        put_pending_llm(req_id, &goal_id);
    }
}

export!(Component);
