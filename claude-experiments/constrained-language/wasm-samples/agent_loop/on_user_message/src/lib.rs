wit_bindgen::generate!({
    world: "on-user-message",
    path: "wit",
});

struct Component;

impl Guest for Component {
    fn handle(event: UserMessagePayload) {
        let session_id = event.session_id;
        let text = event.text;

        // Find the existing session if any; if none, create it. This single
        // handler covers both "first turn of a new conversation" and
        // "follow-up turn of an existing conversation."
        let existing = list_goals()
            .into_iter()
            .find_map(|(k, v)| if k == session_id { Some(v) } else { None });

        let mut goal = match existing {
            Some(g) => g,
            None => GoalRecord {
                id: session_id.clone(),
                prompt: text.clone(),
                status: "thinking".to_string(),
                messages: Vec::new(),
                step_count: 0,
            },
        };

        goal.messages.push(Message {
            role: "user".to_string(),
            text,
        });
        goal.status = "thinking".to_string();
        let messages = goal.messages.clone();
        put_goals(&session_id, &goal);

        let req_id = emit_call_llm(&LlmReq {
            goal_id: session_id.clone(),
            messages,
        });
        put_pending_llm(req_id, &session_id);
    }
}

export!(Component);
