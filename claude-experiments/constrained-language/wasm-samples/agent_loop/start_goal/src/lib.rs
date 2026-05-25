wit_bindgen::generate!({
    world: "start-goal",
    path: "wit",
});

struct Component;

impl Guest for Component {
    fn handle(event: GoalReceivedPayload) {
        let goal_id = event.goal_id;
        let prompt = event.prompt;

        let messages = vec![Message {
            role: "user".to_string(),
            text: prompt.clone(),
        }];

        let goal = GoalRecord {
            id: goal_id.clone(),
            prompt: prompt.clone(),
            status: "thinking".to_string(),
            messages: messages.clone(),
            step_count: 0,
        };
        put_goals(&goal_id, &goal);

        let emit_id = emit_call_llm(&LlmReq {
            goal_id: goal_id.clone(),
            messages,
        });
        put_pending_llm(emit_id, &goal_id);
    }
}

export!(Component);
