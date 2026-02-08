import { query, createSdkMcpServer } from "@anthropic-ai/claude-agent-sdk";
import * as readline from "node:readline";
import { allTools } from "./tools.js";

const server = createSdkMcpServer({
  name: "virtual",
  version: "1.0.0",
  tools: allTools,
});

const disallowedTools = [
  "Read",
  "Write",
  "Edit",
  "Bash",
  "BatchBash",
  "Glob",
  "Grep",
  "WebFetch",
  "WebSearch",
  "NotebookEdit",
  "Task",
  "TodoRead",
  "TodoWrite",
];

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function ask(prompt: string): Promise<string> {
  return new Promise((resolve) => rl.question(prompt, resolve));
}

console.log("Custom Tool Agent Chat");
console.log("Type a message to chat. Ctrl+C to exit.\n");

let sessionId: string | undefined;

while (true) {
  const input = await ask("> ");
  if (!input.trim()) continue;

  console.log("─".repeat(60));

  const options: Record<string, unknown> = {
    systemPrompt: { type: "preset", preset: "claude_code" },
    disallowedTools,
    mcpServers: { virtual: server },
    permissionMode: "bypassPermissions",
    allowDangerouslySkipPermissions: true,
  };

  if (sessionId) {
    options.resume = sessionId;
  }

  for await (const message of query({
    prompt: input,
    options: options as any,
  })) {
    // Capture session ID from any message
    if ("session_id" in message && message.session_id && !sessionId) {
      sessionId = message.session_id as string;
    }

    if (message.type === "assistant" && message.message?.content) {
      for (const block of message.message.content) {
        if ("text" in block && block.text) {
          console.log(block.text);
        }
      }
    }

    if (message.type === "result") {
      console.log("─".repeat(60));
      if (message.subtype === "success") {
        console.log(`Cost: $${message.total_cost_usd?.toFixed(4) ?? "?"}`);
        // Update session ID from result in case it wasn't captured
        if (message.session_id) {
          sessionId = message.session_id;
        }
      } else {
        console.log("Error occurred");
        if ("error" in message) {
          console.log(`${message.error}`);
        }
      }
      console.log("");
    }
  }
}
