#!/usr/bin/env tsx

import { query } from '@anthropic-ai/claude-agent-sdk';
import * as dotenv from 'dotenv';

dotenv.config();

async function testAgentSDK() {
  console.log('Testing Agent SDK integration...\n');

  const result = query({
    prompt: 'What is 2+2? Just answer with the number.',
    options: {
      cwd: process.cwd(),
      maxTurns: 1,
      permissionMode: 'default', // Don't auto-approve anything
      allowedTools: ['Read'], // Only allow reading files, no edits
    }
  });

  let messageCount = 0;
  let response = '';

  for await (const message of result) {
    messageCount++;
    console.log(`\nMessage ${messageCount}:`);
    console.log(`  Type: ${message.type}`);

    if (message.type === 'assistant') {
      const content = (message as any).message?.content || [];
      console.log(`  Content items: ${content.length}`);
      for (const item of content) {
        console.log(`    - ${item.type}: ${item.type === 'text' ? item.text.substring(0, 100) : '[non-text]'}`);
        if (item.type === 'text') {
          response += item.text;
        }
      }
    } else if (message.type === 'result') {
      const usage = (message as any).usage || {};
      console.log(`  Tokens: input=${usage.input_tokens}, output=${usage.output_tokens}`);
      console.log(`  Cost: $${((message as any).total_cost_usd || 0).toFixed(4)}`);
    } else {
      console.log(`  Full message:`, JSON.stringify(message, null, 2).substring(0, 500));
    }
  }

  console.log(`\n\n=== FINAL RESPONSE ===`);
  console.log(response);
  console.log(`\nTotal messages: ${messageCount}`);
}

testAgentSDK().catch(error => {
  console.error('Error:', error.message);
  console.error(error);
  process.exit(1);
});
