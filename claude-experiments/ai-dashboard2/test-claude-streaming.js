const { query } = require('@anthropic-ai/claude-agent-sdk');
const util = require('util');

async function testStreaming() {
  console.log('Starting Claude Agent SDK streaming test...\n');

  const processedTexts = new Set();
  const allMessages = [];

  try {
    const result = query({
      prompt: "Hey, what files can you see?",
      options: {
        model: 'claude-sonnet-4-5-20250929',
        cwd: process.cwd(),
        permissionMode: 'acceptEdits'
      }
    });

    console.log('=== Streaming messages ===\n');

    for await (const msg of result) {
      allMessages.push(msg);

      console.log(`\n--- Message ${allMessages.length} ---`);
      console.log(`Type: ${msg.type}`);

      if (msg.type === 'assistant') {
        const messageId = msg.message?.id;
        const content = msg.message?.content;

        console.log(`Message ID: ${messageId}`);
        console.log(`Content blocks: ${content?.length || 0}`);

        if (Array.isArray(content)) {
          content.forEach((block, idx) => {
            console.log(`\n  Block ${idx}:`);
            console.log(`    Type: ${block.type}`);

            if (block.type === 'text') {
              const text = block.text;
              const textHash = text.trim();
              const isNew = !processedTexts.has(textHash);

              console.log(`    Text length: ${text.length} chars`);
              console.log(`    First 100 chars: ${text.substring(0, 100)}...`);
              console.log(`    Is new? ${isNew}`);

              if (isNew) {
                processedTexts.add(textHash);
                console.log(`    ✓ WOULD DISPLAY THIS`);
              } else {
                console.log(`    ✗ SKIPPING (duplicate)`);
              }
            } else if (block.type === 'tool_use') {
              console.log(`    Tool: ${block.name}`);
            }
          });
        }
      }
    }

    console.log('\n\n=== Summary ===');
    console.log(`Total messages received: ${allMessages.length}`);
    console.log(`Unique text blocks: ${processedTexts.size}`);

    console.log('\n=== All unique texts that would be displayed ===');
    let displayCount = 0;
    processedTexts.forEach(text => {
      displayCount++;
      console.log(`\n${displayCount}. ${text.substring(0, 200)}${text.length > 200 ? '...' : ''}`);
    });

  } catch (error) {
    console.error('Error:', error);
  }
}

testStreaming();
