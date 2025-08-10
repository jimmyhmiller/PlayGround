#!/usr/bin/env node

// Simple demo to show the box without input handling for testing
import React from 'react';
import { render, Box, Text } from 'ink';

const DemoBox = () => {
  const terminalWidth = process.stdout.columns || 80;
  
  return (
    <Box flexDirection="column">
      <Text>Full Width Text Box Demo - Terminal Width: {terminalWidth}</Text>
      <Text> </Text>
      <Box
        borderStyle="round"
        borderColor="cyan"
        width={terminalWidth}
        paddingX={1}
        paddingY={0}
      >
        <Text>This box spans the full width of your terminal window!</Text>
      </Box>
      <Text> </Text>
      <Text color="gray">Box width: {terminalWidth} characters</Text>
      <Text color="gray">Content width: {terminalWidth - 4} characters (accounting for border and padding)</Text>
    </Box>
  );
};

render(<DemoBox />);