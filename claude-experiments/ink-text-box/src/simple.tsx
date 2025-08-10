#!/usr/bin/env node

import React, { useState } from 'react';
import { render, Box, Text, useInput, useApp } from 'ink';

const SimpleTextBox: React.FC = () => {
  const [value, setValue] = useState('');
  const [cursorPos, setCursorPos] = useState(0);
  const { exit } = useApp();

  useInput((input, key) => {
    if (key.ctrl && input === 'c') {
      exit();
      return;
    }

    if (key.escape) {
      exit();
      return;
    }

    if (key.backspace || key.delete) {
      if (cursorPos > 0) {
        setValue(prev => {
          const newValue = prev.slice(0, cursorPos - 1) + prev.slice(cursorPos);
          setCursorPos(cursorPos - 1);
          return newValue;
        });
      }
      return;
    }

    if (key.leftArrow) {
      setCursorPos(Math.max(0, cursorPos - 1));
      return;
    }

    if (key.rightArrow) {
      setCursorPos(Math.min(value.length, cursorPos + 1));
      return;
    }

    if (key.return) {
      if (key.meta || key.option) {
        // Option/Alt + Enter for multiline
        setValue(prev => {
          const newValue = prev.slice(0, cursorPos) + '\n' + prev.slice(cursorPos);
          setCursorPos(cursorPos + 1);
          return newValue;
        });
      }
      return;
    }

    if (input && input.length > 0) {
      setValue(prev => {
        const newValue = prev.slice(0, cursorPos) + input + prev.slice(cursorPos);
        setCursorPos(cursorPos + input.length);
        return newValue;
      });
    }
  });

  // Simple full-width approach without resize handling
  const terminalWidth = process.stdout.columns || 80;
  const displayValue = value || 'Enter your text here...';
  const lines = displayValue.split('\n');
  
  // Calculate cursor position for display
  let cursorLine = 0;
  let cursorCol = cursorPos;
  let charCount = 0;
  
  if (value) {
    for (let i = 0; i < lines.length; i++) {
      if (charCount + lines[i].length >= cursorPos) {
        cursorLine = i;
        cursorCol = cursorPos - charCount;
        break;
      }
      charCount += lines[i].length + 1; // +1 for the \n
    }
  }

  return (
    <Box flexDirection="column">
      <Text>Simple Full-Width Text Input (Press Esc or Ctrl+C to exit, Option+Enter for new line)</Text>
      <Text> </Text>
      <Box
        borderStyle="round"
        borderColor="cyan"
        width="100%"
        paddingX={1}
        paddingY={0}
      >
        <Box flexDirection="column" width="100%">
          {lines.map((line, index) => (
            <Text key={`${index}-${line.length}`}>
              {index === cursorLine && value ? (
                <>
                  {line.slice(0, cursorCol)}
                  <Text backgroundColor="white" color="black">
                    {line[cursorCol] || ' '}
                  </Text>
                  {line.slice(cursorCol + 1)}
                </>
              ) : (
                <Text color={value ? 'white' : 'gray'}>
                  {line}
                  {index === cursorLine && !value && (
                    <Text backgroundColor="white" color="black"> </Text>
                  )}
                </Text>
              )}
            </Text>
          ))}
        </Box>
      </Box>
      <Text> </Text>
      <Text color="gray">Current value: "{value}"</Text>
      <Text color="gray">Terminal width: {terminalWidth} | Cursor: {cursorPos}</Text>
    </Box>
  );
};

render(<SimpleTextBox />, {
  exitOnCtrlC: false
});