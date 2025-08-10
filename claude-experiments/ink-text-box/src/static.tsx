#!/usr/bin/env node

import React, { useState, useCallback } from 'react';
import { render, Box, Text, useInput, useApp, Static } from 'ink';

const StaticTextBox: React.FC = () => {
  const [value, setValue] = useState('');
  const [cursorPos, setCursorPos] = useState(0);
  const [renders, setRenders] = useState<string[]>([]);
  const { exit } = useApp();

  const addRender = useCallback((message: string) => {
    setRenders(prev => [...prev, `${new Date().toISOString()}: ${message}`]);
  }, []);

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
          addRender(`Backspace: "${newValue}"`);
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
        setValue(prev => {
          const newValue = prev.slice(0, cursorPos) + '\n' + prev.slice(cursorPos);
          setCursorPos(cursorPos + 1);
          addRender(`New line: "${newValue}"`);
          return newValue;
        });
      }
      return;
    }

    if (input && input.length > 0) {
      setValue(prev => {
        const newValue = prev.slice(0, cursorPos) + input + prev.slice(cursorPos);
        setCursorPos(cursorPos + input.length);
        addRender(`Added "${input}": "${newValue}"`);
        return newValue;
      });
    }
  });

  const terminalWidth = process.stdout.columns || 80;
  const displayValue = value || 'Enter your text here...';
  const lines = displayValue.split('\n');
  
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
      charCount += lines[i].length + 1;
    }
  }

  return (
    <Box flexDirection="column">
      <Static items={[`Width: ${terminalWidth}`]}>
        {(item, index) => (
          <Text key={index}>Static Text Input - {item}</Text>
        )}
      </Static>
      
      <Box
        borderStyle="round"
        borderColor="cyan"
        width="100%"
        paddingX={1}
        paddingY={0}
      >
        {lines.map((line, index) => (
          <Text key={`line-${index}`}>
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
      
      <Text color="gray">Value: "{value}" | Cursor: {cursorPos}</Text>
      <Text color="yellow">Renders: {renders.length}</Text>
      
      <Static items={renders.slice(-3)}>
        {(item, index) => (
          <Text key={index} color="dim">{item}</Text>
        )}
      </Static>
    </Box>
  );
};

render(<StaticTextBox />, {
  exitOnCtrlC: false
});