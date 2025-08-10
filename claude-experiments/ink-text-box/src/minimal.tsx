#!/usr/bin/env node

import React, { useState } from 'react';
import { render, Box, Text, useInput, useApp } from 'ink';

const MinimalTextBox: React.FC = () => {
  const [value, setValue] = useState('');
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

    if (key.backspace) {
      setValue(prev => prev.slice(0, -1));
      return;
    }

    if (key.return) {
      if (key.meta || key.option) {
        setValue(prev => prev + '\n');
      }
      return;
    }

    if (input && input.length > 0) {
      setValue(prev => prev + input);
    }
  });

  const lines = (value || 'Enter your text here...').split('\n');

  return (
    <Box flexDirection="column">
      <Text>Minimal Text Input (Esc to exit, Option+Enter for newline)</Text>
      <Box borderStyle="round" borderColor="cyan" width="100%" paddingX={1}>
        <Box flexDirection="column">
          {lines.map((line, i) => (
            <Text key={i} color={value ? 'white' : 'gray'}>
              {line}
            </Text>
          ))}
        </Box>
      </Box>
      <Text color="gray">Length: {value.length}</Text>
    </Box>
  );
};

render(<MinimalTextBox />, { exitOnCtrlC: false });