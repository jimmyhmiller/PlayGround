#!/usr/bin/env node

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { render, Box, Text } from 'ink';

const ResizeDemo = () => {
  const [terminalWidth, setTerminalWidth] = useState(process.stdout.columns || 80);
  const [resizeCount, setResizeCount] = useState(0);
  const resizeTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  const handleResize = useCallback(() => {
    if (resizeTimeoutRef.current) {
      clearTimeout(resizeTimeoutRef.current);
    }
    
    resizeTimeoutRef.current = setTimeout(() => {
      const newWidth = process.stdout.columns || 80;
      if (newWidth !== terminalWidth) {
        setTerminalWidth(newWidth);
        setResizeCount(prev => prev + 1);
      }
    }, 100); // 100ms debounce
  }, [terminalWidth]);

  useEffect(() => {
    process.stdout.on('resize', handleResize);
    
    return () => {
      process.stdout.off('resize', handleResize);
      if (resizeTimeoutRef.current) {
        clearTimeout(resizeTimeoutRef.current);
      }
    };
  }, [handleResize]);

  return (
    <Box flexDirection="column">
      <Text>Resize Demo - Terminal Width: {terminalWidth} (Resized {resizeCount} times)</Text>
      <Text> </Text>
      <Box
        borderStyle="round"
        borderColor="green"
        width={terminalWidth}
        paddingX={1}
        paddingY={0}
      >
        <Text>This box adjusts when you resize your terminal window!</Text>
      </Box>
      <Text> </Text>
      <Text color="gray">Try resizing your terminal window to see the box adjust.</Text>
      <Text color="gray">The box will span the full width and update smoothly.</Text>
    </Box>
  );
};

render(<ResizeDemo />);