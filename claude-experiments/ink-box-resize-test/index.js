#!/usr/bin/env node
import React, { useState, useEffect } from 'react';
import { render, Box, Text, useApp, useInput } from 'ink';

const BoxResizeTest = () => {
    const [dimensions, setDimensions] = useState({
        width: process.stdout.columns,
        height: process.stdout.rows
    });
    const { exit } = useApp();

    useEffect(() => {
        const handleResize = () => {
            setDimensions({
                width: process.stdout.columns,
                height: process.stdout.rows
            });
        };

        process.stdout.on('resize', handleResize);
        return () => {
            process.stdout.off('resize', handleResize);
        };
    }, []);

    useInput((input, key) => {
        if (input === 'q' || key.escape) {
            exit();
        }
    });

    return (
        <Box flexDirection="column" paddingY={1}>
            <Box>
                <Text>Terminal size: {dimensions.width} x {dimensions.height}</Text>
            </Box>
            
            <Box marginTop={1}>
                <Text>Press 'q' or ESC to exit. Resize your terminal to test.</Text>
            </Box>

            {/* Test box with border */}
            <Box
                marginTop={1}
                borderStyle="round"
                borderColor="green"
                padding={1}
                width="50%"
            >
                <Text>This is a bordered box with 50% width</Text>
            </Box>

            {/* Another test box */}
            <Box
                marginTop={1}
                borderStyle="single"
                borderColor="blue"
                padding={1}
                flexDirection="column"
            >
                <Text>This box has a single border style</Text>
                <Text>It contains multiple lines</Text>
                <Text>Watch what happens when you resize!</Text>
            </Box>

            {/* Box with fixed width */}
            <Box
                marginTop={1}
                borderStyle="double"
                borderColor="red"
                padding={1}
                width={40}
            >
                <Text>Fixed width (40 chars) box</Text>
            </Box>

            {/* Nested boxes */}
            <Box
                marginTop={1}
                borderStyle="round"
                borderColor="cyan"
                padding={1}
                flexDirection="column"
            >
                <Text>Parent box with nested content:</Text>
                <Box
                    marginTop={1}
                    borderStyle="single"
                    borderColor="yellow"
                    padding={1}
                >
                    <Text>Nested box inside</Text>
                </Box>
            </Box>
        </Box>
    );
};

render(<BoxResizeTest />);