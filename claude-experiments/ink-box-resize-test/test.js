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

    return React.createElement(Box, { flexDirection: "column", paddingY: 1 },
        React.createElement(Box, null,
            React.createElement(Text, null, `Terminal size: ${dimensions.width} x ${dimensions.height}`)
        ),
        
        React.createElement(Box, { marginTop: 1 },
            React.createElement(Text, null, "Press 'q' or ESC to exit. Resize your terminal to test.")
        ),

        // Test box with border
        React.createElement(Box, {
            marginTop: 1,
            borderStyle: "round",
            borderColor: "green",
            padding: 1,
            width: "50%"
        },
            React.createElement(Text, null, "This is a bordered box with 50% width")
        ),

        // Another test box
        React.createElement(Box, {
            marginTop: 1,
            borderStyle: "single",
            borderColor: "blue",
            padding: 1,
            flexDirection: "column"
        },
            React.createElement(Text, null, "This box has a single border style"),
            React.createElement(Text, null, "It contains multiple lines"),
            React.createElement(Text, null, "Watch what happens when you resize!")
        ),

        // Box with fixed width
        React.createElement(Box, {
            marginTop: 1,
            borderStyle: "double",
            borderColor: "red",
            padding: 1,
            width: 40
        },
            React.createElement(Text, null, "Fixed width (40 chars) box")
        ),

        // Nested boxes
        React.createElement(Box, {
            marginTop: 1,
            borderStyle: "round",
            borderColor: "cyan",
            padding: 1,
            flexDirection: "column"
        },
            React.createElement(Text, null, "Parent box with nested content:"),
            React.createElement(Box, {
                marginTop: 1,
                borderStyle: "single",
                borderColor: "yellow",
                padding: 1
            },
                React.createElement(Text, null, "Nested box inside")
            )
        )
    );
};

render(React.createElement(BoxResizeTest));