#!/usr/bin/env node
import React, { useState, useEffect } from 'react';
import { render, Box, Text, Static, useApp, useInput } from 'ink';

const StaticBoxTest = () => {
    const [dimensions, setDimensions] = useState({
        width: process.stdout.columns,
        height: process.stdout.rows
    });
    const [items, setItems] = useState([
        { id: 1, label: 'Static box 1' },
        { id: 2, label: 'Static box 2' },
        { id: 3, label: 'Static box 3' }
    ]);
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
        // Add a new item with 'a'
        if (input === 'a') {
            setItems(prev => [...prev, { id: Date.now(), label: `New box ${prev.length + 1}` }]);
        }
    });

    return React.createElement(Box, { flexDirection: "column", paddingY: 1 },
        React.createElement(Box, null,
            React.createElement(Text, null, `Terminal size: ${dimensions.width} x ${dimensions.height}`)
        ),
        
        React.createElement(Box, { marginTop: 1 },
            React.createElement(Text, null, "Press 'q' to exit, 'a' to add a box. Resize terminal to test.")
        ),

        // Static content with borders
        React.createElement(Static, { items: items },
            (item) => React.createElement(Box, {
                key: item.id,
                marginTop: 1,
                borderStyle: "round",
                borderColor: "green",
                padding: 1,
                width: "80%"
            },
                React.createElement(Text, null, `${item.label} - This should remain static`)
            )
        ),

        // Dynamic content for comparison
        React.createElement(Box, { marginTop: 2 },
            React.createElement(Text, { bold: true }, "Dynamic content below:")
        ),
        
        React.createElement(Box, {
            marginTop: 1,
            borderStyle: "single",
            borderColor: "blue",
            padding: 1,
            flexDirection: "column"
        },
            React.createElement(Text, null, "This is dynamic content"),
            React.createElement(Text, null, `Current width: ${dimensions.width}`),
            React.createElement(Text, null, "It re-renders on resize")
        )
    );
};

render(React.createElement(StaticBoxTest));