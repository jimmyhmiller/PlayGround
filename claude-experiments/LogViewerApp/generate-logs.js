#!/usr/bin/env node

const fs = require('fs');

function randomChoice(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
}

function generateLogEntry(timestamp) {
    const levels = ['INFO', 'WARN', 'ERROR', 'DEBUG'];
    const level = randomChoice(levels);
    
    const messages = {
        'INFO': [
            'Request processed successfully',
            'User authenticated',
            'Cache updated',
            'Configuration reloaded',
            'Database connection established',
            'Service started',
            'Health check passed',
            'Session created',
            'File uploaded successfully',
            'Email sent',
            'Webhook delivered',
            'Task completed'
        ],
        'WARN': [
            'High memory usage detected',
            'API rate limit approaching',
            'Deprecated endpoint used',
            'Cache miss rate high',
            'Slow query detected',
            'Certificate expiring soon',
            'Disk space low',
            'Connection pool nearly exhausted',
            'Retry attempt',
            'Timeout warning'
        ],
        'ERROR': [
            'Database connection failed',
            'Authentication failed',
            'File not found',
            'Permission denied',
            'Invalid request format',
            'Service unavailable',
            'Timeout exceeded',
            'Memory allocation failed',
            'Null pointer exception',
            'Network unreachable'
        ],
        'DEBUG': [
            'Entering function processRequest',
            'Variable state changed',
            'SQL query executed',
            'Cache lookup performed',
            'Event listener triggered',
            'Middleware executed',
            'Token validated',
            'Response serialized',
            'Request parsed',
            'Configuration loaded'
        ]
    };
    
    const message = randomChoice(messages[level]);
    const dateStr = timestamp.toISOString().replace('T', ' ').substring(0, 23);
    
    return `${dateStr} [${level}] ${message}`;
}

function generateLogs() {
    const logs = [];
    let currentTime = new Date('2025-01-15T08:00:00.000Z');
    const endTime = new Date('2025-01-15T20:00:00.000Z'); // 12 hours of logs
    
    while (currentTime < endTime) {
        const scenario = Math.random();
        
        if (scenario < 0.2) {
            // 20% chance: Continuous activity (constant stream with no gaps)
            const duration = Math.floor(Math.random() * 5) + 3; // 3-8 minutes of continuous logs
            const endContinuous = new Date(currentTime.getTime() + duration * 60 * 1000);
            console.log(`Generating continuous activity for ${duration} minutes at ${currentTime.toISOString()}`);
            
            while (currentTime < endContinuous && currentTime < endTime) {
                logs.push(generateLogEntry(new Date(currentTime)));
                currentTime = new Date(currentTime.getTime() + Math.random() * 200 + 50); // 50-250ms between events
            }
            
        } else if (scenario < 0.4) {
            // 20% chance: Burst of activity (many logs in quick succession)
            const burstSize = Math.floor(Math.random() * 50) + 20;
            console.log(`Generating burst of ${burstSize} events at ${currentTime.toISOString()}`);
            
            for (let i = 0; i < burstSize; i++) {
                logs.push(generateLogEntry(new Date(currentTime)));
                currentTime = new Date(currentTime.getTime() + Math.random() * 500); // 0-500ms between burst events
            }
            
        } else if (scenario < 0.55) {
            // 15% chance: Large gap (quiet period)
            const gapMinutes = Math.floor(Math.random() * 30) + 10; // 10-40 minute gap
            console.log(`Creating ${gapMinutes} minute gap at ${currentTime.toISOString()}`);
            currentTime = new Date(currentTime.getTime() + gapMinutes * 60 * 1000);
            
        } else if (scenario < 0.65) {
            // 10% chance: Error burst (simulating an incident)
            const errorCount = Math.floor(Math.random() * 20) + 10;
            console.log(`Generating error burst of ${errorCount} events at ${currentTime.toISOString()}`);
            
            for (let i = 0; i < errorCount; i++) {
                const timestamp = new Date(currentTime);
                const dateStr = timestamp.toISOString().replace('T', ' ').substring(0, 23);
                const errorMessages = [
                    'Connection to database lost',
                    'Service health check failed', 
                    'Out of memory error',
                    'Too many open connections',
                    'Request timeout after 30s'
                ];
                logs.push(`${dateStr} [ERROR] ${randomChoice(errorMessages)}`);
                currentTime = new Date(currentTime.getTime() + Math.random() * 1000);
            }
            
        } else if (scenario < 0.75) {
            // 10% chance: Heavy continuous load (simulating high traffic)
            const heavyDuration = Math.floor(Math.random() * 3) + 2; // 2-5 minutes
            const endHeavy = new Date(currentTime.getTime() + heavyDuration * 60 * 1000);
            console.log(`Generating heavy continuous load for ${heavyDuration} minutes at ${currentTime.toISOString()}`);
            
            while (currentTime < endHeavy && currentTime < endTime) {
                // Mix of levels during heavy load
                logs.push(generateLogEntry(new Date(currentTime)));
                currentTime = new Date(currentTime.getTime() + Math.random() * 100 + 20); // 20-120ms between events (very fast)
            }
            
        } else {
            // 25% chance: Normal activity (steady stream)
            const normalCount = Math.floor(Math.random() * 10) + 5;
            
            for (let i = 0; i < normalCount; i++) {
                logs.push(generateLogEntry(new Date(currentTime)));
                currentTime = new Date(currentTime.getTime() + Math.random() * 5000 + 1000); // 1-6 seconds between normal events
            }
        }
    }
    
    // Add some final entries
    console.log('Adding final entries...');
    for (let i = 0; i < 10; i++) {
        logs.push(generateLogEntry(new Date(currentTime)));
        currentTime = new Date(currentTime.getTime() + Math.random() * 2000);
    }
    
    return logs;
}

// Generate the logs
console.log('Starting log generation...');
const logs = generateLogs();

// Write to file
const outputFile = 'large-sample.log';
fs.writeFileSync(outputFile, logs.join('\n'), 'utf8');

console.log(`\nGenerated ${logs.length} log entries`);
console.log(`Output written to ${outputFile}`);

// Show some statistics
const errorCount = logs.filter(l => l.includes('[ERROR]')).length;
const warnCount = logs.filter(l => l.includes('[WARN]')).length;
const infoCount = logs.filter(l => l.includes('[INFO]')).length;
const debugCount = logs.filter(l => l.includes('[DEBUG]')).length;

console.log('\nLog statistics:');
console.log(`  INFO:  ${infoCount}`);
console.log(`  WARN:  ${warnCount}`);
console.log(`  ERROR: ${errorCount}`);
console.log(`  DEBUG: ${debugCount}`);
console.log(`  TOTAL: ${logs.length}`);