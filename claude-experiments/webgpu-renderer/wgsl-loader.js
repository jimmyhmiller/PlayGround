/**
 * Node.js loader for .wgsl shader files
 * Allows importing .wgsl files as text in Node.js
 */
import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';

export async function load(url, context, nextLoad) {
    // Handle .wgsl files (with or without ?raw query)
    if (url.endsWith('.wgsl') || url.includes('.wgsl?')) {
        const filePath = fileURLToPath(url.split('?')[0]);
        const source = await readFile(filePath, 'utf8');

        return {
            format: 'module',
            source: `export default ${JSON.stringify(source)};`,
            shortCircuit: true,
        };
    }

    // Let Node.js handle other URLs
    return nextLoad(url, context);
}
