const acorn = require('acorn');
const fs = require('fs');
const { execSync } = require('child_process');

if (process.argv.length < 3) {
    console.log('Usage: node compare-ast.js <file-path>');
    process.exit(1);
}

const filePath = process.argv[2];
const source = fs.readFileSync(filePath, 'utf-8');

// Determine source type
let sourceType = 'script';
if (filePath.endsWith('.mjs') || source.includes('import ') || source.includes('export ')) {
    sourceType = 'module';
}

// Parse with Acorn
let acornAst;
try {
    acornAst = acorn.parse(source, {ecmaVersion: 2025, locations: true, sourceType: sourceType});
} catch (e) {
    console.log('Acorn parse failed:', e.message);
    process.exit(1);
}

// Parse with Java parser and get AST
let javaAstJson;
try {
    const isModule = sourceType === 'module';
    const result = execSync(
        `mvn -q exec:java -Dexec.mainClass="com.jsparser.CompareHelper" -Dexec.args="${filePath} ${isModule}"`,
        { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 }
    );
    javaAstJson = result.trim();
} catch (e) {
    console.log('Java parse failed:', e.message);
    process.exit(1);
}

const javaAst = JSON.parse(javaAstJson);

// Deep comparison function
function findFirstDifference(path, acornVal, javaVal, depth = 0) {
    if (depth > 50) return null; // Prevent infinite recursion

    const acornType = typeof acornVal;
    const javaType = typeof javaVal;

    if (acornType !== javaType) {
        return {
            path: path,
            difference: 'type mismatch',
            acorn: acornType,
            java: javaType
        };
    }

    if (acornVal === null && javaVal === null) {
        return null;
    }

    if (acornVal === null || javaVal === null) {
        return {
            path: path,
            difference: 'null mismatch',
            acorn: acornVal,
            java: javaVal
        };
    }

    if (acornType === 'object') {
        if (Array.isArray(acornVal) && Array.isArray(javaVal)) {
            if (acornVal.length !== javaVal.length) {
                return {
                    path: path,
                    difference: 'array length',
                    acorn: acornVal.length,
                    java: javaVal.length
                };
            }
            for (let i = 0; i < acornVal.length; i++) {
                const diff = findFirstDifference(path + '[' + i + ']', acornVal[i], javaVal[i], depth + 1);
                if (diff) return diff;
            }
            return null;
        }

        if (Array.isArray(acornVal) || Array.isArray(javaVal)) {
            return {
                path: path,
                difference: 'array vs object',
                acorn: Array.isArray(acornVal) ? 'array' : 'object',
                java: Array.isArray(javaVal) ? 'array' : 'object'
            };
        }

        // Object comparison
        const acornKeys = Object.keys(acornVal).sort();
        const javaKeys = Object.keys(javaVal).sort();

        // Check for key differences
        const acornOnly = acornKeys.filter(k => !javaKeys.includes(k));
        const javaOnly = javaKeys.filter(k => !acornKeys.includes(k));

        if (acornOnly.length > 0 || javaOnly.length > 0) {
            return {
                path: path,
                difference: 'key mismatch',
                acornOnly: acornOnly,
                javaOnly: javaOnly
            };
        }

        // Compare all keys
        for (const key of acornKeys) {
            const diff = findFirstDifference(path + '.' + key, acornVal[key], javaVal[key], depth + 1);
            if (diff) return diff;
        }

        return null;
    }

    // Primitive comparison
    if (acornVal !== javaVal) {
        return {
            path: path,
            difference: 'value mismatch',
            acorn: acornVal,
            java: javaVal
        };
    }

    return null;
}

const difference = findFirstDifference('root', acornAst, javaAst);

if (difference) {
    console.log('First difference found:');
    console.log(JSON.stringify(difference, null, 2));
} else {
    console.log('ASTs are identical');
}
