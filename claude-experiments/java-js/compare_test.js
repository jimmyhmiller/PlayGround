const acorn = require('acorn');
const fs = require('fs');
const crypto = require('crypto');

const jsPath = 'test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_TypeScript_tests_baselines_reference_binaryIntegerLiteral.js';
const source = fs.readFileSync(jsPath, 'utf-8');
const ast = acorn.parse(source, {ecmaVersion: 2025, locations: true, sourceType: 'script'});

// Write streaming JSON (same as test)
const out = fs.createWriteStream('/tmp/acorn-test.json');
function writeJson(obj, indent) {
    if (obj === null) { out.write('null'); return; }
    if (typeof obj === 'undefined') { out.write('null'); return; }
    if (typeof obj === 'boolean') { out.write(obj.toString()); return; }
    if (typeof obj === 'number') { out.write(obj.toString()); return; }
    if (typeof obj === 'bigint') { out.write('"' + obj.toString() + '"'); return; }
    if (typeof obj === 'string') { out.write(JSON.stringify(obj)); return; }
    if (Array.isArray(obj)) {
        out.write('[');
        for (let i = 0; i < obj.length; i++) {
            if (i > 0) out.write(',');
            out.write('\n' + indent + '  ');
            writeJson(obj[i], indent + '  ');
        }
        if (obj.length > 0) out.write('\n' + indent);
        out.write(']');
        return;
    }
    out.write('{');
    const keys = Object.keys(obj);
    for (let i = 0; i < keys.length; i++) {
        if (i > 0) out.write(',');
        out.write('\n' + indent + '  ');
        out.write(JSON.stringify(keys[i]) + ': ');
        writeJson(obj[keys[i]], indent + '  ');
    }
    if (keys.length > 0) out.write('\n' + indent);
    out.write('}');
}
writeJson(ast, '');
out.write('\n');
out.end(() => {
    console.log('Acorn AST written to /tmp/acorn-test.json');

    // Now check what we get from parsing
    const acornJson = JSON.parse(fs.readFileSync('/tmp/acorn-test.json', 'utf-8'));
    console.log('Acorn sourceType:', acornJson.sourceType);
    console.log('Acorn body length:', acornJson.body.length);
});
