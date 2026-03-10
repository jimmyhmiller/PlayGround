import * as vscode from 'vscode';
import * as net from 'net';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

let connection: BeagleConnection | null = null;
let outputChannel: vscode.OutputChannel;

class BeagleConnection {
    private socket: net.Socket | null = null;
    private buffer = '';
    private messageId = 0;
    private pendingEvals = new Map<string, {
        resolve: (result: EvalResult) => void;
        output: string;
        value: string | null;
        error: string | null;
    }>();
    private connected = false;
    private sessionId: string;

    constructor(
        private host: string,
        private port: number,
    ) {
        this.sessionId = `vscode-${Date.now()}`;
    }

    async connect(): Promise<void> {
        if (this.connected) { return; }

        return new Promise((resolve, reject) => {
            this.socket = new net.Socket();

            this.socket.connect(this.port, this.host, () => {
                this.connected = true;
                outputChannel.appendLine(`Connected to Beagle REPL at ${this.host}:${this.port}`);
                resolve();
            });

            this.socket.on('data', (data) => {
                this.handleData(data.toString());
            });

            this.socket.on('error', (err) => {
                if (!this.connected) {
                    reject(err);
                } else {
                    outputChannel.appendLine(`Connection error: ${err.message}`);
                    this.disconnect();
                }
            });

            this.socket.on('close', () => {
                this.connected = false;
                // Reject all pending evals
                for (const [id, pending] of this.pendingEvals) {
                    pending.resolve({ output: pending.output, value: null, error: 'Connection closed' });
                }
                this.pendingEvals.clear();
            });
        });
    }

    private handleData(data: string) {
        this.buffer += data;
        const lines = this.buffer.split('\n');
        // Last element is incomplete line (or empty if data ended with \n)
        this.buffer = lines.pop() || '';

        for (const line of lines) {
            if (line.trim().length === 0) { continue; }
            try {
                const msg = JSON.parse(line);
                this.handleMessage(msg);
            } catch {
                outputChannel.appendLine(`Invalid JSON from server: ${line}`);
            }
        }
    }

    private handleMessage(msg: Record<string, unknown>) {
        const id = msg.id as string;
        if (!id) { return; }

        const pending = this.pendingEvals.get(id);
        if (!pending) { return; }

        if (msg.out) {
            pending.output += msg.out as string;
        }
        if (msg.value !== undefined) {
            pending.value = msg.value as string;
        }
        if (msg.error) {
            pending.error = msg.error as string;
        }
        if (msg.ex) {
            pending.error = msg.ex as string;
        }

        const status = msg.status as string[] | undefined;
        if (status && (status.includes('done') || status.includes('error'))) {
            this.pendingEvals.delete(id);
            pending.resolve({
                output: pending.output,
                value: pending.value,
                error: pending.error,
            });
        }
    }

    async eval(code: string): Promise<EvalResult> {
        if (!this.connected) {
            await this.connect();
        }

        const id = String(++this.messageId);
        const msg: Record<string, string> = {
            op: 'eval',
            id,
            session: this.sessionId,
            code,
        };
        const request = JSON.stringify(msg);

        return new Promise((resolve) => {
            this.pendingEvals.set(id, { resolve, output: '', value: null, error: null });
            this.socket!.write(request + '\n');
        });
    }

    disconnect() {
        if (this.socket) {
            this.socket.destroy();
            this.socket = null;
        }
        this.connected = false;
        this.pendingEvals.clear();
    }

    isConnected() { return this.connected; }
}

interface EvalResult {
    output: string;
    value: string | null;
    error: string | null;
}

// Inline decoration type for showing eval results
const resultDecorationType = vscode.window.createTextEditorDecorationType({
    after: {
        margin: '0 0 0 2em',
        color: new vscode.ThemeColor('editorCodeLens.foreground'),
    },
    isWholeLine: true,
});

const errorDecorationType = vscode.window.createTextEditorDecorationType({
    after: {
        margin: '0 0 0 2em',
        color: new vscode.ThemeColor('errorForeground'),
    },
    isWholeLine: true,
});

function showResultInline(editor: vscode.TextEditor, line: number, result: EvalResult) {
    const range = new vscode.Range(line, 0, line, editor.document.lineAt(line).text.length);

    if (result.error) {
        editor.setDecorations(errorDecorationType, [{
            range,
            renderOptions: {
                after: { contentText: `  => Error: ${result.error}` },
            },
        }]);
        editor.setDecorations(resultDecorationType, []);
    } else if (result.value !== null) {
        editor.setDecorations(resultDecorationType, [{
            range,
            renderOptions: {
                after: { contentText: `  => ${result.value}` },
            },
        }]);
        editor.setDecorations(errorDecorationType, []);
    }
}

// ============================================================================
// Expression-before-cursor detection (Lisp/Clojure-style eval)
// ============================================================================

const BRACKET_PAIRS: Record<string, string> = { ')': '(', '}': '{', ']': '[' };
const BLOCK_KEYWORDS = new Set([
    'fn', 'if', 'else', 'while', 'for', 'match', 'struct', 'enum',
    'let', 'try', 'catch', 'binding', 'namespace', 'protocol', 'use',
    'handle', 'handler',
]);

function findMatchingBracket(text: string, closePos: number): number {
    const closeChar = text[closePos];
    const openChar = BRACKET_PAIRS[closeChar];
    if (!openChar) { return -1; }

    let depth = 1;
    let i = closePos - 1;

    while (i >= 0 && depth > 0) {
        const ch = text[i];

        // Skip over string literals (scanning backwards)
        if (ch === '"') {
            i--;
            while (i >= 0) {
                if (text[i] === '"') {
                    let bs = 0;
                    let j = i - 1;
                    while (j >= 0 && text[j] === '\\') { bs++; j--; }
                    if (bs % 2 === 0) { break; }
                }
                i--;
            }
            i--;
            continue;
        }

        // Skip line comments: if we see / preceded by /, skip to start of line
        if (ch === '/' && i > 0 && text[i - 1] === '/') {
            i -= 2;
            while (i >= 0 && text[i] !== '\n') { i--; }
            continue;
        }

        if (ch === closeChar) { depth++; }
        else if (ch === openChar) { depth--; }

        if (depth > 0) { i--; }
    }

    return depth === 0 ? i : -1;
}

function extendBackOverFunctionName(text: string, parenPos: number): number {
    let i = parenPos - 1;
    while (i >= 0 && /\s/.test(text[i])) { i--; }
    if (i < 0 || !/[\w\-!?.]/.test(text[i])) { return parenPos; }

    // Scan back over identifier (including qualified names like ns/func)
    while (i > 0 && /[\w\-!?./]/.test(text[i - 1])) { i--; }
    return i;
}

function extendBackOverBlockHeader(text: string, openBracePos: number): number {
    let result = openBracePos;
    let pos = openBracePos - 1;

    while (pos >= 0) {
        // Skip whitespace (but not newlines for now - we'll handle multi-line below)
        while (pos >= 0 && /\s/.test(text[pos])) { pos--; }
        if (pos < 0) { break; }

        // If preceded by '}', this is like `} else {` or `} catch (e) {`
        // Find matching '{' and recurse
        if (text[pos] === '}') {
            const innerOpen = findMatchingBracket(text, pos);
            if (innerOpen < 0) { break; }
            result = innerOpen;
            return extendBackOverBlockHeader(text, innerOpen);
        }

        // If preceded by ')', find matching '(' and continue
        if (text[pos] === ')') {
            const matchPos = findMatchingBracket(text, pos);
            if (matchPos < 0) { break; }
            result = matchPos;
            pos = matchPos - 1;
            continue;
        }

        // If preceded by an identifier, read it
        if (/[\w\-!?]/.test(text[pos])) {
            const wordEnd = pos;
            while (pos > 0 && /[\w\-!?]/.test(text[pos - 1])) { pos--; }
            const word = text.substring(pos, wordEnd + 1);
            result = pos;

            if (BLOCK_KEYWORDS.has(word) && word !== 'else' && word !== 'catch') {
                break; // Found the keyword that starts this construct
            }
            if (word === 'else' || word === 'catch') {
                // These are continuations - keep scanning to find the
                // preceding if/try block
                pos--;
                continue;
            }

            // Not a keyword (function name, condition token, etc.) - keep scanning
            pos--;
            continue;
        }

        // Operator or other character - keep scanning back
        // This handles conditions like `x > 5` in `if x > 5 {`
        pos--;
    }

    return result;
}

function findExpressionBeforeCursor(
    document: vscode.TextDocument,
    position: vscode.Position,
): vscode.Range | null {
    const text = document.getText();
    const offset = document.offsetAt(position);

    // Skip whitespace backwards from cursor
    let end = offset - 1;
    while (end >= 0 && /\s/.test(text[end])) { end--; }
    if (end < 0) { return null; }

    const endChar = text[end];
    let start: number;

    if (endChar in BRACKET_PAIRS) {
        start = findMatchingBracket(text, end);
        if (start < 0) { return null; }

        if (endChar === ')') {
            start = extendBackOverFunctionName(text, start);
        } else if (endChar === '}') {
            start = extendBackOverBlockHeader(text, start);
        }
    } else if (endChar === '"') {
        // Find matching opening quote
        start = end - 1;
        while (start >= 0) {
            if (text[start] === '"') {
                let bs = 0;
                let j = start - 1;
                while (j >= 0 && text[j] === '\\') { bs++; j--; }
                if (bs % 2 === 0) { break; }
            }
            start--;
        }
        if (start < 0) { return null; }
    } else if (/[\w\-!?]/.test(endChar)) {
        // Identifier or number
        start = end;
        while (start > 0 && /[\w\-!?./]/.test(text[start - 1])) { start--; }
    } else {
        start = end;
    }

    // Check if this expression is the RHS of a `let` binding: `let name = <expr>`
    // If so, extend back to include the full `let`.
    start = extendBackOverLetBinding(text, start);

    return new vscode.Range(
        document.positionAt(start),
        document.positionAt(end + 1),
    );
}

function extendBackOverLetBinding(text: string, exprStart: number): number {
    let pos = exprStart - 1;

    // Skip whitespace before the expression
    while (pos >= 0 && /\s/.test(text[pos])) { pos--; }
    if (pos < 0 || text[pos] !== '=') { return exprStart; }

    // Skip the '=' (but not '==' or '=>')
    if (pos > 0 && (text[pos - 1] === '=' || text[pos - 1] === '!' || text[pos - 1] === '<' || text[pos - 1] === '>')) {
        return exprStart; // This is ==, !=, <=, >= — not an assignment
    }
    if (pos + 1 < text.length && text[pos + 1] === '>') {
        return exprStart; // This is => (match arm)
    }
    pos--;

    // Skip whitespace before '='
    while (pos >= 0 && /\s/.test(text[pos])) { pos--; }
    if (pos < 0) { return exprStart; }

    // Read the binding name backwards
    if (!/[\w\-!?]/.test(text[pos])) { return exprStart; }
    let nameEnd = pos;
    while (pos > 0 && /[\w\-!?]/.test(text[pos - 1])) { pos--; }
    const name = text.substring(pos, nameEnd + 1);

    // Skip whitespace before name
    let kwPos = pos - 1;
    while (kwPos >= 0 && /\s/.test(text[kwPos])) { kwPos--; }

    // Check for 'let' or 'let mut' keyword
    if (kwPos >= 2) {
        // Check for 'mut' first (for 'let mut x = ...')
        if (kwPos >= 2 && text.substring(kwPos - 2, kwPos + 1) === 'mut') {
            let letPos = kwPos - 3;
            while (letPos >= 0 && /\s/.test(text[letPos])) { letPos--; }
            if (letPos >= 2 && text.substring(letPos - 2, letPos + 1) === 'let') {
                return letPos - 2;
            }
        }
        // Check for 'let'
        if (text.substring(kwPos - 2, kwPos + 1) === 'let') {
            return kwPos - 2;
        }
    }

    return exprStart;
}

// ============================================================================
// Top-level (outermost) expression detection
// ============================================================================

const TOP_LEVEL_KEYWORDS = new Set([
    'fn', 'let', 'struct', 'enum', 'namespace', 'use', 'protocol',
]);

function findTopLevelExpression(
    document: vscode.TextDocument,
    position: vscode.Position,
): vscode.Range | null {
    const text = document.getText();
    const cursorOffset = document.offsetAt(position);

    // Collect all top-level forms by forward-scanning.
    // A new top-level form starts when we see a top-level keyword at depth 0
    // at the beginning of a line (first non-whitespace).
    // A form ends when: (a) depth returns to 0 after `}`, or
    // (b) a new top-level keyword starts at depth 0.
    const forms: { start: number; end: number }[] = [];
    let depth = 0;
    let inString = false;
    let inLineComment = false;
    let formStart = -1;
    let atLineStart = true; // are we looking at the first token on a line?

    function endCurrentForm(endPos: number) {
        if (formStart >= 0) {
            // Trim trailing whitespace from the form
            let trimEnd = endPos;
            while (trimEnd > formStart && /\s/.test(text[trimEnd - 1])) { trimEnd--; }
            forms.push({ start: formStart, end: trimEnd });
            formStart = -1;
        }
    }

    for (let i = 0; i < text.length; i++) {
        const ch = text[i];

        if (ch === '\n') {
            inLineComment = false;
            atLineStart = true;
            continue;
        }

        if (inLineComment) { continue; }

        // Track whether we're at the first non-whitespace on a line
        if (atLineStart && /\s/.test(ch)) { continue; }
        const isFirstTokenOnLine = atLineStart;
        atLineStart = false;

        if (!inString && ch === '/' && i + 1 < text.length && text[i + 1] === '/') {
            inLineComment = true;
            continue;
        }

        // Handle strings
        if (ch === '"') {
            if (inString) {
                let bs = 0;
                let j = i - 1;
                while (j >= 0 && text[j] === '\\') { bs++; j--; }
                if (bs % 2 === 0) { inString = false; }
            } else {
                inString = true;
                if (depth === 0 && formStart === -1) { formStart = i; }
            }
            continue;
        }
        if (inString) { continue; }

        // At depth 0, check if a new top-level form starts
        if (depth === 0 && isFirstTokenOnLine && /[\w]/.test(ch)) {
            // Read the word at this position
            let wordEnd = i;
            while (wordEnd + 1 < text.length && /[\w\-]/.test(text[wordEnd + 1])) { wordEnd++; }
            const word = text.substring(i, wordEnd + 1);

            if (TOP_LEVEL_KEYWORDS.has(word)) {
                // New top-level form — end the previous one
                endCurrentForm(i);
                formStart = i;
            } else if (formStart === -1) {
                formStart = i;
            }
        } else if (depth === 0 && formStart === -1 && !/\s/.test(ch)) {
            formStart = i;
        }

        // Track brackets
        if (ch === '(' || ch === '{' || ch === '[') {
            depth++;
        } else if (ch === ')' || ch === '}' || ch === ']') {
            depth--;
            if (depth === 0 && formStart >= 0 && ch === '}') {
                // Closing brace at depth 0 ends the form
                endCurrentForm(i + 1);
            }
        }
    }

    // End any trailing form
    endCurrentForm(text.length);

    // Find the form containing the cursor
    for (const form of forms) {
        if (cursorOffset >= form.start && cursorOffset <= form.end) {
            return new vscode.Range(
                document.positionAt(form.start),
                document.positionAt(form.end),
            );
        }
    }

    return null;
}

function getConfig() {
    const config = vscode.workspace.getConfiguration('beagle');
    return {
        host: config.get<string>('repl.host', '127.0.0.1'),
        port: config.get<number>('repl.port', 7888),
    };
}

export function activate(context: vscode.ExtensionContext) {
    outputChannel = vscode.window.createOutputChannel('Beagle REPL');

    const evalCommand = vscode.commands.registerCommand('beagle-vs-code-eval.sendFile', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor.');
            return;
        }

        const text = editor.document.getText();
        const config = getConfig();

        try {
            if (!connection || !connection.isConnected()) {
                connection = new BeagleConnection(config.host, config.port);
                await connection.connect();
            }

            outputChannel.appendLine(`--- Eval: ${editor.document.fileName} ---`);
            const result = await connection.eval(text);

            if (result.output) {
                outputChannel.appendLine(result.output);
            }
            if (result.value !== null) {
                outputChannel.appendLine(`=> ${result.value}`);
            }
            if (result.error) {
                outputChannel.appendLine(`Error: ${result.error}`);
                vscode.window.showErrorMessage(`Beagle: ${result.error}`);
            }

            showResultInline(editor, editor.document.lineCount - 1, result);
            outputChannel.show(true); // Show but don't steal focus
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : String(err);
            vscode.window.showErrorMessage(`Failed to connect to Beagle REPL at ${config.host}:${config.port}: ${message}`);
        }
    });

    const evalSelectionCommand = vscode.commands.registerCommand('beagle-vs-code-eval.evalSelection', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor.');
            return;
        }

        const selection = editor.selection;
        let text: string;
        let displayLine: number;
        let exprRange: vscode.Range | null = null;

        if (!selection.isEmpty) {
            // Selection exists - eval the selection
            text = editor.document.getText(selection);
            displayLine = selection.end.line;
        } else {
            // No selection - find the expression before the cursor
            exprRange = findExpressionBeforeCursor(editor.document, selection.active);
            if (!exprRange) {
                vscode.window.showInformationMessage('No expression found before cursor.');
                return;
            }
            text = editor.document.getText(exprRange);
            displayLine = exprRange.end.line;
        }

        if (!text.trim()) {
            return;
        }

        // Briefly flash the expression range so the user sees what was eval'd
        if (exprRange) {
            const flashDecoration = vscode.window.createTextEditorDecorationType({
                backgroundColor: new vscode.ThemeColor('editor.wordHighlightBackground'),
            });
            editor.setDecorations(flashDecoration, [exprRange]);
            setTimeout(() => flashDecoration.dispose(), 300);
        }

        const config = getConfig();

        try {
            if (!connection || !connection.isConnected()) {
                connection = new BeagleConnection(config.host, config.port);
                await connection.connect();
            }

            outputChannel.appendLine(`--- Eval: ${text.length > 80 ? text.substring(0, 80) + '...' : text} ---`);
            const result = await connection.eval(text);

            if (result.output) {
                outputChannel.appendLine(result.output);
            }
            if (result.value !== null) {
                outputChannel.appendLine(`=> ${result.value}`);
            }
            if (result.error) {
                outputChannel.appendLine(`Error: ${result.error}`);
            }

            showResultInline(editor, displayLine, result);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : String(err);
            vscode.window.showErrorMessage(`Failed to connect to Beagle REPL at ${config.host}:${config.port}: ${message}`);
        }
    });

    const evalTopLevelCommand = vscode.commands.registerCommand('beagle-vs-code-eval.evalTopLevel', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor.');
            return;
        }

        const exprRange = findTopLevelExpression(editor.document, editor.selection.active);
        if (!exprRange) {
            vscode.window.showInformationMessage('No top-level expression found at cursor.');
            return;
        }

        const text = editor.document.getText(exprRange);
        if (!text.trim()) { return; }

        // Flash the expression
        const flashDecoration = vscode.window.createTextEditorDecorationType({
            backgroundColor: new vscode.ThemeColor('editor.wordHighlightBackground'),
        });
        editor.setDecorations(flashDecoration, [exprRange]);
        setTimeout(() => flashDecoration.dispose(), 300);

        const config = getConfig();

        try {
            if (!connection || !connection.isConnected()) {
                connection = new BeagleConnection(config.host, config.port);
                await connection.connect();
            }

            outputChannel.appendLine(`--- Eval top-level: ${text.length > 80 ? text.substring(0, 80) + '...' : text} ---`);
            const result = await connection.eval(text);

            if (result.output) {
                outputChannel.appendLine(result.output);
            }
            if (result.value !== null) {
                outputChannel.appendLine(`=> ${result.value}`);
            }
            if (result.error) {
                outputChannel.appendLine(`Error: ${result.error}`);
            }

            showResultInline(editor, exprRange.end.line, result);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : String(err);
            vscode.window.showErrorMessage(`Failed to connect to Beagle REPL at ${config.host}:${config.port}: ${message}`);
        }
    });

    let runTerminal: vscode.Terminal | null = null;

    const runCommand = vscode.commands.registerCommand('beagle-vs-code-eval.run', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor.');
            return;
        }

        const filePath = editor.document.fileName;
        const text = editor.document.getText();
        const config = getConfig();

        // Extract namespace from the file
        const nsMatch = text.match(/^\s*namespace\s+(\S+)/m);
        if (!nsMatch) {
            vscode.window.showErrorMessage('Could not find namespace declaration in file.');
            return;
        }
        const ns = nsMatch[1];

        // Generate a wrapper that starts REPL server then calls main.
        // The eval("namespace ...") at the end sets the compiler's current namespace
        // so that REPL evals resolve names in the target namespace by default.
        const wrapper = `namespace __repl_runner

use beagle.repl-main as repl-main
use ${ns} as target

fn main() {
    eval("namespace ${ns}")
    repl-main/run-with-repl("${config.host}", ${config.port}, fn() {
        target/main()
    })
}
`;
        const wrapperPath = path.join(os.tmpdir(), '__beagle_repl_runner.bg');
        fs.writeFileSync(wrapperPath, wrapper);

        const sourceDir = path.dirname(filePath);

        // Kill existing run terminal if any
        if (runTerminal) {
            runTerminal.dispose();
        }

        runTerminal = vscode.window.createTerminal({ name: 'Beagle' });
        runTerminal.show();
        runTerminal.sendText(`beag run -I ${sourceDir} ${wrapperPath}`);
    });

    const connectCommand = vscode.commands.registerCommand('beagle-vs-code-eval.connect', async () => {
        const config = getConfig();
        try {
            connection = new BeagleConnection(config.host, config.port);
            await connection.connect();
            vscode.window.showInformationMessage(`Connected to Beagle REPL at ${config.host}:${config.port}`);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : String(err);
            vscode.window.showErrorMessage(`Failed to connect: ${message}`);
        }
    });

    const disconnectCommand = vscode.commands.registerCommand('beagle-vs-code-eval.disconnect', () => {
        if (connection) {
            connection.disconnect();
            connection = null;
            outputChannel.appendLine('Disconnected from Beagle REPL');
            vscode.window.showInformationMessage('Disconnected from Beagle REPL');
        }
    });

    context.subscriptions.push(evalCommand, evalSelectionCommand, evalTopLevelCommand, runCommand, connectCommand, disconnectCommand, outputChannel);
}

export function deactivate() {
    if (connection) {
        connection.disconnect();
        connection = null;
    }
}
