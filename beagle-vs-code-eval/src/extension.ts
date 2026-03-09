import * as vscode from 'vscode';
import * as net from 'net';

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
        const request = JSON.stringify({
            op: 'eval',
            id,
            session: this.sessionId,
            code,
        });

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
        const text = selection.isEmpty
            ? editor.document.lineAt(selection.active.line).text
            : editor.document.getText(selection);
        const displayLine = selection.isEmpty ? selection.active.line : selection.end.line;

        if (!text.trim()) {
            return;
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

    context.subscriptions.push(evalCommand, evalSelectionCommand, connectCommand, disconnectCommand, outputChannel);
}

export function deactivate() {
    if (connection) {
        connection.disconnect();
        connection = null;
    }
}
