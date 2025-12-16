import * as vscode from 'vscode';
import WebSocket from 'ws';
import * as path from 'path';

interface EvalResult {
  type: string;
  requestId: string;
  moduleId: string;
  success: boolean;
  value?: unknown;
  error?: string;
  exprType?: string;
}

class HotReloadConnection {
  private ws: WebSocket | null = null;
  private connected = false;
  private sourceRoot: string = '';
  private pendingEvals = new Map<string, (result: EvalResult) => void>();
  private requestCounter = 0;
  private outputChannel: vscode.OutputChannel;
  private statusBarItem: vscode.StatusBarItem;
  private reconnectTimer: NodeJS.Timeout | null = null;

  constructor() {
    this.outputChannel = vscode.window.createOutputChannel('Hot Reload');
    this.statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    this.statusBarItem.command = 'hotReload.connect';
    this.updateStatusBar();
  }

  private updateStatusBar() {
    if (this.connected) {
      this.statusBarItem.text = '$(zap) Hot';
      this.statusBarItem.tooltip = 'Hot Reload: Connected. Click to reconnect.';
      this.statusBarItem.backgroundColor = undefined;
    } else {
      this.statusBarItem.text = '$(debug-disconnect) Hot';
      this.statusBarItem.tooltip = 'Hot Reload: Disconnected. Click to connect.';
      this.statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
    }
    this.statusBarItem.show();
  }

  async connect(sourceRoot?: string): Promise<void> {
    if (this.ws) {
      this.disconnect();
    }

    // Clear any pending reconnect
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    const config = vscode.workspace.getConfiguration('hotReload');
    const host = config.get<string>('host', 'localhost');
    const port = config.get<number>('port', 3456);

    // Determine source root
    if (!sourceRoot) {
      const workspaceFolders = vscode.workspace.workspaceFolders;
      if (workspaceFolders && workspaceFolders.length > 0) {
        sourceRoot = workspaceFolders[0].uri.fsPath;
      } else {
        sourceRoot = '';
      }
    }
    this.sourceRoot = sourceRoot;

    const url = `ws://${host}:${port}`;
    this.outputChannel.appendLine(`[hot] Connecting to ${url}...`);

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(url);

        this.ws.on('open', () => {
          this.connected = true;
          this.updateStatusBar();

          // Identify as editor
          this.send({
            type: 'identify',
            clientType: 'editor'
          });

          vscode.window.showInformationMessage(`Hot Reload: Connected to ${url}`);
          this.outputChannel.appendLine(`[hot] Connected to ${url}`);
          resolve();
        });

        this.ws.on('message', (data: WebSocket.Data) => {
          try {
            const message = JSON.parse(data.toString());
            this.handleMessage(message);
          } catch (e) {
            // Ignore parse errors
          }
        });

        this.ws.on('close', () => {
          this.connected = false;
          this.updateStatusBar();
          this.outputChannel.appendLine('[hot] Disconnected');
        });

        this.ws.on('error', (err: Error) => {
          this.outputChannel.appendLine(`[hot] Error: ${err.message}`);
          if (!this.connected) {
            reject(err);
          }
        });

      } catch (e) {
        reject(e);
      }
    });
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.connected = false;
    this.updateStatusBar();
  }

  private send(message: object) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  private handleMessage(message: EvalResult) {
    if (message.type === 'eval-result') {
      const { requestId, success, value, error, moduleId, exprType } = message;

      // Resolve pending promise
      const callback = this.pendingEvals.get(requestId);
      if (callback) {
        this.pendingEvals.delete(requestId);
        callback(message);
      }

      // Display result
      if (success) {
        const resultStr = this.formatValue(value);
        this.outputChannel.appendLine(`[hot] ${moduleId} => ${resultStr}`);
        vscode.window.setStatusBarMessage(`[hot] => ${resultStr}`, 3000);

        // Show inline decoration for expressions
        if (exprType === 'expression') {
          this.showInlineResult(resultStr);
        }
      } else {
        this.outputChannel.appendLine(`[hot] Error in ${moduleId}: ${error}`);
        vscode.window.showErrorMessage(`Hot Reload Error: ${error}`);
      }
    }
  }

  private formatValue(value: unknown): string {
    if (value === null) return 'null';
    if (value === undefined) return 'undefined';
    if (typeof value === 'string') return value;
    if (typeof value === 'number' || typeof value === 'boolean') {
      return String(value);
    }
    return JSON.stringify(value);
  }

  private showInlineResult(result: string) {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const decorationType = vscode.window.createTextEditorDecorationType({
      after: {
        contentText: ` => ${result}`,
        color: new vscode.ThemeColor('editorCodeLens.foreground'),
        fontStyle: 'italic'
      }
    });

    const position = editor.selection.active;
    const line = editor.document.lineAt(position.line);
    const range = new vscode.Range(line.range.end, line.range.end);

    editor.setDecorations(decorationType, [{ range }]);

    // Remove after 3 seconds
    setTimeout(() => {
      decorationType.dispose();
    }, 3000);
  }

  private generateRequestId(): string {
    this.requestCounter++;
    return `vscode-${process.pid}-${this.requestCounter}`;
  }

  getModuleId(filePath: string): string {
    if (this.sourceRoot && filePath) {
      try {
        return path.relative(this.sourceRoot, filePath);
      } catch {
        // Fall through
      }
    }
    return path.basename(filePath);
  }

  async evalExpr(expr: string, moduleId: string): Promise<EvalResult> {
    if (!this.connected || !this.ws) {
      throw new Error('Not connected. Run "Hot Reload: Connect" first.');
    }

    const requestId = this.generateRequestId();

    return new Promise((resolve, reject) => {
      // Set timeout for response
      const timeout = setTimeout(() => {
        this.pendingEvals.delete(requestId);
        reject(new Error('Eval request timed out'));
      }, 30000);

      this.pendingEvals.set(requestId, (result) => {
        clearTimeout(timeout);
        resolve(result);
      });

      this.send({
        type: 'eval-request',
        moduleId,
        expr,
        requestId
      });

      this.outputChannel.appendLine(`[hot] Evaluating in ${moduleId}...`);
      vscode.window.setStatusBarMessage(`[hot] Evaluating...`, 1000);
    });
  }

  dispose() {
    this.disconnect();
    this.outputChannel.dispose();
    this.statusBarItem.dispose();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
  }
}

// Global connection
let connection: HotReloadConnection;

export function activate(context: vscode.ExtensionContext) {
  connection = new HotReloadConnection();

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand('hotReload.connect', async () => {
      const workspaceFolders = vscode.workspace.workspaceFolders;
      let sourceRoot = workspaceFolders?.[0]?.uri.fsPath || '';

      // Ask user to confirm/change source root
      const result = await vscode.window.showInputBox({
        prompt: 'Source root directory',
        value: sourceRoot,
        validateInput: (value) => {
          return value ? null : 'Source root is required';
        }
      });

      if (result) {
        try {
          await connection.connect(result);
        } catch (e) {
          vscode.window.showErrorMessage(`Failed to connect: ${(e as Error).message}`);
        }
      }
    }),

    vscode.commands.registerCommand('hotReload.disconnect', () => {
      connection.disconnect();
      vscode.window.showInformationMessage('Hot Reload: Disconnected');
    }),

    vscode.commands.registerCommand('hotReload.evalSelection', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;

      let expr = editor.document.getText(editor.selection);
      if (!expr.trim()) {
        // No selection - try to expand
        const wordRange = editor.document.getWordRangeAtPosition(editor.selection.active);
        if (wordRange) {
          expr = editor.document.getText(wordRange);
        }
      }

      if (!expr.trim()) {
        vscode.window.showWarningMessage('No code selected');
        return;
      }

      const moduleId = connection.getModuleId(editor.document.fileName);

      try {
        await connection.evalExpr(expr, moduleId);
      } catch (e) {
        vscode.window.showErrorMessage(`Eval failed: ${(e as Error).message}`);
      }
    }),

    vscode.commands.registerCommand('hotReload.evalDefun', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;

      const position = editor.selection.active;
      const expr = findDefunAtPosition(editor.document, position);

      if (!expr) {
        vscode.window.showWarningMessage('No function/class found at cursor');
        return;
      }

      const moduleId = connection.getModuleId(editor.document.fileName);

      try {
        await connection.evalExpr(expr, moduleId);
      } catch (e) {
        vscode.window.showErrorMessage(`Eval failed: ${(e as Error).message}`);
      }
    }),

    vscode.commands.registerCommand('hotReload.evalBuffer', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) return;

      const expr = editor.document.getText();
      const moduleId = connection.getModuleId(editor.document.fileName);

      try {
        await connection.evalExpr(expr, moduleId);
      } catch (e) {
        vscode.window.showErrorMessage(`Eval failed: ${(e as Error).message}`);
      }
    }),

    vscode.commands.registerCommand('hotReload.evalExpression', async () => {
      const editor = vscode.window.activeTextEditor;
      const moduleId = editor ? connection.getModuleId(editor.document.fileName) : '';

      const expr = await vscode.window.showInputBox({
        prompt: `Eval in ${moduleId}`,
        placeHolder: 'Enter JavaScript expression'
      });

      if (expr && expr.trim()) {
        try {
          await connection.evalExpr(expr, moduleId);
        } catch (e) {
          vscode.window.showErrorMessage(`Eval failed: ${(e as Error).message}`);
        }
      }
    })
  );

  // Auto-connect if configured
  const config = vscode.workspace.getConfiguration('hotReload');
  if (config.get<boolean>('autoConnect')) {
    connection.connect().catch(() => {
      // Silently fail on auto-connect
    });
  }

  context.subscriptions.push(connection);
}

function findDefunAtPosition(document: vscode.TextDocument, position: vscode.Position): string | null {
  const text = document.getText();

  // Patterns for top-level declarations
  const patterns = [
    // function declarations (including async)
    /^(export\s+)?(async\s+)?function\s+\w+\s*\([^)]*\)\s*\{/gm,
    // class declarations
    /^(export\s+)?class\s+\w+(\s+extends\s+\w+)?\s*\{/gm,
    // arrow functions and function expressions assigned to variables
    /^(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?(\([^)]*\)|[^=])\s*=>/gm,
    /^(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?function/gm,
    // export default
    /^export\s+default\s+(async\s+)?function/gm,
    /^export\s+default\s+class/gm,
  ];

  // Find all matches and their positions
  interface Match {
    start: number;
    index: number;
  }

  const matches: Match[] = [];
  for (const pattern of patterns) {
    let match;
    while ((match = pattern.exec(text)) !== null) {
      matches.push({
        start: match.index,
        index: match.index
      });
    }
  }

  // Sort by position
  matches.sort((a, b) => a.start - b.start);

  // Convert cursor position to offset
  const cursorOffset = document.offsetAt(position);

  // Find which declaration contains our cursor
  let currentMatch: Match | null = null;
  for (const match of matches) {
    if (match.start > cursorOffset) {
      break;
    }
    currentMatch = match;
  }

  if (!currentMatch) {
    return null;
  }

  // Find the end by counting braces
  let braceCount = 0;
  let inString = false;
  let stringChar = '';
  let pos = currentMatch.start;
  let started = false;

  while (pos < text.length) {
    const char = text[pos];

    // Handle string literals
    if ((char === '"' || char === "'" || char === '`') && (pos === 0 || text[pos - 1] !== '\\')) {
      if (!inString) {
        inString = true;
        stringChar = char;
      } else if (char === stringChar) {
        inString = false;
        stringChar = '';
      }
    } else if (!inString) {
      if (char === '{') {
        braceCount++;
        started = true;
      } else if (char === '}') {
        braceCount--;
        if (started && braceCount === 0) {
          return text.substring(currentMatch.start, pos + 1);
        }
      }
    }

    pos++;
  }

  // Fallback: return from start to end of line
  const startPos = document.positionAt(currentMatch.start);
  const endLine = document.lineAt(startPos.line);
  return document.getText(new vscode.Range(startPos, endLine.range.end));
}

export function deactivate() {
  if (connection) {
    connection.dispose();
  }
}
