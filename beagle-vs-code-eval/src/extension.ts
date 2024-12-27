import * as vscode from 'vscode';
import * as net from 'net';

export function activate(context: vscode.ExtensionContext) {
    const disposable = vscode.commands.registerCommand('beagle-vs-code-eval.sendFile', async () => {
        const editor = vscode.window.activeTextEditor;

        if (editor) {
            const document = editor.document;
            const text = document.getText();
            const position = editor.selection.active;

            // Create socket connection
            const socket = new net.Socket();
            const host = '127.0.0.1'; // Replace with your desired host
            const port = 12345; // Replace with your desired port

            try {
                socket.connect(port, host, () => {
                    // for right now, we are just going to send the whole file
                    // and now worry about trying to eval parts
                    const data = text;

                    // Send data over socket
                    socket.write(data);
                    vscode.window.showInformationMessage('File data sent successfully.');
                    socket.end();
                });

                socket.on('error', (err) => {
                    vscode.window.showErrorMessage(`Socket error: ${err.message}`);
                });
            } catch (err) {
                vscode.window.showErrorMessage(`Failed to send file data: ${err}`);
            }
        } else {
            vscode.window.showErrorMessage('No active editor.');
        }
    });

    context.subscriptions.push(disposable);
}

export function deactivate() {}