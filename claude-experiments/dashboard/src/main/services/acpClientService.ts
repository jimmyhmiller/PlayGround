/**
 * ACP Client Service
 *
 * Manages the connection to claude-code-acp via the Agent Client Protocol.
 * Spawns the agent as a subprocess and implements the ACP Client interface.
 */

import { spawn, ChildProcess } from 'child_process';
import { Readable, Writable, PassThrough } from 'stream';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import * as readline from 'readline';

// Get the path to the MCP server (relative to this file in src/main/services/)
function getMcpServerPath(): string {
  // In development, __dirname points to src/main/services/
  // MCP server is at src/main/mcp/server.js
  return path.resolve(__dirname, '../mcp/server.js');
}

// Event emitter interface
interface EventEmitter {
  emit(type: string, payload: unknown): void;
}

// Terminal tracking
interface ManagedTerminal {
  id: string;
  process: ChildProcess;
  output: string;
  exitCode: number | null;
  exitSignal: string | null;
  exited: boolean;
  exitPromise: Promise<void>;
  exitResolve: () => void;
}

// ACP SDK types (loaded dynamically)
type ACPModule = typeof import('@agentclientprotocol/sdk');
type ClientSideConnection = InstanceType<ACPModule['ClientSideConnection']>;
type Client = import('@agentclientprotocol/sdk').Client;
type Agent = import('@agentclientprotocol/sdk').Agent;
type SessionNotification = import('@agentclientprotocol/sdk').SessionNotification;
type RequestPermissionRequest = import('@agentclientprotocol/sdk').RequestPermissionRequest;
type RequestPermissionResponse = import('@agentclientprotocol/sdk').RequestPermissionResponse;
type ContentBlock = import('@agentclientprotocol/sdk').ContentBlock;

// Permission request callback
type PermissionCallback = (
  request: RequestPermissionRequest
) => Promise<RequestPermissionResponse>;

// Cached SDK module
let acpModule: ACPModule | null = null;

async function getACPModule(): Promise<ACPModule> {
  if (!acpModule) {
    acpModule = await import('@agentclientprotocol/sdk');
  }
  return acpModule;
}

// Debug logging for ACP messages - toggle with ACP_DEBUG=0 env var to disable or setAcpDebug(false)
let acpDebugEnabled = process.env.ACP_DEBUG !== '0';

export function setAcpDebug(enabled: boolean): void {
  acpDebugEnabled = enabled;
  console.log(`[acp] Debug logging ${enabled ? 'enabled' : 'disabled'}`);
}

export function isAcpDebugEnabled(): boolean {
  return acpDebugEnabled;
}

/**
 * ACP Client Service
 */
export class ACPClientService {
  private events: EventEmitter;
  private agentProcess: ChildProcess | null = null;
  private connection: ClientSideConnection | null = null;
  private agent: Agent | null = null;
  private terminals: Map<string, ManagedTerminal> = new Map();
  private terminalIdCounter = 0;
  private permissionCallback: PermissionCallback | null = null;
  private isInitialized = false;
  private spawnPromise: Promise<void> | null = null;
  private initPromise: Promise<void> | null = null;
  private currentSessionId: string | null = null;
  private sessionPromise: Promise<{ sessionId: string; modes?: { availableModes: Array<{ id: string; name: string }>; currentModeId: string } }> | null = null;
  private stderrBuffer: string = '';
  private agentInfo: { name: string; title?: string; version?: string } | null = null;
  // Track cwd for each session so we can use it for terminal commands
  private sessionCwdMap: Map<string, string> = new Map();

  constructor(events: EventEmitter) {
    this.events = events;
  }

  /**
   * Check if the connected agent is Claude Code
   */
  isClaudeCode(): boolean {
    return this.agentInfo?.name === '@zed-industries/claude-code-acp' ||
           this.agentInfo?.title === 'Claude Code';
  }

  /**
   * Set the callback for permission requests
   */
  setPermissionCallback(callback: PermissionCallback): void {
    this.permissionCallback = callback;
  }

  private spawnCwd: string | null = null;

  /**
   * Spawn the claude-code-acp agent and establish connection
   * @param cwd - Working directory to spawn the agent in
   */
  async spawn(cwd?: string): Promise<void> {
    console.log('[acp] spawn() called with cwd:', cwd, 'current spawnCwd:', this.spawnCwd, 'isInitialized:', this.isInitialized);
    
    // Already connected - but check if we need to respawn in a different directory
    if (this.isInitialized && this.connection) {
      console.log('[acp] Already initialized, checking if respawn needed');
      if (cwd && this.spawnCwd !== cwd) {
        console.log('[acp] RESPAWNING in new directory:', cwd, '(was:', this.spawnCwd, ')');
        // Kill and wait for cleanup
        if (this.agentProcess) {
          this.agentProcess.kill();
          await new Promise(resolve => setTimeout(resolve, 500));
        }
        this.agentProcess = null;
        this.connection = null;
        this.agent = null;
        this.isInitialized = false;
        this.currentSessionId = null;
        this.agentInfo = null;
        this.spawnCwd = null;
      } else {
        console.log('[acp] Already connected in correct directory:', this.spawnCwd, '- SKIPPING SPAWN');
        return;
      }
    }

    // Already spawning - wait for that to complete
    if (this.spawnPromise) {
      console.log('[acp] Already spawning, waiting for existing spawn...');
      await this.spawnPromise;
      return;
    }

    // Clean up any existing process
    if (this.agentProcess) {
      console.log('[acp] Cleaning up existing agent process');
      this.agentProcess.kill();
      // Wait for the process to actually exit before proceeding
      await new Promise(resolve => setTimeout(resolve, 300));
      this.agentProcess = null;
      this.connection = null;
      this.agent = null;
      this.isInitialized = false;
      this.currentSessionId = null;
      this.agentInfo = null;
    }

    // Store the cwd we're spawning with
    this.spawnCwd = cwd || null;

    // Create spawn promise before any async work
    this.spawnPromise = this.doSpawn(cwd);

    try {
      await this.spawnPromise;
    } finally {
      this.spawnPromise = null;
    }
  }

  /**
   * Internal spawn implementation
   */
  private async doSpawn(cwd?: string): Promise<void> {
    const acp = await getACPModule();

    // Spawn claude-code-acp with auto-approve permissions
    // Capture stderr to detect session errors
    this.stderrBuffer = '';
    const spawnDir = cwd || process.cwd();
    console.log('[acp] Spawning agent in cwd:', spawnDir);
    console.log('[acp] Current process.cwd():', process.cwd());
    console.log('[acp] ENV PWD before:', process.env.PWD);
    
    // Use direct path to claude-code-acp instead of npx to ensure cwd is honored
    this.agentProcess = spawn('claude-code-acp', ['--dangerously-skip-permissions'], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: spawnDir,
      env: {
        ...process.env,
        PWD: spawnDir,  // Also set PWD env var
      },
    });
    
    console.log('[acp] Agent process spawned, pid:', this.agentProcess.pid);

    if (!this.agentProcess.stdin || !this.agentProcess.stdout || !this.agentProcess.stderr) {
      throw new Error('Failed to get stdio pipes from agent process');
    }

    // Monitor stderr for errors and also log to console
    this.agentProcess.stderr.on('data', (data: Buffer) => {
      const text = data.toString();
      this.stderrBuffer += text;
      // Also print to console so we can see it
      process.stderr.write(text);
    });

    // Create PassThrough for stdout to enable logging while still passing to SDK
    const stdoutPassThrough = new PassThrough();
    this.agentProcess.stdout.pipe(stdoutPassThrough);

    // Log incoming messages (from agent) when debug enabled
    this.agentProcess.stdout.on('data', (data: Buffer) => {
      if (acpDebugEnabled) {
        const text = data.toString();
        for (const line of text.split('\n')) {
          if (line.trim()) {
            try {
              const msg = JSON.parse(line);
              console.log('[acp] <<< RECV:', JSON.stringify(msg, null, 2));
            } catch {
              console.log('[acp] <<< RECV (raw):', line);
            }
          }
        }
      }
    });

    // Wrap stdin to log outgoing messages when debug enabled
    const originalStdin = this.agentProcess.stdin;
    const loggingStdin = new Writable({
      write(chunk: Buffer, encoding: BufferEncoding, callback: (error?: Error | null) => void) {
        if (acpDebugEnabled) {
          const text = chunk.toString();
          for (const line of text.split('\n')) {
            if (line.trim()) {
              try {
                const msg = JSON.parse(line);
                console.log('[acp] >>> SEND:', JSON.stringify(msg, null, 2));
              } catch {
                console.log('[acp] >>> SEND (raw):', line);
              }
            }
          }
        }
        originalStdin.write(chunk, encoding, callback);
      },
    });

    // Convert Node streams to Web streams for the SDK
    const input = Writable.toWeb(loggingStdin) as WritableStream<Uint8Array>;
    const output = Readable.toWeb(stdoutPassThrough) as ReadableStream<Uint8Array>;

    // Create the ndjson stream
    const stream = acp.ndJsonStream(input, output);

    // Create the client-side connection
    const client = this.createClient();
    this.connection = new acp.ClientSideConnection((_agent: Agent) => {
      this.agent = _agent;
      return client;
    }, stream);

    // Handle connection close
    this.connection.closed.then(() => {
      console.log('[acp] Connection closed');
      this.events.emit('acp.connection.closed', {});
      this.isInitialized = false;
    });

    // Track if process has exited for early detection
    let processExited = false;
    let processExitCode: number | null = null;
    let processExitSignal: string | null = null;

    // Handle process exit
    this.agentProcess.on('exit', (code, signal) => {
      console.log(`[acp] Agent process exited with code ${code}, signal ${signal}`);
      processExited = true;
      processExitCode = code;
      processExitSignal = signal;
      this.agentProcess = null;
      this.connection = null;
      this.agent = null;
      this.isInitialized = false;
      this.currentSessionId = null;
      this.agentInfo = null;
      this.events.emit('acp.process.exit', { code, signal });
    });

    // Wait a moment to verify the process is still running
    // This catches immediate exits (SIGTERM, command not found, etc.)
    await new Promise(resolve => setTimeout(resolve, 200));

    if (processExited) {
      const exitInfo = processExitSignal ? `signal ${processExitSignal}` : `code ${processExitCode}`;
      throw new Error(`Agent process exited immediately with ${exitInfo}. Check stderr for details.`);
    }

    this.agentProcess.on('error', (err) => {
      console.error('[acp] Agent process error:', err);
      this.events.emit('acp.process.error', { error: err.message });
    });

    console.log('[acp] Agent process spawned');
  }

  /**
   * Initialize the ACP connection
   */
  async initialize(): Promise<void> {
    if (!this.connection) {
      // Check if process exists to give better error message
      if (!this.agentProcess) {
        throw new Error('Agent process not running. The process may have exited immediately after spawn. Check logs for details.');
      }
      throw new Error('Connection not established. Call spawn() first.');
    }

    if (this.isInitialized) {
      console.log('[acp] Already initialized');
      return;
    }

    // Already initializing - wait for that to complete
    if (this.initPromise) {
      console.log('[acp] Already initializing, waiting...');
      await this.initPromise;
      return;
    }

    this.initPromise = this.doInitialize();

    try {
      await this.initPromise;
    } finally {
      this.initPromise = null;
    }
  }

  /**
   * Internal initialize implementation
   */
  private async doInitialize(): Promise<void> {
    const acp = await getACPModule();

    const initParams = {
      protocolVersion: acp.PROTOCOL_VERSION,
      clientCapabilities: {
        fs: {
          readTextFile: true,
          writeTextFile: true,
        },
        terminal: true,
      },
      clientInfo: {
        name: 'dashboard',
        version: '1.0.0',
      },
    };

    const result = await this.connection!.initialize(initParams);

    console.log('[acp] Initialized:', result);
    this.isInitialized = true;

    // Store agent info for later checks
    if (result.agentInfo) {
      this.agentInfo = result.agentInfo as { name: string; title?: string; version?: string };
      console.log('[acp] Agent:', this.agentInfo.name, this.agentInfo.version);
    }

    this.events.emit('acp.initialized', result);
  }

  // Session response type with modes
  private sessionModes: { availableModes: Array<{ id: string; name: string }>; currentModeId: string } | null = null;

  /**
   * Create a new session (or return existing one if already created)
   * @param cwd - Working directory for the session (defaults to spawn directory or process.cwd())
   * @param mcpServers - Optional MCP servers to connect
   * @param force - If true, always creates a new session even if one exists
   */
  async newSession(cwd?: string, mcpServers?: unknown[], force?: boolean): Promise<{
    sessionId: string;
    modes?: { availableModes: Array<{ id: string; name: string }>; currentModeId: string };
  }> {
    if (!this.connection || !this.isInitialized) {
      throw new Error('Not initialized. Call spawn() and initialize() first.');
    }

    // Use provided cwd, or fall back to spawn directory, or process.cwd()
    const effectiveCwd = cwd || this.spawnCwd || process.cwd();
    console.log('[acp] newSession cwd:', effectiveCwd, '(provided:', cwd, ', spawnCwd:', this.spawnCwd, ')');

    // If we already have a session and not forcing, return it with cached modes
    if (this.currentSessionId && !force) {
      console.log('[acp] Returning existing session:', this.currentSessionId);
      return { sessionId: this.currentSessionId, modes: this.sessionModes ?? undefined };
    }

    // If session creation is in progress, wait for it
    if (this.sessionPromise) {
      console.log('[acp] Session creation in progress, waiting...');
      return await this.sessionPromise;
    }

    // Create session with locking
    this.sessionPromise = this.doNewSession(effectiveCwd, mcpServers);

    try {
      return await this.sessionPromise;
    } finally {
      this.sessionPromise = null;
    }
  }

  /**
   * Internal session creation
   */
  private async doNewSession(cwd: string, mcpServers?: unknown[]): Promise<{
    sessionId: string;
    modes?: { availableModes: Array<{ id: string; name: string }>; currentModeId: string };
  }> {
    // Build MCP server list, including our dashboard MCP server
    const dashboardMcpServer = {
      name: 'dashboard',
      command: 'node',
      args: [getMcpServerPath()],
      env: [],  // Required by ACP schema
    };

    // Combine user-provided MCP servers with our dashboard server
    const allMcpServers = [
      dashboardMcpServer,
      ...(mcpServers ?? []),
    ];

    console.log('[acp] Creating session with MCP servers:', allMcpServers.map((s: { name?: string }) => s.name));

    const result = await this.connection!.newSession({
      cwd,
      mcpServers: allMcpServers as never[],
    });

    console.log('[acp] New session:', result.sessionId);
    console.log('[acp] Available modes:', result.modes);
    this.currentSessionId = result.sessionId;
    // Store the session's cwd so we can use it for terminal commands
    this.sessionCwdMap.set(result.sessionId, cwd);
    console.log('[acp] Stored session cwd:', result.sessionId, '->', cwd);
    this.sessionModes = result.modes as { availableModes: Array<{ id: string; name: string }>; currentModeId: string } | null;
    this.events.emit('acp.session.new', {
      sessionId: result.sessionId,
      modes: result.modes,
    });

    return {
      sessionId: result.sessionId,
      modes: this.sessionModes ?? undefined,
    };
  }

  /**
   * Resume an existing session (uses unstable_resumeSession)
   * @param sessionId - The session ID to resume
   * @param cwd - Working directory for the session (defaults to spawn directory or process.cwd())
   */
  async resumeSession(sessionId: string, cwd?: string): Promise<{
    sessionId: string;
    modes?: { availableModes: Array<{ id: string; name: string }>; currentModeId: string };
  }> {
    if (!this.connection || !this.isInitialized) {
      throw new Error('Not initialized');
    }

    // Use provided cwd, or fall back to spawn directory, or process.cwd()
    const effectiveCwd = cwd || this.spawnCwd || process.cwd();
    console.log('[acp] resumeSession cwd:', effectiveCwd, '(provided:', cwd, ', spawnCwd:', this.spawnCwd, ')');

    console.log('[acp] Attempting to resume session:', sessionId);

    // Clear stderr buffer before resume to catch any new errors
    this.stderrBuffer = '';

    // Include dashboard MCP server, same as newSession
    const dashboardMcpServer = {
      name: 'dashboard',
      command: 'node',
      args: [getMcpServerPath()],
      env: [],
    };

    const result = await this.connection.unstable_resumeSession({
      sessionId,
      cwd: effectiveCwd,
      mcpServers: [dashboardMcpServer] as never[],
    });

    // Wait a moment for stderr to be processed
    await new Promise(resolve => setTimeout(resolve, 100));

    // Check if stderr contains "No conversation found" - this means the session doesn't exist
    if (this.stderrBuffer.includes('No conversation found')) {
      console.log('[acp] Session not found on disk, throwing error');
      throw new Error(`Session not found: ${sessionId}`);
    }

    // Double-check we're still connected after resume
    if (!this.isConnected()) {
      throw new Error('Connection lost after resume - session may not exist');
    }

    console.log('[acp] Resume session result:', result);
    this.currentSessionId = sessionId;
    // Store the session's cwd so we can use it for terminal commands
    this.sessionCwdMap.set(sessionId, effectiveCwd);
    console.log('[acp] Stored session cwd:', sessionId, '->', effectiveCwd);
    this.sessionModes = result.modes as { availableModes: Array<{ id: string; name: string }>; currentModeId: string } | null;
    this.events.emit('acp.session.resumed', { sessionId, modes: result.modes });

    return {
      sessionId,
      modes: this.sessionModes ?? undefined,
    };
  }

  /**
   * Send a prompt to the agent
   */
  async prompt(
    sessionId: string,
    content: string | ContentBlock[]
  ): Promise<{ stopReason: string }> {
    if (!this.connection || !this.isInitialized) {
      throw new Error('Not initialized');
    }

    // Convert string to ContentBlock array if needed
    const promptContent: ContentBlock[] =
      typeof content === 'string'
        ? [{ type: 'text', text: content }]
        : content;

    const result = await this.connection.prompt({
      sessionId,
      prompt: promptContent,
    });

    console.log('[acp] Prompt completed:', result.stopReason);
    return { stopReason: result.stopReason };
  }

  /**
   * Cancel an ongoing prompt
   */
  async cancel(sessionId: string): Promise<void> {
    if (!this.connection) {
      console.log('[acp] Cannot cancel - not connected');
      return; // Silently return rather than throw - cancel is best-effort
    }

    try {
      await this.connection.cancel({ sessionId });
      console.log('[acp] Cancelled session:', sessionId);
    } catch (err) {
      // Cancel errors are usually not critical (session may have already ended)
      console.log('[acp] Cancel failed (session may not exist):', (err as Error).message);
    }
  }

  /**
   * Set session mode (e.g., 'plan', 'act')
   */
  async setMode(sessionId: string, modeId: string): Promise<void> {
    if (!this.connection || !this.isInitialized) {
      throw new Error('Not initialized');
    }

    await this.connection.setSessionMode({
      sessionId,
      modeId,
    });

    console.log('[acp] Set mode:', modeId);
    this.events.emit('acp.session.modeChanged', { sessionId, modeId });
  }

  /**
   * Shutdown the agent connection
   */
  async shutdown(): Promise<void> {
    // Kill all managed terminals
    for (const terminal of this.terminals.values()) {
      if (!terminal.exited) {
        terminal.process.kill();
      }
    }
    this.terminals.clear();

    // Kill the agent process
    if (this.agentProcess) {
      this.agentProcess.kill();
      this.agentProcess = null;
    }

    this.connection = null;
    this.agent = null;
    this.isInitialized = false;
    this.currentSessionId = null;
    this.agentInfo = null;

    console.log('[acp] Shutdown complete');
  }

  /**
   * Check if the service is connected and initialized
   */
  isConnected(): boolean {
    return this.isInitialized && this.connection !== null;
  }

  /**
   * Get the path to a session file on disk
   */
  private getSessionFilePath(sessionId: string, cwd: string): string {
    // Claude Code stores sessions in ~/.claude/projects/<encoded-cwd>/<sessionId>.jsonl
    const homeDir = os.homedir();
    const encodedCwd = cwd.replace(/\//g, '-');
    return path.join(homeDir, '.claude', 'projects', encodedCwd, `${sessionId}.jsonl`);
  }

  /**
   * Load conversation history from a session file
   * Returns messages in the format expected by ChatWidget
   * Note: Only works with Claude Code agent - other agents may store sessions differently
   * @param sessionId - The session ID to load
   * @param cwd - Working directory (defaults to spawn directory or process.cwd())
   */
  async loadSessionHistory(sessionId: string, cwd?: string): Promise<Array<{
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: number;
  }>> {
    // Only Claude Code stores sessions in ~/.claude/projects/
    if (!this.isClaudeCode()) {
      console.log('[acp] Session history loading only supported for Claude Code agent');
      return [];
    }

    // Use provided cwd, or fall back to spawn directory, or process.cwd()
    const effectiveCwd = cwd || this.spawnCwd || process.cwd();
    const sessionPath = this.getSessionFilePath(sessionId, effectiveCwd);

    try {
      await fs.access(sessionPath);
    } catch {
      console.log('[acp] Session file not found:', sessionPath);
      return [];
    }

    const messages: Array<{
      id: string;
      role: 'user' | 'assistant';
      content: string;
      timestamp: number;
    }> = [];

    // Read file line by line (JSONL format)
    const fileHandle = await fs.open(sessionPath, 'r');
    const rl = readline.createInterface({
      input: fileHandle.createReadStream(),
      crlfDelay: Infinity,
    });

    for await (const line of rl) {
      if (!line.trim()) continue;

      try {
        const entry = JSON.parse(line);

        // Only process user and assistant messages
        if (entry.type === 'user' && entry.message) {
          const textContent = entry.message.content
            ?.filter((c: { type: string }) => c.type === 'text')
            .map((c: { text: string }) => c.text)
            .join('') || '';

          if (textContent) {
            messages.push({
              id: entry.uuid || `msg-${Date.now()}-${Math.random().toString(36).slice(2)}`,
              role: 'user',
              content: textContent,
              timestamp: entry.timestamp ? new Date(entry.timestamp).getTime() : Date.now(),
            });
          }
        } else if (entry.type === 'assistant' && entry.message) {
          const textContent = entry.message.content
            ?.filter((c: { type: string }) => c.type === 'text')
            .map((c: { text: string }) => c.text)
            .join('') || '';

          if (textContent) {
            messages.push({
              id: entry.uuid || `msg-${Date.now()}-${Math.random().toString(36).slice(2)}`,
              role: 'assistant',
              content: textContent,
              timestamp: entry.timestamp ? new Date(entry.timestamp).getTime() : Date.now(),
            });
          }
        }
      } catch (parseErr) {
        // Skip malformed lines
        console.warn('[acp] Failed to parse session line:', parseErr);
      }
    }

    await fileHandle.close();
    console.log('[acp] Loaded', messages.length, 'messages from session file');
    return messages;
  }

  /**
   * Create the Client implementation for the ACP connection
   */
  private createClient(): Client {
    return {
      // Handle session updates from the agent
      sessionUpdate: async (params: SessionNotification): Promise<void> => {
        // Forward all updates to the event system
        this.events.emit('acp.session.update', params);

        // Also emit specific event types for convenience
        const updateType = (params.update as { sessionUpdate: string }).sessionUpdate;
        this.events.emit(`acp.session.${updateType}`, {
          sessionId: params.sessionId,
          update: params.update,
        });
      },

      // Handle permission requests
      requestPermission: async (
        params: RequestPermissionRequest
      ): Promise<RequestPermissionResponse> => {
        console.log('[acp] Permission request:', params);

        // Emit event for UI to handle
        this.events.emit('acp.permission.request', params);

        // If we have a callback, use it
        if (this.permissionCallback) {
          return await this.permissionCallback(params);
        }

        // Default: cancel if no callback set
        return { outcome: { outcome: 'cancelled' } };
      },

      // File system: read
      readTextFile: async (
        params: { path: string }
      ): Promise<{ content: string }> => {
        try {
          const content = await fs.readFile(params.path, 'utf-8');
          return { content };
        } catch (err) {
          const error = err as NodeJS.ErrnoException;
          if (error.code === 'ENOENT') {
            throw new Error(`File not found: ${params.path}`);
          }
          throw error;
        }
      },

      // File system: write
      writeTextFile: async (
        params: { path: string; content: string }
      ): Promise<Record<string, never>> => {
        // Ensure directory exists
        const dir = path.dirname(params.path);
        await fs.mkdir(dir, { recursive: true });

        await fs.writeFile(params.path, params.content, 'utf-8');
        return {};
      },

      // Terminal: create
      createTerminal: async (
        params: { command: string; args?: string[]; cwd?: string; env?: Record<string, string>; sessionId?: string }
      ): Promise<{ terminalId: string }> => {
        const id = `terminal-${++this.terminalIdCounter}`;

        // Determine the working directory:
        // 1. Use explicit cwd from params if provided
        // 2. Fall back to session's cwd if sessionId is provided
        // 3. Fall back to spawnCwd
        // 4. Fall back to process.cwd()
        let effectiveCwd = params.cwd;
        if (!effectiveCwd && params.sessionId) {
          effectiveCwd = this.sessionCwdMap.get(params.sessionId);
          console.log('[acp] Terminal using session cwd:', params.sessionId, '->', effectiveCwd);
        }
        if (!effectiveCwd) {
          effectiveCwd = this.spawnCwd || process.cwd();
          console.log('[acp] Terminal falling back to spawnCwd:', effectiveCwd);
        }

        // Create exit promise
        let exitResolve: () => void = () => {};
        const exitPromise = new Promise<void>((resolve) => {
          exitResolve = resolve;
        });

        const terminal: ManagedTerminal = {
          id,
          process: spawn(params.command, params.args ?? [], {
            cwd: effectiveCwd,
            shell: true,
            stdio: ['ignore', 'pipe', 'pipe'],
            env: { ...process.env, ...params.env },
          }),
          output: '',
          exitCode: null,
          exitSignal: null,
          exited: false,
          exitPromise,
          exitResolve,
        };

        // Collect output
        terminal.process.stdout?.on('data', (data: Buffer) => {
          terminal.output += data.toString();
          this.events.emit('acp.terminal.output', {
            id,
            data: data.toString(),
          });
        });

        terminal.process.stderr?.on('data', (data: Buffer) => {
          terminal.output += data.toString();
          this.events.emit('acp.terminal.output', {
            id,
            data: data.toString(),
          });
        });

        terminal.process.on('exit', (code, signal) => {
          terminal.exitCode = code;
          terminal.exitSignal = signal;
          terminal.exited = true;
          terminal.exitResolve();
          this.events.emit('acp.terminal.exit', { id, code, signal });
        });

        terminal.process.on('error', (err) => {
          terminal.exited = true;
          terminal.exitResolve();
          this.events.emit('acp.terminal.error', { id, error: err.message });
        });

        this.terminals.set(id, terminal);

        return { terminalId: id };
      },

      // Terminal: get output
      terminalOutput: async (
        params: { terminalId: string }
      ): Promise<{ output: string; exitStatus?: { exitCode: number | null; signal: string | null } }> => {
        const terminal = this.terminals.get(params.terminalId);
        if (!terminal) {
          throw new Error(`Terminal not found: ${params.terminalId}`);
        }

        return {
          output: terminal.output,
          exitStatus: terminal.exited
            ? {
                exitCode: terminal.exitCode,
                signal: terminal.exitSignal,
              }
            : undefined,
        };
      },

      // Terminal: wait for exit
      waitForTerminalExit: async (
        params: { terminalId: string }
      ): Promise<{ exitCode: number | null; signal: string | null }> => {
        const terminal = this.terminals.get(params.terminalId);
        if (!terminal) {
          throw new Error(`Terminal not found: ${params.terminalId}`);
        }

        await terminal.exitPromise;

        return {
          exitCode: terminal.exitCode,
          signal: terminal.exitSignal,
        };
      },

      // Terminal: kill
      killTerminal: async (
        params: { terminalId: string }
      ): Promise<Record<string, never>> => {
        const terminal = this.terminals.get(params.terminalId);
        if (!terminal) {
          throw new Error(`Terminal not found: ${params.terminalId}`);
        }

        if (!terminal.exited) {
          terminal.process.kill();
        }

        return {};
      },

      // Terminal: release
      releaseTerminal: async (
        params: { terminalId: string }
      ): Promise<Record<string, never>> => {
        const terminal = this.terminals.get(params.terminalId);
        if (!terminal) {
          return {};
        }

        if (!terminal.exited) {
          terminal.process.kill();
        }

        this.terminals.delete(params.terminalId);
        return {};
      },
    };
  }
}

// Map of ACP services by widget ID - each ChatWidget gets its own agent
const acpServices: Map<string, ACPClientService> = new Map();

/**
 * Get or create an ACP service for a specific widget
 * @param widgetId - Unique identifier for the widget (e.g., window ID)
 * @param events - Event emitter
 */
export function getACPServiceForWidget(widgetId: string, events: EventEmitter): ACPClientService {
  let service = acpServices.get(widgetId);
  if (!service) {
    service = new ACPClientService(events);
    acpServices.set(widgetId, service);
    console.log('[acp] Created new service for widget:', widgetId);
  }
  return service;
}

/**
 * Remove an ACP service for a widget
 */
export async function removeACPServiceForWidget(widgetId: string): Promise<void> {
  const service = acpServices.get(widgetId);
  if (service) {
    await service.shutdown();
    acpServices.delete(widgetId);
    console.log('[acp] Removed service for widget:', widgetId);
  }
}

/**
 * Shutdown all ACP services
 */
export async function shutdownAllACPServices(): Promise<void> {
  for (const [widgetId, service] of acpServices.entries()) {
    await service.shutdown();
    console.log('[acp] Shutdown service for widget:', widgetId);
  }
  acpServices.clear();
}
