/**
 * ACP Client Service
 *
 * Manages the connection to claude-code-acp via the Agent Client Protocol.
 * Spawns the agent as a subprocess and implements the ACP Client interface.
 */

import { spawn, ChildProcess } from 'child_process';
import { Readable, Writable } from 'stream';
import * as fs from 'fs/promises';
import * as path from 'path';

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
  private sessionPromise: Promise<{ sessionId: string }> | null = null;

  constructor(events: EventEmitter) {
    this.events = events;
  }

  /**
   * Set the callback for permission requests
   */
  setPermissionCallback(callback: PermissionCallback): void {
    this.permissionCallback = callback;
  }

  /**
   * Spawn the claude-code-acp agent and establish connection
   */
  async spawn(): Promise<void> {
    // Already connected
    if (this.isInitialized && this.connection) {
      console.log('[acp] Already connected, skipping spawn');
      return;
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
      this.agentProcess = null;
      this.connection = null;
      this.agent = null;
      this.isInitialized = false;
      this.currentSessionId = null;
    }

    // Create spawn promise before any async work
    this.spawnPromise = this.doSpawn();

    try {
      await this.spawnPromise;
    } finally {
      this.spawnPromise = null;
    }
  }

  /**
   * Internal spawn implementation
   */
  private async doSpawn(): Promise<void> {
    const acp = await getACPModule();

    // Spawn claude-code-acp
    this.agentProcess = spawn('npx', ['@zed-industries/claude-code-acp'], {
      stdio: ['pipe', 'pipe', 'inherit'],
      env: {
        ...process.env,
      },
    });

    if (!this.agentProcess.stdin || !this.agentProcess.stdout) {
      throw new Error('Failed to get stdio pipes from agent process');
    }

    // Convert Node streams to Web streams for the SDK
    const input = Writable.toWeb(this.agentProcess.stdin) as WritableStream<Uint8Array>;
    const output = Readable.toWeb(this.agentProcess.stdout) as ReadableStream<Uint8Array>;

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

    // Handle process exit
    this.agentProcess.on('exit', (code, signal) => {
      console.log(`[acp] Agent process exited with code ${code}, signal ${signal}`);
      this.agentProcess = null;
      this.connection = null;
      this.agent = null;
      this.isInitialized = false;
      this.currentSessionId = null;
      this.events.emit('acp.process.exit', { code, signal });
    });

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

    const result = await this.connection!.initialize({
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
    });

    console.log('[acp] Initialized:', result);
    this.isInitialized = true;
    this.events.emit('acp.initialized', result);
  }

  /**
   * Create a new session (or return existing one if already created)
   */
  async newSession(cwd: string, mcpServers?: unknown[]): Promise<{ sessionId: string }> {
    if (!this.connection || !this.isInitialized) {
      throw new Error('Not initialized. Call spawn() and initialize() first.');
    }

    // If we already have a session, return it
    if (this.currentSessionId) {
      console.log('[acp] Returning existing session:', this.currentSessionId);
      return { sessionId: this.currentSessionId };
    }

    // If session creation is in progress, wait for it
    if (this.sessionPromise) {
      console.log('[acp] Session creation in progress, waiting...');
      return await this.sessionPromise;
    }

    // Create session with locking
    this.sessionPromise = this.doNewSession(cwd, mcpServers);

    try {
      return await this.sessionPromise;
    } finally {
      this.sessionPromise = null;
    }
  }

  /**
   * Internal session creation
   */
  private async doNewSession(cwd: string, mcpServers?: unknown[]): Promise<{ sessionId: string }> {
    const result = await this.connection!.newSession({
      cwd,
      mcpServers: mcpServers as never[] ?? [],
    });

    console.log('[acp] New session:', result.sessionId);
    this.currentSessionId = result.sessionId;
    this.events.emit('acp.session.new', { sessionId: result.sessionId });

    return { sessionId: result.sessionId };
  }

  /**
   * Load an existing session
   */
  async loadSession(sessionId: string, cwd: string): Promise<void> {
    if (!this.connection || !this.isInitialized) {
      throw new Error('Not initialized');
    }

    await this.connection.loadSession({
      sessionId,
      cwd,
      mcpServers: [],
    });

    console.log('[acp] Loaded session:', sessionId);
    this.events.emit('acp.session.loaded', { sessionId });
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
      throw new Error('Not connected');
    }

    await this.connection.cancel({ sessionId });
    console.log('[acp] Cancelled session:', sessionId);
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

    console.log('[acp] Shutdown complete');
  }

  /**
   * Check if the service is connected and initialized
   */
  isConnected(): boolean {
    return this.isInitialized && this.connection !== null;
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

        // Default: deny all permissions if no callback set
        return { outcome: 'deny' };
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
        params: { command: string; args?: string[]; cwd?: string; env?: Record<string, string> }
      ): Promise<{ terminalId: string }> => {
        const id = `terminal-${++this.terminalIdCounter}`;

        // Create exit promise
        let exitResolve: () => void = () => {};
        const exitPromise = new Promise<void>((resolve) => {
          exitResolve = resolve;
        });

        const terminal: ManagedTerminal = {
          id,
          process: spawn(params.command, params.args ?? [], {
            cwd: params.cwd,
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

// Singleton instance
let acpService: ACPClientService | null = null;

/**
 * Initialize the ACP client service
 */
export function initACPService(events: EventEmitter): ACPClientService {
  acpService = new ACPClientService(events);
  console.log('[acp] Service initialized');
  return acpService;
}

/**
 * Get the ACP client service instance
 */
export function getACPService(): ACPClientService | null {
  return acpService;
}
