import WebSocket from 'ws';
import { EventEmitter } from 'events';
import { createServer } from 'net';
import {
  AgentToUIMessage,
  RegisterComponentMessage,
  UpdateDataMessage,
} from '../../src/types/agent-messages';

interface RegisteredComponent {
  agentId: string;
  component: RegisterComponentMessage['component'];
}

interface DataSource {
  agentId: string;
  id: string;
  latestData: any;
}

export class AgentBridge extends EventEmitter {
  private wss: WebSocket.Server | null = null;
  private connections = new Map<string, WebSocket>();
  private components = new Map<string, RegisteredComponent>();
  private dataSources = new Map<string, DataSource>();
  private currentStyle: any = null;
  private pendingRequests = new Map<string, { resolve: (value: any) => void; reject: (error: any) => void }>();
  private requestCounter = 0;
  private port: number = 3000;

  async start() {
    // Find an available port
    this.port = await this.findAvailablePort(3000);

    // Start WebSocket server for external agents
    this.wss = new WebSocket.Server({ port: this.port });

    this.wss.on('connection', (ws: WebSocket) => {
      console.log('[AgentBridge] External agent connected');

      ws.on('message', (data: Buffer) => {
        try {
          const message: AgentToUIMessage = JSON.parse(data.toString());
          this.handleAgentMessage(ws, message);
        } catch (error) {
          console.error('[AgentBridge] Failed to parse message:', error);
        }
      });

      ws.on('close', () => {
        // Remove agent from connections
        for (const [id, conn] of this.connections.entries()) {
          if (conn === ws) {
            this.connections.delete(id);
            console.log(`[AgentBridge] Agent ${id} disconnected`);
            break;
          }
        }
      });

      ws.on('error', (error) => {
        console.error('[AgentBridge] WebSocket error:', error);
      });
    });

    console.log(`[AgentBridge] Started on ws://localhost:${this.port}`);
  }

  private async findAvailablePort(startPort: number): Promise<number> {
    const isPortAvailable = (port: number): Promise<boolean> => {
      return new Promise((resolve) => {
        const server = createServer();

        server.once('error', (err: any) => {
          if (err.code === 'EADDRINUSE') {
            resolve(false);
          } else {
            resolve(false);
          }
        });

        server.once('listening', () => {
          server.close();
          resolve(true);
        });

        server.listen(port);
      });
    };

    let port = startPort;
    while (!(await isPortAvailable(port))) {
      console.log(`[AgentBridge] Port ${port} is in use, trying ${port + 1}...`);
      port++;

      if (port > startPort + 100) {
        throw new Error(`Could not find available port in range ${startPort}-${port}`);
      }
    }

    return port;
  }

  getPort(): number {
    return this.port;
  }

  private handleAgentMessage(ws: WebSocket, message: any) {
    switch (message.type) {
      case 'register:component':
        this.handleComponentRegistration(ws, message);
        break;

      case 'register:data-source':
        this.handleDataSourceRegistration(ws, message);
        break;

      case 'update:data':
        this.handleDataUpdate(message);
        break;

      case 'response':
        this.handleResponse(message);
        break;

      default:
        console.warn('[AgentBridge] Unknown message type:', message.type);
    }
  }

  private handleResponse(message: any) {
    const { queryId, response } = message;
    const pending = this.pendingRequests.get(queryId);

    if (pending) {
      pending.resolve(response);
      this.pendingRequests.delete(queryId);
    } else {
      console.warn('[AgentBridge] Received response for unknown query:', queryId);
    }
  }

  private handleComponentRegistration(
    ws: WebSocket,
    message: RegisterComponentMessage
  ) {
    const { agent, component } = message;

    console.log(
      `[AgentBridge] Agent "${agent.name}" registered component: ${component.id}`
    );

    // Store connection
    this.connections.set(agent.id, ws);

    // Store component
    this.components.set(component.id, {
      agentId: agent.id,
      component,
    });

    // Emit event for main process
    this.emit('registration', message);
  }

  private handleDataSourceRegistration(ws: WebSocket, message: any) {
    const { agent, dataSource } = message;

    console.log(
      `[AgentBridge] Agent "${agent.name}" registered data source: ${dataSource.id}`
    );

    // Store connection
    this.connections.set(agent.id, ws);

    // Store data source
    this.dataSources.set(dataSource.id, {
      agentId: agent.id,
      id: dataSource.id,
      latestData: null,
    });

    // Emit event for main process
    this.emit('registration', message);
  }

  private handleDataUpdate(message: UpdateDataMessage) {
    const { dataSourceId, data } = message;

    // Update stored data
    const dataSource = this.dataSources.get(dataSourceId);
    if (dataSource) {
      dataSource.latestData = data;
    }

    // Emit event for main process to forward to renderer
    this.emit('data-update', message);

    console.log(`[AgentBridge] Data update for: ${dataSourceId}`);
  }

  // Send message to specific agent and wait for response
  async sendToAgent(agentId: string, message: any): Promise<any> {
    const ws = this.connections.get(agentId);
    if (!ws) {
      throw new Error(`Agent ${agentId} not connected`);
    }

    // Generate unique query ID
    const queryId = `query-${++this.requestCounter}`;

    // Add query ID to message
    const messageWithId = {
      ...message,
      queryId,
    };

    return new Promise((resolve, reject) => {
      // Store the pending request
      this.pendingRequests.set(queryId, { resolve, reject });

      // Set timeout
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(queryId);
        reject(new Error(`Request to ${agentId} timed out`));
      }, 30000); // 30 second timeout

      // Send the message
      ws.send(JSON.stringify(messageWithId), (error) => {
        if (error) {
          clearTimeout(timeout);
          this.pendingRequests.delete(queryId);
          reject(error);
        }
      });

      // Clear timeout when resolved
      const originalResolve = this.pendingRequests.get(queryId)?.resolve;
      if (originalResolve) {
        this.pendingRequests.set(queryId, {
          resolve: (value: any) => {
            clearTimeout(timeout);
            originalResolve(value);
          },
          reject,
        });
      }
    });
  }

  // Get all registered components
  getRegisteredComponents(): RegisteredComponent[] {
    return Array.from(this.components.values());
  }

  // Get data from a specific data source
  getData(dataSourceId: string): any {
    const dataSource = this.dataSources.get(dataSourceId);
    return dataSource?.latestData || null;
  }

  // Broadcast message to all connected agents
  broadcast(message: any) {
    const payload = JSON.stringify(message);
    this.connections.forEach((ws) => {
      ws.send(payload);
    });
  }

  // Get current style
  getCurrentStyle(): any {
    return this.currentStyle;
  }

  // Set current style (called when style agent sends a style)
  setCurrentStyle(style: any) {
    this.currentStyle = style;
    this.emit('style-updated', style);
  }
}
