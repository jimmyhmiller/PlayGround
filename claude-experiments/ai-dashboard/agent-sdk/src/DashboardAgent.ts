import WebSocket from 'ws';
import { EventEmitter } from 'events';

export interface AgentConfig {
  id: string;
  name: string;
  connection: {
    type: 'websocket';
    url: string;
  };
}

export interface ComponentRegistration {
  id: string;
  name: string;
  semantic: 'data-series' | 'metric-display' | 'status-item' | 'custom';
  code: string;
  dependencies?: string[];
  dataSchema?: any;
  themeContract: {
    uses: string[];
    providesClasses?: string[];
  };
}

export interface DataSourceRegistration {
  id: string;
  name: string;
  provider: {
    type: 'polling' | 'webhook' | 'streaming';
    url?: string;
    interval?: number;
  };
  compatibleWith: string[];
}

export class DashboardAgent extends EventEmitter {
  private config: AgentConfig;
  private ws: WebSocket | null = null;
  private connected = false;

  constructor(config: AgentConfig) {
    super();
    this.config = config;
  }

  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.config.connection.url);

      this.ws.on('open', () => {
        this.connected = true;
        console.log(`[Agent ${this.config.id}] Connected to dashboard`);
        resolve();
      });

      this.ws.on('message', (data: Buffer) => {
        try {
          const message = JSON.parse(data.toString());
          this.handleMessage(message);
        } catch (error) {
          console.error(`[Agent ${this.config.id}] Failed to parse message:`, error);
        }
      });

      this.ws.on('close', () => {
        this.connected = false;
        console.log(`[Agent ${this.config.id}] Disconnected from dashboard`);
        this.emit('disconnected');
      });

      this.ws.on('error', (error) => {
        console.error(`[Agent ${this.config.id}] WebSocket error:`, error);
        reject(error);
      });
    });
  }

  private handleMessage(message: any) {
    switch (message.type) {
      case 'query':
        this.emit('query', message.query, (response: any) => {
          this.send({
            type: 'response',
            queryId: message.queryId,
            response,
          });
        });
        break;

      case 'theme-changed':
        this.emit('theme-changed', message.theme);
        break;

      case 'user-interaction':
        this.emit('user-interaction', message);
        break;

      default:
        console.warn(`[Agent ${this.config.id}] Unknown message type:`, message.type);
    }
  }

  async registerComponent(component: ComponentRegistration): Promise<void> {
    await this.send({
      type: 'register:component',
      agent: {
        id: this.config.id,
        name: this.config.name,
      },
      component,
    });

    console.log(`[Agent ${this.config.id}] Registered component: ${component.id}`);
  }

  async registerDataSource(dataSource: DataSourceRegistration): Promise<void> {
    await this.send({
      type: 'register:data-source',
      agent: {
        id: this.config.id,
        name: this.config.name,
      },
      dataSource,
    });

    console.log(`[Agent ${this.config.id}] Registered data source: ${dataSource.id}`);
  }

  async updateData(dataSourceId: string, data: any, componentId?: string): Promise<void> {
    await this.send({
      type: 'update:data',
      agent: {
        id: this.config.id,
      },
      dataSourceId,
      componentId,
      data,
      timestamp: Date.now(),
    });
  }

  private async send(message: any): Promise<void> {
    if (!this.ws || !this.connected) {
      throw new Error('Not connected to dashboard');
    }

    return new Promise((resolve, reject) => {
      this.ws!.send(JSON.stringify(message), (error) => {
        if (error) {
          reject(error);
        } else {
          resolve();
        }
      });
    });
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
      this.connected = false;
    }
  }

  isConnected(): boolean {
    return this.connected;
  }
}
