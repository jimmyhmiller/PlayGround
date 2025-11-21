import { ThemeContract } from './theme';

// Messages from External Agent → Dashboard
export type AgentToUIMessage =
  | RegisterComponentMessage
  | RegisterDataSourceMessage
  | UpdateDataMessage;

export interface RegisterComponentMessage {
  type: 'register:component';
  agent: {
    id: string;
    name: string;
  };
  component: {
    id: string;
    name: string;
    semantic: 'data-series' | 'metric-display' | 'status-item' | 'custom';
    code: string;
    dependencies?: string[];
    dataSchema?: any;
    themeContract: ThemeContract;
  };
}

export interface RegisterDataSourceMessage {
  type: 'register:data-source';
  agent: {
    id: string;
    name: string;
  };
  dataSource: {
    id: string;
    name: string;
    provider: {
      type: 'polling' | 'webhook' | 'streaming';
      url?: string;
      interval?: number;
    };
    compatibleWith: string[];
  };
}

export interface UpdateDataMessage {
  type: 'update:data';
  agent: {
    id: string;
  };
  dataSourceId: string;
  componentId?: string;
  data: any;
  timestamp: number;
}

// Messages from Dashboard → External Agent
export type UIToAgentMessage =
  | QueryMessage
  | ThemeChangedMessage
  | UserInteractionMessage;

export interface QueryMessage {
  type: 'query';
  queryId: string;
  query: {
    type: string;
    payload: any;
  };
}

export interface ThemeChangedMessage {
  type: 'theme-changed';
  theme: {
    colors?: Record<string, string>;
    fonts?: Record<string, string>;
  };
}

export interface UserInteractionMessage {
  type: 'user-interaction';
  componentId: string;
  event: string;
  data: any;
}
