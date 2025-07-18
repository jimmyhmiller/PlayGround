import { HLCTimestamp } from '../utils/hlc';
import { Entity } from './entity';

export type MessageType = 'update' | 'sync-request' | 'sync-response';

export interface Message {
  id: string;
  from: string;
  to: string;
  timestamp: HLCTimestamp;
  content: string;
  type: MessageType;
  entityUpdates?: Partial<Entity>;
  fullEntityState?: Entity;
  messageHistory?: Message[];
  receivedAt?: HLCTimestamp;
}

export interface Client {
  id: string;
  name: string;
  isOnline: boolean;
  clockSkew: number;
  messages: Message[];
}

export * from './entity';