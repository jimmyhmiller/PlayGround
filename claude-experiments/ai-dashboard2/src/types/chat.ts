export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  isStreaming?: boolean;
}

export interface ChatOptions {
  projectPath?: string;
  continueSession?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number | string;
  lastMessageAt: number | string;
  sessionId?: string;
}

export interface Todo {
  content: string;
  status: 'pending' | 'in_progress' | 'completed';
  activeForm: string;
}
