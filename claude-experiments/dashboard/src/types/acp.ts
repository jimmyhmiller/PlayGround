/**
 * ACP (Agent Client Protocol) Types
 *
 * Re-exports from @agentclientprotocol/sdk and app-specific types
 * for the chat widget integration.
 */

// Re-export SDK types that we use throughout the app
export type {
  // Core connection types
  Client,
  Agent,

  // Session types
  SessionNotification,
  SessionUpdate,
  SessionId,
  SessionMode,
  SessionModeId,

  // Content types
  ContentBlock,
  TextContent,
  ImageContent,

  // Tool call types
  ToolCall,
  ToolCallUpdate,
  ToolCallId,
  ToolCallStatus,
  ToolCallContent,
  ToolCallLocation,

  // Permission types
  RequestPermissionRequest,
  RequestPermissionResponse,

  // File system types
  ReadTextFileRequest,
  ReadTextFileResponse,
  WriteTextFileRequest,
  WriteTextFileResponse,

  // Terminal types
  CreateTerminalRequest,
  CreateTerminalResponse,
  TerminalOutputRequest,
  TerminalOutputResponse,

  // Initialize types
  InitializeRequest,
  InitializeResponse,
  ClientCapabilities,
  AgentCapabilities,

  // Session setup types
  NewSessionRequest,
  NewSessionResponse,
  LoadSessionRequest,
  LoadSessionResponse,

  // Prompt types
  PromptRequest,
  PromptResponse,
  StopReason,

  // Mode types
  SetSessionModeRequest,
  SetSessionModeResponse,

  // Cancel types
  CancelNotification,
} from '@agentclientprotocol/sdk';

// Note: PROTOCOL_VERSION is accessed dynamically in acpClientService.ts
// to avoid ESM import issues in the main process

/**
 * Message types for our chat UI
 */
export type MessageRole = 'user' | 'assistant';

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: number;
  // For assistant messages, track associated tool calls
  toolCallIds?: string[];
}

/**
 * Extended tool call with UI state
 */
export interface UIToolCall {
  toolCallId: string;
  title: string;
  kind: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  content?: unknown[];
  locations?: Array<{ path: string; lineRange?: { start: number; end: number } }>;
  result?: unknown;
  error?: string;
}

/**
 * Permission request with UI state
 */
export interface UIPermissionRequest {
  requestId: string;
  toolCallId: string;
  title: string;
  description?: string;
  options: Array<{
    id: string;
    label: string;
    description?: string;
  }>;
}

/**
 * Session state for persistence
 */
export interface ChatSessionState {
  sessionId: string | null;
  messages: ChatMessage[];
  mode: string;
  todos: UIPlanTask[];
}

/**
 * Plan task with UI state
 */
export interface UIPlanTask {
  id: string;
  title: string;
  status: 'pending' | 'in_progress' | 'completed';
  description?: string;
}

/**
 * Terminal output for display
 */
export interface UITerminal {
  id: string;
  command: string;
  output: string;
  exitCode?: number;
  isRunning: boolean;
}
