interface ToolInput {
  file_path?: string;
  command?: string;
  pattern?: string;
  description?: string;
  url?: string;
  query?: string;
}

export function formatToolDescription(toolName: string, input?: ToolInput): string {
  if (!input) return toolName;

  switch (toolName) {
    case 'Read':
      return `Read ${input.file_path?.split('/').pop() || input.file_path || ''}`;
    case 'Write':
      return `Write ${input.file_path?.split('/').pop() || input.file_path || ''}`;
    case 'Edit':
      return `Edit ${input.file_path?.split('/').pop() || input.file_path || ''}`;
    case 'Bash': {
      const cmd = input.command || '';
      const shortCmd = cmd.length > 40 ? cmd.substring(0, 40) + '...' : cmd;
      return `Run: ${shortCmd}`;
    }
    case 'Grep':
      return `Search for "${input.pattern || ''}"`;
    case 'Glob':
      return `Find files: ${input.pattern || ''}`;
    case 'Task':
      return `${input.description || 'Start task'}`;
    case 'WebFetch': {
      const url = input.url || '';
      const domain = url.replace(/^https?:\/\//, '').split('/')[0];
      return `Fetch ${domain}`;
    }
    case 'WebSearch':
      return `Search: ${input.query || ''}`;
    default:
      return toolName;
  }
}
