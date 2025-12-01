import { WidgetConfigSchema } from './schemas';

export interface ValidationError {
  field: string;
  message: string;
  expected?: string;
  received?: string;
}

export interface ValidationResult {
  valid: boolean;
  errors?: ValidationError[];
  widgetType?: string;
  suggestion?: string;
  exampleConfig?: any;
}

/**
 * Validate a widget configuration against the schema
 */
export function validateWidget(config: any): ValidationResult {
  try {
    // First check if type exists
    if (!config.type) {
      return {
        valid: false,
        errors: [{
          field: 'type',
          message: 'Missing required field "type"',
          expected: 'One of: stat, barChart, progress, chat, codeEditor, commandRunner, etc.',
        }],
        suggestion: 'Add a "type" field to specify which widget type to create',
      };
    }

    // Validate against schema
    const result = WidgetConfigSchema.safeParse(config);

    if (result.success) {
      return { valid: true };
    }

    // Parse Zod errors into friendly format
    const errors: ValidationError[] = [];
    const zodErrors = result.error.issues;

    for (const error of zodErrors) {
      const field = error.path.join('.');

      // Check for common property name mistakes
      if (error.code === 'invalid_type' && error.path.includes('runCommand')) {
        errors.push({
          field: 'runCommand',
          message: 'Property "runCommand" does not exist on codeEditor widget',
          expected: 'Use "command" instead',
          received: 'runCommand',
        });
      } else if (error.code === 'invalid_type' && error.path.includes('defaultCode')) {
        errors.push({
          field: 'defaultCode',
          message: 'Property "defaultCode" does not exist on codeEditor widget',
          expected: 'Use "content" instead',
          received: 'defaultCode',
        });
      } else {
        errors.push({
          field,
          message: error.message,
          expected: 'expected' in error ? String(error.expected) : undefined,
          received: 'received' in error ? String(error.received) : undefined,
        });
      }
    }

    return {
      valid: false,
      widgetType: config.type,
      errors,
      suggestion: generateSuggestion(config, errors),
      exampleConfig: generateExampleConfig(config.type),
    };
  } catch (error) {
    return {
      valid: false,
      errors: [{
        field: 'unknown',
        message: error instanceof Error ? error.message : 'Unknown validation error',
      }],
    };
  }
}

/**
 * Generate a helpful suggestion based on the errors
 */
function generateSuggestion(_config: any, errors: ValidationError[]): string {
  const firstError = errors[0];
  if (!firstError) return 'Fix the validation errors above';

  // Special suggestions for common mistakes
  if (firstError.field === 'runCommand') {
    return 'Change "runCommand" to "command" in your widget configuration';
  }
  if (firstError.field === 'defaultCode') {
    return 'Change "defaultCode" to "content" in your widget configuration';
  }
  if (firstError.field === 'type') {
    return 'Add a "type" field specifying which widget to create (e.g., "codeEditor", "stat", "chat")';
  }

  return `Fix the "${firstError.field}" field: ${firstError.message}`;
}

/**
 * Generate an example configuration for a widget type
 */
function generateExampleConfig(widgetType: string): any {
  const examples: Record<string, any> = {
    stat: {
      type: 'stat',
      id: 'my-stat',
      label: 'Uptime',
      value: '99.9%',
      x: 0,
      y: 0,
      width: 200,
      height: 100,
    },
    codeEditor: {
      type: 'codeEditor',
      id: 'my-editor',
      label: 'Code Editor',
      language: 'javascript',
      command: 'node {file}',
      content: 'console.log("Hello, world!");',
      x: 0,
      y: 0,
      width: 600,
      height: 400,
    },
    'code-editor': {
      type: 'code-editor',
      id: 'my-editor',
      label: 'Code Editor',
      language: 'javascript',
      command: 'node {file}',
      content: 'console.log("Hello, world!");',
      x: 0,
      y: 0,
      width: 600,
      height: 400,
    },
    commandRunner: {
      type: 'commandRunner',
      id: 'my-runner',
      label: 'Test Runner',
      command: 'npm test',
      autoRun: false,
      showOutput: true,
      x: 0,
      y: 0,
      width: 400,
      height: 300,
    },
    chat: {
      type: 'chat',
      id: 'my-chat',
      label: 'AI Assistant',
      backend: 'claude',
      claudeOptions: {
        model: 'claude-sonnet-4-5-20250929',
      },
      x: 0,
      y: 0,
      width: 400,
      height: 600,
    },
    testResults: {
      type: 'testResults',
      id: 'my-tests',
      label: 'Test Results',
      testRunner: 'jest',
      tests: [
        { name: 'Test 1', status: 'passed', duration: 123 },
        { name: 'Test 2', status: 'failed', error: 'Expected true but got false' },
      ],
      x: 0,
      y: 0,
      width: 400,
      height: 300,
    },
  };

  return examples[widgetType] || {
    type: widgetType,
    id: 'my-widget',
    label: 'Widget Label',
    x: 0,
    y: 0,
    width: 400,
    height: 300,
  };
}
