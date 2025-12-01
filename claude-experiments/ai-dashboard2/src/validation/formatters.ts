import { ValidationResult } from './validate';

/**
 * Format validation error for AI assistant consumption
 * Returns a clear, actionable message with examples
 */
export function formatValidationErrorForAI(result: ValidationResult): string {
  if (result.valid) {
    return '✅ Widget configuration is valid';
  }

  const lines: string[] = [];
  lines.push('❌ Widget Configuration Validation Failed\n');

  if (result.widgetType) {
    lines.push(`Widget Type: ${result.widgetType}\n`);
  }

  if (result.errors && result.errors.length > 0) {
    lines.push('Errors:');
    for (const error of result.errors) {
      lines.push(`  • ${error.field}: ${error.message}`);
      if (error.expected) {
        lines.push(`    Expected: ${error.expected}`);
      }
      if (error.received) {
        lines.push(`    Received: ${error.received}`);
      }
    }
    lines.push('');
  }

  if (result.suggestion) {
    lines.push(`💡 Suggestion: ${result.suggestion}\n`);
  }

  if (result.exampleConfig) {
    lines.push('📋 Example Valid Configuration:');
    lines.push('```json');
    lines.push(JSON.stringify(result.exampleConfig, null, 2));
    lines.push('```');
  }

  return lines.join('\n');
}

/**
 * Format validation error for UI display
 * Returns a simple, user-friendly message
 */
export function formatValidationErrorForUI(result: ValidationResult): string {
  if (result.valid) {
    return '';
  }

  const lines: string[] = [];

  if (result.errors && result.errors.length > 0) {
    for (const error of result.errors) {
      lines.push(`${error.field}: ${error.message}`);
    }
  }

  if (result.suggestion) {
    lines.push(`\n💡 ${result.suggestion}`);
  }

  return lines.join('\n');
}

/**
 * Format validation result as JSON for programmatic use
 */
export function formatValidationErrorAsJSON(result: ValidationResult): string {
  return JSON.stringify(result, null, 2);
}
