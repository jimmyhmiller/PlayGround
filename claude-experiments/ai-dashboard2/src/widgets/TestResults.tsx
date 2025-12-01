import { FC, useState } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';

interface TestResult {
  name: string;
  status: 'passed' | 'failed' | 'skipped';
  duration?: number;
  error?: string;
}

interface TestResultsConfig {
  id: string;
  type: 'testResults' | 'test-results';
  label?: string;
  tests?: TestResult[];
  testRunner?: 'cargo' | 'jest' | 'pytest';
  regenerateCommand?: string;
  regenerateScript?: string;
  regenerate?: boolean;
  autoRun?: boolean;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export const TestResults: FC<BaseWidgetComponentProps> = (props) => {
  const { theme, config, dashboardId } = props;
  const testConfig = config as TestResultsConfig;
  const [hiddenStatuses, setHiddenStatuses] = useState(new Set<string>());
  const [isRegenerating, setIsRegenerating] = useState(false);

  const toggleStatus = (status: string) => {
    setHiddenStatuses(prev => {
      const next = new Set(prev);
      if (next.has(status)) {
        next.delete(status);
      } else {
        next.add(status);
      }
      return next;
    });
  };

  const handleRegenerate = async () => {
    if (!(window as any).dashboardAPI || !dashboardId || !config.id) return;

    setIsRegenerating(true);
    try {
      const result = await (window as any).dashboardAPI.regenerateWidget(dashboardId, config.id);
      if (!result.success) {
        console.error('Regenerate failed:', result.error);
      }
    } catch (error) {
      console.error('Regenerate error:', error);
    } finally {
      setIsRegenerating(false);
    }
  };

  const hasRegenerate = testConfig.regenerateCommand || testConfig.regenerateScript || testConfig.regenerate || testConfig.testRunner;

  // Validation
  const validateTests = () => {
    // If using a built-in test runner, tests array is optional (will be populated on run)
    if (!testConfig.tests && !testConfig.testRunner) {
      return {
        valid: false,
        message: 'Missing "tests" array',
        expected: 'The testResults widget requires a "tests" array property.\n\nExpected format:\n{\n  "type": "testResults",\n  "label": "Test Suite Results",\n  "tests": [\n    {\n      "name": "Test name",\n      "status": "passed" | "failed" | "skipped",\n      "duration": 123,  // optional, in milliseconds\n      "error": "Error message"  // optional, for failed tests\n    }\n  ]\n}\n\nAlternatively, use a built-in test runner:\n{\n  "type": "testResults",\n  "label": "Test Suite",\n  "testRunner": "cargo" | "jest" | "pytest",\n  "autoRun": false\n}'
      };
    }

    // If no tests yet (using testRunner), validation passes
    if (!testConfig.tests) {
      return { valid: true };
    }

    if (!Array.isArray(testConfig.tests)) {
      return {
        valid: false,
        message: '"tests" must be an array',
        expected: `Expected "tests" to be an array, but got ${typeof testConfig.tests}.\n\nExpected format:\n"tests": [\n  {\n    "name": "Test name",\n    "status": "passed",\n    "duration": 123\n  }\n]`
      };
    }

    // Validate each test
    for (let i = 0; i < testConfig.tests.length; i++) {
      const test = testConfig.tests[i];

      if (!test || typeof test !== 'object') {
        return {
          valid: false,
          message: `Test at index ${i} is not an object`,
          expected: `Each test must be an object with "name" and "status" properties.\n\nExpected format for test:\n{\n  "name": "Test name",\n  "status": "passed" | "failed" | "skipped",\n  "duration": 123,  // optional\n  "error": "Error message"  // optional\n}`
        };
      }

      if (!test.name || typeof test.name !== 'string') {
        return {
          valid: false,
          message: `Test at index ${i} missing or invalid "name" property`,
          expected: `Each test must have a "name" property (string).\n\nCurrent test: ${JSON.stringify(test, null, 2)}\n\nExpected: { "name": "Test name", "status": "passed", ... }`
        };
      }

      if (!test.status || !['passed', 'failed', 'skipped'].includes(test.status)) {
        return {
          valid: false,
          message: `Test "${test.name}" has invalid "status" property`,
          expected: `The "status" property must be one of: "passed", "failed", or "skipped".\n\nCurrent value: "${test.status}"\n\nExpected format:\n{\n  "name": "${test.name}",\n  "status": "passed" | "failed" | "skipped"\n}`
        };
      }

      if (test.duration !== undefined && typeof test.duration !== 'number') {
        return {
          valid: false,
          message: `Test "${test.name}" has invalid "duration" property`,
          expected: `The "duration" property must be a number (milliseconds).\n\nCurrent value: ${JSON.stringify(test.duration)} (${typeof test.duration})\n\nExpected format:\n{\n  "name": "${test.name}",\n  "status": "${test.status}",\n  "duration": 123\n}`
        };
      }

      if (test.error !== undefined && typeof test.error !== 'string') {
        return {
          valid: false,
          message: `Test "${test.name}" has invalid "error" property`,
          expected: `The "error" property must be a string.\n\nCurrent value: ${JSON.stringify(test.error)} (${typeof test.error})\n\nExpected format:\n{\n  "name": "${test.name}",\n  "status": "failed",\n  "error": "Error message here"\n}`
        };
      }
    }

    return { valid: true };
  };

  const validation = validateTests();

  if (!validation.valid) {
    return (
      <>
        <div className="widget-label" style={{ fontFamily: theme.textBody }}>{testConfig.label || 'Test Results'}</div>
        <div style={{
          fontFamily: theme.textBody,
          fontSize: '0.8rem',
          padding: '16px',
          color: theme.negative,
          backgroundColor: `${theme.negative}15`,
          border: `1px solid ${theme.negative}40`,
          borderRadius: '8px',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px'
        }}>
          <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>
            ⚠️ Invalid configuration: {validation.message}
          </div>
          <div style={{
            fontFamily: theme.textCode || 'monospace',
            fontSize: '0.7rem',
            backgroundColor: `${theme.negative}10`,
            padding: '12px',
            borderRadius: '6px',
            whiteSpace: 'pre-wrap',
            color: theme.textColor,
            lineHeight: 1.5,
            userSelect: 'text',
            cursor: 'text'
          }}>
            {validation.expected}
          </div>
          <div style={{
            fontSize: '0.7rem',
            opacity: 0.8,
            borderTop: `1px solid ${theme.negative}20`,
            paddingTop: '8px',
            marginTop: '4px'
          }}>
            💡 Tip: Copy the expected format above and provide it to an AI to generate the correct configuration.
          </div>
        </div>
      </>
    );
  }

  const tests = testConfig.tests || [];
  const passedCount = tests.filter(t => t.status === 'passed').length;
  const failedCount = tests.filter(t => t.status === 'failed').length;
  const skippedCount = tests.filter(t => t.status === 'skipped').length;
  const totalCount = tests.length;

  return (
    <>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: hasRegenerate ? '8px' : '0'
      }}>
        <div className="widget-label" style={{ fontFamily: theme.textBody, marginBottom: 0 }}>
          {testConfig.label || 'Test Results'}
        </div>
        {hasRegenerate && (
          <button
            onClick={handleRegenerate}
            disabled={isRegenerating}
            style={{
              background: 'transparent',
              border: `1px solid ${theme.accent}40`,
              borderRadius: '4px',
              color: theme.accent,
              cursor: isRegenerating ? 'wait' : 'pointer',
              padding: '4px 8px',
              fontSize: '0.75rem',
              fontFamily: theme.textBody,
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              opacity: isRegenerating ? 0.5 : 1,
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => {
              if (!isRegenerating) {
                (e.target as HTMLButtonElement).style.backgroundColor = `${theme.accent}15`;
                (e.target as HTMLButtonElement).style.borderColor = `${theme.accent}80`;
              }
            }}
            onMouseLeave={(e) => {
              (e.target as HTMLButtonElement).style.backgroundColor = 'transparent';
              (e.target as HTMLButtonElement).style.borderColor = `${theme.accent}40`;
            }}
          >
            <span style={{ fontSize: '0.9rem' }}>{isRegenerating ? '⟳' : '▶'}</span>
            {isRegenerating ? 'Running...' : 'Run Tests'}
          </button>
        )}
      </div>

      {/* Summary */}
      <div style={{
        fontFamily: theme.textBody,
        fontSize: '0.85rem',
        padding: '12px 0',
        borderBottom: `1px solid ${theme.accent}22`,
        marginBottom: '12px',
        display: 'flex',
        gap: '16px',
        flexWrap: 'wrap'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ color: theme.textColor, opacity: 0.7 }}>Total:</span>
          <span style={{ color: theme.accent, fontWeight: 600 }}>{totalCount}</span>
        </div>
        {passedCount > 0 && (
          <div
            onClick={() => toggleStatus('passed')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              cursor: 'pointer',
              padding: '2px 6px',
              margin: '-2px -6px',
              borderRadius: '4px',
              backgroundColor: hiddenStatuses.has('passed') ? `${theme.positive}10` : 'transparent',
              opacity: hiddenStatuses.has('passed') ? 0.5 : 1,
              transition: 'all 0.2s',
              userSelect: 'none'
            }}
          >
            <span style={{ color: theme.positive }}>✓</span>
            <span style={{ color: theme.positive, fontWeight: 600 }}>{passedCount} passed</span>
          </div>
        )}
        {failedCount > 0 && (
          <div
            onClick={() => toggleStatus('failed')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              cursor: 'pointer',
              padding: '2px 6px',
              margin: '-2px -6px',
              borderRadius: '4px',
              backgroundColor: hiddenStatuses.has('failed') ? `${theme.negative}10` : 'transparent',
              opacity: hiddenStatuses.has('failed') ? 0.5 : 1,
              transition: 'all 0.2s',
              userSelect: 'none'
            }}
          >
            <span style={{ color: theme.negative }}>✗</span>
            <span style={{ color: theme.negative, fontWeight: 600 }}>{failedCount} failed</span>
          </div>
        )}
        {skippedCount > 0 && (
          <div
            onClick={() => toggleStatus('skipped')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              cursor: 'pointer',
              padding: '2px 6px',
              margin: '-2px -6px',
              borderRadius: '4px',
              backgroundColor: hiddenStatuses.has('skipped') ? `${theme.textColor}10` : 'transparent',
              opacity: hiddenStatuses.has('skipped') ? 0.5 : 1,
              transition: 'all 0.2s',
              userSelect: 'none'
            }}
          >
            <span style={{ color: theme.textColor, opacity: 0.5 }}>○</span>
            <span style={{ color: theme.textColor, opacity: 0.7 }}>{skippedCount} skipped</span>
          </div>
        )}
      </div>

      {/* Test List */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        overflow: 'auto',
        maxHeight: '100%'
      }}>
        {tests.length === 0 && testConfig.testRunner ? (
          <div style={{
            fontFamily: theme.textBody,
            fontSize: '0.85rem',
            padding: '24px',
            textAlign: 'center',
            color: theme.textColor,
            opacity: 0.6
          }}>
            <div style={{ fontSize: '2rem', marginBottom: '12px' }}>📋</div>
            <div>No test results yet</div>
            <div style={{ fontSize: '0.75rem', marginTop: '8px' }}>
              Click "Run Tests" to execute {testConfig.testRunner} tests
            </div>
          </div>
        ) : tests.length === 0 ? (
          <div style={{
            fontFamily: theme.textBody,
            fontSize: '0.85rem',
            padding: '24px',
            textAlign: 'center',
            color: theme.textColor,
            opacity: 0.6
          }}>
            No tests configured
          </div>
        ) : null}
        {tests.filter(test => !hiddenStatuses.has(test.status)).map((test, i) => (
          <div
            key={i}
            style={{
              fontFamily: theme.textBody,
              fontSize: '0.75rem',
              padding: '8px 12px',
              borderRadius: '6px',
              backgroundColor: test.status === 'passed'
                ? `${theme.positive}15`
                : test.status === 'failed'
                ? `${theme.negative}15`
                : `${theme.textColor}08`,
              border: `1px solid ${
                test.status === 'passed'
                  ? `${theme.positive}40`
                  : test.status === 'failed'
                  ? `${theme.negative}40`
                  : `${theme.textColor}20`
              }`,
              display: 'flex',
              flexDirection: 'column',
              gap: '4px'
            }}
          >
            {/* Test name with status icon */}
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
              <span style={{
                color: test.status === 'passed'
                  ? theme.positive
                  : test.status === 'failed'
                  ? theme.negative
                  : theme.textColor,
                opacity: test.status === 'skipped' ? 0.5 : 1,
                fontSize: '0.85rem',
                flexShrink: 0
              }}>
                {test.status === 'passed' ? '✓' : test.status === 'failed' ? '✗' : '○'}
              </span>
              <div style={{ flex: 1 }}>
                <div style={{
                  color: theme.textColor,
                  fontWeight: 500,
                  wordBreak: 'break-word'
                }}>
                  {test.name}
                </div>
                {test.duration && (
                  <div style={{
                    color: theme.textColor,
                    opacity: 0.5,
                    fontSize: '0.7rem',
                    marginTop: '2px'
                  }}>
                    {test.duration}ms
                  </div>
                )}
              </div>
            </div>

            {/* Error message for failed tests */}
            {test.status === 'failed' && test.error && (
              <div style={{
                marginTop: '4px',
                padding: '6px 8px',
                backgroundColor: `${theme.negative}10`,
                borderLeft: `2px solid ${theme.negative}`,
                borderRadius: '4px',
                color: theme.negative,
                fontSize: '0.7rem',
                fontFamily: theme.textCode || 'monospace',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word'
              }}>
                {test.error}
              </div>
            )}
          </div>
        ))}
      </div>
    </>
  );
};
