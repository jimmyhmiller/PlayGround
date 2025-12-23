import { memo, useRef } from 'react';
import { useLatestEvent, useEventSubscription } from '../hooks/useEvents';

/**
 * Git Diff Viewer Component
 *
 * Subscribes to git events and displays diffs.
 */

// Simple diff line parser
function parseDiffLines(diff) {
  if (!diff) return [];

  return diff.split('\n').map((line, idx) => {
    let type = 'context';
    if (line.startsWith('+') && !line.startsWith('+++')) {
      type = 'addition';
    } else if (line.startsWith('-') && !line.startsWith('---')) {
      type = 'deletion';
    } else if (line.startsWith('@@')) {
      type = 'hunk';
    } else if (line.startsWith('diff ') || line.startsWith('index ') ||
               line.startsWith('---') || line.startsWith('+++')) {
      type = 'header';
    }

    return { line, type, idx };
  });
}

const lineStyleVars = {
  addition: {
    bg: 'var(--theme-diff-add-bg)',
    color: 'var(--theme-diff-add-text)',
  },
  deletion: {
    bg: 'var(--theme-diff-remove-bg)',
    color: 'var(--theme-diff-remove-text)',
  },
  hunk: {
    bg: 'var(--theme-diff-hunk-bg)',
    color: 'var(--theme-diff-hunk-text)',
  },
  header: {
    bg: 'var(--theme-bg-tertiary)',
    color: 'var(--theme-text-muted)',
  },
  context: {
    bg: 'transparent',
    color: 'var(--theme-code-text)',
  },
};

const GitDiffViewer = memo(function GitDiffViewer({
  instanceId,
  subscribePattern = 'git.**',
  filePath = null,
}) {
  const renderCount = useRef(0);
  renderCount.current += 1;

  // Get latest git events
  const diffEvent = useLatestEvent('git.diff.updated');
  const statusEvent = useLatestEvent('git.status.changed');

  // Get all git events for the log
  const allGitEvents = useEventSubscription(subscribePattern, { maxEvents: 10 });

  // Filter diff by filePath if specified
  const currentDiff = diffEvent?.payload?.diff || null;
  const diffLines = parseDiffLines(currentDiff);

  const files = statusEvent?.payload?.files || [];

  return (
    <div
      className="git-diff"
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: 'var(--theme-bg-secondary)',
        fontFamily: 'var(--theme-font-family)',
      }}
    >
      <div
        className="git-diff-header"
        style={{
          padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
          background: 'var(--theme-bg-tertiary)',
          color: 'var(--theme-text-muted)',
          fontSize: 'var(--theme-font-size-sm)',
          fontFamily: 'var(--theme-font-mono)',
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <span>
          Git Diff {filePath && `(${filePath})`}
          {files.length > 0 && ` • ${files.length} changed files`}
        </span>
        <span style={{ color: 'var(--theme-accent-warning)' }}>
          renders: {renderCount.current}
        </span>
      </div>

      {/* Status bar */}
      {files.length > 0 && (
        <div
          className="git-diff-status"
          style={{
            padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
            background: 'var(--theme-code-bg)',
            borderBottom: '1px solid var(--theme-border-primary)',
            fontSize: 'var(--theme-font-size-md)',
            fontFamily: 'var(--theme-font-mono)',
          }}
        >
          {files.map((f, i) => (
            <span
              key={i}
              className="git-diff-file"
              style={{
                marginRight: '12px',
                color: f.status === 'M' ? 'var(--theme-accent-warning)' :
                       f.status === 'A' ? 'var(--theme-diff-add-text)' :
                       f.status === 'D' ? 'var(--theme-diff-remove-text)' :
                       'var(--theme-text-secondary)',
              }}
            >
              {f.status} {f.path}
            </span>
          ))}
        </div>
      )}

      {/* Diff view */}
      <div
        className="git-diff-content"
        style={{
          flex: 1,
          overflow: 'auto',
          background: 'var(--theme-code-bg)',
          fontFamily: 'var(--theme-font-mono)',
          fontSize: 'var(--theme-font-size-md)',
        }}
      >
        {diffLines.length === 0 ? (
          <div style={{
            padding: 'var(--theme-spacing-xl)',
            color: 'var(--theme-text-muted)',
            textAlign: 'center',
          }}>
            No diff available. Waiting for git.diff.updated event...
          </div>
        ) : (
          diffLines.map(({ line, type, idx }) => (
            <div
              key={idx}
              className={`git-diff-line git-diff-line--${type}`}
              style={{
                padding: '0 var(--theme-spacing-md)',
                whiteSpace: 'pre',
                background: lineStyleVars[type].bg,
                color: lineStyleVars[type].color,
              }}
            >
              {line || ' '}
            </div>
          ))
        )}
      </div>

      {/* Recent git events */}
      <div
        className="git-diff-events"
        style={{
          padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
          background: 'var(--theme-bg-tertiary)',
          borderTop: '1px solid var(--theme-border-primary)',
          fontSize: 'var(--theme-font-size-xs)',
          color: 'var(--theme-text-disabled)',
          maxHeight: '60px',
          overflow: 'auto',
        }}
      >
        <div style={{ marginBottom: '4px' }}>Recent git events:</div>
        {allGitEvents.slice(-5).reverse().map((evt) => (
          <div key={evt.id} style={{ color: 'var(--theme-text-muted)' }}>
            {evt.type} @ {new Date(evt.timestamp).toLocaleTimeString()}
          </div>
        ))}
        {allGitEvents.length === 0 && <span>None yet</span>}
      </div>
    </div>
  );
});

export default GitDiffViewer;
