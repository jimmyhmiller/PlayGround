import { memo, useEffect, useRef, useState } from 'react';
import { EditorView, basicSetup } from 'codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { oneDark } from '@codemirror/theme-one-dark';
import { EditorState } from '@codemirror/state';
import { useLatestEvent, useEmit } from '../hooks/useEvents';
import type { DashboardEvent } from '../../types/events';

interface FileContentPayload {
  filePath: string;
  content: string;
  instanceId?: string;
}

interface CodeMirrorEditorProps {
  instanceId?: string;
  windowId?: string;
  subscribePattern?: string;
  filePath?: string | null;
  content?: string | null;
  onUpdateProps?: (props: { filePath?: string; content?: string }) => void;
  initialContent?: string;
}

/**
 * CodeMirror Editor Component
 *
 * Subscribes to file events and displays/edits code.
 * Persists content to window props for state survival.
 */
const CodeMirrorEditor = memo(function CodeMirrorEditor({
  instanceId,
  windowId: _windowId,
  subscribePattern = 'file.**',
  filePath = null,
  content = null,
  onUpdateProps,
  initialContent = '// Start typing or load a file...',
}: CodeMirrorEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<EditorView | null>(null);
  const [currentPath, setCurrentPath] = useState(filePath);
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const renderCount = useRef(0);
  renderCount.current += 1;

  const emit = useEmit();

  // Subscribe to file events
  const fileEvent = useLatestEvent(subscribePattern) as DashboardEvent<FileContentPayload> | null;

  // Keep a ref to onUpdateProps so we always have the latest
  const onUpdatePropsRef = useRef(onUpdateProps);
  useEffect(() => {
    onUpdatePropsRef.current = onUpdateProps;
  }, [onUpdateProps]);

  // Debounced save to window props
  const saveContent = useRef((newContent: string) => {
    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }
    saveTimeoutRef.current = setTimeout(() => {
      onUpdatePropsRef.current?.({ content: newContent });
    }, 500); // 500ms debounce for typing
  }).current;

  // Initialize CodeMirror
  useEffect(() => {
    if (!containerRef.current) return;

    // Use persisted content if available, otherwise initial
    const startContent = content ?? initialContent;

    const state = EditorState.create({
      doc: startContent,
      extensions: [
        basicSetup,
        javascript(),
        oneDark,
        EditorView.updateListener.of((update) => {
          if (update.docChanged) {
            const newContent = update.state.doc.toString();
            // Emit event for external listeners
            if (currentPath) {
              emit('editor.content.changed', {
                filePath: currentPath,
                content: newContent,
                instanceId,
              });
            }
            // Persist to window props
            saveContent(newContent);
          }
        }),
      ],
    });

    const view = new EditorView({
      state,
      parent: containerRef.current,
    });

    editorRef.current = view;

    return () => {
      // Clear any pending save
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
      view.destroy();
    };
  }, []); // Only run once on mount - content is loaded from props

  // Update editor when file events come in
  useEffect(() => {
    if (!fileEvent || !editorRef.current) return;

    const { type, payload } = fileEvent;

    // Handle file content events
    if (type === 'file.content.loaded' || type === 'file.content.changed') {
      // If we have a filePath filter, only update if it matches
      if (filePath && payload.filePath !== filePath) {
        return;
      }

      setCurrentPath(payload.filePath);

      const view = editorRef.current;
      view.dispatch({
        changes: {
          from: 0,
          to: view.state.doc.length,
          insert: payload.content,
        },
      });

      // Also persist the loaded file path and content
      onUpdatePropsRef.current?.({
        filePath: payload.filePath,
        content: payload.content,
      });
    }
  }, [fileEvent, filePath]);

  return (
    <div
      className="code-editor"
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: 'var(--theme-code-bg)',
      }}
    >
      <div
        className="code-editor-header"
        style={{
          padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
          background: 'var(--theme-bg-tertiary)',
          color: 'var(--theme-text-muted)',
          fontSize: 'var(--theme-font-size-sm)',
          fontFamily: 'var(--theme-font-mono)',
          display: 'flex',
          justifyContent: 'space-between',
          borderBottom: '1px solid var(--theme-border-primary)',
        }}
      >
        <span>{currentPath || 'No file'}</span>
        <span style={{ color: 'var(--theme-accent-warning)' }}>
          renders: {renderCount.current}
        </span>
      </div>
      <div
        ref={containerRef}
        className="code-editor-content"
        style={{
          flex: 1,
          overflow: 'auto',
        }}
      />
    </div>
  );
});

export default CodeMirrorEditor;
