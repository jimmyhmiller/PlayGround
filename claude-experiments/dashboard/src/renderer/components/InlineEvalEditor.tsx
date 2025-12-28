/**
 * InlineEvalEditor
 *
 * A CodeMirror-based editor with inline evaluation results.
 * Supports keyboard shortcuts for evaluating code and displaying results inline.
 *
 * Keyboard shortcuts:
 * - Cmd/Ctrl+Enter: Evaluate current line
 * - Cmd/Ctrl+Shift+Enter: Evaluate selection
 * - Cmd/Ctrl+Shift+C: Clear all results
 */

import { memo, useEffect, useRef, useCallback } from 'react';
import { EditorView, keymap } from '@codemirror/view';
import { basicSetup } from 'codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { oneDark } from '@codemirror/theme-one-dark';
import { EditorState } from '@codemirror/state';
import {
  inlineResultsExtension,
  setLoading,
  setResult,
  clearAllResults,
} from './InlineResultWidget';

interface InlineEvalEditorProps {
  instanceId?: string;
  windowId?: string;
  initialContent?: string;
  language?: 'javascript' | 'typescript';
}

/**
 * InlineEvalEditor Component
 *
 * Light Table-style editor with inline evaluation results.
 */
const InlineEvalEditor = memo(function InlineEvalEditor({
  instanceId: _instanceId,
  windowId: _windowId,
  initialContent = '// Try evaluating some code!\n// Press Cmd+Enter to evaluate a line\n\n1 + 1\n\nMath.PI * 2\n\n[1, 2, 3].map(x => x * 2)\n\n({ name: "Light Table", year: 2012 })\n',
  language = 'javascript',
}: InlineEvalEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<EditorView | null>(null);

  /**
   * Evaluate code and update the editor with results
   */
  const evaluateCode = useCallback(async (view: EditorView, code: string, line: number) => {
    // Show loading state
    setLoading(view, line);

    try {
      const result = await window.evalAPI.execute(code, language);
      setResult(view, line, result);
    } catch (err) {
      setResult(view, line, {
        id: `error-${Date.now()}`,
        success: false,
        displayValue: err instanceof Error ? err.message : 'Unknown error',
        type: 'error',
        executionTimeMs: 0,
        error: err instanceof Error ? err.message : 'Unknown error',
      });
    }
  }, [language]);

  /**
   * Evaluate the current line
   */
  const evaluateLine = useCallback((view: EditorView): boolean => {
    const { state } = view;
    const { from } = state.selection.main;
    const line = state.doc.lineAt(from);
    const code = line.text.trim();

    if (code && !code.startsWith('//')) {
      evaluateCode(view, code, line.number);
    }

    return true;
  }, [evaluateCode]);

  /**
   * Evaluate the current selection or all lines if no selection
   */
  const evaluateSelection = useCallback((view: EditorView): boolean => {
    const { state } = view;
    const { from, to } = state.selection.main;

    if (from === to) {
      // No selection - evaluate all non-empty, non-comment lines
      for (let i = 1; i <= state.doc.lines; i++) {
        const line = state.doc.line(i);
        const code = line.text.trim();
        if (code && !code.startsWith('//')) {
          evaluateCode(view, code, i);
        }
      }
    } else {
      // Evaluate selection
      const selectedText = state.sliceDoc(from, to);
      const startLine = state.doc.lineAt(from);
      evaluateCode(view, selectedText, startLine.number);
    }

    return true;
  }, [evaluateCode]);

  /**
   * Clear all results
   */
  const handleClearResults = useCallback((view: EditorView): boolean => {
    clearAllResults(view);
    return true;
  }, []);

  // Initialize CodeMirror
  useEffect(() => {
    if (!containerRef.current) return;

    const customKeymap = keymap.of([
      {
        key: 'Mod-Enter',
        run: evaluateLine,
      },
      {
        key: 'Mod-Shift-Enter',
        run: evaluateSelection,
      },
      {
        key: 'Mod-Shift-c',
        run: handleClearResults,
      },
    ]);

    const state = EditorState.create({
      doc: initialContent,
      extensions: [
        basicSetup,
        javascript(),
        oneDark,
        inlineResultsExtension(),
        customKeymap,
        EditorView.theme({
          '&': {
            height: '100%',
          },
          '.cm-scroller': {
            overflow: 'auto',
          },
        }),
      ],
    });

    const view = new EditorView({
      state,
      parent: containerRef.current,
    });

    editorRef.current = view;

    return () => {
      view.destroy();
    };
  }, []); // Only run once on mount

  return (
    <div
      className="inline-eval-editor"
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: 'var(--theme-code-bg)',
      }}
    >
      <div
        className="inline-eval-header"
        style={{
          padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
          background: 'var(--theme-bg-tertiary)',
          color: 'var(--theme-text-muted)',
          fontSize: 'var(--theme-font-size-sm)',
          fontFamily: 'var(--theme-font-mono)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '1px solid var(--theme-border-primary)',
        }}
      >
        <span>Inline Eval ({language})</span>
        <div style={{ display: 'flex', gap: '12px', fontSize: '0.85em' }}>
          <span style={{ opacity: 0.7 }}>⌘↵ eval line</span>
          <span style={{ opacity: 0.7 }}>⌘⇧↵ eval all</span>
          <span style={{ opacity: 0.7 }}>⌘⇧C clear</span>
        </div>
      </div>
      <div
        ref={containerRef}
        className="inline-eval-content"
        style={{
          flex: 1,
          overflow: 'auto',
        }}
      />
    </div>
  );
});

export default InlineEvalEditor;
