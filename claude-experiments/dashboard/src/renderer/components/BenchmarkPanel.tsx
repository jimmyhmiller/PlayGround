/**
 * BenchmarkPanel
 *
 * Main benchmarking interface with multiple code variants,
 * execution controls, and result visualization.
 */

import { memo, useState, useCallback, useRef, useEffect } from 'react';
import { EditorView } from '@codemirror/view';
import { basicSetup } from 'codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { oneDark } from '@codemirror/theme-one-dark';
import { EditorState } from '@codemirror/state';
import BenchmarkChart from './BenchmarkChart';
import BenchmarkTable from './BenchmarkTable';
import {
  runBenchmarks,
  createVariant,
  DEFAULT_VARIANTS,
} from '../services/benchmarkRunner';
import type {
  BenchmarkVariant,
  BenchmarkVariantResult,
  BenchmarkConfig,
} from './benchmarkTypes';
import { DEFAULT_BENCHMARK_CONFIG } from './benchmarkTypes';

interface BenchmarkPanelProps {
  instanceId?: string;
  windowId?: string;
  iterations?: number;
}

/**
 * BenchmarkPanel Component
 */
const BenchmarkPanel = memo(function BenchmarkPanel({
  instanceId: _instanceId,
  windowId: _windowId,
  iterations = 100,
}: BenchmarkPanelProps) {
  const [variants, setVariants] = useState<BenchmarkVariant[]>(DEFAULT_VARIANTS);
  const [activeVariantId, setActiveVariantId] = useState<string>(variants[0]?.id ?? '');
  const [baselineId, setBaselineId] = useState<string>(variants[0]?.id ?? '');
  const [results, setResults] = useState<BenchmarkVariantResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState<string>('');

  const [config] = useState<BenchmarkConfig>({
    ...DEFAULT_BENCHMARK_CONFIG,
    iterations,
  });

  const editorContainerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<EditorView | null>(null);

  const activeVariant = variants.find((v) => v.id === activeVariantId);

  // Initialize editor when active variant changes
  useEffect(() => {
    if (!editorContainerRef.current || !activeVariant) return;

    // Destroy existing editor
    if (editorRef.current) {
      editorRef.current.destroy();
    }

    const state = EditorState.create({
      doc: activeVariant.code,
      extensions: [
        basicSetup,
        javascript(),
        oneDark,
        EditorView.updateListener.of((update) => {
          if (update.docChanged) {
            const newCode = update.state.doc.toString();
            setVariants((prev) =>
              prev.map((v) =>
                v.id === activeVariantId ? { ...v, code: newCode } : v
              )
            );
          }
        }),
        EditorView.theme({
          '&': { height: '100%' },
          '.cm-scroller': { overflow: 'auto' },
        }),
      ],
    });

    const view = new EditorView({
      state,
      parent: editorContainerRef.current,
    });

    editorRef.current = view;

    return () => {
      view.destroy();
    };
  }, [activeVariantId]);

  // Update editor content when switching variants
  useEffect(() => {
    if (!editorRef.current || !activeVariant) return;

    const currentContent = editorRef.current.state.doc.toString();
    if (currentContent !== activeVariant.code) {
      editorRef.current.dispatch({
        changes: {
          from: 0,
          to: currentContent.length,
          insert: activeVariant.code,
        },
      });
    }
  }, [activeVariant?.code]);

  const handleAddVariant = useCallback(() => {
    const newVariant = createVariant(
      `Variant ${variants.length + 1}`,
      '// Write your code here\n'
    );
    setVariants((prev) => [...prev, newVariant]);
    setActiveVariantId(newVariant.id);
  }, [variants.length]);

  const handleRemoveVariant = useCallback(
    (id: string) => {
      if (variants.length <= 1) return;

      setVariants((prev) => prev.filter((v) => v.id !== id));

      if (activeVariantId === id) {
        const remaining = variants.filter((v) => v.id !== id);
        setActiveVariantId(remaining[0]?.id ?? '');
      }

      if (baselineId === id) {
        const remaining = variants.filter((v) => v.id !== id);
        setBaselineId(remaining[0]?.id ?? '');
      }
    },
    [variants, activeVariantId, baselineId]
  );

  const handleSetBaseline = useCallback((id: string) => {
    setBaselineId(id);
  }, []);

  const handleRunBenchmark = useCallback(async () => {
    setIsRunning(true);
    setProgress('Starting...');
    setResults([]);

    try {
      const benchmarkResults = await runBenchmarks(variants, config, (completed, total, current) => {
        setProgress(`Running ${current} (${completed}/${total})`);
      });

      setResults(benchmarkResults);
      setProgress('');
    } catch (err) {
      setProgress(`Error: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsRunning(false);
    }
  }, [variants, config]);

  const handleClearResults = useCallback(() => {
    setResults([]);
    setProgress('');
  }, []);

  return (
    <div className="benchmark-panel">
      <div className="benchmark-header">
        <span style={{ fontWeight: 500 }}>Benchmark</span>
        <span style={{ fontSize: '0.85em', color: 'var(--theme-text-muted)' }}>
          {config.iterations} iterations, {config.warmupIterations} warmup
        </span>
      </div>

      <div className="benchmark-tabs">
        {variants.map((variant) => (
          <button
            key={variant.id}
            className={`benchmark-tab ${variant.id === activeVariantId ? 'active' : ''} ${variant.id === baselineId ? 'baseline' : ''}`}
            onClick={() => setActiveVariantId(variant.id)}
            onDoubleClick={() => handleSetBaseline(variant.id)}
            title={variant.id === baselineId ? 'Baseline' : 'Double-click to set as baseline'}
          >
            {variant.name}
            {variants.length > 1 && (
              <span
                style={{
                  marginLeft: '8px',
                  opacity: 0.5,
                  cursor: 'pointer',
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemoveVariant(variant.id);
                }}
              >
                x
              </span>
            )}
          </button>
        ))}
        <button
          className="benchmark-tab"
          onClick={handleAddVariant}
          style={{ opacity: 0.6 }}
        >
          + Add
        </button>
      </div>

      <div className="benchmark-editors">
        <div
          ref={editorContainerRef}
          className="benchmark-editor-container"
          style={{ flex: 1 }}
        />
      </div>

      <div className="benchmark-controls">
        <button
          className="benchmark-btn"
          onClick={handleRunBenchmark}
          disabled={isRunning}
        >
          {isRunning ? 'Running...' : 'Run Benchmark'}
        </button>
        <button
          className="benchmark-btn secondary"
          onClick={handleClearResults}
          disabled={isRunning || results.length === 0}
        >
          Clear Results
        </button>
        {progress && (
          <span style={{ marginLeft: '12px', color: 'var(--theme-text-muted)', fontSize: '0.9em' }}>
            {progress}
          </span>
        )}
      </div>

      {results.length > 0 && (
        <div className="benchmark-results">
          <BenchmarkChart results={results} baselineId={baselineId} />
          <BenchmarkTable results={results} baselineId={baselineId} />
        </div>
      )}
    </div>
  );
});

export default BenchmarkPanel;
