/**
 * Pipeline Widgets
 *
 * Source widgets and pipeline management widgets for the Unix-pipes data flow system.
 */

import React, { memo, useState, useEffect, useCallback, useRef, useId } from 'react';
import { useEmit } from '../hooks/useEvents';
import { usePersistentState } from '../hooks/useWidgetState';
import type { PipelineConfig, PipelineStats } from '../../types/pipeline';

// ========== Shared Styles ==========

const baseWidgetStyle: React.CSSProperties = {
  padding: 'var(--theme-spacing-sm, 8px)',
  background: 'var(--theme-bg-elevated, #252540)',
  borderRadius: 'var(--theme-radius-sm, 4px)',
  fontSize: '0.85em',
};

// ========== FileDrop Widget ==========

export interface FileDropProps {
  /** Event type to emit when file is dropped */
  channel: string;
  /** Accepted file extensions (e.g., ".csv,.json") */
  accept?: string;
  /** Title/label */
  title?: string;
  /** Show file info after drop */
  showInfo?: boolean;
  /** Multiple files allowed */
  multiple?: boolean;
}

/**
 * FileDrop - Drag and drop zone for files
 *
 * Emits to channel with payload: { filePath, fileName, content, type, size }
 */
export const FileDrop = memo(function FileDrop({
  channel,
  accept,
  title = 'Drop file here',
  showInfo = true,
  multiple = false,
}: FileDropProps): React.ReactElement {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const emit = useEmit();
  const inputRef = useRef<HTMLInputElement>(null);

  // Persist last file - usePersistentState is a drop-in for useState
  const [lastFile, setLastFile] = usePersistentState<{ name: string; size: number } | null>('lastFile', null);

  const handleFiles = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    setError(null);

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (!file) continue;

      // Check file extension if accept is specified
      if (accept) {
        const extensions = accept.split(',').map(e => e.trim().toLowerCase());
        const fileExt = '.' + file.name.split('.').pop()?.toLowerCase();
        if (!extensions.some(ext => ext === fileExt || ext === file.type)) {
          setError(`File type not accepted: ${fileExt}`);
          continue;
        }
      }

      try {
        const content = await file.text();
        const payload = {
          filePath: file.name, // Web API doesn't give full path
          fileName: file.name,
          content,
          type: file.type || 'text/plain',
          size: file.size,
        };

        emit(`file.dropped.${channel}`, payload);
        setLastFile({ name: file.name, size: file.size });
      } catch (err) {
        setError(`Failed to read file: ${(err as Error).message}`);
      }

      if (!multiple) break;
    }
  }, [accept, channel, emit, multiple]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files);
    // Reset input so same file can be selected again
    e.target.value = '';
  }, [handleFiles]);

  return (
    <div
      onClick={handleClick}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      style={{
        ...baseWidgetStyle,
        border: `2px dashed ${isDragging ? 'var(--theme-accent-primary)' : 'var(--theme-border-primary)'}`,
        background: isDragging ? 'rgba(99, 102, 241, 0.1)' : 'var(--theme-bg-elevated)',
        cursor: 'pointer',
        textAlign: 'center',
        padding: '16px',
        transition: 'all 0.2s ease',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        onChange={handleInputChange}
        style={{ display: 'none' }}
      />

      <div style={{ color: 'var(--theme-text-secondary)', marginBottom: 4 }}>
        {title}
      </div>

      {accept && (
        <div style={{ fontSize: '0.8em', color: 'var(--theme-text-muted)' }}>
          Accepts: {accept}
        </div>
      )}

      {error && (
        <div style={{ fontSize: '0.8em', color: 'var(--theme-status-error)', marginTop: 8 }}>
          {error}
        </div>
      )}

      {showInfo && lastFile && !error && (
        <div style={{ fontSize: '0.8em', color: 'var(--theme-text-muted)', marginTop: 8 }}>
          Last: {lastFile.name} ({(lastFile.size / 1024).toFixed(1)} KB)
        </div>
      )}
    </div>
  );
});

// ========== Pipeline Widget ==========

export interface PipelineWidgetProps {
  /** Pipeline configuration */
  config: PipelineConfig;
  /** Auto-start on mount */
  autoStart?: boolean;
  /** Show status UI */
  showStatus?: boolean;
}

/**
 * Pipeline - Invisible widget that runs a data pipeline
 *
 * Connects source events through processor stages to sink.
 */
export const Pipeline = memo(function Pipeline({
  config,
  autoStart = true,
  showStatus = false,
}: PipelineWidgetProps): React.ReactElement | null {
  const [running, setRunning] = useState(false);
  const [stats, setStats] = useState<PipelineStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Start pipeline
  const start = useCallback(async () => {
    if (!window.pipelineAPI) {
      setError('Pipeline API not available');
      return;
    }

    const result = await window.pipelineAPI.start(config);
    if (result.success) {
      setRunning(true);
      setError(null);
    } else {
      setError(result.error ?? 'Failed to start pipeline');
    }
  }, [config]);

  // Stop pipeline
  const stop = useCallback(async () => {
    if (!window.pipelineAPI) return;

    const result = await window.pipelineAPI.stop(config.id);
    if (result.success) {
      setRunning(false);
    }
  }, [config.id]);

  // Auto-start on mount
  useEffect(() => {
    if (autoStart) {
      start();
    }

    return () => {
      // Stop on unmount
      if (window.pipelineAPI) {
        window.pipelineAPI.stop(config.id);
      }
    };
  }, [autoStart, config.id, start]);

  // Poll stats
  useEffect(() => {
    if (!running || !showStatus || !window.pipelineAPI) return;

    const interval = setInterval(async () => {
      const s = await window.pipelineAPI.stats(config.id);
      if (s) setStats(s);
    }, 1000);

    return () => clearInterval(interval);
  }, [running, showStatus, config.id]);

  // No UI if showStatus is false
  if (!showStatus) {
    return null;
  }

  return (
    <div style={{
      ...baseWidgetStyle,
      display: 'flex',
      flexDirection: 'column',
      gap: 8,
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <span style={{ fontWeight: 500 }}>
          {config.name ?? config.id}
        </span>
        <button
          onClick={running ? stop : start}
          style={{
            padding: '4px 12px',
            background: running ? 'var(--theme-status-error)' : 'var(--theme-accent-primary)',
            color: 'white',
            border: 'none',
            borderRadius: 'var(--theme-radius-sm)',
            cursor: 'pointer',
            fontSize: '0.8em',
          }}
        >
          {running ? 'Stop' : 'Start'}
        </button>
      </div>

      {error && (
        <div style={{ color: 'var(--theme-status-error)', fontSize: '0.85em' }}>
          {error}
        </div>
      )}

      {running && stats && (
        <div style={{
          display: 'flex',
          gap: 16,
          fontSize: '0.8em',
          color: 'var(--theme-text-muted)',
        }}>
          <span>In: {stats.inputCount}</span>
          <span>Out: {stats.outputCount}</span>
          {stats.errorCount > 0 && (
            <span style={{ color: 'var(--theme-status-error)' }}>
              Errors: {stats.errorCount}
            </span>
          )}
        </div>
      )}

      <div style={{
        fontSize: '0.75em',
        color: 'var(--theme-text-muted)',
        fontFamily: 'var(--theme-font-mono)',
      }}>
        {config.source} → [{config.stages.map(s => s.processor).join(' → ')}] → {config.sink}
      </div>
    </div>
  );
});

// ========== PipelineStatus Widget ==========

export interface PipelineStatusProps {
  /** Pipeline ID to show status for */
  pipelineId?: string;
  /** Show all pipelines */
  showAll?: boolean;
}

/**
 * PipelineStatus - Shows status of running pipelines
 */
export const PipelineStatus = memo(function PipelineStatus({
  pipelineId,
  showAll = false,
}: PipelineStatusProps): React.ReactElement {
  const [pipelines, setPipelines] = useState<Array<{
    id: string;
    config: PipelineConfig;
    stats: PipelineStats;
  }>>([]);

  useEffect(() => {
    if (!window.pipelineAPI) return;

    const refresh = async () => {
      const list = await window.pipelineAPI.listDetailed();
      if (pipelineId && !showAll) {
        setPipelines(list.filter(p => p.id === pipelineId));
      } else {
        setPipelines(list);
      }
    };

    refresh();
    const interval = setInterval(refresh, 1000);
    return () => clearInterval(interval);
  }, [pipelineId, showAll]);

  if (pipelines.length === 0) {
    return (
      <div style={{ ...baseWidgetStyle, color: 'var(--theme-text-muted)' }}>
        No pipelines running
      </div>
    );
  }

  return (
    <div style={{ ...baseWidgetStyle, display: 'flex', flexDirection: 'column', gap: 8 }}>
      {pipelines.map(p => (
        <div key={p.id} style={{
          padding: 8,
          background: 'var(--theme-bg-tertiary)',
          borderRadius: 'var(--theme-radius-sm)',
        }}>
          <div style={{ fontWeight: 500, marginBottom: 4 }}>
            {p.config.name ?? p.id}
          </div>
          <div style={{ fontSize: '0.8em', color: 'var(--theme-text-muted)' }}>
            In: {p.stats.inputCount} | Out: {p.stats.outputCount}
            {p.stats.errorCount > 0 && (
              <span style={{ color: 'var(--theme-status-error)' }}>
                {' '}| Errors: {p.stats.errorCount}
              </span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
});

// ========== ProcessorList Widget ==========

/**
 * ProcessorList - Shows available processors (for LLM discovery)
 */
export const ProcessorList = memo(function ProcessorList(): React.ReactElement {
  const [processors, setProcessors] = useState<Array<{
    name: string;
    description: string;
  }>>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!window.pipelineAPI) {
      setLoading(false);
      return;
    }

    window.pipelineAPI.describeProcessors()
      .then(setProcessors)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div style={{ ...baseWidgetStyle, color: 'var(--theme-text-muted)' }}>
        Loading processors...
      </div>
    );
  }

  return (
    <div style={{
      ...baseWidgetStyle,
      display: 'flex',
      flexDirection: 'column',
      gap: 4,
      maxHeight: 300,
      overflow: 'auto',
    }}>
      <div style={{
        fontSize: '0.75em',
        color: 'var(--theme-text-muted)',
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        marginBottom: 4,
      }}>
        Available Processors ({processors.length})
      </div>
      {processors.map(p => (
        <div key={p.name} style={{
          padding: '4px 8px',
          background: 'var(--theme-bg-tertiary)',
          borderRadius: 'var(--theme-radius-sm)',
        }}>
          <div style={{ fontWeight: 500, fontFamily: 'var(--theme-font-mono)', fontSize: '0.9em' }}>
            {p.name}
          </div>
          <div style={{ fontSize: '0.8em', color: 'var(--theme-text-muted)' }}>
            {p.description}
          </div>
        </div>
      ))}
    </div>
  );
});

// ========== Inline Pipeline Widget ==========

export interface InlinePipelineProps {
  /** Source event pattern */
  source: string;
  /** Sink event type */
  sink: string;
  /** Processor stages */
  stages: Array<{
    processor: string;
    config?: Record<string, unknown>;
  }>;
  /** Auto-start */
  autoStart?: boolean;
}

/**
 * InlinePipeline - Shorthand for simple pipelines without full config
 */
export const InlinePipeline = memo(function InlinePipeline({
  source,
  sink,
  stages,
  autoStart = true,
}: InlinePipelineProps): React.ReactElement | null {
  const id = useId().replace(/:/g, '');

  const config: PipelineConfig = {
    id: `inline-${id}`,
    source,
    sink,
    stages,
  };

  return <Pipeline config={config} autoStart={autoStart} showStatus={false} />;
});
