/**
 * ErrorBoundary - Catches React errors and displays fallback UI
 */

import React, { Component, ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    console.error('[ErrorBoundary] Caught error:', error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div style={{
          padding: '12px',
          backgroundColor: 'var(--theme-bg-secondary, #1a1a2e)',
          border: '1px solid var(--theme-status-error, #f44)',
          borderRadius: 'var(--theme-radius-md, 6px)',
          color: 'var(--theme-status-error, #f44)',
          fontSize: '0.85em',
          fontFamily: 'monospace',
        }}>
          <div style={{ fontWeight: 600, marginBottom: '8px' }}>
            Component Error
          </div>
          <div style={{
            color: 'var(--theme-text-muted, #888)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}>
            {this.state.error?.message || 'Unknown error'}
          </div>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              marginTop: '8px',
              padding: '4px 8px',
              backgroundColor: 'transparent',
              border: '1px solid var(--theme-border-primary, #333)',
              borderRadius: '4px',
              color: 'var(--theme-text-muted, #888)',
              cursor: 'pointer',
              fontSize: '0.85em',
            }}
          >
            Retry
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * WidgetErrorBoundary - Specialized error boundary for widgets
 * Shows a compact error message suitable for dashboard widgets
 */
export class WidgetErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    console.error('[WidgetErrorBoundary] Caught error:', error, errorInfo);
    this.props.onError?.(error, errorInfo);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div style={{
          padding: '8px 12px',
          backgroundColor: 'rgba(244, 67, 54, 0.1)',
          border: '1px solid var(--theme-status-error, #f44)',
          borderRadius: 'var(--theme-radius-sm, 4px)',
          color: 'var(--theme-status-error, #f44)',
          fontSize: '0.8em',
        }}>
          <span style={{ fontWeight: 500 }}>Error: </span>
          {this.state.error?.message || 'Component failed to render'}
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              marginLeft: '8px',
              padding: '2px 6px',
              backgroundColor: 'transparent',
              border: '1px solid currentColor',
              borderRadius: '3px',
              color: 'inherit',
              cursor: 'pointer',
              fontSize: '0.9em',
            }}
          >
            ↻
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
