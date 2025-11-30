import { Component, ReactNode, ErrorInfo } from 'react';
import type { Theme } from '../../types';

interface ErrorBoundaryProps {
  children: ReactNode;
  theme: Theme;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(_error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({
      error,
      errorInfo
    });
  }

  render(): ReactNode {
    if (this.state.hasError) {
      const { theme } = this.props;

      return (
        <div style={{
          padding: '20px',
          fontFamily: theme?.textBody || 'sans-serif',
          color: theme?.negative || '#ff4757',
          backgroundColor: theme?.widgetBg || 'rgba(22, 27, 34, 0.85)',
          border: `1px solid ${theme?.negative || '#ff4757'}`,
          borderRadius: theme?.widgetRadius || '8px',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          gap: '12px'
        }}>
          <div style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>
            ⚠️ Widget Error
          </div>
          <div style={{
            fontSize: '0.9rem',
            opacity: 0.9,
            fontFamily: 'monospace',
            whiteSpace: 'pre-wrap',
            overflow: 'auto',
            flex: 1
          }}>
            {this.state.error?.toString()}
            {this.state.errorInfo?.componentStack && (
              <details style={{ marginTop: '10px', fontSize: '0.8rem', opacity: 0.7 }}>
                <summary style={{ cursor: 'pointer' }}>Stack Trace</summary>
                <pre>{this.state.errorInfo.componentStack}</pre>
              </details>
            )}
          </div>
          <button
            onClick={() => this.setState({ hasError: false, error: null, errorInfo: null })}
            style={{
              padding: '8px 16px',
              backgroundColor: theme?.accent || '#00d9ff',
              color: theme?.bgApp || '#0d1117',
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontFamily: theme?.textBody || 'sans-serif',
              fontSize: '0.9rem',
              fontWeight: 'bold'
            }}
          >
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
