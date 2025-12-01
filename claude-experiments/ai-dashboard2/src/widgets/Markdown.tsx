import { FC, useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';

interface MarkdownConfig {
  id: string;
  type: 'markdown';
  label: string;
  content?: string;
  filePath?: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export const Markdown: FC<BaseWidgetComponentProps> = ({ theme, config, dashboard }) => {
  const markdownConfig = config as MarkdownConfig;
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Extract only the project root to avoid re-rendering on every dashboard change
  const projectRoot = dashboard?._projectRoot;

  useEffect(() => {
    // If markdown content is provided inline, use it directly
    if (markdownConfig.content !== undefined) {
      setContent(markdownConfig.content);
      setLoading(false);
      setError(null);
      return;
    }

    // If filePath is provided, load from file
    if (markdownConfig.filePath) {
      setLoading(true);
      setError(null);

      // Resolve relative paths based on project root
      let fullPath = markdownConfig.filePath;
      if (!fullPath.startsWith('/') && projectRoot) {
        fullPath = `${projectRoot}/${markdownConfig.filePath}`;
      }

      // Use the dashboard API to load the file as text
      if (window.dashboardAPI && window.dashboardAPI.loadTextFile) {
        console.log('[Markdown] Loading file via dashboardAPI:', fullPath);
        window.dashboardAPI.loadTextFile(fullPath)
          .then(text => {
            console.log('[Markdown] Loaded text, length:', text.length);
            setContent(text);
            setLoading(false);
          })
          .catch(err => {
            console.error('[Markdown] Failed to load file from', fullPath, err);
            setError(err.message || 'Failed to load file');
            setLoading(false);
          });
      } else {
        console.error('[Markdown] dashboardAPI.loadTextFile not available');
        setError('File loading API not available');
        setLoading(false);
      }
    }
  }, [markdownConfig.content, markdownConfig.filePath, projectRoot]);

  if (loading) {
    return (
      <>
        <div className="widget-label" style={{ fontFamily: theme.textBody }}>
          {markdownConfig.label}
        </div>
        <div style={{
          fontFamily: theme.textBody,
          color: theme.textColor,
          opacity: 0.6,
          padding: '20px',
          textAlign: 'center'
        }}>
          Loading...
        </div>
      </>
    );
  }

  if (error) {
    return (
      <>
        <div className="widget-label" style={{ fontFamily: theme.textBody }}>
          {markdownConfig.label}
        </div>
        <div style={{
          fontFamily: theme.textBody,
          color: theme.negative,
          padding: '20px'
        }}>
          ⚠️ Error: {error}
        </div>
      </>
    );
  }

  if (!content) {
    return (
      <>
        <div className="widget-label" style={{ fontFamily: theme.textBody }}>
          {markdownConfig.label}
        </div>
        <div style={{
          fontFamily: theme.textBody,
          color: theme.textColor,
          opacity: 0.6,
          padding: '20px',
          textAlign: 'center'
        }}>
          No content provided. Add "content" or "filePath" to widget config.
        </div>
      </>
    );
  }

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>
        {markdownConfig.label}
      </div>
      <div className="markdown-widget" style={{
        fontFamily: theme.textBody,
        color: theme.textColor,
        fontSize: '0.85rem',
        lineHeight: '1.6',
        padding: '12px',
        overflowY: 'auto',
        height: 'calc(100% - 40px)'
      }}>
        <ReactMarkdown
          components={{
            code({ node, inline, className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || '');
              return !inline && match ? (
                <SyntaxHighlighter
                  style={oneDark}
                  language={match[1]}
                  PreTag="div"
                  {...props}
                >
                  {String(children).replace(/\n$/, '')}
                </SyntaxHighlighter>
              ) : (
                <code
                  className={className}
                  style={{
                    backgroundColor: 'rgba(255,255,255,0.1)',
                    padding: '2px 6px',
                    borderRadius: '3px',
                    fontFamily: theme.textCode
                  }}
                  {...props}
                >
                  {children}
                </code>
              );
            }
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    </>
  );
};
