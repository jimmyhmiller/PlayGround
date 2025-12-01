import { FC, useState, useEffect, useRef } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';

interface WebViewConfig {
  id: string;
  type: 'webView' | 'web-view';
  label: string;
  url?: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export const WebView: FC<BaseWidgetComponentProps> = (props) => {
  const { theme, config } = props;
  const webViewConfig = config as WebViewConfig;
  const [currentUrl, setCurrentUrl] = useState(webViewConfig.url || '');
  const [inputUrl, setInputUrl] = useState(webViewConfig.url || '');
  const [canGoBack, setCanGoBack] = useState(false);
  const [canGoForward, setCanGoForward] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const widgetId = config.id;

  // Create WebContentsView on mount
  useEffect(() => {
    const createView = async () => {
      if (!(window as any).webContentsViewAPI || !containerRef.current) return;

      // Get bounds from container
      const rect = containerRef.current.getBoundingClientRect();
      const bounds = {
        x: Math.round(rect.left),
        y: Math.round(rect.top),
        width: Math.round(rect.width),
        height: Math.round(rect.height)
      };

      // Use widget background color from theme, fallback to bgApp
      // Convert rgba to rgb by removing alpha channel if present
      let backgroundColor = theme.widgetBg || theme.bgApp;
      if (typeof backgroundColor === 'string' && backgroundColor.startsWith('rgba')) {
        // Extract rgb values from rgba(r, g, b, a) format
        const match = backgroundColor.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
        if (match) {
          backgroundColor = `rgb(${match[1]}, ${match[2]}, ${match[3]})`;
        }
      }

      console.log('[WebView] Creating WebContentsView:', { widgetId, bounds, url: currentUrl, backgroundColor });

      await (window as any).webContentsViewAPI.create(widgetId, currentUrl, bounds, backgroundColor);

      // Check initial navigation state
      const navState = await (window as any).webContentsViewAPI.canNavigate(widgetId);
      if (navState?.success) {
        setCanGoBack(navState.canGoBack || false);
        setCanGoForward(navState.canGoForward || false);
      }
    };

    createView();

    // Listen for navigation events
    const handleNavigated = ({ widgetId: navWidgetId, url, canGoBack: navCanGoBack, canGoForward: navCanGoForward }: any) => {
      if (navWidgetId === widgetId) {
        console.log('[WebView] Navigated to:', url);
        setCurrentUrl(url);
        setInputUrl(url);
        setCanGoBack(navCanGoBack || false);
        setCanGoForward(navCanGoForward || false);
      }
    };

    const handler = (window as any).webContentsViewAPI.onNavigated(handleNavigated);

    // Cleanup on unmount
    return () => {
      (window as any).webContentsViewAPI.offNavigated(handler);
      (window as any).webContentsViewAPI.destroy(widgetId);
    };
  }, [widgetId]);

  // Update bounds when container resizes
  useEffect(() => {
    if (!containerRef.current || !(window as any).webContentsViewAPI) return;

    const resizeObserver = new ResizeObserver(() => {
      const rect = containerRef.current!.getBoundingClientRect();
      const bounds = {
        x: Math.round(rect.left),
        y: Math.round(rect.top),
        width: Math.round(rect.width),
        height: Math.round(rect.height)
      };

      (window as any).webContentsViewAPI.updateBounds(widgetId, bounds);
    });

    resizeObserver.observe(containerRef.current);

    return () => resizeObserver.disconnect();
  }, [widgetId]);

  const handleNavigate = async () => {
    let url = inputUrl.trim();
    if (url && !url.startsWith('http://') && !url.startsWith('https://')) {
      url = 'http://' + url;
    }

    if ((window as any).webContentsViewAPI) {
      await (window as any).webContentsViewAPI.navigate(widgetId, url);
    }
  };

  const handleReload = async () => {
    if ((window as any).webContentsViewAPI) {
      await (window as any).webContentsViewAPI.reload(widgetId);
    }
  };

  const goBack = async () => {
    if ((window as any).webContentsViewAPI) {
      const result = await (window as any).webContentsViewAPI.goBack(widgetId);
      if (!result.success) {
        setCanGoBack(false);
      }
    }
  };

  const goForward = async () => {
    if ((window as any).webContentsViewAPI) {
      const result = await (window as any).webContentsViewAPI.goForward(widgetId);
      if (!result.success) {
        setCanGoForward(false);
      }
    }
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div className="widget-label" style={{ fontFamily: theme.textBody, marginBottom: 8 }}>{webViewConfig.label}</div>

      {/* Navigation Bar */}
      <div style={{
        display: 'flex',
        gap: 6,
        marginBottom: 8,
        alignItems: 'center'
      }}>
        <button
          onClick={goBack}
          disabled={!canGoBack}
          style={{
            background: 'rgba(255,255,255,0.1)',
            border: `1px solid ${theme.accent}44`,
            borderRadius: 4,
            padding: '6px 10px',
            color: theme.textColor,
            cursor: !canGoBack ? 'not-allowed' : 'pointer',
            fontSize: '0.75rem',
            opacity: !canGoBack ? 0.4 : 1
          }}
          title="Back"
        >
          ←
        </button>

        <button
          onClick={goForward}
          disabled={!canGoForward}
          style={{
            background: 'rgba(255,255,255,0.1)',
            border: `1px solid ${theme.accent}44`,
            borderRadius: 4,
            padding: '6px 10px',
            color: theme.textColor,
            cursor: !canGoForward ? 'not-allowed' : 'pointer',
            fontSize: '0.75rem',
            opacity: !canGoForward ? 0.4 : 1
          }}
          title="Forward"
        >
          →
        </button>

        <button
          onClick={handleReload}
          style={{
            background: 'rgba(255,255,255,0.1)',
            border: `1px solid ${theme.accent}44`,
            borderRadius: 4,
            padding: '6px 10px',
            color: theme.textColor,
            cursor: 'pointer',
            fontSize: '0.75rem'
          }}
          title="Reload"
        >
          ↻
        </button>

        <input
          type="text"
          value={inputUrl}
          onChange={(e) => setInputUrl(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleNavigate()}
          placeholder="Enter URL..."
          style={{
            flex: 1,
            background: 'rgba(255,255,255,0.05)',
            border: `1px solid ${theme.accent}44`,
            borderRadius: 4,
            padding: '6px 10px',
            color: theme.textColor,
            fontFamily: theme.textCode,
            fontSize: '0.75rem',
            outline: 'none'
          }}
        />

        <button
          onClick={handleNavigate}
          style={{
            background: theme.accent,
            border: 'none',
            borderRadius: 4,
            padding: '6px 12px',
            color: '#000',
            cursor: 'pointer',
            fontSize: '0.75rem',
            fontWeight: 600
          }}
        >
          Go
        </button>
      </div>

      {/* WebContentsView Placeholder */}
      <div
        ref={containerRef}
        style={{
          flex: 1,
          position: 'relative',
          border: `1px solid ${theme.accent}22`,
          borderRadius: 4,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        {!currentUrl && (
          <div style={{
            fontFamily: theme.textBody,
            color: theme.neutral,
            fontSize: '0.8rem',
            padding: '12px',
            textAlign: 'center',
            position: 'absolute'
          }}>
            Enter a URL to navigate
          </div>
        )}
      </div>
    </div>
  );
};
