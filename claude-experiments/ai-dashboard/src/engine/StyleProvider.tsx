// Style Provider - injects agent-generated CSS into the page

import { ReactNode, useEffect } from 'react';
import { GeneratedStyle } from '../types/theme';

interface StyleProviderProps {
  style: GeneratedStyle | null;
  children: ReactNode;
}

export function StyleProvider({ style, children }: StyleProviderProps) {
  useEffect(() => {
    if (!style) return;

    // Inject agent-generated CSS
    let styleEl = document.getElementById('agent-styles') as HTMLStyleElement;

    if (!styleEl) {
      styleEl = document.createElement('style');
      styleEl.id = 'agent-styles';
      document.head.appendChild(styleEl);
    }

    // Smoothly transition between styles
    styleEl.textContent = style.css;

    console.log(`[StyleProvider] Applied CSS, length: ${style.css.length}, id: ${style.id}`);
  }, [style]);

  return <>{children}</>;
}

// SVG Definitions Provider
export function SVGDefs({ content }: { content?: string }) {
  if (!content) return null;

  return (
    <svg
      style={{
        position: 'absolute',
        width: 0,
        height: 0,
        pointerEvents: 'none',
      }}
      aria-hidden="true"
    >
      <defs dangerouslySetInnerHTML={{ __html: content }} />
    </svg>
  );
}
