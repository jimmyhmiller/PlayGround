import { FC, useState } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';

export const FlippableTest: FC<BaseWidgetComponentProps> = ({ theme, config }) => {
  const [isFlipped, setIsFlipped] = useState(false);

  return (
    <div
      className={`widget-flip-container ${isFlipped ? 'flipped' : ''}`}
      style={{ width: '100%', height: '100%' }}
      onContextMenu={(e) => {
        e.preventDefault();
        setIsFlipped(!isFlipped);
      }}
    >
      <div className="widget-flip-inner">
        {/* Front - Just apply .widget class like normal */}
        <div
          className="widget widget-flip-front"
          style={{
            background: theme.widgetBg,
            border: theme.widgetBorder,
            borderRadius: theme.widgetRadius,
          }}
        >
          <div className="widget-label" style={{ fontFamily: theme.textBody }}>
            Flippable Test Widget
          </div>
          <p style={{ marginBottom: '8px', color: theme.accent, fontSize: '0.85rem' }}>
            Right-click to flip and see JSON
          </p>
          {[...Array(20)].map((_, i) => (
            <div key={i} style={{
              padding: '8px 12px',
              marginBottom: '4px',
              background: 'rgba(255,255,255,0.05)',
              borderRadius: '4px',
              border: '1px solid rgba(255,255,255,0.1)',
              color: theme.textColor,
              fontSize: '0.85rem',
            }}>
              Item {i + 1} - This is scrollable content to test the flip animation
            </div>
          ))}
        </div>

        {/* Back - Just apply .widget class like normal */}
        <div
          className="widget widget-flip-back"
          style={{
            background: theme.widgetBg,
            border: theme.widgetBorder,
            borderRadius: theme.widgetRadius,
          }}
        >
          <div className="widget-label" style={{ fontFamily: theme.textBody, marginBottom: '12px' }}>
            Back Side - JSON Configuration
          </div>
          <textarea
            style={{
              width: '100%',
              height: 'calc(100% - 40px)',
              background: 'rgba(0,0,0,0.3)',
              border: '1px solid rgba(255,255,255,0.1)',
              borderRadius: '4px',
              padding: '12px',
              color: theme.textColor,
              fontFamily: 'monospace',
              fontSize: '0.75rem',
              resize: 'none',
            }}
            value={JSON.stringify(config, null, 2)}
            readOnly
          />
        </div>
      </div>
    </div>
  );
};
