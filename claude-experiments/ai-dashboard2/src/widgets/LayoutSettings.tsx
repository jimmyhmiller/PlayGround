import { FC, useState, useEffect } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';
import type { LayoutSettings as LayoutSettingsType } from '../types';

interface LayoutSettingsConfig {
  id: string;
  type: 'layoutSettings' | 'layout-settings';
  label?: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export const LayoutSettings: FC<BaseWidgetComponentProps> = (props) => {
  const { theme, config, dashboardId, layout } = props;
  const settingsConfig = config as LayoutSettingsConfig;
  const [widgetGap, setWidgetGap] = useState(layout?.widgetGap ?? 10);
  const [buffer, setBuffer] = useState(layout?.buffer ?? 20);
  const [layoutMode, setLayoutMode] = useState(layout?.mode || 'single-pane');


  // Update local state when layout prop changes
  useEffect(() => {
    if (layout?.mode && layout.mode !== layoutMode) {
      setLayoutMode(layout.mode);
    }
  }, [layout?.mode]);

  const handleGapChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    setWidgetGap(value);
    if ((window as any).dashboardAPI && dashboardId) {
      (window as any).dashboardAPI.updateLayoutSettings(dashboardId, { widgetGap: value });
    }
  };

  const handleBufferChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    setBuffer(value);
    if ((window as any).dashboardAPI && dashboardId) {
      (window as any).dashboardAPI.updateLayoutSettings(dashboardId, { buffer: value });
    }
  };

  const handleModeChange = (mode: string) => {
    console.log('[LayoutSettings] Changing mode to:', mode);
    console.log('[LayoutSettings] DashboardId:', dashboardId);
    setLayoutMode(mode);
    if ((window as any).dashboardAPI && dashboardId) {
      console.log('[LayoutSettings] Calling updateLayoutSettings...');
      (window as any).dashboardAPI.updateLayoutSettings(dashboardId, { mode })
        .then(() => console.log('[LayoutSettings] Update successful'))
        .catch((err: any) => console.error('[LayoutSettings] Update failed:', err));
    } else {
      console.error('[LayoutSettings] dashboardAPI not available or no dashboardId');
    }
  };

  const modes = [
    { value: 'single-pane', label: '📄 Single Pane', description: 'Fixed viewport, no scrolling' },
    { value: 'infinite-canvas', label: '🎨 Infinite Canvas', description: 'Hold Option/Alt to pan around' },
    { value: 'vertical-scroll', label: '↕️ Vertical Scroll', description: 'Scroll up and down' },
    { value: 'horizontal-scroll', label: '↔️ Horizontal Scroll', description: 'Scroll left and right' }
  ];

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>Layout Settings</div>
      <div style={{ fontFamily: theme.textBody, fontSize: '0.85rem', color: theme.textColor, padding: '12px' }}>
        {/* Layout Mode Selector */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ marginBottom: 8, fontWeight: 'bold', color: theme.accent }}>Layout Mode</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {modes.map(mode => (
              <div
                key={mode.value}
                onClick={() => handleModeChange(mode.value)}
                style={{
                  padding: '10px 12px',
                  borderRadius: '6px',
                  border: `2px solid ${layoutMode === mode.value ? theme.accent : 'rgba(255,255,255,0.1)'}`,
                  backgroundColor: layoutMode === mode.value ? `${theme.accent}22` : 'rgba(255,255,255,0.05)',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onMouseEnter={(e) => {
                  if (layoutMode !== mode.value) {
                    e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.08)';
                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (layoutMode !== mode.value) {
                    e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)';
                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)';
                  }
                }}
              >
                <div style={{ fontWeight: 'bold', marginBottom: 2 }}>{mode.label}</div>
                <div style={{ fontSize: '0.75rem', opacity: 0.7 }}>{mode.description}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Widget Gap Slider */}
        <div style={{ marginBottom: 15 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
            <label>Widget Gap</label>
            <span style={{ color: theme.accent }}>{widgetGap}px</span>
          </div>
          <input
            type="range"
            min="0"
            max="40"
            value={widgetGap}
            onChange={handleGapChange}
            style={{ width: '100%', accentColor: theme.accent }}
          />
        </div>

        {/* Buffer Slider */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
            <label>Min Size Buffer</label>
            <span style={{ color: theme.accent }}>{buffer}px</span>
          </div>
          <input
            type="range"
            min="0"
            max="50"
            value={buffer}
            onChange={handleBufferChange}
            style={{ width: '100%', accentColor: theme.accent }}
          />
        </div>
      </div>
    </>
  );
};
