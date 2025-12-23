import { memo } from 'react';
import { useSettings } from '../settings/SettingsProvider';

/**
 * Settings Editor Component
 *
 * Controls for global settings that persist independently of theme.
 */
const SettingsEditor = memo(function SettingsEditor() {
  const {
    settings,
    updateSetting,
    resetSettings,
    fontSizeOptions,
    spacingOptions,
  } = useSettings();

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: 'var(--theme-bg-secondary)',
      color: 'var(--theme-text-primary)',
      fontFamily: 'var(--theme-font-family)',
      fontSize: 'var(--theme-font-size-md)',
    }}>
      <div style={{
        padding: 'var(--theme-spacing-md)',
        borderBottom: '1px solid var(--theme-border-primary)',
      }}>
        <div style={{
          fontSize: 'var(--theme-font-size-lg)',
          fontWeight: 500,
          marginBottom: 'var(--theme-spacing-xs)',
        }}>
          Global Settings
        </div>
        <div style={{
          fontSize: 'var(--theme-font-size-sm)',
          color: 'var(--theme-text-muted)',
        }}>
          These persist across theme changes
        </div>
      </div>

      <div style={{ flex: 1, overflow: 'auto', padding: 'var(--theme-spacing-md)' }}>
        {/* Font Size */}
        <SettingRow label="Font Size">
          <div style={{ display: 'flex', gap: '4px' }}>
            {fontSizeOptions.map((option) => (
              <button
                key={option}
                onClick={() => updateSetting('fontSize', option)}
                style={{
                  padding: '6px 10px',
                  background: settings.fontSize === option
                    ? 'var(--theme-accent-primary)'
                    : 'var(--theme-bg-tertiary)',
                  border: '1px solid',
                  borderColor: settings.fontSize === option
                    ? 'var(--theme-accent-primary)'
                    : 'var(--theme-border-secondary)',
                  borderRadius: 'var(--theme-radius-sm)',
                  color: settings.fontSize === option
                    ? '#fff'
                    : 'var(--theme-text-secondary)',
                  fontSize: 'var(--theme-font-size-sm)',
                  cursor: 'pointer',
                  textTransform: 'capitalize',
                }}
              >
                {option}
              </button>
            ))}
          </div>
        </SettingRow>

        {/* Font Scale */}
        <SettingRow label="Font Scale">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input
              type="range"
              min="0.8"
              max="1.5"
              step="0.05"
              value={settings.fontScale}
              onChange={(e) => updateSetting('fontScale', parseFloat(e.target.value))}
              style={{ flex: 1 }}
            />
            <span style={{
              width: '40px',
              textAlign: 'right',
              fontSize: 'var(--theme-font-size-sm)',
              fontFamily: 'var(--theme-font-mono)',
            }}>
              {settings.fontScale.toFixed(2)}x
            </span>
          </div>
        </SettingRow>

        {/* Spacing */}
        <SettingRow label="Spacing">
          <div style={{ display: 'flex', gap: '4px' }}>
            {spacingOptions.map((option) => (
              <button
                key={option}
                onClick={() => updateSetting('spacing', option)}
                style={{
                  padding: '6px 10px',
                  background: settings.spacing === option
                    ? 'var(--theme-accent-primary)'
                    : 'var(--theme-bg-tertiary)',
                  border: '1px solid',
                  borderColor: settings.spacing === option
                    ? 'var(--theme-accent-primary)'
                    : 'var(--theme-border-secondary)',
                  borderRadius: 'var(--theme-radius-sm)',
                  color: settings.spacing === option
                    ? '#fff'
                    : 'var(--theme-text-secondary)',
                  fontSize: 'var(--theme-font-size-sm)',
                  cursor: 'pointer',
                  textTransform: 'capitalize',
                }}
              >
                {option}
              </button>
            ))}
          </div>
        </SettingRow>
      </div>

      {/* Reset */}
      <div style={{
        padding: 'var(--theme-spacing-md)',
        borderTop: '1px solid var(--theme-border-primary)',
      }}>
        <button
          onClick={resetSettings}
          style={{
            width: '100%',
            padding: '8px',
            background: 'var(--theme-bg-tertiary)',
            border: '1px solid var(--theme-border-secondary)',
            borderRadius: 'var(--theme-radius-sm)',
            color: 'var(--theme-text-secondary)',
            fontSize: 'var(--theme-font-size-sm)',
            cursor: 'pointer',
          }}
        >
          Reset to Defaults
        </button>
      </div>
    </div>
  );
});

const SettingRow = memo(function SettingRow({ label, children }) {
  return (
    <div style={{ marginBottom: 'var(--theme-spacing-lg)' }}>
      <div style={{
        fontSize: 'var(--theme-font-size-sm)',
        color: 'var(--theme-text-muted)',
        marginBottom: 'var(--theme-spacing-sm)',
      }}>
        {label}
      </div>
      {children}
    </div>
  );
});

export default SettingsEditor;
