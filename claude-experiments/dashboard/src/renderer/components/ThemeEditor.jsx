import { memo, useState, useEffect } from 'react';
import { useTheme, THEME_CATEGORIES } from '../theme/ThemeProvider';

/**
 * Theme Editor Component
 *
 * Live editor for theme CSS variables.
 * Can switch presets and override individual values.
 */
const ThemeEditor = memo(function ThemeEditor() {
  const {
    currentTheme,
    presetThemes,
    setTheme,
    setVariable,
    getVariable,
    resetVariable,
    resetAllOverrides,
    overrides,
    exportTheme,
  } = useTheme();

  const [expandedCategory, setExpandedCategory] = useState('Structure');
  const [showExport, setShowExport] = useState(false);

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
      {/* Preset Themes */}
      <div style={{
        padding: 'var(--theme-spacing-md)',
        borderBottom: '1px solid var(--theme-border-primary)',
      }}>
        <div style={{
          fontSize: 'var(--theme-font-size-sm)',
          color: 'var(--theme-text-muted)',
          marginBottom: 'var(--theme-spacing-sm)',
        }}>
          Preset Theme
        </div>
        <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
          {presetThemes.map((theme) => (
            <button
              key={theme}
              onClick={() => setTheme(theme)}
              style={{
                padding: '4px 8px',
                background: currentTheme === theme
                  ? 'var(--theme-accent-primary)'
                  : 'var(--theme-bg-tertiary)',
                border: '1px solid',
                borderColor: currentTheme === theme
                  ? 'var(--theme-accent-primary)'
                  : 'var(--theme-border-secondary)',
                borderRadius: 'var(--theme-radius-sm)',
                color: currentTheme === theme
                  ? '#fff'
                  : 'var(--theme-text-secondary)',
                fontSize: 'var(--theme-font-size-xs)',
                cursor: 'pointer',
              }}
            >
              {theme}
            </button>
          ))}
        </div>
      </div>

      {/* Category Accordion */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        {Object.entries(THEME_CATEGORIES).map(([category, variables]) => (
          <div key={category}>
            <button
              onClick={() => setExpandedCategory(
                expandedCategory === category ? null : category
              )}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                width: '100%',
                padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
                background: 'var(--theme-bg-tertiary)',
                border: 'none',
                borderBottom: '1px solid var(--theme-border-primary)',
                color: 'var(--theme-text-primary)',
                fontSize: 'var(--theme-font-size-sm)',
                fontWeight: 500,
                cursor: 'pointer',
                textAlign: 'left',
              }}
            >
              <span>{category}</span>
              <span style={{ color: 'var(--theme-text-muted)' }}>
                {expandedCategory === category ? '▼' : '▶'}
              </span>
            </button>

            {expandedCategory === category && (
              <div style={{
                padding: 'var(--theme-spacing-sm)',
                background: 'var(--theme-bg-secondary)',
              }}>
                {variables.map(({ var: varName, label, type }) => (
                  <ThemeVariable
                    key={varName}
                    varName={varName}
                    label={label}
                    type={type || 'color'}
                    value={getVariable(varName)}
                    isOverridden={!!overrides[varName]}
                    onChange={(value) => setVariable(varName, value)}
                    onReset={() => resetVariable(varName)}
                  />
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Actions */}
      <div style={{
        padding: 'var(--theme-spacing-md)',
        borderTop: '1px solid var(--theme-border-primary)',
        display: 'flex',
        gap: 'var(--theme-spacing-sm)',
      }}>
        <button
          onClick={resetAllOverrides}
          style={{
            flex: 1,
            padding: '8px',
            background: 'var(--theme-bg-tertiary)',
            border: '1px solid var(--theme-border-secondary)',
            borderRadius: 'var(--theme-radius-sm)',
            color: 'var(--theme-text-secondary)',
            fontSize: 'var(--theme-font-size-sm)',
            cursor: 'pointer',
          }}
        >
          Reset ({Object.keys(overrides).length})
        </button>
        <button
          onClick={() => setShowExport(!showExport)}
          style={{
            flex: 1,
            padding: '8px',
            background: 'var(--theme-accent-primary)',
            border: 'none',
            borderRadius: 'var(--theme-radius-sm)',
            color: '#fff',
            fontSize: 'var(--theme-font-size-sm)',
            cursor: 'pointer',
          }}
        >
          Export
        </button>
      </div>

      {/* Export Modal */}
      {showExport && (
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'rgba(0,0,0,0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 'var(--theme-spacing-lg)',
        }}>
          <div style={{
            background: 'var(--theme-bg-secondary)',
            borderRadius: 'var(--theme-radius-lg)',
            padding: 'var(--theme-spacing-lg)',
            maxWidth: '100%',
            maxHeight: '100%',
            overflow: 'auto',
          }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              marginBottom: 'var(--theme-spacing-md)',
            }}>
              <strong>Theme Export</strong>
              <button
                onClick={() => setShowExport(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--theme-text-muted)',
                  cursor: 'pointer',
                  fontSize: '16px',
                }}
              >
                ×
              </button>
            </div>
            <pre style={{
              background: 'var(--theme-bg-tertiary)',
              padding: 'var(--theme-spacing-md)',
              borderRadius: 'var(--theme-radius-sm)',
              fontSize: 'var(--theme-font-size-xs)',
              overflow: 'auto',
              maxHeight: '200px',
            }}>
              {JSON.stringify(exportTheme(), null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
});

/**
 * Single theme variable editor
 * Supports both color and text types
 */
const ThemeVariable = memo(function ThemeVariable({
  varName,
  label,
  type,
  value,
  isOverridden,
  onChange,
  onReset,
}) {
  const [inputValue, setInputValue] = useState(value);

  useEffect(() => {
    setInputValue(value);
  }, [value]);

  const handleChange = (e) => {
    const newValue = e.target.value;
    setInputValue(newValue);
    onChange(newValue);
  };

  const isColor = type === 'color';

  // Try to determine if value looks like a color
  const looksLikeColor = isColor ||
    (value && (value.startsWith('#') || value.startsWith('rgb') || value.startsWith('hsl')));

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 'var(--theme-spacing-sm)',
      padding: '4px 0',
    }}>
      {/* Color picker (only for color types) */}
      {looksLikeColor && (
        <input
          type="color"
          value={inputValue?.startsWith('#') ? inputValue : '#888888'}
          onChange={handleChange}
          style={{
            width: 24,
            height: 24,
            padding: 0,
            border: '2px solid var(--theme-border-secondary)',
            borderRadius: 'var(--theme-radius-sm)',
            cursor: 'pointer',
            flexShrink: 0,
          }}
        />
      )}

      {/* Label */}
      <span style={{
        flex: 1,
        fontSize: 'var(--theme-font-size-sm)',
        color: isOverridden ? 'var(--theme-accent-warning)' : 'var(--theme-text-secondary)',
        minWidth: 0,
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
      }}>
        {label}
        {isOverridden && ' *'}
      </span>

      {/* Text input */}
      <input
        type="text"
        value={inputValue}
        onChange={handleChange}
        style={{
          width: isColor ? 70 : 100,
          padding: '4px 6px',
          background: 'var(--theme-bg-input)',
          border: '1px solid var(--theme-border-secondary)',
          borderRadius: 'var(--theme-radius-sm)',
          color: 'var(--theme-text-primary)',
          fontSize: 'var(--theme-font-size-xs)',
          fontFamily: 'var(--theme-font-mono)',
          flexShrink: 0,
        }}
      />

      {/* Reset button */}
      {isOverridden && (
        <button
          onClick={onReset}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--theme-text-muted)',
            cursor: 'pointer',
            fontSize: '12px',
            padding: '2px 4px',
            flexShrink: 0,
          }}
          title="Reset to theme default"
        >
          ↩
        </button>
      )}
    </div>
  );
});

export default ThemeEditor;
