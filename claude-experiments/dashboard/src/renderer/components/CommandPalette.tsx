import { memo, useState, useEffect, useRef, useMemo, useCallback } from 'react';

/**
 * Command configuration interface
 */
export interface Command {
  id: string;
  label: string;
  description?: string;
  icon?: string;
  category?: string;
  keywords?: string[];
  action: () => void;
}

/**
 * Keyboard shortcut configuration
 */
export interface ShortcutConfig {
  key: string;
  meta: boolean;
  ctrl: boolean;
  shift: boolean;
  alt: boolean;
}

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  commands: Command[];
  onExecute: (command: Command) => void;
}

/**
 * Command Palette - A themable command interface
 *
 * Opens with Cmd+K (Mac) or Ctrl+K (Windows/Linux)
 * Provides quick access to all actions in the app
 */
const CommandPalette = memo(function CommandPalette({
  isOpen,
  onClose,
  commands,
  onExecute,
}: CommandPaletteProps) {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Filter commands based on search query
  const filteredCommands = useMemo(() => {
    if (!query.trim()) return commands;

    const lowerQuery = query.toLowerCase();
    return commands.filter(cmd =>
      cmd.label.toLowerCase().includes(lowerQuery) ||
      cmd.keywords?.some(kw => kw.toLowerCase().includes(lowerQuery)) ||
      cmd.category?.toLowerCase().includes(lowerQuery)
    );
  }, [commands, query]);

  // Reset selection when query changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  // Focus input when palette opens
  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedIndex(0);
      // Small delay to ensure the element is rendered
      setTimeout(() => inputRef.current?.focus(), 10);
    }
  }, [isOpen]);

  // Scroll selected item into view
  useEffect(() => {
    if (listRef.current && filteredCommands.length > 0) {
      const selectedEl = listRef.current.children[selectedIndex] as HTMLElement;
      if (selectedEl) {
        selectedEl.scrollIntoView({ block: 'nearest' });
      }
    }
  }, [selectedIndex, filteredCommands.length]);

  const executeCommand = useCallback((command: Command) => {
    onExecute(command);
    onClose();
  }, [onExecute, onClose]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex(prev =>
          prev < filteredCommands.length - 1 ? prev + 1 : prev
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex(prev => prev > 0 ? prev - 1 : prev);
        break;
      case 'Enter':
        e.preventDefault();
        if (filteredCommands[selectedIndex]) {
          executeCommand(filteredCommands[selectedIndex]);
        }
        break;
      case 'Escape':
        e.preventDefault();
        onClose();
        break;
      default:
        break;
    }
  }, [filteredCommands, selectedIndex, executeCommand, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="command-palette-overlay"
      onClick={onClose}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'var(--theme-palette-overlay)',
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingTop: '15vh',
        zIndex: 10000,
        backdropFilter: 'var(--theme-backdrop-blur)',
      }}
    >
      <div
        className="command-palette"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '500px',
          maxWidth: '90vw',
          maxHeight: '60vh',
          background: 'var(--theme-palette-bg)',
          border: '1px solid var(--theme-palette-border)',
          borderRadius: 'var(--theme-palette-radius)',
          boxShadow: 'var(--theme-palette-shadow)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* Search input */}
        <div
          className="command-palette-input-wrapper"
          style={{
            padding: 'var(--theme-spacing-md)',
            borderBottom: '1px solid var(--theme-palette-border)',
          }}
        >
          <input
            ref={inputRef}
            type="text"
            className="command-palette-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a command..."
            style={{
              width: '100%',
              padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
              background: 'var(--theme-palette-input-bg)',
              border: '1px solid var(--theme-palette-input-border)',
              borderRadius: 'var(--theme-radius-sm)',
              color: 'var(--theme-text-primary)',
              fontSize: 'var(--theme-font-size-lg)',
              fontFamily: 'var(--theme-font-family)',
              outline: 'none',
            }}
          />
        </div>

        {/* Command list */}
        <div
          ref={listRef}
          className="command-palette-list"
          style={{
            flex: 1,
            overflow: 'auto',
            padding: 'var(--theme-spacing-xs)',
          }}
        >
          {filteredCommands.length === 0 ? (
            <div
              style={{
                padding: 'var(--theme-spacing-lg)',
                textAlign: 'center',
                color: 'var(--theme-text-muted)',
                fontSize: 'var(--theme-font-size-md)',
              }}
            >
              No commands found
            </div>
          ) : (
            filteredCommands.map((command, index) => (
              <CommandItem
                key={command.id}
                command={command}
                isSelected={index === selectedIndex}
                onClick={() => executeCommand(command)}
                onMouseEnter={() => setSelectedIndex(index)}
              />
            ))
          )}
        </div>

        {/* Footer hint */}
        <div
          className="command-palette-footer"
          style={{
            padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
            borderTop: '1px solid var(--theme-palette-border)',
            display: 'flex',
            gap: 'var(--theme-spacing-md)',
            fontSize: 'var(--theme-font-size-xs)',
            color: 'var(--theme-text-muted)',
          }}
        >
          <span>↑↓ Navigate</span>
          <span>↵ Select</span>
          <span>Esc Close</span>
        </div>
      </div>
    </div>
  );
});

interface CommandItemProps {
  command: Command;
  isSelected: boolean;
  onClick: () => void;
  onMouseEnter: () => void;
}

/**
 * Individual command item in the palette
 */
const CommandItem = memo(function CommandItem({
  command,
  isSelected,
  onClick,
  onMouseEnter
}: CommandItemProps) {
  return (
    <div
      className={`command-palette-item ${isSelected ? 'selected' : ''}`}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      style={{
        padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
        borderRadius: 'var(--theme-radius-sm)',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: 'var(--theme-spacing-sm)',
        background: isSelected
          ? 'var(--theme-palette-item-hover)'
          : 'var(--theme-palette-item-bg)',
        transition: 'background var(--theme-transition-fast)',
      }}
    >
      {command.icon && (
        <span
          style={{
            fontSize: 'var(--theme-font-size-lg)',
            width: '24px',
            textAlign: 'center',
          }}
        >
          {command.icon}
        </span>
      )}
      <div style={{ flex: 1 }}>
        <div
          style={{
            color: isSelected
              ? 'var(--theme-text-primary)'
              : 'var(--theme-text-secondary)',
            fontSize: 'var(--theme-font-size-md)',
            fontWeight: 'var(--theme-font-weight-medium)',
          }}
        >
          {command.label}
        </div>
        {command.description && (
          <div
            style={{
              color: 'var(--theme-text-muted)',
              fontSize: 'var(--theme-font-size-xs)',
              marginTop: '2px',
            }}
          >
            {command.description}
          </div>
        )}
      </div>
      {command.category && (
        <span
          style={{
            color: 'var(--theme-text-disabled)',
            fontSize: 'var(--theme-font-size-xs)',
            padding: '2px 6px',
            background: 'var(--theme-bg-tertiary)',
            borderRadius: 'var(--theme-radius-sm)',
          }}
        >
          {command.category}
        </span>
      )}
    </div>
  );
});

/**
 * Default shortcut configuration
 */
export const DEFAULT_PALETTE_SHORTCUT: ShortcutConfig = {
  key: 'p',
  meta: true,    // Cmd on Mac
  ctrl: false,   // Ctrl key (also used as Cmd alternative on Windows/Linux)
  shift: true,
  alt: false,
};

/**
 * Parse a shortcut string like "cmd+shift+p" into a shortcut object
 */
export function parseShortcut(str: string): ShortcutConfig {
  if (!str || typeof str !== 'string') return DEFAULT_PALETTE_SHORTCUT;

  const parts = str.toLowerCase().split('+').map(p => p.trim());
  const key = parts.find(p => !['cmd', 'ctrl', 'shift', 'alt', 'meta', 'option'].includes(p)) || 'p';

  return {
    key,
    meta: parts.includes('cmd') || parts.includes('meta'),
    ctrl: parts.includes('ctrl'),
    shift: parts.includes('shift'),
    alt: parts.includes('alt') || parts.includes('option'),
  };
}

/**
 * Format a shortcut object to a display string
 */
export function formatShortcut(shortcut: ShortcutConfig): string {
  if (!shortcut) return '';

  const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
  const parts: string[] = [];

  if (shortcut.meta) parts.push(isMac ? '⌘' : 'Ctrl');
  if (shortcut.ctrl && !shortcut.meta) parts.push('Ctrl');
  if (shortcut.alt) parts.push(isMac ? '⌥' : 'Alt');
  if (shortcut.shift) parts.push(isMac ? '⇧' : 'Shift');
  if (shortcut.key) parts.push(shortcut.key.toUpperCase());

  return parts.join(isMac ? '' : '+');
}

/**
 * Check if a keyboard event matches a shortcut configuration
 */
function matchesShortcut(event: KeyboardEvent, shortcut: ShortcutConfig | null): boolean {
  if (!shortcut) return false;

  const keyMatches = event.key.toLowerCase() === shortcut.key.toLowerCase();
  const metaMatches = shortcut.meta ? (event.metaKey || event.ctrlKey) : true;
  const ctrlMatches = shortcut.ctrl ? event.ctrlKey : true;
  const shiftMatches = shortcut.shift ? event.shiftKey : !event.shiftKey;
  const altMatches = shortcut.alt ? event.altKey : !event.altKey;

  // For meta shortcuts, allow either meta or ctrl (for cross-platform)
  if (shortcut.meta) {
    return keyMatches && (event.metaKey || event.ctrlKey) && shiftMatches && altMatches;
  }

  return keyMatches && metaMatches && ctrlMatches && shiftMatches && altMatches;
}

export interface UseCommandPaletteResult {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
  shortcut: ShortcutConfig;
}

/**
 * Hook to use command palette with keyboard shortcut
 */
export function useCommandPalette(shortcut: ShortcutConfig | string = DEFAULT_PALETTE_SHORTCUT): UseCommandPaletteResult {
  const [isOpen, setIsOpen] = useState(false);

  // Parse shortcut if it's a string
  const shortcutConfig = useMemo(() => {
    if (typeof shortcut === 'string') {
      return parseShortcut(shortcut);
    }
    return shortcut || DEFAULT_PALETTE_SHORTCUT;
  }, [shortcut]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (matchesShortcut(e, shortcutConfig)) {
        e.preventDefault();
        setIsOpen(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcutConfig]);

  const open = useCallback(() => setIsOpen(true), []);
  const close = useCallback(() => setIsOpen(false), []);
  const toggle = useCallback(() => setIsOpen(prev => !prev), []);

  return { isOpen, open, close, toggle, shortcut: shortcutConfig };
}

export default CommandPalette;
