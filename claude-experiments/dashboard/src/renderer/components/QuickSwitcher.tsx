import { memo, useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { useProjectDashboardTree, useDashboardCommands } from '../hooks/useBackendState';
import type { ProjectState, DashboardState } from '../../types/state';

/**
 * Quick Switcher - Fast dashboard switching with Cmd+P
 *
 * Shows dashboards grouped by projects for quick navigation.
 * Supports keyboard navigation and search filtering.
 */

interface QuickSwitcherItem {
  type: 'project-header' | 'dashboard';
  id: string;
  label: string;
  projectId?: string;
  projectName?: string;
  isActive: boolean;
  dashboard?: DashboardState;
  project?: ProjectState;
}

interface QuickSwitcherProps {
  isOpen: boolean;
  onClose: () => void;
}

const QuickSwitcher = memo(function QuickSwitcher({ isOpen, onClose }: QuickSwitcherProps) {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const { tree, activeProjectId, activeDashboardId, loading } = useProjectDashboardTree();
  const { switchDashboard } = useDashboardCommands();

  // Build flat list of items (project headers + dashboards)
  const allItems = useMemo((): QuickSwitcherItem[] => {
    const items: QuickSwitcherItem[] = [];

    tree.forEach(({ project, dashboards }) => {
      // Add project header
      items.push({
        type: 'project-header',
        id: `header-${project.id}`,
        label: project.name,
        isActive: project.id === activeProjectId,
        project,
      });

      // Add dashboards under this project
      dashboards.forEach((dashboard) => {
        items.push({
          type: 'dashboard',
          id: dashboard.id,
          label: dashboard.name,
          projectId: project.id,
          projectName: project.name,
          isActive: dashboard.id === activeDashboardId,
          dashboard,
        });
      });
    });

    return items;
  }, [tree, activeProjectId, activeDashboardId]);

  // Filter items based on search query
  const filteredItems = useMemo(() => {
    if (!query.trim()) return allItems;

    const lowerQuery = query.toLowerCase();
    const matchingDashboards = allItems.filter(
      (item) =>
        item.type === 'dashboard' &&
        (item.label.toLowerCase().includes(lowerQuery) ||
          item.projectName?.toLowerCase().includes(lowerQuery))
    );

    // Get unique project IDs that have matching dashboards
    const projectIdsWithMatches = new Set(
      matchingDashboards.map((item) => item.projectId)
    );

    // Return project headers + matching dashboards
    return allItems.filter(
      (item) =>
        (item.type === 'project-header' && projectIdsWithMatches.has(item.project?.id)) ||
        (item.type === 'dashboard' && matchingDashboards.includes(item))
    );
  }, [allItems, query]);

  // Get only selectable items (dashboards, not headers)
  const selectableItems = useMemo(
    () => filteredItems.filter((item) => item.type === 'dashboard'),
    [filteredItems]
  );

  // Reset selection when query changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  // Focus input when switcher opens
  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 10);
    }
  }, [isOpen]);

  // Scroll selected item into view
  useEffect(() => {
    if (listRef.current && selectableItems.length > 0) {
      const selectedItem = selectableItems[selectedIndex];
      if (selectedItem) {
        const selectedEl = listRef.current.querySelector(
          `[data-id="${selectedItem.id}"]`
        ) as HTMLElement;
        if (selectedEl) {
          selectedEl.scrollIntoView({ block: 'nearest' });
        }
      }
    }
  }, [selectedIndex, selectableItems]);

  const handleSelect = useCallback(
    async (item: QuickSwitcherItem) => {
      if (item.type === 'dashboard' && item.dashboard) {
        await switchDashboard(item.dashboard.id);
        onClose();
      }
    },
    [switchDashboard, onClose]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex((prev) =>
            prev < selectableItems.length - 1 ? prev + 1 : prev
          );
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex((prev) => (prev > 0 ? prev - 1 : prev));
          break;
        case 'Enter':
          e.preventDefault();
          if (selectableItems[selectedIndex]) {
            handleSelect(selectableItems[selectedIndex]);
          }
          break;
        case 'Escape':
          e.preventDefault();
          onClose();
          break;
        default:
          break;
      }
    },
    [selectableItems, selectedIndex, handleSelect, onClose]
  );

  if (!isOpen) return null;

  const isEmpty = tree.length === 0;

  return (
    <div
      className="quick-switcher-overlay"
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
        className="quick-switcher"
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
          className="quick-switcher-input-wrapper"
          style={{
            padding: 'var(--theme-spacing-md)',
            borderBottom: '1px solid var(--theme-palette-border)',
          }}
        >
          <input
            ref={inputRef}
            type="text"
            className="quick-switcher-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Switch to dashboard..."
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

        {/* Dashboard list */}
        <div
          ref={listRef}
          className="quick-switcher-list"
          style={{
            flex: 1,
            overflow: 'auto',
            padding: 'var(--theme-spacing-xs)',
          }}
        >
          {loading ? (
            <div
              style={{
                padding: 'var(--theme-spacing-lg)',
                textAlign: 'center',
                color: 'var(--theme-text-muted)',
                fontSize: 'var(--theme-font-size-md)',
              }}
            >
              Loading...
            </div>
          ) : isEmpty ? (
            <div
              style={{
                padding: 'var(--theme-spacing-lg)',
                textAlign: 'center',
                color: 'var(--theme-text-muted)',
                fontSize: 'var(--theme-font-size-md)',
              }}
            >
              No projects yet. Use the command palette to create one.
            </div>
          ) : filteredItems.length === 0 ? (
            <div
              style={{
                padding: 'var(--theme-spacing-lg)',
                textAlign: 'center',
                color: 'var(--theme-text-muted)',
                fontSize: 'var(--theme-font-size-md)',
              }}
            >
              No dashboards found
            </div>
          ) : (
            filteredItems.map((item) => {
              if (item.type === 'project-header') {
                return (
                  <ProjectHeader
                    key={item.id}
                    label={item.label}
                    isActive={item.isActive}
                  />
                );
              }

              const isSelected =
                selectableItems[selectedIndex]?.id === item.id;

              return (
                <DashboardItem
                  key={item.id}
                  item={item}
                  isSelected={isSelected}
                  onClick={() => handleSelect(item)}
                  onMouseEnter={() => {
                    const idx = selectableItems.findIndex(
                      (i) => i.id === item.id
                    );
                    if (idx !== -1) setSelectedIndex(idx);
                  }}
                />
              );
            })
          )}
        </div>

        {/* Footer hint */}
        <div
          className="quick-switcher-footer"
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
          <span>↵ Switch</span>
          <span>Esc Close</span>
        </div>
      </div>
    </div>
  );
});

interface ProjectHeaderProps {
  label: string;
  isActive: boolean;
}

const ProjectHeader = memo(function ProjectHeader({
  label,
  isActive,
}: ProjectHeaderProps) {
  return (
    <div
      className="quick-switcher-project-header"
      style={{
        padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
        marginTop: 'var(--theme-spacing-sm)',
        fontSize: 'var(--theme-font-size-xs)',
        fontWeight: 'var(--theme-font-weight-bold)',
        color: isActive
          ? 'var(--theme-accent-primary)'
          : 'var(--theme-text-muted)',
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        display: 'flex',
        alignItems: 'center',
        gap: 'var(--theme-spacing-xs)',
      }}
    >
      <span style={{ fontSize: '10px' }}>▸</span>
      {label}
      {isActive && (
        <span
          style={{
            marginLeft: 'auto',
            fontSize: 'var(--theme-font-size-xs)',
            color: 'var(--theme-accent-primary)',
            fontWeight: 'normal',
            textTransform: 'none',
          }}
        >
          active
        </span>
      )}
    </div>
  );
});

interface DashboardItemProps {
  item: QuickSwitcherItem;
  isSelected: boolean;
  onClick: () => void;
  onMouseEnter: () => void;
}

const DashboardItem = memo(function DashboardItem({
  item,
  isSelected,
  onClick,
  onMouseEnter,
}: DashboardItemProps) {
  return (
    <div
      className={`quick-switcher-item ${isSelected ? 'selected' : ''}`}
      data-id={item.id}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      style={{
        padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
        paddingLeft: 'var(--theme-spacing-xl)',
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
      <span
        style={{
          fontSize: 'var(--theme-font-size-md)',
          width: '20px',
          textAlign: 'center',
          color: 'var(--theme-text-muted)',
        }}
      >
        ◦
      </span>
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
          {item.label}
        </div>
      </div>
      {item.isActive && (
        <span
          style={{
            color: 'var(--theme-accent-primary)',
            fontSize: 'var(--theme-font-size-xs)',
            padding: '2px 6px',
            background: 'var(--theme-bg-tertiary)',
            borderRadius: 'var(--theme-radius-sm)',
          }}
        >
          current
        </span>
      )}
    </div>
  );
});

/**
 * Keyboard shortcut configuration for quick switcher
 */
export interface QuickSwitcherShortcutConfig {
  key: string;
  meta: boolean;
  ctrl: boolean;
  shift: boolean;
  alt: boolean;
}

/**
 * Default shortcut: Cmd+P (no shift)
 */
export const DEFAULT_QUICK_SWITCHER_SHORTCUT: QuickSwitcherShortcutConfig = {
  key: 'p',
  meta: true,
  ctrl: false,
  shift: false,
  alt: false,
};

/**
 * Check if a keyboard event matches a shortcut configuration
 */
function matchesShortcut(
  event: KeyboardEvent,
  shortcut: QuickSwitcherShortcutConfig | null
): boolean {
  if (!shortcut) return false;

  const keyMatches = event.key.toLowerCase() === shortcut.key.toLowerCase();
  const shiftMatches = shortcut.shift ? event.shiftKey : !event.shiftKey;
  const altMatches = shortcut.alt ? event.altKey : !event.altKey;

  // For meta shortcuts, allow either meta or ctrl (for cross-platform)
  if (shortcut.meta) {
    return keyMatches && (event.metaKey || event.ctrlKey) && shiftMatches && altMatches;
  }

  const metaMatches = !event.metaKey;
  const ctrlMatches = shortcut.ctrl ? event.ctrlKey : !event.ctrlKey;

  return keyMatches && metaMatches && ctrlMatches && shiftMatches && altMatches;
}

export interface UseQuickSwitcherResult {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
  shortcut: QuickSwitcherShortcutConfig;
}

/**
 * Hook to use quick switcher with keyboard shortcut
 */
export function useQuickSwitcher(
  shortcut: QuickSwitcherShortcutConfig = DEFAULT_QUICK_SWITCHER_SHORTCUT
): UseQuickSwitcherResult {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (matchesShortcut(e, shortcut)) {
        e.preventDefault();
        setIsOpen((prev) => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcut]);

  const open = useCallback(() => setIsOpen(true), []);
  const close = useCallback(() => setIsOpen(false), []);
  const toggle = useCallback(() => setIsOpen((prev) => !prev), []);

  return { isOpen, open, close, toggle, shortcut };
}

export default QuickSwitcher;
