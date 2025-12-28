/**
 * RouteSelector
 *
 * Dropdown component for selecting routes to view.
 * Groups routes by HTTP method and supports filtering.
 */

import { memo, useState, useCallback, useMemo, useRef, useEffect } from 'react';
import type { RouteDefinition } from '../services/routeParser';

interface RouteSelectorProps {
  routes: RouteDefinition[];
  selectedRoute: RouteDefinition | null;
  onSelectRoute: (route: RouteDefinition) => void;
}

/**
 * RouteSelector Component
 */
const RouteSelector = memo(function RouteSelector({
  routes,
  selectedRoute,
  onSelectRoute,
}: RouteSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [filter, setFilter] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Focus input when dropdown opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  // Filter and group routes
  const groupedRoutes = useMemo(() => {
    const filtered = routes.filter((route) =>
      route.path.toLowerCase().includes(filter.toLowerCase())
    );

    const groups: Record<string, RouteDefinition[]> = {};
    for (const route of filtered) {
      if (!groups[route.method]) {
        groups[route.method] = [];
      }
      groups[route.method]!.push(route);
    }

    return groups;
  }, [routes, filter]);

  const handleToggle = useCallback(() => {
    setIsOpen((prev) => !prev);
    setFilter('');
  }, []);

  const handleSelect = useCallback(
    (route: RouteDefinition) => {
      onSelectRoute(route);
      setIsOpen(false);
      setFilter('');
    },
    [onSelectRoute]
  );

  const methodOrder: RouteDefinition['method'][] = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'];

  return (
    <div className="route-selector" ref={dropdownRef}>
      <button className="route-selector-trigger" onClick={handleToggle}>
        {selectedRoute ? (
          <>
            <span className={`route-method ${selectedRoute.method.toLowerCase()}`}>
              {selectedRoute.method}
            </span>
            <span>{selectedRoute.path}</span>
          </>
        ) : (
          <span style={{ color: 'var(--theme-text-muted)' }}>Select a route...</span>
        )}
        <span style={{ marginLeft: 'auto', opacity: 0.5 }}>v</span>
      </button>

      {isOpen && (
        <div className="route-selector-dropdown">
          <div className="route-selector-search">
            <input
              ref={inputRef}
              type="text"
              placeholder="Filter routes..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
            />
          </div>

          {methodOrder.map((method) => {
            const methodRoutes = groupedRoutes[method];
            if (!methodRoutes || methodRoutes.length === 0) return null;

            return (
              <div key={method} className="route-selector-group">
                <div className="route-selector-group-title">{method}</div>
                {methodRoutes.map((route) => (
                  <div
                    key={route.id}
                    className={`route-selector-item ${route.id === selectedRoute?.id ? 'selected' : ''}`}
                    onClick={() => handleSelect(route)}
                  >
                    <span className={`route-method ${route.method.toLowerCase()}`}>
                      {route.method}
                    </span>
                    <span>{route.path}</span>
                  </div>
                ))}
              </div>
            );
          })}

          {Object.keys(groupedRoutes).length === 0 && (
            <div style={{ padding: '12px', textAlign: 'center', color: 'var(--theme-text-muted)' }}>
              No routes found
            </div>
          )}
        </div>
      )}
    </div>
  );
});

export default RouteSelector;
