/**
 * RouteView
 *
 * Unified view showing all code related to a route:
 * handler, model, and template in collapsible sections.
 */

import { memo, useState, useCallback, useRef, useEffect } from 'react';
import { EditorView } from '@codemirror/view';
import { basicSetup } from 'codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { html } from '@codemirror/lang-html';
import { oneDark } from '@codemirror/theme-one-dark';
import { EditorState } from '@codemirror/state';
import RouteSelector from './RouteSelector';
import type { RouteDefinition, RelatedCode } from '../services/routeParser';
import { DEMO_ROUTES, DEMO_CODE } from '../services/routeParser';

interface RouteViewProps {
  instanceId?: string;
  windowId?: string;
}

type LayerType = 'handler' | 'model' | 'template';

interface SectionState {
  collapsed: boolean;
}

/**
 * CodeSection Component - A collapsible code section with embedded editor
 */
const CodeSection = memo(function CodeSection({
  code,
  collapsed,
  onToggle,
}: {
  code: RelatedCode;
  collapsed: boolean;
  onToggle: () => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<EditorView | null>(null);

  useEffect(() => {
    if (!containerRef.current || collapsed) return;

    // Destroy existing editor
    if (editorRef.current) {
      editorRef.current.destroy();
    }

    const language = code.type === 'template' ? html() : javascript();

    const state = EditorState.create({
      doc: code.code,
      extensions: [
        basicSetup,
        language,
        oneDark,
        EditorState.readOnly.of(true),
        EditorView.theme({
          '&': { maxHeight: '250px' },
          '.cm-scroller': { overflow: 'auto' },
        }),
      ],
    });

    const view = new EditorView({
      state,
      parent: containerRef.current,
    });

    editorRef.current = view;

    return () => {
      view.destroy();
    };
  }, [code.code, code.type, collapsed]);

  const typeIcons: Record<RelatedCode['type'], string> = {
    handler: 'f',
    model: 'M',
    template: 'T',
    middleware: 'm',
  };

  const typeLabels: Record<RelatedCode['type'], string> = {
    handler: 'Handler',
    model: 'Model',
    template: 'Template',
    middleware: 'Middleware',
  };

  return (
    <div className={`route-section ${collapsed ? 'collapsed' : ''}`}>
      <div className="route-section-header" onClick={onToggle}>
        <span className="route-section-icon">&gt;</span>
        <span style={{ opacity: 0.5 }}>[{typeIcons[code.type]}]</span>
        <span className="route-section-title">
          {typeLabels[code.type]}: {code.name}
        </span>
        <span className="route-section-file">
          {code.file}:{code.startLine}
        </span>
      </div>
      <div className="route-section-content">
        <div ref={containerRef} />
      </div>
    </div>
  );
});

/**
 * RouteView Component
 */
const RouteView = memo(function RouteView({
  instanceId: _instanceId,
  windowId: _windowId,
}: RouteViewProps) {
  const [routes] = useState<RouteDefinition[]>(DEMO_ROUTES);
  const [selectedRoute, setSelectedRoute] = useState<RouteDefinition | null>(DEMO_ROUTES[1] ?? null);
  const [visibleLayers, setVisibleLayers] = useState<Set<LayerType>>(
    new Set(['handler', 'model', 'template'])
  );
  const [sectionStates, setSectionStates] = useState<Record<string, SectionState>>({});

  const handleSelectRoute = useCallback((route: RouteDefinition) => {
    setSelectedRoute(route);
  }, []);

  const handleToggleLayer = useCallback((layer: LayerType) => {
    setVisibleLayers((prev) => {
      const next = new Set(prev);
      if (next.has(layer)) {
        next.delete(layer);
      } else {
        next.add(layer);
      }
      return next;
    });
  }, []);

  const handleToggleSection = useCallback((key: string) => {
    setSectionStates((prev) => ({
      ...prev,
      [key]: { collapsed: !prev[key]?.collapsed },
    }));
  }, []);

  // Get related code for selected route
  const getRelatedCode = useCallback((route: RouteDefinition): Record<LayerType, RelatedCode | null> => {
    // For demo, use hardcoded demo code
    const handlerKey = `${route.id}-handler`;
    const modelKey = `${route.id}-model`;
    const templateKey = `${route.id}-template`;

    return {
      handler: DEMO_CODE[handlerKey] ?? null,
      model: DEMO_CODE[modelKey] ?? null,
      template: DEMO_CODE[templateKey] ?? null,
    };
  }, []);

  const relatedCode = selectedRoute ? getRelatedCode(selectedRoute) : null;

  return (
    <div className="route-view">
      <div className="route-view-header">
        <RouteSelector
          routes={routes}
          selectedRoute={selectedRoute}
          onSelectRoute={handleSelectRoute}
        />

        {selectedRoute && (
          <>
            <span className={`route-method ${selectedRoute.method.toLowerCase()}`}>
              {selectedRoute.method}
            </span>
            <span className="route-path">{selectedRoute.path}</span>
          </>
        )}

        <div className="route-layers">
          {(['handler', 'model', 'template'] as LayerType[]).map((layer) => (
            <button
              key={layer}
              className={`route-layer-toggle ${visibleLayers.has(layer) ? 'active' : ''}`}
              onClick={() => handleToggleLayer(layer)}
            >
              {layer}
            </button>
          ))}
        </div>
      </div>

      <div className="route-sections">
        {selectedRoute && relatedCode ? (
          <>
            {visibleLayers.has('handler') && relatedCode.handler && (
              <CodeSection
                code={relatedCode.handler}
                collapsed={sectionStates['handler']?.collapsed ?? false}
                onToggle={() => handleToggleSection('handler')}
              />
            )}
            {visibleLayers.has('model') && relatedCode.model && (
              <CodeSection
                code={relatedCode.model}
                collapsed={sectionStates['model']?.collapsed ?? false}
                onToggle={() => handleToggleSection('model')}
              />
            )}
            {visibleLayers.has('template') && relatedCode.template && (
              <CodeSection
                code={relatedCode.template}
                collapsed={sectionStates['template']?.collapsed ?? false}
                onToggle={() => handleToggleSection('template')}
              />
            )}
            {!relatedCode.handler && !relatedCode.model && !relatedCode.template && (
              <div style={{ padding: '40px', textAlign: 'center', color: 'var(--theme-text-muted)' }}>
                No related code found for this route.
                <br />
                <span style={{ fontSize: '0.85em' }}>
                  Try selecting GET /api/users/:id for a demo.
                </span>
              </div>
            )}
          </>
        ) : (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--theme-text-muted)' }}>
            Select a route to view related code.
          </div>
        )}
      </div>
    </div>
  );
});

export default RouteView;
