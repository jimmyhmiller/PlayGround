/**
 * Global UI Renderer
 *
 * Renders widgets from globalUI state in their designated slots.
 * Reads slot and widget configuration from backend state.
 */

import { memo, type ReactElement, type ReactNode } from 'react';
import { useBackendStateSelector } from '../hooks/useBackendState';
import { CornerSlot, BarSlot, PanelSlot } from './SlotRenderers';
import { WIDGET_TYPES } from '../widgets/BuiltinWidgets';
import type { GlobalUIState, SlotState, WidgetState } from '../../types/state';

interface SlotContainerProps {
  slot: SlotState;
  children: ReactNode;
}

/**
 * Renders the appropriate slot container based on position type
 */
function SlotContainer({ slot, children }: SlotContainerProps): ReactElement | null {
  const { position } = slot;

  switch (position.type) {
    case 'corner':
      return (
        <CornerSlot corner={position.corner} zIndex={slot.zIndex}>
          {children}
        </CornerSlot>
      );

    case 'bar':
      return (
        <BarSlot edge={position.edge} zIndex={slot.zIndex}>
          {children}
        </BarSlot>
      );

    case 'panel':
      return (
        <PanelSlot
          side={position.side}
          width={position.width}
          zIndex={slot.zIndex}
        >
          {children}
        </PanelSlot>
      );

    default:
      return null;
  }
}

/**
 * Renders a widget instance using its registered type
 */
function WidgetInstance({ widget }: { widget: WidgetState }): ReactElement | null {
  const typeConfig = WIDGET_TYPES[widget.type];

  if (!typeConfig) {
    console.warn(`Unknown widget type: ${widget.type}`);
    return null;
  }

  const Component = typeConfig.component;
  const props = { ...typeConfig.defaultProps, ...widget.props };

  return <Component {...props} />;
}

/**
 * Main Global UI Renderer
 *
 * Subscribes to globalUI state and renders slots with their widgets.
 */
export const GlobalUIRenderer = memo(function GlobalUIRenderer(): ReactElement | null {
  const [globalUI, loading] = useBackendStateSelector<GlobalUIState, GlobalUIState | null>(
    'globalUI',
    (state) => state
  );

  if (loading || !globalUI) {
    return null;
  }

  const { slots, widgets } = globalUI;

  return (
    <div
      className="global-ui-layer"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        pointerEvents: 'none',
        zIndex: 50,
      }}
    >
      {slots.map((slot) => {
        // Get widgets for this slot, filter by visibility, sort by priority
        const slotWidgets = widgets
          .filter((w) => w.slot === slot.id && w.visible !== false)
          .sort((a, b) => (a.priority ?? 0) - (b.priority ?? 0));

        // Don't render empty slots
        if (slotWidgets.length === 0) {
          return null;
        }

        return (
          <SlotContainer key={slot.id} slot={slot}>
            {slotWidgets.map((widget) => (
              <WidgetInstance key={widget.id} widget={widget} />
            ))}
          </SlotContainer>
        );
      })}
    </div>
  );
});
