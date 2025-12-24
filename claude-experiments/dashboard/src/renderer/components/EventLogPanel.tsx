import { memo, useRef, useMemo } from 'react';
import { FixedSizeList as List } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import { useEventSubscription } from '../hooks/useEvents';
import type { DashboardEvent } from '../../types/events';

// Fixed row height
const ROW_HEIGHT = 70;

interface EventRowProps {
  event: DashboardEvent;
  style: React.CSSProperties;
}

/**
 * Single event row component
 */
const EventRow = memo(function EventRow({ event, style }: EventRowProps) {
  const hasPayload = Object.keys(event.payload as object).length > 0;

  return (
    <div style={{ ...style, padding: '0 8px' }}>
      <div
        className="event-log-item"
        style={{
          height: ROW_HEIGHT - 4,
          padding: '6px 8px',
          background: 'var(--theme-bg-tertiary)',
          borderRadius: 'var(--theme-radius-sm)',
          borderLeft: `3px solid var(${getEventColorVar(event.type)})`,
          boxSizing: 'border-box',
          overflow: 'hidden',
        }}
      >
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginBottom: '2px',
        }}>
          <span style={{
            color: `var(${getEventColorVar(event.type)})`,
            fontWeight: 500,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}>
            {event.type}
          </span>
          <span style={{
            color: 'var(--theme-text-disabled)',
            fontSize: 'var(--theme-font-size-xs)',
            flexShrink: 0,
            marginLeft: 8,
          }}>
            {new Date(event.timestamp).toLocaleTimeString()}
          </span>
        </div>
        <div style={{
          color: 'var(--theme-text-muted)',
          fontSize: 'var(--theme-font-size-xs)',
        }}>
          [{event.meta.source}]
          {event.meta.correlationId && (
            <span style={{ marginLeft: '8px', color: 'var(--theme-text-disabled)' }}>
              corr: {event.meta.correlationId.slice(-8)}
            </span>
          )}
        </div>
        {hasPayload && (
          <div style={{
            marginTop: '2px',
            padding: '2px 6px',
            background: 'var(--theme-bg-secondary)',
            borderRadius: '2px',
            color: 'var(--theme-code-string)',
            fontSize: 'var(--theme-font-size-xs)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}>
            {JSON.stringify(event.payload)}
          </div>
        )}
      </div>
    </div>
  );
});

interface RowProps {
  index: number;
  style: React.CSSProperties;
  data: DashboardEvent[];
}

/**
 * Row renderer for the virtualized list
 */
const Row = ({ index, style, data }: RowProps) => {
  const event = data[index];
  if (!event) return null;
  return <EventRow event={event} style={style} />;
};

interface EventLogPanelProps {
  subscribePattern?: string;
  maxEvents?: number;
}

/**
 * Event Log Panel
 * Shows live stream of events matching a pattern using virtualized list
 */
const EventLogPanel = memo(function EventLogPanel({
  subscribePattern = '**',
  maxEvents = 100,
}: EventLogPanelProps) {
  const events = useEventSubscription(subscribePattern, { maxEvents });
  const renderCount = useRef(0);
  renderCount.current += 1;

  // Reverse events for display (newest first)
  const reversedEvents = useMemo(() => events.slice().reverse(), [events]);

  return (
    <div
      className="event-log"
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        background: 'var(--theme-bg-secondary)',
        fontFamily: 'var(--theme-font-family)',
      }}
    >
      {/* Header */}
      <div
        className="event-log-header"
        style={{
          padding: 'var(--theme-spacing-sm) var(--theme-spacing-md)',
          background: 'var(--theme-bg-tertiary)',
          borderBottom: '1px solid var(--theme-border-primary)',
          fontSize: 'var(--theme-font-size-sm)',
          color: 'var(--theme-text-muted)',
          display: 'flex',
          justifyContent: 'space-between',
        }}
      >
        <span>
          Pattern: <code style={{ color: 'var(--theme-code-keyword)' }}>{subscribePattern}</code>
          {' • '}
          {events.length} events
        </span>
        <span style={{ color: 'var(--theme-accent-warning)' }}>renders: {renderCount.current}</span>
      </div>

      {/* Event list */}
      <div
        className="event-log-list"
        style={{
          flex: 1,
          fontFamily: 'var(--theme-font-mono)',
          fontSize: 'var(--theme-font-size-sm)',
        }}
      >
        {events.length === 0 ? (
          <div style={{
            color: 'var(--theme-text-disabled)',
            textAlign: 'center',
            padding: 'var(--theme-spacing-xl)',
          }}>
            Waiting for events...
          </div>
        ) : (
          <AutoSizer>
            {({ height, width }: { height: number; width: number }) => (
              <List
                height={height}
                width={width}
                itemCount={reversedEvents.length}
                itemSize={ROW_HEIGHT}
                itemData={reversedEvents}
              >
                {Row}
              </List>
            )}
          </AutoSizer>
        )}
      </div>
    </div>
  );
});

function getEventColorVar(type: string): string {
  if (type.startsWith('user.')) return '--theme-event-user';
  if (type.startsWith('data.')) return '--theme-event-data';
  if (type.startsWith('file.')) return '--theme-event-file';
  if (type.startsWith('git.')) return '--theme-event-git';
  if (type.startsWith('system.')) return '--theme-event-system';
  if (type.startsWith('command.')) return '--theme-event-command';
  if (type.startsWith('editor.')) return '--theme-event-editor';
  return '--theme-text-muted';
}

export default EventLogPanel;
