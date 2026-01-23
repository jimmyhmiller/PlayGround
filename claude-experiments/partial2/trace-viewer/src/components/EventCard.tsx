'use client';

import React from 'react';
import { TraceEntry } from '../types';

interface EventCardProps {
  entry: TraceEntry;
  isCurrent: boolean;
  onClick: () => void;
}

export function EventCard({ entry, isCurrent, onClick }: EventCardProps) {
  const { event, depth, seq } = entry;

  return (
    <div
      className={`event-card cursor-pointer ${isCurrent ? 'current' : ''}`}
      onClick={onClick}
      style={{ marginLeft: depth * 16 }}
    >
      <div className="flex items-center gap-2 mb-1">
        <span className="text-xs text-gray-500">#{seq}</span>
        <EventIcon type={event.type} />
        <span className="font-medium text-sm">{formatEventTitle(event)}</span>
      </div>
      <EventSummary event={event} />
    </div>
  );
}

function EventIcon({ type }: { type: string }) {
  const icons: Record<string, { icon: string; color: string }> = {
    binding_created: { icon: '+', color: 'text-green-400' },
    binding_updated: { icon: '↺', color: 'text-blue-400' },
    function_enter: { icon: '→', color: 'text-purple-400' },
    function_exit: { icon: '←', color: 'text-purple-400' },
    loop_iteration: { icon: '↻', color: 'text-yellow-400' },
    became_dynamic: { icon: '?', color: 'text-orange-400' },
    bailed_out: { icon: '!', color: 'text-red-400' },
    emitted_residual: { icon: '⇥', color: 'text-rose-400' },
  };

  const { icon, color } = icons[type] || { icon: '·', color: 'text-gray-400' };

  return (
    <span className={`font-bold ${color}`}>{icon}</span>
  );
}

function formatEventTitle(event: TraceEntry['event']): string {
  switch (event.type) {
    case 'binding_created':
      return `Create: ${event.name}`;
    case 'binding_updated':
      return `Update: ${event.name}`;
    case 'function_enter':
      return `Enter: ${event.name}()`;
    case 'function_exit':
      return `Exit: ${event.name}()`;
    case 'loop_iteration':
      return `${event.loop_type} iteration #${event.iteration}`;
    case 'became_dynamic':
      return 'Became Dynamic';
    case 'bailed_out':
      return 'Bailed Out';
    case 'emitted_residual':
      return `Residual: ${event.construct}`;
    default:
      return 'Unknown Event';
  }
}

function EventSummary({ event }: { event: TraceEntry['event'] }) {
  switch (event.type) {
    case 'binding_created':
      return (
        <div className="text-xs">
          <span className={event.is_static ? 'static-value' : 'dynamic-value'}>
            {event.value}
          </span>
          {event.cause && (
            <span className="text-gray-500 ml-2">({event.cause})</span>
          )}
        </div>
      );

    case 'binding_updated':
      return (
        <div className="text-xs flex items-center gap-1">
          <span className={event.was_static ? 'static-value' : 'dynamic-value'}>
            {event.old}
          </span>
          <span className="text-gray-500">→</span>
          <span className={event.is_static ? 'static-value' : 'dynamic-value'}>
            {event.new}
          </span>
        </div>
      );

    case 'function_enter':
      return (
        <div className="text-xs">
          {event.args.length === 0 ? (
            <span className="text-gray-500">(no args)</span>
          ) : (
            <span>
              {event.args.map(([repr, isStatic], i) => (
                <span key={i}>
                  {i > 0 && ', '}
                  <span className={isStatic ? 'static-value' : 'dynamic-value'}>
                    {repr}
                  </span>
                </span>
              ))}
            </span>
          )}
        </div>
      );

    case 'function_exit':
      return (
        <div className="text-xs">
          <span className="text-gray-500">returns </span>
          <span className={event.is_static ? 'static-value' : 'dynamic-value'}>
            {event.result}
          </span>
        </div>
      );

    case 'loop_iteration':
      return (
        <div className="text-xs">
          <span className="text-gray-500">condition: </span>
          <span className={event.condition_static ? 'static-value' : 'dynamic-value'}>
            {event.condition}
          </span>
        </div>
      );

    case 'became_dynamic':
      return (
        <div className="text-xs">
          <code className="dynamic-value">{event.expr}</code>
          <span className="text-gray-500 ml-2">{event.reason}</span>
        </div>
      );

    case 'bailed_out':
      return (
        <div className="text-xs text-red-400">
          {event.reason}
          {event.context && (
            <span className="text-gray-500 ml-2">at {event.context}</span>
          )}
        </div>
      );

    case 'emitted_residual':
      return (
        <div className="text-xs">
          <div className="text-rose-400 mb-1">{event.reason}</div>
          {event.residual_preview && event.residual_preview !== '(parent emits)' && (
            <code className="text-gray-400 text-[10px] block truncate max-w-xs">
              {event.residual_preview}
            </code>
          )}
        </div>
      );

    default:
      return null;
  }
}
