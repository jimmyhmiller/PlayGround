'use client';

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Trace, TraceEntry, BindingSnapshot } from '../types';
import { EventCard } from './EventCard';
import { EnvironmentPanel } from './EnvironmentPanel';
import { StepControls } from './StepControls';
import { CallStackPanel } from './CallStackPanel';
import { SourceCodePanel } from './SourceCodePanel';

// Event types that are residual-related
const RESIDUAL_TYPES = ['emitted_residual', 'became_dynamic', 'bailed_out'] as const;

interface TraceViewerProps {
  trace: Trace;
}

export function TraceViewer({ trace }: TraceViewerProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [residualOnly, setResidualOnly] = useState(false);
  const totalSteps = trace.events.length;

  // Get indices of residual-related events
  const residualIndices = useMemo(() => {
    return trace.events
      .map((e, i) => ({ event: e, index: i }))
      .filter(({ event }) => RESIDUAL_TYPES.includes(event.event.type as any))
      .map(({ index }) => index);
  }, [trace.events]);

  // Filter events if residualOnly is enabled
  const filteredEvents = useMemo(() => {
    if (!residualOnly) {
      return trace.events.map((entry, index) => ({ entry, index }));
    }
    return trace.events
      .map((entry, index) => ({ entry, index }))
      .filter(({ entry }) => RESIDUAL_TYPES.includes(entry.event.type as any));
  }, [trace.events, residualOnly]);

  // Find next/prev residual from current position
  const nextResidualIndex = useMemo(() => {
    return residualIndices.find(i => i > currentStep) ?? null;
  }, [residualIndices, currentStep]);

  const prevResidualIndex = useMemo(() => {
    return [...residualIndices].reverse().find(i => i < currentStep) ?? null;
  }, [residualIndices, currentStep]);

  // Current entry
  const currentEntry = trace.events[currentStep];

  // Get environment from the trace entry (if available)
  const environment: BindingSnapshot[] = currentEntry?.env || [];

  // Keyboard navigation
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'ArrowRight' || e.key === 'j' || e.key === 'n') {
      setCurrentStep(s => Math.min(s + 1, totalSteps - 1));
    } else if (e.key === 'ArrowLeft' || e.key === 'k' || e.key === 'p') {
      setCurrentStep(s => Math.max(s - 1, 0));
    } else if (e.key === 'Home' || e.key === 'g') {
      setCurrentStep(0);
    } else if (e.key === 'End' || e.key === 'G') {
      setCurrentStep(totalSteps - 1);
    } else if (e.key === 'r') {
      // Toggle residual filter
      setResidualOnly(v => !v);
    } else if (e.key === ']') {
      // Jump to next residual
      const next = residualIndices.find(i => i > currentStep);
      if (next !== undefined) setCurrentStep(next);
    } else if (e.key === '[') {
      // Jump to prev residual
      const prev = [...residualIndices].reverse().find(i => i < currentStep);
      if (prev !== undefined) setCurrentStep(prev);
    }
  }, [totalSteps, residualIndices, currentStep]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  // Get surrounding events for context (works with filtered view)
  const visibleEvents = useMemo(() => {
    if (residualOnly) {
      // In filtered mode, find current position in filtered list
      const currentFilteredIndex = filteredEvents.findIndex(({ index }) => index === currentStep);
      const centerIndex = currentFilteredIndex >= 0 ? currentFilteredIndex : 0;
      const start = Math.max(0, centerIndex - 5);
      const end = Math.min(filteredEvents.length, centerIndex + 6);
      return filteredEvents.slice(start, end);
    } else {
      const start = Math.max(0, currentStep - 5);
      const end = Math.min(totalSteps, currentStep + 6);
      return trace.events.slice(start, end).map((entry, i) => ({
        entry,
        index: start + i,
      }));
    }
  }, [trace, currentStep, totalSteps, residualOnly, filteredEvents]);

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold">Trace Viewer</h1>
            {/* Residual filter toggle */}
            <button
              onClick={() => setResidualOnly(v => !v)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                residualOnly
                  ? 'bg-rose-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Residual Only ({residualIndices.length})
            </button>
            {/* Jump buttons */}
            <div className="flex items-center gap-1">
              <button
                onClick={() => prevResidualIndex !== null && setCurrentStep(prevResidualIndex)}
                disabled={prevResidualIndex === null}
                className="px-2 py-1 rounded text-sm bg-gray-700 text-gray-300 hover:bg-gray-600 disabled:opacity-40 disabled:cursor-not-allowed"
                title="Previous residual ([)"
              >
                ← Prev
              </button>
              <button
                onClick={() => nextResidualIndex !== null && setCurrentStep(nextResidualIndex)}
                disabled={nextResidualIndex === null}
                className="px-2 py-1 rounded text-sm bg-gray-700 text-gray-300 hover:bg-gray-600 disabled:opacity-40 disabled:cursor-not-allowed"
                title="Next residual (])"
              >
                Next →
              </button>
            </div>
          </div>
          <div className="text-sm text-gray-400">
            Step {currentStep + 1} of {totalSteps}
            {residualOnly && (
              <span className="ml-2 text-rose-400">
                (showing {filteredEvents.length} residual events)
              </span>
            )}
          </div>
        </div>
      </header>

      {/* Main content - 3 column layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left panel: Source Code */}
        {trace.source && (
          <div className="w-1/3 border-r border-gray-700 flex flex-col">
            <SourceCodePanel
              source={trace.source}
              highlightLine={currentEntry?.location?.line}
            />
          </div>
        )}

        {/* Middle panel: Event list */}
        <div className={`${trace.source ? 'w-1/3' : 'w-1/2'} border-r border-gray-700 flex flex-col`}>
          {/* Event timeline */}
          <div className="flex-1 overflow-y-auto p-4 space-y-2">
            {visibleEvents.map(({ entry, index }) => (
              <EventCard
                key={entry.seq}
                entry={entry}
                isCurrent={index === currentStep}
                onClick={() => setCurrentStep(index)}
              />
            ))}
          </div>

          {/* Step controls */}
          <StepControls
            currentStep={currentStep}
            totalSteps={totalSteps}
            onStepChange={setCurrentStep}
          />
        </div>

        {/* Right panel: State view */}
        <div className={`${trace.source ? 'w-1/3' : 'w-1/2'} flex flex-col`}>
          {/* Call stack */}
          <div className="border-b border-gray-700">
            <CallStackPanel stack={currentEntry?.stack || []} />
          </div>

          {/* Current event detail */}
          <div className="border-b border-gray-700 p-4">
            <h2 className="text-sm font-semibold text-gray-400 mb-2">Current Event</h2>
            {currentEntry && (
              <EventDetail entry={currentEntry} />
            )}
          </div>

          {/* Environment */}
          <div className="flex-1 overflow-y-auto">
            <EnvironmentPanel
              bindings={environment}
              currentSeq={currentStep}
            />
          </div>
        </div>
      </div>

      {/* Footer with keyboard hints */}
      <footer className="border-t border-gray-700 p-2 text-xs text-gray-500">
        <span className="mr-4">← → or j/k: Navigate</span>
        <span className="mr-4">Home/g: First</span>
        <span className="mr-4">End/G: Last</span>
        <span className="mr-4 text-rose-400">r: Toggle residual filter</span>
        <span className="text-rose-400">[ ]: Jump prev/next residual</span>
      </footer>
    </div>
  );
}

function EventDetail({ entry }: { entry: TraceEntry }) {
  const { event, location } = entry;

  return (
    <div className="space-y-2 text-sm">
      <div className="flex items-center gap-2 flex-wrap">
        <EventTypeBadge type={event.type} />
        <span className="text-gray-400">Depth: {entry.depth}</span>
        {location && (
          <span className="text-gray-500 text-xs">
            Line {location.line}:{location.column}
          </span>
        )}
      </div>

      <div className="bg-gray-900 rounded p-3 font-mono text-xs">
        <pre className="whitespace-pre-wrap">{JSON.stringify(event, null, 2)}</pre>
      </div>
    </div>
  );
}

function EventTypeBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
    binding_created: 'bg-green-900 text-green-300',
    binding_updated: 'bg-blue-900 text-blue-300',
    function_enter: 'bg-purple-900 text-purple-300',
    function_exit: 'bg-purple-900 text-purple-300',
    loop_iteration: 'bg-yellow-900 text-yellow-300',
    became_dynamic: 'bg-orange-900 text-orange-300',
    bailed_out: 'bg-red-900 text-red-300',
    emitted_residual: 'bg-rose-900 text-rose-300',
  };

  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${colors[type] || 'bg-gray-700'}`}>
      {type}
    </span>
  );
}
