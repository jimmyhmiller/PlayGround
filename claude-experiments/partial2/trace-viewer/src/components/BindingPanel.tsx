'use client';

import React from 'react';
import { BindingState } from '../types';

interface BindingPanelProps {
  bindings: BindingState[];
  currentSeq: number;
}

export function BindingPanel({ bindings, currentSeq }: BindingPanelProps) {
  // Sort bindings: most recently updated first
  const sortedBindings = [...bindings].sort((a, b) => b.lastUpdatedAt - a.lastUpdatedAt);

  return (
    <div className="p-4">
      <h2 className="text-sm font-semibold text-gray-400 mb-3">Bindings</h2>

      {sortedBindings.length === 0 ? (
        <div className="text-gray-500 text-sm">No bindings yet</div>
      ) : (
        <div className="space-y-2">
          {sortedBindings.map((binding) => (
            <BindingRow
              key={binding.name}
              binding={binding}
              isRecent={binding.lastUpdatedAt === currentSeq}
            />
          ))}
        </div>
      )}

      {/* Legend */}
      <div className="mt-6 pt-4 border-t border-gray-700">
        <h3 className="text-xs font-semibold text-gray-500 mb-2">Legend</h3>
        <div className="flex gap-4 text-xs">
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-green-500"></span>
            <span className="text-gray-400">Static</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-amber-500"></span>
            <span className="text-gray-400">Dynamic</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function BindingRow({ binding, isRecent }: { binding: BindingState; isRecent: boolean }) {
  return (
    <div
      className={`
        p-2 rounded border transition-all
        ${isRecent ? 'border-blue-500 bg-blue-950/30' : 'border-gray-700 bg-gray-900/50'}
      `}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              binding.isStatic ? 'bg-green-500' : 'bg-amber-500'
            }`}
          />
          <span className="font-medium text-sm">{binding.name}</span>
        </div>
        <span className="text-xs text-gray-500">@{binding.lastUpdatedAt}</span>
      </div>
      <div className="mt-1 pl-4">
        <code
          className={`text-xs ${binding.isStatic ? 'static-value' : 'dynamic-value'}`}
        >
          {binding.value}
        </code>
      </div>
    </div>
  );
}
