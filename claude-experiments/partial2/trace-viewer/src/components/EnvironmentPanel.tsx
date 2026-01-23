'use client';

import React from 'react';
import { BindingSnapshot } from '../types';

interface EnvironmentPanelProps {
  bindings: BindingSnapshot[];
  currentSeq: number;
}

export function EnvironmentPanel({ bindings, currentSeq }: EnvironmentPanelProps) {
  // Group bindings by scope
  const byScope = bindings.reduce((acc, binding) => {
    const scope = binding.scope;
    if (!acc[scope]) acc[scope] = [];
    acc[scope].push(binding);
    return acc;
  }, {} as Record<number, BindingSnapshot[]>);

  const scopes = Object.keys(byScope).map(Number).sort();

  return (
    <div className="p-4">
      <h2 className="text-sm font-semibold text-gray-400 mb-3">Environment</h2>

      {bindings.length === 0 ? (
        <div className="text-gray-500 text-sm">No bindings</div>
      ) : (
        <div className="space-y-4">
          {scopes.map((scopeIdx) => (
            <ScopeSection
              key={scopeIdx}
              scopeIdx={scopeIdx}
              bindings={byScope[scopeIdx]}
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
            <span className="text-gray-400">Static (known at compile time)</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-amber-500"></span>
            <span className="text-gray-400">Dynamic (runtime value)</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function ScopeSection({ scopeIdx, bindings }: { scopeIdx: number; bindings: BindingSnapshot[] }) {
  const scopeLabel = scopeIdx === 0 ? 'Global' : `Scope ${scopeIdx}`;

  return (
    <div>
      <div className="text-xs text-gray-500 mb-2 flex items-center gap-2">
        <span className="font-medium">{scopeLabel}</span>
        <span className="text-gray-600">({bindings.length} bindings)</span>
      </div>
      <div className="space-y-1">
        {bindings.map((binding) => (
          <BindingRow key={`${scopeIdx}-${binding.name}`} binding={binding} />
        ))}
      </div>
    </div>
  );
}

function BindingRow({ binding }: { binding: BindingSnapshot }) {
  return (
    <div className="p-2 rounded border border-gray-700 bg-gray-900/50">
      <div className="flex items-center gap-2">
        <span
          className={`w-2 h-2 rounded-full flex-shrink-0 ${
            binding.is_static ? 'bg-green-500' : 'bg-amber-500'
          }`}
        />
        <span className="font-medium text-sm">{binding.name}</span>
        <span className="text-gray-500">=</span>
        <code
          className={`text-xs flex-1 truncate ${
            binding.is_static ? 'text-green-400' : 'text-amber-400'
          }`}
          title={binding.value}
        >
          {binding.value}
        </code>
      </div>
    </div>
  );
}
