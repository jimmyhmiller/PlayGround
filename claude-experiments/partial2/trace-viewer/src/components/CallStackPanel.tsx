'use client';

import React from 'react';

interface CallStackPanelProps {
  stack: string[];
}

export function CallStackPanel({ stack }: CallStackPanelProps) {
  return (
    <div className="p-4">
      <h2 className="text-sm font-semibold text-gray-400 mb-2">Call Stack</h2>
      {stack.length === 0 ? (
        <div className="text-gray-500 text-sm">(top level)</div>
      ) : (
        <div className="flex flex-wrap gap-1">
          {stack.map((fn, i) => (
            <React.Fragment key={i}>
              {i > 0 && <span className="text-gray-500">→</span>}
              <span className="px-2 py-0.5 bg-purple-900/50 text-purple-300 rounded text-xs">
                {fn}()
              </span>
            </React.Fragment>
          ))}
        </div>
      )}
    </div>
  );
}
