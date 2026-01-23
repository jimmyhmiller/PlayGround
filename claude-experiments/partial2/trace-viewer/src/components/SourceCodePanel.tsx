'use client';

import React, { useEffect, useRef } from 'react';
import Prism from 'prismjs';
import 'prismjs/components/prism-javascript';

interface SourceCodePanelProps {
  source: string;
  highlightLine?: number;
}

export function SourceCodePanel({ source, highlightLine }: SourceCodePanelProps) {
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (codeRef.current) {
      Prism.highlightElement(codeRef.current);
    }
  }, [source]);

  // Split source into lines for line numbers
  const lines = source.split('\n');

  return (
    <div className="h-full flex flex-col">
      <h2 className="text-sm font-semibold text-gray-400 p-3 border-b border-gray-700">
        Source Code
      </h2>
      <div className="flex-1 overflow-auto">
        <div className="relative">
          {/* Line numbers */}
          <div className="absolute left-0 top-0 text-right pr-2 select-none text-gray-500 text-xs font-mono leading-6 pt-4 pl-2">
            {lines.map((_, i) => (
              <div
                key={i}
                className={`${
                  highlightLine === i + 1
                    ? 'bg-yellow-500/20 text-yellow-400'
                    : ''
                }`}
              >
                {i + 1}
              </div>
            ))}
          </div>

          {/* Code content */}
          <div className="pl-12 pt-4 pr-4 pb-4">
            <pre className="!bg-transparent !p-0 !m-0">
              <code ref={codeRef} className="language-javascript text-xs leading-6">
                {source}
              </code>
            </pre>

            {/* Highlight overlay */}
            {highlightLine && (
              <div
                className="absolute left-0 right-0 bg-yellow-500/10 pointer-events-none"
                style={{
                  top: `${(highlightLine - 1) * 24 + 16}px`,
                  height: '24px',
                }}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
