'use client';

import React, { useState, useCallback } from 'react';
import { Trace } from '../types';
import { TraceViewer } from '../components/TraceViewer';

export default function Home() {
  const [trace, setTrace] = useState<Trace | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setError(null);
    setFileName(file.name);

    try {
      const text = await file.text();
      const data = JSON.parse(text);

      // Validate trace format
      if (!data.events || !Array.isArray(data.events)) {
        throw new Error('Invalid trace format: expected { events: [...] }');
      }

      setTrace(data as Trace);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to parse trace file');
      setTrace(null);
    }
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (!file) return;

    setError(null);
    setFileName(file.name);

    try {
      const text = await file.text();
      const data = JSON.parse(text);

      if (!data.events || !Array.isArray(data.events)) {
        throw new Error('Invalid trace format: expected { events: [...] }');
      }

      setTrace(data as Trace);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to parse trace file');
      setTrace(null);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  if (trace) {
    return (
      <div className="h-screen flex flex-col">
        {/* File info bar */}
        <div className="bg-gray-800 px-4 py-2 flex items-center justify-between text-sm">
          <span className="text-gray-400">
            Loaded: <span className="text-white">{fileName}</span>
          </span>
          <button
            onClick={() => {
              setTrace(null);
              setFileName(null);
            }}
            className="px-3 py-1 rounded bg-gray-700 hover:bg-gray-600 text-xs"
          >
            Load Different File
          </button>
        </div>
        <div className="flex-1">
          <TraceViewer trace={trace} />
        </div>
      </div>
    );
  }

  return (
    <main className="min-h-screen flex items-center justify-center p-8">
      <div
        className="w-full max-w-lg"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <h1 className="text-3xl font-bold text-center mb-2">Trace Viewer</h1>
        <p className="text-gray-400 text-center mb-8">
          Step through jspartial evaluation traces
        </p>

        {/* Drop zone */}
        <label
          className="
            block border-2 border-dashed border-gray-600 rounded-lg p-12
            text-center cursor-pointer hover:border-gray-500 hover:bg-gray-900/50
            transition-colors
          "
        >
          <input
            type="file"
            accept=".json"
            onChange={handleFileSelect}
            className="hidden"
          />
          <div className="text-4xl mb-4">📁</div>
          <div className="text-lg mb-2">Drop trace file here</div>
          <div className="text-gray-500 text-sm">or click to browse</div>
        </label>

        {error && (
          <div className="mt-4 p-4 bg-red-900/30 border border-red-700 rounded-lg text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* Instructions */}
        <div className="mt-8 p-4 bg-gray-900 rounded-lg text-sm">
          <h2 className="font-semibold mb-2">Generate a trace:</h2>
          <code className="block bg-gray-800 p-2 rounded text-xs">
            cargo run -- --trace output.json input.js
          </code>
        </div>
      </div>
    </main>
  );
}
