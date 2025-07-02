'use client';

import { useState } from 'react';

interface DataItem {
  id: number;
  name: string;
}

interface ClientComponentProps {
  data: {
    message: string;
    timestamp: string;
    items: DataItem[];
  };
}

export default function ClientComponent({ data }: ClientComponentProps) {
  const [clickCount, setClickCount] = useState(0);

  return (
    <div className="bg-gray-100 dark:bg-gray-800 p-8 rounded-lg shadow-lg max-w-2xl w-full">
      <h2 className="text-2xl font-semibold mb-4">Client Component</h2>
      
      <div className="space-y-4">
        <div>
          <p className="text-gray-600 dark:text-gray-400">Message from server:</p>
          <p className="text-lg font-medium">{data.message}</p>
        </div>
        
        <div>
          <p className="text-gray-600 dark:text-gray-400">Server timestamp:</p>
          <p className="text-sm font-mono">{data.timestamp}</p>
        </div>
        
        <div>
          <p className="text-gray-600 dark:text-gray-400">Items from server:</p>
          <ul className="list-disc list-inside mt-2">
            {data.items.map(item => (
              <li key={item.id}>{item.name}</li>
            ))}
          </ul>
        </div>
        
        <div className="pt-4 border-t border-gray-300 dark:border-gray-600">
          <p className="text-gray-600 dark:text-gray-400 mb-2">Client-side interaction:</p>
          <button
            onClick={() => setClickCount(clickCount + 1)}
            className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded transition-colors"
          >
            Clicked {clickCount} times
          </button>
        </div>
      </div>
    </div>
  );
}