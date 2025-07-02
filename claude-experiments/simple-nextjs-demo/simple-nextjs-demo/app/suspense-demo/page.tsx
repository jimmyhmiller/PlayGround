import { Suspense } from 'react';
import ServerFetchComponent from './server-fetch-component';
import ClientFetchComponent from './client-fetch-component';
import FastServerComponent from './fast-server-component';
import LoadingSkeleton from './loading-skeleton';
import FastLoadingSkeleton from './fast-loading-skeleton';
import DelayedSuspense from './delayed-suspense';

export default function SuspenseDemo() {
  return (
    <div className="min-h-screen p-8">
      <h1 className="text-4xl font-bold text-center mb-12">
        Suspense vs Client Fetching Demo
      </h1>
      
      <div className="grid md:grid-cols-2 gap-8 max-w-6xl mx-auto mb-8">
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Server Component with Suspense</h2>
          <p className="text-gray-600 dark:text-gray-400">
            Data fetching happens on the server, HTML is streamed progressively
          </p>
          <div className="border-2 border-blue-500 rounded-lg p-6">
            <Suspense fallback={<LoadingSkeleton />}>
              <ServerFetchComponent />
            </Suspense>
          </div>
        </div>
        
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Client Component Fetching</h2>
          <p className="text-gray-600 dark:text-gray-400">
            Component loads first, then fetches data from the browser
          </p>
          <div className="border-2 border-orange-500 rounded-lg p-6">
            <ClientFetchComponent />
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto mb-8">
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Fast Server Component with Delayed Loading</h2>
          <p className="text-gray-600 dark:text-gray-400">
            Loading state only appears if the request takes longer than 200ms
          </p>
          <div className="border-2 border-green-500 rounded-lg p-6">
            <DelayedSuspense fallback={<FastLoadingSkeleton />} delay={200}>
              <FastServerComponent />
            </DelayedSuspense>
          </div>
        </div>
      </div>
      
      <div className="mt-12 max-w-4xl mx-auto bg-gray-100 dark:bg-gray-800 p-6 rounded-lg">
        <h3 className="text-xl font-semibold mb-4">Key Differences:</h3>
        <ul className="space-y-2 list-disc list-inside">
          <li><strong>Initial Load:</strong> Suspense shows loading state immediately while streaming HTML</li>
          <li><strong>SEO:</strong> Server components have content in initial HTML</li>
          <li><strong>Performance:</strong> No client-server roundtrip with server components</li>
          <li><strong>Bundle Size:</strong> Server components don't add to JavaScript bundle</li>
          <li><strong>Delayed Loading:</strong> Fast requests avoid showing loading state for better UX</li>
        </ul>
      </div>
    </div>
  );
}