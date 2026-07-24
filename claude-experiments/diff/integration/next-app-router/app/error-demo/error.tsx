"use client";
// The `error.tsx` convention: a client component rendered by the generated
// ErrorBoundary when its subtree throws. Receives `{ error, reset }`.
export default function Err({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <main id="error-demo">
      error caught: {error.message}
      <button id="reset" onClick={reset}>
        retry
      </button>
    </main>
  );
}
