"use client";
// The `error.tsx` boundary convention: a client component rendered when its subtree
// throws. The adapter interns it as an `error: M<i>` level around the boom route.
export default function Err({ error, reset }: { error: Error; reset: () => void }) {
  return (
    <main id="boom-error">
      error caught: {error.message}
      <button id="reset" onClick={reset}>
        retry
      </button>
    </main>
  );
}
