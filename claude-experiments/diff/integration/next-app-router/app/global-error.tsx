"use client";
// The `global-error.tsx` convention: the app-root error boundary that owns the entire
// document (its own <html>/<body>). When a throw escapes every nested error.tsx
// (including one in the root layout), diffpack's generated document tree replaces the
// whole tree with this component. It receives `{ error, reset }` like error.tsx.
export default function GlobalError({
  error,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body>
        <main id="global-error">global-error boundary: {error.message}</main>
      </body>
    </html>
  );
}
