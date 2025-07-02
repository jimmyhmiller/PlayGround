'use client';

import { Suspense, useState, useEffect, ReactNode, useRef } from 'react';

interface DelayedSuspenseProps {
  fallback: ReactNode;
  delay?: number;
  children: ReactNode;
}

function DelayedFallback({ fallback, delay = 200 }: { fallback: ReactNode; delay?: number }) {
  const [show, setShow] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    timeoutRef.current = setTimeout(() => {
      setShow(true);
    }, delay);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [delay]);

  if (!show) return null;
  return <>{fallback}</>;
}

export default function DelayedSuspense({ 
  fallback, 
  delay = 200, 
  children 
}: DelayedSuspenseProps) {
  return (
    <Suspense fallback={<DelayedFallback fallback={fallback} delay={delay} />}>
      {children}
    </Suspense>
  );
}