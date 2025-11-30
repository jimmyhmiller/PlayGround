import { ReactElement } from 'react';

export const icons: Record<string, ReactElement> = {
  botanist: (
    <svg viewBox="0 0 60 60">
      <path d="M10 10 Q30 5 50 10 Q55 30 50 50 Q30 55 10 50 Q5 30 10 10" />
      <path d="M30 50 Q 45 30 30 10 Q 15 30 30 50" />
    </svg>
  ),
  grid: (
    <svg viewBox="0 0 60 60">
      <path d="M10 5 L50 5 L55 15 L55 55 L15 55 L5 45 Z" />
      <rect x="20" y="20" width="20" height="20" />
    </svg>
  ),
  dream: (
    <svg viewBox="0 0 60 60">
      <rect x="5" y="5" width="50" height="50" rx="20" />
      <circle cx="30" cy="30" r="10" />
    </svg>
  ),
  console: (
    <svg viewBox="0 0 60 60">
      <rect x="5" y="10" width="50" height="40" rx="4" />
      <path d="M15 25 L25 32 L15 39" />
      <line x1="30" y1="39" x2="45" y2="39" />
    </svg>
  ),
  square: (
    <svg viewBox="0 0 60 60">
      <rect x="10" y="10" width="40" height="40" />
    </svg>
  ),
  circle: (
    <svg viewBox="0 0 60 60">
      <circle cx="30" cy="30" r="20" />
    </svg>
  ),
};
