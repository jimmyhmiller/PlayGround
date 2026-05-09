import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDuration(us: number): string {
  if (us < 1000) return `${Math.round(us)}us`
  if (us < 1_000_000) return `${(us / 1000).toFixed(1)}ms`
  return `${(us / 1_000_000).toFixed(2)}s`
}

export function middleTruncate(str: string, maxLen: number): string {
  if (str.length <= maxLen) return str
  const half = Math.floor((maxLen - 1) / 2)
  return str.slice(0, half) + '…' + str.slice(str.length - half)
}

export const EVENT_KIND_COLORS: Record<string, string> = {
  TaskCreated: 'bg-blue-600',
  TaskScheduled: 'bg-yellow-600',
  TaskStarted: 'bg-sky-600',
  TaskCompleted: 'bg-green-600',
  TaskInvalidated: 'bg-red-600',
  CellUpdated: 'bg-purple-600',
  ChildConnected: 'bg-indigo-600',
  DependencyAdded: 'bg-orange-600',
  StepMarker: 'bg-amber-600',
  ResumeMarker: 'bg-green-600',
}
