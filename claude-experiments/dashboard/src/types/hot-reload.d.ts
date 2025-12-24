/**
 * Type declarations for hot-reload package
 */

declare module 'hot-reload/api' {
  export function getMainPath(): string;
  export function getRendererPath(): string;
  export function reload(): void;
  export function hmrAccept(): void;
  export function defonce<T>(value: T): T;
  export function once<T>(value: T): T;
}
