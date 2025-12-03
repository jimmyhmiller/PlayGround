import type { Vec2 } from "./Graph.js";

export interface LayoutProvider {
  // DOM element creation
  createElement(tag: string): HTMLElement;
  createSVGElement(tag: string): SVGElement;

  // Element manipulation
  appendChild(parent: HTMLElement | SVGElement, child: HTMLElement | SVGElement): void;
  setAttribute(element: HTMLElement | SVGElement, name: string, value: string): void;
  setInnerHTML(element: HTMLElement, html: string): void;
  setInnerText(element: HTMLElement, text: string): void;

  // CSS classes
  addClass(element: HTMLElement | SVGElement, className: string): void;
  addClasses(element: HTMLElement | SVGElement, classNames: string[]): void;
  removeClass(element: HTMLElement | SVGElement, className: string): void;
  toggleClass(element: HTMLElement | SVGElement, className: string, force?: boolean): void;

  // Style manipulation
  setStyle(element: HTMLElement, property: string, value: string): void;
  setCSSProperty(element: HTMLElement, property: string, value: string): void;

  // Measurements
  getBoundingClientRect(element: HTMLElement): DOMRect;
  getClientWidth(element: HTMLElement): number;
  getClientHeight(element: HTMLElement): number;

  // Event handling
  addEventListener<K extends keyof HTMLElementEventMap>(
    element: HTMLElement,
    type: K,
    listener: (ev: HTMLElementEventMap[K]) => void
  ): void;

  // Query selectors
  querySelector<T extends HTMLElement = HTMLElement>(parent: HTMLElement, selector: string): T | null;
  querySelectorAll<T extends HTMLElement = HTMLElement>(parent: HTMLElement, selector: string): NodeListOf<T>;

  // Resize observation
  observeResize(element: HTMLElement, callback: (size: Vec2) => void): () => void;

  // Pointer capture
  setPointerCapture(element: HTMLElement, pointerId: number): void;
  releasePointerCapture(element: HTMLElement, pointerId: number): void;
  hasPointerCapture(element: HTMLElement, pointerId: number): boolean;
}
