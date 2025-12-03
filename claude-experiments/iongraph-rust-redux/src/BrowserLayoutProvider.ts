import type { LayoutProvider } from "./LayoutProvider.js";
import type { Vec2 } from "./Graph.js";

export class BrowserLayoutProvider implements LayoutProvider {
  createElement(tag: string): HTMLElement {
    return document.createElement(tag);
  }

  createSVGElement(tag: string): SVGElement {
    return document.createElementNS("http://www.w3.org/2000/svg", tag);
  }

  appendChild(parent: HTMLElement | SVGElement, child: HTMLElement | SVGElement): void {
    parent.appendChild(child);
  }

  setAttribute(element: HTMLElement | SVGElement, name: string, value: string): void {
    element.setAttribute(name, value);
  }

  setInnerHTML(element: HTMLElement, html: string): void {
    element.innerHTML = html;
  }

  setInnerText(element: HTMLElement, text: string): void {
    element.innerText = text;
  }

  addClass(element: HTMLElement | SVGElement, className: string): void {
    element.classList.add(className);
  }

  addClasses(element: HTMLElement | SVGElement, classNames: string[]): void {
    element.classList.add(...classNames);
  }

  removeClass(element: HTMLElement | SVGElement, className: string): void {
    element.classList.remove(className);
  }

  toggleClass(element: HTMLElement | SVGElement, className: string, force?: boolean): void {
    element.classList.toggle(className, force);
  }

  setStyle(element: HTMLElement, property: string, value: string): void {
    (element.style as any)[property] = value;
  }

  setCSSProperty(element: HTMLElement, property: string, value: string): void {
    element.style.setProperty(property, value);
  }

  getBoundingClientRect(element: HTMLElement): DOMRect {
    return element.getBoundingClientRect();
  }

  getClientWidth(element: HTMLElement): number {
    return element.clientWidth;
  }

  getClientHeight(element: HTMLElement): number {
    return element.clientHeight;
  }

  addEventListener<K extends keyof HTMLElementEventMap>(
    element: HTMLElement,
    type: K,
    listener: (ev: HTMLElementEventMap[K]) => void
  ): void {
    element.addEventListener(type, listener);
  }

  querySelector<T extends HTMLElement = HTMLElement>(parent: HTMLElement, selector: string): T | null {
    return parent.querySelector<T>(selector);
  }

  querySelectorAll<T extends HTMLElement = HTMLElement>(parent: HTMLElement, selector: string): NodeListOf<T> {
    return parent.querySelectorAll<T>(selector);
  }

  observeResize(element: HTMLElement, callback: (size: Vec2) => void): () => void {
    const observer = new ResizeObserver(entries => {
      if (entries.length > 0) {
        const rect = entries[0].contentRect;
        callback({ x: rect.width, y: rect.height });
      }
    });
    observer.observe(element);
    return () => observer.disconnect();
  }

  setPointerCapture(element: HTMLElement, pointerId: number): void {
    element.setPointerCapture(pointerId);
  }

  releasePointerCapture(element: HTMLElement, pointerId: number): void {
    element.releasePointerCapture(pointerId);
  }

  hasPointerCapture(element: HTMLElement, pointerId: number): boolean {
    return element.hasPointerCapture(pointerId);
  }
}
