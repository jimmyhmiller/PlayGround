import type { LayoutProvider } from "./LayoutProvider.js";
import type { Vec2 } from "./Graph.js";

/**
 * Element representation for pure SVG (text-based, no foreignObject)
 */
interface SVGTextNode {
  type: 'svg' | 'g' | 'rect' | 'text' | 'path' | 'line';
  attributes: Map<string, string>;
  children: SVGTextNode[];
  textContent?: string;
  classList: Set<string>;
  parent?: SVGTextNode;
}

/**
 * Pure SVG layout provider that uses native SVG text elements instead of foreignObject.
 * This ensures maximum compatibility and visibility.
 */
export class PureSVGTextLayoutProvider implements LayoutProvider {
  private elementSizes = new WeakMap<SVGTextNode, Vec2>();
  private eventListeners = new WeakMap<SVGTextNode, Map<string, Function[]>>();

  // Font metrics for monospace
  private charWidth = 7;
  private charHeight = 14;
  private padding = 8;

  createElement(tag: string): any {
    // Create a group to represent an HTML element
    const g = this.createNode('g');
    const rect = this.createNode('rect');
    this.appendChild(g, rect);
    (g as any).__rect = rect;
    (g as any).__htmlElement = g;
    return g;
  }

  createSVGElement(tag: string): any {
    return this.createNode(tag as any);
  }

  private createNode(type: SVGTextNode['type']): SVGTextNode {
    return {
      type,
      attributes: new Map(),
      children: [],
      classList: new Set(),
    };
  }

  appendChild(parent: any, child: any): void {
    parent.children.push(child);
    child.parent = parent;
  }

  setAttribute(element: any, name: string, value: string): void {
    element.attributes.set(name, value);
  }

  setInnerHTML(element: any, html: string): void {
    // Strip HTML tags and convert to text
    const text = html.replace(/<[^>]*>/g, '').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
    element.textContent = text;
    this.updateEstimatedSize(element);
  }

  setInnerText(element: any, text: string): void {
    element.textContent = text;
    this.updateEstimatedSize(element);
  }

  addClass(element: any, className: string): void {
    element.classList.add(className);
  }

  addClasses(element: any, classNames: string[]): void {
    classNames.forEach(c => element.classList.add(c));
  }

  removeClass(element: any, className: string): void {
    element.classList.delete(className);
  }

  toggleClass(element: any, className: string, force?: boolean): void {
    if (force === undefined) {
      if (element.classList.has(className)) {
        element.classList.delete(className);
      } else {
        element.classList.add(className);
      }
    } else if (force) {
      element.classList.add(className);
    } else {
      element.classList.delete(className);
    }
  }

  setStyle(element: any, property: string, value: string): void {
    // Convert CSS to SVG attributes where applicable
    if (property === 'left') {
      const x = parseFloat(value);
      this.setAttribute(element, 'transform', `translate(${x}, ${element.attributes.get('transform')?.match(/translate\([^,]+,\s*([^)]+)\)/)?.[1] || 0})`);
    } else if (property === 'top') {
      const y = parseFloat(value);
      const x = element.attributes.get('transform')?.match(/translate\(([^,]+)/)?.[1] || 0;
      this.setAttribute(element, 'transform', `translate(${x}, ${y})`);
    }
  }

  setCSSProperty(element: any, property: string, value: string): void {
    // Store as data attribute
    this.setAttribute(element, `data-${property}`, value);
  }

  getBoundingClientRect(element: any): DOMRect {
    const size = this.elementSizes.get(element) || { x: 100, y: 50 };
    const transform = element.attributes.get('transform') || '';
    const match = transform.match(/translate\(([^,]+),\s*([^)]+)\)/);
    const x = match ? parseFloat(match[1]) : 0;
    const y = match ? parseFloat(match[2]) : 0;

    return {
      x, y,
      width: size.x,
      height: size.y,
      left: x,
      right: x + size.x,
      top: y,
      bottom: y + size.y,
      toJSON: () => ({}),
    } as DOMRect;
  }

  getClientWidth(element: any): number {
    const size = this.elementSizes.get(element);
    if (size) return size.x;
    this.updateEstimatedSize(element);
    return this.elementSizes.get(element)?.x || 100;
  }

  getClientHeight(element: any): number {
    const size = this.elementSizes.get(element);
    if (size) return size.y;
    this.updateEstimatedSize(element);
    return this.elementSizes.get(element)?.y || 50;
  }

  addEventListener(element: any, type: any, listener: any): void {
    if (!this.eventListeners.has(element)) {
      this.eventListeners.set(element, new Map());
    }
    const listeners = this.eventListeners.get(element)!;
    if (!listeners.has(type)) {
      listeners.set(type, []);
    }
    listeners.get(type)!.push(listener);
  }

  querySelector(parent: any, selector: string): any {
    return this.queryNode(parent, selector);
  }

  querySelectorAll(parent: any, selector: string): any {
    const results: any[] = [];
    this.queryAllNodes(parent, selector, results);
    return {
      ...results,
      length: results.length,
      item: (index: number) => results[index],
      forEach: (callback: any) => results.forEach(callback),
      [Symbol.iterator]: function* () { yield* results; }
    };
  }

  private queryNode(node: SVGTextNode, selector: string): SVGTextNode | null {
    if (selector.startsWith('.')) {
      const className = selector.substring(1);
      if (node.classList.has(className)) return node;
    } else if (selector.startsWith('[')) {
      const match = selector.match(/\[([^=]+)="?([^"\]]+)"?\]/);
      if (match) {
        const [, attr, value] = match;
        if (node.attributes.get(attr) === value) return node;
      }
    }
    for (const child of node.children) {
      const result = this.queryNode(child, selector);
      if (result) return result;
    }
    return null;
  }

  private queryAllNodes(node: SVGTextNode, selector: string, results: any[]): void {
    let matches = false;
    if (selector.startsWith('.')) {
      const className = selector.substring(1);
      if (node.classList.has(className)) matches = true;
    } else if (selector.startsWith('[')) {
      const match = selector.match(/\[([^=]+)="?([^"\]]+)"?\]/);
      if (match) {
        const [, attr, value] = match;
        if (node.attributes.get(attr) === value) matches = true;
      }
    }
    if (matches) results.push(node);
    for (const child of node.children) {
      this.queryAllNodes(child, selector, results);
    }
  }

  observeResize(element: any, callback: (size: Vec2) => void): () => void {
    const size = this.elementSizes.get(element) || { x: 800, y: 600 };
    callback(size);
    return () => {};
  }

  setPointerCapture(): void {}
  releasePointerCapture(): void {}
  hasPointerCapture(): boolean { return false; }

  private updateEstimatedSize(element: any): void {
    const text = element.textContent || '';

    // Special handling for blocks - need to account for multi-column layout
    if (element.classList.has('ig-block')) {
      // Calculate actual width based on instruction content
      const { width, height } = this.calculateBlockSize(element);
      this.elementSizes.set(element, { x: width, y: height });
      return;
    }

    const lines = text.split('\n');
    const maxLineLength = Math.max(...lines.map((line: string) => line.length), 10);
    const width = maxLineLength * this.charWidth + this.padding * 2;
    const height = lines.length * this.charHeight + this.padding * 2;
    this.elementSizes.set(element, { x: width, y: height });
  }

  private calculateBlockSize(element: any): { width: number, height: number } {
    const rows = this.findInstructionRows(element);

    // Measure maximum width of each column
    let maxNumWidth = 0;
    let maxOpcodeWidth = 0;
    let maxTypeWidth = 0;

    for (const row of rows) {
      const numText = row.num?.textContent || '';
      const opcodeText = row.opcode?.textContent || '';
      const typeText = row.type?.textContent || '';

      maxNumWidth = Math.max(maxNumWidth, numText.length * this.charWidth);
      maxOpcodeWidth = Math.max(maxOpcodeWidth, opcodeText.length * this.charWidth);
      maxTypeWidth = Math.max(maxTypeWidth, typeText.length * this.charWidth);
    }

    // Also consider header width
    for (const child of element.children) {
      if (child.classList.has('ig-block-header') && child.textContent) {
        const headerWidth = child.textContent.length * this.charWidth;
        const minWidth = headerWidth + this.padding * 2;
        const calculatedWidth = this.padding + maxNumWidth + 8 + maxOpcodeWidth + 8 + maxTypeWidth + this.padding;
        const width = Math.max(minWidth, calculatedWidth, 150);
        const height = 30 + (rows.length * this.charHeight) + this.padding * 2;
        return { width, height };
      }
    }

    // Calculate total width: padding + num + gap + opcode + gap + type + padding
    const width = Math.max(
      this.padding + maxNumWidth + 8 + maxOpcodeWidth + 8 + maxTypeWidth + this.padding,
      150 // minimum width
    );
    const height = 30 + (rows.length * this.charHeight) + this.padding * 2;

    return { width, height };
  }

  private countInstructions(element: any): number {
    let count = 0;
    const countInNode = (node: any) => {
      if (node.classList && node.classList.has('ig-ins')) {
        count++;
      }
      if (node.children) {
        for (const child of node.children) {
          countInNode(child);
        }
      }
    };
    countInNode(element);
    return count;
  }

  setViewportSize(element: any, width: number, height: number): void {
    this.elementSizes.set(element, { x: width, y: height });
  }

  /**
   * Convert to SVG string with actual rendering
   */
  toSVGString(root: any): string {
    return this.renderNode(root, 0);
  }

  private renderNode(node: SVGTextNode, depth: number): string {
    const indent = '  '.repeat(depth);
    const attrs: string[] = [];

    // Add attributes
    node.attributes.forEach((value, key) => {
      attrs.push(`${key}="${this.escape(value)}"`);
    });

    // Add class
    if (node.classList.size > 0) {
      attrs.push(`class="${Array.from(node.classList).join(' ')}"`);
    }

    const attrStr = attrs.length > 0 ? ' ' + attrs.join(' ') : '';

    // Handle different node types
    if (node.type === 'g' && (node as any).__rect && node.classList.has('ig-block')) {
      // This is a block element, render specially
      return this.renderBlockGroup(node, depth);
    }

    // Regular SVG elements
    if (node.children.length === 0 && !node.textContent) {
      return `${indent}<${node.type}${attrStr}/>`;
    }

    let result = `${indent}<${node.type}${attrStr}>`;

    if (node.textContent && node.type === 'text') {
      result += this.escape(node.textContent);
    } else if (node.children.length > 0) {
      result += '\n';
      for (const child of node.children) {
        result += this.renderNode(child, depth + 1) + '\n';
      }
      result += indent;
    }

    result += `</${node.type}>`;
    return result;
  }

  private renderBlockGroup(node: SVGTextNode, depth: number): string {
    const indent = '  '.repeat(depth);
    const transform = node.attributes.get('transform') || '';
    const match = transform.match(/translate\(([^,]+),\s*([^)]+)\)/);
    const x = match ? parseFloat(match[1]) : 0;
    const y = match ? parseFloat(match[2]) : 0;

    const size = this.elementSizes.get(node) || { x: 100, y: 50 };
    const isBlock = node.classList.has('ig-block');
    const isHeader = node.classList.has('ig-block-header');

    let result = `${indent}<g transform="translate(${x}, ${y})">`;

    if (isBlock) {
      // Render block background with light fill and dark border
      result += `\n${indent}  <rect x="0" y="0" width="${size.x}" height="${size.y}" fill="#f9f9f9" stroke="#0c0c0d" stroke-width="1"/>`;

      // Render the block content recursively with proper positioning
      result += this.renderBlockContent(node, depth + 1, 0);
    } else if (node.textContent) {
      const lines = node.textContent.split('\n');
      let textY = isHeader ? 16 : this.charHeight;
      for (const line of lines) {
        if (line.trim()) { // Only render non-empty lines
          result += `\n${indent}  <text x="${this.padding}" y="${textY}" font-family="monospace" font-size="12" fill="black">${this.escape(line)}</text>`;
        }
        textY += this.charHeight;
      }
    }

    result += `\n${indent}</g>`;
    return result;
  }

  private renderBlockContent(node: SVGTextNode, depth: number, startY: number): string {
    const indent = '  '.repeat(depth);
    let result = '';
    let currentY = startY;

    // Check if this is a loop header
    const isLoopHeader = node.classList.has('ig-block-att-loopheader');
    const headerBg = isLoopHeader ? '#1fa411' : '#0c0c0d';

    // Render header first if present
    for (const child of node.children) {
      if (child.classList.has('ig-block-header') && child.textContent) {
        const size = this.elementSizes.get(node) || { x: 250, y: 50 };
        // Draw header background rectangle
        result += `\n${indent}<rect x="0" y="0" width="${size.x}" height="28" fill="${headerBg}"/>`;
        // Draw header text (centered)
        result += `\n${indent}<text x="${size.x / 2}" y="18" font-family="monospace" font-size="12" fill="white" font-weight="bold" text-anchor="middle">${this.escape(child.textContent)}</text>`;
        currentY = 28 + this.padding + 4; // Start instructions below header with extra padding
        break;
      }
    }

    // Render instructions table
    for (const child of node.children) {
      if (child.classList.has('ig-instructions')) {
        result += this.renderInstructions(child, depth, currentY);
        break;
      }
    }

    // Render edge labels (for blocks with 2 successors)
    const size = this.elementSizes.get(node) || { x: 250, y: 50 };
    for (const child of node.children) {
      if (child.classList.has('ig-edge-label') && child.textContent) {
        // Get the x position from the transform attribute
        const transform = child.attributes.get('transform') || '';
        const match = transform.match(/translate\(([^,]+)/);
        const x = match ? parseFloat(match[1]) : 0;

        // Position at bottom of block (labels use bottom: -1em in CSS, which is ~14px below)
        const labelY = size.y + 12;
        result += `\n${indent}<text x="${x + 4}" y="${labelY}" font-family="monospace" font-size="9" fill="#777">${this.escape(child.textContent)}</text>`;
      }
    }

    return result;
  }

  private renderInstructions(node: SVGTextNode, depth: number, startY: number): string {
    const indent = '  '.repeat(depth);
    let result = '';
    let currentY = startY;

    // Find all instruction rows
    const rows = this.findInstructionRows(node);

    // Measure maximum width of each column
    let maxNumWidth = 0;
    let maxOpcodeWidth = 0;

    for (const row of rows) {
      const numText = row.num?.textContent || '';
      const opcodeText = row.opcode?.textContent || '';
      maxNumWidth = Math.max(maxNumWidth, numText.length * this.charWidth);
      maxOpcodeWidth = Math.max(maxOpcodeWidth, opcodeText.length * this.charWidth);
    }

    // Calculate column positions with proper spacing
    const numX = this.padding;
    const opcodeX = this.padding + maxNumWidth + 8;
    const typeX = opcodeX + maxOpcodeWidth + 8;

    for (const row of rows) {
      const numText = row.num?.textContent || '';
      const opcodeText = row.opcode?.textContent || '';
      const typeText = row.type?.textContent || '';

      // Check for attributes that affect styling
      const hasMovable = row.rowNode.classList.has('ig-ins-att-Movable');
      const hasGuard = row.rowNode.classList.has('ig-ins-att-Guard');
      const hasROB = row.rowNode.classList.has('ig-ins-att-RecoveredOnBailout');

      // Determine text color and decoration
      let textColor = 'black';
      let textDecoration = '';
      if (hasMovable) textColor = '#1048af';
      if (hasROB) textColor = '#444';
      if (hasGuard) textDecoration = ' text-decoration="underline"';

      // Render instruction with dynamically positioned columns
      result += `\n${indent}<text x="${numX}" y="${currentY}" font-family="monospace" font-size="11" fill="#777">${this.escape(numText)}</text>`;
      result += `\n${indent}<text x="${opcodeX}" y="${currentY}" font-family="monospace" font-size="11" fill="${textColor}"${textDecoration}>${this.escape(opcodeText)}</text>`;
      if (typeText) {
        result += `\n${indent}<text x="${typeX}" y="${currentY}" font-family="monospace" font-size="11" fill="#1048af">${this.escape(typeText)}</text>`;
      }

      currentY += this.charHeight;
    }

    return result;
  }

  private findInstructionRows(node: SVGTextNode): Array<{num?: SVGTextNode, opcode?: SVGTextNode, type?: SVGTextNode, rowNode: SVGTextNode}> {
    const rows: Array<{num?: SVGTextNode, opcode?: SVGTextNode, type?: SVGTextNode, rowNode: SVGTextNode}> = [];

    const findRows = (n: SVGTextNode) => {
      if (n.classList.has('ig-ins')) {
        const row: {num?: SVGTextNode, opcode?: SVGTextNode, type?: SVGTextNode, rowNode: SVGTextNode} = {
          rowNode: n
        };

        for (const child of n.children) {
          if (child.classList.has('ig-ins-num')) {
            row.num = child;
          } else if (child.classList.has('ig-ins-type')) {
            row.type = child;
          } else if (!child.classList.has('ig-ins-num') && !child.classList.has('ig-ins-type')) {
            row.opcode = child;
          }
        }

        rows.push(row);
      }

      for (const child of n.children) {
        findRows(child);
      }
    };

    findRows(node);
    return rows;
  }

  private escape(str: string): string {
    return str.replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }
}
