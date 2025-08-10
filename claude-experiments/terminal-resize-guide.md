# Terminal Box Resize Implementation Guide

## Overview

This document details the techniques used in bcherny's Ink fork to achieve proper terminal box resizing without visual corruption or flicker. These methods can be adapted for any terminal UI library.

## The Problem

Standard terminal UI libraries (including vanilla Ink) suffer from visual corruption when the terminal is resized, particularly:

- Box borders break or disappear
- Content gets misaligned
- Partial redraws cause visual artifacts
- Tall content causes rendering issues

## The Solution: Smart Clearing Strategy

The key insight is that **proper terminal resize handling requires aggressive clearing strategies** rather than trying to be clever with incremental updates.

## Core Implementation Techniques

### 1. Smart Terminal Clearing Logic

The most critical improvement is knowing when to clear the entire terminal vs. doing incremental updates:

```typescript
if (
  outputHeight >= this.options.stdout.rows ||
  this.lastOutputHeight >= this.options.stdout.rows
) {
  // Full clear and redraw - prevents corruption
  this.options.stdout.write(
    ansiEscapes.clearTerminal + this.fullStaticOutput + output + '\n'
  );
  this.lastOutput = output;
  this.lastOutputHeight = outputHeight;
  this.log.updateLineCount(output + '\n');
  return;
}
```

**Why this works:** When content is taller than the terminal, partial updates cause corruption. Full clear prevents this.

### 2. Forced Redraw on Resize Events

```typescript
resized = () => {
  this.calculateLayout();
  this.onRender(true); // Force parameter bypasses throttling
};

// In the render function:
if (didResize) {
  this.options.stdout.write(
    ansiEscapes.clearTerminal + this.fullStaticOutput + output + '\n'
  );
  this.lastOutput = output;
  this.lastOutputHeight = outputHeight;
  this.log.updateLineCount(output + '\n');
  return;
}
```

**Why this works:** Forces immediate full redraw instead of trying to incrementally update during resize, which is where most corruption occurs.

### 3. Output Height Tracking & Management

```typescript
// Track both current and previous output heights
this.lastOutputHeight = outputHeight;

// Use in rendering decisions
const needsFullClear = 
  outputHeight >= this.options.stdout.rows ||
  this.lastOutputHeight >= this.options.stdout.rows;
```

**Why this works:** Detects when terminal size changes would cause rendering issues before they occur.

### 4. Throttled Rendering with Critical Event Bypass

```typescript
// Normal rendering is throttled to prevent flicker
this.throttledLog = throttle(this.log, 32, {
  leading: true,
  trailing: true,
});

// But resize and overflow events bypass throttling
this.rootNode.onRender = options.debug
  ? this.onRender
  : throttle(this.onRender, 32, {
      leading: true,
      trailing: true,
    });
    
// Immediate render for critical events
this.rootNode.onImmediateRender = this.onRender; // No throttling
```

**Why this works:** Prevents flicker during normal updates but allows immediate response to critical events like resizes.

### 5. Flicker Detection & Handling

```typescript
if (this.options.onFlicker) {
  this.options.onFlicker(); // Allow app to handle flicker events
}
```

**Why this works:** Gives applications a chance to respond to known problematic situations (e.g., show loading state).

## Implementation Architecture for Non-Ink Libraries

### Core Data Structures

```typescript
interface TerminalState {
  width: number;
  height: number;
  lastOutputHeight: number;
  lastOutput: string;
  staticOutput: string; // Content that doesn't change
}

interface RenderOptions {
  forceRedraw?: boolean;
  isResize?: boolean;
  debug?: boolean;
}
```

### Essential Components

#### 1. Output Buffer Management

```typescript
class TerminalRenderer {
  private currentOutput: string = '';
  private previousOutput: string = '';
  private currentHeight: number = 0;
  private previousHeight: number = 0;
  private staticOutput: string = '';
  
  // Track terminal dimensions
  private terminalWidth: number;
  private terminalHeight: number;
}
```

#### 2. Resize Event Handler

```typescript
private setupResizeHandler() {
  process.stdout.on('resize', () => {
    this.terminalWidth = process.stdout.columns || 80;
    this.terminalHeight = process.stdout.rows || 24;
    this.recalculateLayout();
    this.render({ forceRedraw: true, isResize: true });
  });
}
```

#### 3. Smart Rendering Decision Engine

```typescript
private render(options: RenderOptions = {}) {
  const { forceRedraw = false, isResize = false } = options;
  const content = this.generateOutput();
  const contentHeight = this.calculateHeight(content);
  
  // Decision tree for rendering strategy
  if (this.shouldFullClear(contentHeight, forceRedraw, isResize)) {
    this.fullClearAndRedraw(content, contentHeight);
  } else {
    this.incrementalUpdate(content, contentHeight);
  }
}

private shouldFullClear(
  contentHeight: number, 
  forceRedraw: boolean, 
  isResize: boolean
): boolean {
  return (
    forceRedraw ||
    isResize ||
    contentHeight >= this.terminalHeight ||
    this.previousHeight >= this.terminalHeight
  );
}
```

#### 4. Terminal Output Methods

```typescript
private fullClearAndRedraw(content: string, height: number) {
  process.stdout.write(ansiEscapes.clearTerminal + this.staticOutput + content);
  this.updateTracking(content, height);
}

private incrementalUpdate(content: string, height: number) {
  if (content !== this.previousOutput) {
    this.logUpdate(content); // Uses ansiEscapes.eraseLines for efficiency
  }
  this.updateTracking(content, height);
}

private updateTracking(content: string, height: number) {
  this.previousOutput = content;
  this.previousHeight = height;
}
```

### Box Drawing Considerations

When implementing bordered boxes:

1. **Pre-calculate dimensions** before drawing
2. **Use Unicode box characters** consistently
3. **Store box layout separately** from content
4. **Always redraw entire boxes** on resize, not just content

```typescript
interface BoxLayout {
  x: number;
  y: number;
  width: number;
  height: number;
  borderStyle: 'single' | 'double' | 'round';
}

private drawBox(layout: BoxLayout, content: string): string {
  // Calculate box with proper Unicode characters
  // Handle wrapping and truncation within box bounds
  // Return complete box as string for atomic rendering
}
```

## Key ANSI Escape Sequences

```typescript
import ansiEscapes from 'ansi-escapes';

// Essential sequences for proper rendering
const sequences = {
  clearTerminal: ansiEscapes.clearTerminal,     // Full screen clear
  eraseLines: ansiEscapes.eraseLines,           // Clear specific lines
  cursorTo: ansiEscapes.cursorTo,               // Position cursor
  hideCursor: ansiEscapes.cursorHide,           // Hide cursor during render
  showCursor: ansiEscapes.cursorShow,           // Show cursor when done
};
```

## Critical Success Factors

### 1. Always Clear Terminal on Resize
- **Never** try incremental updates during resize
- Always use `ansiEscapes.clearTerminal` + full redraw
- Update all tracking variables after clear

### 2. Track Output Height Accurately
- Include newlines and text wrapping in calculations
- Account for box borders in height calculations
- Use actual terminal dimensions, not assumed values

### 3. Separate Static from Dynamic Content
- Render static content (headers, unchanging boxes) separately
- Only update dynamic portions during normal renders
- Combine both during full redraws

### 4. Use Throttling with Escape Hatches
- Throttle normal rendering to prevent flicker (16-32ms)
- Bypass throttling for critical events (resize, overflow)
- Provide immediate render option for urgent updates

### 5. Test with Edge Cases
- Content taller than terminal height
- Rapid terminal resizing
- Very small terminal dimensions
- Box borders spanning multiple lines

## Common Pitfalls to Avoid

1. **Trying to preserve content during resize** - This causes most corruption
2. **Not tracking previous output height** - Leads to rendering decisions based on incomplete data
3. **Using incremental updates for tall content** - Causes visual artifacts
4. **Forgetting to update tracking variables** - Causes subsequent renders to fail
5. **Not handling static vs dynamic content differently** - Causes unnecessary redraws

## Implementation Checklist

- [ ] Terminal resize event handler implemented
- [ ] Output height tracking in place
- [ ] Smart clearing logic implemented
- [ ] Throttling with bypass mechanisms
- [ ] Static/dynamic content separation
- [ ] Proper ANSI escape sequence usage
- [ ] Edge case testing completed
- [ ] Box drawing handles resize properly

## Conclusion

The key insight from bcherny's Ink fork is that robust terminal UIs require **defensive rendering strategies**. Instead of optimizing for minimal updates, optimize for correctness by detecting problematic conditions and falling back to full redraws.

This approach trades some performance for reliability, but the performance cost is minimal while the reliability improvement is substantial. Modern terminals can handle full screen clears very efficiently, making this a practical solution.

The main failures in standard terminal UIs happen when content is taller than the terminal, when terminal size changes cause text reflow, and when box borders span multiple lines. The solution is to detect these conditions and always fall back to full terminal clearing and redrawing rather than trying to preserve existing content.