/**
 * InlineResultWidget
 *
 * A CodeMirror 6 widget that displays evaluation results inline after code.
 * Supports loading, success, and error states with appropriate styling.
 */

import { WidgetType, EditorView, Decoration } from '@codemirror/view';
import { StateField, StateEffect, Range } from '@codemirror/state';
import type { EvaluationResult } from '../../types/ipc';

/**
 * State for a single inline result
 */
export interface InlineResult {
  line: number;
  result: EvaluationResult | null;
  loading: boolean;
}

/**
 * Effect to set a result for a line
 */
export const setResultEffect = StateEffect.define<InlineResult>();

/**
 * Effect to clear all results
 */
export const clearResultsEffect = StateEffect.define<void>();

/**
 * Effect to clear result for a specific line
 */
export const clearLineResultEffect = StateEffect.define<number>();

/**
 * Widget class for rendering inline results
 */
class InlineResultWidgetType extends WidgetType {
  constructor(
    readonly result: EvaluationResult | null,
    readonly loading: boolean
  ) {
    super();
  }

  eq(other: InlineResultWidgetType): boolean {
    if (this.loading !== other.loading) return false;
    if (this.result === null && other.result === null) return true;
    if (this.result === null || other.result === null) return false;
    return (
      this.result.displayValue === other.result.displayValue &&
      this.result.success === other.result.success
    );
  }

  toDOM(): HTMLElement {
    const wrapper = document.createElement('span');
    wrapper.className = 'inline-result-wrapper';

    const span = document.createElement('span');
    span.className = 'inline-result';

    if (this.loading) {
      span.classList.add('loading');
      span.textContent = '...';
    } else if (this.result) {
      if (this.result.success) {
        span.classList.add('success');
        // Show type and value
        const typeSpan = document.createElement('span');
        typeSpan.className = 'inline-result-type';
        typeSpan.textContent = this.result.type;

        const valueSpan = document.createElement('span');
        valueSpan.className = 'inline-result-value';
        valueSpan.textContent = this.result.displayValue;

        span.appendChild(typeSpan);
        span.appendChild(document.createTextNode(' '));
        span.appendChild(valueSpan);

        // Add execution time on hover
        span.title = `${this.result.executionTimeMs.toFixed(2)}ms`;
      } else {
        span.classList.add('error');
        span.textContent = this.result.error || 'Error';
        span.title = this.result.error || 'Evaluation error';
      }
    }

    wrapper.appendChild(span);
    return wrapper;
  }

  ignoreEvent(): boolean {
    return false;
  }
}

/**
 * State field that tracks inline results per line
 */
export const inlineResultsField = StateField.define<Map<number, InlineResult>>({
  create() {
    return new Map();
  },

  update(results, tr) {
    let newResults = results;

    for (const effect of tr.effects) {
      if (effect.is(setResultEffect)) {
        newResults = new Map(newResults);
        newResults.set(effect.value.line, effect.value);
      } else if (effect.is(clearResultsEffect)) {
        newResults = new Map();
      } else if (effect.is(clearLineResultEffect)) {
        newResults = new Map(newResults);
        newResults.delete(effect.value);
      }
    }

    // If lines changed, we might need to adjust line numbers
    if (tr.docChanged && newResults.size > 0) {
      const updated = new Map<number, InlineResult>();
      newResults.forEach((result, line) => {
        // Find the new position of this line
        const pos = tr.changes.mapPos(tr.startState.doc.line(line).from);
        const newLine = tr.state.doc.lineAt(pos).number;
        updated.set(newLine, { ...result, line: newLine });
      });
      newResults = updated;
    }

    return newResults;
  },
});

/**
 * Decoration provider that renders InlineResultWidget for each result
 */
export const inlineResultsDecorations = EditorView.decorations.compute(
  [inlineResultsField],
  (state) => {
    const results = state.field(inlineResultsField);
    const decorations: Range<Decoration>[] = [];

    results.forEach((result) => {
      const line = state.doc.line(result.line);
      const widget = Decoration.widget({
        widget: new InlineResultWidgetType(result.result, result.loading),
        side: 1, // After the line content
      });
      decorations.push(widget.range(line.to));
    });

    return Decoration.set(decorations.sort((a, b) => a.from - b.from));
  }
);

/**
 * Helper function to set a loading state for a line
 */
export function setLoading(view: EditorView, line: number): void {
  view.dispatch({
    effects: setResultEffect.of({ line, result: null, loading: true }),
  });
}

/**
 * Helper function to set a result for a line
 */
export function setResult(view: EditorView, line: number, result: EvaluationResult): void {
  view.dispatch({
    effects: setResultEffect.of({ line, result, loading: false }),
  });
}

/**
 * Helper function to clear all results
 */
export function clearAllResults(view: EditorView): void {
  view.dispatch({
    effects: clearResultsEffect.of(undefined),
  });
}

/**
 * Helper function to clear a specific line's result
 */
export function clearLineResult(view: EditorView, line: number): void {
  view.dispatch({
    effects: clearLineResultEffect.of(line),
  });
}

/**
 * Get the extension bundle for inline results
 */
export function inlineResultsExtension() {
  return [inlineResultsField, inlineResultsDecorations];
}
