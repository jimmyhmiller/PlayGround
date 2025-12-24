/**
 * Register all available component types
 */

import { registerComponent } from './ComponentRegistry';
import CodeMirrorEditor from './CodeMirrorEditor';
import GitDiffViewer from './GitDiffViewer';

export function registerAllComponents(): void {
  registerComponent('codemirror', CodeMirrorEditor, {
    subscribePattern: 'file.**',
    initialContent: '// Loading...',
  });

  registerComponent('git-diff', GitDiffViewer, {
    subscribePattern: 'git.**',
  });

  console.log('[components] Registered: codemirror, git-diff');
}
