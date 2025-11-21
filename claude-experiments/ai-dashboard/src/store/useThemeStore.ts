// Theme state management

import { create } from 'zustand';
import { GeneratedStyle } from '../types/theme';

interface ThemeStore {
  currentStyle: GeneratedStyle | null;
  streamingCSS: string;
  projectStyles: Record<string, GeneratedStyle>; // projectId -> style
  currentProjectId: string | null;
  history: GeneratedStyle[];
  isGenerating: boolean;

  setProjectStyle: (projectId: string, style: GeneratedStyle) => void;
  switchProject: (projectId: string) => void;
  setStreamingStyle: (chunk: string) => void;
  setGenerating: (generating: boolean) => void;
  clearHistory: () => void;
}

export const useThemeStore = create<ThemeStore>((set, get) => ({
  currentStyle: null,
  streamingCSS: '',
  projectStyles: {},
  currentProjectId: null,
  history: [],
  isGenerating: false,

  setProjectStyle: async (projectId, style) => {
    // Save to backend
    await window.electronAPI.setProjectStyle(projectId, style);

    set((state) => {
      const newProjectStyles = { ...state.projectStyles, [projectId]: style };
      console.log(`[Store] Saved style for project ${projectId}:`, style.prompt);
      console.log('[Store] All project styles:', Object.keys(newProjectStyles));
      return {
        projectStyles: newProjectStyles,
        currentStyle: state.currentProjectId === projectId ? style : state.currentStyle,
        history: [...state.history, style],
        streamingCSS: '', // Reset streaming CSS when complete
      };
    });
  },

  switchProject: async (projectId) => {
    // Load style from backend
    const style = await window.electronAPI.getProjectStyle(projectId);

    set((state) => {
      console.log(`[Store] Switching to project ${projectId}`);
      console.log('[Store] Existing style for project:', style?.prompt || 'none');

      // Update projectStyles cache
      const newProjectStyles = { ...state.projectStyles };
      if (style) {
        newProjectStyles[projectId] = style;
      }

      return {
        currentProjectId: projectId,
        currentStyle: style || null,
        projectStyles: newProjectStyles,
      };
    });
  },

  setStreamingStyle: (chunk) =>
    set((state) => {
      const newCSS = state.streamingCSS + chunk;
      console.log('[Store] Streaming CSS length:', newCSS.length, 'chunk:', chunk.substring(0, 20));
      // Create a temporary style object for streaming
      const streamingStyle: GeneratedStyle = {
        id: 'streaming',
        prompt: 'Generating...',
        timestamp: Date.now(),
        css: newCSS,
        svgDefs: '',
      };
      return {
        streamingCSS: newCSS,
        currentStyle: streamingStyle,
      };
    }),

  setGenerating: (generating) =>
    set((state) => ({
      isGenerating: generating,
      streamingCSS: generating ? '' : state.streamingCSS, // Reset on start
    })),

  clearHistory: () => set({ history: [] }),
}));
