interface GeneratedStyle {
  id: string;
  prompt: string;
  timestamp: number;
  css: string;
  svgDefs?: string;
  metadata?: any;
}

interface StyleRequest {
  prompt: string;
  context?: {
    currentTheme?: GeneratedStyle;
    components?: string[];
  };
}

type ChunkCallback = (chunk: string) => void;
type CompleteCallback = (style: GeneratedStyle) => void;
type ErrorCallback = (error: Error) => void;

export class StyleAgent {
  private currentStyle: GeneratedStyle | null = null;
  private query: any = null;

  async initialize() {
    // Use eval to force a true dynamic import (bypasses TypeScript transpilation)
    const importFunc = new Function('specifier', 'return import(specifier)');
    const sdk = await importFunc('@anthropic-ai/claude-agent-sdk');
    this.query = sdk.query;
    console.log('[StyleAgent] Initialized (using Claude subscription - no API key)');
  }

  async generateStyleStreaming(
    request: StyleRequest,
    onChunk: ChunkCallback,
    onComplete: CompleteCallback,
    onError: ErrorCallback
  ): Promise<void> {
    if (!this.query) {
      onError(new Error('StyleAgent not initialized. Call initialize() first.'));
      return;
    }

    console.log('[StyleAgent] Generating style (streaming) for:', request.prompt);

    const stylePrompt = `Generate CSS for a dashboard with this aesthetic: "${request.prompt}"

${request.context?.currentTheme ? `Current theme: ${request.context.currentTheme.prompt}\n` : ''}

Style these selectors:
- .dashboard-root, .sidebar, .main-content
- .project-selector, .project-item
- .widget-container, .widget-header, .widget-title, .widget-content
- .metric-display, .metric-label, .metric-value, .metric-unit
- .data-series, .data-point
- .status-item, .status-label, .status-value

Output ONLY valid CSS. No markdown, no explanations. Start with :root for variables.`;

    let generatedCSS = '';

    try {
      const queryResult = this.query({
        prompt: stylePrompt,
        options: {
          allowedTools: [],
          systemPrompt: 'Output only raw CSS code. No markdown code blocks, no explanations, no backticks. Just pure CSS starting with :root',
        }
      });

      for await (const message of queryResult) {
        // Handle streaming text deltas
        if (message.type === 'stream_event' && message.event?.type === 'content_block_delta') {
          const text = message.event.delta?.text;
          if (text) {
            generatedCSS += text;
            onChunk(text);
          }
        }
        // Handle complete assistant messages (fallback)
        else if (message.type === 'assistant' && message.message?.content) {
          for (const block of message.message.content) {
            if (block.type === 'text') {
              generatedCSS += block.text;
              onChunk(block.text);
            }
          }
        }
      }

      // Clean up any markdown that snuck through
      generatedCSS = generatedCSS.replace(/```css\n?/g, '').replace(/```\n?/g, '').trim();

      const style: GeneratedStyle = {
        id: `style-${Date.now()}`,
        prompt: request.prompt,
        timestamp: Date.now(),
        css: generatedCSS,
        svgDefs: '',
        metadata: {
          mood: request.prompt
        }
      };

      this.currentStyle = style;
      console.log('[StyleAgent] Generated style:', style.id);

      onComplete(style);
    } catch (error) {
      onError(error as Error);
    }
  }

  getCurrentStyle(): GeneratedStyle | null {
    return this.currentStyle;
  }
}
