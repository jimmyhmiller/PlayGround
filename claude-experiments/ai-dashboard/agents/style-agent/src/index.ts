import { query } from '@anthropic-ai/claude-agent-sdk';
import { DashboardAgent } from '@ai-dashboard/agent-sdk';

async function generateStyle(prompt: string, currentTheme?: any): Promise<any> {
  console.log('[StyleAgent] Generating CSS for:', prompt);

  // Use Claude Agent SDK to generate CSS
  // This uses your Claude subscription, NOT an API key
  const stylePrompt = `You are a CSS design expert. Generate a complete, cohesive stylesheet for a dashboard application.

Aesthetic request: "${prompt}"

${currentTheme ? `Current theme to iterate on: ${currentTheme.prompt}\n` : ''}

Generate CSS for these semantic components:
- .dashboard-root (main container)
- .sidebar (navigation sidebar)
- .main-content (main content area)
- .project-selector, .project-item (project list)
- .widget-container, .widget-header, .widget-title, .widget-content (widgets)
- .metric-display, .metric-label, .metric-value, .metric-unit (metrics)
- .data-series, .data-point (data visualization, .data-point has --value CSS variable)
- .status-item, .status-label, .status-value (status rows)

Requirements:
- Use modern CSS (grid, flexbox, custom properties, gradients, animations)
- Make it cohesive and beautiful
- Ensure text is readable
- Be creative with the aesthetic

Return ONLY the CSS code, nothing else. Start with .dashboard-root and include all component styles.`;

  let generatedCSS = '';

  const queryResult = query({
    prompt: stylePrompt,
    options: {
      allowedTools: [], // No tools needed, just want Claude's response
      systemPrompt: 'You are a CSS design expert. Generate only CSS code, no explanations.',
    }
  });

  for await (const message of queryResult) {
    if (message.type === 'assistant') {
      // Extract text from assistant message
      const textBlocks = message.message.content.filter((block: any) => block.type === 'text');
      for (const block of textBlocks) {
        generatedCSS += block.text;
      }
    }
  }

  // Clean up the CSS (remove markdown code blocks if present)
  generatedCSS = generatedCSS.replace(/```css\n?/g, '').replace(/```\n?/g, '').trim();

  return {
    id: `style-${Date.now()}`,
    prompt,
    timestamp: Date.now(),
    css: generatedCSS,
    svgDefs: '', // Could be enhanced to extract SVG patterns from CSS
    metadata: {
      mood: prompt
    }
  };
}

async function main() {
  console.log('[StyleAgent] Starting...');
  console.log('[StyleAgent] Using Claude subscription (no API key needed)');

  // Connect to dashboard (port can be configured via env var)
  const dashboardPort = process.env.DASHBOARD_PORT || '3000';
  const dashboardAgent = new DashboardAgent({
    id: 'style-agent',
    name: 'Style Agent',
    connection: {
      type: 'websocket',
      url: `ws://localhost:${dashboardPort}`,
    },
  });

  await dashboardAgent.connect();
  console.log('[StyleAgent] Connected to dashboard');

  // Handle style generation requests
  dashboardAgent.on('query', async (queryMsg: any, respond: (response: any) => void) => {
    if (queryMsg.type === 'generate-style') {
      try {
        const style = await generateStyle(
          queryMsg.request.prompt,
          queryMsg.request.context?.currentTheme
        );

        console.log('[StyleAgent] Generated style:', style.id);
        respond(style);
      } catch (error) {
        console.error('[StyleAgent] Error generating style:', error);
        respond({ error: (error as Error).message });
      }
    }
  });

  console.log('[StyleAgent] Ready to generate styles!');
}

main().catch((error) => {
  console.error('[StyleAgent] Fatal error:', error);
  process.exit(1);
});
