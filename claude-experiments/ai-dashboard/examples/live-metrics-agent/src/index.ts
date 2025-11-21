import { DashboardAgent } from '@ai-dashboard/agent-sdk';

async function main() {
  // Create agent instance
  const agent = new DashboardAgent({
    id: 'live-metrics-agent',
    name: 'Live Metrics Agent',
    connection: {
      type: 'websocket',
      url: 'ws://localhost:3000',
    },
  });

  // Connect to dashboard
  try {
    await agent.connect();
    console.log('[LiveMetrics] Connected to dashboard!');
  } catch (error) {
    console.error('[LiveMetrics] Failed to connect:', error);
    process.exit(1);
  }

  // Register a custom component for displaying network activity
  await agent.registerComponent({
    id: 'network-activity',
    name: 'Network Activity Chart',
    semantic: 'data-series',
    code: `
      export default function NetworkActivity({ data, theme }) {
        if (!data || !data.points) return null;

        return (
          <div className="network-activity">
            <div className="data-series">
              {data.points.map((value, index) => (
                <div
                  key={index}
                  className="data-point network-point"
                  style={{
                    '--value': value,
                    height: value + '%'
                  }}
                  data-label={data.labels?.[index]}
                />
              ))}
            </div>
          </div>
        );
      }
    `,
    themeContract: {
      uses: ['colors.accent', 'fonts.body'],
      providesClasses: ['network-activity', 'network-point'],
    },
  });

  // Register data source
  await agent.registerDataSource({
    id: 'network-metrics',
    name: 'Network Metrics',
    provider: {
      type: 'polling',
      interval: 2000,
    },
    compatibleWith: ['data-series', 'network-activity'],
  });

  console.log('[LiveMetrics] Registered component and data source');

  // Simulate live metrics
  let iteration = 0;
  setInterval(async () => {
    iteration++;

    // Generate random network activity data
    const points = Array.from({ length: 20 }, () => Math.random() * 100);

    const data = {
      points,
      labels: points.map((_, i) => `T-${20 - i}s`),
      timestamp: Date.now(),
      iteration,
    };

    // Send data update
    await agent.updateData('network-metrics', data);

    if (iteration % 10 === 0) {
      console.log(`[LiveMetrics] Sent update #${iteration}`);
    }
  }, 2000);

  // Also send periodic metrics
  setInterval(async () => {
    const cpuUsage = Math.random() * 100;
    const memoryUsage = 60 + Math.random() * 30;

    await agent.updateData('system-metrics', {
      cpu: cpuUsage.toFixed(1),
      memory: memoryUsage.toFixed(1),
      timestamp: Date.now(),
    });
  }, 5000);

  // Handle theme changes
  agent.on('theme-changed', (theme) => {
    console.log('[LiveMetrics] Theme changed:', theme);
  });

  // Handle queries from dashboard
  agent.on('query', (query, respond) => {
    console.log('[LiveMetrics] Received query:', query);

    if (query.type === 'get-status') {
      respond({
        status: 'healthy',
        uptime: process.uptime(),
        iterations: iteration,
      });
    }
  });

  // Handle disconnection
  agent.on('disconnected', () => {
    console.log('[LiveMetrics] Disconnected from dashboard. Exiting...');
    process.exit(0);
  });

  console.log('[LiveMetrics] Agent is running. Press Ctrl+C to exit.');
}

main().catch((error) => {
  console.error('[LiveMetrics] Fatal error:', error);
  process.exit(1);
});
