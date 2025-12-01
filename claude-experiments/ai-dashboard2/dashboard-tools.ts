import * as fs from 'fs';
import { tool, createSdkMcpServer } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';
import type { Dashboard } from './src/types';

interface DashboardContext {
  config: Dashboard;
  filePath: string;
}

interface WatchedPathEntry {
  dashboard: Dashboard;
  lastWriteTime: number | null;
  watcher?: fs.FSWatcher | fs.StatWatcher;
  projectId?: string;
}

interface UpdateResult {
  error?: string;
  message?: string;
  success?: boolean;
}

type BroadcastCallback = () => void;
type PermissionMode = 'plan' | 'bypassPermissions';

/**
 * Generates dashboard-specific MCP tools based on the dashboard configuration
 */
export function createDashboardTools(
  dashboardContext: DashboardContext,
  watchedPaths: Map<string, WatchedPathEntry>,
  broadcastCallback: BroadcastCallback,
  permissionMode: PermissionMode = 'bypassPermissions'
) {
  const { config, filePath } = dashboardContext;
  const tools: Array<ReturnType<typeof tool<any>>> = [];

  // Helper function to safely update dashboard JSON
  function updateDashboard(updater: (dashboard: Dashboard) => UpdateResult): UpdateResult {
    try {
      const entry = watchedPaths.get(filePath);
      if (!entry) {
        return { error: 'Dashboard not found in watched paths' };
      }

      // Apply the update
      const result = updater(entry.dashboard);
      if (result.error) {
        return result;
      }

      // Update timestamp to prevent reload loop
      entry.lastWriteTime = Date.now();

      // Write to file
      fs.writeFileSync(filePath, JSON.stringify(entry.dashboard, null, 2), 'utf8');

      // Manually broadcast the update to the UI since file watcher will ignore it
      if (broadcastCallback) {
        broadcastCallback();
      }

      return { success: true, ...result };
    } catch (error: any) {
      return { error: error.message };
    }
  }

  // Generate widget-specific tools based on widget type
  config.widgets?.forEach(widget => {
    const widgetId = widget.id;
    const widgetType = widget.type;
    const widgetLabel = Array.isArray(widget.label) ? widget.label.join(' - ') : widget.label || widgetId;

    // Skip read-only widgets
    if (widget.readOnly || widget.derived || widgetType === 'claude-todo-list') {
      console.log(`[Dashboard Tools] Skipping read-only/derived widget: ${widgetId} (${widgetType})`);
      return;
    }

    // Tool for stat widgets - quick value updates
    if (widgetType === 'stat') {
      tools.push(
        tool(
          `update_stat_${widgetId}`,
          `Update the value of the "${widgetLabel}" stat widget`,
          {
            value: z.string().describe('The new value to display (e.g., "99%", "1.2s", "Active")')
          },
          async ({ value }: { value: string }) => {
            const result = updateDashboard((dashboard) => {
              const targetWidget = dashboard.widgets.find(w => w.id === widgetId);
              if (!targetWidget) {
                return { error: `Widget ${widgetId} not found` };
              }
              if ('value' in targetWidget) {
                targetWidget.value = value;
              }
              return { message: `Updated ${widgetLabel} to: ${value}` };
            });

            return {
              content: [
                {
                  type: 'text',
                  text: result.error || result.message
                }
              ],
              isError: !!result.error
            };
          }
        )
      );
    }

    // Tool for barChart widgets - update data arrays
    if (widgetType === 'bar-chart') {
      tools.push(
        tool(
          `update_chart_${widgetId}`,
          `Update the data for the "${widgetLabel}" chart widget`,
          {
            data: z.array(z.number()).describe('Array of numbers representing chart data points')
          },
          async ({ data }: { data: number[] }) => {
            const result = updateDashboard((dashboard) => {
              const targetWidget = dashboard.widgets.find(w => w.id === widgetId);
              if (!targetWidget) {
                return { error: `Widget ${widgetId} not found` };
              }
              if ('data' in targetWidget) {
                targetWidget.data = data;
              }
              return { message: `Updated ${widgetLabel} with ${data.length} data points` };
            });

            return {
              content: [
                {
                  type: 'text',
                  text: result.error || result.message
                }
              ],
              isError: !!result.error
            };
          }
        )
      );
    }

    // Tool for progress widgets
    if (widgetType === 'progress') {
      tools.push(
        tool(
          `update_progress_${widgetId}`,
          `Update the progress value for the "${widgetLabel}" widget`,
          {
            value: z.number().min(0).max(100).describe('Progress percentage (0-100)'),
            text: z.string().optional().describe('Optional text label to display')
          },
          async ({ value, text }: { value: number; text?: string }) => {
            const result = updateDashboard((dashboard) => {
              const targetWidget = dashboard.widgets.find(w => w.id === widgetId);
              if (!targetWidget) {
                return { error: `Widget ${widgetId} not found` };
              }
              if ('value' in targetWidget) {
                targetWidget.value = value;
              }
              if (text !== undefined && 'text' in targetWidget) {
                targetWidget.text = text;
              }
              return { message: `Updated ${widgetLabel} to ${value}%` };
            });

            return {
              content: [
                {
                  type: 'text',
                  text: result.error || result.message
                }
              ],
              isError: !!result.error
            };
          }
        )
      );
    }

    // Tool for todoList widgets - add items
    if (widgetType === 'todo-list') {
      tools.push(
        tool(
          `add_todo_${widgetId}`,
          `Add a todo item to the "${widgetLabel}" list`,
          {
            text: z.string().describe('The todo item text'),
            done: z.boolean().optional().default(false).describe('Whether the todo is completed')
          },
          async ({ text, done }: { text: string; done?: boolean }) => {
            const result = updateDashboard((dashboard) => {
              const targetWidget = dashboard.widgets.find(w => w.id === widgetId);
              if (!targetWidget) {
                return { error: `Widget ${widgetId} not found` };
              }
              if (targetWidget.type !== 'todo-list') {
                return { error: `Widget ${widgetId} is not a todo-list widget` };
              }
              if (!targetWidget.items) {
                targetWidget.items = [];
              }
              targetWidget.items.push({ text, done: done || false });
              return { message: `Added todo: "${text}" to ${widgetLabel}` };
            });

            return {
              content: [
                {
                  type: 'text',
                  text: result.error || result.message
                }
              ],
              isError: !!result.error
            };
          }
        )
      );

      tools.push(
        tool(
          `toggle_todo_${widgetId}`,
          `Toggle the completion status of a todo item in "${widgetLabel}"`,
          {
            index: z.number().int().min(0).describe('Index of the todo item to toggle (0-based)')
          },
          async ({ index }: { index: number }) => {
            const result = updateDashboard((dashboard) => {
              const targetWidget = dashboard.widgets.find(w => w.id === widgetId);
              if (!targetWidget) {
                return { error: `Widget ${widgetId} not found` };
              }
              if (targetWidget.type !== 'todo-list') {
                return { error: `Widget ${widgetId} is not a todo-list widget` };
              }
              if (!targetWidget.items || targetWidget.items.length === 0) {
                return { error: `Widget ${widgetId} has no items` };
              }
              if (index >= targetWidget.items.length) {
                return { error: `Todo index ${index} out of range (max: ${targetWidget.items.length - 1})` };
              }
              const item = targetWidget.items[index];
              item.done = !item.done;
              const status = item.done ? 'completed' : 'incomplete';
              return { message: `Marked todo "${item.text}" as ${status}` };
            });

            return {
              content: [
                {
                  type: 'text',
                  text: result.error || result.message
                }
              ],
              isError: !!result.error
            };
          }
        )
      );
    }

    // Tool for keyValue widgets - update key-value pairs
    if (widgetType === 'key-value') {
      tools.push(
        tool(
          `update_keyvalue_${widgetId}`,
          `Update or add a key-value pair in "${widgetLabel}"`,
          {
            key: z.string().describe('The key to update or add'),
            value: z.string().describe('The value to set')
          },
          async ({ key, value }: { key: string; value: string }) => {
            const result = updateDashboard((dashboard) => {
              const targetWidget = dashboard.widgets.find(w => w.id === widgetId);
              if (!targetWidget) {
                return { error: `Widget ${widgetId} not found` };
              }
              if (targetWidget.type !== 'key-value') {
                return { error: `Widget ${widgetId} is not a key-value widget` };
              }
              if (!targetWidget.items) {
                targetWidget.items = [];
              }
              const existing = targetWidget.items.find(item => item.key === key);
              if (existing) {
                existing.value = value;
                return { message: `Updated ${key} = ${value} in ${widgetLabel}` };
              } else {
                targetWidget.items.push({ key, value });
                return { message: `Added ${key} = ${value} to ${widgetLabel}` };
              }
            });

            return {
              content: [
                {
                  type: 'text',
                  text: result.error || result.message
                }
              ],
              isError: !!result.error
            };
          }
        )
      );
    }
  });

  // Create the MCP server with these tools
  return createSdkMcpServer({
    name: `dashboard-${config.id}`,
    version: '1.0.0',
    tools
  });
}
