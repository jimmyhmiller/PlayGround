import * as fs from 'fs';
import { tool, createSdkMcpServer } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';
import type { Dashboard } from './src/types';
import { validateWidget, formatValidationErrorForAI } from './src/validation';

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
            done: z.boolean().default(false).describe('Whether the todo is completed')
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

  // Add validation tool (available for all dashboards)
  tools.push(
    tool(
      'validate_widget',
      'Validate a widget configuration before adding it to the dashboard. Use this to check if your widget config is correct and get helpful error messages if not.',
      {
        widgetConfig: z.object({}).passthrough().describe('The widget configuration object to validate')
      },
      async ({ widgetConfig }: { widgetConfig: any }) => {
        const result = validateWidget(widgetConfig);
        const formatted = formatValidationErrorForAI(result);

        return {
          content: [
            {
              type: 'text',
              text: formatted
            }
          ],
          isError: !result.valid
        };
      }
    )
  );

  // Generic tool to update any widget property
  tools.push(
    tool(
      'update_widget',
      'Update any property of a widget in the dashboard',
      {
        widgetId: z.string().describe('ID of the widget to update'),
        property: z.string().describe('Property name to update (e.g., "label", "width", "height")'),
        value: z.any().optional().describe('New value for the property')
      },
      async ({ widgetId, property, value }: { widgetId: string; property: string; value?: any }) => {
        const result = updateDashboard((dashboard) => {
          const targetWidget = dashboard.widgets.find(w => w.id === widgetId);
          if (!targetWidget) {
            return { error: `Widget ${widgetId} not found` };
          }
          (targetWidget as any)[property] = value;
          return { message: `Updated widget ${widgetId}: ${property} = ${JSON.stringify(value)}` };
        });

        return {
          content: [
            {
              type: 'text',
              text: result.error || result.message || 'Updated successfully'
            }
          ],
          isError: !!result.error
        };
      }
    )
  );

  // Tool to add a new widget
  tools.push(
    tool(
      'add_widget',
      'Add a new widget to the dashboard',
      {
        type: z.string().describe('Type of widget to add (e.g., "bar-chart", "stat", "progress", "chat", "todo-list", "key-value", etc.)'),
        id: z.string().describe('Unique ID for the widget'),
        x: z.number().describe('X position'),
        y: z.number().describe('Y position'),
        width: z.number().describe('Widget width'),
        height: z.number().describe('Widget height'),
        config: z.object({}).passthrough().optional().describe('Additional widget configuration (label, data, etc.)')
      },
      async ({ type, id, x, y, width, height, config }: {
        type: string;
        id: string;
        x: number;
        y: number;
        width: number;
        height: number;
        config?: Record<string, any>
      }) => {
        const result = updateDashboard((dashboard) => {
          // Check if ID already exists
          if (dashboard.widgets.find(w => w.id === id)) {
            return { error: `Widget with ID ${id} already exists` };
          }

          const newWidget: any = {
            id,
            type,
            x,
            y,
            width,
            height,
            ...config
          };

          dashboard.widgets.push(newWidget);
          return { message: `Added ${type} widget with ID: ${id}` };
        });

        return {
          content: [
            {
              type: 'text',
              text: result.error || result.message || 'Widget added successfully'
            }
          ],
          isError: !!result.error
        };
      }
    )
  );

  // Tool to remove a widget
  tools.push(
    tool(
      'remove_widget',
      'Remove a widget from the dashboard',
      {
        widgetId: z.string().describe('ID of the widget to remove')
      },
      async ({ widgetId }: { widgetId: string }) => {
        const result = updateDashboard((dashboard) => {
          const index = dashboard.widgets.findIndex(w => w.id === widgetId);
          if (index === -1) {
            return { error: `Widget ${widgetId} not found` };
          }
          const removed = dashboard.widgets.splice(index, 1)[0];
          return { message: `Removed ${removed.type} widget: ${widgetId}` };
        });

        return {
          content: [
            {
              type: 'text',
              text: result.error || result.message || 'Widget removed successfully'
            }
          ],
          isError: !!result.error
        };
      }
    )
  );

  // Ask user a question with multiple choice or custom input (available in all modes)
  tools.push(
    tool(
      'AskUserQuestion',
      'Ask the user one or more questions during planning. Can ask a single question or multiple questions at once. All questions are shown together and all answers collected before returning. Use this when you need clarification before proposing a plan.',
      {
        questions: z.union([
          // Single question (backward compatible)
          z.object({
            question: z.string().describe('The question to ask'),
            options: z.array(z.string()).optional().describe('Optional predefined choices'),
            allowMultiple: z.boolean().default(false).describe('Allow multiple selections'),
            allowCustom: z.boolean().default(true).describe('Show "type my own answer" option')
          }),
          // Multiple questions
          z.array(z.object({
            id: z.string().describe('Unique ID for this question (e.g., "widget-type", "position")'),
            question: z.string().describe('The question to ask'),
            options: z.array(z.string()).optional().describe('Optional predefined choices'),
            allowMultiple: z.boolean().default(false).describe('Allow multiple selections'),
            allowCustom: z.boolean().default(true).describe('Show "type my own answer" option')
          }))
        ]).describe('Either a single question object or an array of question objects to ask together')
      },
      async ({ questions }: { questions: any }) => {
        try {
          const { ipcMain, BrowserWindow } = await import('electron');
          const mainWindow = BrowserWindow.getAllWindows()[0];

          if (!mainWindow) {
            return {
              content: [{
                type: 'text',
                text: 'Error: No window available to display question'
              }],
              isError: true
            };
          }

          // Normalize to array format
          const questionArray = Array.isArray(questions)
            ? questions
            : [{ id: 'single', ...questions }];

          // Send the questions to the renderer and wait for response
          return new Promise((resolve) => {
            // Set up one-time listener for the answer
            const answerId = `question-${Date.now()}`;

            ipcMain.once(`question-answer-${answerId}`, (_event: any, answers: any) => {
              // If it was a single question, return just the answer value
              const result = Array.isArray(questions)
                ? answers
                : answers.single;

              resolve({
                content: [{
                  type: 'text',
                  text: `User answered: ${JSON.stringify(result)}`
                }],
                isError: false
              });
            });

            // Send questions to renderer
            mainWindow.webContents.send('ask-user-question', {
              id: answerId,
              questions: questionArray,
              isMultiple: Array.isArray(questions)
            });
          });
        } catch (error: any) {
          return {
            content: [{
              type: 'text',
              text: `Error asking question: ${error.message}`
            }],
            isError: true
          };
        }
      }
    )
  );

  // Create the MCP server with these tools
  return createSdkMcpServer({
    name: `dashboard-${config.id}`,
    version: '1.0.0',
    tools
  });
}
