const fs = require('fs');
const { tool, createSdkMcpServer } = require('@anthropic-ai/claude-agent-sdk');
const { z } = require('zod');

/**
 * Generates dashboard-specific MCP tools based on the dashboard configuration
 * @param {Object} dashboardContext - Contains dashboard config and file path
 * @param {Map} watchedPaths - Map of watched dashboard files for coordination
 * @param {Function} broadcastCallback - Function to call to broadcast dashboard updates to UI
 * @param {string} permissionMode - The permission mode ('plan' or 'bypassPermissions')
 * @returns {Object} MCP server configuration with custom tools
 */
function createDashboardTools(dashboardContext, watchedPaths, broadcastCallback, permissionMode = 'bypassPermissions') {
  const { config, filePath } = dashboardContext;
  const tools = [];

  // Helper function to safely update dashboard JSON
  function updateDashboard(updater) {
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
    } catch (error) {
      return { error: error.message };
    }
  }

  // Generate widget-specific tools based on widget type
  config.widgets.forEach(widget => {
    const widgetId = widget.id;
    const widgetType = widget.type;
    const widgetLabel = Array.isArray(widget.label) ? widget.label.join(' - ') : widget.label;

    // Skip read-only widgets (derived from SDK, not manually editable)
    // claudeTodos is always read-only as it's derived from TodoWrite tool usage
    if (widget.readOnly || widget.derived || widgetType === 'claudeTodos') {
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
          async ({ value }) => {
            const result = updateDashboard((dashboard) => {
              const targetWidget = dashboard.widgets.find(w => w.id === widgetId);
              if (!targetWidget) {
                return { error: `Widget ${widgetId} not found` };
              }
              targetWidget.value = value;
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
    if (widgetType === 'barChart') {
      tools.push(
        tool(
          `update_chart_${widgetId}`,
          `Update the data for the "${widgetLabel}" chart widget`,
          {
            data: z.array(z.number()).describe('Array of numbers representing chart data points')
          },
          async ({ data }) => {
            const result = updateDashboard((dashboard) => {
              const targetWidget = dashboard.widgets.find(w => w.id === widgetId);
              if (!targetWidget) {
                return { error: `Widget ${widgetId} not found` };
              }
              targetWidget.data = data;
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
          async ({ value, text }) => {
            const result = updateDashboard((dashboard) => {
              const targetWidget = dashboard.widgets.find(w => w.id === widgetId);
              if (!targetWidget) {
                return { error: `Widget ${widgetId} not found` };
              }
              targetWidget.value = value;
              if (text !== undefined) {
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
    if (widgetType === 'todoList') {
      tools.push(
        tool(
          `add_todo_${widgetId}`,
          `Add a todo item to the "${widgetLabel}" list`,
          {
            text: z.string().describe('The todo item text'),
            done: z.boolean().optional().default(false).describe('Whether the todo is completed')
          },
          async ({ text, done }) => {
            const result = updateDashboard((dashboard) => {
              const targetWidget = dashboard.widgets.find(w => w.id === widgetId);
              if (!targetWidget) {
                return { error: `Widget ${widgetId} not found` };
              }
              if (!targetWidget.items) {
                targetWidget.items = [];
              }
              targetWidget.items.push({ text, done });
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
          async ({ index }) => {
            const result = updateDashboard((dashboard) => {
              const targetWidget = dashboard.widgets.find(w => w.id === widgetId);
              if (!targetWidget || !targetWidget.items) {
                return { error: `Widget ${widgetId} not found or has no items` };
              }
              if (index >= targetWidget.items.length) {
                return { error: `Todo index ${index} out of range (max: ${targetWidget.items.length - 1})` };
              }
              targetWidget.items[index].done = !targetWidget.items[index].done;
              const status = targetWidget.items[index].done ? 'completed' : 'incomplete';
              return { message: `Marked todo "${targetWidget.items[index].text}" as ${status}` };
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
    if (widgetType === 'keyValue') {
      tools.push(
        tool(
          `update_keyvalue_${widgetId}`,
          `Update or add a key-value pair in "${widgetLabel}"`,
          {
            key: z.string().describe('The key to update or add'),
            value: z.string().describe('The value to set')
          },
          async ({ key, value }) => {
            const result = updateDashboard((dashboard) => {
              const targetWidget = dashboard.widgets.find(w => w.id === widgetId);
              if (!targetWidget) {
                return { error: `Widget ${widgetId} not found` };
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

    // Tool for regenerating widget data (if widget has regenerateCommand or regenerateScript)
    if (widget.regenerateCommand || widget.regenerateScript) {
      tools.push(
        tool(
          `regenerate_${widgetId}`,
          `Regenerate data for "${widgetLabel}" by running its configured command`,
          {},
          async () => {
            try {
              // Call the IPC handler via electron (need to access ipcRenderer from renderer process)
              // Since MCP tools run in main process, we need to use the IPC mechanism directly
              const { ipcMain } = require('electron');

              // Get the regenerate-widget handler
              const handlers = ipcMain._events['regenerate-widget'];
              if (!handlers || handlers.length === 0) {
                return {
                  content: [{
                    type: 'text',
                    text: 'Regenerate handler not available'
                  }],
                  isError: true
                };
              }

              // Call the handler
              const handler = Array.isArray(handlers) ? handlers[0] : handlers;
              const result = await handler(null, {
                dashboardId: config.id,
                widgetId: widgetId
              });

              return {
                content: [
                  {
                    type: 'text',
                    text: result.success
                      ? `Successfully regenerated "${widgetLabel}"`
                      : `Failed to regenerate "${widgetLabel}": ${result.error}`
                  }
                ],
                isError: !result.success
              };
            } catch (error) {
              return {
                content: [
                  {
                    type: 'text',
                    text: `Error regenerating "${widgetLabel}": ${error.message}`
                  }
                ],
                isError: true
              };
            }
          }
        )
      );
    }
  });

  // Generic tool to update any widget property
  tools.push(
    tool(
      'update_widget',
      'Update any property of a widget in the dashboard',
      {
        widgetId: z.string().describe('ID of the widget to update'),
        property: z.string().describe('Property name to update (e.g., "label", "width", "height")'),
        value: z.any().describe('New value for the property')
      },
      async ({ widgetId, property, value }) => {
        const result = updateDashboard((dashboard) => {
          const targetWidget = dashboard.widgets.find(w => w.id === widgetId);
          if (!targetWidget) {
            return { error: `Widget ${widgetId} not found` };
          }
          targetWidget[property] = value;
          return { message: `Updated widget ${widgetId}: ${property} = ${JSON.stringify(value)}` };
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

  // Tool to add a new widget
  tools.push(
    tool(
      'add_widget',
      'Add a new widget to the dashboard',
      {
        type: z.enum(['barChart', 'stat', 'progress', 'chat', 'diffList', 'fileList', 'todoList', 'keyValue', 'layoutSettings']).describe('Type of widget to add'),
        id: z.string().describe('Unique ID for the widget'),
        x: z.number().describe('X position'),
        y: z.number().describe('Y position'),
        width: z.number().describe('Widget width'),
        height: z.number().describe('Widget height'),
        config: z.record(z.any()).optional().describe('Additional widget configuration')
      },
      async ({ type, id, x, y, width, height, config }) => {
        const result = updateDashboard((dashboard) => {
          // Check if ID already exists
          if (dashboard.widgets.find(w => w.id === id)) {
            return { error: `Widget with ID ${id} already exists` };
          }

          const newWidget = {
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
              text: result.error || result.message
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
      async ({ widgetId }) => {
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
              text: result.error || result.message
            }
          ],
          isError: !!result.error
        };
      }
    )
  );

  // Global tool to regenerate all widgets with commands
  tools.push(
    tool(
      'regenerate_all_widgets',
      'Regenerate all widgets in this dashboard that have regenerate commands configured',
      {},
      async () => {
        try {
          const { ipcMain } = require('electron');

          // Get the regenerate-all-widgets handler
          const handlers = ipcMain._events['regenerate-all-widgets'];
          if (!handlers || handlers.length === 0) {
            return {
              content: [{
                type: 'text',
                text: 'Regenerate-all handler not available'
              }],
              isError: true
            };
          }

          // Call the handler
          const handler = Array.isArray(handlers) ? handlers[0] : handlers;
          const result = await handler(null, {
            dashboardId: config.id
          });

          if (!result.success) {
            return {
              content: [
                {
                  type: 'text',
                  text: `Failed to regenerate widgets: ${result.error}`
                }
              ],
              isError: true
            };
          }

          // Format results
          const successResults = result.results.filter(r => r.success);
          const failureResults = result.results.filter(r => !r.success);

          let message = `Regenerated ${successResults.length} widget(s)`;
          if (failureResults.length > 0) {
            message += `\nFailed: ${failureResults.length} widget(s)`;
            failureResults.forEach(r => {
              message += `\n  - ${r.label || r.widgetId}: ${r.error}`;
            });
          }

          return {
            content: [
              {
                type: 'text',
                text: message
              }
            ],
            isError: failureResults.length > 0
          };
        } catch (error) {
          return {
            content: [
              {
                type: 'text',
                text: `Error regenerating widgets: ${error.message}`
              }
            ],
            isError: true
          };
        }
      }
    )
  );

  // Plan mode only: Ask user a question with multiple choice or custom input
  if (permissionMode === 'plan') {
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
              allowMultiple: z.boolean().optional().default(false).describe('Allow multiple selections'),
              allowCustom: z.boolean().optional().default(true).describe('Show "type my own answer" option')
            }),
            // Multiple questions
            z.array(z.object({
              id: z.string().describe('Unique ID for this question (e.g., "widget-type", "position")'),
              question: z.string().describe('The question to ask'),
              options: z.array(z.string()).optional().describe('Optional predefined choices'),
              allowMultiple: z.boolean().optional().default(false).describe('Allow multiple selections'),
              allowCustom: z.boolean().optional().default(true).describe('Show "type my own answer" option')
            }))
          ]).describe('Either a single question object or an array of question objects to ask together')
        },
        async ({ questions }) => {
          try {
            const { ipcMain, BrowserWindow } = require('electron');
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

              ipcMain.once(`question-answer-${answerId}`, (event, answers) => {
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
          } catch (error) {
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
  }

  // Create and return the MCP server
  return createSdkMcpServer({
    name: `dashboard-${config.id}`,
    version: '1.0.0',
    tools
  });
}

module.exports = { createDashboardTools };
