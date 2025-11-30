"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.createDashboardTools = createDashboardTools;
const fs = __importStar(require("fs"));
const claude_agent_sdk_1 = require("@anthropic-ai/claude-agent-sdk");
const zod_1 = require("zod");
/**
 * Generates dashboard-specific MCP tools based on the dashboard configuration
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
        }
        catch (error) {
            return { error: error.message };
        }
    }
    // Generate widget-specific tools based on widget type
    config.widgets?.forEach(widget => {
        const widgetId = widget.id;
        const widgetType = widget.type;
        const widgetLabel = Array.isArray(widget.label) ? widget.label.join(' - ') : widget.label || widgetId;
        const widgetConfig = widget;
        // Skip read-only widgets
        if (widgetConfig.readOnly || widgetConfig.derived || widgetType === 'claude-todo-list') {
            console.log(`[Dashboard Tools] Skipping read-only/derived widget: ${widgetId} (${widgetType})`);
            return;
        }
        // Tool for stat widgets - quick value updates
        if (widgetType === 'stat') {
            tools.push((0, claude_agent_sdk_1.tool)(`update_stat_${widgetId}`, `Update the value of the "${widgetLabel}" stat widget`, {
                value: zod_1.z.string().describe('The new value to display (e.g., "99%", "1.2s", "Active")')
            }, async ({ value }) => {
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
            }));
        }
        // Tool for barChart widgets - update data arrays
        if (widgetType === 'bar-chart') {
            tools.push((0, claude_agent_sdk_1.tool)(`update_chart_${widgetId}`, `Update the data for the "${widgetLabel}" chart widget`, {
                data: zod_1.z.array(zod_1.z.number()).describe('Array of numbers representing chart data points')
            }, async ({ data }) => {
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
            }));
        }
        // Tool for progress widgets
        if (widgetType === 'progress') {
            tools.push((0, claude_agent_sdk_1.tool)(`update_progress_${widgetId}`, `Update the progress value for the "${widgetLabel}" widget`, {
                value: zod_1.z.number().min(0).max(100).describe('Progress percentage (0-100)'),
                text: zod_1.z.string().optional().describe('Optional text label to display')
            }, async ({ value, text }) => {
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
            }));
        }
        // Tool for todoList widgets - add items
        if (widgetType === 'todo-list') {
            tools.push((0, claude_agent_sdk_1.tool)(`add_todo_${widgetId}`, `Add a todo item to the "${widgetLabel}" list`, {
                text: zod_1.z.string().describe('The todo item text'),
                done: zod_1.z.boolean().optional().default(false).describe('Whether the todo is completed')
            }, async ({ text, done }) => {
                const result = updateDashboard((dashboard) => {
                    const targetWidget = dashboard.widgets.find(w => w.id === widgetId);
                    if (!targetWidget) {
                        return { error: `Widget ${widgetId} not found` };
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
            }));
            tools.push((0, claude_agent_sdk_1.tool)(`toggle_todo_${widgetId}`, `Toggle the completion status of a todo item in "${widgetLabel}"`, {
                index: zod_1.z.number().int().min(0).describe('Index of the todo item to toggle (0-based)')
            }, async ({ index }) => {
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
            }));
        }
        // Tool for keyValue widgets - update key-value pairs
        if (widgetType === 'key-value') {
            tools.push((0, claude_agent_sdk_1.tool)(`update_keyvalue_${widgetId}`, `Update or add a key-value pair in "${widgetLabel}"`, {
                key: zod_1.z.string().describe('The key to update or add'),
                value: zod_1.z.string().describe('The value to set')
            }, async ({ key, value }) => {
                const result = updateDashboard((dashboard) => {
                    const targetWidget = dashboard.widgets.find(w => w.id === widgetId);
                    if (!targetWidget) {
                        return { error: `Widget ${widgetId} not found` };
                    }
                    if (!targetWidget.items) {
                        targetWidget.items = [];
                    }
                    const existing = targetWidget.items.find((item) => item.key === key);
                    if (existing) {
                        existing.value = value;
                        return { message: `Updated ${key} = ${value} in ${widgetLabel}` };
                    }
                    else {
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
            }));
        }
    });
    // Create the MCP server with these tools
    return (0, claude_agent_sdk_1.createSdkMcpServer)({
        name: `dashboard-${config.id}`,
        version: '1.0.0',
        tools
    });
}
