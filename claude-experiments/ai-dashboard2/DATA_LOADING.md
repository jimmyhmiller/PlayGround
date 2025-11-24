# Widget Data Loading

The dashboard now supports loading data for widgets from both inline JSON and external files.

## Usage

### Inline Data

You can provide data directly in the widget configuration:

```json
{
  "id": "chart1",
  "type": "barChart",
  "label": ["Network Traffic", "LIVE"],
  "data": [85, 92, 78, 65, 88, 95, 72, 80, 90, 75]
}
```

### File Reference

You can load data from an external JSON file using the `dataSource` property:

```json
{
  "id": "chart1",
  "type": "barChart",
  "label": ["Packet Flow", "LIVE"],
  "dataSource": "network-data.json"
}
```

The file path can be:
- **Relative**: Resolved relative to the dashboard JSON file's directory
- **Absolute**: Used as-is

## Supported Data Formats

The BarChart widget supports multiple data formats:

### 1. Array of Numbers
```json
[20, 45, 60, 80, 35, 70]
```

### 2. Array of Objects
```json
[
  {"value": 20},
  {"value": 45},
  {"value": 60}
]
```

### 3. Object with Values Array
```json
{
  "values": [20, 45, 60, 80, 35, 70]
}
```

## Examples

### Example 1: Inline Data (resizable-test.json)
```json
{
  "id": "chart1",
  "type": "barChart",
  "label": ["Network Traffic", "LIVE"],
  "data": [85, 92, 78, 65, 88, 95, 72, 80, 90, 75, 83, 88],
  "width": "620px",
  "height": "320px"
}
```

### Example 2: File Reference (grid.json)
```json
{
  "id": "chart",
  "type": "barChart",
  "area": "chart",
  "label": ["Packet Flow", "LIVE"],
  "dataSource": "network-data.json"
}
```

With `network-data.json`:
```json
{
  "values": [12, 34, 56, 78, 45, 67, 89, 32, 54, 76]
}
```

## Loading States

The widget will display:
- **Loading**: "Loading data..." while fetching from file
- **Error**: "Error: [message]" if loading fails
- **Fallback**: Random data if no data source is provided

## Implementation Details

### useWidgetData Hook

The `useWidgetData` hook handles data loading:

```javascript
const { data, loading, error } = useWidgetData(config);
```

It checks for:
1. `config.data` - inline data (highest priority)
2. `config.dataSource` - file path to load
3. Falls back to widget's default behavior if neither is provided

### Backend API

The Electron main process provides the `load-data-file` IPC handler that:
1. Resolves relative paths from dashboard directories
2. Reads and parses JSON files
3. Returns the data or error message

Exposed in renderer via `window.dashboardAPI.loadDataFile(filePath)`.

## Adding Data Support to Other Widgets

To add data loading to a widget:

1. Use the `useWidgetData` hook:
```javascript
function MyWidget({ theme, config }) {
  const { data, loading, error } = useWidgetData(config);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  // Use data here...
}
```

2. Handle different data formats as needed
3. Provide a fallback if no data is available
