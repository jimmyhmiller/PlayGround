#!/usr/bin/env python3
"""Simple plotter for git-of-theseus JSON output"""
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
except ImportError:
    print("matplotlib not available, showing data summary instead")
    matplotlib = None

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_stack(data, output_path, title):
    """Create a stacked area chart"""
    if matplotlib is None:
        print(f"Skipping plot generation for {title}")
        return

    ts = [datetime.fromisoformat(t) for t in data['ts']]
    y_arrays = data['y']
    labels = data['labels']

    # Limit to top 20 series
    if len(labels) > 20:
        # Sum all values for each series to find the top contributors
        totals = [(sum(y), i) for i, y in enumerate(y_arrays)]
        totals.sort(reverse=True)
        top_indices = [i for _, i in totals[:20]]

        # Combine the rest into "other"
        other_y = [0] * len(ts)
        for i, y in enumerate(y_arrays):
            if i not in top_indices:
                other_y = [a + b for a, b in zip(other_y, y)]

        y_arrays = [y_arrays[i] for i in top_indices] + [other_y]
        labels = [labels[i] for i in top_indices] + ['other']

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.stackplot(ts, *y_arrays, labels=labels, alpha=0.8)

    ax.set_xlabel('Date')
    ax.set_ylabel('Lines of Code')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close()

def summarize_data(data, title):
    """Print a text summary of the data"""
    labels = data['labels']
    y_arrays = data['y']
    ts = data['ts']

    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Time range: {ts[0]} to {ts[-1]}")
    print(f"Number of data points: {len(ts)}")
    print(f"Number of series: {len(labels)}")

    if len(y_arrays) > 0 and len(y_arrays[0]) > 0:
        # Show final values
        final_values = [(labels[i], y_arrays[i][-1]) for i in range(len(labels))]
        final_values.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 10 contributors (final values):")
        for label, value in final_values[:10]:
            print(f"  {label:40s}: {value:>10,} lines")

        total = sum(val for _, val in final_values)
        print(f"\nTotal lines: {total:,}")

def main():
    output_dir = Path("analysis-output")

    # Load and plot/summarize each dataset
    datasets = [
        ("cohorts.json", "Code by Year Added", "cohorts_stack.png"),
        ("exts.json", "Code by File Extension", "exts_stack.png"),
        ("authors.json", "Code by Author", "authors_stack.png"),
        ("dirs.json", "Code by Directory", "dirs_stack.png"),
    ]

    for json_file, title, output_file in datasets:
        json_path = output_dir / json_file
        if not json_path.exists():
            print(f"Skipping {json_file} - file not found")
            continue

        data = load_json(json_path)
        summarize_data(data, title)

        if matplotlib is not None:
            output_path = output_dir / output_file
            plot_stack(data, output_path, title)

if __name__ == "__main__":
    main()
