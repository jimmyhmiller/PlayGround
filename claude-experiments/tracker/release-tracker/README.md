# Release Tracker

A small macOS app + WidgetKit widget for tracking what's left before
shipping Ease.

## Architecture

- **Host app** (`Sources/ReleaseTracker/`) — SwiftUI window with the full
  categorized checklist. Tick boxes here.
- **Widget extension** (`Sources/ReleaseTrackerWidget/`) — read-only
  desktop widget. Shows progress meter + next 3–8 unchecked tasks
  depending on widget size. Tap to open the host app.
- **Shared** (`Sources/Shared/`) — `ChecklistItem` model, `ChecklistStore`
  (reads/writes JSON in the App Group container), and `SeedData` with all
  the pre-populated tasks.

The two binaries share state through an App Group container:
`group.com.jimmyhmiller.ReleaseTracker`. The host app calls
`WidgetCenter.shared.reloadAllTimelines()` after every change so the
widget refreshes immediately.

## Build & install

```bash
./build.sh
cp -R ReleaseTracker.app /Applications/
open /Applications/ReleaseTracker.app
```

Launching once registers the widget with the system. Then:

1. Right-click your desktop → **Edit Widgets**
2. Search for **Ease Release**
3. Drag the widget onto the desktop in your preferred size (small,
   medium, or large)

## Modifying the checklist

Edit `Sources/Shared/SeedData.swift` and bump `currentSeedVersion`. On
next launch the store will merge the new items into your existing
checklist *without* losing any check marks you've already made — items
are matched by title. New items show up unchecked; items that no longer
exist in the seed are dropped.

If you want a clean slate, click the **Reset** toolbar button in the host
app.

## Data location

- Signed and inside an App Group:
  `~/Library/Group Containers/group.com.jimmyhmiller.ReleaseTracker/Library/Application Support/checklist.json`
- Running unsigned (e.g. `swift run`):
  `~/Library/Application Support/ReleaseTracker/checklist.json`
  *(the widget can't see this — needs the signed build)*

## Limitations

- Widget is read-only. Toggling an item from the widget would need an
  AppIntent — feasible but adds complexity. For now, tap the widget to
  open the host app and tick it there.
- Built with SwiftPM, not Xcode. The widget extension `.appex` is
  hand-assembled by `build.sh`. This works but is fiddlier than Xcode
  would be.
