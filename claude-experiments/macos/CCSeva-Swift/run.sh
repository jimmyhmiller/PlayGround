#!/bin/bash
echo "Building CCSeva..."
swift build

echo "Running CCSeva in background..."
.build/debug/CCSeva &

echo "CCSeva should now be running in the menu bar!"
echo "Look for '75%' in your menu bar and click it."
echo ""
echo "To stop the app, run: pkill -f CCSeva"
echo "To see debug output, run: .build/debug/CCSeva"