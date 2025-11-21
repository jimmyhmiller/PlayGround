#!/bin/bash
# Find files in git by their hash

cd /Users/jimmyhmiller/Documents/Code/PlayGround

echo "Finding files for the top discrepant hashes..."
echo

# Test a few hashes
for hash in "59171f5554dda9389c9ac7a437b1dd0684931f90" "2522918ca89993e251fd1ce3a2ad0def70227ff1"; do
    echo "Hash: $hash"
    git ls-tree -r HEAD | grep "$hash" || echo "  Not found at HEAD"
    echo
done
