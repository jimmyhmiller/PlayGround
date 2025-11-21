#!/bin/bash

# Script to add start/end fields to AST node records

for file in src/main/java/com/jsparser/ast/*.java; do
    if grep -q "public record.*implements.*Node" "$file" || grep -q "public record.*implements.*Statement" "$file" || grep -q "public record.*implements.*Expression" "$file"; then
        echo "Processing: $file"

        # Add start and end fields after "public record Name("
        sed -i 's/public record \([A-Za-z]*\)(/public record \1(\n    int start,\n    int end,/' "$file"

    fi
done

echo "Done!"
