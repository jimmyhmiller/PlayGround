#!/bin/bash

# Rename PDFs to clean names based on title and author
# Format: Title_by_Author.pdf or just Title.pdf

WORK_DIR="/Users/jimmyhmiller/Documents/Code/readings"
cd "$WORK_DIR" || exit 1

# Function to clean text for filename
clean_filename() {
    local text="$1"
    # Remove leading/trailing whitespace
    text=$(echo "$text" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    # Replace spaces and multiple whitespace with underscore
    text=$(echo "$text" | sed 's/[[:space:]]\+/_/g')
    # Remove special characters except underscore, hyphen, period
    text=$(echo "$text" | sed 's/[^a-zA-Z0-9._-]//g')
    # Remove multiple underscores
    text=$(echo "$text" | sed 's/__\+/_/g')
    # Remove leading/trailing underscores
    text=$(echo "$text" | sed 's/^_\+//;s/_\+$//')
    # Limit length to 200 chars
    text=$(echo "$text" | cut -c1-200)
    echo "$text"
}

# Function to extract title from first page content
extract_title_from_content() {
    local pdf="$1"
    # Extract first page, get first few non-empty lines
    local content=$(pdftotext -l 1 -q "$pdf" - 2>/dev/null | head -20)

    # Try to find title (usually one of the first few lines, often in title case or all caps)
    local title=$(echo "$content" | grep -v '^[[:space:]]*$' | head -5 | tail -1)

    if [ -n "$title" ]; then
        echo "$title"
    fi
}

# Function to get a unique filename
get_unique_filename() {
    local base="$1"
    local ext="$2"
    local counter=1
    local filename="${base}${ext}"

    while [ -f "$filename" ]; do
        filename="${base}_${counter}${ext}"
        ((counter++))
    done

    echo "$filename"
}

echo "Renaming PDFs to clean titles..."
echo ""

count=0
renamed=0
skipped=0

# Process each PDF
find . -maxdepth 1 -name "*.pdf" -type f | while read -r pdf; do
    ((count++))

    filename=$(basename "$pdf")

    # Skip if already has a reasonable name (not all hex/numbers)
    if [[ ! "$filename" =~ ^[0-9a-fA-F_-]+\.pdf$ ]] && [[ ! "$filename" =~ ^[0-9]+.*\.pdf$ ]]; then
        # Check if name is already reasonable (has some letters in a pattern)
        if [[ ${#filename} -lt 100 ]] && [[ "$filename" =~ [a-zA-Z]{3,} ]]; then
            ((skipped++))
            if [ $((count % 50)) -eq 0 ]; then
                echo "... processed $count PDFs (renamed: $renamed, skipped: $skipped)"
            fi
            continue
        fi
    fi

    # Extract metadata
    title=$(pdfinfo "$pdf" 2>/dev/null | grep "^Title:" | sed 's/^Title:[[:space:]]*//' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    author=$(pdfinfo "$pdf" 2>/dev/null | grep "^Author:" | sed 's/^Author:[[:space:]]*//' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    # If no title in metadata, try extracting from content
    if [ -z "$title" ] || [ "$title" = "Untitled" ] || [ "$title" = "untitled" ]; then
        title=$(extract_title_from_content "$pdf")
    fi

    # Clean title and author
    title=$(clean_filename "$title")
    author=$(clean_filename "$author")

    # Build new filename
    new_name=""
    if [ -n "$title" ] && [ -n "$author" ]; then
        new_name="${title}_by_${author}"
    elif [ -n "$title" ]; then
        new_name="$title"
    elif [ -n "$author" ]; then
        new_name="by_${author}"
    else
        # No metadata, keep original name
        ((skipped++))
        if [ $((count % 50)) -eq 0 ]; then
            echo "... processed $count PDFs (renamed: $renamed, skipped: $skipped)"
        fi
        continue
    fi

    # Ensure name isn't too short or weird
    if [ ${#new_name} -lt 5 ]; then
        ((skipped++))
        if [ $((count % 50)) -eq 0 ]; then
            echo "... processed $count PDFs (renamed: $renamed, skipped: $skipped)"
        fi
        continue
    fi

    # Get unique filename
    new_filename=$(get_unique_filename "$new_name" ".pdf")

    # Rename
    if [ "$filename" != "$new_filename" ]; then
        mv "$pdf" "$new_filename" && {
            echo "✓ $filename"
            echo "  → $new_filename"
            ((renamed++))
        }
    else
        ((skipped++))
    fi

    # Progress update
    if [ $((count % 50)) -eq 0 ]; then
        echo "... processed $count PDFs (renamed: $renamed, skipped: $skipped)"
    fi
done

echo ""
echo "========================================"
echo "Processed: $count PDFs"
echo "Renamed: $renamed PDFs"
echo "Skipped: $skipped PDFs"
