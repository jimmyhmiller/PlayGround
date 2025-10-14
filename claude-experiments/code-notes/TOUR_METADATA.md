# Tour Metadata Convention

## Overview

Tours in code-notes are created using the existing note system with specific metadata fields. This approach leverages the extensible metadata system rather than creating separate tour data structures.

## Metadata Fields

### Required Fields

- **`tour_id`** (string): Unique identifier for the tour (e.g., "auth-tour", "onboarding-basics")
- **`tour_order`** (number): Position of this note in the tour sequence (1, 2, 3, ...)

### Optional Fields

- **`tour_title`** (string): Human-readable title for the tour (typically on first note)
- **`tour_description`** (string): Description of what the tour covers (typically on first note)
- **`tour_path`** (string): Which path this note belongs to (default: "main")
- **`tour_branches`** (array): Available branches from this note
  - Each branch: `{ "label": "Branch Name", "to_path": "path-id" }`
- **`tour_return_to`** (string): For branch paths, which path to return to
- **`tour_tags`** (array): Tags for categorizing tours (e.g., ["beginner", "authentication"])

## Examples

### Simple Linear Tour

```json
// Note 1
{
  "id": "uuid-1",
  "content": "Welcome to the authentication tour!",
  "metadata": {
    "tour_id": "auth-tour",
    "tour_title": "Understanding Authentication",
    "tour_description": "Learn how our auth system works",
    "tour_order": 1,
    "tour_tags": ["beginner", "security"]
  }
}

// Note 2
{
  "id": "uuid-2",
  "content": "This function validates JWT tokens...",
  "metadata": {
    "tour_id": "auth-tour",
    "tour_order": 2
  }
}

// Note 3
{
  "id": "uuid-3",
  "content": "User claims are extracted here...",
  "metadata": {
    "tour_id": "auth-tour",
    "tour_order": 3
  }
}
```

### Tour with Branching

```json
// Main path note with branch point
{
  "id": "uuid-2",
  "content": "JWT validation happens here",
  "metadata": {
    "tour_id": "auth-tour",
    "tour_order": 2,
    "tour_path": "main",
    "tour_branches": [
      {
        "label": "JWT Deep Dive",
        "to_path": "jwt-details"
      },
      {
        "label": "Error Handling",
        "to_path": "error-path"
      }
    ]
  }
}

// Branch path note
{
  "id": "uuid-10",
  "content": "JWT structure explained in detail...",
  "metadata": {
    "tour_id": "auth-tour",
    "tour_order": 1,
    "tour_path": "jwt-details",
    "tour_return_to": "main"
  }
}
```

## Usage Patterns

### Creating a Tour

Using the existing `code-notes add` command with metadata:

```bash
# Note 1
code-notes add \
  --file src/auth.rs \
  --line 10 \
  --column 0 \
  --content "Welcome to the authentication tour!" \
  --author "Alice" \
  --collection onboarding \
  --metadata '{"tour_id":"auth-tour","tour_title":"Understanding Authentication","tour_order":1,"tour_tags":["beginner"]}'

# Note 2
code-notes add \
  --file src/auth.rs \
  --line 42 \
  --column 0 \
  --content "JWT validation happens here" \
  --author "Alice" \
  --collection onboarding \
  --metadata '{"tour_id":"auth-tour","tour_order":2}'
```

### Querying Tour Notes

The TUI viewer queries notes by:
1. Filter by `tour_id` metadata field
2. Sort by `tour_order`
3. Group by `tour_path` (default: "main")
4. Handle branches via `tour_branches` array

## Benefits of Metadata-Based Tours

1. **No New Data Structures**: Reuses existing note system
2. **Flexible**: Easy to add new tour-related metadata fields
3. **Searchable**: Can query tours using existing note queries
4. **Backward Compatible**: Notes without tour metadata still work normally
5. **Lightweight**: No separate storage or management overhead
6. **Composable**: Notes can belong to multiple tours (different `tour_id` values)

## Implementation in TUI Viewer

The Node.js TUI viewer will:

```javascript
// Load all notes for a tour
function loadTour(tourId, collection) {
  const notes = loadCollection(collection);
  const tourNotes = notes.filter(n =>
    n.metadata && n.metadata.tour_id === tourId
  );

  // Group by path
  const paths = {};
  tourNotes.forEach(note => {
    const path = note.metadata.tour_path || 'main';
    if (!paths[path]) paths[path] = [];
    paths[path].push(note);
  });

  // Sort each path by tour_order
  Object.keys(paths).forEach(pathId => {
    paths[pathId].sort((a, b) =>
      (a.metadata.tour_order || 0) - (b.metadata.tour_order || 0)
    );
  });

  return {
    id: tourId,
    title: tourNotes[0]?.metadata?.tour_title || tourId,
    description: tourNotes[0]?.metadata?.tour_description,
    paths
  };
}
```

## CLI Helper Commands

We can add convenience commands to make tour creation easier:

```bash
# List available tours
code-notes list --filter-metadata tour_id

# View a specific tour
code-notes list --filter-metadata tour_id=auth-tour --sort-by tour_order

# Set tour metadata on existing note
code-notes update <note-id> --set-metadata tour_id=auth-tour
code-notes update <note-id> --set-metadata tour_order=3

# Add branch metadata
code-notes update <note-id> --set-metadata 'tour_branches=[{"label":"Deep Dive","to_path":"details"}]'
```

## Tour Discovery

To discover available tours, the TUI can:

1. Load all notes from a collection
2. Extract unique `tour_id` values
3. Find the note with `tour_order=1` for each tour (has title/description)
4. Present a menu of available tours

## Migration from Old Tour Format

If you had the old tour data structures, they can be converted to metadata:

```javascript
function convertOldTourToMetadata(oldTour) {
  const metadataUpdates = [];

  oldTour.paths.forEach(path => {
    path.notes.forEach((noteId, index) => {
      metadataUpdates.push({
        noteId,
        metadata: {
          tour_id: oldTour.id,
          tour_order: index + 1,
          tour_path: path.id,
          ...(index === 0 && path.id === 'main' ? {
            tour_title: oldTour.title,
            tour_description: oldTour.description
          } : {})
        }
      });
    });
  });

  return metadataUpdates;
}
```

## Best Practices

1. **Use Consistent IDs**: Use kebab-case for `tour_id` (e.g., "auth-tour", "api-basics")
2. **Number Sequentially**: Use sequential integers for `tour_order` (1, 2, 3...)
3. **First Note Metadata**: Put `tour_title` and `tour_description` on the first note (`tour_order=1`)
4. **Branch Labels**: Use descriptive branch labels ("Deep Dive", "Alternative Approach")
5. **Path Naming**: Use simple path names ("main", "jwt-details", "error-handling")
6. **Tags**: Add relevant tags for discovery and filtering
7. **Collections**: Group tour notes in appropriate collections (e.g., "onboarding", "architecture")

## Example: Complete Tour

```bash
# Create onboarding tour with 3 notes and 1 branch

# Main path note 1 (intro)
code-notes add --file README.md --line 1 --column 0 \
  --content "Welcome! This tour will guide you through our auth system." \
  --author "Alice" --collection onboarding \
  --metadata '{"tour_id":"auth-tour","tour_title":"Auth System Tour","tour_description":"Learn authentication","tour_order":1,"tour_path":"main","tour_tags":["beginner","auth"]}'

# Main path note 2 (JWT validation with branch)
code-notes add --file src/auth.rs --line 42 --column 0 \
  --content "JWT tokens are validated here. See RS256 algorithm." \
  --author "Alice" --collection onboarding \
  --metadata '{"tour_id":"auth-tour","tour_order":2,"tour_path":"main","tour_branches":[{"label":"JWT Deep Dive","to_path":"jwt-details"}]}'

# Main path note 3 (user extraction)
code-notes add --file src/auth.rs --line 65 --column 0 \
  --content "User claims are extracted and User struct is created." \
  --author "Alice" --collection onboarding \
  --metadata '{"tour_id":"auth-tour","tour_order":3,"tour_path":"main"}'

# Branch path note 1
code-notes add --file src/jwt.rs --line 10 --column 0 \
  --content "JWT structure: header.payload.signature" \
  --author "Alice" --collection onboarding \
  --metadata '{"tour_id":"auth-tour","tour_order":1,"tour_path":"jwt-details","tour_return_to":"main"}'

# Branch path note 2
code-notes add --file src/jwt.rs --line 50 --column 0 \
  --content "RS256 uses RSA public/private key pairs" \
  --author "Alice" --collection onboarding \
  --metadata '{"tour_id":"auth-tour","tour_order":2,"tour_path":"jwt-details","tour_return_to":"main"}'
```

Then view in TUI:
```bash
code-notes-tour auth-tour onboarding
```

## Future Extensions

The metadata system allows easy extension:

- **`tour_difficulty`**: "beginner", "intermediate", "advanced"
- **`tour_duration`**: Estimated time in minutes
- **`tour_prerequisites`**: Array of tour IDs that should be completed first
- **`tour_quiz`**: Interactive questions at certain points
- **`tour_links`**: External documentation links
- **`tour_video`**: Associated video URL
- **`tour_last_updated`**: Timestamp for maintenance
- **`tour_author_notes`**: Private notes for tour maintainers
