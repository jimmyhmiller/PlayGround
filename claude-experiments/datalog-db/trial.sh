#!/usr/bin/env bash
set -euo pipefail

# Trial run: load JSONPlaceholder data, run queries, test retract + time travel.
# Usage: ./trial.sh

ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR=$(mktemp -d)
DOWNLOAD_DIR=$(mktemp -d)
PORT=15557
ADDR="127.0.0.1:$PORT"
SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$DATA_DIR" "$DOWNLOAD_DIR"
}
trap cleanup EXIT

cli() {
    "$ROOT/target/debug/datalog-cli" --host "$ADDR" "$@"
}

# ── Build ──────────────────────────────────────────────────────

echo "Building..."
cargo build --manifest-path "$ROOT/Cargo.toml" 2>&1 | tail -1

# ── Start server ───────────────────────────────────────────────

echo "Starting server on $ADDR..."
"$ROOT/target/debug/datalog-db" --data-dir "$DATA_DIR" --bind "$ADDR" 2>/dev/null &
SERVER_PID=$!
sleep 1

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: Server failed to start"
    exit 1
fi
echo "Server running (pid $SERVER_PID)"

# ── Download data ──────────────────────────────────────────────

echo ""
echo "Downloading JSONPlaceholder data..."
curl -sf https://jsonplaceholder.typicode.com/users    -o "$DOWNLOAD_DIR/users.json"
curl -sf https://jsonplaceholder.typicode.com/posts    -o "$DOWNLOAD_DIR/posts.json"
curl -sf https://jsonplaceholder.typicode.com/comments -o "$DOWNLOAD_DIR/comments.json"
curl -sf https://jsonplaceholder.typicode.com/todos    -o "$DOWNLOAD_DIR/todos.json"
echo "  users:    $(python3 -c "import json; print(len(json.load(open('$DOWNLOAD_DIR/users.json'))))")"
echo "  posts:    $(python3 -c "import json; print(len(json.load(open('$DOWNLOAD_DIR/posts.json'))))")"
echo "  comments: $(python3 -c "import json; print(len(json.load(open('$DOWNLOAD_DIR/comments.json'))))")"
echo "  todos:    $(python3 -c "import json; print(len(json.load(open('$DOWNLOAD_DIR/todos.json'))))")"

# ── Define schema ──────────────────────────────────────────────

echo ""
echo "Defining schema..."
cli define 'User { name: string required, username: string required, email: string required unique, phone: string, website: string, company_name: string, city: string }'
cli define 'Post { title: string required, body: string required, author: ref(User) required indexed }'
cli define 'Comment { post: ref(Post) required indexed, commenter_name: string required, commenter_email: string required, body: string required }'
cli define 'Todo { title: string required, completed: bool required, assignee: ref(User) required indexed }'

# ── Load data ──────────────────────────────────────────────────

echo ""
echo "Loading data..."

# The loader writes user_map.json and post_map.json for later use, and prints stats.
python3 - "$DOWNLOAD_DIR" "$ROOT/target/debug/datalog-cli" "$ADDR" <<'PYEOF'
import json, subprocess, sys, time

download_dir, cli_bin, addr = sys.argv[1], sys.argv[2], sys.argv[3]
start = time.time()

def cli_json(payload):
    r = subprocess.run([cli_bin, '--host', addr, 'json', json.dumps(payload)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"ERROR: {r.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return r.stdout.strip()

def parse_eids(out):
    for part in out.split():
        if part.startswith('entity_ids='):
            return [int(x) for x in part.split('=')[1].strip('[]').split(',')]
    return []

# Users
with open(f'{download_dir}/users.json') as f:
    users = json.load(f)
ops = [{'assert': 'User', 'data': {
    'name': u['name'], 'username': u['username'], 'email': u['email'],
    'phone': u['phone'], 'website': u['website'],
    'company_name': u['company']['name'], 'city': u['address']['city'],
}} for u in users]
out = cli_json({'type': 'transact', 'ops': ops})
eids = parse_eids(out)
user_map = {u['id']: eids[i] for i, u in enumerate(users)}
print(f"  Users:    {len(users):>4}  ({out})")

# Posts
with open(f'{download_dir}/posts.json') as f:
    posts = json.load(f)
post_map = {}
for i in range(0, len(posts), 25):
    batch = posts[i:i+25]
    ops = [{'assert': 'Post', 'data': {
        'title': p['title'], 'body': p['body'],
        'author': {'ref': user_map[p['userId']]},
    }} for p in batch]
    out = cli_json({'type': 'transact', 'ops': ops})
    for j, eid in enumerate(parse_eids(out)):
        post_map[batch[j]['id']] = eid
print(f"  Posts:    {len(posts):>4}")

# Comments
with open(f'{download_dir}/comments.json') as f:
    comments = json.load(f)
for i in range(0, len(comments), 50):
    batch = comments[i:i+50]
    ops = [{'assert': 'Comment', 'data': {
        'post': {'ref': post_map[c['postId']]},
        'commenter_name': c['name'], 'commenter_email': c['email'], 'body': c['body'],
    }} for c in batch]
    cli_json({'type': 'transact', 'ops': ops})
print(f"  Comments: {len(comments):>4}")

# Todos
with open(f'{download_dir}/todos.json') as f:
    todos = json.load(f)
for i in range(0, len(todos), 50):
    batch = todos[i:i+50]
    ops = [{'assert': 'Todo', 'data': {
        'title': t['title'], 'completed': t['completed'],
        'assignee': {'ref': user_map[t['userId']]},
    }} for t in batch]
    cli_json({'type': 'transact', 'ops': ops})
print(f"  Todos:    {len(todos):>4}")

total = len(users) + len(posts) + len(comments) + len(todos)
elapsed = time.time() - start
print(f"  Total:    {total:>4} entities in {elapsed:.1f}s")

# Save user map so the shell script can find Chelsey's entity ID
with open(f'{download_dir}/user_map.json', 'w') as f:
    json.dump(user_map, f)
PYEOF

# Read Chelsey's entity ID (userId 5 in JSONPlaceholder)
CHELSEY_EID=$(python3 -c "import json; m=json.load(open('$DOWNLOAD_DIR/user_map.json')); print(m['5'])")
echo "  Chelsey Dietrich is entity #$CHELSEY_EID"

# ── Queries ────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════"
echo "  QUERIES"
echo "═══════════════════════════════════════════"

echo ""
echo "── All Users ──"
cli query 'find ?name, ?username, ?city where ?u: User { name: ?name, username: ?username, city: ?city }'

echo ""
echo "── Posts by Leanne Graham ──"
cli query 'find ?title where ?u: User { name: "Leanne Graham" }, ?p: Post { author: ?u, title: ?title }'

echo ""
echo "── 3-way join: Comments on Ervin Howell's posts ──"
cli query 'find ?post_title, ?commenter_name where ?u: User { name: "Ervin Howell" }, ?p: Post { author: ?u, title: ?post_title }, ?c: Comment { post: ?p, commenter_name: ?commenter_name }'

echo ""
echo "── Incomplete Todos for Chelsey Dietrich ──"
cli query 'find ?title where ?u: User { name: "Chelsey Dietrich" }, ?t: Todo { assignee: ?u, title: ?title, completed: false }'

# ── Retract entity + time travel ───────────────────────────────

echo ""
echo "═══════════════════════════════════════════"
echo "  RETRACT ENTITY + TIME TRAVEL"
echo "═══════════════════════════════════════════"

echo ""
echo "Soft-deleting Chelsey Dietrich (#$CHELSEY_EID)..."
cli json "{\"type\":\"transact\",\"ops\":[{\"retract_entity\":\"User\",\"entity\":$CHELSEY_EID}]}"

echo ""
echo "── Users after retraction (9 rows, no Chelsey) ──"
cli query 'find ?name, ?city where ?u: User { name: ?name, city: ?city }'

echo ""
echo "── Time travel: as_of tx 5 (all 10 users, Chelsey reappears) ──"
cli query 'find ?name, ?city where ?u: User { name: ?name, city: ?city } as_of 5'

echo ""
echo "── Chelsey's todos after retraction (should be empty) ──"
cli query 'find ?title where ?u: User { name: "Chelsey Dietrich" }, ?t: Todo { assignee: ?u, title: ?title }'

# ── Summary ────────────────────────────────────────────────────

echo ""
echo "── DB size ──"
du -sh "$DATA_DIR"

echo ""
echo "Done!"
