# verify

A super-simple utility: hand it **instructions** and an **input**, and it asks an
LLM whether the input follows the instructions. The model is forced to call one
tool — `report_verdict` — returning `{ follows, confidence, reason }`, and the
**process exit code is the answer**:

| exit | meaning |
|------|---------|
| `0`  | the input **follows** the instructions (yes) |
| `1`  | the input does **not** follow them (no) |
| `2`  | error (bad usage, missing API key, API/tool-call failure) |

So it drops straight into shell logic:

```sh
if echo "$DATA" | verify -I rules.txt --quiet; then echo valid; else echo rejected; fi
```

## Install

```sh
./install.sh            # symlinks bin/verify -> ~/.local/bin/verify
export VERIFY_API_KEY=sk-...
verify --help
```

`bin/verify` is a single self-contained Node file (needs Node ≥ 18); no deps.

## Usage

Instructions: positional, `-i <text>`, or `-I <file>`.
Input: stdin (default), `--input <text>`, or `--input-file <file>`.

```sh
# instructions positional, input on stdin
echo '{"name":"x"}' | verify "Must be valid JSON containing a name field"

# both from files
verify -I rules.txt --input-file data.json

# inline, full verdict as JSON to stdout
verify -i "Must be all lowercase" --input "Hello" --json
# {"follows":false,"confidence":97,"reason":"contains an uppercase 'H'"}
```

`--quiet` suppresses all output (use the exit code only). Default human output
goes to **stderr**, so `--json` on stdout stays clean for piping.

## Configuration

Same OpenAI-compatible setup as `hivemind` — any `/chat/completions` endpoint
that supports tool calls.

| env var | default | meaning |
|---|---|---|
| `VERIFY_API_KEY` | — (required; falls back to `DEEPSEEK_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`) | API key |
| `VERIFY_BASE_URL` | `https://api.deepseek.com` | endpoint base URL |
| `VERIFY_MODEL` | `deepseek-chat` | model name |
| `VERIFY_TEMPERATURE` | `0` | sampling temperature |

`--model` and `--base-url` flags override the env vars per-invocation.
