You are creating one executable Flowline workflow for this repository.

Read `.flowline/create.request` for the user's goal. Inspect the repository to understand the project before designing the workflow.

Create exactly one new file under `workflows/` with a unique descriptive kebab-case filename ending in `.flow`. Do not modify any other file.

The grammar is line-oriented:

```text
workflow Human Readable Name
runner codex

agent unique-node-name
title Human readable node title
prompt A complete standalone instruction for this agent. Keep it on one line.
after none

approval unique-approval-name
title Human readable approval title
after previous-node-name
```

Rules:

- Define between 2 and 7 nodes.
- Every node needs a unique kebab-case name, a title, and an `after` line.
- Agent nodes also need one single-line prompt.
- `after none` defines a root node.
- `after first, second` joins two dependencies.
- Dependencies must refer to earlier nodes.
- Use independent sibling agents when work can happen concurrently.
- Use an approval node before consequential or irreversible work.
- Prompts must name the repository files or project areas agents should inspect when useful.
- The workflow must be useful for this actual repository and the user's stated goal, not generic filler.
