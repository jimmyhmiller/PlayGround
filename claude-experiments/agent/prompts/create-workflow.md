You are creating one executable Flowline workflow for this repository.

The user's goal (`.flowline/create.request`) and a map of the repository are included below as labeled sections. Design the workflow for this actual repository and that stated goal, not generic filler.

Reply with a line `file: workflows/<unique-descriptive-kebab-name>.flow` followed by exactly one fenced code block containing the complete workflow file. No other prose.

The grammar is line-oriented:

```text
workflow Human Readable Name
runner deepseek

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
- Agents run as single language-model calls; they cannot execute commands or edit files. Write prompts that ask for analysis, review, plans, or drafted content returned as text.
- Each agent receives a repository file map, and the full contents of repository files its prompt names. Name the exact files the agent should work from in its prompt.
