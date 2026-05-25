#!/bin/sh
# "10 open issues" — requires `gh` CLI authenticated to the current
# repo. Pass `REPO=owner/name` to target a specific repo; otherwise the
# current directory's GitHub remote is used.
#
# Emits one frame and exits. Reopen the pane (or wire a refresh button)
# to re-fetch.

set -e

REPO_ARG=""
if [ -n "$REPO" ]; then
    REPO_ARG="--repo $REPO"
fi

# Fetch as NDJSON-friendly compact JSON, build the widget frame in jq.
gh issue list $REPO_ARG --json number,title,state,url --limit 10 | jq -c '
  {type:"frame", root:{
    type:"vstack", gap:4, pad:10,
    children:
      ([
        {type:"text", value:"Open Issues", weight:"bold", size:14},
        {type:"divider"},
        {type:"spacer", size:2}
      ]
      +
      [ .[] | {type:"hstack", gap:8, children:[
          {type:"badge", value:.state, color:(if .state=="OPEN" then "#3a7a3a" else "#7a3a3a" end)},
          {type:"link", url:.url, label:("#"+(.number|tostring))},
          {type:"text", value:.title, color:"#cfd"}
        ]}
      ])
  }}
'
