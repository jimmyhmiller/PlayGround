file: examples/assistant.scry
stdin: exit\n
expr: Orchestrator.instance(0).useClaude()
expr: Orchestrator.instance(0)
expr: CliModel.instances()
expr: Orchestrator.instance(0).allowWorkspaceWrites()
expr: Orchestrator.instance(0).setModel("opus")
expr: Orchestrator.instance(0)
readonly: no
contains: "provider":{"type":"String","value":"claude"}
contains: "access":{"type":"String","value":"read-only"}
contains: "elementType":"CliModel","length":1
contains: "access":{"type":"String","value":"workspace-write"}
contains: "modelId":{"type":"String","value":"opus"}
contains: "brainName":{"type":"String","value":"Claude subscription (workspace-write)"}
