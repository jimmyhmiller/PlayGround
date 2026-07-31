file: examples/assistant.scry
stdin: exit\n
expr: Orchestrator.instance(0).useClaude()
expr: Orchestrator.instance(0)
expr: CliModel.instances()
expr: Orchestrator.instance(0).setModel("opus")
expr: Orchestrator.instance(0).setTimeoutMs(90000)
expr: Orchestrator.instance(0).setMaxOutputBytes(1048576)
expr: Orchestrator.instance(0)
readonly: no
contains: "provider":{"type":"String","value":"claude"}
contains: "elementType":"CliModel","length":1
contains: "modelId":{"type":"String","value":"opus"}
contains: "brainName":{"type":"String","value":"Claude subscription"}
contains: "timeoutMs":{"type":"Int","value":90000}
contains: "maxOutputBytes":{"type":"Int","value":1048576}
