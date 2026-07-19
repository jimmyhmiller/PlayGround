file: examples/assistant.scry
expr: actions()
contains: "type":"Actions"
contains: {"label":"Pause","target":"Agent","invoke":"__action_0","params":[],"returns":"Void"}
contains: {"label":"Ask","target":"Agent","invoke":"__action_2","params":[{"name":"question","type":"String"}],"returns":"Void"}
contains: {"label":"Spawn researcher","target":"Orchestrator","invoke":"__action_3","params":[{"name":"topic","type":"String"}],"returns":"Void"}
