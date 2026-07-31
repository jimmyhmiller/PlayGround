file: examples/assistant.scry
stdin: /models\n5\nstatus\nexit\n
contains: Select model
contains: Fable 5
contains: Opus
contains: Sonnet
contains: GPT-5.6 Sol
contains: GPT-5.6 Terra
contains: GPT-5.6 Luna
contains: model switched to Codex subscription
contains: model:        gpt-5.6-terra
notcontains: [gpt-5.6-terra]
contains: goodbye
