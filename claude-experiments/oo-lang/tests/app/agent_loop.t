file: examples/assistant.scry
stdin: what is 17 times 23?\nweather in Tokyo?\nexit\n
contains: brain: ScriptedModel - offline
contains: [agent] -> tool_use: calculate({"a":17,"b":23,"op":"mul"})
contains: [agent] <- tool_result: calculate => 17 * 23 = 391
contains: Here you go: 17 * 23 = 391
contains: [agent] -> tool_use: get_weather({"location":"Tokyo"})
contains: Tokyo: 18C, cloudy
contains: goodbye
