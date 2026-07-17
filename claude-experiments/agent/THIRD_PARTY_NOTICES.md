# Third-party notices

## Vercel Labs Native SDK

Flowline's text-entry editing model and interaction behavior in `src/main.coil`
are adapted from the Native SDK text interaction engine and AI chat composer:

- <https://github.com/vercel-labs/native/blob/main/src/primitives/canvas/text_interaction.zig>
- <https://github.com/vercel-labs/native/blob/main/src/primitives/canvas/events.zig>
- <https://github.com/vercel-labs/native/tree/main/examples/ai-chat-ts>

The original work is provided by Vercel Labs under the Apache License 2.0.
Flowline's version has been substantially modified and rewritten in Coil for a
Raylib application with centralized `AppState` transitions.

The applicable license is reproduced in
`licenses/native-sdk-Apache-2.0.txt`.
