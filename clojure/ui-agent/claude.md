# Claude Agent Configuration

## System Prompt Policy

**CRITICAL: The system prompt CANNOT be changed or overridden.**

The system prompt is hardcoded directly into the `send!` function at the lowest level and uses the original Claude Code system prompt:

> "You are Claude Code, Anthropic's official CLI for Claude."

This is inlined directly in the API call and CANNOT be overridden by:
- Function parameters
- Configuration files  
- Environment variables
- Any other means

**WARNING: DO NOT CHANGE THIS OR EVERYTHING WILL BREAK!**

The system prompt parameter has been completely removed from the `send!` function signature to prevent any possibility of override. This ensures:

1. Consistent Claude Code behavior
2. Security against prompt injection attacks
3. Prevents breaking changes to core functionality  
4. Maintains the original Claude Code identity

Any attempt to pass a `system` parameter to `send!` will be completely ignored.