# Magical `src/shared`

The contents of this folder get copied into every Lambda function `node_modules/@architect/shared` when:

- running `npm start`
- running `npm test`
- running `npm run deploy`

This means modules in this folder can be required from any function in this `@app`

For example:

```javascript
var layout = require('@architect/shared/layout')
```

Resolves locally to `src/shared/layout.js`.

## Organizing

Organize shared code however it makes sense for your project.

Here is an example starting structure:

- src/shared/views
- src/shared/middleware
- src/shared/helpers
- src/shared/components

Overwrite the contents of this file to describe the structure you've chosen.

## Use Caution!

Everything in src/shared will be copied into all Lambdas which could quickly bloat an application. Remember you want to keep your Lambda functions sub 5 MB for optimal coldstart performance.
