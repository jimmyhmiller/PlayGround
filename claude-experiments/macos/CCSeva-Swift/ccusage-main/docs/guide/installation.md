# Installation

ccusage can be installed and used in several ways depending on your preferences and use case.

## Quick Start (Recommended)

The fastest way to try ccusage is to run it directly without installation:

::: code-group

```bash [npx]
npx ccusage@latest
```

```bash [bunx]
bunx ccusage
```

```bash [pnpm]
pnpm dlx ccusage
```

```bash [deno]
deno run -E -R=$HOME/.claude/projects/ -S=homedir -N='raw.githubusercontent.com:443' npm:ccusage@latest
```

:::

This approach:

- ✅ Always uses the latest version
- ✅ No global installation required
- ✅ Works across different systems
- ✅ Perfect for occasional use

## Global Installation

If you use ccusage frequently, install it globally:

::: code-group

```bash [npm]
npm install -g ccusage
```

```bash [bun]
bun install -g ccusage
```

```bash [yarn]
yarn global add ccusage
```

```bash [pnpm]
pnpm add -g ccusage
```

:::

After global installation, run commands directly:

```bash
ccusage daily
ccusage monthly --breakdown
ccusage blocks --live
```

## Development Installation

For development or contributing to ccusage:

```bash
# Clone the repository
git clone https://github.com/ryoppippi/ccusage.git
cd ccusage

# Install dependencies
bun install

# Run directly from source
bun run start daily
bun run start monthly --json
```

### Development Scripts

```bash
# Run tests
bun run test

# Type checking
bun typecheck

# Build distribution
bun run build

# Lint and format
bun run format
```

## Runtime Requirements

### Node.js

- **Minimum**: Node.js 20.x
- **Recommended**: Node.js 20.x or later
- **LTS versions** are fully supported

### Bun (Alternative)

- **Minimum**: Bun 1.2+
- **Recommended**: Latest stable release
- Often faster than Node.js for ccusage

### Deno

Deno 2.0+ is fully supported with proper permissions:

```bash
deno run \
  -E \
  -R=$HOME/.claude/projects/ \
  -S=homedir \
  -N='raw.githubusercontent.com:443' \
  npm:ccusage@latest
```

Also you can use `offline` mode to run ccusage without network access:

```bash
deno run \
  -E \
  -R=$HOME/.claude/projects/ \
  -S=homedir \
  npm:ccusage@latest --offline
```

## Verification

After installation, verify ccusage is working:

```bash
# Check version
ccusage --version

# Run help command
ccusage --help

# Test with daily report
ccusage daily
```

## Updating

### Direct Execution (npx/bunx)

Always gets the latest version automatically.

### Global Installation

```bash
# Update with npm
npm update -g ccusage

# Update with bun
bun update -g ccusage
```

### Check Current Version

```bash
ccusage --version
```

## Uninstalling

### Global Installation

::: code-group

```bash [npm]
npm uninstall -g ccusage
```

```bash [bun]
bun remove -g ccusage
```

```bash [yarn]
yarn global remove ccusage
```

```bash [pnpm]
pnpm remove -g ccusage
```

:::

### Development Installation

```bash
# Remove cloned repository
rm -rf ccusage/
```

## Troubleshooting Installation

### Permission Errors

If you get permission errors during global installation:

::: code-group

```bash [npm]
# Use npx instead of global install
npx ccusage@latest

# Or configure npm to use a different directory
npm config set prefix ~/.npm-global
export PATH=~/.npm-global/bin:$PATH
```

```bash [Node Version Managers]
# Use nvm (recommended)
nvm install node
npm install -g ccusage

# Or use fnm
fnm install node
npm install -g ccusage
```

:::

### Network Issues

If installation fails due to network issues:

```bash
# Try with different registry
npm install -g ccusage --registry https://registry.npmjs.org

# Or use bunx for offline-capable runs
bunx ccusage
```

### Version Conflicts

If you have multiple versions installed:

```bash
# Check which version is being used
which ccusage
ccusage --version

# Uninstall and reinstall
npm uninstall -g ccusage
npm install -g ccusage@latest
```

## Next Steps

After installation, check out:

- [Getting Started Guide](/guide/getting-started) - Your first usage report
- [Configuration](/guide/configuration) - Customize ccusage behavior
- [Daily Reports](/guide/daily-reports) - Understand daily usage patterns
