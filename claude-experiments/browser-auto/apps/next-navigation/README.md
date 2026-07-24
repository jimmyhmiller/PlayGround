# bat vs Next.js e2e — navigation suite comparison

The app under test is `test/e2e/app-dir/navigation` from vercel/next.js,
built and served unmodified. Only the bat integration (`bat.config.json`,
`e2e/`) is checked in here; restore the app with:

    # from a vercel/next.js checkout:
    cp -R test/e2e/app-dir/navigation/{app,pages,next.config.js,middleware.js} apps/next-navigation/
    cd apps/next-navigation && npm i && npx next build && npx next start -p 3100

Then, from the repo root:

    npx tsx src/cli.ts run --config apps/next-navigation
    npx tsx src/cli.ts run --config apps/next-navigation --latency 0-1500 --seed 1

These 4 flows rewrite 6 of the suite's user-journey tests (query strings,
nested navigation, config+middleware redirects, not-found boundaries). The
other ~half of the 52-test suite asserts framework internals (RSC headers,
_rsc params, meta tags, dev overlays, server logs, React render identity) —
out of scope for a user-journey DSL by design.
