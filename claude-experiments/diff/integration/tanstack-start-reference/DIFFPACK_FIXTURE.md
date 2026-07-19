# Diffpack TanStack Start reference fixture

This fixture was imported from TanStack Router's official `start-basic`
example at commit `00d00b3155e6e8bdbdae806ff12de129c4915d86` on 2026-07-17.

Source:
<https://github.com/TanStack/router/tree/00d00b3155e6e8bdbdae806ff12de129c4915d86/examples/react/start-basic>

Package versions are exact rather than ranges so this remains a stable
acceptance target. The reference build uses the officially supported Vite,
TanStack Start, React, Tailwind, and Nitro plugins. Diffpack is expected to
eventually replace that build execution path while preserving the observable
client/server behavior and production artifact contracts.

Do not casually update this fixture. Version updates must regenerate the npm
lockfile, reference artifact inventory, and acceptance expectations together.
