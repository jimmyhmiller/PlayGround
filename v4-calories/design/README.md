# Design source

The visual design this app implements lives in the Claude Design project:

- Project: **Cumulative calorie and weight tracker**
  https://claude.ai/design/p/e0c7bf11-085f-4a08-87b6-536d86f7fdcf
- File: `Cumulative Tracker.dc.html`

## Tokens (extracted into `App/Sources/Theme.swift`)

- Background `#0A0A0B`, sheet `#0E0E10`, keypad key `#191A1D`
- Accent green `#7DD3A8`, amber `#F0A878`, on-green text `#08130D`
- Primary text `#F3F3F5`; secondary `rgba(235,235,245, α)`
- Numbers use JetBrains Mono in the design → SF Mono (`.monospaced`) in the app

## Screens

- **Today** — running calories in / budget, scale-anchored cumulative deficit (`✓ MATCHES SCALE`),
  projected loss vs plan, "The Model" (TDEE ± band, logging bias), entry list + quick add.
- **Weight** — trend weight + rate, reconciliation chart (trend / scale dots / logs-only),
  reconciliation breakdown, recent weigh-ins with deviation-from-trend.
- **Trends** — goal progress + ETA, cumulative-deficit chart, converging TDEE band, logging-bias.
- **Sheets** — numeric keypad quick entry with shortcuts; weigh-in keypad.
