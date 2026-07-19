# Cumulative — calorie + weight tracker

An iPhone app that tracks calories **and** weight together and continuously learns the link
between them. Unlike a daily-deficit tracker, the headline number is a **cumulative deficit
that is anchored to the scale** — so missing a day of logging is fine: the scale fills the gap
and the running total never resets.

The whole thing is built on a statistical model that is **proven against thousands of simulated
dieters** before any of it reaches the screen (water swings, missed weigh-ins, under-logging,
HealthKit noise). See "Proving it doesn't go crazy" below.

## What it does

- **Name-free, one-tap entry.** The fast path is just a number on a keypad. Names are optional.
- **Shortcuts** for common foods (tap to add, "+ save" to create from the current amount).
- **Cumulative deficit, scale-anchored.** `deficit = (trendStart − trendNow) × 3500 kcal`. Days
  since your last weigh-in get a model-based projection that **snaps back to the scale** the
  moment you step on it.
- **Two TDEEs.**
  - *effective TDEE* — the intake level, in **your own logged units**, that holds weight steady.
    The daily budget is built from this, so it self-corrects for systematic over/under-logging
    without needing to know the bias.
  - *true TDEE* — your actual burn from Apple Health (active + resting energy).
  - Their ratio is your **logging bias** (e.g. "you under-log by 8%"). HealthKit is the external
    anchor that makes bias identifiable at all; without it the app honestly says "unknown".
- **Weight trend** that filters water/sodium/glycogen swings, with an **honest uncertainty band**.
- **Goal + ETA** projected from the trend's actual rate, adjusting over time.
- **Apple Health**: reads active + resting energy and imports smart-scale weigh-ins; can write
  weigh-ins and logged calories back. Entirely optional — the app works fully without it.

## Architecture

```
Package.swift              SwiftPM — shared, headless, fully testable
Sources/CalorieModel/      the model: trend filter, TDEE+bias estimator, reconciliation, budget
Sources/ScenarioRunner/    Monte-Carlo robustness harness (run it: see below)
Tests/CalorieModelTests/   XCTest invariants
App/                       SwiftUI iPhone app (Today / Weight / Trends + entry sheets)
App/project.yml            xcodegen project definition (the .xcodeproj is generated, not committed)
```

The same `CalorieModel` library powers the headless scenario runner, the unit tests, and the app.

### The model, and why it's robust

1. **Trend weight** — a local-linear-trend (level + slope) Kalman filter run as an RTS smoother.
   Modelling the slope means a steady diet is tracked with no lag and the loss *rate* comes for
   free. The backward pass denoises the start anchor (critical for a start→end deficit). During
   weigh-in gaps the level extrapolates along the slope with variance that grows until the next
   reading. Measurement (water) noise is estimated per-user from the data, so the uncertainty band
   self-calibrates. A `+5 lb` single-day water spike moves the trend less than `~0.7 lb`.
2. **Cumulative deficit** is scale-anchored, not log-anchored — that's what makes missing a day OK.
3. **Budget** is in *as-logged* units (from effective TDEE), so you log straight against it.
4. **Goal / ETA** from the trend's real rate.

### Proving it doesn't go crazy

`ScenarioRunner` generates known ground-truth weight trajectories, then corrupts them realistically
(two-period water sinusoids + AR(1) noise + sodium spikes, missing weigh-ins, missing logs,
under-logging bias, HealthKit noise) across 4000 randomized dieters, plus targeted stress tests.
It gates on what the model can actually control:

- no NaN / absurd budgets — ever
- the reported ±2σ deficit band covers the truth **≥90%** of the time (honesty)
- the deficit is **unbiased** (|mean error| < 0.2 lb)
- trend RMSE p95 < 1.4 lb; deficit |error| p95 < 4 lb (scale-noise-limited)
- effective-TDEE p95 error < 320 kcal; bias identifiable with HealthKit

```bash
swift run -c release ScenarioRunner   # prints the full robustness report
swift test                            # the same invariants as XCTest
```

## Building the app

```bash
cd App
xcodegen generate                     # regenerate CumulativeTracker.xcodeproj from project.yml
open CumulativeTracker.xcodeproj      # then run on a device/simulator from Xcode
```

To set your Apple Developer team for on-device runs, set `DEVELOPMENT_TEAM` in `App/project.yml`
(HealthKit needs a provisioning profile on a real device; the simulator runs without it).

Debug launch arguments (used for QA/screenshots): `--demo`, `--reset`,
`--tab=weight|trends`, `--entry`.
