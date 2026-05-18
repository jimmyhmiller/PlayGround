# Releasing Ease

Ease ships through two channels with separate build pipelines:

| Channel | Script | Signing | Updates | Distribution |
|---|---|---|---|---|
| Mac App Store | `build-mas.sh` | `3rd Party Mac Developer Application` + `3rd Party Mac Developer Installer` | App Store | `.pkg` via Transporter |
| Website / Lemon Squeezy | `build-direct.sh` | `Developer ID Application` (notarized) | Sparkle | `.dmg` |

`install.sh` and `build-app.sh` remain for unsigned local development only — do not use them for distribution builds.

---

## One-time setup (Apple)

These steps are required before either build script will succeed. They must be done by hand at developer.apple.com / App Store Connect.

### Certificates

In **Keychain Access → Certificate Assistant → Request a Certificate From a CA**, generate a CSR for each of the three certs and upload at https://developer.apple.com/account/resources/certificates :

1. **Developer ID Application** — needed for the website build.
2. **Apple Distribution** (or `3rd Party Mac Developer Application`) — needed for the App Store build.
3. **3rd Party Mac Developer Installer** — signs the `.pkg`.

Verify with:

```bash
security find-identity -v -p codesigning
```

### App Store Connect record

1. Sign in at https://appstoreconnect.apple.com → **My Apps → +**.
2. Platform: macOS · Name: `Ease` · Primary Language: English · Bundle ID: `com.jimmyhmiller.Ease` (register at https://developer.apple.com/account/resources/identifiers if not already) · SKU: `ease-macos-001`.
3. Under **App Information** set Primary Category: `Productivity`. Set the support URL (your website) and privacy policy URL.
4. Under **Pricing and Availability**, set the tier (or Free if applicable).
5. Under **App Privacy**, declare CloudKit usage:
   - Data Linked to You: *User Content* (the goals/entries) — purpose: App Functionality.
   - Data Used to Track You: None.

### Provisioning profile

1. https://developer.apple.com/account/resources/identifiers — confirm `com.jimmyhmiller.Ease` has CloudKit + Push Notifications enabled and uses the `iCloud.com.jimmyhmiller.Ease` container.
2. https://developer.apple.com/account/resources/profiles → **+** → *Mac App Store Distribution*. Select the app id, the Apple Distribution cert. Download the `.provisionprofile`.

### CloudKit production schema

CloudKit schemas live separately in Development and Production. The Direct build (`aps-environment = production`) and the MAS build both require the schema to exist in Production.

1. https://icloud.developer.apple.com → container `iCloud.com.jimmyhmiller.Ease` → **Deploy Schema Changes**.
2. Push Dev → Production (one-way). Do this once the schema is stable.
3. Confirm the `Goal` and `Entry` record types and the `EaseZone` zone exist in Production.

### Notarization credentials

1. Generate an app-specific password at https://appleid.apple.com → Sign-In and Security → App-Specific Passwords.
2. Store the password in your shell environment for `build-direct.sh` (see below).

---

## Building the Mac App Store release

```bash
export MAS_APP_IDENTITY="3rd Party Mac Developer Application: Jimmy Miller (7J8U597P7P)"
export MAS_INSTALLER_IDENTITY="3rd Party Mac Developer Installer: Jimmy Miller (7J8U597P7P)"
export MAS_PROVISION_PROFILE="$HOME/Downloads/Ease_MAS_Distribution.provisionprofile"
./build-mas.sh
```

This produces `Ease.pkg`. Upload via **Transporter.app** (free, from the Mac App Store). Once it shows up under *Activity* in App Store Connect, attach it to a version, fill in screenshots / description / app review notes, and submit.

**App Review notes** should include:
- "Ease is a menu bar app (LSUIElement). After launch, the icon appears in the macOS menu bar."
- A short note about CloudKit being optional (the app works without iCloud).
- That clearing data requires holding ⌘⌥ in the context menu (intentional).

**Screenshots**: Required at 2880×1800 or 2560×1600. Take them with the popover open against a clean menu bar.

---

## Building the website / Lemon Squeezy release

```bash
export DEVELOPER_ID_APPLICATION="Developer ID Application: Jimmy Miller (7J8U597P7P)"
export APPLE_ID="jimmyhmiller@gmail.com"
export APPLE_TEAM_ID="7J8U597P7P"
export APPLE_APP_PASSWORD="xxxx-xxxx-xxxx-xxxx"   # app-specific password
./build-direct.sh
```

Produces `Ease.dmg`, notarized and stapled. To verify:

```bash
spctl -a -vvv -t install Ease.dmg
# should print: source=Notarized Developer ID
```

### Publishing the appcast (Sparkle)

After building the DMG:

```bash
# Generate the appcast from the Sparkle release tools
.build/checkouts/Sparkle/bin/generate_appcast \
    --download-url-prefix https://github.com/jimmyhmiller/ease/releases/latest/download/ \
    .   # the directory containing Ease.dmg
```

This produces `appcast.xml` next to the DMG. Then create a GitHub release tagged `v1.0.0` and upload both `Ease.dmg` and `appcast.xml`. Sparkle inside the app fetches `https://github.com/jimmyhmiller/ease/releases/latest/download/appcast.xml` (see `Info.Direct.plist`).

**Sparkle EdDSA keys** must already be set up (the public key is baked into `Info.Direct.plist` as `SUPublicEDKey`). The private key lives in the Keychain on whichever machine signs releases. To re-derive on a new machine: `.build/checkouts/Sparkle/bin/generate_keys`.

### Lemon Squeezy

1. https://app.lemonsqueezy.com → **Products → New** → digital product.
2. Upload `Ease.dmg` as the downloadable file (or link to the GitHub release URL).
3. Enable license keys if you want to gate "Pro" features later. The endpoint is `https://api.lemonsqueezy.com/v1/licenses/activate` (POST with `license_key` + `instance_name`).
4. Embed the Lemon Squeezy checkout link in the website Buy button.

---

## Per-release checklist

For every shipped version:

- [ ] `./bump-version.sh 1.0.1` — updates both plists atomically. (Manual edits drift; the MAS rejects any submission whose `CFBundleVersion` doesn't increase.)
- [ ] Update release notes (changelog) — for the App Store *What's New* field and the Sparkle appcast `<description>`.
- [ ] Run `./build-direct.sh` and verify `spctl -a -vvv -t install Ease.dmg` reports `Notarized Developer ID`.
- [ ] Run `./build-mas.sh` and dry-run-validate with `xcrun altool --validate-app -f Ease.pkg -t macos -u APPLE_ID -p APPLE_APP_PASSWORD` (or use Transporter's Verify).
- [ ] Tag the commit (e.g. `v1.0.1`), push, attach DMG + appcast.xml to a GitHub release.
- [ ] Upload PKG via Transporter, submit in App Store Connect.

## Local sandbox smoke test

The MAS build requires a real Mac App Store provisioning profile, which only Apple Developer Portal can issue. Until you have one, you can run a *partial* sandbox test:

```bash
./build-sandbox-smoketest.sh
open Ease-Sandbox.app
```

This signs the MAS variant with your local Development cert and stripped-down entitlements (sandbox + network + file picker, no CloudKit/APS, since the dev profile doesn't grant those). It catches structural failures (missing files, unresolved imports, sandbox-incompatible API usage) but **cannot** validate CloudKit / push under sandbox — that requires the real MAS profile. Run this before every MAS submission to catch the obvious stuff.

## Lemon Squeezy license activation

The Direct build now includes a license entry sheet (right-click menu bar → *Enter License…*). It calls `api.lemonsqueezy.com/v1/licenses/activate` with the user's key and a stable machine instance name (IOPlatformUUID), then stores the activation in the Keychain.

To finish wiring it up:

1. In `Ease/Views/LicenseSheet.swift`, replace `https://your-lemon-squeezy-link.example.com` with your real Lemon Squeezy product URL.
2. Decide your enforcement policy. The current implementation only stores the activation — it does *not* block the app from running without a license. To gate the app, check `LicenseManager.shared.state` at startup and present the sheet immediately if `.unactivated`.
3. Optionally call `LicenseManager.shared.validate()` on a weekly cadence to catch revoked licenses (refunds, charge-backs). The implementation is forgiving on network failure (keeps the user activated when offline).

---

## Decisions left to confirm

- **Pricing**: same on MAS and website, or website is cheaper to absorb Apple's 15-30% cut? The current code has no paywall — both builds ship the full feature set.
- **License gating**: if you later want Lemon Squeezy license activation on the website build, add a `LicenseManager` that calls `licenses/activate` and store the activation in the Keychain. (Not implemented yet.)
- **Privacy policy URL**: required by both Apple and Lemon Squeezy. The website scaffold under `website/privacy.html` is a starting point — fill in your real terms.
- **Marketing site copy**: `website/index.html` is a minimal scaffold. Replace placeholder copy, screenshots, and the Lemon Squeezy checkout link before pointing a domain at it.
