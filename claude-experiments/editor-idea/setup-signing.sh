#!/bin/sh
# One-time setup: create a STABLE self-signed code-signing identity for
# the TerminalBevy bundle.
#
# Why: macOS TCC keys permission grants (Microphone, Documents/Desktop/
# Downloads, Full Disk Access, …) on the app's *code identity*. Ad-hoc
# signing (`codesign --sign -`) derives that identity from the binary's
# cdhash, which changes on every rebuild — so macOS treats each rebuilt
# app as brand new and forgets every grant, re-prompting "all the time".
# A self-signed identity anchored to a persistent certificate keeps the
# code identity constant across rebuilds, so grants persist.
#
# Safe to run repeatedly: it no-ops if the identity already exists. The
# private key never leaves this machine; it's only used for local signing.
set -e

IDENTITY_NAME="${TB_SIGN_IDENTITY:-TerminalBevy Local Signing}"
KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"

# No -v: a self-signed cert is Gatekeeper-untrusted (which -v filters
# out), but codesign uses it fine and TCC keys on its stable cert-leaf
# designated requirement.
if security find-identity -p codesigning 2>/dev/null | grep -qF "$IDENTITY_NAME"; then
    echo "[setup-signing] identity '$IDENTITY_NAME' already exists — nothing to do."
    exit 0
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Self-signed leaf cert with the codeSigning extended key usage. 20-year
# validity so it doesn't silently expire and break signing.
openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$TMP/key.pem" -out "$TMP/cert.pem" -days 7300 \
    -subj "/CN=$IDENTITY_NAME" \
    -addext "basicConstraints=critical,CA:FALSE" \
    -addext "keyUsage=critical,digitalSignature" \
    -addext "extendedKeyUsage=critical,codeSigning"

# Package into PKCS#12. OpenSSL 3's default PKCS12 MAC (AES/SHA-256) is
# unreadable by Apple's `security import` ("MAC verification failed").
# For OpenSSL 3+, force the legacy SHA1/3DES encoding macOS understands;
# LibreSSL (the system openssl) already defaults to it.
P12_COMPAT=""
case "$(openssl version 2>/dev/null)" in
    OpenSSL\ 3*|OpenSSL\ 4*)
        P12_COMPAT="-legacy -macalg sha1 -keypbe PBE-SHA1-3DES -certpbe PBE-SHA1-3DES"
        ;;
esac
# A non-empty transit password is required: Apple's `security` fails MAC
# verification on empty-password PKCS#12 bundles. It's only used to move
# the key from openssl into the keychain, then discarded.
P12_PASS="terminalbevy"
openssl pkcs12 -export $P12_COMPAT -out "$TMP/identity.p12" \
    -inkey "$TMP/key.pem" -in "$TMP/cert.pem" -passout "pass:$P12_PASS"

# Import the key+cert into the login keychain. -A lets codesign use the
# key without a per-build keychain-access prompt (acceptable for a local
# dev signing key). If macOS still prompts on first signing, click
# "Always Allow".
security import "$TMP/identity.p12" -k "$KEYCHAIN" -P "$P12_PASS" -A

echo "[setup-signing] created code-signing identity:"
security find-identity -p codesigning | grep -F "$IDENTITY_NAME" || true
echo "[setup-signing] done. Rebuild with ./dev-restart.sh; the bundle will sign with it."
echo "[setup-signing] First launch after this, re-grant Microphone / Full Disk Access once;"
echo "[setup-signing] the grants will then survive future rebuilds."
