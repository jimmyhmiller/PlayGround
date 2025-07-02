# Creating a Code Signing Certificate for QuasiLauncher

The app needs to be properly signed (not ad-hoc) for macOS to recognize accessibility permissions.

## Option 1: Using Keychain Access (Recommended)

1. Open **Keychain Access** (in /Applications/Utilities/)

2. Menu: **Keychain Access** > **Certificate Assistant** > **Create a Certificate...**

3. Fill in the form:
   - **Name**: `QuasiLauncher Developer`
   - **Identity Type**: `Self Signed Root`
   - **Certificate Type**: `Code Signing`
   - ✅ Check **"Let me override defaults"**

4. Click **Continue** through all the screens (accept all defaults)

5. When done, the certificate will be in your login keychain

## Option 2: Command Line

Run this in Terminal:

```bash
# Create certificate
security create-keychain-certificate \
    -t codeSign \
    -s "QuasiLauncher Developer" \
    -a "$(whoami)@$(hostname)" \
    -k ~/Library/Keychains/login.keychain-db \
    -r Self-Signed \
    -A

# Trust it
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "" ~/Library/Keychains/login.keychain-db
```

## After Creating the Certificate

1. Run `./build.sh` again - it will now use the proper certificate
2. You may need to remove and re-add QuasiLauncher in System Preferences permissions
3. The app should now properly detect granted permissions

## Why This Matters

- Ad-hoc signed apps (`codesign -s -`) don't work properly with accessibility APIs
- macOS requires a proper code signing identity for permission detection to work
- Even a self-signed certificate is better than ad-hoc signing