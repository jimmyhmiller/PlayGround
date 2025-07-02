#\!/bin/bash

# Alternative method: Create certificate using security command line
CERT_NAME="QuasiLauncher Developer"

# Generate a private key
openssl genrsa -out temp_key.pem 2048

# Create a self-signed certificate
openssl req -new -x509 -key temp_key.pem -out temp_cert.pem -days 365 -subj "/CN=$CERT_NAME/O=QuasiLauncher"

# Convert to PKCS12 format
openssl pkcs12 -export -out temp_cert.p12 -inkey temp_key.pem -in temp_cert.pem -passout pass:

# Import into keychain
security import temp_cert.p12 -k ~/Library/Keychains/login.keychain-db -P "" -T /usr/bin/codesign

# Clean up temp files
rm temp_key.pem temp_cert.pem temp_cert.p12

# Trust the certificate for code signing
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "" ~/Library/Keychains/login.keychain-db

echo "Certificate created successfully\!"
