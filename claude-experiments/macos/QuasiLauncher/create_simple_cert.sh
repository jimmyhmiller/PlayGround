#\!/bin/bash

# Create a simple certificate directly in keychain
CERT_NAME="QuasiLauncher Developer"

# Create a certificate request
cat > cert_config.txt << CONFIG
[ req ]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[ req_distinguished_name ]
CN = $CERT_NAME

[ v3_req ]
keyUsage = digitalSignature
extendedKeyUsage = codeSigning
CONFIG

# Generate key and certificate
openssl req -new -x509 -keyout temp_key.pem -out temp_cert.pem -days 365 -nodes -config cert_config.txt

# Convert to proper format and import
security add-certificates temp_cert.pem
security import temp_key.pem -k ~/Library/Keychains/login.keychain-db -T /usr/bin/codesign

# Clean up
rm temp_key.pem temp_cert.pem cert_config.txt

echo "Certificate creation attempted"
