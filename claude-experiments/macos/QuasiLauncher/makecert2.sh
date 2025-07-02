#\!/bin/bash

# Create a self-signed certificate for code signing

# Step 1: Generate a private key
openssl genrsa -out private_key.pem 2048

# Step 2: Create extensions file
cat > extensions.conf << EXT
[codesign_ext]
keyUsage=digitalSignature
extendedKeyUsage=codeSigning
EXT

# Step 3: Create a certificate signing request
openssl req -new -key private_key.pem -out cert_request.csr -subj "/CN=QuasiLauncher Developer/O=QuasiLauncher/C=US"

# Step 4: Create a self-signed certificate
openssl x509 -req -in cert_request.csr -signkey private_key.pem -out certificate.pem -days 365 -extensions codesign_ext -extfile extensions.conf

# Step 5: Create a PKCS#12 bundle
openssl pkcs12 -export -out certificate.p12 -inkey private_key.pem -in certificate.pem -name "QuasiLauncher Developer" -passout pass:test

# Step 6: Import into keychain
security import certificate.p12 -k ~/Library/Keychains/login.keychain-db -P test -A

# Step 7: Trust the certificate
security add-trusted-cert -d -r trustRoot -k ~/Library/Keychains/login.keychain-db certificate.pem

# Clean up temporary files
rm private_key.pem cert_request.csr certificate.pem certificate.p12 extensions.conf

echo "Certificate 'QuasiLauncher Developer' created and imported\!"
