#!/bin/bash
set -e

REMOTE_HOST="192.168.0.55"
REMOTE_USER="${REMOTE_USER:-jimmyhmiller}"

echo "==> Installing LLVM/MLIR 20 on remote AMD machine..."
echo "==> You will be prompted for your sudo password on the remote machine"
echo ""

# Create a temporary script on the remote machine and execute it
ssh -t "${REMOTE_USER}@${REMOTE_HOST}" '
set -e

# Add LLVM APT repository for latest versions
echo "Adding LLVM apt repository..."
wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc

# Detect Ubuntu codename
CODENAME=$(lsb_release -cs)
echo "Detected Ubuntu codename: $CODENAME"

# Add repository
echo "deb http://apt.llvm.org/$CODENAME/ llvm-toolchain-$CODENAME-20 main" | sudo tee /etc/apt/sources.list.d/llvm.list

# Update package list
echo "Updating package list..."
sudo apt update

# Uninstall LLVM 18 if present
echo "Removing LLVM 18..."
sudo apt remove -y llvm-18 llvm-18-dev llvm-18-tools libmlir-18-dev mlir-18-tools 2>/dev/null || true
sudo apt autoremove -y

# Install LLVM/MLIR packages
echo "Installing LLVM 20 and MLIR..."
sudo apt install -y \
  llvm-20 \
  llvm-20-dev \
  llvm-20-tools \
  libmlir-20-dev \
  mlir-20-tools

# Check installation
echo ""
echo "Checking installation..."
llvm-config-20 --version
echo "LLVM installed at: $(llvm-config-20 --prefix)"
echo ""
echo "Installation complete!"
'

echo ""
echo "==> Done! LLVM/MLIR 20 is now installed on the remote machine."
