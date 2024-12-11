#!/bin/bash

# 1. Set up writable directories for Rust
export CARGO_HOME="$HOME/.cargo"
export RUSTUP_HOME="$HOME/.rustup"
export PATH="$CARGO_HOME/bin:$PATH"

# 2. Install Rust toolchain (locally)
echo "Installing Rust toolchain..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 3. Ensure Rust environment variables are loaded
source "$HOME/.cargo/env"

# 4. Upgrade pip and install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install maturin --no-cache-dir
pip install --only-binary :all: -r requirements.txt
