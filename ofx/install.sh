#!/bin/bash
# Nimbus Diffusion — build + install for macOS
# Run with: sudo bash install.sh
# (sudo is needed to write to /Library/OFX/Plugins)

set -e
cd "$(dirname "$0")"

# check dependencies
for cmd in cmake git; do
    if ! command -v $cmd &>/dev/null; then
        echo "ERROR: $cmd not found. Install it first."
        echo "  cmake: https://cmake.org/download/ or: brew install cmake"
        echo "  git:   xcode-select --install"
        exit 1
    fi
done

BUILD_DIR="$(pwd)/build"
BUNDLE_SRC="$BUILD_DIR/NimbusDiffusor.ofx.bundle"
BUNDLE_DST="/Library/OFX/Plugins/NimbusDiffusor.ofx.bundle"

echo "[1/3] Configuring..."
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

echo "[2/3] Building..."
cmake --build "$BUILD_DIR" --config Release

echo "[3/3] Installing to $BUNDLE_DST ..."
mkdir -p "$BUNDLE_DST/Contents/MacOS"
mkdir -p "$BUNDLE_DST/Contents/Resources"

cp "$BUNDLE_SRC/Contents/MacOS/NimbusDiffusor.ofx" \
   "$BUNDLE_DST/Contents/MacOS/NimbusDiffusor.ofx"

cp "$(dirname "$0")/NimbusDiffusion.png" \
   "$BUNDLE_DST/Contents/Resources/NimbusDiffusion.png"

echo ""
echo "Done! Restart DaVinci Resolve."
echo "If controls look wrong, clear the OFX cache:"
echo "  Preferences > System > Memory and GPU > Clear OFX cache"
