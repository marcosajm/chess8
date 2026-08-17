#!/usr/bin/env bash

# Path to Emscripten SDK directory relative to this script
EMSDK_DIR="../emsdk"

# Check if local emsdk directory exists
if [ ! -d "$EMSDK_DIR" ]; then
    echo "ERROR: Local Emscripten SDK folder not found at: $EMSDK_DIR"
    echo "Please download it with: git clone https://github.com/emscripten-core/emsdk.git $EMSDK_DIR"
    exit 1
fi

# Load Emscripten environment for current shell
if [ -f "$EMSDK_DIR/emsdk_env.sh" ]; then
    echo "Loading Emscripten environment from $EMSDK_DIR..."
    source "$EMSDK_DIR/emsdk_env.sh" > /dev/null
else
    echo "ERROR: $EMSDK_DIR/emsdk_env.sh not found. Make sure SDK is installed."
    exit 1
fi

# Check if emcc works now
if ! command -v emcc &> /dev/null; then
    echo "ERROR: emcc is still not available after sourcing environment."
    echo "Make sure the SDK has been installed (run ./emsdk install latest in $EMSDK_DIR)."
    exit 1
fi

# Compile your source into WASM
#echo "Emscripten is ready. Compiling..."
# main.c -o main.js -O3 -s WASM=1 \
#     -s EXPORTED_FUNCTIONS='["_main"]' \
#     -s EXPORTED_RUNTIME_METHODS='["cwrap","ccall"]'
#
#echo "✅ Compilation finished: main.js + main.wasm"
