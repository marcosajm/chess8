# === Configuration ===
# Source files (adjust based on whether engine.c and zobrist.c exist)
SRC_C = bitboard.c movegen.c search.c engine.c zobrist.c fen.c utilities.c occupancy.c nnue.c uci.c
SRC_CT = main.c bitboard.c movegen.c search.c engine.c zobrist.c fen.c utilities.c occupancy.c uci.c

# Output targets
TARGET_WASM = ./public/main
TARGET_NATIVE = chess_engine

# Emscripten settings
EMSDK_DIR = ../emsdk
EMCC = emcc
EMCC_FLAGS = -O0 -std=c23 -s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=256MB -s MAXIMUM_MEMORY=2048MB --preload-file public/nnue.bin@nnue.bin \
             -s EXPORTED_FUNCTIONS="['_init_board_wasm', '_get_board_ptr_wasm', '_make_move_wasm', '_get_current_turn_wasm', '_find_ai_move_wasm', '_find_ai_move_wasm_depth', '_promote_pawn_wasm', '_get_pawn_promotion_pending_index_wasm', '_get_game_state_wasm', '_evaluate_board', '_get_empty']" \
             -s EXPORTED_RUNTIME_METHODS="['cwrap', 'ccall', 'getValue', 'setValue', 'HEAP8']" \
             -s ENVIRONMENT=web -s MODULARIZE=1 -s EXPORT_NAME=ChessModule -g

# Native compiler settings
GCC = gcc
GCC_FLAGS = -std=c23 -O2

# === Phony targets ===
.PHONY: all wasm wasm_build native clean

# === Default target ===
all: wasm

# === WebAssembly build === source ../emsdk/emsdk_env.sh
wasm:
	@echo "🔹 Checking for Emscripten..."
	@if ! command -v $(EMCC) >/dev/null 2>&1; then \
		if [ -f "$(EMSDK_DIR)/emsdk_env.sh" ]; then \
			echo "🔹 Loading Emscripten from $(EMSDK_DIR)"; \
			. "$(EMSDK_DIR)/emsdk_env.sh"; \
			$(MAKE) wasm_build; \
		else \
			echo "❌ Emscripten SDK not found in $(EMSDK_DIR)"; \
			echo "Please install: git clone https://github.com/emscripten-core/emsdk.git $(EMSDK_DIR)"; \
			echo "Then run: $(EMSDK_DIR)/emsdk install latest && $(EMSDK_DIR)/emsdk activate latest"; \
			exit 1; \
		fi \
	else \
		$(MAKE) wasm_build; \
	fi

wasm_build:
	@echo "⚙️  Compiling WebAssembly: $(SRC_C) → $(TARGET_WASM).wasm"
	$(EMCC) $(SRC_C) -o $(TARGET_WASM).js $(EMCC_FLAGS)
	@echo "✅ WebAssembly build complete: $(TARGET_WASM).js, $(TARGET_WASM).wasm"

# === Native build ===
native:
	@echo "⚙️  Compiling native binary: $(SRC_CT) → $(TARGET_NATIVE)"
	$(GCC) $(SRC_CT) -o $(TARGET_NATIVE) $(GCC_FLAGS)
	@echo "✅ Native build complete: ./$(TARGET_NATIVE)"

# === Clean ===
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -f $(TARGET_WASM).js $(TARGET_WASM).wasm $(TARGET_NATIVE)
	@echo "✅ Clean complete"