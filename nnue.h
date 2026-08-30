#ifndef NNUE_H
#define NNUE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "bitboard.h"   // for Position

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif

// === Network dimensions - ENHANCED VERSION ===
enum { 
    NNUE_INPUT_DIM = 780,    // feature size (unchanged)
    NNUE_H1 = 256,           // unchanged
    NNUE_H2 = 64,            // CHANGED: 32 -> 64 (better defense)
    NNUE_H3 = 32,            // NEW: added 3rd hidden layer
    NNUE_OUT = 1             // unchanged
};

// --- NNUE weights for 4 layers ---
// Layer 1: [256 x 780], bias [256]
// Layer 2: [64  x 256], bias [64]      <- CHANGED
// Layer 3: [32  x 64 ], bias [32]      <- NEW
// Layer 4: [1   x 32 ], bias [1]       <- NEW
typedef struct {
    float *w1; float *b1;    // Layer 1: 780 -> 256
    float *w2; float *b2;    // Layer 2: 256 -> 64
    float *w3; float *b3;    // Layer 3: 64 -> 32    (NEW)
    float *w4; float *b4;    // Layer 4: 32 -> 1     (NEW)
    bool   loaded;
} NNUEWeights;

// Global weights instance
extern NNUEWeights g_nnue;

// --- API (mostly unchanged) ---
void featurize_board(const Position *pos, float *out_features);
bool nnue_load_weights_from_file(NNUEWeights *nn, const char *path);
void nnue_free(NNUEWeights *nn);

float nnue_forward_vec(const NNUEWeights *nn, const float *x);
float nnue_forward_pos(const NNUEWeights *nn, const Position *pos);

// Optional: Add info function for debugging
//void nnue_print_info(void);

#endif