#include "nnue.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Global instance
NNUEWeights g_nnue;

// ============== Feature extraction (UNCHANGED) ==============
void featurize_board(const Position *pos, float *out_features) {
    memset(out_features, 0, sizeof(float) * NNUE_INPUT_DIM);

    for (int side = 0; side < 2; side++) {
        for (int piece = 0; piece < PIECE_NB; piece++) {
            uint64_t bb = pos->pieces[side][piece];
            while (bb) {
                int sq = __builtin_ctzll(bb);
                bb &= bb - 1;
                int idx = side * (PIECE_NB * 64) + piece * 64 + sq;
                out_features[idx] = 1.0f;
            }
        }
    }
}

static inline float relu(float x) { return x > 0 ? x : 0.0f; }

// ============== Dimensions - UPDATED for 4 layers ==============
static const size_t W1_F = (size_t)NNUE_H1 * NNUE_INPUT_DIM;    // 256 * 780
static const size_t B1_F = (size_t)NNUE_H1;                     // 256
static const size_t W2_F = (size_t)NNUE_H2 * NNUE_H1;           // 64 * 256   (CHANGED)
static const size_t B2_F = (size_t)NNUE_H2;                     // 64         (CHANGED)
static const size_t W3_F = (size_t)NNUE_H3 * NNUE_H2;           // 32 * 64    (NEW)
static const size_t B3_F = (size_t)NNUE_H3;                     // 32         (NEW)
static const size_t W4_F = (size_t)NNUE_OUT * NNUE_H3;          // 1 * 32     (NEW)
static const size_t B4_F = (size_t)NNUE_OUT;                    // 1          (NEW)

// TOTAL_F changes from 200,705 to 218,305
static const size_t TOTAL_F = W1_F + B1_F + W2_F + B2_F + W3_F + B3_F + W4_F + B4_F;
static const size_t TOTAL_BYTES = TOTAL_F * sizeof(float);

// ============== Memory management - UPDATED ==============
void nnue_free(NNUEWeights *nn) {
    if (!nn) return;
    free(nn->w1); free(nn->b1);
    free(nn->w2); free(nn->b2);
    free(nn->w3); free(nn->b3);
    free(nn->w4); free(nn->b4);  // NEW
    memset(nn, 0, sizeof(*nn));
}

static bool alloc_layers(NNUEWeights *nn) {
    memset(nn, 0, sizeof(*nn));
    nn->w1 = malloc(W1_F * sizeof(float));
    nn->b1 = malloc(B1_F * sizeof(float));
    nn->w2 = malloc(W2_F * sizeof(float));
    nn->b2 = malloc(B2_F * sizeof(float));
    nn->w3 = malloc(W3_F * sizeof(float));  // NEW
    nn->b3 = malloc(B3_F * sizeof(float));  // NEW
    nn->w4 = malloc(W4_F * sizeof(float));  // NEW
    nn->b4 = malloc(B4_F * sizeof(float));  // NEW
    return nn->w1 && nn->b1 && nn->w2 && nn->b2 && 
           nn->w3 && nn->b3 && nn->w4 && nn->b4;  // UPDATED
}

static bool load_into_layers(NNUEWeights *nn, const float *src, size_t floats) {
    if (floats != TOTAL_F) {
       fprintf(stderr, "[NNUE] Size mismatch: got %zu, expected %zu\n", floats, TOTAL_F);
        return false;
    }
    
    size_t off = 0;
    // Layer 1: w1 [256 x 780], b1 [256]
    memcpy(nn->w1, src + off, W1_F * sizeof(float)); off += W1_F;
    memcpy(nn->b1, src + off, B1_F * sizeof(float)); off += B1_F;
    
    // Layer 2: w2 [64 x 256], b2 [64]     (CHANGED)
    memcpy(nn->w2, src + off, W2_F * sizeof(float)); off += W2_F;
    memcpy(nn->b2, src + off, B2_F * sizeof(float)); off += B2_F;
    
    // Layer 3: w3 [32 x 64], b3 [32]      (NEW)
    memcpy(nn->w3, src + off, W3_F * sizeof(float)); off += W3_F;
    memcpy(nn->b3, src + off, B3_F * sizeof(float)); off += B3_F;
    
    // Layer 4: w4 [1 x 32], b4 [1]         (NEW)
    memcpy(nn->w4, src + off, W4_F * sizeof(float)); off += W4_F;
    memcpy(nn->b4, src + off, B4_F * sizeof(float)); off += B4_F;
    
    nn->loaded = true;
    return true;
}

// ============== Load from .bin file (UNCHANGED) ==============
bool nnue_load_weights_from_file(NNUEWeights *nn, const char *path) {
    if (!nn || !path) return false;

    FILE *fp = fopen(path, "rb");
    if (!fp) { 
       fprintf(stderr, "[NNUE] Failed to open %s\n", path); 
        return false; 
    }

    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (sz != (long)TOTAL_BYTES) {
       fprintf(stderr, "[NNUE] File size mismatch: got %ld, expected %zu\n", 
                sz, TOTAL_BYTES);
        fclose(fp);
        return false;
    }

    float *buf = malloc(TOTAL_BYTES);
    if (!buf) { 
        fclose(fp); 
        return false; 
    }
    
    size_t rd = fread(buf, 1, TOTAL_BYTES, fp);
    fclose(fp);

    if (rd != TOTAL_BYTES) { 
        free(buf); 
        return false; 
    }
    
    bool ok = alloc_layers(nn) && load_into_layers(nn, buf, TOTAL_F);
    free(buf);

    if (!ok) nnue_free(nn);
    return ok;
}

// ============== Forward passes - UPDATED for 4 layers ==============
float nnue_forward_vec(const NNUEWeights *nn, const float *x) {
    if (!nn || !nn->loaded) return 0.0f;

    // Layer 1: 780 -> 256
    float h1[NNUE_H1];
    for (int o = 0; o < NNUE_H1; o++) {
        float sum = nn->b1[o];
        const float *wrow = nn->w1 + o * NNUE_INPUT_DIM;
        for (int i = 0; i < NNUE_INPUT_DIM; i++) {
            sum += wrow[i] * x[i];
        }
        h1[o] = relu(sum);
    }

    // Layer 2: 256 -> 64 (CHANGED: H1->H2)
    float h2[NNUE_H2];
    for (int o = 0; o < NNUE_H2; o++) {
        float sum = nn->b2[o];
        const float *wrow = nn->w2 + o * NNUE_H1;
        for (int i = 0; i < NNUE_H1; i++) {
            sum += wrow[i] * h1[i];
        }
        h2[o] = relu(sum);
    }

    // Layer 3: 64 -> 32 (NEW)
    float h3[NNUE_H3];
    for (int o = 0; o < NNUE_H3; o++) {
        float sum = nn->b3[o];
        const float *wrow = nn->w3 + o * NNUE_H2;
        for (int i = 0; i < NNUE_H2; i++) {
            sum += wrow[i] * h2[i];
        }
        h3[o] = relu(sum);
    }

    // Layer 4: 32 -> 1 (NEW - replaces old output layer)
    float out = nn->b4[0];
    for (int i = 0; i < NNUE_H3; i++) {
        out += nn->w4[i] * h3[i];
    }

    return out;
}

float nnue_forward_pos(const NNUEWeights *nn, const Position *pos) {
    float feat[NNUE_INPUT_DIM];
    featurize_board(pos, feat);
    return nnue_forward_vec(nn, feat);
}

// ============== Debug/Info function (NEW) ==============
/* void nnue_print_info(void) {
    printf("=== NNUE Network Information ===\n");
    printf("Architecture: %d -> %d -> %d -> %d -> %d\n", 
           NNUE_INPUT_DIM, NNUE_H1, NNUE_H2, NNUE_H3, NNUE_OUT);
    printf("Layer 1: %d -> %d (%zu weights, %zu biases)\n", 
           NNUE_INPUT_DIM, NNUE_H1, W1_F, B1_F);
    printf("Layer 2: %d -> %d (%zu weights, %zu biases)\n", 
           NNUE_H1, NNUE_H2, W2_F, B2_F);
    printf("Layer 3: %d -> %d (%zu weights, %zu biases)\n", 
           NNUE_H2, NNUE_H3, W3_F, B3_F);
    printf("Layer 4: %d -> %d (%zu weights, %zu biases)\n", 
           NNUE_H3, NNUE_OUT, W4_F, B4_F);
    printf("Total parameters: %zu\n", TOTAL_F);
    printf("Total bytes: %zu (%.2f MB)\n", TOTAL_BYTES, 
           (float)TOTAL_BYTES / (1024 * 1024));
    printf("Loaded: %s\n", g_nnue.loaded ? "Yes" : "No");
} */