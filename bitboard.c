//bitboard.c
#include "bitboard.h"
#include <stdio.h>
#include <stddef.h>

// Export constants
int get_empty() { return 0; }

/* ---------------- helpers ---------------- */
int pop_lsb(uint64_t *bb) {
    if (!bb || !*bb) return -1;
#if defined(__GNUC__) || defined(__clang__)
    int sq = __builtin_ctzll(*bb);
#else
    uint64_t b = *bb;
    int sq = 0;
    while ((b & 1ULL) == 0ULL) { b >>= 1; sq++; }
#endif
    *bb &= *bb - 1ULL;
    return sq;
}

int popcount64(uint64_t bb) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(bb);
#else
    int count = 0;
    while (bb) { bb &= bb - 1ULL; count++; }
    return count;
#endif
}

int lsb_index(uint64_t bb) {
    if (!bb) return -1;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(bb);
#else
    int sq = 0;
    while ((bb & 1ULL) == 0ULL) { bb >>= 1; sq++; }
    return sq;
#endif
}