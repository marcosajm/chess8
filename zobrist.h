#ifndef ZOBRIST_H
#define ZOBRIST_H

#include <stdint.h>
#include "bitboard.h"

#define NO_ENPASSANT   (-1)

// Zobrist random key tables
extern uint64_t zobrist_piece[2][PIECE_NB][64];
extern uint64_t zobrist_castling[16];   // 4-bit castling rights, so 16 combos
extern uint64_t zobrist_enpassant[64];  // only file of en-passant target is relevant
extern uint64_t zobrist_side;           // side to move

// Init tables with random values
void init_zobrist(void);

void update_zobrist_key(Position *pos);

// Compute full hash from scratch for a position
//uint64_t compute_hash(const Position *pos);

// Incremental XOR helpers
static inline uint64_t xor_piece(uint64_t hash, Side color, PieceType piece, int square) {
    return hash ^ zobrist_piece[color][piece][square];
}

static inline uint64_t xor_castling(uint64_t hash, int castling_rights) {
    return hash ^ zobrist_castling[castling_rights & 0xF];
}

static inline uint64_t xor_enpassant(uint64_t hash, int square) {
    return (square >= 0 && square < 64) ? hash ^ zobrist_enpassant[square] : hash;
}

static inline uint64_t xor_side(uint64_t hash) {
    return hash ^ zobrist_side;
}

#endif // ZOBRIST_H
