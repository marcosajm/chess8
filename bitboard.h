#ifndef BITBOARD_H
#define BITBOARD_H

#pragma once
#include <stdint.h>

/* === Bitboard definitions === */
typedef uint64_t Bitboard;

// Bitboard helper macros
#define BIT(sq) (1ULL << (sq))

/*
 * Square indexing (little-endian rank-file mapping):
 *   a1 = 0, b1 = 1, ..., h1 = 7
 *   a2 = 8, b2 = 9, ..., h8 = 63
 */
// Convert rank/file to square index (a1 = 0, h8 = 63)
#define SQ(file, rank)  ((rank) * 8 + (file))
#define SQ_INDEX(file, rank)  SQ(file, rank)

#define sq_file(sq)     ((sq) & 7)
#define sq_rank(sq)     ((sq) >> 3)

/* === Enumerations === */
typedef enum { WHITE = 0, BLACK = 1, COLOR_NB = 2 } Side;

typedef enum { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_NB } PieceType;

/* === Position structure === */
typedef struct {
    uint64_t pieces[2][PIECE_NB];  // Bitboards for each color/piece type
    uint64_t occupancy[2];         // Occupancy for each side
    uint64_t occupancy_both;       // All pieces combined

    uint64_t zobrist_key;          // Hash key
    uint64_t history[256];         // For repetition detection

    int side_to_move;              // WHITE=0, BLACK=1
    int castling_rights;           // bitmask WK=1, WQ=2, BK=4, BQ=8
    int en_passant;                // Target square index or -1
    int halfmove_clock;            // For 50-move rule
    int fullmove_number;           // Increments after Black’s move
    int history_count;
} Position;

/* === Inline helpers === */
static inline void set_bit(uint64_t *bb, int sq)   { *bb |=  (1ULL << sq); }
static inline void clear_bit(uint64_t *bb, int sq) { *bb &= ~(1ULL << sq); }
static inline int  get_bit(uint64_t bb, int sq)    { return (int)((bb >> sq) & 1ULL); }

/* === Function declarations === */
int pop_lsb(uint64_t *bb);
int popcount64(uint64_t bb);
int lsb_index(uint64_t bb);

#endif /* BITBOARD_H */