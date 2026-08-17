#ifndef MOVEGEN_H
#define MOVEGEN_H

#include <stdint.h>
#include "bitboard.h"

/*
 * Notes:
 * - Sliding-ray generation (bishops/rooks/queens) must guard against
 *   wrap-around across file boundaries when adding direction offsets
 * * (±1, ±7, ±9, ±8 offsets). The implementation in movegen.c ensures
 *   rays stop when a step would change file/rank illegally.
 */

typedef uint32_t Move;

#define MOVE(from,to,flags)    ((Move)(((from) & 63) | (((to) & 63) << 6) | (((uint32_t)(flags)) << 12)))
#define MOVE_FROM(m)           ((int)((m) & 63))
#define MOVE_TO(m)             ((int)(((m) >> 6) & 63))
#define MOVE_FLAGS(m)          ((int)(((m) >> 12) & 0xFF))  // 8-bit flags

// Basic flags (keep existing bit positions)
#define FLAG_NONE        0
#define FLAG_CAPTURE     (1 << 0)
#define FLAG_CASTLING    (1 << 1)
#define FLAG_EN_PASSANT  (1 << 2)

// Promotion bits (we keep them at the old bit positions but mask them together)
#define FLAG_PROMO_Q     (1 << 3)
#define FLAG_PROMO_R     (1 << 4)
#define FLAG_PROMO_B     (1 << 5)
#define FLAG_PROMO_N     (1 << 6)

// Promotion mask (convenience)
#define FLAG_PROMO_MASK  (FLAG_PROMO_Q | FLAG_PROMO_R | FLAG_PROMO_B | FLAG_PROMO_N)

// Map promotion flags to PieceType (returns PieceType; if none -> PAWN as sentinel)
#define PROMO_TO_PIECETYPE(flags) \
    ( (flags) & FLAG_PROMO_Q ? QUEEN : \
      (flags) & FLAG_PROMO_R ? ROOK  : \
      (flags) & FLAG_PROMO_B ? BISHOP: \
      (flags) & FLAG_PROMO_N ? KNIGHT: PAWN )

// Map PieceType to a promo flag (useful when generating promotion moves)
// NOTE: expects a PieceType (QUEEN,ROOK,BISHOP,KNIGHT)
#define PROMO_FLAG_FOR_PIECE(pt) \
    ( (pt) == QUEEN  ? FLAG_PROMO_Q : \
      (pt) == ROOK   ? FLAG_PROMO_R : \
      (pt) == BISHOP ? FLAG_PROMO_B : \
      (pt) == KNIGHT ? FLAG_PROMO_N : 0 )

// Castling rights bits (unchanged)
#define CASTLE_WK 1
#define CASTLE_WQ 2
#define CASTLE_BK 4
#define CASTLE_BQ 8

// At top near MoveList
#ifndef MAX_MOVES
#define MAX_MOVES 256
#endif

typedef struct {
    Move moves[MAX_MOVES];
    int count;
} MoveList;

typedef enum {
    PLAYING,
    WHITE_CHECKMATE,
    BLACK_CHECKMATE,
    STALEMATE,
    DRAW_50_MOVE,
    DRAW_REPETITION,
    DRAW_INSUFFICIENT
} GameState;

// prototypes
int is_king_in_check(const Position *pos, Side side);

int is_square_attacked(const Position *pos, int sq, Side attacker);

void generate_all_moves(const Position *pos, MoveList *list);
void filter_legal_moves(const Position *pos, MoveList *legal);
void apply_move(Position *pos, Move m);

/* Helpers for WASM/JS testing (implemented in movegen.c) */
int init_movegen_tables(); // call once (safe to call multiple times)
int check_threefold_repetition(const Position *pos);

#endif /* MOVEGEN_H */