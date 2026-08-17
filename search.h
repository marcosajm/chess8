#ifndef SEARCH_H
#define SEARCH_H

#include "bitboard.h"
#include "movegen.h"

// Piece values for simple evaluation
extern int piece_values[PIECE_NB];

// Evaluate a position (material only for demo)
int evaluate_position(const Position *pos);

// Alpha-beta search
int minimax(Position *pos, int depth, int alpha, int beta, int maximizingPlayer);

// Search best move at given depth
Move search_best_move(Position *pos, int depth);

#endif
