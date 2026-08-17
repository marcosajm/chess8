#ifndef UCI_H
#define UCI_H

#include "bitboard.h"
#include "movegen.h"

extern int promotion_pending;

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

/* Convert an internal Move to a zero-terminated UCI string (max 6 chars). */
void move_to_uci(Move m, char out[6]);

/*
 * Parse a UCI move string and find the corresponding legal move.
 * Returns 1 and writes *out on success, else 0.
 */
//int uci_to_legal_move(const Position *pos, const char *uci, Move *out);

/* Set the global position to the standard startpos followed by
   the optional space-separated move list. */
void uci_set_position_startpos_with_moves(const char *moves_start);

/* Set the global position from a FEN string optionally followed by
   " moves <move1> <move2> ...". */
void uci_set_position_fen_with_moves(const char *fen_and_moves);

#endif /* UCI_H */
