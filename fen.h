#ifndef FEN_H
#define FEN_H

#include <stddef.h>
#include "bitboard.h"

int parse_fen(const char *fen, Position *pos);

#endif
