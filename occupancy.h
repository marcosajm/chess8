#ifndef OCCUPANCY_H
#define OCCUPANCY_H

#include "bitboard.h"
#include <stdint.h>
#include <stdbool.h>

extern Position global_pos;

/* ----- public helpers ----------------------------------------------------- */
void sq_name_idx(int sq, char out[3]);
int idx_from_file_rank(char f, char r);
void update_occ(Position *pos);
void fill_board_array(int8_t board[64], const Position *pos);
//void print_board_state(void);
//void print_board(const Position *pos);

#endif /* OCCUPANCY_H */