//occupancy.c
#include "occupancy.h"
#include <stdio.h>
#include <string.h>

/* globally visible demo board */
Position global_pos;

inline void sq_name_idx(int sq, char out[3])
{
    out[0] = 'a' + (sq & 7);
    out[1] = '1' + (sq >> 3);
    out[2] = '\0';
}

inline int idx_from_file_rank(char f, char r)
{
    return (r - '1') * 8 + (f - 'a');
}

void fill_board_array(int8_t board[64], const Position *pos) {
    // Reserve 1..6 for white pieces, 7..12 for black. 0 = empty.
    memset(board, 0, 64 * sizeof(board[0]));
    //memset(board, 0, 64);
    for (int s = 0; s < 2; s++) {
        for (int p = 0; p < PIECE_NB; p++) {
            uint64_t bb = pos->pieces[s][p];
            while (bb) {
                int sq = pop_lsb(&bb);
                board[sq] = (s == WHITE) ? (p + 1) : (p + 7);
            }
        }
    }
}   

/* ---------------- core routines ---------------- */
// Recompute occupancies from piece bitboards (local copy to avoid external deps)
void update_occ(Position *pos) {
    pos->occupancy[WHITE] = 0;
    pos->occupancy[BLACK] = 0;

 /*    uint64_t occW = 0, occB = 0;
    for (int p = 0; p < PIECE_NB; p++) occW |= pos->pieces[WHITE][p];
    for (int p = 0; p < PIECE_NB; p++) occB |= pos->pieces[BLACK][p];
    pos->occupancy[WHITE] = occW;
    pos->occupancy[BLACK] = occB;
    pos->occupancy_both = occW | occB;
    printf("[DEBUG] Occupancy: WHITE=0x%lx, BLACK=0x%lx, BOTH=0x%lx\n",
           pos->occupancy[WHITE], pos->occupancy[BLACK], pos->occupancy_both);
     */

/* 
    for (int p = 0; p < PIECE_NB; ++p) {
        pos->occupancy[WHITE] |= pos->pieces[WHITE][p];
        pos->occupancy[BLACK] |= pos->pieces[BLACK][p];
    } */

    for (int s = 0; s < 2; s++) {
        for (int p = 0; p < PIECE_NB; p++) {
            pos->occupancy[s] |= pos->pieces[s][p];
        }
    }
    pos->occupancy_both = pos->occupancy[WHITE] | pos->occupancy[BLACK];
    //printf("[DEBUG] update_occ: occupancy_both = 0x%lx\n", pos->occupancy_both);
    
    // Validation
    uint64_t calculated_white = 0;
    uint64_t calculated_black = 0;
    for (int p = 0; p < PIECE_NB; p++) {
        calculated_white |= pos->pieces[WHITE][p];
        calculated_black |= pos->pieces[BLACK][p];
    }
    
    if (calculated_white != pos->occupancy[WHITE] || 
        calculated_black != pos->occupancy[BLACK]) {
        //printf("[ERROR] Occupancy mismatch detected!\n");
    }
}

void print_board(const Position *pos) {
    for (int rank = 7; rank >= 0; rank--) {
        printf("%d | ", rank + 1);
        for (int file = 0; file < 8; file++) {
            int sq = SQ(file, rank);
        }
        printf("\n");
    }
    printf("    a b c d e f g h\n");
}


void print_board_state(void){

    static const char piece_chars[] = ".PNBRQKpnbrqk"; // <-- Change this line!
    
    int8_t board[64];
    fill_board_array(board, &global_pos);

    puts("\nBoard state:");
    for (int r = 7; r >= 0; --r) {
        printf("%d | ", r + 1);
        for (int f = 0; f < 8; ++f) {
            char pc = piece_chars[board[r * 8 + f]];
            printf("%c ", pc);
        }
        putchar('\n');
    }
    puts("    a b c d e f g h");
    printf("Side to move: %s\n", global_pos.side_to_move == WHITE ? "White" : "Black");
}
