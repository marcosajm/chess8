#include "fen.h"
#include "bitboard.h"
#include "movegen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static int square_index_from_coords(char file, char rank) {
    if (file < 'a' || file > 'h') return -1;
    if (rank < '1' || rank > '8') return -1;
    int f = file - 'a';          // 0..7
    int r = rank - '1';          // '1'->0, '8'->7
    //int r = 8 - (rank - '0');   // invert rank: '8'→0, '1'→7
    //int r = 8 - (rank - '1') - 1; // rank '8'→0, '1'→7
    return r * 8 + f;            // a1=0, a8=56
}

int parse_fen(const char *fen, Position *pos) {
    if (!fen || !pos) return -1;
    memset(pos, 0, sizeof(Position));
    pos->en_passant = -1;
    pos->side_to_move = WHITE;
    pos->castling_rights = 0;
    pos->halfmove_clock = 0;
    pos->fullmove_number = 1;

    const char *p = fen;
    
    int rank = 7;
    int file = 0;

    /* --- Piece placement --- */
    while (*p && *p != ' ') {
        char c = *p++;
        if (c == '/') {
            rank--;
            file = 0;
            continue;
        }
        if (isdigit((unsigned char)c)) {
            file += c - '0';
            continue;
        }

        Side side = isupper((unsigned char)c) ? WHITE : BLACK;
        char lower = tolower((unsigned char)c);

        PieceType pt;
        switch (lower) {
            case 'p': pt = PAWN; break;
            case 'n': pt = KNIGHT; break;
            case 'b': pt = BISHOP; break;
            case 'r': pt = ROOK; break;
            case 'q': pt = QUEEN; break;
            case 'k': pt = KING; break;
            default: return -1;
        }

        //int sq = rank * 8 + file;
        //int sq = (7 - rank) * 8 + file;  // a8→0, a1→56 if your engine expects top→bottom
        int sq = SQ(file, rank);  // Correct mapping
        set_bit(&pos->pieces[side][pt], sq);
        pos->occupancy[side] |= BIT(sq);
        file++;
    }

    /* --- Side to move --- */
    if (*p == ' ') p++;
    if (*p == 'w') pos->side_to_move = WHITE;
    else if (*p == 'b') pos->side_to_move = BLACK;
    else return -1;
    while (*p && *p != ' ') p++;

    /* --- Castling rights --- */
    if (*p == ' ') p++;
    if (*p == '-') p++;
    else {
        while (*p && *p != ' ') {
            char c = *p++;
            if (c == 'K') pos->castling_rights |= CASTLE_WK;
            else if (c == 'Q') pos->castling_rights |= CASTLE_WQ;
            else if (c == 'k') pos->castling_rights |= CASTLE_BK;
            else if (c == 'q') pos->castling_rights |= CASTLE_BQ;
        }
    }

    /* --- En passant --- */
    if (*p == ' ') p++;
    if (*p == '-') p++;
    else if (isalpha((unsigned char)p[0]) && isdigit((unsigned char)p[1])) {
        int sqidx = square_index_from_coords(p[0], p[1]);
        if (sqidx >= 0) pos->en_passant = sqidx;
        p += 2;
    }
    
    pos->occupancy_both = pos->occupancy[WHITE] | pos->occupancy[BLACK];

    for (int sq = 0; sq < 64; sq++) {
        int r = sq / 8;
        int f = sq % 8;
      //  printf("%2d -> %c%d\n", sq, 'a'+f, r+1);
    }
    
   // printf("[FEN] Parsed position, side=%s, castling=%d, ep=%d\n",
 //       pos->side_to_move == WHITE ? "White" : "Black",
 //       pos->castling_rights,
 //       pos->en_passant);

    return 0;
}