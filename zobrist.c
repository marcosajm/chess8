// zobrist.c
#include "zobrist.h"
#include <stdlib.h>
#include <time.h>

uint64_t zobrist_piece[2][PIECE_NB][64];
uint64_t zobrist_castling[16];
uint64_t zobrist_enpassant[64];
uint64_t zobrist_side;

static uint64_t rand64(void) {
    uint64_t r = 0;
    for (int i = 0; i < 4; i++) {
        r <<= 16;
        r |= (uint64_t)(rand() & 0xFFFF);
    }
    return r;
}

void init_zobrist(void) {
    srand((unsigned)time(NULL));
    for (int c = 0; c < 2; c++) {
        for (int p = 0; p < PIECE_NB; p++) {
            for (int sq = 0; sq < 64; sq++) {
                zobrist_piece[c][p][sq] = rand64();
            }
        }
    }
    for (int i = 0; i < 16; i++) zobrist_castling[i] = rand64();
    for (int sq = 0; sq < 64; sq++) zobrist_enpassant[sq] = rand64();
    zobrist_side = rand64();
}

void update_zobrist_key(Position *pos) {
  pos->zobrist_key = 0;
  for (int s = 0; s < 2; s++) {
    for (int p = 0; p < PIECE_NB; p++) {
      uint64_t bb = pos->pieces[s][p];
      while (bb) {
        int sq = pop_lsb(&bb);
        pos->zobrist_key ^= zobrist_piece[s][p][sq];
      }
    }
  }
  pos->zobrist_key ^= zobrist_castling[pos->castling_rights];
  if (pos->en_passant != -1)
    pos->zobrist_key ^= zobrist_enpassant[pos->en_passant];
  if (pos->side_to_move == BLACK)
    pos->zobrist_key ^= zobrist_side;
}
/* 
uint64_t compute_hash(const Position *pos) {
    uint64_t h = 0ULL;
    for (int c = 0; c < 2; c++) {
        for (int p = 0; p < PIECE_NB; p++) {
            uint64_t bb = pos->pieces[c][p];
            while (bb) {
                int sq = __builtin_ctzll(bb);
                h ^= zobrist_piece[c][p][sq];
                bb &= bb - 1;
            }
        }
    }
    h ^= zobrist_castling[pos->castling_rights & 0xF];
    if (pos->en_passant >= 0 && pos->en_passant < 64)
        h ^= zobrist_enpassant[pos->en_passant];
    if (pos->side_to_move == BLACK) h ^= zobrist_side;
    return h;
} */