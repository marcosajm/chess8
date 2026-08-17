// movgen.c
#include "movegen.h"
#include "bitboard.h"
#include "zobrist.h"
#include "utilities.h"
#include "fen.h"
#include "occupancy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

static uint64_t knight_attacks[64];
static uint64_t king_attacks[64];
static uint64_t pawn_attacks[2][64]; // [color][square]
static int mg_initialized = 0;

int is_king_in_check(const Position *pos, Side side) {
    uint64_t king_bb = pos->pieces[side][KING];
    if (!king_bb) return 1; // no king found, treat as in check (fail safe)
    int king_sq = __builtin_ctzll(king_bb);
    Side opponent = (side == WHITE) ? BLACK : WHITE;
    return is_square_attacked(pos, king_sq, opponent);
}

static uint64_t mask_knight_attacks(int sq) {
    uint64_t bb = 0ULL;
    int r = sq / 8, f = sq % 8;
    int dr[8] = {2,1,-1,-2,-2,-1,1,2};
    int df[8] = {1,2,2,1,-1,-2,-2,-1};
    for (int i=0;i<8;i++){
        int nr=r+dr[i], nf=f+df[i];
        if (nr>=0 && nr<8 && nf>=0 && nf<8) bb |= 1ULL << (nr*8 + nf);
    }
    return bb;
}

static uint64_t mask_king_attacks(int sq) {
    uint64_t bb = 0ULL;
    int r = sq/8, f = sq%8;
    for (int dr=-1; dr<=1; dr++){
        for (int df=-1; df<=1; df++){
            if (!dr && !df) continue;
            int nr=r+dr, nf=f+df;
            if (nr>=0 && nr<8 && nf>=0 && nf<8) bb |= 1ULL << (nr*8 + nf);
        }
    }
    return bb;
}

static uint64_t mask_pawn_attacks(int sq, Side side) {
    uint64_t bb = 0ULL;
    int r = sq / 8, f = sq % 8;
    
    if (side == WHITE) {
        if (r < 7) {
            if (f > 0) bb |= 1ULL << ((r+1)*8 + (f-1));
            if (f < 7) bb |= 1ULL << ((r+1)*8 + (f+1));
        }
    } else {
        if (r > 0) {
            if (f > 0) bb |= 1ULL << ((r-1)*8 + (f-1));
            if (f < 7) bb |= 1ULL << ((r-1)*8 + (f+1));
        }
    }
    return bb;
}

int is_square_attacked(const Position *pos, int sq, Side attacker) {
    // pawns: compute squares from which an attacker pawn would attack `sq`.
    {
        int r = sq / 8, f = sq % 8;
        uint64_t pawn_src = 0ULL;
        if (attacker == WHITE) {
            // white pawns attack one rank up: attackers must be on rank r-1
            if (r > 0) {
                if (f > 0) pawn_src |= 1ULL << (sq - 9);
                if (f < 7) pawn_src |= 1ULL << (sq - 7);
            }
        } else {
            // black pawns attack one rank down: attackers must be on rank r+1
            if (r < 7) {
                if (f > 0) pawn_src |= 1ULL << (sq + 7);
                if (f < 7) pawn_src |= 1ULL << (sq + 9);
            }
        }
        if (pawn_src & pos->pieces[attacker][PAWN]) return 1;
    }
    
    // knights
    if (knight_attacks[sq] & pos->pieces[attacker][KNIGHT]) return 1;
    
    // king
    if (king_attacks[sq] & pos->pieces[attacker][KING]) return 1;

    // bishops/queens (diagonals)
    static const int bd[4] = {9,7,-7,-9};
    for (int i=0;i<4;i++){
        int to = sq;
        while (1){
            int prev = to;
            to += bd[i];
            if (to < 0 || to >= 64) break;
            // ensure we didn't wrap across files
            if (abs((to % 8) - (prev % 8)) != 1) break;
            
            if (get_bit(pos->occupancy_both, to)) {
                if (get_bit(pos->occupancy[attacker], to) && 
                    (get_bit(pos->pieces[attacker][BISHOP], to) ||
                     get_bit(pos->pieces[attacker][QUEEN], to))) 
                    return 1;
                break;
            }
        }
    }
    
    // rooks/queens (straights)
    static const int rd[4] = {8,-8,1,-1};
    for (int i=0;i<4;i++){
        int to = sq;
        while (1){
            int prev = to;
            to += rd[i];
            if (to < 0 || to >= 64) break;
            if ((rd[i] == 1 || rd[i] == -1) && (to / 8 != prev / 8)) break;

            if (get_bit(pos->occupancy_both, to)) {
                if (get_bit(pos->occupancy[attacker], to) && 
                    (get_bit(pos->pieces[attacker][ROOK], to) ||
                     get_bit(pos->pieces[attacker][QUEEN], to))) 
                    return 1;
                break;
            }
        }
    }
    return 0;
}

void generate_pawn_moves(const Position *pos, MoveList *list, Side s) {
    int dir = (s == WHITE ? 8 : -8);
    int start_rank = (s == WHITE ? 1 : 6);
    //int promo_rank = (s == WHITE ? 6 : 1);    
    // 🛑 BUG FIX: White promotes on rank 8 (index 7), Black promotes on rank 1 (index 0).
    int promo_rank = (s == WHITE ? 7 : 0); // <-- FIX THIS LINE
    
    uint64_t pawns = pos->pieces[s][PAWN];
    
    while (pawns) {
        int sq = pop_lsb(&pawns);
        int rank = sq_rank(sq);
        int fwd = sq + dir;
        
        // Single push
        if (fwd >= 0 && fwd < 64 && !get_bit(pos->occupancy_both, fwd)) {
            if (sq_rank(fwd) == promo_rank) {
                list->moves[list->count++] = MOVE(sq, fwd, FLAG_PROMO_Q);
                list->moves[list->count++] = MOVE(sq, fwd, FLAG_PROMO_R);
                list->moves[list->count++] = MOVE(sq, fwd, FLAG_PROMO_B);
                list->moves[list->count++] = MOVE(sq, fwd, FLAG_PROMO_N);
            } else {
                list->moves[list->count++] = MOVE(sq, fwd, FLAG_NONE);

                // Double push
                if (rank == start_rank) {
                    int fwd2 = sq + 2*dir;
                    if (fwd2 >= 0 && fwd2 < 64 && !get_bit(pos->occupancy_both, fwd2)) {
                        list->moves[list->count++] = MOVE(sq, fwd2, FLAG_NONE);
                    }
                }
            }
        }
        
        // Corrected logic for pawn captures in generate_pawn_moves:

        // Captures
        uint64_t attacks = pawn_attacks[s][sq] & pos->occupancy[!s];
        while (attacks) {
            int to = pop_lsb(&attacks);

            if (sq_rank(to) == promo_rank) {
                // If the pawn reaches the promotion rank via a capture,
                // generate all four promotion capture moves.
                list->moves[list->count++] = MOVE(sq, to, FLAG_PROMO_Q | FLAG_CAPTURE);
                list->moves[list->count++] = MOVE(sq, to, FLAG_PROMO_R | FLAG_CAPTURE);
                list->moves[list->count++] = MOVE(sq, to, FLAG_PROMO_B | FLAG_CAPTURE);
                list->moves[list->count++] = MOVE(sq, to, FLAG_PROMO_N | FLAG_CAPTURE);
            } else {
                // Regular capture.
                list->moves[list->count++] = MOVE(sq, to, FLAG_CAPTURE);
            }
        }
                
        // En passant
        if (pos->en_passant != -1) {
            uint64_t ep_mask = 1ULL << pos->en_passant;
            if (pawn_attacks[s][sq] & ep_mask) {
                // Verify that en passant is legal (the captured pawn is actually there)
                int ep_pawn_sq = pos->en_passant + (s == WHITE ? -8 : 8);
                if (get_bit(pos->pieces[!s][PAWN], ep_pawn_sq)) {
                    list->moves[list->count++] = MOVE(sq, pos->en_passant, FLAG_CAPTURE | FLAG_EN_PASSANT);
                }
            }
        }
    }
}

void generate_knight_moves(const Position *pos, MoveList *list, Side s) {
    uint64_t knights = pos->pieces[s][KNIGHT];
    uint64_t own = pos->occupancy[s];
    
    while (knights) {
        int sq = pop_lsb(&knights);
        uint64_t moves = knight_attacks[sq] & ~own;
        
        while (moves) {
            int to = pop_lsb(&moves);
            int flags = get_bit(pos->occupancy[!s], to) ? FLAG_CAPTURE : FLAG_NONE;
            list->moves[list->count++] = MOVE(sq, to, flags);
        }
    }
}

static uint64_t generate_sliding_attacks(int sq, uint64_t occupancy, const int dirs[], int dircount) {
    uint64_t attacks = 0ULL;
    
    //printf("[DEBUG] Generating sliding attacks from sq %d, occupancy: 0x%lx\n", sq, occupancy);
    
    for (int i = 0; i < dircount; i++) {
        int to = sq;
        
        while (true) {
            int prev = to;
            to += dirs[i];
            
            if (to < 0 || to >= 64) break;

            /* Prevent wrap-around across files for horizontal and diagonal rays.
             * Horizontal (±1) must stay on same rank; diagonals (±7,±9)
             * must change file by exactly one each step. */
            if (dirs[i] == 1 || dirs[i] == -1) {
                if (to / 8 != prev / 8) break;
            } else if (dirs[i] == 9 || dirs[i] == 7 || dirs[i] == -7 || dirs[i] == -9) {
                if (abs((to % 8) - (prev % 8)) != 1) break;
            }

            attacks |= 1ULL << to;
            if (get_bit(occupancy, to)) {
                /* include the blocking square then stop the ray */
                break;
            }
        }
    }
    
    return attacks;
}

static void gen_sliding_moves(const Position *pos, MoveList *list, Side s, PieceType pt, const int dirs[], int dircount) {
    uint64_t pieces = pos->pieces[s][pt];
    uint64_t own = pos->occupancy[s];
    
    while (pieces) {
        int sq = pop_lsb(&pieces);
        uint64_t attacks = generate_sliding_attacks(sq, pos->occupancy_both, dirs, dircount);
        uint64_t moves = attacks & ~own;
        
        while (moves) {
            int to = pop_lsb(&moves);
            int flags = get_bit(pos->occupancy[!s], to) ? FLAG_CAPTURE : FLAG_NONE;
            list->moves[list->count++] = MOVE(sq, to, flags);
        }
    }
}

void generate_bishop_moves(const Position *pos, MoveList *list, Side s) {
    static const int dirs[4] = {9,7,-7,-9};
    gen_sliding_moves(pos, list, s, BISHOP, dirs, 4);
}

void generate_rook_moves(const Position *pos, MoveList *list, Side s) {
    static const int dirs[4] = {8,-8,1,-1};
    gen_sliding_moves(pos, list, s, ROOK, dirs, 4);
}

void generate_queen_moves(const Position *pos, MoveList *list, Side s) {
    static const int dirs[8] = {9,7,-7,-9,8,-8,1,-1};
    gen_sliding_moves(pos, list, s, QUEEN, dirs, 8);
}

void generate_king_moves(const Position *pos, MoveList *list, Side s) {
    uint64_t king = pos->pieces[s][KING];
    if (!king) return;
    
    int sq = __builtin_ctzll(king);
    uint64_t moves = king_attacks[sq] & ~pos->occupancy[s];
    
    while (moves) {
        int to = pop_lsb(&moves);
        int flags = get_bit(pos->occupancy[!s], to) ? FLAG_CAPTURE : FLAG_NONE;
        list->moves[list->count++] = MOVE(sq, to, flags);
    }
}

void generate_castling_moves(const Position *pos, MoveList *list, Side s) {
    // King must not be in check
    if (is_king_in_check(pos, s)) return;
    
    uint64_t occ = pos->occupancy_both;
    
    if (s == WHITE) {
        if ((pos->castling_rights & CASTLE_WK) &&
            !(occ & ((1ULL<<5)|(1ULL<<6))) &&
            !is_square_attacked(pos, 5, BLACK) &&
            !is_square_attacked(pos, 6, BLACK)) {
            list->moves[list->count++] = MOVE(4, 6, FLAG_CASTLING);
        }
        
        if ((pos->castling_rights & CASTLE_WQ) &&
            !(occ & ((1ULL<<1)|(1ULL<<2)|(1ULL<<3))) &&
            !is_square_attacked(pos, 2, BLACK) &&
            !is_square_attacked(pos, 3, BLACK)) {
            list->moves[list->count++] = MOVE(4, 2, FLAG_CASTLING);
        }
    } else {
        if ((pos->castling_rights & CASTLE_BK) &&
            !(occ & ((1ULL<<61)|(1ULL<<62))) &&
            !is_square_attacked(pos, 61, WHITE) &&
            !is_square_attacked(pos, 62, WHITE)) {
            list->moves[list->count++] = MOVE(60, 62, FLAG_CASTLING);
        }
        
        if ((pos->castling_rights & CASTLE_BQ) &&
            !(occ & ((1ULL<<57)|(1ULL<<58)|(1ULL<<59))) &&
            !is_square_attacked(pos, 58, WHITE) &&
            !is_square_attacked(pos, 59, WHITE)) {
            list->moves[list->count++] = MOVE(60, 58, FLAG_CASTLING);
        }
    }
}

int init_movegen_tables(void) {
    if (mg_initialized) return 0;
    
    for (int i = 0; i < 64; i++) {
        knight_attacks[i] = mask_knight_attacks(i);
        king_attacks[i] = mask_king_attacks(i);
        pawn_attacks[WHITE][i] = mask_pawn_attacks(i, WHITE);
        pawn_attacks[BLACK][i] = mask_pawn_attacks(i, BLACK);
    }
    
    mg_initialized = 1;
    return 1;
}

int validate_board(const Position *pos) {
    // Check that pieces don't overlap
    if (pos->occupancy[WHITE] & pos->occupancy[BLACK]) {
        //printf("[ERROR] Pieces overlap between white and black!\n");
        return 0;
    }
    
    // Check that occupancy matches pieces
    uint64_t calculated_white = 0;
    uint64_t calculated_black = 0;
    
    for (int p = 0; p < PIECE_NB; p++) {
        calculated_white |= pos->pieces[WHITE][p];
        calculated_black |= pos->pieces[BLACK][p];
    }
    
    if (calculated_white != pos->occupancy[WHITE]) {
        //printf("[ERROR] White occupancy mismatch!\n");
        return 0;
    }
    
    if (calculated_black != pos->occupancy[BLACK]) {
        //printf("[ERROR] Black occupancy mismatch!\n");
        return 0;
    }
    
    return 1;
}

void generate_all_moves(const Position *pos, MoveList *list) {
    if (!mg_initialized) init_movegen_tables();
    
    // EMERGENCY FIX: Rebuild occupancy if corrupted
    if (pos->occupancy_both == 0) {
        //printf("[WARNING] Rebuilding corrupted occupancy bitboards\n");
        // Cast away const to fix the corruption
        Position *temp_pos = (Position *)pos;
        update_occ(temp_pos);
    }
    
    list->count = 0;
    Side s = pos->side_to_move;
  /*     if (!validate_board(pos)) {
        printf("[ERROR] Invalid board state!\n");
        list->count = 0;
        return;
    }  */
    
    //printf("[DEBUG] Generating moves for %s\n", s == WHITE ? "White" : "Black");
    //printf("[DEBUG] Board state before movegen:\n");
   // print_board_state(); // Add this function if you don't have it    
    
    generate_pawn_moves(pos, list, s);
    generate_knight_moves(pos, list, s);
    generate_bishop_moves(pos, list, s);
    generate_rook_moves(pos, list, s);
    generate_queen_moves(pos, list, s);
    generate_king_moves(pos, list, s);
    generate_castling_moves(pos, list, s);

    //printf("[DEBUG] Generated %d moves\n", list->count);
    for (int i = 0; i < list->count; i++) {
        Move m = list->moves[i];
        char from[3], to[3];
        sq_name_idx(MOVE_FROM(m), from);
        sq_name_idx(MOVE_TO(m), to);
        //printf("  %s -> %s (flags: %d)\n", from, to, MOVE_FLAGS(m));
    }
}

void filter_legal_moves(const Position *pos, MoveList *legal) {
    MoveList all_moves;
    generate_all_moves(pos, &all_moves);
    
    legal->count = 0;
    
    for (int i = 0; i < all_moves.count; i++) {
        Position tmp = *pos;
        apply_move(&tmp, all_moves.moves[i]);
        
        // Check if the moving side's king is not in check after the move
        if (!is_king_in_check(&tmp, pos->side_to_move)) {
            if (legal->count < MAX_MOVES) {
                legal->moves[legal->count++] = all_moves.moves[i];
            }
        }
    }
}

static inline void move_piece(Position *pos, Side color, PieceType piece, int from, int to) {
    // Clear the piece from the original square
    clear_bit(&pos->pieces[color][piece], from);
    clear_bit(&pos->occupancy[color], from);
    clear_bit(&pos->occupancy_both, from);
    pos->zobrist_key = xor_piece(pos->zobrist_key, color, piece, from);
    
    // Set the piece on the new square
    set_bit(&pos->pieces[color][piece], to);
    set_bit(&pos->occupancy[color], to);
    set_bit(&pos->occupancy_both, to);
    pos->zobrist_key = xor_piece(pos->zobrist_key, color, piece, to);
}

void apply_move(Position *pos, Move m) {
    const int from  = MOVE_FROM(m);
    const int to    = MOVE_TO(m);
    const int flags = MOVE_FLAGS(m);

    const Side side = (Side)pos->side_to_move;
    const Side opp  = !side;

    // ---- 1.  understand which piece is moving ----------------------
    PieceType pt = PAWN;                 // default for promotions
    for (PieceType p = PAWN; p < PIECE_NB; ++p) {
        if (get_bit(pos->pieces[side][p], from)) {
            pt = p;
            break;
        }
    }

    // ---- 2.  snapshot old values into local vars -------------------
    const int old_castling = pos->castling_rights;
    const int old_ep = pos->en_passant;

    // ---- 3.  key: XOR out side + castling + ep --------- ONE TIME ---
    pos->zobrist_key = xor_side       (pos->zobrist_key);
    pos->zobrist_key = xor_castling   (pos->zobrist_key, old_castling);
    pos->zobrist_key = xor_enpassant  (pos->zobrist_key, old_ep);

    
        // ---- 4.  handle castling moves: king + rook ---------------------
    if ((flags & FLAG_CASTLING) && pt == KING) {
        // castling flag is not combined with any other flag in practice,
        // but we just test the bit to be robust.
        int rook_from, rook_to;
        if (side == WHITE) {
            if (to ==  6) { rook_from = 7;  rook_to = 5;  } // e1g1
            else          { rook_from = 0;  rook_to = 3;  } // e1c1
        } else {
            if (to == 62) { rook_from = 63; rook_to = 61; } // e8g8
            else          { rook_from = 56; rook_to = 59; } // e8c8
        }
        // move the king normally
        move_piece(pos, side, KING, from, to);

        // move the rook
        move_piece(pos, side, ROOK, rook_from, rook_to);
    }
    // ---- 5.  non-castling moves ------------------------------------
    else {
        // 5a. remove any captured piece
        if (flags & FLAG_CAPTURE) {
            int cap_sq = to;
            if ((flags & FLAG_EN_PASSANT) && pt == PAWN) {
                cap_sq = (side == WHITE ? to - 8 : to + 8);
            }
            
            for (PieceType p = PAWN; p < PIECE_NB; ++p) {
                if (get_bit(pos->pieces[opp][p], cap_sq)) {
                    clear_bit(&pos->pieces[opp][p], cap_sq);
                    pos->zobrist_key = xor_piece(pos->zobrist_key, opp, p, cap_sq);
                    
                    // Keep occupancy consistent immediately
                    clear_bit(&pos->occupancy[opp], cap_sq);
                    clear_bit(&pos->occupancy_both, cap_sq);
                    
                    break;
                }
            }
        }
        // 5b.  promotion? change promoted piece type
        PieceType place_type = pt;
       /*  if (flags & (FLAG_PROMO_N | FLAG_PROMO_B | FLAG_PROMO_R | FLAG_PROMO_Q)) {
            if      (flags & FLAG_PROMO_Q) place_type = QUEEN;
            else if (flags & FLAG_PROMO_R) place_type = ROOK;
            else if (flags & FLAG_PROMO_B) place_type = BISHOP;
            else                           place_type = KNIGHT;
           
        } */
        if (flags & FLAG_PROMO_MASK) {
            // Determine which promotion flag is present and pick the correct piece type.
            place_type = PROMO_TO_PIECETYPE(flags);
            // clear original pawn before promotion
            clear_bit(&pos->pieces[side][pt], from);
            pos->zobrist_key = xor_piece(pos->zobrist_key, side, pt, from);
            set_bit(&pos->pieces[side][place_type], to);
            pos->zobrist_key = xor_piece(pos->zobrist_key, side, place_type, to);
        } else {
            // simple move or non-capture
            move_piece(pos, side, pt, from, to);
        }
    }

    // ---- 6.  castling rights adjustment -----------------------------
    int new_rights = old_castling;
    
    // If a rook moves or is captured, remove specific castling right
    if (from == 0 || to == 0)  new_rights &= ~CASTLE_WQ;
    if (from == 7 || to == 7)  new_rights &= ~CASTLE_WK;
    if (from == 56 || to == 56) new_rights &= ~CASTLE_BQ;
    if (from == 63 || to == 63) new_rights &= ~CASTLE_BK;

    // king/rook of same color having moved removes whole side:
    if (pt == KING) {
        if (side == WHITE) new_rights &= ~(CASTLE_WK | CASTLE_WQ);
        else               new_rights &= ~(CASTLE_BK | CASTLE_BQ);
    }
    if (pt == ROOK) {
        if (side == WHITE) {
            if (from == 7) new_rights &= ~CASTLE_WK;
            if (from == 0) new_rights &= ~CASTLE_WQ;
        } else {
            if (from == 63) new_rights &= ~CASTLE_BK;
            if (from == 56) new_rights &= ~CASTLE_BQ;
        }
    }
    pos->castling_rights = new_rights;

    // ---- 7.  en-passant square update ------------------------------- 
    pos->en_passant = -1;
    if (pt == PAWN && abs(to - from) == 16) {
        pos->en_passant = (side == WHITE ? to - 8 : to + 8);
    }

    // ---- 8.  half-move clock & ply increment ----------------------- 
    if (pt == PAWN || (flags & FLAG_CAPTURE))
        pos->halfmove_clock = 0;
    else
        ++pos->halfmove_clock;

    // ---- 9.  key: XOR in updated side + castling + ep -------------- 
    pos->side_to_move = opp;
    pos->zobrist_key  = xor_castling   (pos->zobrist_key, new_rights);
    pos->zobrist_key  = xor_enpassant  (pos->zobrist_key, pos->en_passant);
    pos->zobrist_key  = xor_side       (pos->zobrist_key);

    // ---- 10. store new key in repetition table --------------------- 
    if (pos->history_count < 256) {
        pos->history[pos->history_count++] = pos->zobrist_key;
    }

     // ---- 11. recompute occupancy bitmaps (8 total bitboards)
    for (Side c = WHITE; c <= BLACK; ++c) {
        pos->occupancy[c] = 0;
        for (PieceType p = PAWN; p < PIECE_NB; ++p)
            pos->occupancy[c] |= pos->pieces[c][p];
    }
    pos->occupancy_both = pos->occupancy[WHITE] | pos->occupancy[BLACK];
    
    // ADD VALIDATION:
    //printf("[DEBUG] After apply_move: occupancy_both = 0x%lx\n", pos->occupancy_both);
    if (pos->occupancy_both == 0) {
        //printf("[ERROR] Empty board after move!\n");
        // Add emergency recovery or abort
    }
}

bool position_repeated(const Position *pos, int ply) {
    const uint64_t key = pos->zobrist_key;
    int cnt = 0;
    for (int i = (int)pos->history_count - 2; i >= 0 && cnt < 2; --i) {
        if (pos->history[i] == key) {
            if (++cnt == 2) return true;
            i -= 1;                 /* skip intervening irreversible moves */
        }
    }
    return false;
}

int check_threefold_repetition(const Position *pos) {
    if (pos->history_count < 5) {
        return 0; // Not enough moves for three-fold repetition
    }
    int count = 0;
    // The current position is the last one in the history array
    uint64_t current_key = pos->zobrist_key;

    // Iterate through the history, skipping the current key
    for (int i = 0; i < pos->history_count - 1; i++) {
        if (pos->history[i] == current_key) {
            count++;
        }
    }
    //return count >= 3;
    return count >= 2; // Returns true if the current position has been reached twice before
}

static inline bool insufficient_material(const Position *pos) {
    uint64_t kpk = pos->pieces[WHITE][KING] | pos->pieces[BLACK][KING];

    //static const uint64_t kpk  = pos->pieces[WHITE][KING] | pos->pieces[BLACK][KING];
    uint64_t minors = pos->occupancy_both ^ kpk;

    /* bare Kings ⇒ yes
       lone Bishop each or lone Knight each ⇒ yes */
    if (popcount64(minors) <= 1) return true;
    /* opposite-coloured bishops only ⇒ yes
       otherwise continue playing */
    return false;
}

void print_move_path(int from, int to) {
    //printf("Move from %s to %s: ", square_to_algebraic(from), square_to_algebraic(to));
    
    int file_from = from % 8;
    int rank_from = from / 8;
    int file_to = to % 8;
    int rank_to = to / 8;
    
    int file_dir = (file_to > file_from) ? 1 : (file_to < file_from) ? -1 : 0;
    int rank_dir = (rank_to > rank_from) ? 1 : (rank_to < rank_from) ? -1 : 0;
    
    //printf("File direction: %d, Rank direction: %d\n", file_dir, rank_dir);
    
    int current = from;
    while (current != to) {
        current += file_dir + (8 * rank_dir);
        //printf("  Path through: %s\n", square_to_algebraic(current));
    }
}
