// engine.c — WASM + UCI integrated
#include "bitboard.h"
#include "movegen.h"
#include "zobrist.h"
#include "search.h"
#include "occupancy.h"
#include "fen.h"
#include "uci.h"
#include "nnue.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

// ===============
// WASM interface
// ===============

static int nnue_level = 1;

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#ifdef __EMSCRIPTEN__
#define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EXPORT
#endif

EXPORT void init_board_wasm() {
    init_movegen_tables();
    init_zobrist();
    parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &global_pos);
    update_occ(&global_pos);
    update_zobrist_key(&global_pos);
    promotion_pending = -1;
    printf("[INIT] Board initialized with starting FEN\n");
    print_board_state();
    if (!g_nnue.loaded) {
        if (nnue_load_weights_from_file(&g_nnue, "nnue.bin")) {
            nnue_level = 1;
            printf("[NNUE] Active model: Level %d (nnue.bin)\n", nnue_level);
        } else {
            fprintf(stderr, "Error loading NNUE weights\n");
        }
    }
}

EXPORT int8_t* get_board_ptr_wasm() {
    static int8_t board[64];
    fill_board_array(board, &global_pos);
    //printf("[DEBUG] Returning board pointer\n");
    return board;
}

EXPORT int get_current_turn_wasm() {
   // printf("[TURN] Current: %s\n", global_pos.side_to_move == WHITE ? "White" : "Black");
    return global_pos.side_to_move;
}

// Returns packed (from | (to << 8)) for JS side, does NOT apply move.
// UCI uses the actual move string; see UCI loop below.
EXPORT int find_ai_move_wasm_depth(int ai_color, int depth) {
    if (global_pos.side_to_move != ai_color) {
        printf("[AI] Skipped (AI: %s, Current: %s)\n", ai_color == WHITE ? "White" : "Black", global_pos.side_to_move == WHITE ? "White" : "Black");
        return -1;
    }

    if (depth <= 0) {
        depth = 1;
    }

    Move m = search_best_move(&global_pos, depth);
    if (!m) {
        printf("[AI] No valid move found\n");
        return -1;
    }
    int from = MOVE_FROM(m), to = MOVE_TO(m);
    char fs[3], ts[3];
    sq_name_idx(from, fs);
    sq_name_idx(to, ts);
    printf("[AI] Selected %s -> %s (depth=%d)\n", fs, ts, depth);
    return from | (to << 8);
}

EXPORT int find_ai_move_wasm(int ai_color) {
    return find_ai_move_wasm_depth(ai_color, 1);
}

EXPORT void promote_pawn_wasm(int sq, int piece_type) {
    if (promotion_pending != sq) {
        //printf("[PROMO] Failed: no pending at %d\n", sq);
        return;
    }

    // Determine which side actually owns the pawn on the square. After
    // applying the base move the `side_to_move` has already been flipped,
    // so the promoting side is the opposite of the current side.
    int side = (global_pos.side_to_move == WHITE) ? BLACK : WHITE;

    // Map incoming JS piece codes into internal PieceType index. Support both
    // white (1..6) and black (7..12) encodings; fall back to queen if invalid.
    int piece = QUEEN; // default
    if (piece_type >= 1 && piece_type <= 6) {
        piece = piece_type - 1; // white: 1->PAWN(0), 5->QUEEN(4)
    } else if (piece_type >= 7 && piece_type <= 12) {
        piece = piece_type - 7; // black: 7->PAWN(0), 11->QUEEN(4)
    }

    static const char *names[] = {"Pawn","Knight","Bishop","Rook","Queen","King"};
    char sc[3]; sq_name_idx(sq, sc);
    printf("[PROMO] %s at %s -> %s\n", side==WHITE?"White":"Black", sc, names[piece]);

    // Replace pawn with promoted piece on that square and keep Zobrist consistent.
    // The base move left a pawn on `sq`; remove that pawn's bit/hash and add
    // the promoted piece's bit/hash, then refresh occupancies.
    clear_bit(&global_pos.pieces[side][PAWN], sq);
    global_pos.zobrist_key = xor_piece(global_pos.zobrist_key, side, PAWN, sq);

    set_bit(&global_pos.pieces[side][piece], sq);
    global_pos.zobrist_key = xor_piece(global_pos.zobrist_key, side, piece, sq);

    update_occ(&global_pos);

    promotion_pending = -1;
    print_board_state();
}

EXPORT int get_pawn_promotion_pending_index_wasm() {
    //printf("[DEBUG] Promotion pending index: %d\n", promotion_pending);
    return promotion_pending;
}

EXPORT int get_game_state_wasm() {
    MoveList list;
    generate_all_moves(&global_pos, &list);
    filter_legal_moves(&global_pos, &list);

    if (list.count == 0) {
        // Find king square quickly without modifying bitboards
        uint64_t kbb = global_pos.pieces[global_pos.side_to_move][KING];
        int king_sq = (kbb ? __builtin_ctzll(kbb) : -1);
        if (king_sq >= 0 && is_square_attacked(&global_pos, king_sq, (Side)(global_pos.side_to_move ^ 1))) {
            printf("[STATE] Checkmate\n");
            return 1;
        }
        printf("[STATE] Stalemate\n");
        return 2;
    }

    if (global_pos.halfmove_clock >= 50) {
        printf("[STATE] Draw (50-move rule)\n");
        return 4;
    }

    if (check_threefold_repetition(&global_pos)) {
        printf("[STATE] Draw (threefold repetition)\n");
        return 3; // Use a new enum value for this
    }

    int piece_count = 0;
    for (int s = 0; s < 2; s++)
        for (int p = 0; p < PIECE_NB; p++)
            piece_count += popcount64(global_pos.pieces[s][p]);

    if (piece_count <= 4) {
        printf("[STATE] Draw (insufficient material)\n");
        return 5;
    }

     // ... Insufficient material check ...
    
    printf("[STATE] Ongoing\n");
    return 0;
}

//EXPORT int evaluate_board(Position *pos) {
EXPORT int evaluate_board() {
    int score = evaluate_position(&global_pos);
    //printf("[EVAL] Score: %d\n", score);
    return score;
}

EXPORT int set_difficulty_wasm(int level) {
    if (level < 1 || level > 5) {
        fprintf(stderr, "[DIFFICULTY] Invalid level: %d (must be 1-5)\n", level);
        return 0;
    }

    char model_path[16];
    if (level == 1) {
        snprintf(model_path, sizeof(model_path), "nnue.bin");
    } else {
        snprintf(model_path, sizeof(model_path), "nnue%d.bin", level);
    }

    NNUEWeights candidate = {0};
    if (nnue_load_weights_from_file(&candidate, model_path)) {
        nnue_free(&g_nnue);
        g_nnue = candidate;
        nnue_level = level;
        printf("[DIFFICULTY] Successfully loaded %s\n", model_path);
        return 1;
    }

    fprintf(stderr, "[DIFFICULTY] Failed to load %s\n", model_path);
    return 0;
}

EXPORT int make_move_wasm(int from, int to) {
    char fs[3], ts[3];
    sq_name_idx(from, fs);
    sq_name_idx(to, ts);
    //printf("[MOVE] Attempting %s -> %s\n", fs, ts);

    MoveList list;
    generate_all_moves(&global_pos, &list);
    filter_legal_moves(&global_pos, &list);

    for (int i = 0; i < list.count; i++) {
        Move m = list.moves[i];
        if (MOVE_FROM(m) == from && MOVE_TO(m) == to) {
            int flags = MOVE_FLAGS(m);

            if (flags & FLAG_PROMO_MASK) {
                // frontend must call promote_pawn_wasm() afterwards
                promotion_pending = to;

                 // 1. Temporarily remove the promotion flag before applying move
                Move base_move = MOVE(from, to, flags & ~FLAG_PROMO_MASK); 
                // 2. Apply move, but remove the piece at 'from' and place the pawn at 'to', 
                //    AND flip the side-to-move.
                apply_move(&global_pos, base_move);

                // Apply the full move including the promotion flag so apply_move can do bookkeeping
                //apply_move(&global_pos, m);
                // apply_move(&global_pos, MOVE(from, to, FLAG_NONE));
                update_occ(&global_pos);
                //printf("[PROMO] Pending at %s\n", ts);
                print_board_state();
                return 2;
            }
            // Regular move (non-promotion)
            apply_move(&global_pos, m);
            update_occ(&global_pos);
            //printf("[MOVE] Applied %s -> %s\n", fs, ts);
            print_board_state();
            return 1;
        }
    }
    printf("[INVALID] %s -> %s\n", fs, ts);
    return 0;
}

EXPORT void reset_game(Position *pos) {
    parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", pos);
    init_zobrist();
    update_zobrist_key(pos);
    update_occ(pos);
    printf("[RESET] Game reset\n");
}

// =======================
// UCI loop (native only)
// =======================

#ifndef __EMSCRIPTEN__

static void uci_loop(void) {
    char line[4096];

    // Print ID on engine start if GUI sends input immediately
    // Engine identifies again when "uci" is received (per spec it's fine).
    while (fgets(line, sizeof(line), stdin)) {
        // Trim trailing newline
        size_t n = strlen(line);
        while (n && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = '\0';

       // if (!strncmp(line, "uci", 3)) {
            //printf("id name Togoby WASM/Native Engine\n");
            //printf("id author Marcos Marques\n");
            // (Optional) options here via "option name ...".
            //printf("uciok\n");
       // }
        //else if (!strncmp(line, "isready", 7)) {
            //printf("readyok\n");
       //}
        //else 
        if (!strncmp(line, "ucinewgame", 10)) {
            init_board_wasm();
        }
        else if (!strncmp(line, "position startpos", 17)) {
            const char *moves = strstr(line, " moves ");
            if (moves) {
                uci_set_position_startpos_with_moves(moves + 7);
            } else {
                uci_set_position_startpos_with_moves(NULL);
            }
        }
        else if (!strncmp(line, "position fen ", 13)) {
            uci_set_position_fen_with_moves(line + 13);
        }
        else if (!strncmp(line, "go", 2)) {
            // Simple support: optional "depth N", else default depth 3
            int depth = 3;
            const char *dptr = strstr(line, "depth ");
            if (dptr) depth = atoi(dptr + 6);
            if (depth <= 0) depth = 3;

            Move best = search_best_move(&global_pos, depth);
            if (!best) {
                //printf("bestmove 0000\n");
            } else {
                char uci[6];
                move_to_uci(best, uci);
                // Apply the move to internal state (common behavior in many engines)
                apply_move(&global_pos, best);
                update_occ(&global_pos);
                //printf("bestmove %s\n", uci);
            }
        }
        else if (!strncmp(line, "stop", 4)) {
            // Our search is synchronous; nothing special needed.
        }
        else if (!strncmp(line, "quit", 4)) {
            break;
        }
        else if (!strncmp(line, "d", 1)) {
            // Handy debug: print board
            print_board_state();
        }

        fflush(stdout);
    }
}

int main(void) {
    // Initialize defaults so the engine is in a valid state
    init_board_wasm();
    uci_loop();
    return 0;
}
#endif  // __EMSCRIPTEN__
