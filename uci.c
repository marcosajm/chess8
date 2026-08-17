// uci.c
#include "uci.h"
#include "bitboard.h"
#include "movegen.h"
#include "occupancy.h"
#include "zobrist.h"
#include "fen.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

int promotion_pending = -1;

// =========================
// Internal state & helpers
// =========================

// Convert internal Move -> UCI string (e.g., "e2e4", "e7e8q")
void move_to_uci(Move m, char out[6]) {
    int from = MOVE_FROM(m);
    int to   = MOVE_TO(m);

    out[0] = 'a' + (from & 7);
    out[1] = '1' + (from >> 3);
    out[2] = 'a' + (to & 7);
    out[3] = '1' + (to >> 3);
    out[4] = '\0';

    int flags = MOVE_FLAGS(m);
    if (flags == FLAG_PROMO_N || flags == FLAG_PROMO_B ||
        flags == FLAG_PROMO_R || flags == FLAG_PROMO_Q) {
        char promo = 'q';
        switch (flags) {
            case FLAG_PROMO_N: promo = 'n'; break;
            case FLAG_PROMO_B: promo = 'b'; break;
            case FLAG_PROMO_R: promo = 'r'; break;
            case FLAG_PROMO_Q: promo = 'q'; break;
        }
        out[4] = promo;
        out[5] = '\0';
    }
}

// Parse a UCI move string into a legal Move by matching generated legal moves
// Returns 1 and writes *out on success, else 0.
static int uci_to_legal_move(const Position *pos, const char *uci, Move *out) {
    size_t len = strlen(uci);
    if (len < 4) return 0;

    int from = idx_from_file_rank(uci[0], uci[1]);
    int to   = idx_from_file_rank(uci[2], uci[3]);
    int want_promo = 0; // 0 none, else one of FLAG_PROMO_*

    if (len >= 5) {
        switch (uci[4]) {
            case 'n': want_promo = FLAG_PROMO_N; break;
            case 'b': want_promo = FLAG_PROMO_B; break;
            case 'r': want_promo = FLAG_PROMO_R; break;
            case 'q': want_promo = FLAG_PROMO_Q; break;
            default: return 0;
        }
    }

    MoveList list;
    generate_all_moves(pos, &list);
    filter_legal_moves(pos, &list);

    for (int i = 0; i < list.count; i++) {
        Move m = list.moves[i];
        if (MOVE_FROM(m) == from && MOVE_TO(m) == to) {
            int flags = MOVE_FLAGS(m);
            if (want_promo) {
                if (flags == want_promo) { *out = m; return 1; }
                // Some engines encode promotion + capture combinations.
                // Your flags schema has promotion as 1..4 (mutually exclusive) and capture as 8.
                // If your generator sets FLAG_CAPTURE alongside promotion, the compare above
                // would fail. So allow FLAG_CAPTURE bit when promotion matches:
                if ((flags & 0x7) == want_promo) { *out = m; return 1; }
            } else {
                // No promotion requested in UCI — accept only non-promotion moves
                if (flags != FLAG_PROMO_N && flags != FLAG_PROMO_B &&
                    flags != FLAG_PROMO_R && flags != FLAG_PROMO_Q) {
                    *out = m; return 1;
                }
            }
        }
    }
    return 0;
}

void uci_set_position_startpos_with_moves(const char *moves_start) {
    init_movegen_tables();
    init_zobrist();
    parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &global_pos);
    update_occ(&global_pos);
    update_zobrist_key(&global_pos);
    promotion_pending = -1;

    if (!moves_start) return;

    // token-by-token parse of move list
    char buf[4096];
    strncpy(buf, moves_start, sizeof(buf)-1);
    buf[sizeof(buf)-1] = '\0';
    char *tok = strtok(buf, " \t\r\n");
    while (tok) {
        Move m;
        if (uci_to_legal_move(&global_pos, tok, &m)) {
            apply_move(&global_pos, m);
            update_occ(&global_pos);
        } else {
            //printf("info string illegal move in position moves: %s\n", tok);
        }
        tok = strtok(NULL, " \t\r\n");
    }
}

void uci_set_position_fen_with_moves(const char *fen_and_moves) {
    // fen_and_moves is the substring after "position fen "
    // format: <fen string> [moves <m1> <m2> ...]
    char fen[256] = {0};
    const char *moves_kw = strstr(fen_and_moves, " moves ");
    if (moves_kw) {
        size_t fen_len = (size_t)(moves_kw - fen_and_moves);
        if (fen_len >= sizeof(fen)) fen_len = sizeof(fen) - 1;
        memcpy(fen, fen_and_moves, fen_len);
        fen[fen_len] = '\0';

        init_movegen_tables();
        init_zobrist();
        if (parse_fen(fen, &global_pos) != 0) {
            //printf("info string invalid FEN\n");
            return;
        }
        update_occ(&global_pos);
        update_zobrist_key(&global_pos);
        promotion_pending = -1;

        const char *moves_start = moves_kw + 7; // skip " moves "
        uci_set_position_startpos_with_moves(moves_start);
    } else {
        // only FEN, no moves
        init_movegen_tables();
        init_zobrist();
        if (parse_fen(fen_and_moves, &global_pos) != 0) {
            //printf("info string invalid FEN\n");
            return;
        }
        update_occ(&global_pos);
        update_zobrist_key(&global_pos);
        promotion_pending = -1;
    }
}

/* 
static void uci_loop() {
    char line[4096];
    bool running = true;

    init_board_wasm(); // Start position by default

    while (running && fgets(line, sizeof(line), stdin)) {
        if (strncmp(line, "uci", 3) == 0) {
            printf("id name MarcosChessEngine\n");
            printf("id author Marcos Marques\n");
            printf("uciok\n");
        }
        else if (strncmp(line, "isready", 7) == 0) {
            printf("readyok\n");
        }
        else if (strncmp(line, "ucinewgame", 10) == 0) {
            init_board_wasm();
        }
        else if (strncmp(line, "position startpos", 17) == 0) {
            init_board_wasm();
            char *moves = strstr(line, "moves");
            if (moves) {
                moves += 6; // skip "moves "
                char *token = strtok(moves, " \n");
                while (token) {
                    make_move_from_uci(token); // you need to implement this parser
                    token = strtok(NULL, " \n");
                }
            }
        }
        else if (strncmp(line, "position fen", 12) == 0) {
            char fen[128];
            sscanf(line + 13, "%127[^\n]", fen);
            load_fen_wasm(fen); // implement if not yet in your engine
        }
        else if (strncmp(line, "go", 2) == 0) {
            char bestmove[8];
            find_ai_move_wasm(bestmove); // adapt to fill bestmove in UCI format
            printf("bestmove %s\n", bestmove);
        }
        else if (strncmp(line, "stop", 4) == 0) {
            // In your simple engine, search is blocking, so nothing needed here
        }
        else if (strncmp(line, "quit", 4) == 0) {
            running = false;
        }
        fflush(stdout);
    }
}

int main() {
    uci_loop();
    return 0;
}
 */