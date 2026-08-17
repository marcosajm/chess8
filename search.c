// search.c
#include "search.h"
#include "movegen.h"
#include "occupancy.h"
#include "nnue.h"
#include <limits.h>
#include <stdio.h>

// External NNUE instance (defined in nnue.c)
extern NNUEWeights g_nnue;

// Load once before any search
bool nnue_initialized = false;

// Simple material fallback (not used once NNUE is stable, but kept for backup)
int piece_values[PIECE_NB] = {100, 320, 330, 500, 900, 20000};

// --- Evaluation using NNUE ---
int evaluate_position(const Position *pos) {
   
    float score = nnue_forward_pos(&g_nnue, pos);
    int cp_score = (int)(score * 1000.0f);

    // Perspective correction
    if (pos->side_to_move == BLACK)
        cp_score = -cp_score;

    //printf("[EVAL] side=%s score=%.4f → %d cp\n", pos->side_to_move == WHITE ? "WHITE" : "BLACK", score, cp_score);

    return cp_score;
}

// --- Minimax with alpha-beta pruning ---
int minimax(Position *pos, int depth, int alpha, int beta, int maximizingPlayer) {
    if (depth <= 0) {
        int eval = evaluate_position(pos);
        printf("  [MINIMAX d=%d] leaf eval=%d\n", depth, eval);
        return eval;
    }

    MoveList legal_moves;
    filter_legal_moves(pos, &legal_moves);

    if (legal_moves.count == 0) {
        int king_sq = -1;
        if (pos->pieces[pos->side_to_move][KING])
            king_sq = __builtin_ctzll(pos->pieces[pos->side_to_move][KING]);

        bool in_check = (king_sq != -1 && is_square_attacked(pos, king_sq, !pos->side_to_move));
        if (in_check) {
            int mateScore = maximizingPlayer ? (-1000000 + depth) : (1000000 - depth);
            //printf("  [TERMINAL] Checkmate detected → %d\n", mateScore);
            return mateScore;
        }
        //printf("  [TERMINAL] Stalemate detected → 0\n");
        return 0;
    }

    if (maximizingPlayer) {
        int maxEval = INT_MIN;
        for (int i = 0; i < legal_moves.count; i++) {
            Position child = *pos;
            apply_move(&child, legal_moves.moves[i]);

            int eval = minimax(&child, depth - 1, alpha, beta, 0);
            if (eval > maxEval) {
                maxEval = eval;
                //printf("  [MAX d=%d] New best eval=%d move=%d\n", depth, eval, legal_moves.moves[i]);
            }
            if (eval > alpha) alpha = eval;
            if (beta <= alpha) {
                //printf("  [MAX d=%d] Beta cutoff\n", depth);
                break;
            }
        }
        return maxEval;
    } else {
        int minEval = INT_MAX;
        for (int i = 0; i < legal_moves.count; i++) {
            Position child = *pos;
            apply_move(&child, legal_moves.moves[i]);

            int eval = minimax(&child, depth - 1, alpha, beta, 1);
            if (eval < minEval) {
                minEval = eval;
                //printf("  [MIN d=%d] New best eval=%d move=%d\n", depth, eval, legal_moves.moves[i]);
            }
            if (eval < beta) beta = eval;
            if (beta <= alpha) {
                //printf("  [MIN d=%d] Alpha cutoff\n", depth);
                break;
            }
        }
        return minEval;
    }
}

// --- Top-level search function ---
Move search_best_move(Position *pos, int depth) {
    if (depth <= 0) {
        depth = 1;
    }

    MoveList legal_moves;
    generate_all_moves(pos, &legal_moves);
    filter_legal_moves(pos, &legal_moves);

    if (legal_moves.count == 0) {
        return 0;
    }

    int bestScore = INT_MIN;
    Move bestMove = 0;

    int alpha = INT_MIN;
    int beta = INT_MAX;

    //printf("[SEARCH] Starting depth=%d, legal_moves=%d\n", depth, legal_moves.count);
   
    for (int i = 0; i < legal_moves.count; i++) {
        //printf("  move[%d]=%d\n", i, legal_moves.moves[i]);
        Position child = *pos;
        apply_move(&child, legal_moves.moves[i]);
        update_occ(&child);

        int score = minimax(&child, depth - 1, alpha, beta, 0);

        //printf("[SEARCH] Move=%d Score=%d\n", legal_moves.moves[i], score);

        if (score > bestScore) {
            bestScore = score;
            bestMove = legal_moves.moves[i];
            //printf("[SEARCH] New best move=%d score=%d\n", bestMove, bestScore);
        }
        if (score > alpha) alpha = score;
    }

    printf("[SEARCH] Final best move=%d score=%d\n", bestMove, bestScore);
    return bestMove;
}