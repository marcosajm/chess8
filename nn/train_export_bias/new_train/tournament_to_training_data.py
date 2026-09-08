#!/usr/bin/env python3
"""
NNUE Tournament → Training Data Generator
Plays every discovered NNUE model vs Stockfish and saves the games
with Stockfish evaluations in the same .bin format as the data-generator.
"""

import chess
import chess.engine
import numpy as np
import torch
import torch.nn as nn
import struct
import time
import os
import re
import glob
from dataclasses import dataclass
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================
class Config:
    # Network
    NNUE_INPUT_DIM = 780
    NNUE_H1 = 256
    NNUE_H2 = 64
    NNUE_H3 = 32
    NNUE_OUT = 1

    # Tournament settings
    GAMES_PER_MODEL = 1          # how many games each NNUE plays vs Stockfish
    NNUE_TIME = 1.08             # seconds the NNUE is allowed to think
    SF_TIME = 1.25               # seconds Stockfish is allowed when moving
    SF_DEPTH = 24                # depth used only for the *stored* evaluation
    MAX_MOVES = 160
    STOCKFISH_PATH = "stockfish" # change if needed

    # Output
    OUTPUT_PREFIX = "tournament_sf_data"


# ============================================================
# NNUE Model (same architecture as your engine)
# ============================================================
class NNUEProduction(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(Config.NNUE_INPUT_DIM, Config.NNUE_H1)
        self.fc2 = nn.Linear(Config.NNUE_H1, Config.NNUE_H2)
        self.fc3 = nn.Linear(Config.NNUE_H2, Config.NNUE_H3)
        self.fc4 = nn.Linear(Config.NNUE_H3, Config.NNUE_OUT)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x.squeeze(-1)


# ============================================================
# Feature extraction (copied from data-generator – keep identical)
# ============================================================
def calculate_tactical_threats(board: chess.Board) -> float:
    score = 0.0
    turn = board.turn
    value_map = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                 chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}

    for sq in range(64):
        piece = board.piece_at(sq)
        if piece and piece.color == turn:
            if board.is_attacked_by(not turn, sq):
                value = value_map.get(piece.piece_type, 0)
                score += value / 100.0
                attackers = board.attackers(not turn, sq)
                score += len(attackers) * 0.1

            defenders = board.attackers(turn, sq)
            attackers = board.attackers(not turn, sq)
            if len(attackers) > len(defenders):
                value = value_map.get(piece.piece_type, 0)
                score += (value / 100.0) * (len(attackers) - len(defenders))
    return max(-1.0, min(1.0, score))


def calculate_king_safety(board: chess.Board) -> float:
    score = 0.0
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is None:
            continue
        attackers = 0
        for sq in range(64):
            p = board.piece_at(sq)
            if p and p.color != color and board.is_attacked_by(color, sq):
                attackers += 1

        pawn_shield = 0
        king_rank = chess.square_rank(king_sq)
        king_file = chess.square_file(king_sq)
        for df in [-1, 0, 1]:
            f = king_file + df
            if 0 <= f < 8:
                r = king_rank + (1 if color == chess.WHITE else -1)
                if 0 <= r < 8:
                    sq = chess.square(f, r)
                    p = board.piece_at(sq)
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        pawn_shield += 1
        king_score = (pawn_shield / 3.0) - (attackers / 4.0)
        score += king_score
    return max(-1.0, min(1.0, score / 2.0))


def featurize_board_prod(board: chess.Board) -> np.ndarray:
    features = np.zeros(780, dtype=np.float32)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    # 768 standard piece-square features
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            side = 1 if piece.color == chess.BLACK else 0
            piece_idx = piece_map[piece.piece_type]
            idx = side * (6 * 64) + piece_idx * 64 + sq
            features[idx] = 1.0

    idx = 768
    # castling
    features[idx]     = float(board.has_kingside_castling_rights(chess.WHITE))
    features[idx + 1] = float(board.has_queenside_castling_rights(chess.WHITE))
    features[idx + 2] = float(board.has_kingside_castling_rights(chess.BLACK))
    features[idx + 3] = float(board.has_queenside_castling_rights(chess.BLACK))
    idx += 4
    # en passant
    if board.ep_square is not None:
        features[idx]     = chess.square_file(board.ep_square) / 7.0
        features[idx + 1] = chess.square_rank(board.ep_square) / 7.0
    idx += 2
    # clocks
    features[idx]     = min(board.halfmove_clock, 50) / 50.0
    features[idx + 1] = min(board.fullmove_number, 50) / 50.0
    idx += 2
    # side to move
    features[idx] = 1.0 if board.turn == chess.BLACK else 0.0
    idx += 1
    # material
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    white_mat = black_mat = 0
    for sq in range(64):
        p = board.piece_at(sq)
        if p:
            v = values.get(p.piece_type, 0)
            if p.color == chess.WHITE:
                white_mat += v
            else:
                black_mat += v
    features[idx] = (white_mat - black_mat) / 39.0
    idx += 1
    # tactical + king safety
    features[idx]     = calculate_tactical_threats(board)
    features[idx + 1] = calculate_king_safety(board)
    return features


# ============================================================
# NNUE Evaluator (loads your weights)
# ============================================================
class NNUEEvaluator:
    def __init__(self, weights_file: str, bias_file: str):
        self.model = NNUEProduction()
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load(weights_file, bias_file)
        self.model = self.model.to(self.device)
        self.cache = {}

    def _load(self, weights_file, bias_file):
        weights = np.fromfile(weights_file, dtype=np.float32)
        biases  = np.fromfile(bias_file, dtype=np.float32)

        sizes = [
            (Config.NNUE_INPUT_DIM * Config.NNUE_H1, Config.NNUE_H1),
            (Config.NNUE_H1 * Config.NNUE_H2, Config.NNUE_H2),
            (Config.NNUE_H2 * Config.NNUE_H3, Config.NNUE_H3),
            (Config.NNUE_H3 * Config.NNUE_OUT, Config.NNUE_OUT),
        ]
        idx = 0
        layers = [self.model.fc1, self.model.fc2, self.model.fc3, self.model.fc4]
        for (w_size, b_size), layer in zip(sizes, layers):
            w = weights[idx:idx + w_size].reshape(layer.out_features, layer.in_features)
            idx += w_size + b_size
            layer.weight.data = torch.tensor(w, dtype=torch.float32)

        b_idx = 0
        for (_, b_size), layer in zip(sizes, layers):
            b = biases[b_idx:b_idx + b_size]
            b_idx += b_size
            layer.bias.data = torch.tensor(b, dtype=torch.float32)

        print(f"  ✅ Loaded {os.path.basename(weights_file)}")

    def board_to_features(self, board: chess.Board) -> np.ndarray:
        # simple 780 one-hot (same as your original engine)
        features = np.zeros(Config.NNUE_INPUT_DIM, dtype=np.float32)
        piece_map = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                     chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                color_offset = 0 if piece.color == chess.WHITE else 6
                features[sq * 12 + color_offset + piece_map[piece.piece_type]] = 1.0
        return features

    def evaluate_board(self, board: chess.Board) -> float:
        key = board.fen()
        if key in self.cache:
            return self.cache[key]
        feats = self.board_to_features(board)
        t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model(t).item() * 50.0
        if board.turn == chess.BLACK:
            score = -score
        self.cache[key] = score
        if len(self.cache) > 8000:
            self.cache.clear()
        return score


class NNUE_Engine:
    def __init__(self, weights_file: str, bias_file: str):
        self.evaluator = NNUEEvaluator(weights_file, bias_file)
        self.board = chess.Board()

    def get_best_move(self, time_limit: float = 0.05) -> Optional[chess.Move]:
        if self.board.is_game_over():
            return None
        moves = list(self.board.legal_moves)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]

        scores = []
        for m in moves:
            self.board.push(m)
            s = -self.evaluator.evaluate_board(self.board)
            self.board.pop()
            scores.append((m, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]


# ============================================================
# Stockfish helper
# ============================================================
class StockfishHelper:
    def __init__(self, path: str = Config.STOCKFISH_PATH):
        self.engine = None
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
            self.engine.configure({"Hash": 128, "Threads": 2})
            print("✅ Stockfish ready")
        except Exception as e:
            print(f"❌ Cannot start Stockfish: {e}")

    def close(self):
        if self.engine:
            self.engine.quit()

    def get_move(self, board: chess.Board, time_limit: float = 0.25) -> Optional[chess.Move]:
        if not self.engine:
            return None
        try:
            result = self.engine.play(board, chess.engine.Limit(time=time_limit))
            return result.move if result.move in board.legal_moves else None
        except:
            return None

    def get_eval(self, board: chess.Board, depth: int = Config.SF_DEPTH) -> float:
        """Return score from White’s perspective in pawns."""
        if not self.engine:
            return 0.0
        try:
            info = self.engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].white().score()
            return (score / 100.0) if score is not None else 0.0
        except:
            return 0.0


# ============================================================
# Data structures
# ============================================================
@dataclass
class TrainingPosition:
    features: np.ndarray
    score: float
    result: float
    tactical_score: float


# ============================================================
# Model discovery
# ============================================================
def discover_models(weights_dir: str = ".") -> List[Dict]:
    paths = glob.glob(os.path.join(weights_dir, "nnue*_weights_wasm.bin"))
    models = []
    for p in paths:
        base = os.path.basename(p)
        stem = base[:-len("_weights_wasm.bin")]
        bias_candidates = [
            os.path.join(weights_dir, f"{stem}_bias.bin"),
            os.path.join(weights_dir, "nnue_bias.bin"),
        ]
        bias = next((c for c in bias_candidates if os.path.exists(c)), None)
        if bias is None:
            print(f"⚠️  Skipping {base} (no bias file)")
            continue
        num = int(re.search(r"(\d+)", base).group(1)) if re.search(r"(\d+)", base) else 1
        models.append({"name": base, "weights": p, "bias": bias, "number": num})
    models.sort(key=lambda m: (m["number"], m["name"]))
    return models


# ============================================================
# Main generator
# ============================================================
def generate_tournament_data(weights_dir: str = "."):
    print("=" * 80)
    print("🏆 NNUE Tournament → Training Data Generator")
    print("=" * 80)

    models = discover_models(weights_dir)
    if not models:
        print("❌ No usable NNUE models found.")
        return

    print(f"\nFound {len(models)} model(s):")
    for m in models:
        print(f"  • {m['name']}")

    sf = StockfishHelper()
    if not sf.engine:
        return

    all_positions: List[TrainingPosition] = []
    total_games = len(models) * Config.GAMES_PER_MODEL
    done = 0
    start = time.time()

    print(f"\nPlaying {Config.GAMES_PER_MODEL} games per model "
          f"({total_games} total) …")
    print("-" * 80)

    for model in models:
        eng = NNUE_Engine(model["weights"], model["bias"])

        for g in range(Config.GAMES_PER_MODEL):
            nnue_is_white = (g % 2 == 0)
            board = chess.Board()
            eng.board = board
            game_pos = []
            move_count = 0

            while not board.is_game_over() and move_count < Config.MAX_MOVES:
                # 1. Record position with Stockfish evaluation
                sf_score = sf.get_eval(board, depth=Config.SF_DEPTH)
                # convert to side-to-move perspective (same convention as data-gen)
                score_for_stm = sf_score if board.turn == chess.WHITE else -sf_score

                features = featurize_board_prod(board)
                tactical = calculate_tactical_threats(board) * 0.3

                game_pos.append(TrainingPosition(
                    features=features,
                    score=score_for_stm,
                    result=0.0,          # filled later
                    tactical_score=tactical
                ))

                # 2. Make a move
                if board.turn == (chess.WHITE if nnue_is_white else chess.BLACK):
                    move = eng.get_best_move(time_limit=Config.NNUE_TIME)
                else:
                    move = sf.get_move(board, time_limit=Config.SF_TIME)

                if move is None or move not in board.legal_moves:
                    break
                board.push(move)
                move_count += 1

            # 3. Game result from NNUE’s point of view
            result = 0.5
            if board.is_checkmate():
                # the side that just moved won
                winner_white = (board.turn == chess.BLACK)
                nnue_won = (nnue_is_white and winner_white) or (not nnue_is_white and not winner_white)
                result = 1.0 if nnue_won else 0.0
            elif (board.is_stalemate() or board.is_insufficient_material()
                  or board.is_seventyfive_moves() or board.is_fivefold_repetition()):
                result = 0.5

            for p in game_pos:
                p.result = result

            all_positions.extend(game_pos)
            done += 1
            color = "W" if nnue_is_white else "B"
            print(f"  [{done:>3}/{total_games}] {model['name']:<28} ({color}) "
                  f"→ {len(game_pos):3d} pos  result={result}")

    sf.close()

    # 4. Save binary file (identical format to your data-generator)
    ts = int(time.time())
    filename = f"{Config.OUTPUT_PREFIX}_d{Config.SF_DEPTH}_g{Config.GAMES_PER_MODEL}_{ts}.bin"

    with open(filename, "wb") as f:
        f.write(struct.pack("4sI", b"NNUE", len(all_positions)))
        for pos in all_positions:
            f.write(pos.features.tobytes())
            f.write(struct.pack("fff", pos.score, pos.result, pos.tactical_score))

    size_mb = os.path.getsize(filename) / (1024 * 1024)
    elapsed = time.time() - start

    print("-" * 80)
    print(f"✅ Done!")
    print(f"   Positions saved : {len(all_positions)}")
    print(f"   File            : {filename}")
    print(f"   Size            : {size_mb:.2f} MB")
    print(f"   Time            : {elapsed/60:.1f} min")
    print("=" * 80)


# ============================================================
if __name__ == "__main__":
    generate_tournament_data(weights_dir=".")