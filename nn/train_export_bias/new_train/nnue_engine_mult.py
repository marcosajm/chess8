#!/usr/bin/env python3
"""
NNUE Chess Engine - Complete Working Version
Includes a multi-model tournament that ranks every nnue*_weights_wasm.bin model.
"""


import chess
import chess.engine
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Dict
import time
import os
import re
import glob
import random
import warnings
warnings.filterwarnings('ignore')


# ============== Configuration ==============
class EngineConfig:
    NNUE_INPUT_DIM = 780
    NNUE_H1 = 256
    NNUE_H2 = 64
    NNUE_H3 = 32
    NNUE_OUT = 1
    
    WEIGHTS_FILE = "nnue_weights_wasm.bin"
    BIAS_FILE = "nnue_bias.bin"
    
    TIME_LIMIT = 1.0
    STOCKFISH_PATH = "stockfish"


# ============== NNUE Model ==============
class NNUEProduction(nn.Module):
    def __init__(self):
        super(NNUEProduction, self).__init__()
        self.fc1 = nn.Linear(EngineConfig.NNUE_INPUT_DIM, EngineConfig.NNUE_H1)
        self.fc2 = nn.Linear(EngineConfig.NNUE_H1, EngineConfig.NNUE_H2)
        self.fc3 = nn.Linear(EngineConfig.NNUE_H2, EngineConfig.NNUE_H3)
        self.fc4 = nn.Linear(EngineConfig.NNUE_H3, EngineConfig.NNUE_OUT)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x.squeeze(-1)


# ============== NNUE Evaluator ==============
class NNUEEvaluator:
    def __init__(self, weights_file: str, bias_file: str):
        self.model = NNUEProduction()
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_weights(weights_file, bias_file)
        self.model = self.model.to(self.device)
        print(f"NNUE evaluator ready on {self.device}")
        self.cache = {}
    
    def load_weights(self, weights_file: str, bias_file: str):
        weights = np.fromfile(weights_file, dtype=np.float32)
        biases = np.fromfile(bias_file, dtype=np.float32)
        
        w1_size = EngineConfig.NNUE_INPUT_DIM * EngineConfig.NNUE_H1
        b1_size = EngineConfig.NNUE_H1
        w2_size = EngineConfig.NNUE_H1 * EngineConfig.NNUE_H2
        b2_size = EngineConfig.NNUE_H2
        w3_size = EngineConfig.NNUE_H2 * EngineConfig.NNUE_H3
        b3_size = EngineConfig.NNUE_H3
        w4_size = EngineConfig.NNUE_H3 * EngineConfig.NNUE_OUT
        b4_size = EngineConfig.NNUE_OUT
        
        idx = 0
        w1 = weights[idx:idx+w1_size].reshape(EngineConfig.NNUE_H1, EngineConfig.NNUE_INPUT_DIM)
        idx += w1_size + b1_size
        w2 = weights[idx:idx+w2_size].reshape(EngineConfig.NNUE_H2, EngineConfig.NNUE_H1)
        idx += w2_size + b2_size
        w3 = weights[idx:idx+w3_size].reshape(EngineConfig.NNUE_H3, EngineConfig.NNUE_H2)
        idx += w3_size + b3_size
        w4 = weights[idx:idx+w4_size].reshape(EngineConfig.NNUE_OUT, EngineConfig.NNUE_H3)
        
        b1 = biases[0:b1_size]
        b2 = biases[b1_size:b1_size+b2_size]
        b3 = biases[b1_size+b2_size:b1_size+b2_size+b3_size]
        b4 = biases[b1_size+b2_size+b3_size:b1_size+b2_size+b3_size+b4_size]
        
        self.model.fc1.weight.data = torch.tensor(w1, dtype=torch.float32)
        self.model.fc1.bias.data = torch.tensor(b1, dtype=torch.float32)
        self.model.fc2.weight.data = torch.tensor(w2, dtype=torch.float32)
        self.model.fc2.bias.data = torch.tensor(b2, dtype=torch.float32)
        self.model.fc3.weight.data = torch.tensor(w3, dtype=torch.float32)
        self.model.fc3.bias.data = torch.tensor(b3, dtype=torch.float32)
        self.model.fc4.weight.data = torch.tensor(w4, dtype=torch.float32)
        self.model.fc4.bias.data = torch.tensor(b4, dtype=torch.float32)
        
        print(f"✅ Weights loaded: {len(weights)+len(biases)} params")
    
    def board_to_features(self, board: chess.Board) -> np.ndarray:
        features = np.zeros(EngineConfig.NNUE_INPUT_DIM, dtype=np.float32)
        piece_map = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                color_offset = 0 if piece.color == chess.WHITE else 6
                piece_idx = color_offset + piece_map[piece.piece_type]
                features[square * 12 + piece_idx] = 1.0
        
        return features
    
    def evaluate(self, board: chess.Board) -> float:
        board_key = board.fen()
        if board_key in self.cache:
            return self.cache[board_key]
        
        features = self.board_to_features(board)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            score = self.model(features_tensor.unsqueeze(0)).item()
        
        self.cache[board_key] = score
        if len(self.cache) > 10000:
            self.cache.clear()
        
        return score * 50.0
    
    def evaluate_board(self, board: chess.Board) -> float:
        score = self.evaluate(board)
        if board.turn == chess.BLACK:
            return -score
        return score


# ============== Chess Engine ==============
class NNUE_Engine:
    def __init__(self, weights_file: str = None, bias_file: str = None):
        if weights_file is None:
            weights_file = EngineConfig.WEIGHTS_FILE
        if bias_file is None:
            bias_file = EngineConfig.BIAS_FILE
            
        self.evaluator = NNUEEvaluator(weights_file, bias_file)
        self.board = chess.Board()
        self.move_history = []
        self.eval_history = []
        self.debug = False
    
    def reset(self):
        self.board = chess.Board()
        self.move_history = []
        self.eval_history = []
        # Also clear the eval cache so a new game isn't polluted by
        # stale scores from a previous game with the same positions.
        self.evaluator.cache.clear()
    
    def make_move(self, move: chess.Move) -> bool:
        try:
            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_history.append(move)
                self.eval_history.append(self.evaluator.evaluate_board(self.board))
                return True
            return False
        except:
            return False
    
    def get_best_move(self, time_limit: float = EngineConfig.TIME_LIMIT) -> Optional[chess.Move]:
        """Get best move - GUARANTEED to return a move if any legal moves exist"""
        if self.board.is_game_over():
            return None
        
        moves = list(self.board.legal_moves)
        if not moves:
            return None
        
        # If only one move, return it immediately
        if len(moves) == 1:
            return moves[0]
        
        # Evaluate all moves with simple NNUE evaluation.
        # evaluate_board returns the score from the perspective of the side to
        # move in the PUSHED position (the opponent), so we negate to express it
        # from our own perspective. Highest = best for the side to move.
        move_scores = []
        for move in moves:
            self.board.push(move)
            score = self.evaluator.evaluate_board(self.board)
            self.board.pop()
            score = -score
            move_scores.append((move, score))
        
        # Sort by score (highest first - best for the side to move)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Debug output
        if self.debug and len(moves) <= 5:
            print(f"\n  Move evaluation:")
            for move, score in move_scores[:5]:
                print(f"    {move.uci()}: {score:.2f}")
        
        # ALWAYS return the best move (first in sorted list)
        best_move = move_scores[0][0]
        best_score = move_scores[0][1]
        
        # Try simple one-ply lookahead for the top 3 moves.
        # This helps avoid blunders. NOTE: only runs when time_limit > 0.1;
        # tournament matches use a small time_limit so this is skipped there
        # (keeps the shared board safe from half-applied push/pop on error).
        if len(moves) >= 3 and time_limit > 0.1:
            try:
                for move, score in move_scores[:3]:
                    self.board.push(move)
                    
                    if not self.board.is_game_over():
                        opponent_moves = list(self.board.legal_moves)
                        if opponent_moves:
                            best_response_score = float('-inf') if self.board.turn == chess.WHITE else float('inf')
                            
                            for opp_move in opponent_moves[:5]:
                                self.board.push(opp_move)
                                opp_score = self.evaluator.evaluate_board(self.board)
                                self.board.pop()
                                
                                if self.board.turn == chess.WHITE:
                                    opp_score = -opp_score
                                
                                if self.board.turn == chess.WHITE:
                                    best_response_score = min(best_response_score, opp_score)
                                else:
                                    best_response_score = max(best_response_score, opp_score)
                            
                            if best_response_score > best_score + 50:
                                best_move = move
                    
                    self.board.pop()
            except Exception as e:
                if self.debug:
                    print(f"  Lookahead failed: {e}")
                pass
        
        return best_move
    
    def play_move(self, time_limit: float = EngineConfig.TIME_LIMIT) -> Optional[chess.Move]:
        """Get and make a move"""
        move = self.get_best_move(time_limit)
        if move:
            self.make_move(move)
        return move
    
    def evaluate_position(self) -> float:
        return self.evaluator.evaluate_board(self.board)


# ============== Stockfish Integration ==============
class StockfishGame:
    def __init__(self, stockfish_path: str = EngineConfig.STOCKFISH_PATH):
        self.stockfish_path = stockfish_path
        self.engine = None
        self._start_engine()
    
    def _start_engine(self):
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            print(f"✅ Stockfish started")
            return True
        except Exception as e:
            print(f"❌ Stockfish error: {e}")
            return False
    
    def close(self):
        if self.engine:
            try:
                self.engine.quit()
            except:
                pass
    
    def get_stockfish_move(self, board: chess.Board, time_limit: float = 0.3) -> Optional[chess.Move]:
        if not self.engine:
            return None
        
        try:
            board_copy = board.copy()
            result = self.engine.play(board_copy, chess.engine.Limit(time=time_limit))
            move = result.move
            
            if move in board.legal_moves:
                return move
            return None
        except Exception as e:
            return None
    
    def play_game(self, nnue_engine: NNUE_Engine, stockfish_color: chess.Color = chess.BLACK) -> dict:
        """Play a complete game using nnue_engine.board as the SINGLE source of truth."""
        nnue_engine.reset()
        board = nnue_engine.board
        
        moves = []
        move_count = 0
        max_moves = 100
        
        print("  Playing...", end="", flush=True)
        
        while not board.is_game_over() and move_count < max_moves:
            try:
                if board.turn == stockfish_color:
                    move = self.get_stockfish_move(board, time_limit=0.3)
                    if move and move in board.legal_moves:
                        board.push(move)
                        moves.append(('Stockfish', move))
                    else:
                        print(" SF failed", end="", flush=True)
                        break
                else:
                    if move_count == 0:
                        nnue_engine.debug = True
                    
                    move = nnue_engine.get_best_move(time_limit=0.5)
                    nnue_engine.debug = False
                    
                    if move and move in board.legal_moves:
                        board.push(move)
                        moves.append(('NNUE', move))
                    else:
                        print(" NNUE failed", end="", flush=True)
                        break
                
                move_count += 1
                if move_count % 10 == 0:
                    print(".", end="", flush=True)
                    
            except Exception as e:
                print(f" Error: {e}", end="", flush=True)
                break
        
        print(" done!", flush=True)
        
        final_eval = 0
        if len(moves) > 0:
            try:
                final_eval = nnue_engine.evaluate_position()
            except:
                pass
        
        return {
            'result': board.result(),
            'move_count': len(moves),
            'moves': moves,
            'final_eval': final_eval
        }


# ============== Model discovery ==============
def _model_number(name: str) -> int:
    """Extract the model number from a filename. 'nnue' (no digits) -> 1."""
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else 1


def discover_nnue_models(weights_dir: str = ".") -> List[Dict]:
    """Find every nnue*_weights_wasm.bin file and locate its bias file.

    For a weights file 'nnue3_weights_wasm.bin' we look for, in order:
      1. nnue3_bias.bin        (per-model bias)
      2. nnue_bias.bin          (shared bias fallback)
    Models with no findable bias file are still returned with bias=None so the
    caller can warn/skip them.
    """
    paths = glob.glob(os.path.join(weights_dir, "nnue*_weights_wasm.bin"))
    models = []
    for p in paths:
        base = os.path.basename(p)
        stem = base[:-len("_weights_wasm.bin")]  # e.g. 'nnue3' or 'nnue'
        bias_candidates = [
            os.path.join(weights_dir, f"{stem}_bias.bin"),
            os.path.join(weights_dir, "nnue_bias.bin"),
        ]
        bias = next((c for c in bias_candidates if os.path.exists(c)), None)
        models.append({
            "name": base,
            "weights": p,
            "bias": bias,
            "number": _model_number(base),
        })
    models.sort(key=lambda m: (m["number"], m["name"]))
    return models


# ============== Tournament ==============
class NNUETournament:
    """Play every NNUE model against the others and produce an ordered ranking.

    Modes:
      - round_robin(): each model plays every other model (home & away).
        This directly answers "which NNUE is the strongest".
      - vs_stockfish_benchmark(): each model plays N games vs Stockfish.
        Gives an absolute strength rating relative to Stockfish.
    """

    SF_ELO = 1500.0  # fixed reference rating for Stockfish in benchmark mode

    def __init__(self, weights_dir: str = "."):
        all_models = discover_nnue_models(weights_dir)
        self.models = []
        for m in all_models:
            if m["bias"] is None:
                print(f"⚠️  No bias file found for {m['name']} - skipping")
            else:
                self.models.append(m)
        if not self.models:
            raise FileNotFoundError(
                f"No usable NNUE models found in {weights_dir} "
                "(expected nnue*_weights_wasm.bin + matching bias file)"
            )
        self.results = {m["name"]: {"wins": 0, "draws": 0, "losses": 0,
                                    "score": 0.0, "played": 0} for m in self.models}
        self.elo = {m["name"]: 1000.0 for m in self.models}
        self.method = ""
        self._engine_cache: Dict[str, NNUE_Engine] = {}
    
    # ---------- engine caching ----------
    def _get_engine(self, model: Dict) -> NNUE_Engine:
        """Load each model's engine once and reuse it across matches."""
        key = model["weights"]
        if key not in self._engine_cache:
            self._engine_cache[key] = NNUE_Engine(
                weights_file=model["weights"], bias_file=model["bias"]
            )
        return self._engine_cache[key]
    
    # ---------- core match ----------
    def play_match(self, white_model: Dict, black_model: Dict,
                   time_limit: float = 0.05, max_moves: int = 60) -> str:
        """Play one game between two NNUE models. Returns '1-0'/'0-1'/'1/2-1/2'.

        Both engines share ONE board object, so they always see the same
        position (this is the fix for the old 'NNUE failed after 2 moves' bug).
        """
        w_eng = self._get_engine(white_model)
        b_eng = self._get_engine(black_model)
        board = chess.Board()
        w_eng.board = board  # shared board -> both engines stay in sync
        b_eng.board = board
        
        for _ in range(max_moves):
            if board.is_game_over():
                break
            eng = w_eng if board.turn == chess.WHITE else b_eng
            try:
                move = eng.get_best_move(time_limit=time_limit)
            except Exception:
                move = None
            if not move or move not in board.legal_moves:
                # side to move couldn't produce a legal move -> it loses
                return "0-1" if board.turn == chess.WHITE else "1-0"
            board.push(move)
        
        res = board.result()
        return res if res != "*" else "1/2-1/2"  # move-limit reached -> draw
    
    def play_match_vs_stockfish(self, nnue_model: Dict, sf: StockfishGame,
                                nnue_color: chess.Color, time_limit: float = 0.05,
                                sf_time: float = 0.3, max_moves: int = 60) -> str:
        """Play one game between a NNUE model and Stockfish."""
        eng = self._get_engine(nnue_model)
        board = chess.Board()
        eng.board = board
        
        for _ in range(max_moves):
            if board.is_game_over():
                break
            if board.turn == nnue_color:
                try:
                    move = eng.get_best_move(time_limit=time_limit)
                except Exception:
                    move = None
            else:
                move = sf.get_stockfish_move(board, time_limit=sf_time)
            if not move or move not in board.legal_moves:
                return "0-1" if board.turn == chess.WHITE else "1-0"
            board.push(move)
        
        res = board.result()
        return res if res != "*" else "1/2-1/2"
    
    # ---------- elo ----------
    @staticmethod
    def _update_elo(elo: Dict[str, float], a: str, b: str, score_a: float, k: float = 32.0):
        """Update Elo for two players. score_a is a's result (1.0/0.5/0.0)."""
        Ra, Rb = elo[a], elo[b]
        Ea = 1.0 / (1.0 + 10 ** ((Rb - Ra) / 400.0))
        elo[a] = Ra + k * (score_a - Ea)
        elo[b] = Rb + k * ((1.0 - score_a) - (1.0 - Ea))
    
    def _record(self, white_name: str, black_name: str, result: str):
        sw = self.results[white_name]
        sb = self.results[black_name]
        sw["played"] += 1
        sb["played"] += 1
        if result == "1-0":
            sw["wins"] += 1; sw["score"] += 1.0; sb["losses"] += 1; score_a = 1.0
        elif result == "0-1":
            sb["wins"] += 1; sb["score"] += 1.0; sw["losses"] += 1; score_a = 0.0
        else:  # draw (or move-limit draw)
            sw["draws"] += 1; sw["score"] += 0.5
            sb["draws"] += 1; sb["score"] += 0.5; score_a = 0.5
        self._update_elo(self.elo, white_name, black_name, score_a)
    
    # ---------- modes ----------
    def round_robin(self, games_per_pairing: int = 2, time_limit: float = 0.05,
                    max_moves: int = 60):
        """Each model plays every other model. games_per_pairing=2 = home & away."""
        from itertools import combinations
        self.method = f"Round-robin ({games_per_pairing} games/pairing)"
        
        if len(self.models) < 2:
            print("❌ Need at least 2 models for a round-robin.")
            return
        
        pairings = list(combinations(range(len(self.models)), 2))
        total = len(pairings) * games_per_pairing
        done = 0
        print(f"\nRound-robin: {len(self.models)} models, {len(pairings)} pairings, "
              f"{total} games (time_limit={time_limit}s, max_moves={max_moves})")
        print("-" * 80)
        
        for (i, j) in pairings:
            a, b = self.models[i], self.models[j]
            for g in range(games_per_pairing):
                # alternate colors for fairness
                if games_per_pairing > 1 and g % 2 == 1:
                    white, black = b, a
                else:
                    white, black = a, b
                result = self.play_match(white, black, time_limit, max_moves)
                self._record(white["name"], black["name"], result)
                done += 1
                print(f"  [{done:>3}/{total}] {white['name']:<26} (W) vs "
                      f"{black['name']:<26} (B) -> {result}")
        
        print("-" * 80)
        print(f"Done: {done} games played.")
    
    def vs_stockfish_benchmark(self, num_games: int = 5, time_limit: float = 0.05,
                               sf_time: float = 0.3, max_moves: int = 60):
        """Each model plays num_games vs Stockfish (alternating colors)."""
        self.method = f"vs Stockfish ({num_games} games/model)"
        sf = StockfishGame()
        if not sf.engine:
            print("❌ Stockfish not available - cannot benchmark.")
            return
        
        total = len(self.models) * num_games
        done = 0
        print(f"\nBenchmark: {len(self.models)} models x {num_games} games vs Stockfish "
              f"({total} games)")
        print("-" * 80)
        
        for model in self.models:
            for g in range(num_games):
                nnue_color = chess.WHITE if g % 2 == 0 else chess.BLACK
                result = self.play_match_vs_stockfish(
                    model, sf, nnue_color, time_limit, sf_time, max_moves
                )
                # convert result (white's perspective) -> NNUE perspective
                if result == "1-0":
                    outcome = "win" if nnue_color == chess.WHITE else "loss"
                    score_nnue = 1.0 if nnue_color == chess.WHITE else 0.0
                elif result == "0-1":
                    outcome = "win" if nnue_color == chess.BLACK else "loss"
                    score_nnue = 1.0 if nnue_color == chess.BLACK else 0.0
                else:
                    outcome = "draw"
                    score_nnue = 0.5
                
                st = self.results[model["name"]]
                st["played"] += 1
                if outcome == "win":
                    st["wins"] += 1; st["score"] += 1.0
                elif outcome == "draw":
                    st["draws"] += 1; st["score"] += 0.5
                else:
                    st["losses"] += 1
                
                # Elo update vs fixed Stockfish rating (SF Elo is not changed)
                Ra = self.elo[model["name"]]
                Ea = 1.0 / (1.0 + 10 ** ((self.SF_ELO - Ra) / 400.0))
                self.elo[model["name"]] = Ra + 32.0 * (score_nnue - Ea)
                
                done += 1
                color_char = "W" if nnue_color == chess.WHITE else "B"
                print(f"  [{done:>3}/{total}] {model['name']:<26} ({color_char}) "
                      f"vs Stockfish -> {result} [{outcome}]")
        
        sf.close()
        print("-" * 80)
        print(f"Done: {done} games played.")
    
    # ---------- ranking ----------
    def ranking(self, by: str = "elo") -> List[tuple]:
        rows = []
        for m in self.models:
            name = m["name"]
            st = self.results[name]
            played = st["played"]
            score = st["score"]
            winrate = (score / played) if played else 0.0
            rows.append((name, played, st["wins"], st["draws"], st["losses"],
                         score, winrate, self.elo[name]))
        if by == "elo":
            rows.sort(key=lambda x: (x[7], x[5]), reverse=True)
        else:
            rows.sort(key=lambda x: (x[5], x[7]), reverse=True)
        return rows
    
    def print_ranking(self, by: str = "elo"):
        rows = self.ranking(by)
        print("\n" + "=" * 92)
        print(f"🏆 NNUE Tournament Ranking  —  {self.method}")
        print("=" * 92)
        print(f"{'#':>3}  {'Model':<28} {'P':>4} {'W':>4} {'D':>4} {'L':>4} "
              f"{'Score':>7} {'Win%':>6} {'Elo':>7}")
        print("-" * 92)
        medals = ["🥇", "🥈", "🥉"]
        for i, (name, played, w, d, l, score, winrate, elo) in enumerate(rows, 1):
            tag = medals[i - 1] + " " if i <= 3 else "   "
            print(f"{i:>3} {tag}{name:<26} {played:>4} {w:>4} {d:>4} {l:>4} "
                  f"{score:>7.1f} {winrate*100:>5.1f}% {elo:>7.0f}")
        print("=" * 92)
        if rows:
            print(f"Best NNUE: {rows[0][0]}  (Elo {rows[0][7]:.0f}, "
                  f"score {rows[0][5]:.1f}/{rows[0][1]})")

class EnhancedNNUETournament(NNUETournament):
    def __init__(self, weights_dir: str = "."):
        super().__init__(weights_dir)
        # Track additional metrics per model
        self.metrics = {
            m["name"]: {
                "total_games": 0,
                "total_moves": 0,
                "avg_moves": 0.0,
                "total_eval_score": 0.0,
                "avg_eval_score": 0.0,
                "max_eval": float('-inf'),
                "min_eval": float('inf'),
                "total_game_time": 0.0,
                "avg_game_time": 0.0,
                "positions_evaluated": 0,
                "avg_positions_per_game": 0.0,
                "total_capture_moves": 0,
                "avg_captures_per_game": 0.0,
                "total_promotions": 0,
                "avg_promotions_per_game": 0.0,
                "total_checks": 0,
                "avg_checks_per_game": 0.0,
                "total_checkmates": 0,
                "avg_checkmates_per_game": 0.0,
                # New: Track wins/losses by color
                "wins_as_white": 0,
                "wins_as_black": 0,
                "losses_as_white": 0,
                "losses_as_black": 0,
                "draws_as_white": 0,
                "draws_as_black": 0,
            } for m in self.models
        }
        self.game_logs = []
        self.stockfish_games = []  # Track games vs Stockfish separately
    
    def play_match_with_metrics(self, white_model: Dict, black_model: Dict,
                                time_limit: float = 0.05, max_moves: int = 60) -> Dict:
        """Play one game and collect detailed metrics"""
        start_time = time.time()
        
        w_eng = self._get_engine(white_model)
        b_eng = self._get_engine(black_model)
        board = chess.Board()
        w_eng.board = board
        b_eng.board = board
        
        moves_played = []
        positions_evaluated = 0
        captures = 0
        promotions = 0
        checks = 0
        checkmates = 0
        evals = []
        white_moves = 0
        black_moves = 0
        
        for move_num in range(max_moves):
            if board.is_game_over():
                break
            
            # Track who made the move
            is_white_turn = board.turn == chess.WHITE
            
            eng = w_eng if is_white_turn else b_eng
            move = eng.get_best_move(time_limit=time_limit)
            
            if not move or move not in board.legal_moves:
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                break
            
            # Track move statistics
            if board.is_capture(move):
                captures += 1
            if move.promotion:
                promotions += 1
            if board.gives_check(move):
                checks += 1
            if board.is_checkmate():
                checkmates += 1
            
            board.push(move)
            moves_played.append(move)
            
            if is_white_turn:
                white_moves += 1
            else:
                black_moves += 1
            
            # Evaluate position after move
            eval_score = eng.evaluator.evaluate_board(board)
            evals.append(eval_score)
            positions_evaluated += 1
        
        result = board.result() if board.is_game_over() else "1/2-1/2"
        elapsed = time.time() - start_time
        
        return {
            'result': result,
            'white': white_model["name"],
            'black': black_model["name"],
            'moves': len(moves_played),
            'white_moves': white_moves,
            'black_moves': black_moves,
            'time': elapsed,
            'captures': captures,
            'promotions': promotions,
            'checks': checks,
            'checkmates': checkmates,
            'positions_evaluated': positions_evaluated,
            'evals': evals,
            'avg_eval': sum(evals) / len(evals) if evals else 0,
            'max_eval': max(evals) if evals else 0,
            'min_eval': min(evals) if evals else 0,
            'final_eval': evals[-1] if evals else 0,
            'is_stockfish_game': False  # Mark as regular match
        }
    
    def play_match_vs_stockfish_with_metrics(self, nnue_model: Dict, sf: StockfishGame,
                                             nnue_color: chess.Color, time_limit: float = 0.05,
                                             sf_time: float = 0.3, max_moves: int = 60) -> Dict:
        """Play one game vs Stockfish and collect detailed metrics"""
        start_time = time.time()
        
        eng = self._get_engine(nnue_model)
        board = chess.Board()
        eng.board = board
        
        moves_played = []
        positions_evaluated = 0
        captures = 0
        promotions = 0
        checks = 0
        checkmates = 0
        evals = []
        nnue_moves = 0
        sf_moves = 0
        
        for move_num in range(max_moves):
            if board.is_game_over():
                break
            
            is_nnue_turn = board.turn == nnue_color
            
            if is_nnue_turn:
                move = eng.get_best_move(time_limit=time_limit)
                nnue_moves += 1
            else:
                move = sf.get_stockfish_move(board, time_limit=sf_time)
                sf_moves += 1
            
            if not move or move not in board.legal_moves:
                result = "0-1" if board.turn == chess.WHITE else "1-0"
                break
            
            # Track move statistics
            if board.is_capture(move):
                captures += 1
            if move.promotion:
                promotions += 1
            if board.gives_check(move):
                checks += 1
            if board.is_checkmate():
                checkmates += 1
            
            board.push(move)
            moves_played.append(move)
            
            # Evaluate position after move (from NNUE perspective)
            eval_score = eng.evaluator.evaluate_board(board)
            evals.append(eval_score)
            positions_evaluated += 1
        
        result = board.result() if board.is_game_over() else "1/2-1/2"
        elapsed = time.time() - start_time
        
        # Convert result to NNUE perspective
        if result == "1-0":
            nnue_won = (nnue_color == chess.WHITE)
        elif result == "0-1":
            nnue_won = (nnue_color == chess.BLACK)
        else:
            nnue_won = None  # Draw
        
        return {
            'result': result,
            'nnue_model': nnue_model["name"],
            'nnue_color': 'White' if nnue_color == chess.WHITE else 'Black',
            'nnue_won': nnue_won,
            'moves': len(moves_played),
            'nnue_moves': nnue_moves,
            'sf_moves': sf_moves,
            'time': elapsed,
            'captures': captures,
            'promotions': promotions,
            'checks': checks,
            'checkmates': checkmates,
            'positions_evaluated': positions_evaluated,
            'evals': evals,
            'avg_eval': sum(evals) / len(evals) if evals else 0,
            'max_eval': max(evals) if evals else 0,
            'min_eval': min(evals) if evals else 0,
            'final_eval': evals[-1] if evals else 0,
            'is_stockfish_game': True
        }
    
    def update_metrics(self, game_data: Dict):
        """Update metrics for both players based on game data"""
        if game_data.get('is_stockfish_game', False):
            # Game vs Stockfish
            self._update_stockfish_metrics(game_data)
        else:
            # Regular match between two NNUE models
            self._update_match_metrics(game_data)
        
        # Store game log
        self.game_logs.append(game_data)
    
    def _update_match_metrics(self, game_data: Dict):
        """Update metrics for a match between two NNUE models"""
        white = game_data['white']
        black = game_data['black']
        
        # Update for White
        self._update_single_metrics(white, game_data, is_white=True)
        # Update for Black
        self._update_single_metrics(black, game_data, is_white=False)
        
        # Track color-specific results
        result = game_data['result']
        if result == "1-0":
            self.metrics[white]["wins_as_white"] += 1
            self.metrics[black]["losses_as_black"] += 1
        elif result == "0-1":
            self.metrics[white]["losses_as_white"] += 1
            self.metrics[black]["wins_as_black"] += 1
        else:
            self.metrics[white]["draws_as_white"] += 1
            self.metrics[black]["draws_as_black"] += 1
    
    def _update_stockfish_metrics(self, game_data: Dict):
        """Update metrics for a game vs Stockfish"""
        model_name = game_data['nnue_model']
        metrics = self.metrics[model_name]
        
        # Basic stats
        metrics["total_games"] += 1
        metrics["total_moves"] += game_data['moves']
        metrics["avg_moves"] = metrics["total_moves"] / metrics["total_games"]
        
        # Evaluation stats
        if game_data['evals']:
            metrics["total_eval_score"] += sum(game_data['evals'])
            metrics["avg_eval_score"] = metrics["total_eval_score"] / metrics["total_games"]
            metrics["max_eval"] = max(metrics["max_eval"], max(game_data['evals']))
            metrics["min_eval"] = min(metrics["min_eval"], min(game_data['evals']))
        
        # Position evaluation count
        metrics["positions_evaluated"] += game_data['positions_evaluated']
        metrics["avg_positions_per_game"] = metrics["positions_evaluated"] / metrics["total_games"]
        
        # Move quality metrics (all NNUE moves since SF moves are tracked separately)
        metrics["total_capture_moves"] += game_data['captures'] * 0.5  # Approximate
        metrics["total_promotions"] += game_data['promotions'] * 0.5
        metrics["total_checks"] += game_data['checks'] * 0.5
        metrics["total_checkmates"] += game_data['checkmates'] * 0.5
        
        metrics["avg_captures_per_game"] = metrics["total_capture_moves"] / metrics["total_games"]
        metrics["avg_promotions_per_game"] = metrics["total_promotions"] / metrics["total_games"]
        metrics["avg_checks_per_game"] = metrics["total_checks"] / metrics["total_games"]
        metrics["avg_checkmates_per_game"] = metrics["total_checkmates"] / metrics["total_games"]
        
        # Time stats
        metrics["total_game_time"] += game_data['time']
        metrics["avg_game_time"] = metrics["total_game_time"] / metrics["total_games"]
    
    def _update_single_metrics(self, model_name: str, game_data: Dict, is_white: bool):
        """Update metrics for a single player in a match"""
        metrics = self.metrics[model_name]
        
        # Basic stats
        metrics["total_games"] += 1
        metrics["total_moves"] += game_data['moves']
        metrics["avg_moves"] = metrics["total_moves"] / metrics["total_games"]
        
        # Evaluation stats
        if game_data['evals']:
            if is_white:
                eval_scores = game_data['evals']
            else:
                # For black, negate evaluations (from white's perspective)
                eval_scores = [-e for e in game_data['evals']]
            
            metrics["total_eval_score"] += sum(eval_scores)
            metrics["avg_eval_score"] = metrics["total_eval_score"] / metrics["total_games"]
            metrics["max_eval"] = max(metrics["max_eval"], max(eval_scores))
            metrics["min_eval"] = min(metrics["min_eval"], min(eval_scores))
        
        # Position evaluation count
        metrics["positions_evaluated"] += game_data['positions_evaluated']
        metrics["avg_positions_per_game"] = metrics["positions_evaluated"] / metrics["total_games"]
        
        # Move quality metrics
        if is_white:
            metrics["total_capture_moves"] += game_data['captures']
            metrics["total_promotions"] += game_data['promotions']
            metrics["total_checks"] += game_data['checks']
            metrics["total_checkmates"] += game_data['checkmates']
        else:
            metrics["total_capture_moves"] += game_data['captures'] * 0.5
            metrics["total_promotions"] += game_data['promotions'] * 0.5
            metrics["total_checks"] += game_data['checks'] * 0.5
            metrics["total_checkmates"] += game_data['checkmates'] * 0.5
        
        metrics["avg_captures_per_game"] = metrics["total_capture_moves"] / metrics["total_games"]
        metrics["avg_promotions_per_game"] = metrics["total_promotions"] / metrics["total_games"]
        metrics["avg_checks_per_game"] = metrics["total_checks"] / metrics["total_games"]
        metrics["avg_checkmates_per_game"] = metrics["total_checkmates"] / metrics["total_games"]
        
        # Time stats
        metrics["total_game_time"] += game_data['time']
        metrics["avg_game_time"] = metrics["total_game_time"] / metrics["total_games"]
    
    def round_robin(self, games_per_pairing: int = 2, time_limit: float = 0.05,
                    max_moves: int = 60):
        """Override to use enhanced match tracking"""
        from itertools import combinations
        self.method = f"Round-robin ({games_per_pairing} games/pairing)"
        
        if len(self.models) < 2:
            print("❌ Need at least 2 models for a round-robin.")
            return
        
        pairings = list(combinations(range(len(self.models)), 2))
        total = len(pairings) * games_per_pairing
        done = 0
        
        print(f"\nRound-robin: {len(self.models)} models, {len(pairings)} pairings, "
              f"{total} games (time_limit={time_limit}s, max_moves={max_moves})")
        print("-" * 80)
        
        for (i, j) in pairings:
            a, b = self.models[i], self.models[j]
            for g in range(games_per_pairing):
                if games_per_pairing > 1 and g % 2 == 1:
                    white, black = b, a
                else:
                    white, black = a, b
                result = self.play_match(white, black, time_limit, max_moves)
                done += 1
                print(f"  [{done:>3}/{total}] {white['name']:<26} (W) vs "
                      f"{black['name']:<26} (B) -> {result}")
        
        print("-" * 80)
        print(f"Done: {done} games played.")
    
    def vs_stockfish_benchmark(self, num_games: int = 5, time_limit: float = 0.05,
                               sf_time: float = 0.3, max_moves: int = 60):
        """Override to use enhanced Stockfish match tracking"""
        self.method = f"vs Stockfish ({num_games} games/model)"
        sf = StockfishGame()
        if not sf.engine:
            print("❌ Stockfish not available - cannot benchmark.")
            return
        
        total = len(self.models) * num_games
        done = 0
        print(f"\nBenchmark: {len(self.models)} models x {num_games} games vs Stockfish "
              f"({total} games)")
        print("-" * 80)
        
        for model in self.models:
            for g in range(num_games):
                nnue_color = chess.WHITE if g % 2 == 0 else chess.BLACK
                
                # Use enhanced match function
                game_data = self.play_match_vs_stockfish_with_metrics(
                    model, sf, nnue_color, time_limit, sf_time, max_moves
                )
                
                # Update metrics
                self.update_metrics(game_data)
                
                # Update results (for ranking)
                result = game_data['result']
                nnue_won = game_data['nnue_won']
                
                # Convert result to NNUE perspective for recording
                if nnue_won is True:
                    outcome = "win"
                    score_nnue = 1.0
                elif nnue_won is False:
                    outcome = "loss"
                    score_nnue = 0.0
                else:
                    outcome = "draw"
                    score_nnue = 0.5
                
                # Update parent class results
                st = self.results[model["name"]]
                st["played"] += 1
                if outcome == "win":
                    st["wins"] += 1
                    st["score"] += 1.0
                elif outcome == "draw":
                    st["draws"] += 1
                    st["score"] += 0.5
                else:
                    st["losses"] += 1
                
                # Update Elo
                Ra = self.elo[model["name"]]
                Ea = 1.0 / (1.0 + 10 ** ((self.SF_ELO - Ra) / 400.0))
                self.elo[model["name"]] = Ra + 32.0 * (score_nnue - Ea)
                
                done += 1
                color_char = "W" if nnue_color == chess.WHITE else "B"
                print(f"  [{done:>3}/{total}] {model['name']:<26} ({color_char}) "
                      f"vs Stockfish -> {result} [{outcome}]")
        
        sf.close()
        print("-" * 80)
        print(f"Done: {done} games played.")
    
    def print_detailed_ranking(self, by: str = "elo"):
        """Print ranking with extended performance metrics"""
        rows = self.ranking(by)
        
        print("\n" + "=" * 170)
        print(f"🏆 NNUE Tournament Detailed Ranking  —  {self.method}")
        print("=" * 170)
        print(f"{'#':>3}  {'Model':<28} {'P':>4} {'W':>4} {'D':>4} {'L':>4} "
              f"{'Score':>7} {'Win%':>6} {'Elo':>7} {'Avg Moves':>9} {'Avg Eval':>9} "
              f"{'Avg Cap':>7} {'Avg Chk':>7} {'W%W':>5} {'W%B':>5}")
        print("-" * 170)
        
        medals = ["🥇", "🥈", "🥉"]
        for i, (name, played, w, d, l, score, winrate, elo) in enumerate(rows, 1):
            tag = medals[i - 1] + " " if i <= 3 else "   "
            metrics = self.metrics[name]
            
            # Calculate win rates by color
            w_as_white = metrics["wins_as_white"]
            games_as_white = w_as_white + metrics["losses_as_white"] + metrics["draws_as_white"]
            w_as_black = metrics["wins_as_black"]
            games_as_black = w_as_black + metrics["losses_as_black"] + metrics["draws_as_black"]
            
            winrate_white = (w_as_white / games_as_white * 100) if games_as_white > 0 else 0
            winrate_black = (w_as_black / games_as_black * 100) if games_as_black > 0 else 0
            
            print(f"{i:>3} {tag}{name:<26} {played:>4} {w:>4} {d:>4} {l:>4} "
                  f"{score:>7.1f} {winrate*100:>5.1f}% {elo:>7.0f} "
                  f"{metrics['avg_moves']:>8.1f} {metrics['avg_eval_score']:>8.1f} "
                  f"{metrics['avg_captures_per_game']:>6.1f} {metrics['avg_checks_per_game']:>6.1f} "
                  f"{winrate_white:>4.0f}% {winrate_black:>4.0f}%")
        
        print("=" * 170)
        
        # Print category winners
        self.print_category_winners()
    
    def print_category_winners(self):
        """Print category winners for all tracked metrics"""
        print("\n🏆 Category Winners:")
        print("-" * 80)
        
        categories = [
            ("Best Win Rate", "winrate", max),
            ("Highest Elo Rating", "elo", max),
            ("Highest Avg Evaluation", "avg_eval_score", max),
            ("Longest Games", "avg_moves", max),
            ("Most Captures/Game", "avg_captures_per_game", max),
            ("Most Checks/Game", "avg_checks_per_game", max),
            ("Most Promotions/Game", "avg_promotions_per_game", max),
            ("Best as White", "wins_as_white", max),
            ("Best as Black", "wins_as_black", max),
        ]
        
        for category_name, metric_key, func in categories:
            if metric_key == "winrate":
                best = max(self.results.items(), key=lambda x: 
                          x[1]['score'] / x[1]['played'] if x[1]['played'] > 0 else 0)
                value = (best[1]['score'] / best[1]['played'] * 100) if best[1]['played'] > 0 else 0
                print(f"  {category_name}: {best[0]} ({value:.1f}%)")
            elif metric_key == "elo":
                best = max(self.elo.items(), key=lambda x: x[1])
                print(f"  {category_name}: {best[0]} ({best[1]:.0f})")
            elif metric_key in ["wins_as_white", "wins_as_black"]:
                best = max(self.metrics.items(), key=lambda x: x[1][metric_key])
                print(f"  {category_name}: {best[0]} ({best[1][metric_key]} wins)")
            else:
                best = func(self.metrics.items(), key=lambda x: x[1][metric_key])
                print(f"  {category_name}: {best[0]} ({best[1][metric_key]:.2f})")
    
    def print_game_stats_summary(self):
        """Print summary statistics for all games played"""
        total_games = len(self.game_logs)
        if total_games == 0:
            print("No games played yet.")
            return
        
        # Separate games by type
        nnue_vs_nnue = [g for g in self.game_logs if not g.get('is_stockfish_game', False)]
        nnue_vs_sf = [g for g in self.game_logs if g.get('is_stockfish_game', False)]
        
        print("\n📊 Game Statistics Summary")
        print("=" * 80)
        print(f"Total Games: {total_games}")
        print(f"  NNUE vs NNUE: {len(nnue_vs_nnue)}")
        print(f"  NNUE vs Stockfish: {len(nnue_vs_sf)}")
        
        if total_games > 0:
            print(f"\nOverall Statistics:")
            print(f"  Average Game Length: {sum(g['moves'] for g in self.game_logs) / total_games:.1f} moves")
            print(f"  Average Game Time: {sum(g['time'] for g in self.game_logs) / total_games:.2f}s")
            print(f"  Total Captures: {sum(g['captures'] for g in self.game_logs)}")
            print(f"  Total Promotions: {sum(g['promotions'] for g in self.game_logs)}")
            print(f"  Total Checks: {sum(g['checks'] for g in self.game_logs)}")
            print(f"  Total Checkmates: {sum(g['checkmates'] for g in self.game_logs)}")
            
            # Result distribution
            results = {}
            for game in self.game_logs:
                results[game['result']] = results.get(game['result'], 0) + 1
            
            print("\nResult Distribution:")
            for result in ['1-0', '0-1', '1/2-1/2']:
                count = results.get(result, 0)
                print(f"  {result}: {count} ({count/total_games*100:.1f}%)")
            
            # Evaluation score distribution
            all_evals = []
            for game in self.game_logs:
                all_evals.extend(game['evals'])
            
            if all_evals:
                print(f"\nEvaluation Score Statistics (from white's perspective):")
                print(f"  Min: {min(all_evals):.2f}")
                print(f"  Max: {max(all_evals):.2f}")
                print(f"  Mean: {sum(all_evals)/len(all_evals):.2f}")
                print(f"  Std Dev: {(sum((x - sum(all_evals)/len(all_evals))**2 for x in all_evals) / len(all_evals))**0.5:.2f}")
    
    def export_stats_to_csv(self, filename: str = "tournament_stats.csv"):
        """Export all statistics to CSV for further analysis"""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'Model', 'Games', 'Wins', 'Draws', 'Losses', 
                'Score', 'WinRate', 'Elo',
                'AvgMoves', 'AvgEval', 'MaxEval', 'MinEval',
                'AvgCaptures', 'AvgPromotions', 'AvgChecks', 'AvgCheckmates',
                'AvgPositionsEval', 'AvgGameTime',
                'WinsAsWhite', 'WinsAsBlack', 'WinRateWhite', 'WinRateBlack'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for model in self.models:
                name = model["name"]
                stats = self.results[name]
                metrics = self.metrics[name]
                
                # Calculate win rates by color
                w_as_white = metrics["wins_as_white"]
                games_as_white = w_as_white + metrics["losses_as_white"] + metrics["draws_as_white"]
                w_as_black = metrics["wins_as_black"]
                games_as_black = w_as_black + metrics["losses_as_black"] + metrics["draws_as_black"]
                
                winrate_white = (w_as_white / games_as_white * 100) if games_as_white > 0 else 0
                winrate_black = (w_as_black / games_as_black * 100) if games_as_black > 0 else 0
                
                row = {
                    'Model': name,
                    'Games': stats['played'],
                    'Wins': stats['wins'],
                    'Draws': stats['draws'],
                    'Losses': stats['losses'],
                    'Score': stats['score'],
                    'WinRate': (stats['score'] / stats['played'] * 100) if stats['played'] > 0 else 0,
                    'Elo': self.elo[name],
                    'AvgMoves': metrics['avg_moves'],
                    'AvgEval': metrics['avg_eval_score'],
                    'MaxEval': metrics['max_eval'],
                    'MinEval': metrics['min_eval'],
                    'AvgCaptures': metrics['avg_captures_per_game'],
                    'AvgPromotions': metrics['avg_promotions_per_game'],
                    'AvgChecks': metrics['avg_checks_per_game'],
                    'AvgCheckmates': metrics['avg_checkmates_per_game'],
                    'AvgPositionsEval': metrics['avg_positions_per_game'],
                    'AvgGameTime': metrics['avg_game_time'],
                    'WinsAsWhite': w_as_white,
                    'WinsAsBlack': w_as_black,
                    'WinRateWhite': winrate_white,
                    'WinRateBlack': winrate_black
                }
                writer.writerow(row)
        
        print(f"✅ Statistics exported to {filename}")
    
    # Override parent methods to use enhanced versions
    def play_match(self, white_model: Dict, black_model: Dict,
                   time_limit: float = 0.05, max_moves: int = 60) -> str:
        """Override to use enhanced match with metrics"""
        game_data = self.play_match_with_metrics(white_model, black_model, time_limit, max_moves)
        self.update_metrics(game_data)
        
        # Also record the result in the parent class
        self._record(white_model["name"], black_model["name"], game_data['result'])
        
        return game_data['result']


# Update the tournament runner function
def run_enhanced_tournament():
    """Run enhanced tournament with both methods and detailed stats"""
    print("\n" + "=" * 80)
    print("🏆 Enhanced NNUE Tournament - With Detailed Metrics")
    print("=" * 80)
    
    try:
        tournament = EnhancedNNUETournament(weights_dir=".")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    print(f"\nFound {len(tournament.models)} NNUE model(s):")
    for m in tournament.models:
        print(f"  • {m['name']:<28} bias: {os.path.basename(m['bias'])}")
    
    if len(tournament.models) < 2:
        print("\n⚠️  Need at least 2 models to run a tournament.")
        return
    
    # Method selection
    print("\nMethod:")
    print("  1. Round-robin  — each NNUE plays every other NNUE (recommended)")
    print("  2. vs Stockfish — each NNUE plays N games vs Stockfish")
    method = input("Choose method (1-2, default 1): ").strip() or "1"
    
    if method == "2":
        n = input("Games per model vs Stockfish (default 5): ").strip()
        n = int(n) if n else 5
        tournament.vs_stockfish_benchmark(num_games=n)
    else:
        g = input("Games per pairing (default 2 = home & away): ").strip()
        g = int(g) if g else 2
        tournament.round_robin(games_per_pairing=g)
    
    # Print comprehensive results
    tournament.print_detailed_ranking()
    tournament.print_game_stats_summary()
    
    # Export to CSV
    export = input("\nExport stats to CSV? (y/n, default n): ").strip().lower()
    if export == 'y':
        tournament.export_stats_to_csv()
        
def run_enhanced_tournament():
    """Run tournament with extended metrics"""
    print("\n" + "=" * 80)
    print("🏆 Enhanced NNUE Tournament - With Detailed Metrics")
    print("=" * 80)
    
    try:
        tournament = EnhancedNNUETournament(weights_dir=".")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    print(f"\nFound {len(tournament.models)} NNUE model(s):")
    for m in tournament.models:
        print(f"  • {m['name']:<28} bias: {os.path.basename(m['bias'])}")
    
    if len(tournament.models) < 2:
        print("\n⚠️  Need at least 2 models to run a tournament.")
        return
    
    # Run tournament
    games_per_pairing = 2  # Home and away
    print(f"\nRunning round-robin with {games_per_pairing} games per pairing...")
    tournament.round_robin(games_per_pairing=games_per_pairing)
    
    # Print detailed results
    tournament.print_detailed_ranking()
    tournament.print_game_stats_summary()
    
    # Export to CSV
    tournament.export_stats_to_csv()

# ============== Test Functions ==============
def run_stockfish_test(num_games: int = 5):
    """Test NNUE against Stockfish"""
    print("\n" + "=" * 80)
    print("🏆 NNUE vs Stockfish")
    print("=" * 80)
    
    nnue_engine = NNUE_Engine()
    stockfish = StockfishGame()
    
    if not stockfish.engine:
        print("❌ Stockfish not available")
        return
    
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
    total_moves = 0
    
    print(f"\nPlaying {num_games} games (NNUE White, Stockfish Black)")
    print("-" * 80)
    
    for game_num in range(num_games):
        print(f"\nGame {game_num + 1}/{num_games}")
        nnue_engine.reset()
        game_data = stockfish.play_game(nnue_engine, stockfish_color=chess.BLACK)
        
        result = game_data['result']
        results[result] = results.get(result, 0) + 1
        total_moves += game_data['move_count']
        
        if result == '1-0':
            print(f"  ✅ NNUE wins in {game_data['move_count']} moves!")
        elif result == '0-1':
            print(f"  ❌ Stockfish wins in {game_data['move_count']} moves")
        else:
            print(f"  🤝 Draw in {game_data['move_count']} moves")
        
        if game_data['final_eval']:
            print(f"  Final eval: {game_data['final_eval']:.2f}")
    
    print("\n" + "=" * 80)
    print("📊 Results")
    print("=" * 80)
    print(f"NNUE wins:  {results['1-0']} ({results['1-0']/num_games*100:.1f}%)")
    print(f"Stockfish:  {results['0-1']} ({results['0-1']/num_games*100:.1f}%)")
    print(f"Draws:      {results['1/2-1/2']} ({results['1/2-1/2']/num_games*100:.1f}%)")
    print(f"Avg moves:  {total_moves/max(1, num_games):.1f}")
    
    stockfish.close()


def interactive_play():
    """Play interactively against NNUE"""
    print("\n" + "=" * 80)
    print("🤖 Play against NNUE")
    print("=" * 80)
    print("You: White, NNUE: Black")
    print("Enter moves in UCI format (e.g., 'e2e4')")
    print("-" * 80)
    
    nnue_engine = NNUE_Engine()
    board = chess.Board()
    nnue_engine.board = board  # single shared board
    
    while not board.is_game_over():
        print("\n" + board.__str__())
        print(f"\nYour turn (White)")
        eval_score = nnue_engine.evaluate_position()
        print(f"Evaluation: {eval_score:.2f}")
        
        user_move = input("Your move: ").strip().lower()
        if user_move == 'quit':
            break
        
        try:
            move = chess.Move.from_uci(user_move)
            if move in board.legal_moves:
                board.push(move)
            else:
                print("❌ Illegal move")
                continue
        except Exception:
            print("❌ Invalid format")
            continue
        
        if board.is_game_over():
            break
        
        print("\n🤖 NNUE thinking...", end="", flush=True)
        move = nnue_engine.get_best_move(time_limit=1.0)
        if move and move in board.legal_moves:
            board.push(move)
            print(f" NNUE plays: {move.uci()}")
        else:
            print(" NNUE couldn't move")
            break
    
    print("\n" + "=" * 80)
    print(f"Result: {board.result()}")
    print("=" * 80)


def performance_test():
    """Test evaluation speed"""
    print("\n" + "=" * 80)
    print("⚡ Performance Test")
    print("=" * 80)
    
    nnue_engine = NNUE_Engine()
    
    positions = []
    board = chess.Board()
    for _ in range(200):
        moves = list(board.legal_moves)
        if moves and len(board.move_stack) < 30:
            board.push(random.choice(moves))
        else:
            board = chess.Board()
        positions.append(board.copy())
    
    start_time = time.time()
    for board in positions:
        nnue_engine.evaluator.evaluate_board(board)
    
    elapsed = time.time() - start_time
    speed = len(positions) / elapsed
    
    print(f"Evaluated {len(positions)} positions")
    print(f"Time: {elapsed:.2f}s")
    print(f"Speed: {speed:.1f} pos/sec")
    print(f"Avg: {elapsed/len(positions)*1000:.2f} ms/pos")


def self_play_analysis():
    """Analyze self-play game"""
    print("\n" + "=" * 80)
    print("📈 Self-play Analysis")
    print("=" * 80)
    
    nnue_engine = NNUE_Engine()
    moves = []
    
    for i in range(30):
        if nnue_engine.board.is_game_over():
            break
        move = nnue_engine.play_move(time_limit=0.3)
        if move:
            eval_score = nnue_engine.evaluate_position()
            moves.append((i+1, move.uci(), eval_score))
            print(f"{i+1:2d}. {move.uci():6s} (eval: {eval_score:7.2f})")
        else:
            break
    
    print(f"\nResult: {nnue_engine.board.result()}")


def evaluate_fen():
    """Evaluate a FEN position"""
    fen = input("Enter FEN (or press Enter for start): ").strip()
    if not fen:
        fen = chess.STARTING_FEN
    
    try:
        board = chess.Board(fen)
        nnue_engine = NNUE_Engine()
        nnue_engine.board = board
        eval_score = nnue_engine.evaluate_position()
        print(f"\n{board}\n")
        print(f"NNUE eval: {eval_score:.2f} centipawns")
        print(f"Advantage: {eval_score/100:.2f} pawns")
    except Exception as e:
        print(f"❌ Error: {e}")


def run_tournament():
    """Discover all NNUE models and rank them via a tournament."""
    print("\n" + "=" * 80)
    print("🏆 NNUE Tournament - rank all models")
    print("=" * 80)
    
    try:
        tournament = NNUETournament(weights_dir=".")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    print(f"\nFound {len(tournament.models)} NNUE model(s):")
    for m in tournament.models:
        print(f"  • {m['name']:<28} bias: {os.path.basename(m['bias'])}")
    
    if len(tournament.models) < 2:
        print("\n⚠️  Need at least 2 models to run a tournament.")
        return
    
    print("\nMethod:")
    print("  1. Round-robin  — each NNUE plays every other NNUE (recommended)")
    print("  2. vs Stockfish — each NNUE plays N games vs Stockfish")
    method = input("Choose method (1-2, default 1): ").strip() or "1"
    
    if method == "2":
        n = input("Games per model vs Stockfish (default 5): ").strip()
        n = int(n) if n else 5
        tournament.vs_stockfish_benchmark(num_games=n)
    else:
        g = input("Games per pairing (default 2 = home & away): ").strip()
        g = int(g) if g else 2
        tournament.round_robin(games_per_pairing=g)
    
    tournament.print_ranking()


# ============== Main ==============
def main():
    print("=" * 80)
    print("♟️ NNUE Chess Engine with Enhanced Tournament")
    print("=" * 80)
    
    while True:
        print("\n" + "=" * 80)
        print("📋 MENU")
        print("=" * 80)
        print("1. Play against NNUE")
        print("2. Test vs Stockfish")
        print("3. Performance test")
        print("4. Evaluate FEN position")
        print("5. Self-play analysis")
        print("6. Tournament - rank all NNUE models (with detailed stats)")
        print("7. Exit")
        
        choice = input("\nChoose (1-7): ").strip()
        
        if choice == '1':
            interactive_play()
        elif choice == '2':
            n = input("Number of games (default 3): ").strip()
            run_stockfish_test(int(n) if n else 3)
        elif choice == '3':
            performance_test()
        elif choice == '4':
            evaluate_fen()
        elif choice == '5':
            self_play_analysis()
        elif choice == '6':
            run_enhanced_tournament()
        elif choice == '7':
            print("Goodbye! 👋")
            break
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()