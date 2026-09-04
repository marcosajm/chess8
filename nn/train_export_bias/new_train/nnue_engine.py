#!/usr/bin/env python3
"""
NNUE Chess Engine - Complete Working Version
"""


import chess
import chess.engine
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List
import time
import os
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
        # NOTE: the side to move is `self.board.turn`. evaluate_board returns
        # the score from the perspective of the side to move, so after pushing a
        # candidate move the position is from the OPPONENT's perspective and we
        # must negate to get it back into our own perspective.
        side_to_move = self.board.turn
        move_scores = []
        for move in moves:
            self.board.push(move)
            score = self.evaluator.evaluate_board(self.board)
            self.board.pop()
            # evaluate_board already negates for black-to-move positions, so the
            # value is from the perspective of the side to move in the pushed
            # position (i.e. the opponent). Negate to express it from our side.
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
        
        # Try simple one-ply lookahead for the top 3 moves
        # This helps avoid blunders
        if len(moves) >= 3 and time_limit > 0.1:
            try:
                # Check if any move is significantly better with one-ply lookahead
                for move, score in move_scores[:3]:
                    self.board.push(move)
                    
                    if not self.board.is_game_over():
                        # Get opponent's best response
                        opponent_moves = list(self.board.legal_moves)
                        if opponent_moves:
                            # Evaluate opponent's best response
                            best_response_score = float('-inf') if self.board.turn == chess.WHITE else float('inf')
                            
                            for opp_move in opponent_moves[:5]:
                                self.board.push(opp_move)
                                opp_score = self.evaluator.evaluate_board(self.board)
                                self.board.pop()
                                
                                # Adjust score based on who's to move
                                if self.board.turn == chess.WHITE:
                                    opp_score = -opp_score
                                
                                if self.board.turn == chess.WHITE:
                                    # White wants to maximize, so opponent's best response minimizes
                                    best_response_score = min(best_response_score, opp_score)
                                else:
                                    # Black wants to minimize, so opponent's best response maximizes
                                    best_response_score = max(best_response_score, opp_score)
                            
                            # If this move is significantly better than the simple best move
                            if best_response_score > best_score + 50:
                                best_move = move
                    
                    self.board.pop()
            except Exception as e:
                # If lookahead fails, use the simple best move
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
            # Use a copy of the board to avoid state issues
            board_copy = board.copy()
            result = self.engine.play(board_copy, chess.engine.Limit(time=time_limit))
            move = result.move
            
            # Verify the move is legal on the original board
            if move in board.legal_moves:
                return move
            return None
        except Exception as e:
            return None
    
    def play_game(self, nnue_engine: NNUE_Engine, stockfish_color: chess.Color = chess.BLACK) -> dict:
        """Play a complete game.

        IMPORTANT: use nnue_engine.board as the SINGLE source of truth so the
        NNUE engine and Stockfish always see the same position. Previously a
        separate local board was kept in sync only for NNUE's own moves, which
        caused Stockfish's moves to be missed by the NNUE board and made the
        engine return an illegal move on move 3 ('NNUE failed' after 2 moves).
        """
        nnue_engine.reset()
        board = nnue_engine.board  # single shared board
        
        moves = []
        move_count = 0
        max_moves = 100
        
        print("  Playing...", end="", flush=True)
        
        while not board.is_game_over() and move_count < max_moves:
            try:
                if board.turn == stockfish_color:
                    # Stockfish's turn
                    move = self.get_stockfish_move(board, time_limit=0.3)
                    if move and move in board.legal_moves:
                        board.push(move)  # shared board -> NNUE sees it too
                        moves.append(('Stockfish', move))
                    else:
                        print(" SF failed", end="", flush=True)
                        break
                else:
                    # NNUE's turn
                    if move_count == 0:
                        nnue_engine.debug = True
                    
                    move = nnue_engine.get_best_move(time_limit=0.5)
                    nnue_engine.debug = False
                    
                    if move and move in board.legal_moves:
                        board.push(move)  # shared board
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
        
        # Get final evaluation (board == nnue_engine.board, so this is correct)
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
    
    # Generate random positions
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


# ============== Main ==============
def main():
    print("=" * 80)
    print("♟️ NNUE Chess Engine")
    print("=" * 80)
    
    if not os.path.exists(EngineConfig.WEIGHTS_FILE):
        print(f"❌ Weights not found: {EngineConfig.WEIGHTS_FILE}")
        return
    
    while True:
        print("\n" + "=" * 80)
        print("📋 MENU")
        print("=" * 80)
        print("1. Play against NNUE")
        print("2. Test vs Stockfish")
        print("3. Performance test")
        print("4. Evaluate FEN position")
        print("5. Self-play analysis")
        print("6. Exit")
        
        choice = input("\nChoose (1-6): ").strip()
        
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
            print("Goodbye! 👋")
            break
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
