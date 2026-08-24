#!/usr/bin/env python3
"""
NNUE Data Generator - Separated from Training
Generates training data with configurable bot play styles
"""

import chess
import chess.engine
import numpy as np
import struct
import random
import time
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

ts = int(time.time())

# ============== Configuration ==============
class Config:
    # Network architecture - WASM optimized
    NNUE_INPUT_DIM = 780
    
    # Data generation
    DEPTH = 24
    NUM_GAMES = 80
    MAX_MOVES = 212
    STOCKFISH_PATH = "/usr/games/stockfish"
    
    # Stockfish skill levels (0-20)
    STOCKFISH_SKILL_LEVELS = [20]
    
    # Bot play style options:
    # 'worst' = always worst moves
    # 'average' = average moves
    # 'best' = always best moves
    # 'alternating' = 1 best move, then 2 worst moves (repeating)
    OUR_BOT_PLAY_STYLE = 'alternating'  # Options: 'worst', 'average', 'best', 'alternating'
    
    # Output file
    DATA_FILE = "training_data" + str(DEPTH) + str(NUM_GAMES)  + str(MAX_MOVES) +  str(ts) + "_prod.bin"

# ============== Feature Extraction ==============
def calculate_tactical_threats(board: chess.Board) -> float:
    """Calculate tactical threats for defensive awareness"""
    score = 0.0
    turn = board.turn
    
    # Count attacks on our pieces
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece and piece.color == turn:
            if board.is_attacked_by(not turn, sq):
                value_map = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                           chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
                value = value_map.get(piece.piece_type, 0)
                score += value / 100.0
                
                attackers = board.attackers(not turn, sq)
                score += len(attackers) * 0.1
    
    # Check for hanging pieces
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece and piece.color == turn:
            defenders = board.attackers(turn, sq)
            attackers = board.attackers(not turn, sq)
            
            if len(attackers) > len(defenders):
                value_map = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                           chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
                value = value_map.get(piece.piece_type, 0)
                score += (value / 100.0) * (len(attackers) - len(defenders))
    
    return max(-1, min(1, score))

def calculate_king_safety(board: chess.Board) -> float:
    """Calculate king safety score"""
    score = 0.0
    
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is None:
            continue
        
        attackers = 0
        for sq in range(64):
            if board.piece_at(sq) and board.piece_at(sq).color != color:
                if board.is_attacked_by(color, sq):
                    attackers += 1
        
        pawn_shield = 0
        king_rank = chess.square_rank(king_sq)
        king_file = chess.square_file(king_sq)
        
        for df in [-1, 0, 1]:
            file = king_file + df
            if 0 <= file < 8:
                if color == chess.WHITE:
                    rank = king_rank + 1
                else:
                    rank = king_rank - 1
                
                if 0 <= rank < 8:
                    sq = chess.square(file, rank)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        pawn_shield += 1
        
        king_score = (pawn_shield / 3.0) - (attackers / 4.0)
        score += king_score
    
    return max(-1, min(1, score / 2.0))

def featurize_board_prod(board: chess.Board) -> np.ndarray:
    """Production feature extraction with defensive features"""
    features = np.zeros(780, dtype=np.float32)
    
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    # Standard 768 features
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            side = 1 if piece.color == chess.BLACK else 0
            piece_idx = piece_map[piece.piece_type]
            idx = side * (6 * 64) + piece_idx * 64 + sq
            features[idx] = 1.0
    
    feature_idx = 768
    
    # 1-4: Castling rights
    features[feature_idx] = float(board.has_kingside_castling_rights(chess.WHITE))
    features[feature_idx + 1] = float(board.has_queenside_castling_rights(chess.WHITE))
    features[feature_idx + 2] = float(board.has_kingside_castling_rights(chess.BLACK))
    features[feature_idx + 3] = float(board.has_queenside_castling_rights(chess.BLACK))
    feature_idx += 4
    
    # 5-6: En passant
    if board.ep_square is not None:
        features[feature_idx] = chess.square_file(board.ep_square) / 7.0
        features[feature_idx + 1] = chess.square_rank(board.ep_square) / 7.0
    feature_idx += 2
    
    # 7: Half-move clock
    features[feature_idx] = min(board.halfmove_clock, 50) / 50.0
    feature_idx += 1
    
    # 8: Full move number
    features[feature_idx] = min(board.fullmove_number, 50) / 50.0
    feature_idx += 1
    
    # 9: Side to move
    features[feature_idx] = 1.0 if board.turn == chess.BLACK else 0.0
    feature_idx += 1
    
    # 10: Material balance
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                   chess.ROOK: 5, chess.QUEEN: 9}
    white_material = 0
    black_material = 0
    
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
    
    features[feature_idx] = (white_material - black_material) / 39.0
    feature_idx += 1
    
    # 11: Tactical threat indicator (defensive feature)
    features[feature_idx] = calculate_tactical_threats(board)
    feature_idx += 1
    
    # 12: King safety indicator
    features[feature_idx] = calculate_king_safety(board)
    
    return features

# ============== Bot Move Selector ==============
class OurBotMoveSelector:
    """
    Selects moves for OUR bot based on the configured play style:
    - 'worst': Always chooses the move with the minimum evaluation
    - 'average': Chooses moves in the middle of the evaluation range
    - 'best': Chooses the move with the maximum evaluation
    - 'alternating': 1 best move, then 2 worst moves (repeating pattern)
    """
    
    def __init__(self, style: str = 'worst'):
        self.style = style
        self.move_counter = 0  # Track moves for alternating pattern
    
    def select_move(self, board: chess.Board, engine: chess.engine.SimpleEngine, 
                    depth: int, is_opening: bool = False) -> Tuple[chess.Move, float]:
        """
        Select a move based on the specified style.
        
        Args:
            board: Current chess board
            engine: Stockfish engine instance (used for evaluation)
            depth: Analysis depth
            is_opening: If True, forces random move (100% random)
            
        Returns:
            Tuple of (selected_move, evaluation_score)
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0.0
        
        # If only one legal move, return it
        if len(legal_moves) == 1:
            return legal_moves[0], 0.0
        
        # ALWAYS PLAY RANDOM ON OPENING (first move)
        if is_opening:
            selected = random.choice(legal_moves)
            print(f"  🎲 OUR Bot playing RANDOM opening move")
            return selected, 0.0
        
        # For non-opening moves, use the configured style
        # Small chance of random move for variety (5%)
        if random.random() < 0.05:
            selected = random.choice(legal_moves)
            print(f"  🎲 OUR Bot playing RANDOM move (variety)")
            return selected, 0.0
        
        # Analyze all legal moves using Stockfish
        move_scores = []
        for move in legal_moves:
            try:
                # Make the move on a copy
                board_copy = board.copy()
                board_copy.push(move)
                
                # Analyze the resulting position using Stockfish
                analysis = engine.analyse(board_copy, limit=chess.engine.Limit(depth=depth))
                score = analysis['score'].white().score()
                
                if score is None:
                    score = 0
                
                # Adjust score based on whose turn it is
                if board.turn == chess.BLACK:
                    score = -score
                
                move_scores.append((move, score))
            except:
                move_scores.append((move, 0))
        
        if not move_scores:
            return random.choice(legal_moves), 0.0
        
        # Sort by score
        move_scores.sort(key=lambda x: x[1])
        
        # Handle alternating pattern
        if self.style == 'alternating':
            # Pattern: 1 best, 2 worst, 1 best, 2 worst, ...
            # Increment counter for each non-opening move
            self.move_counter += 1
            
            # Determine if this should be a best or worst move
            # Pattern: move 1 = best, moves 2-3 = worst, move 4 = best, moves 5-6 = worst, ...
            if self.move_counter % 2 == 5: #3 == 1
                # Best move (every 3rd move starting from 1)
                selected = move_scores[-1]
                print(f"  🎯 OUR Bot playing BEST move (alternating pattern #{self.move_counter})")
                return selected
            else:
                # Worst move (moves 2-3, 5-6, 8-9, ...)
                selected = move_scores[0]
                print(f"  🎯 OUR Bot playing WORST move (alternating pattern #{self.move_counter})")
                return selected
        
        elif self.style == 'worst':
            # Select the move with the lowest score (worst move)
            selected = move_scores[0]
            print(f"  🎯 OUR Bot playing WORST move (score: {selected[1]:.2f})")
            return selected
        
        elif self.style == 'average':
            # Select a move in the middle of the range
            # Choose from the middle 30% of moves for variety
            lower_bound = max(0, int(len(move_scores) * 0.35))
            upper_bound = min(len(move_scores), int(len(move_scores) * 0.65))
            
            if upper_bound <= lower_bound:
                # Fallback to median
                mid_idx = len(move_scores) // 2
                selected = move_scores[mid_idx]
            else:
                # Randomly select from the middle range
                random_idx = random.randint(lower_bound, upper_bound - 1)
                selected = move_scores[random_idx]
            
            print(f"  🎯 OUR Bot playing AVERAGE move (score: {selected[1]:.2f}, "
                  f"range: {move_scores[0][1]:.2f} to {move_scores[-1][1]:.2f})")
            return selected
        
        else:  # 'best' or default
            # Select the move with the highest score
            selected = move_scores[-1]
            print(f"  🎯 OUR Bot playing BEST move (score: {selected[1]:.2f})")
            return selected

# ============== Data Generator ==============
@dataclass
class TrainingPosition:
    features: np.ndarray
    score: float
    result: float
    tactical_score: float

class DataGenerator:
    def __init__(self, stockfish_path: str = Config.STOCKFISH_PATH):
        self.engine = None
        self.stockfish_path = stockfish_path
        self.our_bot_style = Config.OUR_BOT_PLAY_STYLE
        self.bot_selector = OurBotMoveSelector(self.our_bot_style)
    
    def start_stockfish_engine(self, skill_level: Optional[int] = None):
        """Start Stockfish engine with specific skill level"""
        if not os.path.exists(self.stockfish_path):
            raise FileNotFoundError(f"Stockfish not found: {self.stockfish_path}")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        self.engine.configure({"Hash": 128, "Threads": 2})
        
        # Configure Stockfish with skill level (this is the opponent)
        if skill_level is not None:
            try:
                self.engine.configure({"Skill Level": skill_level})
                print(f"  ⚙️  Stockfish configured with Skill Level: {skill_level}")
            except chess.engine.EngineError as e:
                print(f"  Warning: Could not set Skill Level {skill_level}: {e}")
                # If Skill Level fails, use UCI_Elo instead (alternative)
                try:
                    # Convert skill level to approximate Elo
                    elo = 800 + (skill_level * 100)
                    self.engine.configure({"UCI_Elo": elo})
                    print(f"  ⚙️  Stockfish configured with UCI_Elo: {elo}")
                except:
                    pass  # Fall back to default strength
    
    def stop_engine(self):
        if self.engine:
            self.engine.quit()
            self.engine = None
    
    def generate_data(self, num_games: int = Config.NUM_GAMES):
        print(f"\n{'='*80}")
        print(f"📊 NNUE DATA GENERATOR")
        print(f"{'='*80}")
        print(f"  Stockfish path: {self.stockfish_path}")
        print(f"  Stockfish skill levels: {Config.STOCKFISH_SKILL_LEVELS}")
        print(f"  OUR Bot play style: {self.our_bot_style.upper()}")
        if self.our_bot_style == 'alternating':
            print(f"    Pattern: 1 BEST move, then 2 WORST moves (repeating)")
        print(f"  Games: {num_games}")
        print(f"  Max moves per game: {Config.MAX_MOVES}")
        print(f"  Opening moves: 100% RANDOM")
        print(f"{'='*80}")
        
        all_positions = []
        openings = self._get_openings()
        
        # Use skill levels from config
        stockfish_levels = Config.STOCKFISH_SKILL_LEVELS
        games_per_level = max(1, num_games // len(stockfish_levels))
        
        start_time = time.time()
        
        for level_idx, skill_level in enumerate(stockfish_levels):
            print(f"\n  🎯 Playing against Stockfish at skill level: {skill_level}")
            self.start_stockfish_engine(skill_level)
            
            # Reset the bot selector's move counter for each level
            self.bot_selector = OurBotMoveSelector(self.our_bot_style)
            
            for game_idx in range(games_per_level):
                opening = openings[(game_idx + level_idx * 100) % len(openings)]
                positions = self.play_game(opening, skill_level)
                all_positions.extend(positions)
                
                if (game_idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    total_games = level_idx * games_per_level + game_idx + 1
                    avg_pos = len(all_positions) // total_games if total_games > 0 else 0
                    print(f"    Game {total_games}/{num_games} - {len(all_positions)} positions "
                          f"(~{avg_pos}/game) - {elapsed/60:.1f}min")
            
            self.stop_engine()
        
        self._save_data(all_positions)
        return all_positions
    
    def play_game(self, opening_fen: Optional[str] = None, stockfish_level: int = 0) -> List[TrainingPosition]:
        board = chess.Board(opening_fen) if opening_fen else chess.Board()
        positions = []
        move_count = 0
        depth_limit = chess.engine.Limit(depth=Config.DEPTH)
        visited_positions = set()
        
        while not board.is_game_over() and move_count < Config.MAX_MOVES:
            # Analyze current position using Stockfish
            try:
                analysis = self.engine.analyse(board, limit=depth_limit)
                score = analysis['score'].white().score()
            except:
                break
            
            if score is None:
                break
            
            score_float = score / 100.0
            if board.turn == chess.BLACK:
                score_float = -score_float
            
            tactical_score = calculate_tactical_threats(board) * 0.3
            
            features = featurize_board_prod(board)
            
            positions.append(TrainingPosition(
                features=features,
                score=score_float,
                result=0.0,
                tactical_score=tactical_score
            ))
            
            # Determine whose turn it is
            # Our bot plays on even moves (starts as white), Stockfish on odd moves
            is_our_turn = (move_count % 2 == 0)
            
            if is_our_turn:
                # OUR BOT's turn
                if move_count == 0:
                    # First move - 100% random (opening)
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        move = random.choice(legal_moves)
                        print(f"  🎲 OUR Bot playing RANDOM opening move: {board.san(move)}")
                    else:
                        break
                else:
                    # Non-opening moves - use configured style
                    try:
                        move, _ = self.bot_selector.select_move(
                            board, 
                            self.engine, 
                            depth=Config.DEPTH - 4,
                            is_opening=False
                        )
                        if move is None:
                            # Fallback to random move if no move selected
                            legal_moves = list(board.legal_moves)
                            if legal_moves:
                                move = random.choice(legal_moves)
                            else:
                                break
                    except Exception as e:
                        print(f"  Warning: Error in OUR bot move selection: {e}")
                        # Fallback to random
                        legal_moves = list(board.legal_moves)
                        if legal_moves:
                            move = random.choice(legal_moves)
                        else:
                            break
            else:
                # STOCKFISH's turn - let Stockfish play normally with its skill level
                try:
                    result = self.engine.play(board, limit=chess.engine.Limit(depth=Config.DEPTH-2))
                    move = result.move
                except Exception as e:
                    print(f"  Warning: Stockfish move error: {e}")
                    # Fallback to random move
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        move = random.choice(legal_moves)
                    else:
                        break
            
            board.push(move)
            move_count += 1
            
            pos_key = board.fen()
            if pos_key in visited_positions:
                break
            visited_positions.add(pos_key)
        
        # Determine result
        result = 0.0
        if board.is_checkmate():
            # White (OUR bot) wins on odd moves, Black (Stockfish) wins on even moves
            result = 1.0 if move_count % 2 == 1 else 0.0
        elif board.is_stalemate() or board.is_insufficient_material():
            result = 0.5
        
        for pos in positions:
            pos.result = result
        
        return positions
    
    def _get_openings(self):
        return [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2",
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
            "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 4",
        ]
    
    def _save_data(self, positions: List[TrainingPosition]):
        with open(Config.DATA_FILE, 'wb') as f:
            f.write(struct.pack('4sI', b'NNUE', len(positions)))
            for pos in positions:
                f.write(pos.features.tobytes())
                f.write(struct.pack('fff', pos.score, pos.result, pos.tactical_score))
        
        file_size = os.path.getsize(Config.DATA_FILE) / (1024 * 1024)
        print(f"\n{'='*80}")
        print(f"✅ Data generation complete!")
        print(f"  Saved {len(positions)} positions to {Config.DATA_FILE}")
        print(f"  File size: {file_size:.2f} MB")
        print(f"  Bot style used: {self.our_bot_style.upper()}")
        if self.our_bot_style == 'alternating':
            print(f"  Pattern: 1 BEST, 2 WORST (repeating)")
        print(f"{'='*80}")

# ============== Main ==============
def main():
    # Check Stockfish
    if not os.path.exists(Config.STOCKFISH_PATH):
        print(f"\n⚠️  Stockfish not found at: {Config.STOCKFISH_PATH}")
        print("Please install Stockfish or update STOCKFISH_PATH in Config")
        return
    
    # Generate data
    generator = DataGenerator()
    generator.generate_data()

if __name__ == "__main__":
    main()