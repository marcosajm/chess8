#!/usr/bin/env python3
"""
NNUE Training Pipeline - Fixed for Stockfish compatibility
Network: 780 -> 256 -> 64 -> 32 -> 1 (WASM optimized)
"""

import chess
import chess.engine
import numpy as np
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Optional, Tuple
import random
import time
import os
import json
from pathlib import Path

ts = int(time.time())

# ============== Configuration ==============
class Config:
    # Network architecture - WASM optimized
    NNUE_INPUT_DIM = 780
    NNUE_H1 = 256    # WASM optimized
    NNUE_H2 = 64     # Better defense
    NNUE_H3 = 32     # Small 3rd layer
    NNUE_OUT = 1
    
    # Training parameters
    BATCH_SIZE = 2048
    LEARNING_RATE = 0.001
    EPOCHS = 200
    VALIDATION_SPLIT = 0.1
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 1.0
    PATIENCE = 10
    DROPOUT_RATE = 0.15
    
    # Data generation - FIXED for Stockfish compatibility
    DEPTH = 24
    NUM_GAMES = 4  # Start with fewer games for testing
    MAX_MOVES = 212
    STOCKFISH_PATH = "/usr/games/stockfish"
    
    # Stockfish skill levels (0-20 for newer versions)
    # 0 = weakest, 20 = strongest
    STOCKFISH_SKILL_LEVELS = [20]  # Stockfish opponent levels
    
    # Bot play style configuration (OUR bot, not Stockfish)
    # 'worst' = play the worst possible moves (minimum evaluation)
    # 'average' = play average moves (middle of evaluation range)
    # 'best' = play the best moves (maximum evaluation) - default behavior
    OUR_BOT_PLAY_STYLE = 'worst'  # Options: 'worst', 'average', 'best'
    
    # Output files
    DATA_FILE = "training_data" + str(DEPTH) + str(NUM_GAMES)  + str(MAX_MOVES) +  str(ts) + "_prod.bin"
    MODEL_FILE = "nnue_model_prod.pt"
    WEIGHTS_FILE = "nnue_weights_prod.bin"
    WASM_WEIGHTS_FILE = "nnue_weights_wasm.bin"
    CHECKPOINT_DIR = "checkpoints_prod"

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

# ============== Model ==============
class NNUEProduction(nn.Module):
    """Optimized for WASM deployment"""
    def __init__(self):
        super(NNUEProduction, self).__init__()
        self.fc1 = nn.Linear(Config.NNUE_INPUT_DIM, Config.NNUE_H1)
        self.fc2 = nn.Linear(Config.NNUE_H1, Config.NNUE_H2)
        self.fc3 = nn.Linear(Config.NNUE_H2, Config.NNUE_H3)
        self.fc4 = nn.Linear(Config.NNUE_H3, Config.NNUE_OUT)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)
        self.relu = nn.ReLU()
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x.squeeze(-1)
    
    def export_weights_wasm(self, filepath: str):
        """Export weights in WASM-compatible format"""
        weights = {
            'w1': self.fc1.weight.detach().cpu().numpy().T,
            'b1': self.fc1.bias.detach().cpu().numpy(),
            'w2': self.fc2.weight.detach().cpu().numpy().T,
            'b2': self.fc2.bias.detach().cpu().numpy(),
            'w3': self.fc3.weight.detach().cpu().numpy().T,
            'b3': self.fc3.bias.detach().cpu().numpy(),
            'w4': self.fc4.weight.detach().cpu().numpy().T,
            'b4': self.fc4.bias.detach().cpu().numpy()
        }
        
        print("\n📊 Exporting weights for WASM:")
        total_params = 0
        for name, data in weights.items():
            print(f"  {name}: {data.shape} ({data.size:,} floats)")
            total_params += data.size
        
        # Flatten in order: w1, b1, w2, b2, w3, b3, w4, b4
        flat_weights = []
        for key in ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4']:
            flat_weights.extend(weights[key].flatten())
        
        flat_array = np.array(flat_weights, dtype=np.float32)
        flat_array.tofile(filepath)
        
        print(f"\n💾 Exported {len(flat_array):,} floats to {filepath}")
        print(f"   File size: {len(flat_array) * 4 / 1024 / 1024:.2f} MB")
        
        return flat_array

# ============== Bot Move Selector (OUR Bot) ==============
class OurBotMoveSelector:
    """
    Selects moves for OUR bot based on the configured play style:
    - 'worst': Always chooses the move with the minimum evaluation
    - 'average': Chooses moves in the middle of the evaluation range
    - 'best': Chooses the move with the maximum evaluation (default)
    - 'random': Chooses a completely random move (used only for opening)
    """
    
    @staticmethod
    def select_move(board: chess.Board, engine: chess.engine.SimpleEngine, 
                    depth: int, style: str = 'worst', is_opening: bool = False) -> Tuple[chess.Move, float]:
        """
        Select a move based on the specified style.
        
        Args:
            board: Current chess board
            engine: Stockfish engine instance (used for evaluation)
            depth: Analysis depth
            style: 'worst', 'average', 'best', or 'random'
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
        if random.random() < 0.25:
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
        
        if style == 'worst':
            # Select the move with the lowest score (worst move)
            selected = move_scores[0]
            print(f"  🎯 OUR Bot playing WORST move (score: {selected[1]:.2f})")
            return selected
        
        elif style == 'average':
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
        self.opening_moves_played = 0  # Track opening moves
    
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
        print(f"\n📊 Generating {num_games} games...")
        print(f"   Stockfish path: {self.stockfish_path}")
        print(f"   Stockfish skill levels: {Config.STOCKFISH_SKILL_LEVELS}")
        print(f"   OUR Bot play style: {self.our_bot_style.upper()}")
        print(f"   Note: OUR Bot plays RANDOM on first move (opening), then uses '{self.our_bot_style}' style")
        
        all_positions = []
        openings = self._get_openings()
        
        # Use skill levels from config
        stockfish_levels = Config.STOCKFISH_SKILL_LEVELS
        games_per_level = max(1, num_games // len(stockfish_levels))
        
        start_time = time.time()
        
        for level_idx, skill_level in enumerate(stockfish_levels):
            print(f"\n  🎯 Playing against Stockfish at skill level: {skill_level}")
            self.start_stockfish_engine(skill_level)
            
            for game_idx in range(games_per_level):
                opening = openings[(game_idx + level_idx * 100) % len(openings)]
                positions = self.play_game(opening, skill_level)
                all_positions.extend(positions)
                
                if (game_idx + 1) % 50 == 0:
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
        
        # Flag to track if this is the first move
        is_opening_move = True
        
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
                # Check if this is the opening move (move_count == 0)
                if move_count == 0:
                    # First move - 100% random
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        move = random.choice(legal_moves)
                        print(f"  🎲 OUR Bot playing RANDOM opening move: {board.san(move)}")
                    else:
                        break
                else:
                    # Non-opening moves - use configured style
                    # 4% random moves for variety (in addition to the style)
                    if random.random() < 0.08:
                        legal_moves = list(board.legal_moves)
                        if legal_moves:
                            move = random.choice(legal_moves)
                            print(f"  🎲 OUR Bot playing RANDOM move (variety)")
                        else:
                            break
                    else:
                        try:
                            move, _ = OurBotMoveSelector.select_move(
                                board, 
                                self.engine, 
                                depth=Config.DEPTH - 4,
                                style=self.our_bot_style,
                                is_opening=False  # Not opening anymore
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
        print(f"\n✅ Saved {len(positions)} positions to {Config.DATA_FILE}")
        print(f"   File size: {file_size:.2f} MB")

# ============== Dataset ==============
class NNUE_Dataset(Dataset):
    def __init__(self, data_file: str):
        self.features = []
        self.scores = []
        self._load_data(data_file)
    
    def _load_data(self, filename: str):
        print(f"Loading training data from {filename}...")
        with open(filename, 'rb') as f:
            magic, count = struct.unpack('4sI', f.read(8))
            if magic != b'NNUE':
                raise ValueError(f"Invalid file format")
            
            print(f"Found {count} positions")
            
            for i in range(count):
                if i % 50000 == 0 and i > 0:
                    print(f"  Loaded {i} positions...")
                
                feat_data = f.read(Config.NNUE_INPUT_DIM * 4)
                if len(feat_data) < Config.NNUE_INPUT_DIM * 4:
                    break
                
                feat = np.frombuffer(feat_data, dtype=np.float32)
                self.features.append(feat)
                score, result, tactical = struct.unpack('fff', f.read(12))
                
                # Combine scores - emphasize tactical awareness
                combined_score = score * 0.7 + tactical * 0.3
                self.scores.append(combined_score)
        
        print(f"Loaded {len(self.features)} positions")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        # Data augmentation: mirror position (simplified)
        if random.random() < 0.3:
            features = self._mirror_features(features)
        
        return (torch.tensor(features, dtype=torch.float32),
                torch.tensor(self.scores[idx], dtype=torch.float32))
    
    def _mirror_features(self, features: np.ndarray) -> np.ndarray:
        """Mirror the board (simplified)"""
        mirrored = features.copy()
        # For production, implement proper mirroring
        return mirrored

# ============== Training ==============
def train_model(model, train_loader, val_features, val_scores):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"\n🔧 Using device: {device}")
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, 
                           weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5)
    
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32).to(device)
    val_scores_tensor = torch.tensor(val_scores, dtype=torch.float32).to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    print(f"\n🏋️  Training on {len(train_loader.dataset)} positions")
    print(f"   Validation: {len(val_features)} positions")
    print(f"   Batch size: {Config.BATCH_SIZE}")
    
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for features, scores in train_loader:
            features = features.to(device)
            scores = scores.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, scores)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = train_loss / batch_count
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features_tensor)
            val_loss = criterion(val_outputs, val_scores_tensor).item()
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, 'best_model.pt'))
            patience_counter = 0
            print(f"  ✅ New best model saved")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"  ⏹️  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    return model

# ============== Data File Selection Routine ==============
def select_data_file() -> str:
    """
    Ask the user whether to use the existing training_data.bin or specify a different file.
    Returns the path to the selected data file.
    """
    print("\n" + "=" * 80)
    print("📁 DATA FILE SELECTION")
    print("=" * 80)
    
    default_file = Config.DATA_FILE
    print(f"Default data file: {default_file}")
    
    # Check if default file exists
    default_exists = os.path.exists(default_file)
    if default_exists:
        file_size = os.path.getsize(default_file) / (1024 * 1024)
        print(f"✅ Default file exists: {default_file} ({file_size:.2f} MB)")
    else:
        print(f"❌ Default file does not exist: {default_file}")
    
    print("\nOptions:")
    print("  1. Use default file (training_data_prod.bin)")
    print("  2. Specify a different data file")
    print("  3. Generate new data")
    
    while True:
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            if default_exists:
                print(f"✅ Using default file: {default_file}")
                return default_file
            else:
                print("❌ Default file does not exist. Please choose another option.")
                continue
        
        elif choice == '2':
            while True:
                custom_file = input("Enter the path to your training data file: ").strip()
                if not custom_file:
                    print("❌ File path cannot be empty.")
                    continue
                
                if os.path.exists(custom_file):
                    file_size = os.path.getsize(custom_file) / (1024 * 1024)
                    print(f"✅ Using custom file: {custom_file} ({file_size:.2f} MB)")
                    return custom_file
                else:
                    print(f"❌ File not found: {custom_file}")
                    retry = input("Try again? (y/n): ").strip().lower()
                    if retry != 'y':
                        break
            
            # If user gives up on custom file, fall back to generating new data
            print("Falling back to generating new data...")
            return None
        
        elif choice == '3':
            print("📊 Will generate new training data...")
            return None
        
        else:
            print("❌ Invalid option. Please enter 1, 2, or 3.")
    
    return None

# ============== Bot Style Selection ==============
def select_bot_style():
    """
    Ask the user to select the play style for OUR bot.
    Returns the selected style.
    """
    print("\n" + "=" * 80)
    print("🎮 OUR BOT PLAY STYLE SELECTION")
    print("=" * 80)
    
    print("Select how OUR bot should play against Stockfish:")
    print("  1. WORST - Always play the worst possible moves (minimum evaluation)")
    print("  2. AVERAGE - Play average moves (middle of evaluation range)")
    print("  3. BEST - Play the best moves (maximum evaluation) [default]")
    print("\n📌 NOTE: The first move (opening) will ALWAYS be 100% RANDOM")
    print("   Stockfish will play with skill levels 0, 5, 10, 15, 20")
    
    current_style = Config.OUR_BOT_PLAY_STYLE
    print(f"\nCurrent style: {current_style.upper()}")
    
    while True:
        choice = input("\nSelect option (1-3) or press Enter to keep current: ").strip()
        
        if choice == '':
            print(f"✅ Keeping current style: {current_style.upper()}")
            return current_style
        
        elif choice == '1':
            print("✅ OUR Bot will play WORST moves (after random opening)")
            return 'worst'
        
        elif choice == '2':
            print("✅ OUR Bot will play AVERAGE moves (after random opening)")
            return 'average'
        
        elif choice == '3':
            print("✅ OUR Bot will play BEST moves (after random opening)")
            return 'best'
        
        else:
            print("❌ Invalid option. Please enter 1, 2, 3, or press Enter.")

# ============== Main ==============
def main():
    print("=" * 80)
    print("NNUE Production Training - WASM Optimized")
    print(f"Network: {Config.NNUE_INPUT_DIM} -> {Config.NNUE_H1} -> {Config.NNUE_H2} -> {Config.NNUE_H3} -> 1")
    print("=" * 80)
    
    # Check Stockfish
    if not os.path.exists(Config.STOCKFISH_PATH):
        print(f"\n⚠️  Stockfish not found at: {Config.STOCKFISH_PATH}")
        print("Please install Stockfish or update STOCKFISH_PATH in Config")
        return
    
    # Step 1: Select bot play style for OUR bot
    Config.OUR_BOT_PLAY_STYLE = select_bot_style()
    
    # Step 2: Select data file or generate new data
    data_file = select_data_file()
    
    if data_file is None:
        # Generate new data
        print("\n📊 Generating training data...")
        generator = DataGenerator()
        generator.generate_data()
        data_file = Config.DATA_FILE
    else:
        print(f"\n📂 Using data file: {data_file}")
    
    # Step 3: Load and prepare data
    print("\n📂 Loading data...")
    try:
        dataset = NNUE_Dataset(data_file)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Please check the file format or generate new data.")
        return
    
    if len(dataset) == 0:
        print("❌ No data loaded!")
        return
    
    # Split data
    total = len(dataset)
    val_size = int(total * Config.VALIDATION_SPLIT)
    train_size = total - val_size
    
    indices = list(range(total))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    val_features = np.array([dataset.features[i] for i in val_indices])
    val_scores = np.array([dataset.scores[i] for i in val_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, num_workers=0, pin_memory=True)
    
    print(f"\n📊 Data split:")
    print(f"  Training: {len(train_dataset):,} positions")
    print(f"  Validation: {len(val_dataset):,} positions")
    
    # Step 4: Train model
    print(f"\n🧠 Creating production model...")
    model = NNUEProduction()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    model = train_model(model, train_loader, val_features, val_scores)
    
    # Step 5: Export for WASM
    print(f"\n💾 Exporting WASM weights...")
    model.export_weights_wasm(Config.WASM_WEIGHTS_FILE)
    
    # Also save PyTorch model
    torch.save(model.state_dict(), Config.MODEL_FILE)
    
    # Step 6: Verify
    verify_export(Config.WASM_WEIGHTS_FILE)
    
    print("\n" + "=" * 80)
    print("✅ Production training complete!")
    print(f"  WASM weights: {Config.WASM_WEIGHTS_FILE}")
    print(f"  Model: {Config.MODEL_FILE}")
    print(f"  Stockfish levels used: {Config.STOCKFISH_SKILL_LEVELS}")
    print(f"  OUR Bot play style: {Config.OUR_BOT_PLAY_STYLE.upper()}")
    print(f"  Opening moves: 100% RANDOM")
    print("=" * 80)

def verify_export(filepath):
    """Verify exported weights file"""
    try:
        file_size = os.path.getsize(filepath)
        expected_params = (Config.NNUE_INPUT_DIM * Config.NNUE_H1 +
                          Config.NNUE_H1 +
                          Config.NNUE_H1 * Config.NNUE_H2 +
                          Config.NNUE_H2 +
                          Config.NNUE_H2 * Config.NNUE_H3 +
                          Config.NNUE_H3 +
                          Config.NNUE_H3 * Config.NNUE_OUT +
                          Config.NNUE_OUT)
        
        expected_bytes = expected_params * 4
        
        print(f"\n🔍 Verification:")
        print(f"  File: {filepath}")
        print(f"  Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        print(f"  Expected: {expected_bytes:,} bytes ({expected_bytes/1024/1024:.2f} MB)")
        
        if file_size == expected_bytes:
            print("  ✅ File size matches!")
            
            # Test read
            weights = np.fromfile(filepath, dtype=np.float32)
            print(f"  ✅ Successfully read {len(weights):,} floats")
        else:
            print(f"  ⚠️  File size mismatch!")
            
    except Exception as e:
        print(f"  ❌ Error verifying: {e}")

if __name__ == "__main__":
    main()