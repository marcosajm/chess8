#!/usr/bin/env python3
"""
NNUE Training Pipeline with 780 Input Features
Matches C code: 780 -> 256 -> 32 -> 1
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

# ============== Configuration ==============
class Config:
    # Network architecture (matches C code)
    PIECE_NB = 6
    NNUE_INPUT_DIM = 780  # Feature size
    NNUE_H1 = 256
    NNUE_H2 = 32
    NNUE_OUT = 1
    
    # Training parameters
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001
    EPOCHS = 20
    VALIDATION_SPLIT = 0.1
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0
    PATIENCE = 5
    
    # Data generation
    DEPTH = 8
    NUM_GAMES = 100  # Start small for testing
    MAX_MOVES = 200
    STOCKFISH_PATH = "/usr/games/stockfish"
    
    # Output files
    DATA_FILE = "training_data.bin"
    MODEL_FILE = "nnue_model.pt"
    WEIGHTS_FILE = "nnue_weights.bin"

# ============== Feature Extraction (780 features) ==============
def featurize_board(board: chess.Board) -> np.ndarray:
    """Convert chess position to NNUE input features (780 features)
    
    Standard NNUE: 768 features (2 sides × 6 pieces × 64 squares)
    Extra 12 features: castling rights, en passant, half-move clock, etc.
    """
    features = np.zeros(780, dtype=np.float32)
    
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    # Standard 768 features (2 sides × 6 pieces × 64 squares)
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            side = 1 if piece.color == chess.BLACK else 0
            piece_idx = piece_map[piece.piece_type]
            idx = side * (6 * 64) + piece_idx * 64 + sq
            features[idx] = 1.0
    
    # Extra 12 features (indices 768-779)
    feature_idx = 768
    
    # 1-4: Castling rights (4 features)
    if board.has_kingside_castling_rights(chess.WHITE):
        features[feature_idx] = 1.0
    feature_idx += 1
    if board.has_queenside_castling_rights(chess.WHITE):
        features[feature_idx] = 1.0
    feature_idx += 1
    if board.has_kingside_castling_rights(chess.BLACK):
        features[feature_idx] = 1.0
    feature_idx += 1
    if board.has_queenside_castling_rights(chess.BLACK):
        features[feature_idx] = 1.0
    feature_idx += 1
    
    # 5-6: En passant target square (2 features)
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        ep_rank = chess.square_rank(board.ep_square)
        features[feature_idx] = ep_file / 7.0
        features[feature_idx + 1] = ep_rank / 7.0
    feature_idx += 2
    
    # 7: Half-move clock (1 feature)
    features[feature_idx] = min(board.halfmove_clock, 100) / 100.0
    feature_idx += 1
    
    # 8: Full move number (1 feature)
    features[feature_idx] = min(board.fullmove_number, 100) / 100.0
    feature_idx += 1
    
    # 9: Side to move (1 feature)
    features[feature_idx] = 1.0 if board.turn == chess.BLACK else 0.0
    feature_idx += 1
    
    # 10: Piece count difference (1 feature)
    white_pieces = board.occupied_co[chess.WHITE]
    black_pieces = board.occupied_co[chess.BLACK]
    piece_diff = (white_pieces.bit_count() - black_pieces.bit_count()) / 16.0
    features[feature_idx] = piece_diff
    feature_idx += 1
    
    # 11-12: Spare features (2 features)
    features[feature_idx] = 0.0
    features[feature_idx + 1] = 0.0
    
    return features

# ============== Data Generator ==============
@dataclass
class TrainingPosition:
    features: np.ndarray
    score: float
    result: float

class DataGenerator:
    def __init__(self, stockfish_path: str = Config.STOCKFISH_PATH):
        self.engine = None
        self.stockfish_path = stockfish_path
        
    def start_engine(self):
        if not os.path.exists(self.stockfish_path):
            raise FileNotFoundError(f"Stockfish not found: {self.stockfish_path}")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        self.engine.configure({"Hash": 128, "Threads": 1})
    
    def stop_engine(self):
        if self.engine:
            self.engine.quit()
            self.engine = None
    
    def play_game(self, opening_fen: Optional[str] = None) -> List[TrainingPosition]:
        board = chess.Board(opening_fen) if opening_fen else chess.Board()
        positions = []
        move_count = 0
        depth_limit = chess.engine.Limit(depth=Config.DEPTH)
        
        while not board.is_game_over() and move_count < Config.MAX_MOVES:
            analysis = self.engine.analyse(board, limit=depth_limit)
            score = analysis['score'].white().score()
            
            if score is None:
                break
            
            score_float = score / 100.0
            if board.turn == chess.BLACK:
                score_float = -score_float
            
            features = featurize_board(board)  # Now uses 780 features
            positions.append(TrainingPosition(
                features=features,
                score=score_float,
                result=0.0
            ))
            
            result = self.engine.play(board, limit=depth_limit)
            board.push(result.move)
            move_count += 1
        
        # Determine result
        result = 0.0
        if board.is_checkmate():
            result = 1.0 if move_count % 2 == 1 else 0.0
        elif board.is_stalemate() or board.is_insufficient_material():
            result = 0.5
        
        for pos in positions:
            pos.result = result
            
        return positions
    
    def generate_data(self, num_games: int = Config.NUM_GAMES):
        #print(f"Generating {num_games} self-play games at depth {Config.DEPTH}...")
        
        self.start_engine()
        all_positions = []
        openings = self._get_openings()
        
        start_time = time.time()
        
        for game_idx in range(num_games):
            opening = openings[game_idx % len(openings)]
            positions = self.play_game(opening)
            all_positions.extend(positions)
            
            if (game_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg = len(all_positions) // (game_idx + 1)
                #print(f"Game {game_idx + 1}/{num_games} - {len(all_positions)} positions ({avg}/game) - {elapsed/60:.1f}min")
        
        self.stop_engine()
        self._save_data(all_positions)
        return all_positions
    
    def _get_openings(self):
        return [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2",
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
        ]
    
    def _save_data(self, positions: List[TrainingPosition]):
        with open(Config.DATA_FILE, 'wb') as f:
            f.write(struct.pack('4sI', b'NNUE', len(positions)))
            for pos in positions:
                f.write(pos.features.tobytes())
                f.write(struct.pack('ff', pos.score, pos.result))
        
        file_size = os.path.getsize(Config.DATA_FILE) / (1024 * 1024)
        #print(f"Saved {len(positions)} positions to {Config.DATA_FILE}")
        #print(f"File size: {file_size:.2f} MB")

# ============== Dataset ==============
class NNUE_Dataset(Dataset):
    def __init__(self, data_file: str):
        self.features = []
        self.scores = []
        self._load_data(data_file)
    
    def _load_data(self, filename: str):
        #print(f"Loading training data from {filename}...")
        with open(filename, 'rb') as f:
            magic, count = struct.unpack('4sI', f.read(8))
            if magic != b'NNUE':
                raise ValueError(f"Invalid file format")
            
            #print(f"Found {count} positions")
            
            for i in range(count):
                if i % 10000 == 0 and i > 0:
                    #print(f"  Loaded {i} positions...")
                
                feat_data = f.read(Config.NNUE_INPUT_DIM * 4)
                if len(feat_data) < Config.NNUE_INPUT_DIM * 4:
                    #print(f"Warning: Incomplete data at position {i}")
                    break
                    
                feat = np.frombuffer(feat_data, dtype=np.float32)
                self.features.append(feat)
                score, result = struct.unpack('ff', f.read(8))
                self.scores.append(score)
        
        #print(f"Loaded {len(self.features)} positions")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx], dtype=torch.float32),
                torch.tensor(self.scores[idx], dtype=torch.float32))

# ============== NNUE Model ==============
class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        self.fc1 = nn.Linear(Config.NNUE_INPUT_DIM, Config.NNUE_H1)
        self.fc2 = nn.Linear(Config.NNUE_H1, Config.NNUE_H2)
        self.fc3 = nn.Linear(Config.NNUE_H2, 1)
        self.relu = nn.ReLU()
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)
    
    def export_weights(self, filepath: str):
        """Export weights in C format: w1, b1, w2, b2, w3, b3"""
        weights = {
            'w1': self.fc1.weight.detach().cpu().numpy().T,
            'b1': self.fc1.bias.detach().cpu().numpy(),
            'w2': self.fc2.weight.detach().cpu().numpy().T,
            'b2': self.fc2.bias.detach().cpu().numpy(),
            'w3': self.fc3.weight.detach().cpu().numpy().T,
            'b3': self.fc3.bias.detach().cpu().numpy()
        }
        
        #print("\n📊 Exporting weights:")
        #print(f"  w1: {weights['w1'].shape} ({weights['w1'].size:,} floats)")
        #print(f"  b1: {weights['b1'].shape} ({weights['b1'].size:,} floats)")
        #print(f"  w2: {weights['w2'].shape} ({weights['w2'].size:,} floats)")
        #print(f"  b2: {weights['b2'].shape} ({weights['b2'].size:,} floats)")
        #print(f"  w3: {weights['w3'].shape} ({weights['w3'].size:,} floats)")
        #print(f"  b3: {weights['b3'].shape} ({weights['b3'].size:,} floats)")
        
        # Flatten in order: w1, b1, w2, b2, w3, b3
        flat_weights = []
        flat_weights.extend(weights['w1'].flatten())
        flat_weights.extend(weights['b1'].flatten())
        flat_weights.extend(weights['w2'].flatten())
        flat_weights.extend(weights['b2'].flatten())
        flat_weights.extend(weights['w3'].flatten())
        flat_weights.extend(weights['b3'].flatten())
        
        flat_array = np.array(flat_weights, dtype=np.float32)
        flat_array.tofile(filepath)
        
        expected = (Config.NNUE_INPUT_DIM * Config.NNUE_H1 +
                   Config.NNUE_H1 +
                   Config.NNUE_H1 * Config.NNUE_H2 +
                   Config.NNUE_H2 +
                   Config.NNUE_H2 * 1 +
                   1)
        
        #print(f"\n💾 Exported {len(flat_array):,} floats to {filepath}")
        #print(f"Expected: {expected:,} floats")
        
        if len(flat_array) == expected:
            #print("✅ Size matches C code!")
        else:
            #print(f"⚠️  Size mismatch! Got {len(flat_array)}, expected {expected}")
        
        return flat_array

# ============== Training ==============
def train_model(model, train_loader, val_features, val_scores):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #print(f"Using device: {device}")
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, 
                          weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=3)
    
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32).to(device)
    val_scores_tensor = torch.tensor(val_scores, dtype=torch.float32).to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    #print(f"\n🏋️  Training on {len(train_loader.dataset)} positions")
    #print(f"   Validation: {len(val_features)} positions")
    #print(f"   Batch size: {Config.BATCH_SIZE}")
    
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
        
        #print(f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_FILE)
            patience_counter = 0
            #print(f"  ✅ New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                #print(f"  ⏹️  Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(torch.load(Config.MODEL_FILE))
    return model

# ============== Main Pipeline ==============
def main():
    #print("=" * 70)
    #print("NNUE Training Pipeline")
    #print(f"Network: {Config.NNUE_INPUT_DIM} -> {Config.NNUE_H1} -> {Config.NNUE_H2} -> 1")
    #print("=" * 70)
    
    # Step 1: Generate data if needed
    if not os.path.exists(Config.DATA_FILE):
        #print("\n📊 Generating training data...")
        generator = DataGenerator()
        generator.generate_data()
    else:
        #print(f"\n📂 Data file exists: {Config.DATA_FILE}")
        response = input("Regenerate data? (y/n): ")
        if response.lower() == 'y':
            os.remove(Config.DATA_FILE)
            generator = DataGenerator()
            generator.generate_data()
    
    # Step 2: Load data
    #print("\n📂 Loading data...")
    dataset = NNUE_Dataset(Config.DATA_FILE)
    
    if len(dataset) == 0:
        #print("❌ No data loaded!")
        return
    
    # Step 3: Split data
    total = len(dataset)
    val_size = int(total * Config.VALIDATION_SPLIT)
    train_size = total - val_size
    
    indices = list(range(total))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Prepare validation data
    val_features = np.array([dataset.features[i] for i in val_indices])
    val_scores = np.array([dataset.scores[i] for i in val_indices])
    
    # Data loader
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, num_workers=0)
    
    #print(f"\n📊 Data split:")
    #print(f"  Training: {len(train_dataset)} positions")
    #print(f"  Validation: {len(val_dataset)} positions")
    
    # Step 4: Create and train model
    #print(f"\n🧠 Creating model...")
    model = NNUE()
    total_params = sum(p.numel() for p in model.parameters())
    #print(f"  Total parameters: {total_params:,}")
    
    model = train_model(model, train_loader, val_features, val_scores)
    
    # Step 5: Export weights
    #print(f"\n💾 Exporting weights...")
    model.export_weights(Config.WEIGHTS_FILE)
    
    # Step 6: Verify
    verify_exported_weights()
    
    #print("\n" + "=" * 70)
    #print("✅ Training complete!")
    #print(f"  Weights saved to: {Config.WEIGHTS_FILE}")
    #print(f"  Model saved to: {Config.MODEL_FILE}")
    #print("=" * 70)

def verify_exported_weights():
    """Verify the exported weights file"""
    try:
        file_size = os.path.getsize(Config.WEIGHTS_FILE)
        expected = (Config.NNUE_INPUT_DIM * Config.NNUE_H1 +
                   Config.NNUE_H1 +
                   Config.NNUE_H1 * Config.NNUE_H2 +
                   Config.NNUE_H2 +
                   Config.NNUE_H2 * 1 +
                   1) * 4
        
        #print(f"\n🔍 Verification:")
        #print(f"  File: {Config.WEIGHTS_FILE}")
        #print(f"  Size: {file_size:,} bytes")
        #print(f"  Expected: {expected:,} bytes")
        
        if file_size == expected:
            #print("  ✅ File size matches C code expectations!")
            
            # Test read
            weights = np.fromfile(Config.WEIGHTS_FILE, dtype=np.float32)
            #print(f"  ✅ Successfully read {len(weights):,} floats")
        else:
            #print(f"  ⚠️  File size mismatch!")
            
    except Exception as e:
        #print(f"Error verifying file: {e}")

if __name__ == "__main__":
    main()