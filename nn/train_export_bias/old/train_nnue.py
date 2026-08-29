#!/usr/bin/env python3
"""
Complete NNUE Training Pipeline
Generates self-play data, trains a neural network, and exports weights
"""

import chess
import chess.engine
import chess.pgn
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
from pathlib import Path

# ============== Configuration ==============
class Config:
    # Network architecture (must match your C code)
    PIECE_NB = 6
    NNUE_INPUT_DIM = 2 * PIECE_NB * 64  # 768
    NNUE_H1 = 256
    NNUE_H2 = 256
    NNUE_OUT = 1
    
    # Training parameters
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001
    EPOCHS = 20
    VALIDATION_SPLIT = 0.1
    
    # Data generation
    DEPTH = 8  # Stockfish search depth
    NUM_GAMES = 100
    POSITIONS_PER_GAME = 50
    STOCKFISH_PATH = "/usr/games/stockfish"  # Change 
    
    # Output files
    DATA_FILE = "training_data.bin"
    MODEL_FILE = "nnue_model.pt"
    WEIGHTS_FILE = "nnue_weights.bin"

# ============== NNUE Architecture (PyTorch) ==============
class NNUE(nn.Module):
    """Neural network matching the C implementation"""
    
    def __init__(self, input_dim=Config.NNUE_INPUT_DIM, 
                 h1=Config.NNUE_H1, h2=Config.NNUE_H2):
        super(NNUE, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to match typical NNUE initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)
    
    def export_weights(self, filepath):
        """Export weights in the format expected by your C code"""
        weights = {
            'w1': self.fc1.weight.detach().cpu().numpy(),
            'b1': self.fc1.bias.detach().cpu().numpy(),
            'w2': self.fc2.weight.detach().cpu().numpy(),
            'b2': self.fc2.bias.detach().cpu().numpy(),
            'w3': self.fc3.weight.detach().cpu().numpy(),
            'b3': self.fc3.bias.detach().cpu().numpy()
        }
        
        # Flatten in order: w1, b1, w2, b2, w3, b3
        flat_weights = []
        flat_weights.extend(weights['w1'].flatten())
        flat_weights.extend(weights['b1'].flatten())
        flat_weights.extend(weights['w2'].flatten())
        flat_weights.extend(weights['b2'].flatten())
        flat_weights.extend(weights['w3'].flatten())
        flat_weights.extend(weights['b3'].flatten())
        
        # Save as float32 binary
        flat_array = np.array(flat_weights, dtype=np.float32)
        flat_array.tofile(filepath)
        
        # Verify size matches C code expectations
        total_size = (Config.NNUE_INPUT_DIM * Config.NNUE_H1 +
                     Config.NNUE_H1 +
                     Config.NNUE_H1 * Config.NNUE_H2 +
                     Config.NNUE_H2 +
                     Config.NNUE_H2 * Config.NNUE_OUT +
                     Config.NNUE_OUT)
        
        #print(f"Exported {len(flat_array)} floats to {filepath}")
        #print(f"Expected size: {total_size} floats")
        
        if len(flat_array) != total_size:
            #print(f"WARNING: Size mismatch! Got {len(flat_array)}, expected {total_size}")
        
        return flat_array

# ============== Feature Extraction ==============
def featurize_board(board: chess.Board) -> np.ndarray:
    """Convert chess position to NNUE input features"""
    features = np.zeros(Config.NNUE_INPUT_DIM, dtype=np.float32)
    
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            side = 1 if piece.color == chess.BLACK else 0
            piece_idx = piece_map[piece.piece_type]
            idx = side * (Config.PIECE_NB * 64) + piece_idx * 64 + sq
            features[idx] = 1.0
    
    return features

# ============== Data Generation ==============
@dataclass
class TrainingPosition:
    features: np.ndarray
    score: float  # Evaluation score from the current player's perspective
    result: float  # 1.0 = win, 0.5 = draw, 0.0 = loss

class DataGenerator:
    """Generate training data using Stockfish self-play"""
    
    def __init__(self, stockfish_path: str = Config.STOCKFISH_PATH):
        self.engine = None
        self.stockfish_path = stockfish_path
        self.positions = []
        
    def start_engine(self):
        """Start Stockfish engine"""
        if not os.path.exists(self.stockfish_path):
            #print(f"Stockfish not found at {self.stockfish_path}")
            #print("Download from: https://stockfishchess.org/download/")
            raise FileNotFoundError(f"Stockfish executable not found: {self.stockfish_path}")
        
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        # Set some engine options
        self.engine.configure({"Hash": 128, "Threads": 1})
        
    def stop_engine(self):
        """Stop Stockfish engine"""
        if self.engine:
            self.engine.quit()
            self.engine = None
    
    def play_game(self, opening_fen: Optional[str] = None, 
                  max_moves: int = 200) -> List[TrainingPosition]:
        """Play one self-play game with fixed depth"""
        board = chess.Board(opening_fen) if opening_fen else chess.Board()
        positions = []
        
        move_count = 0
        # Store original depth for consistent evaluation
        depth_limit = chess.engine.Limit(depth=Config.DEPTH)
        
        #print(f"  Playing game...", end="", flush=True)
        
        while not board.is_game_over() and move_count < max_moves:
            # Get evaluation from Stockfish
            analysis = self.engine.analyse(board, limit=depth_limit)
            score = analysis['score'].white().score()
            
            # Convert to float (centipawns)
            if score is None:
                break
            score_float = score / 100.0
            
            # Store position (from current player's perspective)
            if board.turn == chess.BLACK:
                score_float = -score_float
            
            features = featurize_board(board)
            positions.append(TrainingPosition(
                features=features,
                score=score_float,
                result=0.0  # Will be filled after game ends
            ))
            
            # Play a move
            result = self.engine.play(board, limit=depth_limit)
            board.push(result.move)
            move_count += 1
            
            if move_count % 20 == 0:
                #print(".", end="", flush=True)
        
        #print(f" done ({move_count} moves)")
        
        # Determine game result
        result = 0.0
        if board.is_checkmate():
            # The player who just moved won
            result = 1.0 if move_count % 2 == 1 else 0.0
        elif board.is_stalemate() or board.is_insufficient_material():
            result = 0.5
        elif board.is_fivefold_repetition() or board.is_seventyfive_moves():
            result = 0.5
        
        # Update positions with final result
        for pos in positions:
            pos.result = result
            
        return positions
    
    def generate_data(self, num_games: int = Config.NUM_GAMES, 
                      output_file: str = Config.DATA_FILE):
        """Generate and save training data"""
        #print(f"Generating {num_games} self-play games at depth {Config.DEPTH}...")
        #print(f"Using Stockfish from: {self.stockfish_path}")
        
        self.start_engine()
        
        all_positions = []
        openings = self._get_openings()
        
        start_time = time.time()
        
        for game_idx in range(num_games):
            # Alternate openings for diversity
            opening = openings[game_idx % len(openings)]
            
            try:
                positions = self.play_game(opening)
                all_positions.extend(positions)
                
                if (game_idx + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_positions = len(all_positions) // (game_idx + 1)
                    #print(f"Game {game_idx + 1}/{num_games} completed. "
                          f"Total positions: {len(all_positions)} "
                          f"({avg_positions} avg/game) "
                          f"Time: {elapsed/60:.1f} min")
                          
            except Exception as e:
                #print(f"Error in game {game_idx}: {e}")
                continue
        
        self.stop_engine()
        
        # Save data
        self._save_data(all_positions, output_file)
        #print(f"Generated {len(all_positions)} positions from {num_games} games")
        return all_positions
    
    def _get_openings(self) -> List[str]:
        """Get a list of opening FENs for variety"""
        return [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
            "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2",
            "rnbqkbnr/pp1ppppp/8/2p5/2P5/8/PP1PPPPP/RNBQKBNR w KQkq - 0 2",
            "rnbqkb1r/pp1ppppp/5n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
            "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        ]
    
    def _save_data(self, positions: List[TrainingPosition], filename: str):
        """Save positions to binary file"""
        with open(filename, 'wb') as f:
            # Write header: magic number and count
            f.write(struct.pack('4sI', b'NNUE', len(positions)))
            
            for pos in positions:
                # Write features
                f.write(pos.features.tobytes())
                # Write score and result as floats
                f.write(struct.pack('ff', pos.score, pos.result))
        
        #print(f"Saved {len(positions)} positions to {filename}")
        file_size = os.path.getsize(filename) / (1024 * 1024)
        #print(f"File size: {file_size:.2f} MB")

# ============== Dataset Class ==============
class NNUE_Dataset(Dataset):
    """PyTorch dataset for NNUE training"""
    
    def __init__(self, data_file: str):
        self.features = []
        self.scores = []
        self.results = []
        self._load_data(data_file)
    
    def _load_data(self, filename: str):
        #print(f"Loading training data from {filename}...")
        
        with open(filename, 'rb') as f:
            # Read header
            magic, count = struct.unpack('4sI', f.read(8))
            if magic != b'NNUE':
                raise ValueError(f"Invalid file format: expected 'NNUE', got {magic}")
            
            #print(f"Found {count} positions")
            
            # Read data in chunks for memory efficiency
            for i in range(count):
                if i % 100000 == 0 and i > 0:
                    #print(f"  Loaded {i} positions...")
                
                # Read features
                feat_data = f.read(Config.NNUE_INPUT_DIM * 4)
                feat = np.frombuffer(feat_data, dtype=np.float32)
                self.features.append(feat)
                
                # Read score and result
                score, result = struct.unpack('ff', f.read(8))
                self.scores.append(score)
                self.results.append(result)
        
        #print(f"Loaded {len(self.features)} positions")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.features[idx], dtype=torch.float32),
                torch.tensor(self.scores[idx], dtype=torch.float32))

# ============== Training Function ==============
def train_model(model: NNUE, train_loader: DataLoader, 
                val_features: np.ndarray, val_scores: np.ndarray,
                epochs: int = Config.EPOCHS,
                learning_rate: float = Config.LEARNING_RATE):
    """Train the NNUE model"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #print(f"Using device: {device}")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler verbose=True
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Convert validation data to tensor
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32).to(device)
    val_scores_tensor = torch.tensor(val_scores, dtype=torch.float32).to(device)
    
    best_val_loss = float('inf')
    
    #print(f"Starting training for {epochs} epochs...")
    #print(f"Training samples: {len(train_loader.dataset)}")
    #print(f"Validation samples: {len(val_features)}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for batch_idx, (features, scores) in enumerate(train_loader):
            features = features.to(device)
            scores = scores.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, scores)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 100 == 0:
                #print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / batch_count
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features_tensor)
            val_loss = criterion(val_outputs, val_scores_tensor).item()
        
        # Update learning rate
        scheduler.step(val_loss)
        
        elapsed = time.time() - start_time
        #print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
              f"Time: {elapsed/60:.1f} min")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_FILE)
            #print(f"  New best model saved to {Config.MODEL_FILE} (val loss: {val_loss:.6f})")
    
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_FILE))
    #print(f"Training completed! Best validation loss: {best_val_loss:.6f}")
    
    return model

# ============== Main Pipeline ==============
def main():
    """Complete training pipeline"""
    #print("=" * 60)
    #print("NNUE Training Pipeline")
    #print("=" * 60)
    
    # Step 1: Generate data
    #print("\n[Step 1] Generating training data...")
    generator = DataGenerator()
    
    if os.path.exists(Config.DATA_FILE):
        #print(f"Data file {Config.DATA_FILE} already exists.")
        response = input("Skip data generation? (y/n): ")
        if response.lower() == 'y':
            #print("Skipping data generation...")
        else:
            generator.generate_data()
    else:
        generator.generate_data()
    
    # Step 2: Load and prepare data
    #print("\n[Step 2] Loading and preparing data...")
    dataset = NNUE_Dataset(Config.DATA_FILE)
    
    # Split into train and validation
    total_size = len(dataset)
    val_size = int(total_size * Config.VALIDATION_SPLIT)
    train_size = total_size - val_size
    
    # Create validation set first
    indices = list(range(total_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Get validation data as numpy arrays
    val_features = np.array([dataset.features[i] for i in val_indices])
    val_scores = np.array([dataset.scores[i] for i in val_indices])
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    #print(f"Training samples: {len(train_dataset)}")
    #print(f"Validation samples: {len(val_dataset)}")
    
    # Step 3: Create and train model
    #print("\n[Step 3] Training model...")
    model = NNUE()
    
    # #print model architecture
    #print(f"Model architecture:")
    #print(f"  Input: {Config.NNUE_INPUT_DIM}")
    #print(f"  Hidden1: {Config.NNUE_H1}")
    #print(f"  Hidden2: {Config.NNUE_H2}")
    #print(f"  Output: 1")
    #print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model = train_model(model, train_loader, val_features, val_scores)
    
    # Step 4: Export weights
    #print("\n[Step 4] Exporting weights to .bin file...")
    model.export_weights(Config.WEIGHTS_FILE)
    
    # Step 5: Verify exported file
    #print("\n[Step 5] Verifying exported weights...")
    verify_exported_weights()
    
    #print("\n" + "=" * 60)
    #print("Training pipeline completed successfully!")
    #print(f"Your C code can now load weights from: {Config.WEIGHTS_FILE}")
    #print("=" * 60)

def verify_exported_weights():
    """Verify that the exported weights file is correct"""
    try:
        file_size = os.path.getsize(Config.WEIGHTS_FILE)
        expected_size = (Config.NNUE_INPUT_DIM * Config.NNUE_H1 +
                        Config.NNUE_H1 +
                        Config.NNUE_H1 * Config.NNUE_H2 +
                        Config.NNUE_H2 +
                        Config.NNUE_H2 * Config.NNUE_OUT +
                        Config.NNUE_OUT) * 4
        
        #print(f"File: {Config.WEIGHTS_FILE}")
        #print(f"  Size: {file_size:,} bytes")
        #print(f"  Expected: {expected_size:,} bytes")
        
        if file_size == expected_size:
            #print("  ✅ File size matches expected size")
        else:
            #print("  ⚠️ File size mismatch!")
            
    except Exception as e:
        #print(f"Error verifying file: {e}")

# ============== Quick Test ==============
def test_loading():
    """Test loading the exported weights in C-like format"""
    #print("\nTesting loading of exported weights...")
    
    try:
        # Read weights
        weights = np.fromfile(Config.WEIGHTS_FILE, dtype=np.float32)
        
        # Calculate offsets
        w1_size = Config.NNUE_INPUT_DIM * Config.NNUE_H1
        b1_size = Config.NNUE_H1
        w2_size = Config.NNUE_H1 * Config.NNUE_H2
        b2_size = Config.NNUE_H2
        w3_size = Config.NNUE_H2 * Config.NNUE_OUT
        b3_size = Config.NNUE_OUT
        
        offset = 0
        
        w1 = weights[offset:offset + w1_size].reshape((Config.NNUE_H1, Config.NNUE_INPUT_DIM))
        offset += w1_size
        
        b1 = weights[offset:offset + b1_size]
        offset += b1_size
        
        w2 = weights[offset:offset + w2_size].reshape((Config.NNUE_H2, Config.NNUE_H1))
        offset += w2_size
        
        b2 = weights[offset:offset + b2_size]
        offset += b2_size
        
        w3 = weights[offset:offset + w3_size].reshape((Config.NNUE_OUT, Config.NNUE_H2))
        offset += w3_size
        
        b3 = weights[offset:offset + b3_size]
        
        #print("  ✅ Successfully loaded and parsed weights")
        #print(f"  w1 shape: {w1.shape}")
        #print(f"  b1 shape: {b1.shape}")
        #print(f"  w2 shape: {w2.shape}")
        #print(f"  b2 shape: {b2.shape}")
        #print(f"  w3 shape: {w3.shape}")
        #print(f"  b3 shape: {b3.shape}")
        
    except Exception as e:
        #print(f"Error testing weights: {e}")

if __name__ == "__main__":
    # Run the pipeline
    main()
    
    # Test loading
    test_loading()