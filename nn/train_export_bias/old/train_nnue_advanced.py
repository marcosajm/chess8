#!/usr/bin/env python3
"""
Complete NNUE Training Pipeline with Advanced Options
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
from typing import List, Optional, Tuple, Dict
import random
import time
import os
import json
import sys
from pathlib import Path
from datetime import datetime

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
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 1.0
    PATIENCE = 5  # Early stopping patience
    
    # Data generation
    DEPTH = 8  # Stockfish search depth
    NUM_GAMES = 100
    POSITIONS_PER_GAME = 50
    STOCKFISH_PATH = "/usr/games/stockfish"
    MAX_MOVES = 200
    
    # Output files
    DATA_FILE = "training_data.bin"
    MODEL_FILE = "nnue_model.pt"
    WEIGHTS_FILE = "nnue_weights.bin"
    CONFIG_FILE = "training_config.json"
    LOG_FILE = "training_log.txt"

# ============== NNUE Architecture ==============
class NNUE(nn.Module):
    """Neural network matching the C implementation"""
    
    def __init__(self, input_dim=Config.NNUE_INPUT_DIM, 
                 h1=Config.NNUE_H1, h2=Config.NNUE_H2):
        super(NNUE, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
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
        
        flat_weights = []
        flat_weights.extend(weights['w1'].flatten())
        flat_weights.extend(weights['b1'].flatten())
        flat_weights.extend(weights['w2'].flatten())
        flat_weights.extend(weights['b2'].flatten())
        flat_weights.extend(weights['w3'].flatten())
        flat_weights.extend(weights['b3'].flatten())
        
        flat_array = np.array(flat_weights, dtype=np.float32)
        flat_array.tofile(filepath)
        
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
    score: float
    result: float

class DataGenerator:
    def __init__(self, stockfish_path: str = Config.STOCKFISH_PATH):
        self.engine = None
        self.stockfish_path = stockfish_path
        self.positions = []
    
    def start_engine(self):
        if not os.path.exists(self.stockfish_path):
            #print(f"Stockfish not found at {self.stockfish_path}")
            #print("Download from: https://stockfishchess.org/download/")
            raise FileNotFoundError(f"Stockfish executable not found: {self.stockfish_path}")
        
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        self.engine.configure({"Hash": 128, "Threads": 1})
    
    def stop_engine(self):
        if self.engine:
            self.engine.quit()
            self.engine = None
    
    def play_game(self, opening_fen: Optional[str] = None, 
                  max_moves: int = Config.MAX_MOVES) -> List[TrainingPosition]:
        board = chess.Board(opening_fen) if opening_fen else chess.Board()
        positions = []
        move_count = 0
        depth_limit = chess.engine.Limit(depth=Config.DEPTH)
        
        #print(f"  Playing game...", end="", flush=True)
        
        while not board.is_game_over() and move_count < max_moves:
            analysis = self.engine.analyse(board, limit=depth_limit)
            score = analysis['score'].white().score()
            
            if score is None:
                break
            score_float = score / 100.0
            
            if board.turn == chess.BLACK:
                score_float = -score_float
            
            features = featurize_board(board)
            positions.append(TrainingPosition(
                features=features,
                score=score_float,
                result=0.0
            ))
            
            result = self.engine.play(board, limit=depth_limit)
            board.push(result.move)
            move_count += 1
            
            if move_count % 20 == 0:
                #print(".", end="", flush=True)
        
        #print(f" done ({move_count} moves)")
        
        result = 0.0
        if board.is_checkmate():
            result = 1.0 if move_count % 2 == 1 else 0.0
        elif board.is_stalemate() or board.is_insufficient_material():
            result = 0.5
        elif board.is_fivefold_repetition() or board.is_seventyfive_moves():
            result = 0.5
        
        for pos in positions:
            pos.result = result
            
        return positions
    
    def generate_data(self, num_games: int = Config.NUM_GAMES, 
                      output_file: str = Config.DATA_FILE):
        #print(f"Generating {num_games} self-play games at depth {Config.DEPTH}...")
        #print(f"Using Stockfish from: {self.stockfish_path}")
        
        self.start_engine()
        all_positions = []
        openings = self._get_openings()
        start_time = time.time()
        
        for game_idx in range(num_games):
            opening = openings[game_idx % len(openings)]
            
            try:
                positions = self.play_game(opening)
                all_positions.extend(positions)
                
                if (game_idx + 1) % 10 == 0:
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
        self._save_data(all_positions, output_file)
        #print(f"Generated {len(all_positions)} positions from {num_games} games")
        return all_positions
    
    def _get_openings(self) -> List[str]:
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
        with open(filename, 'wb') as f:
            f.write(struct.pack('4sI', b'NNUE', len(positions)))
            for pos in positions:
                f.write(pos.features.tobytes())
                f.write(struct.pack('ff', pos.score, pos.result))
        
        #print(f"Saved {len(positions)} positions to {filename}")
        file_size = os.path.getsize(filename) / (1024 * 1024)
        #print(f"File size: {file_size:.2f} MB")

# ============== Dataset Class ==============
class NNUE_Dataset(Dataset):
    def __init__(self, data_file: str):
        self.features = []
        self.scores = []
        self.results = []
        self._load_data(data_file)
    
    def _load_data(self, filename: str):
        #print(f"Loading training data from {filename}...")
        
        with open(filename, 'rb') as f:
            magic, count = struct.unpack('4sI', f.read(8))
            if magic != b'NNUE':
                raise ValueError(f"Invalid file format: expected 'NNUE', got {magic}")
            
            #print(f"Found {count} positions")
            
            for i in range(count):
                if i % 100000 == 0 and i > 0:
                    #print(f"  Loaded {i} positions...")
                
                feat_data = f.read(Config.NNUE_INPUT_DIM * 4)
                feat = np.frombuffer(feat_data, dtype=np.float32)
                self.features.append(feat)
                
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #print(f"Using device: {device}")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                          weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    # VERSAO CORRIGIDA - sem verbose=True
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32).to(device)
    val_scores_tensor = torch.tensor(val_scores, dtype=torch.float32).to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
    
    #print(f"Starting training for {epochs} epochs...")
    #print(f"Training samples: {len(train_loader.dataset)}")
    #print(f"Validation samples: {len(val_features)}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for batch_idx, (features, scores) in enumerate(train_loader):
            features = features.to(device)
            scores = scores.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, scores)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRADIENT_CLIP)
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 100 == 0:
                #print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / batch_count
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_features_tensor)
            val_loss = criterion(val_outputs, val_scores_tensor).item()
        
        # Atualiza scheduler e mostra learning rate manualmente
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        lr_changed = old_lr != new_lr
        
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        elapsed = time.time() - start_time
        #print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
              f"Time: {elapsed/60:.1f} min")
        
        if lr_changed:
            #print(f"  📉 Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_FILE)
            #print(f"  ✅ New best model saved (val loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                #print(f"  ⏹️ Early stopping triggered (patience: {Config.PATIENCE})")
                break
    
    # Carregar melhor modelo
    model.load_state_dict(torch.load(Config.MODEL_FILE))
    #print(f"Training completed! Best validation loss: {best_val_loss:.6f}")
    
    # Salvar histórico de treinamento
    with open('training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    return model

# ============== Menu System ==============
defprint_menu():
    """#print the main menu"""
    #print("\n" + "=" * 70)
    #print(" 🧠 NNUE Training Pipeline - Advanced Menu")
    #print("=" * 70)
    #print("""
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Generate Training Data                                   │
  │ 2. Load Existing Data and Train                             │
  │ 3. Full Pipeline (Generate + Train + Export)                │
  │ 4. Export Only (from existing model)                        │
  │ 5. Advanced Configuration                                   │
  │ 6. View Training Statistics                                 │
  │ 7. Test Model on Positions                                  │
  │ 8. Compare with Stockfish                                   │
  │ 9. Exit                                                     │
  └─────────────────────────────────────────────────────────────┘
  """)
    #print("=" * 70)

def get_user_input(prompt: str, default=None, type_func=str):
    """Get validated user input"""
    while True:
        user_input = input(prompt)
        if user_input == '' and default is not None:
            return default
        try:
            return type_func(user_input)
        except ValueError:
            #print(f"Invalid input. Please enter a valid {type_func.__name__}.")

def show_config():
    """Display current configuration"""
    #print("\n" + "=" * 70)
    #print("Current Configuration")
    #print("=" * 70)
    
    config_items = {
        'Network Architecture': {
            'Input Dimension': Config.NNUE_INPUT_DIM,
            'Hidden Layer 1': Config.NNUE_H1,
            'Hidden Layer 2': Config.NNUE_H2,
            'Output Dimension': Config.NNUE_OUT,
        },
        'Training Parameters': {
            'Batch Size': Config.BATCH_SIZE,
            'Learning Rate': Config.LEARNING_RATE,
            'Epochs': Config.EPOCHS,
            'Validation Split': f"{Config.VALIDATION_SPLIT*100}%",
            'Weight Decay': Config.WEIGHT_DECAY,
            'Gradient Clip': Config.GRADIENT_CLIP,
            'Early Stopping Patience': Config.PATIENCE,
        },
        'Data Generation': {
            'Stockfish Path': Config.STOCKFISH_PATH,
            'Search Depth': Config.DEPTH,
            'Number of Games': Config.NUM_GAMES,
            'Max Moves per Game': Config.MAX_MOVES,
        },
        'Files': {
            'Data File': Config.DATA_FILE,
            'Model File': Config.MODEL_FILE,
            'Weights File': Config.WEIGHTS_FILE,
        }
    }
    
    for section, items in config_items.items():
        #print(f"\n📁 {section}:")
        for key, value in items.items():
            #print(f"  {key:20} : {value}")

def advanced_config():
    """Advanced configuration menu"""
    #print("\n" + "=" * 70)
    #print("Advanced Configuration")
    #print("=" * 70)
    #print("""
  1. Network Architecture (Hidden Layer Sizes)
  2. Training Parameters (LR, Batch Size, Epochs)
  3. Data Generation (Depth, Games, Stockfish Path)
  4. Save Configuration to File
  5. Load Configuration from File
  6. Reset to Defaults
  7. Return to Main Menu
  """)
    
    choice = get_user_input("Select option (1-7): ", type_func=int)
    
    if choice == 1:
        #print("\nNetwork Architecture Configuration:")
        Config.NNUE_H1 = get_user_input(f"Hidden Layer 1 size (current: {Config.NNUE_H1}): ", 
                                       default=Config.NNUE_H1, type_func=int)
        Config.NNUE_H2 = get_user_input(f"Hidden Layer 2 size (current: {Config.NNUE_H2}): ", 
                                       default=Config.NNUE_H2, type_func=int)
        #print("✅ Network architecture updated!")
    
    elif choice == 2:
        #print("\nTraining Parameters Configuration:")
        Config.LEARNING_RATE = get_user_input(f"Learning Rate (current: {Config.LEARNING_RATE}): ", 
                                             default=Config.LEARNING_RATE, type_func=float)
        Config.BATCH_SIZE = get_user_input(f"Batch Size (current: {Config.BATCH_SIZE}): ", 
                                          default=Config.BATCH_SIZE, type_func=int)
        Config.EPOCHS = get_user_input(f"Epochs (current: {Config.EPOCHS}): ", 
                                      default=Config.EPOCHS, type_func=int)
        Config.WEIGHT_DECAY = get_user_input(f"Weight Decay (current: {Config.WEIGHT_DECAY}): ", 
                                            default=Config.WEIGHT_DECAY, type_func=float)
        Config.PATIENCE = get_user_input(f"Early Stopping Patience (current: {Config.PATIENCE}): ", 
                                        default=Config.PATIENCE, type_func=int)
        #print("✅ Training parameters updated!")
    
    elif choice == 3:
        #print("\nData Generation Configuration:")
        Config.STOCKFISH_PATH = get_user_input(f"Stockfish Path (current: {Config.STOCKFISH_PATH}): ", 
                                              default=Config.STOCKFISH_PATH, type_func=str)
        Config.DEPTH = get_user_input(f"Search Depth (current: {Config.DEPTH}): ", 
                                     default=Config.DEPTH, type_func=int)
        Config.NUM_GAMES = get_user_input(f"Number of Games (current: {Config.NUM_GAMES}): ", 
                                         default=Config.NUM_GAMES, type_func=int)
        Config.MAX_MOVES = get_user_input(f"Max Moves per Game (current: {Config.MAX_MOVES}): ", 
                                         default=Config.MAX_MOVES, type_func=int)
        #print("✅ Data generation parameters updated!")
    
    elif choice == 4:
        save_config()
    elif choice == 5:
        load_config()
    elif choice == 6:
        reset_config()
    elif choice == 7:
        return

def save_config():
    """Save configuration to JSON file"""
    config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('__')}
    with open(Config.CONFIG_FILE, 'w') as f:
        json.dump(config_dict, f, indent=4, default=str)
    #print(f"✅ Configuration saved to {Config.CONFIG_FILE}")

def load_config():
    """Load configuration from JSON file"""
    try:
        with open(Config.CONFIG_FILE, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            if hasattr(Config, key):
                setattr(Config, key, value)
        #print(f"✅ Configuration loaded from {Config.CONFIG_FILE}")
    except FileNotFoundError:
        #print(f"❌ Configuration file {Config.CONFIG_FILE} not found")
    except Exception as e:
        #print(f"❌ Error loading configuration: {e}")

def reset_config():
    """Reset configuration to defaults"""
    # Store important paths
    old_data_file = Config.DATA_FILE
    old_model_file = Config.MODEL_FILE
    old_weights_file = Config.WEIGHTS_FILE
    
    # Reset class
    class DefaultConfig:
        PIECE_NB = 6
        NNUE_INPUT_DIM = 2 * PIECE_NB * 64
        NNUE_H1 = 256
        NNUE_H2 = 256
        NNUE_OUT = 1
        BATCH_SIZE = 1024
        LEARNING_RATE = 0.001
        EPOCHS = 20
        VALIDATION_SPLIT = 0.1
        WEIGHT_DECAY = 1e-5
        GRADIENT_CLIP = 1.0
        PATIENCE = 5
        DEPTH = 8
        NUM_GAMES = 100
        POSITIONS_PER_GAME = 50
        STOCKFISH_PATH = "/usr/games/stockfish"
        MAX_MOVES = 200
        DATA_FILE = old_data_file
        MODEL_FILE = old_model_file
        WEIGHTS_FILE = old_weights_file
        CONFIG_FILE = "training_config.json"
        LOG_FILE = "training_log.txt"
    
    for key, value in DefaultConfig.__dict__.items():
        if not key.startswith('__'):
            setattr(Config, key, value)
    
    #print("✅ Configuration reset to defaults!")

def view_statistics():
    """View training statistics"""
    try:
        with open('training_history.json', 'r') as f:
            history = json.load(f)
        
        #print("\n" + "=" * 70)
        #print("Training Statistics")
        #print("=" * 70)
        
        #print(f"\n📊 Training Progress:")
        #print(f"  Epochs Completed: {len(history['train_loss'])}")
        #print(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
        #print(f"  Final Val Loss: {history['val_loss'][-1]:.6f}")
        #print(f"  Best Val Loss: {min(history['val_loss']):.6f}")
        
        #print(f"\n📉 Learning Rates:")
        for i, lr in enumerate(history['learning_rates']):
            #print(f"  Epoch {i+1}: {lr:.8f}")
        
    except FileNotFoundError:
        #print("❌ No training history found. Train a model first.")
    except Exception as e:
        #print(f"❌ Error viewing statistics: {e}")

def test_model():
    """Test the trained model on specific positions"""
    try:
        # Load model
        model = NNUE()
        model.load_state_dict(torch.load(Config.MODEL_FILE))
        model.eval()
        
        #print("\n" + "=" * 70)
        #print("Test Model on Positions")
        #print("=" * 70)
        #print("Enter FEN positions (or 'q' to quit):")
        
        while True:
            fen = input("\nFEN: ").strip()
            if fen.lower() == 'q':
                break
            
            try:
                board = chess.Board(fen)
                features = featurize_board(board)
                
                with torch.no_grad():
                    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    score = model(features_tensor).item()
                
                #print(f"  Evaluation: {score:.2f}")
                #print(f"  Board: {board.fen()}")
                #print(f"  To move: {'White' if board.turn == chess.WHITE else 'Black'}")
                
            except Exception as e:
                #print(f"  ❌ Error evaluating position: {e}")
                
    except FileNotFoundError:
        #print(f"❌ Model file {Config.MODEL_FILE} not found. Train a model first.")
    except Exception as e:
        #print(f"❌ Error: {e}")

def compare_with_stockfish():
    """Compare model evaluation with Stockfish"""
    try:
        import chess.engine
        
        # Load model
        model = NNUE()
        model.load_state_dict(torch.load(Config.MODEL_FILE))
        model.eval()
        
        # Start Stockfish
        engine = chess.engine.SimpleEngine.popen_uci(Config.STOCKFISH_PATH)
        
        #print("\n" + "=" * 70)
        #print("Compare with Stockfish")
        #print("=" * 70)
        #print("Enter FEN positions (or 'q' to quit):")
        
        while True:
            fen = input("\nFEN: ").strip()
            if fen.lower() == 'q':
                break
            
            try:
                board = chess.Board(fen)
                features = featurize_board(board)
                
                # NNUE evaluation
                with torch.no_grad():
                    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    nnue_score = model(features_tensor).item()
                
                # Stockfish evaluation
                analysis = engine.analyse(board, chess.engine.Limit(depth=10))
                stockfish_score = analysis['score'].white().score()
                if stockfish_score is None:
                    stockfish_score = 0
                stockfish_score = stockfish_score / 100.0
                
                #print(f"\n  NNUE:      {nnue_score:.2f}")
                #print(f"  Stockfish: {stockfish_score:.2f}")
                #print(f"  Difference: {abs(nnue_score - stockfish_score):.2f}")
                
            except Exception as e:
                #print(f"  ❌ Error: {e}")
        
        engine.quit()
        
    except FileNotFoundError:
        #print(f"❌ Model or Stockfish not found")
    except Exception as e:
        #print(f"❌ Error: {e}")

# ============== Main Pipeline ==============
def generate_data_only():
    """Only generate training data"""
    #print("\n" + "=" * 70)
    #print("Generate Training Data Only")
    #print("=" * 70)
    
    generator = DataGenerator()
    
    if os.path.exists(Config.DATA_FILE):
        response = get_user_input(f"Data file {Config.DATA_FILE} exists. Overwrite? (y/n): ", 
                                 default='n')
        if response.lower() != 'y':
            #print("Skipping data generation...")
            return
    
    generator.generate_data()

def load_and_train():
    """Load existing data and train"""
    #print("\n" + "=" * 70)
    #print("Load Data and Train")
    #print("=" * 70)
    
    if not os.path.exists(Config.DATA_FILE):
        #print(f"❌ Data file {Config.DATA_FILE} not found!")
        #print("Generate data first (option 1) or run full pipeline (option 3)")
        return
    
    # Load data
    #print("\nLoading training data...")
    dataset = NNUE_Dataset(Config.DATA_FILE)
    
    # Split data
    total_size = len(dataset)
    val_size = int(total_size * Config.VALIDATION_SPLIT)
    train_size = total_size - val_size
    
    indices = list(range(total_size))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    val_features = np.array([dataset.features[i] for i in val_indices])
    val_scores = np.array([dataset.scores[i] for i in val_indices])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    #print(f"Training samples: {len(train_dataset)}")
    #print(f"Validation samples: {len(val_dataset)}")
    
    # Create and train model
    model = NNUE()
    #print(f"\nModel Architecture:")
    #print(f"  Input: {Config.NNUE_INPUT_DIM}")
    #print(f"  Hidden: {Config.NNUE_H1} -> {Config.NNUE_H2}")
    #print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model = train_model(model, train_loader, val_features, val_scores)
    
    # Export weights
    #print("\nExporting weights...")
    model.export_weights(Config.WEIGHTS_FILE)

def full_pipeline():
    """Run the complete pipeline"""
    #print("\n" + "=" * 70)
    #print("Full Training Pipeline")
    #print("=" * 70)
    
    # Step 1: Generate data
    generate_data_only()
    
    # Step 2: Train
    load_and_train()
    
    # Step 3: Verify
    verify_weights()
    
    #print("\n" + "=" * 70)
    #print("✅ Full pipeline completed successfully!")
    #print("=" * 70)

def export_only():
    """Export weights from existing model"""
    #print("\n" + "=" * 70)
    #print("Export Weights Only")
    #print("=" * 70)
    
    if not os.path.exists(Config.MODEL_FILE):
        #print(f"❌ Model file {Config.MODEL_FILE} not found!")
        return
    
    try:
        model = NNUE()
        model.load_state_dict(torch.load(Config.MODEL_FILE))
        model.eval()
        
        model.export_weights(Config.WEIGHTS_FILE)
        verify_weights()
        
    except Exception as e:
        #print(f"❌ Error exporting weights: {e}")

def verify_weights():
    """Verify exported weights"""
    try:
        if not os.path.exists(Config.WEIGHTS_FILE):
            #print(f"❌ Weights file {Config.WEIGHTS_FILE} not found!")
            return
        
        weights = np.fromfile(Config.WEIGHTS_FILE, dtype=np.float32)
        
        w1_size = Config.NNUE_INPUT_DIM * Config.NNUE_H1
        b1_size = Config.NNUE_H1
        w2_size = Config.NNUE_H1 * Config.NNUE_H2
        b2_size = Config.NNUE_H2
        w3_size = Config.NNUE_H2 * Config.NNUE_OUT
        b3_size = Config.NNUE_OUT
        total_expected = w1_size + b1_size + w2_size + b2_size + w3_size + b3_size
        
        #print(f"\n📊 Weight Verification:")
        #print(f"  File: {Config.WEIGHTS_FILE}")
        #print(f"  Floats: {len(weights)}")
        #print(f"  Expected: {total_expected}")
        
        if len(weights) == total_expected:
            #print("  ✅ Size matches C code expectations!")
            
            # Parse layers
            offset = 0
            w1 = weights[offset:offset + w1_size]
            offset += w1_size
            b1 = weights[offset:offset + b1_size]
            offset += b1_size
            w2 = weights[offset:offset + w2_size]
            offset += w2_size
            b2 = weights[offset:offset + b2_size]
            offset += b2_size
            w3 = weights[offset:offset + w3_size]
            offset += w3_size
            b3 = weights[offset:offset + b3_size]
            
            #print(f"  w1: {w1.shape} (mean: {w1.mean():.6f}, std: {w1.std():.6f})")
            #print(f"  b1: {b1.shape} (mean: {b1.mean():.6f}, std: {b1.std():.6f})")
            #print(f"  w2: {w2.shape} (mean: {w2.mean():.6f}, std: {w2.std():.6f})")
            #print(f"  b2: {b2.shape} (mean: {b2.mean():.6f}, std: {b2.std():.6f})")
            #print(f"  w3: {w3.shape} (mean: {w3.mean():.6f}, std: {w3.std():.6f})")
            #print(f"  b3: {b3.shape} (mean: {b3.mean():.6f}, std: {b3.std():.6f})")
        else:
            #print("  ❌ Size mismatch!")
            
    except Exception as e:
        #print(f"❌ Error verifying weights: {e}")

# ============== Main Program ==============
def main():
    """Main program with menu"""
    #print("\n" + "=" * 70)
    #print(" 🧠 NNUE Training Pipeline - Advanced Edition")
    #print(f" 📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    #print("=" * 70)
    
    # Check dependencies
    try:
        import chess
        import torch
        import numpy
        #print("✅ All dependencies satisfied")
    except ImportError as e:
        #print(f"❌ Missing dependency: {e}")
        #print("Please install required packages:")
        #print("  pip install python-chess torch numpy")
        return
    
    # Check Stockfish
    if not os.path.exists(Config.STOCKFISH_PATH):
        #print(f"⚠️ Stockfish not found at: {Config.STOCKFISH_PATH}")
        #print("  Install with: sudo apt-get install stockfish")
        #print("  Or update STOCKFISH_PATH in configuration")
    
    while True:
       print_menu()
        choice = get_user_input("Select option (1-9): ", type_func=int)
        
        if choice == 1:
            generate_data_only()
        elif choice == 2:
            load_and_train()
        elif choice == 3:
            full_pipeline()
        elif choice == 4:
            export_only()
        elif choice == 5:
            advanced_config()
        elif choice == 6:
            view_statistics()
        elif choice == 7:
            test_model()
        elif choice == 8:
            compare_with_stockfish()
        elif choice == 9:
            #print("\n👋 Goodbye!")
            break
        else:
            #print("❌ Invalid option. Please select 1-9.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()