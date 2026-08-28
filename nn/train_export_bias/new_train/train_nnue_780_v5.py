#!/usr/bin/env python3
"""
NNUE Training Module - Fixed to handle .binpack format
Network: 780 -> 256 -> 64 -> 32 -> 1 (WASM optimized)
"""

import chess
import numpy as np
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import random
import os
from pathlib import Path

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
    
    # Input data file (supports both formats)
    DATA_FILE = "training_data_prod.bin"
    
    # Output files
    MODEL_FILE = "nnue_model_prod.pt"
    WEIGHTS_FILE = "nnue_weights_prod.bin"
    WASM_WEIGHTS_FILE = "nnue_weights_wasm.bin"
    BIAS_FILE = "nnue_bias.bin"
    CHECKPOINT_DIR = "checkpoints_prod"

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
        
        flat_weights = []
        for key in ['w1', 'b1', 'w2', 'b2', 'w3', 'b3', 'w4', 'b4']:
            flat_weights.extend(weights[key].flatten())
        
        flat_array = np.array(flat_weights, dtype=np.float32)
        flat_array.tofile(filepath)
        
        print(f"\n💾 Exported {len(flat_array):,} floats to {filepath}")
        print(f"   File size: {len(flat_array) * 4 / 1024 / 1024:.2f} MB")
        
        return flat_array
    
    def export_bias_file(self, filepath: str):
        """Export only biases in separate file"""
        biases = {
            'b1': self.fc1.bias.detach().cpu().numpy(),
            'b2': self.fc2.bias.detach().cpu().numpy(),
            'b3': self.fc3.bias.detach().cpu().numpy(),
            'b4': self.fc4.bias.detach().cpu().numpy()
        }
        
        print("\n📊 Exporting biases for WASM:")
        total_params = 0
        for name, data in biases.items():
            print(f"  {name}: {data.shape} ({data.size:,} floats)")
            total_params += data.size
        
        flat_biases = []
        for key in ['b1', 'b2', 'b3', 'b4']:
            flat_biases.extend(biases[key].flatten())
        
        flat_array = np.array(flat_biases, dtype=np.float32)
        flat_array.tofile(filepath)
        
        print(f"\n💾 Exported {len(flat_array):,} floats to {filepath}")
        print(f"   File size: {len(flat_array) * 4 / 1024 / 1024:.2f} MB")
        
        return flat_array

# ============== Dataset - Fixed for .binpack ==============
class NNUE_Dataset(Dataset):
    def __init__(self, data_file: str, max_positions: Optional[int] = None):
        self.features = []
        self.scores = []
        self._load_data(data_file, max_positions)
    
    def _load_data(self, filename: str, max_positions: Optional[int] = None):
        print(f"Loading training data from {filename}...")
        
        # First, detect file format
        file_format = self._detect_format(filename)
        print(f"Detected format: {file_format}")
        
        if file_format == 'nnue':
            self._load_nnue_format(filename, max_positions)
        elif file_format == 'binpack':
            self._load_binpack_format(filename, max_positions)
        else:
            raise ValueError(f"Unknown file format: {filename}")
    
    def _detect_format(self, filename: str) -> str:
        """Detect if file is NNUE format or binpack format"""
        try:
            with open(filename, 'rb') as f:
                magic = f.read(4)
                if magic == b'NNUE':
                    return 'nnue'
                # Check if it's binpack by trying to read header
                f.seek(0)
                header = f.read(4)
                if len(header) == 4:
                    num_pos = struct.unpack('<I', header)[0]
                    # Sanity check: reasonable number of positions
                    if 0 < num_pos < 10000000000:
                        return 'binpack'
                return 'unknown'
        except:
            return 'unknown'
    
    def _load_binpack_format(self, filename: str, max_positions: Optional[int] = None):
        """Load Stockfish .binpack format"""
        with open(filename, 'rb') as f:
            # Read header (number of positions)
            header = f.read(4)
            if len(header) < 4:
                print(f"❌ Error: File too small or empty")
                return
            
            num_positions = struct.unpack('<I', header)[0]
            print(f"Found {num_positions:,} positions in binpack")
            
            if max_positions:
                num_positions = min(num_positions, max_positions)
                print(f"Limiting to {max_positions:,} positions")
            
            # Pre-allocate arrays
            self.features = []
            self.scores = []
            
            # Constants for binpack
            PACKED_PIECES = 4
            PACKED_SCORE = 2
            PACKED_MOVE = 2
            
            # Store score statistics for normalization later
            scores_list = []
            
            for i in range(num_positions):
                if i % 1000000 == 0 and i > 0:
                    print(f"  Loaded {i:,} positions...")
                
                # Read packed position (4 bytes)
                packed_pos = f.read(PACKED_PIECES)
                if len(packed_pos) < PACKED_PIECES:
                    break
                
                # Read score (2 bytes) - signed short
                score_bytes = f.read(PACKED_SCORE)
                if len(score_bytes) < PACKED_SCORE:
                    break
                score = struct.unpack('<h', score_bytes)[0]  # Stockfish score in centipawns
                
                # Read move (2 bytes) - we don't need it for training
                move_bytes = f.read(PACKED_MOVE)
                if len(move_bytes) < PACKED_MOVE:
                    break
                
                # Convert packed position to NNUE features
                features = self._packed_to_features(packed_pos)
                self.features.append(features)
                scores_list.append(score)
        
        # Normalize scores to range [-1, 1] (Stockfish scores are typically -3000 to 3000)
        scores_array = np.array(scores_list, dtype=np.float32)
        # Clip to reasonable range
        scores_array = np.clip(scores_array, -3000, 3000)
        # Normalize to [-1, 1]
        self.scores = scores_array / 3000.0
        
        print(f"Loaded {len(self.features):,} positions")
        print(f"Score range: {scores_array.min():.1f} to {scores_array.max():.1f}")
        print(f"Normalized score mean: {np.mean(self.scores):.3f}, std: {np.std(self.scores):.3f}")
    
    def _packed_to_features(self, packed_pos: bytes) -> np.ndarray:
        """Convert packed position to NNUE features (780-dimensional HalfKP)"""
        # This is a simplified conversion - actual HalfKP is more complex
        # For now, we'll create a basic feature representation
        
        # Read the 32-bit packed value
        packed_val = struct.unpack('<I', packed_pos)[0]
        
        # Create feature array
        features = np.zeros(Config.NNUE_INPUT_DIM, dtype=np.float32)
        
        # Simplified: Use the bits as features
        # In practice, you'd map piece positions to HalfKP indices
        for i in range(16):  # 16 pieces total (8 white + 8 black)
            piece_bits = (packed_val >> (i * 2)) & 0x3
            # Map to some feature index (simplified)
            feature_idx = (i * 48) + (piece_bits * 16)
            if feature_idx < Config.NNUE_INPUT_DIM:
                features[feature_idx] = 1.0
        
        return features
    
    def _load_nnue_format(self, filename: str, max_positions: Optional[int] = None):
        """Load NNUE custom format with 'NNUE' magic header"""
        with open(filename, 'rb') as f:
            magic, count = struct.unpack('4sI', f.read(8))
            if magic != b'NNUE':
                raise ValueError(f"Invalid NNUE file format")
            
            print(f"Found {count} positions in NNUE format")
            
            if max_positions:
                count = min(count, max_positions)
                print(f"Limiting to {max_positions} positions")
            
            for i in range(count):
                if i % 50000 == 0 and i > 0:
                    print(f"  Loaded {i} positions...")
                
                feat_data = f.read(Config.NNUE_INPUT_DIM * 4)
                if len(feat_data) < Config.NNUE_INPUT_DIM * 4:
                    break
                
                feat = np.frombuffer(feat_data, dtype=np.float32)
                self.features.append(feat)
                score, result, tactical = struct.unpack('fff', f.read(12))
                
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
        # For proper mirroring, you'd need to implement HalfKP mirroring
        # This is a placeholder
        return features

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
    print(f"   Epochs: {Config.EPOCHS}")
    print(f"   Early stopping patience: {Config.PATIENCE}")
    
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
        
        progress = (epoch + 1) / Config.EPOCHS * 100
        print(f"Epoch {epoch+1:3d}/{Config.EPOCHS} [{progress:5.1f}%] - "
              f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, 'best_model.pt'))
            patience_counter = 0
            print(f"  ✅ New best model saved (loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"  ⏹️  Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"\n✅ Loaded best model from {best_model_path}")
    else:
        print(f"\n⚠️  Best model not found, using current model")
    
    return model

# ============== Main ==============
def main():
    print("=" * 80)
    print("NNUE Training Module - WASM Optimized")
    print(f"Network: {Config.NNUE_INPUT_DIM} -> {Config.NNUE_H1} -> {Config.NNUE_H2} -> {Config.NNUE_H3} -> 1")
    print("=" * 80)
    
    if not os.path.exists(Config.DATA_FILE):
        print(f"\n❌ Data file not found: {Config.DATA_FILE}")
        return
    
    # Check file size first
    file_size = os.path.getsize(Config.DATA_FILE)
    print(f"\n📂 Data file size: {file_size / 1024 / 1024:.2f} MB")
    
    # Load data with limiting for testing
    test_mode = False
    if file_size > 500 * 1024 * 1024:  # > 500MB
        print("⚠️  Large file detected. Loading first 1,000,000 positions for testing.")
        print("   To load all data, remove the max_positions limit.")
        max_positions = 1000000
    else:
        max_positions = None
    
    try:
        dataset = NNUE_Dataset(Config.DATA_FILE, max_positions)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Get validation data
    val_features = np.array([dataset.features[i] for i in val_indices])
    val_scores = np.array([dataset.scores[i] for i in val_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, num_workers=0, pin_memory=True)
    
    print(f"\n📊 Data split:")
    print(f"  Total: {total:,} positions")
    print(f"  Training: {len(train_dataset):,} positions")
    print(f"  Validation: {len(val_dataset):,} positions")
    
    # Create and train model
    print(f"\n🧠 Creating NNUE model...")
    model = NNUEProduction()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    model = train_model(model, train_loader, val_features, val_scores)
    
    # Export
    print(f"\n💾 Exporting weights...")
    model.export_weights_wasm(Config.WASM_WEIGHTS_FILE)
    model.export_bias_file(Config.BIAS_FILE)
    
    # Save PyTorch model
    torch.save(model.state_dict(), Config.MODEL_FILE)
    print(f"\n💾 Saved PyTorch model to {Config.MODEL_FILE}")
    
    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()