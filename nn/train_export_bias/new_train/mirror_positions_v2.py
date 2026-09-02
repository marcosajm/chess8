#!/usr/bin/env python3
"""
NNUE Data Mirror Tool - CORRECTED VERSION
Properly handles result flipping for mirrored positions
"""

import chess
import numpy as np
import struct
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

# ============== Configuration ==============
class Config:
    INPUT_FILE = "training_data.bin"
    OUTPUT_FILE = None  # Auto-generated
    NNUE_INPUT_DIM = 780

# ============== Feature Extraction ==============
def calculate_tactical_threats(board: chess.Board) -> float:
    """Calculate tactical threats for defensive awareness"""
    score = 0.0
    turn = board.turn
    
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
    
    # 11: Tactical threat indicator
    features[feature_idx] = calculate_tactical_threats(board)
    feature_idx += 1
    
    # 12: King safety indicator
    features[feature_idx] = calculate_king_safety(board)
    
    return features

# ============== Mirror Functions ==============
def mirror_features_direct(features: np.ndarray) -> np.ndarray:
    """
    Mirrors the feature vector directly
    """
    mirrored = np.zeros(780, dtype=np.float32)
    
    # 1. Mirror piece features (768)
    for side in range(2):
        for piece_type in range(6):
            base_idx = side * 384 + piece_type * 64
            mirrored_base_idx = (1 - side) * 384 + piece_type * 64
            
            for sq in range(64):
                file = sq % 8
                rank = sq // 8
                mirrored_sq = (7 - file) + rank * 8
                mirrored[mirrored_base_idx + mirrored_sq] = features[base_idx + sq]
    
    # 2. Mirror castling rights (768-771)
    mirrored[768] = features[769]
    mirrored[769] = features[768]
    mirrored[770] = features[771]
    mirrored[771] = features[770]
    
    # 3. En passant (772-773) - unchanged
    mirrored[772] = features[772]
    mirrored[773] = features[773]
    
    # 4. Half-move clock (774) - unchanged
    mirrored[774] = features[774]
    
    # 5. Full move number (775) - unchanged
    mirrored[775] = features[775]
    
    # 6. Side to move (776) - unchanged
    mirrored[776] = features[776]
    
    # 7. Material balance (777) - flipped sign
    mirrored[777] = -features[777]
    
    # 8. Tactical threats (778) - approximate
    mirrored[778] = features[778]
    
    # 9. King safety (779) - approximate
    mirrored[779] = features[779]
    
    return mirrored

# ============== Main Mirror Tool ==============
@dataclass
class Position:
    features: np.ndarray
    score: float
    result: float  # 1.0 = White wins, 0.0 = Black wins, 0.5 = Draw
    tactical_score: float

def read_nnue_binary(filename: str) -> Tuple[List[Position], int]:
    """Reads NNUE binary file"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    positions = []
    
    with open(filename, 'rb') as f:
        header = f.read(8)
        if len(header) < 8:
            raise ValueError("Invalid file: header too small")
        
        magic, num_positions = struct.unpack('4sI', header)
        
        if magic != b'NNUE':
            raise ValueError(f"Invalid magic number: {magic}. Expected 'NNUE'")
        
        print(f"📖 Reading {num_positions} positions from {filename}")
        
        for i in range(num_positions):
            features_bytes = f.read(780 * 4)
            if len(features_bytes) < 780 * 4:
                break
            
            features = np.frombuffer(features_bytes, dtype=np.float32).copy()
            
            score_bytes = f.read(12)
            if len(score_bytes) < 12:
                break
            
            score, result, tactical_score = struct.unpack('fff', score_bytes)
            
            positions.append(Position(
                features=features,
                score=score,
                result=result,
                tactical_score=tactical_score
            ))
    
    print(f"✅ Successfully read {len(positions)} positions")
    return positions, len(positions)

def write_nnue_binary(filename: str, positions: List[Position]):
    """Writes NNUE binary file"""
    print(f"💾 Writing {len(positions)} positions to {filename}")
    
    with open(filename, 'wb') as f:
        f.write(struct.pack('4sI', b'NNUE', len(positions)))
        
        for pos in positions:
            f.write(pos.features.tobytes())
            f.write(struct.pack('fff', pos.score, pos.result, pos.tactical_score))
    
    file_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"✅ Successfully saved to {filename} ({file_size:.2f} MB)")

def mirror_positions_corrected(positions: List[Position]) -> List[Position]:
    """
    Creates mirrored versions with CORRECT result flipping.
    
    CRITICAL: When we mirror a position:
    - White becomes Black and vice versa
    - If original result = 1.0 (White wins), mirrored result = 0.0 (Black wins)
    - If original result = 0.0 (Black wins), mirrored result = 1.0 (White wins)
    - If draw (0.5), stays 0.5
    """
    print(f"🔄 Mirroring {len(positions)} positions with CORRECT result flipping...")
    
    mirrored_positions = []
    total = len(positions)
    progress_interval = max(1, total // 20)
    
    # Statistics for verification
    results_original = {1.0: 0, 0.0: 0, 0.5: 0}
    results_mirrored = {1.0: 0, 0.0: 0, 0.5: 0}
    
    for idx, pos in enumerate(positions):
        # Count original results
        results_original[pos.result] = results_original.get(pos.result, 0) + 1
        
        # Mirror features
        mirrored_features = mirror_features_direct(pos.features)
        
        # CRITICAL FIX: Flip the result!
        # Original: 1.0 (White wins) -> Mirrored: 0.0 (Black wins)
        # Original: 0.0 (Black wins) -> Mirrored: 1.0 (White wins)
        # Original: 0.5 (Draw)      -> Mirrored: 0.5 (Draw)
        if pos.result == 1.0:
            mirrored_result = 0.0  # White wins -> Black wins
        elif pos.result == 0.0:
            mirrored_result = 1.0  # Black wins -> White wins
        else:
            mirrored_result = 0.5  # Draw stays draw
        
        # Count mirrored results
        results_mirrored[mirrored_result] = results_mirrored.get(mirrored_result, 0) + 1
        
        # Score is negated (since sides are swapped)
        mirrored_score = -pos.score
        
        mirrored_positions.append(Position(
            features=mirrored_features,
            score=mirrored_score,
            result=mirrored_result,  # FLIPPED!
            tactical_score=pos.tactical_score  # Approximation
        ))
        
        if (idx + 1) % progress_interval == 0:
            progress = (idx + 1) / total * 100
            print(f"  Progress: {progress:.0f}% ({idx + 1}/{total})")
    
    # #print statistics
    print(f"\n📊 Result distribution:")
    print(f"  Original:  White wins: {results_original[1.0]}, Black wins: {results_original[0.0]}, Draws: {results_original[0.5]}")
    print(f"  Mirrored:  White wins: {results_mirrored[1.0]}, Black wins: {results_mirrored[0.0]}, Draws: {results_mirrored[0.5]}")
    
    # Verify balance
    total_orig_wins = results_original[1.0] + results_original[0.0]
    total_mirr_wins = results_mirrored[1.0] + results_mirrored[0.0]
    
    if results_original[1.0] > 0 and results_mirrored[0.0] > 0:
        print(f"\n✅ Result flipping correct!")
        print(f"   Original White wins: {results_original[1.0]} -> Mirrored Black wins: {results_mirrored[0.0]}")
    else:
        print(f"\n⚠️  Warning: No result flipping detected! Check your data.")
    
    print(f"✅ Created {len(mirrored_positions)} mirrored positions")
    return mirrored_positions

def analyze_dataset(positions: List[Position], name: str = "Dataset"):
    """Analyzes dataset statistics"""
    total = len(positions)
    if total == 0:
        return
    
    white_wins = sum(1 for p in positions if p.result == 1.0)
    black_wins = sum(1 for p in positions if p.result == 0.0)
    draws = sum(1 for p in positions if p.result == 0.5)
    
    print(f"\n📊 {name} Analysis:")
    print(f"  Total positions: {total}")
    print(f"  White wins: {white_wins} ({white_wins/total*100:.1f}%)")
    print(f"  Black wins: {black_wins} ({black_wins/total*100:.1f}%)")
    print(f"  Draws: {draws} ({draws/total*100:.1f}%)")
    
    # Check balance
    if white_wins > 0 and black_wins > 0:
        ratio = max(white_wins, black_wins) / min(white_wins, black_wins)
        if ratio > 1.5:
            print(f"  ⚠️  Dataset is imbalanced (ratio: {ratio:.2f}:1)")
        else:
            print(f"  ✅ Dataset is well balanced (ratio: {ratio:.2f}:1)")

# ============== Main ==============
def main():
    print("\n" + "="*80)
    print("🪞 NNUE DATA MIRROR TOOL - CORRECTED VERSION")
    print("   (With proper result flipping for mirrored positions)")
    print("="*80)
    
    # Configure input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = Config.INPUT_FILE
    
    # Configure output file
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_mirrored_corrected.bin"
    
    print(f"📁 Input file:  {input_file}")
    print(f"📁 Output file: {output_file}")
    print("="*80)
    
    try:
        # Read original data
        positions, num_orig = read_nnue_binary(input_file)
        
        # Analyze original dataset
        analyze_dataset(positions, "Original Dataset")
        
        # Mirror positions with correct result flipping
        mirrored_positions = mirror_positions_corrected(positions)
        
        # Analyze mirrored dataset
        analyze_dataset(mirrored_positions, "Mirrored Dataset")
        
        # Combine original and mirrored
        all_positions = positions + mirrored_positions
        
        # Analyze combined dataset
        analyze_dataset(all_positions, "Combined Dataset")
        
        # Save combined data
        write_nnue_binary(output_file, all_positions)
        
        print("\n" + "="*80)
        print("✅ Mirroring complete!")
        print(f"  Original: {input_file} ({len(positions)} positions)")
        print(f"  Output:   {output_file} ({len(all_positions)} positions)")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.#print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()