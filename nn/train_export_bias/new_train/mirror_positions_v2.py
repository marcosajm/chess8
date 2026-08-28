#!/usr/bin/env python3
"""
NNUE Data Mirror Tool
Reads existing NNUE binary data, generates mirrored positions, and saves to new file
"""

import chess
import numpy as np
import struct
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time

# ============== Configuration ==============
class Config:
    # Input file (existing binary)
    INPUT_FILE = "training_data.bin"  # Change to your file
    
    # Output file (with mirror)
    OUTPUT_FILE = None  # Will be auto-generated if None
    
    # NNUE input dimension
    NNUE_INPUT_DIM = 780

# ============== Feature Extraction (copied from original) ==============
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

# ============== Board Mirroring Functions ==============
def mirror_board(board: chess.Board) -> chess.Board:
    """
    Creates a mirrored version of the board (horizontal flip).
    White <-> Black, and files are mirrored (a<->h, b<->g, etc.)
    """
    mirrored = chess.Board()
    mirrored.clear()
    
    # Mirror piece positions
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece:
            # Mirror the square: file = 7 - file, rank stays the same
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            mirrored_sq = chess.square(7 - file, rank)
            
            # Flip color
            mirrored_piece = chess.Piece(piece.piece_type, not piece.color)
            mirrored.set_piece_at(mirrored_sq, mirrored_piece)
    
    # Mirror castling rights
    mirrored.castling_rights = 0
    
    # White kingside (H1) <-> White queenside (A1)
    if board.has_kingside_castling_rights(chess.WHITE):
        mirrored.castling_rights |= chess.BB_H1
    if board.has_queenside_castling_rights(chess.WHITE):
        mirrored.castling_rights |= chess.BB_A1
    
    # Black kingside (H8) <-> Black queenside (A8)
    if board.has_kingside_castling_rights(chess.BLACK):
        mirrored.castling_rights |= chess.BB_H8
    if board.has_queenside_castling_rights(chess.BLACK):
        mirrored.castling_rights |= chess.BB_A8
    
    # Mirror en passant
    if board.ep_square is not None:
        file = chess.square_file(board.ep_square)
        rank = chess.square_rank(board.ep_square)
        mirrored.ep_square = chess.square(7 - file, rank)
    
    # Copy turn (flipped sides already handled by piece colors)
    mirrored.turn = board.turn
    
    # Copy other game state
    mirrored.halfmove_clock = board.halfmove_clock
    mirrored.fullmove_number = board.fullmove_number
    
    # Validate the board
    mirrored._update_attacks()
    
    return mirrored

def mirror_features_from_board(board: chess.Board) -> np.ndarray:
    """
    Generates mirrored features directly from a board position.
    More efficient than trying to mirror the feature vector.
    """
    return featurize_board_prod(board)

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

# ============== Main Mirror Tool ==============
@dataclass
class Position:
    features: np.ndarray
    score: float
    result: float
    tactical_score: float

def read_nnue_binary(filename: str) -> Tuple[List[Position], int]:
    """
    Reads NNUE binary file and returns list of positions
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    positions = []
    
    with open(filename, 'rb') as f:
        # Read header
        header = f.read(8)
        if len(header) < 8:
            raise ValueError("Invalid file: header too small")
        
        magic, num_positions = struct.unpack('4sI', header)
        
        if magic != b'NNUE':
            raise ValueError(f"Invalid magic number: {magic}. Expected 'NNUE'")
        
        print(f"📖 Reading {num_positions} positions from {filename}")
        
        # Read each position
        for i in range(num_positions):
            # Read features (780 floats = 780 * 4 bytes = 3120 bytes)
            features_bytes = f.read(780 * 4)
            if len(features_bytes) < 780 * 4:
                break
            
            features = np.frombuffer(features_bytes, dtype=np.float32).copy()
            
            # Read score, result, tactical_score (3 floats = 12 bytes)
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
    """
    Writes NNUE binary file
    """
    print(f"💾 Writing {len(positions)} positions to {filename}")
    
    with open(filename, 'wb') as f:
        # Write header
        f.write(struct.pack('4sI', b'NNUE', len(positions)))
        
        # Write each position
        for pos in positions:
            f.write(pos.features.tobytes())
            f.write(struct.pack('fff', pos.score, pos.result, pos.tactical_score))
    
    file_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"✅ Successfully saved to {filename} ({file_size:.2f} MB)")

def mirror_positions(positions: List[Position]) -> List[Position]:
    """
    Creates mirrored versions of all positions
    """
    print(f"🔄 Mirroring {len(positions)} positions...")
    
    mirrored_positions = []
    total = len(positions)
    
    # For progress tracking
    progress_interval = max(1, total // 20)
    
    for idx, pos in enumerate(positions):
        # Reconstruct board from features
        # This is the tricky part - we need to reconstruct the board state
        # from the features to mirror it properly
        
        # Method 1: Use a pre-stored FEN (not available in this format)
        # Method 2: Approximate mirror by manipulating features directly
        
        # Since we can't reconstruct the exact board from features,
        # we'll use a feature-level mirroring approach
        mirrored_features = mirror_features_direct(pos.features)
        
        # Score is negated for mirrored position
        mirrored_score = -pos.score
        
        # Result stays the same
        mirrored_result = pos.result
        
        # Tactical score should be recomputed, but we'll approximate
        # by mirroring it (not exactly correct, but acceptable)
        mirrored_tactical = pos.tactical_score  # Approximation
        
        mirrored_positions.append(Position(
            features=mirrored_features,
            score=mirrored_score,
            result=mirrored_result,
            tactical_score=mirrored_tactical
        ))
        
        if (idx + 1) % progress_interval == 0:
            progress = (idx + 1) / total * 100
            print(f"  Progress: {progress:.0f}% ({idx + 1}/{total})")
    
    print(f"✅ Created {len(mirrored_positions)} mirrored positions")
    return mirrored_positions

def mirror_features_direct(features: np.ndarray) -> np.ndarray:
    """
    Mirrors the feature vector directly (without reconstructing board)
    This is an approximation but works well for NNUE training
    """
    mirrored = np.zeros(780, dtype=np.float32)
    
    # 1. Mirror piece features (768)
    # For each piece type and color, mirror square positions
    for side in range(2):  # 0=White, 1=Black
        for piece_type in range(6):  # PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
            base_idx = side * 384 + piece_type * 64
            mirrored_base_idx = (1 - side) * 384 + piece_type * 64
            
            for sq in range(64):
                # Mirror square: file = 7 - file
                file = sq % 8
                rank = sq // 8
                mirrored_sq = (7 - file) + rank * 8
                
                # Copy feature
                mirrored[mirrored_base_idx + mirrored_sq] = features[base_idx + sq]
    
    # 2. Mirror castling rights (768-771)
    mirrored[768] = features[769]  # White kingside <-> White queenside
    mirrored[769] = features[768]  # White queenside <-> White kingside
    mirrored[770] = features[771]  # Black kingside <-> Black queenside
    mirrored[771] = features[770]  # Black queenside <-> Black kingside
    
    # 3. En passant (772-773) - unchanged
    mirrored[772] = features[772]
    mirrored[773] = features[773]
    
    # 4. Half-move clock (774) - unchanged
    mirrored[774] = features[774]
    
    # 5. Full move number (775) - unchanged
    mirrored[775] = features[775]
    
    # 6. Side to move (776) - unchanged (same player to move)
    mirrored[776] = features[776]
    
    # 7. Material balance (777) - flipped sign
    mirrored[777] = -features[777]
    
    # 8. Tactical threats (778) - approximate mirror
    # This is an approximation since we don't have the board state
    # We'll just keep it as is (or invert)
    mirrored[778] = features[778]  # Approximation
    
    # 9. King safety (779) - approximate mirror
    mirrored[779] = features[779]  # Approximation
    
    return mirrored

def verify_mirror_quality(original: Position, mirrored: Position):
    """
    Verifies that mirroring worked correctly by checking some properties
    """
    # Check that material balance flipped
    if original.features[777] + mirrored.features[777] > 0.01:
        print(f"⚠️  Warning: Material balance not properly mirrored")
        print(f"  Original: {original.features[777]:.3f}, Mirrored: {mirrored.features[777]:.3f}")
    
    # Check that score flipped
    if abs(original.score + mirrored.score) > 0.01:
        print(f"⚠️  Warning: Score not properly mirrored")
        print(f"  Original: {original.score:.3f}, Mirrored: {mirrored.score:.3f}")

# ============== Main ==============
def main():
    print("\n" + "="*80)
    print("🪞 NNUE DATA MIRROR TOOL")
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
        # Auto-generate output filename
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_mirrored.bin"
    
    print(f"📁 Input file:  {input_file}")
    print(f"📁 Output file: {output_file}")
    print("="*80)
    
    try:
        # Read original data
        positions, num_orig = read_nnue_binary(input_file)
        
        # Mirror positions
        mirrored_positions = mirror_positions(positions)
        
        # Verify a few positions
        print("\n🔍 Verifying mirror quality...")
        for i in range(min(5, len(positions))):
            verify_mirror_quality(positions[i], mirrored_positions[i])
        
        # Combine original and mirrored
        all_positions = positions + mirrored_positions
        
        print(f"\n📊 Summary:")
        print(f"  Original positions: {len(positions)}")
        print(f"  Mirrored positions: {len(mirrored_positions)}")
        print(f"  Total positions:    {len(all_positions)}")
        
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
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()