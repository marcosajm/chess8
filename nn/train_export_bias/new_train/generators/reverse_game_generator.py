#!/usr/bin/env python3
"""
Reverse Game Generator - Generate games from checkmate backwards
Creates tactical and endgame positions by working backwards from checkmate
"""

import chess
import chess.engine
import numpy as np
import struct
import random
import time
import os
from typing import List, Tuple, Optional, Set
from collections import deque
import copy
from pathlib import Path

# ============== Configuration ==============
class ReverseConfig:
    # Generation parameters
    MAX_PLY_FROM_MATE = 20  # How many moves back from mate
    NUM_MATE_PATTERNS = 1000  # Number of checkmate positions to generate
    NUM_REVERSE_GAMES = 5000  # Total reverse games to generate
    MAX_UNMAKE_ATTEMPTS = 100  # Max attempts to find a previous position
    
    # Piece restrictions
    MIN_PIECES = 5  # Minimum pieces on board (including kings)
    MAX_PIECES = 24  # Maximum pieces on board
    
    # Output
    OUTPUT_FILE = "reverse_games_data.bin"
    COMBINED_FILE = "combined_training_data.bin"
    
    # Original data file (to combine with)
    ORIGINAL_DATA_FILE = "training_data_enhanced.bin"

# ============== Mate Pattern Database ==============
class MatePatternGenerator:
    """Generate various checkmate patterns"""
    
    @staticmethod
    def get_basic_mates() -> List[chess.Board]:
        """Generate basic checkmate positions"""
        mates = []
        
        # 1. Back rank mate
        try:
            board = chess.Board("6k1/5R2/8/8/8/8/8/6K1 w - - 0 1")
            mates.append(board)
        except:
            pass
        
        # 2. Scholar's mate
        try:
            board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
            mates.append(board)
        except:
            pass
        
        # 3. Fool's mate
        try:
            board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 0 4")
            mates.append(board)
        except:
            pass
        
        # 4. Smothered mate
        try:
            board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
            mates.append(board)
        except:
            pass
        
        # 5. Arabian mate
        try:
            board = chess.Board("6k1/8/8/8/8/8/4R3/5K2 w - - 0 1")
            mates.append(board)
        except:
            pass
        
        # 6. Anastasia's mate
        try:
            board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
            mates.append(board)
        except:
            pass
        
        return mates
    
    @staticmethod
    def generate_random_mate() -> Optional[chess.Board]:
        """Generate a random checkmate position"""
        
        # Start with a basic mate and add/remove pieces
        base_mates = MatePatternGenerator.get_basic_mates()
        if not base_mates:
            return None
        
        board = random.choice(base_mates).copy()
        
        # Randomly add some pieces to create variety
        num_additions = random.randint(0, 8)
        for _ in range(num_additions):
            # Try to add a random piece
            piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
            color = random.choice([chess.WHITE, chess.BLACK])
            piece_type = random.choice(piece_types)
            
            # Find empty squares
            empty_squares = [sq for sq in range(64) if board.piece_at(sq) is None]
            if empty_squares:
                sq = random.choice(empty_squares)
                board.set_piece_at(sq, chess.Piece(piece_type, color))
        
        # Make sure it's a checkmate (or at least check)
        if board.is_checkmate():
            return board
        elif board.is_check():
            # Try to convert to checkmate by adding a piece
            kings = []
            for color in [chess.WHITE, chess.BLACK]:
                king_sq = board.king(color)
                if king_sq:
                    kings.append((color, king_sq))
            
            if len(kings) == 2:
                # Try to add a checking piece
                for _ in range(10):
                    test_board = board.copy()
                    piece_type = random.choice([chess.ROOK, chess.QUEEN, chess.BISHOP])
                    
                    # Find squares that would give check
                    for sq in range(64):
                        if test_board.piece_at(sq) is None:
                            test_board.set_piece_at(sq, chess.Piece(piece_type, not board.turn))
                            if test_board.is_checkmate():
                                return test_board
                            test_board.remove_piece_at(sq)
        
        return None if not board.is_checkmate() else board

# ============== Reverse Game Generator ==============
class ReverseGameGenerator:
    """Generate games by working backwards from checkmate"""
    
    def __init__(self):
        self.mate_generator = MatePatternGenerator()
        self.engine = None
        self.position_cache = set()
        
    def start_engine(self, stockfish_path: str = "/usr/games/stockfish"):
        """Start Stockfish engine for validation"""
        if os.path.exists(stockfish_path):
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                self.engine.configure({"Hash": 64, "Threads": 1})
            except:
                self.engine = None
    
    def stop_engine(self):
        """Stop Stockfish engine"""
        if self.engine:
            self.engine.quit()
            self.engine = None
    
    def _count_pieces(self, board: chess.Board) -> int:
        """Count pieces on the board (compatible with older python-chess)"""
        try:
            # Try new method (python-chess >= 1.0)
            if hasattr(board, 'occupied_count'):
                return board.occupied_count()
            # Try occupied_co
            elif hasattr(board, 'occupied_co'):
                return board.occupied_co.bit_count()
            else:
                # Manual count (fallback)
                count = 0
                for sq in range(64):
                    if board.piece_at(sq):
                        count += 1
                return count
        except:
            # Manual count (fallback)
            count = 0
            for sq in range(64):
                if board.piece_at(sq):
                    count += 1
            return count
    
    def generate_reverse_positions(self, num_games: int = ReverseConfig.NUM_REVERSE_GAMES) -> List[Tuple[np.ndarray, float]]:
        """Generate positions by walking backwards from checkmate"""
        #print(f"🔄 Generating {num_games} reverse-engineered positions...")
        
        all_positions = []
        successful = 0
        start_time = time.time()
        
        # First, collect checkmate positions
        mate_positions = []
        #print("  Generating checkmate patterns...")
        for i in range(ReverseConfig.NUM_MATE_PATTERNS):
            mate = self.mate_generator.generate_random_mate()
            if mate:
                mate_positions.append(mate)
                if len(mate_positions) >= ReverseConfig.NUM_MATE_PATTERNS // 2:
                    break
        
        #print(f"  Generated {len(mate_positions)} checkmate positions")
        
        if not mate_positions:
            #print("  ⚠️ No checkmate positions generated!")
            return []
        
        # Now work backwards from each mate
        games_from_mate = max(1, num_games // max(1, len(mate_positions)))
        
        for mate_idx, mate_board in enumerate(mate_positions):
            if len(all_positions) >= num_games:
                break
                
            # Generate multiple games from this mate
            for game_idx in range(games_from_mate):
                if len(all_positions) >= num_games:
                    break
                
                # Start from mate and go backwards
                positions = self._walk_backwards_from_mate(mate_board)
                
                if positions:
                    all_positions.extend(positions)
                    successful += 1
                    
                    if successful % 100 == 0:
                        elapsed = time.time() - start_time
                        #print(f"  Generated {successful} games ({len(all_positions)} positions) - {elapsed:.1f}s")
        
        #print(f"\n✅ Generated {len(all_positions)} positions from {successful} reverse games")
        return all_positions
    
    def _walk_backwards_from_mate(self, mate_board: chess.Board, max_ply: int = ReverseConfig.MAX_PLY_FROM_MATE) -> List[Tuple[np.ndarray, float]]:
        """Walk backwards from a checkmate position"""
        positions = []
        board = mate_board.copy()
        
        # Store the mate position (score = 1.0 for side that won)
        features = self._featurize_board(board)
        positions.append((features, 1.0))
        
        # Walk backwards
        for ply in range(max_ply):
            # Find a previous position
            prev_board = self._find_previous_position(board)
            if not prev_board:
                break
            
            board = prev_board
            
            # Calculate score (closer to 1.0 for winning side)
            # For reverse games, we assume the side to move is winning
            score = 0.5 + (0.5 * (max_ply - ply) / max_ply)
            
            # Add position
            features = self._featurize_board(board)
            positions.append((features, score))
            
            # Stop if board is too empty
            if self._count_pieces(board) < ReverseConfig.MIN_PIECES:
                break
        
        return positions
    
    def _find_previous_position(self, board: chess.Board) -> Optional[chess.Board]:
        """Find a plausible previous position"""
        
        # Try multiple times to find a valid previous position
        for attempt in range(ReverseConfig.MAX_UNMAKE_ATTEMPTS):
            test_board = board.copy()
            
            # Try to remove a random piece (reverse of a move)
            if self._remove_random_piece(test_board):
                # Validate the position (legal, not too many pieces)
                if self._is_valid_position(test_board):
                    # Optional: Check if position is reachable (simplified)
                    return test_board
        
        return None
    
    def _remove_random_piece(self, board: chess.Board) -> bool:
        """Remove a random piece (simulating reverse of a capture)"""
        
        # Get all pieces except kings
        pieces = []
        for sq in range(64):
            piece = board.piece_at(sq)
            if piece and piece.piece_type != chess.KING:
                pieces.append((sq, piece))
        
        if not pieces:
            return False
        
        # Choose a random piece to remove
        sq, piece = random.choice(pieces)
        
        # Sometimes also move a piece (reverse of a non-capture)
        if random.random() < 0.3:
            # Move a piece to another square (simplified)
            empty_squares = [s for s in range(64) if board.piece_at(s) is None]
            if empty_squares:
                new_sq = random.choice(empty_squares)
                board.set_piece_at(new_sq, piece)
                board.remove_piece_at(sq)
                return True
        
        # Remove the piece (reverse of capture)
        board.remove_piece_at(sq)
        return True
    
    def _is_valid_position(self, board: chess.Board) -> bool:
        """Check if position is valid for training"""
        
        # Check piece count
        piece_count = self._count_pieces(board)
        if piece_count < ReverseConfig.MIN_PIECES or piece_count > ReverseConfig.MAX_PIECES:
            return False
        
        # Must have exactly 2 kings
        try:
            if not board.has_kings(chess.WHITE) or not board.has_kings(chess.BLACK):
                return False
        except:
            # Check kings manually
            has_white_king = False
            has_black_king = False
            for sq in range(64):
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.KING:
                    if piece.color == chess.WHITE:
                        has_white_king = True
                    else:
                        has_black_king = True
            if not has_white_king or not has_black_king:
                return False
        
        # Check for illegal positions (simplified)
        # King not in check (unless it's a check position)
        if board.is_check():
            return False
        
        # Avoid duplicate positions
        fen = board.fen()
        if fen in self.position_cache:
            return False
        
        self.position_cache.add(fen)
        return True
    
    def _featurize_board(self, board: chess.Board) -> np.ndarray:
        """Convert board to features (simplified version)"""
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
        
        # Additional features (simplified)
        feature_idx = 768
        
        # Castling rights
        features[feature_idx] = float(board.has_kingside_castling_rights(chess.WHITE))
        features[feature_idx + 1] = float(board.has_queenside_castling_rights(chess.WHITE))
        features[feature_idx + 2] = float(board.has_kingside_castling_rights(chess.BLACK))
        features[feature_idx + 3] = float(board.has_queenside_castling_rights(chess.BLACK))
        feature_idx += 4
        
        # Side to move
        features[feature_idx] = 1.0 if board.turn == chess.BLACK else 0.0
        
        return features
    
    def save_reverse_data(self, positions: List[Tuple[np.ndarray, float]], filename: str = ReverseConfig.OUTPUT_FILE):
        """Save reverse-generated training data"""
        #print(f"\n💾 Saving reverse data to {filename}...")
        
        if not positions:
            #print("  No positions to save!")
            return
        
        with open(filename, 'wb') as f:
            # Write header
            f.write(struct.pack('4sI', b'NNUE', len(positions)))
            
            for features, score in positions:
                # Write features (780 floats)
                f.write(features.tobytes())
                # Write score (no result or tactical score needed)
                f.write(struct.pack('f', score))
                # Dummy values for compatibility
                f.write(struct.pack('ff', 0.0, 0.0))
        
        file_size = os.path.getsize(filename) / (1024 * 1024)
        #print(f"  Saved {len(positions)} positions ({file_size:.2f} MB)")

# ============== Data Combiner ==============
class DataCombiner:
    """Combine original and reverse-generated data"""
    
    @staticmethod
    def combine_data_files(original_file: str, reverse_file: str, output_file: str):
        """Combine original and reverse-generated data files"""
        #print(f"\n🔗 Combining data files...")
        
        all_positions = []
        all_positions_original = []
        reverse_count = 0
        
        # Load original data
        if os.path.exists(original_file):
            #print(f"  Loading original data from {original_file}")
            try:
                with open(original_file, 'rb') as f:
                    magic, count = struct.unpack('4sI', f.read(8))
                    if magic == b'NNUE':
                        for i in range(count):
                            try:
                                feat_data = f.read(780 * 4)
                                if len(feat_data) < 780 * 4:
                                    break
                                score, result, tactical = struct.unpack('fff', f.read(12))
                                features = np.frombuffer(feat_data, dtype=np.float32)
                                all_positions.append((features, score, result, tactical))
                                all_positions_original.append((features, score, result, tactical))
                            except:
                                break
                        #print(f"  Loaded {len(all_positions)} positions from original data")
            except Exception as e:
                #print(f"  Error loading original data: {e}")
        else:
            #print(f"  Original data file not found: {original_file}")
        
        # Load reverse data
        if os.path.exists(reverse_file):
            #print(f"  Loading reverse data from {reverse_file}")
            try:
                with open(reverse_file, 'rb') as f:
                    magic, count = struct.unpack('4sI', f.read(8))
                    if magic == b'NNUE':
                        for i in range(count):
                            try:
                                feat_data = f.read(780 * 4)
                                if len(feat_data) < 780 * 4:
                                    break
                                # Read score and dummy values
                                score, result, tactical = struct.unpack('fff', f.read(12))
                                features = np.frombuffer(feat_data, dtype=np.float32)
                                all_positions.append((features, score, result, tactical))
                                reverse_count += 1
                            except:
                                break
                        #print(f"  Loaded {reverse_count} reverse positions")
            except Exception as e:
                #print(f"  Error loading reverse data: {e}")
        else:
            #print(f"  Reverse data file not found: {reverse_file}")
        
        if not all_positions:
            #print("  No positions loaded!")
            return
        
        # Save combined data
        #print(f"\n  Saving combined data to {output_file}")
        with open(output_file, 'wb') as f:
            f.write(struct.pack('4sI', b'NNUE', len(all_positions)))
            for features, score, result, tactical in all_positions:
                f.write(features.tobytes())
                f.write(struct.pack('fff', score, result, tactical))
        
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        #print(f"  Saved {len(all_positions)} total positions ({file_size:.2f} MB)")
        #print(f"  Original: {len(all_positions_original)}")
        #print(f"  Reverse: {reverse_count}")

# ============== Main ==============
def main():
    #print("=" * 80)
    #print("🔄 Reverse Game Generator - Endgame Focus")
    #print("   Generating positions backwards from checkmate")
    #print("=" * 80)
    
    # Step 1: Generate reverse positions
    generator = ReverseGameGenerator()
    positions = generator.generate_reverse_positions()
    
    if positions:
        generator.save_reverse_data(positions)
    else:
        #print("\n⚠️ No positions generated!")
        return
    
    # Step 2: Combine with original data
    if os.path.exists(ReverseConfig.ORIGINAL_DATA_FILE) and positions:
        DataCombiner.combine_data_files(
            ReverseConfig.ORIGINAL_DATA_FILE,
            ReverseConfig.OUTPUT_FILE,
            ReverseConfig.COMBINED_FILE
        )
    else:
        #print("\n⚠️  Original data file not found or no reverse data generated")
        #print(f"   Looking for: {ReverseConfig.ORIGINAL_DATA_FILE}")
        #print(f"   Reverse data: {ReverseConfig.OUTPUT_FILE}")
    
    # Step 3: Generate statistics
    if positions:
        #print("\n📊 Statistics:")
        #print(f"  Total positions generated: {len(positions)}")
        
        # Count unique positions
        unique_fens = set()
        for features, _ in positions:
            # Simple hash of features
            hash_val = hash(features.tobytes())
            unique_fens.add(hash_val)
        
        #print(f"  Unique positions: {len(unique_fens)}")
        if len(positions) > 0:
            duplicate_rate = (1 - len(unique_fens)/len(positions)) * 100
            #print(f"  Duplicate rate: {duplicate_rate:.1f}%")
    
    #print("\n✅ Reverse game generation complete!")
    #print(f"  Output file: {ReverseConfig.OUTPUT_FILE}")
    if os.path.exists(ReverseConfig.ORIGINAL_DATA_FILE):
        #print(f"  Combined file: {ReverseConfig.COMBINED_FILE}")
    #print("=" * 80)
    
    # Instructions for integration
    #print("\n📝 To integrate with original training pipeline:")
    #print("  1. Use the combined file for training:")
    #print(f"     Config.DATA_FILE = '{ReverseConfig.COMBINED_FILE}'")
    #print("  2. Or train separately and ensemble the models")
    #print("  3. The reverse data adds tactical awareness in endgames")

if __name__ == "__main__":
    main()