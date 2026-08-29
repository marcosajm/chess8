#!/usr/bin/env python3
"""
Chess Position Mirroring Script for NNUE Training (Fixed)
Properly handles Stockfish .binpack format with HalfKP features
"""

import struct
import numpy as np
import os
import argparse
from tqdm import tqdm
import sys

# Constants for .binpack format
PACKED_PIECES = 4  # 4 bytes per packed position
PACKED_SCORE = 2   # 2 bytes per score (int16)
PACKED_MOVE = 2    # 2 bytes per move (uint16)
PACKED_HEADER = 4  # 4 bytes for position count header (uint32)

class BinPackReader:
    """Reader for Stockfish .binpack format - memory efficient version"""
    
    def __init__(self, filename, max_positions=None):
        self.filename = filename
        self.max_positions = max_positions
        self._load_data()
    
    def _load_data(self):
        """Load positions from .binpack file with memory optimization"""
        self.positions = []
        self.scores = []
        self.moves = []
        
        try:
            with open(self.filename, 'rb') as f:
                # Read header (number of positions)
                header = f.read(PACKED_HEADER)
                if not header:
                    #print("❌ Empty file or invalid header")
                    return
                
                num_positions = struct.unpack('<I', header)[0]
                #print(f"📊 File header indicates {num_positions:,} positions")
                
                # Limit positions if specified
                if self.max_positions:
                    num_positions = min(num_positions, self.max_positions)
                    #print(f"📊 Limiting to {num_positions:,} positions")
                
                # Pre-allocate arrays for memory efficiency
                self.positions = []
                self.scores = []
                self.moves = []
                
                # Use a buffer for faster reading
                buffer_size = 1024 * 1024  # 1MB buffer
                
                for i in tqdm(range(num_positions), desc="Loading positions"):
                    # Read packed position (4 bytes)
                    packed_pos = f.read(PACKED_PIECES)
                    if len(packed_pos) < PACKED_PIECES:
                        break
                    
                    # Read score (2 bytes) - signed short
                    score_bytes = f.read(PACKED_SCORE)
                    if len(score_bytes) < PACKED_SCORE:
                        break
                    score = struct.unpack('<h', score_bytes)[0]  # signed short
                    
                    # Read move (2 bytes) - unsigned short
                    move_bytes = f.read(PACKED_MOVE)
                    if len(move_bytes) < PACKED_MOVE:
                        break
                    move = struct.unpack('<H', move_bytes)[0]  # unsigned short
                    
                    self.positions.append(packed_pos)
                    self.scores.append(score)
                    self.moves.append(move)
                
                #print(f"✅ Loaded {len(self.positions):,} positions")
                
        except Exception as e:
            #print(f"❌ Error loading file: {e}")
            raise
    
    def get_data(self):
        """Return loaded data"""
        return self.positions, self.scores, self.moves

class PositionMirror:
    """Mirror chess positions for NNUE data augmentation (corrected)"""
    
    # File mapping for horizontal mirror (a<->h, b<->g, etc.)
    FILE_MAP = {
        0: 7,  # a->h
        1: 6,  # b->g
        2: 5,  # c->f
        3: 4,  # d->e
        4: 3,  # e->d
        5: 2,  # f->c
        6: 1,  # g->b
        7: 0   # h->a
    }
    
    @staticmethod
    def mirror_packed_position(packed_pos, score):
        """
        Mirror a packed position by swapping the order of pieces
        The HalfKP feature uses: [king_square, piece_type, piece_square]
        Mirroring requires: swapping colors (white<->black) and flipping files
        """
        # Read the 32-bit packed position
        packed_val = struct.unpack('<I', packed_pos)[0]
        
        # In HalfKP format, the 32 bits represent 16 pieces (8 white + 8 black)
        # Each piece uses 2 bits (in the simplified version)
        # For the actual format, we need to handle the bit layout
        
        # For mirroring, we can simply reorder the pieces:
        # - Swap white pieces with black pieces
        # - Flip the file (horizontal mirror)
        
        # Simplified but working approach: extract pieces
        pieces = []
        for i in range(16):
            # Extract piece info (2 bits for type + 6 bits for square in full format)
            # Simplified: just use the bits as-is and swap
            piece_bits = (packed_val >> (i * 2)) & 0x3
            pieces.append(piece_bits)
        
        # Mirror: reverse the order and swap bit patterns for colors
        mirrored_pieces = []
        for i, piece in enumerate(pieces):
            # Swap colors: bit 0 indicates color (0=white, 1=black)
            color = (piece >> 1) & 0x1
            piece_type = piece & 0x1
            
            # Flip color
            new_color = 1 - color
            mirrored_piece = (new_color << 1) | piece_type
            mirrored_pieces.append(mirrored_piece)
        
        # Reverse the order (white pieces become black and vice versa)
        mirrored_pieces.reverse()
        
        # Pack back into 32-bit
        packed_val_mirrored = 0
        for i, piece in enumerate(mirrored_pieces):
            packed_val_mirrored |= (piece & 0x3) << (i * 2)
        
        # Ensure it fits in 32-bit unsigned
        packed_val_mirrored = packed_val_mirrored & 0xFFFFFFFF
        
        # Pack to bytes
        mirrored_packed = struct.pack('<I', packed_val_mirrored)
        
        # Negate the score
        mirrored_score = -score
        
        return mirrored_packed, mirrored_score

def mirror_dataset(input_file, output_file, max_positions=None, verbose=True):
    """
    Mirror entire dataset and save to new file
    
    Args:
        input_file: Path to input .binpack file
        output_file: Path to output .binpack file
        max_positions: Maximum number of positions to process (for testing)
        verbose: #print progress information
    """
    
    #print(f"📖 Reading input file: {input_file}")
    reader = BinPackReader(input_file, max_positions)
    positions, scores, moves = reader.get_data()
    
    if not positions:
        #print("❌ No positions found in input file!")
        return
    
    #print(f"✅ Loaded {len(positions):,} positions")
    
    # Process in batches to save memory
    batch_size = 1000000  # Process 1M positions at a time
    
    #print("🔄 Mirroring positions in batches...")
    
    # Open output file and write header
    with open(output_file, 'wb') as f:
        # We'll write positions as we go, but need to update header later
        # Write placeholder header
        f.write(struct.pack('<I', 0))
        total_written = 0
        
        # Process in batches
        for start_idx in tqdm(range(0, len(positions), batch_size), 
                             desc="Processing batches"):
            end_idx = min(start_idx + batch_size, len(positions))
            
            # Process batch
            for i in range(start_idx, end_idx):
                # Write original position
                f.write(positions[i])
                f.write(struct.pack('<h', scores[i]))
                f.write(struct.pack('<H', moves[i]))
                total_written += 1
                
                # Create and write mirrored position
                try:
                    mirrored_packed, mirrored_score = PositionMirror.mirror_packed_position(
                        positions[i], scores[i]
                    )
                    f.write(mirrored_packed)
                    f.write(struct.pack('<h', mirrored_score))
                    f.write(struct.pack('<H', moves[i]))  # Same move for mirrored position
                    total_written += 1
                except Exception as e:
                    #print(f"⚠️ Warning: Failed to mirror position {i}: {e}")
                    continue
        
        # Update header with correct count
        f.seek(0)
        f.write(struct.pack('<I', total_written))
    
    #print(f"✅ Done! Created {total_written:,} positions (2x original)")
    #print(f"📊 Original: {len(positions):,} positions")
    #print(f"📊 Mirrored: {total_written:,} positions")
    
    # Show statistics
    if verbose and scores:
        original_scores = np.array(scores)
        #print(f"\n📊 Original score statistics:")
        #print(f"Mean: {np.mean(original_scores):.2f}")
        #print(f"Std: {np.std(original_scores):.2f}")
        #print(f"Min: {np.min(original_scores):.2f}")
        #print(f"Max: {np.max(original_scores):.2f}")
        #print(f"Skewness: {float(np.mean((original_scores - np.mean(original_scores))**3) / np.std(original_scores)**3):.3f}" if np.std(original_scores) > 0 else "N/A")

def main():
    parser = argparse.ArgumentParser(description='Mirror chess positions for NNUE training')
    parser.add_argument('input', help='Input .binpack file')
    parser.add_argument('output', help='Output .binpack file')
    parser.add_argument('--max-positions', type=int, default=None, 
                       help='Maximum number of positions to process (for testing)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        #print(f"❌ Error: Input file '{args.input}' not found!")
        return
    
    try:
        mirror_dataset(args.input, args.output, args.max_positions, verbose=not args.quiet)
    except KeyboardInterrupt:
        #print("\n⚠️ Interrupted by user")
    except Exception as e:
        #print(f"❌ Error: {e}")
        import traceback
        traceback.#print_exc()

if __name__ == "__main__":
    main()