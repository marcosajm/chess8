#!/usr/bin/env python3
"""
Filter NNUE .bin data – remove positions where a side is in check.
"""

import struct
import numpy as np
import chess
import argparse
from pathlib import Path


FEATURE_DIM = 780
HEADER_MAGIC = b"NNUE"
SIDE_TO_MOVE_IDX = 776


def read_positions(path: str):
    with open(path, "rb") as f:
        magic, count = struct.unpack("4sI", f.read(8))
        if magic != HEADER_MAGIC:
            raise ValueError(f"Not a valid NNUE data file: {path}")

        positions = []
        for _ in range(count):
            features = np.frombuffer(f.read(FEATURE_DIM * 4), dtype=np.float32).copy()
            score, result, tactical = struct.unpack("fff", f.read(12))
            positions.append((features, score, result, tactical))
    return positions


def write_positions(path: str, positions):
    with open(path, "wb") as f:
        f.write(struct.pack("4sI", HEADER_MAGIC, len(positions)))
        for feats, score, result, tactical in positions:
            f.write(feats.astype(np.float32).tobytes())
            f.write(struct.pack("fff", score, result, tactical))


def features_to_board(features: np.ndarray) -> chess.Board:
    """Reconstruct a chess.Board from the 780-float feature vector."""
    board = chess.Board(None)          # empty board

    piece_map = {
        0: chess.PAWN, 1: chess.KNIGHT, 2: chess.BISHOP,
        3: chess.ROOK, 4: chess.QUEEN, 5: chess.KING
    }

    # 768 piece-square features
    for sq in range(64):
        for color in [0, 1]:           # 0 = White, 1 = Black
            for pt in range(6):
                idx = color * 384 + pt * 64 + sq
                if features[idx] > 0.5:
                    piece = chess.Piece(piece_map[pt], chess.WHITE if color == 0 else chess.BLACK)
                    board.set_piece_at(sq, piece)

    # Castling rights
    board.castling_rights = 0
    if features[768] > 0.5: board.castling_rights |= chess.BB_H1
    if features[769] > 0.5: board.castling_rights |= chess.BB_A1
    if features[770] > 0.5: board.castling_rights |= chess.BB_H8
    if features[771] > 0.5: board.castling_rights |= chess.BB_A8

    # En passant
    ep_file = int(round(features[772] * 7))
    ep_rank = int(round(features[773] * 7))
    if 0 <= ep_file <= 7 and 0 <= ep_rank <= 7:
        board.ep_square = chess.square(ep_file, ep_rank)
    else:
        board.ep_square = None

    # Side to move
    board.turn = chess.BLACK if features[SIDE_TO_MOVE_IDX] > 0.5 else chess.WHITE

    # Clocks (optional, not critical for is_check)
    board.halfmove_clock = int(round(features[774] * 50))
    board.fullmove_number = max(1, int(round(features[775] * 50)))

    return board


def filter_check(positions, remove_white_check=False, remove_black_check=False):
    filtered = []
    removed = 0

    for feats, score, result, tactical in positions:
        board = features_to_board(feats)

        white_in_check = board.is_check() and board.turn == chess.WHITE
        black_in_check = board.is_check() and board.turn == chess.BLACK

        if remove_white_check and white_in_check:
            removed += 1
            continue
        if remove_black_check and black_in_check:
            removed += 1
            continue

        filtered.append((feats, score, result, tactical))

    return filtered, removed


def main():
    parser = argparse.ArgumentParser(description="Remove positions where a side is in check")
    parser.add_argument("input", help="Input .bin file")
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("--remove-white-check", action="store_true",
                        help="Remove positions where White is in check")
    parser.add_argument("--remove-black-check", action="store_true",
                        help="Remove positions where Black is in check")
    parser.add_argument("--remove-any-check", action="store_true",
                        help="Remove any position where the side to move is in check")
    args = parser.parse_args()

    if not (args.remove_white_check or args.remove_black_check or args.remove_any_check):
        print("You must specify at least one of:")
        print("  --remove-white-check")
        print("  --remove-black-check")
        print("  --remove-any-check")
        return

    input_path = Path(args.input)
    print(f"Reading {input_path} …")
    positions = read_positions(str(input_path))
    print(f"  Loaded {len(positions):,} positions")

    remove_white = args.remove_white_check or args.remove_any_check
    remove_black = args.remove_black_check or args.remove_any_check

    filtered, removed = filter_check(
        positions,
        remove_white_check=remove_white,
        remove_black_check=remove_black
    )

    print(f"  Removed      : {removed:,}")
    print(f"  Kept         : {len(filtered):,}")

    if args.output:
        out_path = Path(args.output)
    else:
        tags = []
        if args.remove_white_check or args.remove_any_check:
            tags.append("nowhitecheck")
        if args.remove_black_check or args.remove_any_check:
            tags.append("noblackcheck")
        out_path = input_path.with_name(input_path.stem + "_" + "_".join(tags) + ".bin")

    write_positions(str(out_path), filtered)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()