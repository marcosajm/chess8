#!/usr/bin/env python3
"""
Filter NNUE training data (.bin) by side-to-move.
Removes all positions where it is Black to move, or all where it is White to move.
"""

import struct
import numpy as np
import os
import argparse
from pathlib import Path


FEATURE_DIM = 780
HEADER_MAGIC = b"NNUE"


def read_positions(path: str):
    """Return list of (features, score, result, tactical) tuples."""
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
    """Write positions in the exact same binary format."""
    with open(path, "wb") as f:
        f.write(struct.pack("4sI", HEADER_MAGIC, len(positions)))
        for feats, score, result, tactical in positions:
            f.write(feats.astype(np.float32).tobytes())
            f.write(struct.pack("fff", score, result, tactical))


def is_black_to_move(features: np.ndarray) -> bool:
    """
    Side-to-move flag is stored at index 777
    (after 768 piece-square + 4 castling + 2 EP + 2 clocks).
    1.0 = Black to move, 0.0 = White to move.
    """
    return features[777] > 0.5


def filter_positions(positions, remove_black: bool = True, remove_white: bool = False):
    """
    remove_black=True  → drop every position where it is Black to move
    remove_white=True  → drop every position where it is White to move
    """
    filtered = []
    for feats, score, result, tactical in positions:
        black = is_black_to_move(feats)
        if black and remove_black:
            continue
        if not black and remove_white:
            continue
        filtered.append((feats, score, result, tactical))
    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Create a filtered copy of an NNUE .bin dataset by side-to-move"
    )
    parser.add_argument("input", help="Input .bin file")
    parser.add_argument("-o", "--output", help="Output .bin file (default: auto-generated)")
    parser.add_argument(
        "--remove",
        choices=["black", "white", "both"],
        default="black",
        help="Which side-to-move positions to remove (default: black)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    print(f"Reading {input_path} …")
    positions = read_positions(str(input_path))
    print(f"  Loaded {len(positions):,} positions")

    remove_black = args.remove in ("black", "both")
    remove_white = args.remove in ("white", "both")

    filtered = filter_positions(positions, remove_black=remove_black, remove_white=remove_white)
    print(f"  Kept   {len(filtered):,} positions "
          f"(removed {len(positions) - len(filtered):,})")

    if args.output:
        out_path = Path(args.output)
    else:
        suffix = f"_no{args.remove}"
        out_path = input_path.with_name(input_path.stem + suffix + ".bin")

    write_positions(str(out_path), filtered)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote   {out_path}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()