#!/usr/bin/env python3
"""
Filter NNUE training data by evaluation quality.

Options:
  A) Remove positions with the best (highest) evaluations for White
  B) Remove positions with the best (highest) evaluations for Black
  C) Remove positions that are winning for one side (|score| > threshold)
"""

import struct
import numpy as np
import argparse
from pathlib import Path


FEATURE_DIM = 780
HEADER_MAGIC = b"NNUE"
SIDE_TO_MOVE_IDX = 776          # 1.0 = Black to move, 0.0 = White to move


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


def is_black_to_move(features: np.ndarray) -> bool:
    return features[SIDE_TO_MOVE_IDX] > 0.5


def get_white_score(features, score_stm: float) -> float:
    """Convert side-to-move score to White’s perspective."""
    if is_black_to_move(features):
        return -score_stm
    return score_stm


def filter_positions(positions, mode: str, threshold: float = 1.5, top_percent: float = 20.0):
    """
    mode:
      "best_white"  → remove the best (highest) evaluations for White
      "best_black"  → remove the best (highest) evaluations for Black
      "winning"     → remove positions that are winning for one side (|score_white| > threshold)
    """
    if mode == "winning":
        filtered = []
        removed = 0
        for feats, score_stm, result, tactical in positions:
            white_score = get_white_score(feats, score_stm)
            if abs(white_score) > threshold:
                removed += 1
                continue
            filtered.append((feats, score_stm, result, tactical))
        return filtered, removed

    # For best_white / best_black we need to rank positions
    scored = []
    for i, (feats, score_stm, result, tactical) in enumerate(positions):
        white_score = get_white_score(feats, score_stm)
        scored.append((i, white_score, feats, score_stm, result, tactical))

    if mode == "best_white":
        # Highest white_score first
        scored.sort(key=lambda x: x[1], reverse=True)
    elif mode == "best_black":
        # Lowest white_score first (best for Black)
        scored.sort(key=lambda x: x[1])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    n_remove = int(len(scored) * (top_percent / 100.0))
    to_remove = set(item[0] for item in scored[:n_remove])

    filtered = []
    for i, (feats, score_stm, result, tactical) in enumerate(positions):
        if i not in to_remove:
            filtered.append((feats, score_stm, result, tactical))

    return filtered, n_remove


def main():
    parser = argparse.ArgumentParser(
        description="Remove best evaluations or winning positions from NNUE .bin"
    )
    parser.add_argument("input", help="Input .bin file")
    parser.add_argument("-o", "--output", help="Output .bin file")
    parser.add_argument(
        "--mode",
        choices=["best_white", "best_black", "winning"],
        required=True,
        help=(
            "best_white  = remove highest evaluations for White\n"
            "best_black  = remove highest evaluations for Black\n"
            "winning     = remove positions with |eval| > threshold"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Threshold for --mode winning (default: 1.5)",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=20.0,
        help="Percentage of best positions to remove for best_white / best_black (default: 20)",
    )
    parser.add_argument("--stats", action="store_true", help="Only show statistics")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    print(f"Reading {input_path} …")
    positions = read_positions(str(input_path))
    print(f"  Loaded {len(positions):,} positions")

    # Quick stats
    white_scores = [get_white_score(f, s) for f, s, _, _ in positions]
    print(f"  White-score  min / mean / max : "
          f"{min(white_scores):+.2f} / {np.mean(white_scores):+.2f} / {max(white_scores):+.2f}")

    if args.stats:
        return

    filtered, removed = filter_positions(
        positions,
        mode=args.mode,
        threshold=args.threshold,
        top_percent=args.top_percent,
    )

    print(f"  Mode         : {args.mode}")
    if args.mode == "winning":
        print(f"  Threshold    : ±{args.threshold}")
    else:
        print(f"  Top percent  : {args.top_percent}%")
    print(f"  Removed      : {removed:,}")
    print(f"  Kept         : {len(filtered):,}")

    if args.output:
        out_path = Path(args.output)
    else:
        suffix = f"_{args.mode}"
        if args.mode == "winning":
            suffix += f"_t{args.threshold}"
        else:
            suffix += f"_p{int(args.top_percent)}"
        out_path = input_path.with_name(input_path.stem + suffix + ".bin")

    write_positions(str(out_path), filtered)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()

# A) Remove the best 20% evaluations for White
#python filter_evals.py training_data_prod.bin --mode best_white --top-percent 20

# B) Remove the best 20% evaluations for Black
#python filter_evals.py training_data_prod.bin --mode best_black --top-percent 20

# C) Remove all positions that are winning for one side (|eval| > 1.5)
#python filter_evals.py training_data_prod.bin --mode winning --threshold 1.5

# C with a stricter threshold
#python filter_evals.py training_data_prod.bin --mode winning --threshold 2.0

# Just see statistics (no file written)
#python filter_evals.py training_data_prod.bin --mode winning --stats