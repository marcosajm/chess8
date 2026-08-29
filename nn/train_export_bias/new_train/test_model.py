#!/usr/bin/env python3
"""
Test script to verify the trained model works with your C code
"""

import chess
import numpy as np
import torch
import struct
from nn.train_export_bias.old.train_nnue import Config, NNUE, featurize_board

def test_with_position(fen: str):
    """Test model evaluation on a specific position"""
    board = chess.Board(fen)
    features = featurize_board(board)
    
    # Load model
    model = NNUE()
    model.load_state_dict(torch.load(Config.MODEL_FILE))
    model.eval()
    
    with torch.no_grad():
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        score = model(features_tensor).item()
    
    #print(f"Position: {fen}")
    #print(f"Model evaluation: {score:.2f}")
    #print(f"Board:\n{board}")
    
    return score

def compare_with_stockfish(fen: str, stockfish_path: str):
    """Compare model evaluation with Stockfish"""
    import chess.engine
    
    board = chess.Board(fen)
    
    # Get model score
    model_score = test_with_position(fen)
    
    # Get Stockfish score
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    analysis = engine.analyse(board, chess.engine.Limit(depth=10))
    engine.quit()
    
    stockfish_score = analysis['score'].white().score()
    if stockfish_score is None:
        stockfish_score = 0
    
    #print(f"Stockfish evaluation: {stockfish_score/100:.2f}")
    #print(f"Difference: {abs(model_score - stockfish_score/100):.2f}")
    
    return model_score, stockfish_score

if __name__ == "__main__":
    # Test with some positions
    positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6",
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
    ]
    
    #print("Testing trained model...")
    for fen in positions:
        #print("\n" + "="*50)
        test_with_position(fen)