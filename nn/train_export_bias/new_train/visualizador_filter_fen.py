#!/usr/bin/env python3
"""
Visualizador de Dados NNUE - Versão Adaptada
Exibe posições, detalhes, avaliações e features extras
"""

import struct
import numpy as np
import chess
import json
import csv
from typing import Dict, List, Tuple, Optional

# ============== CONFIGURAÇÃO ==============
CONFIG = {
    'NNUE_INPUT_DIM': 780,
    'DATA_FILE': 'training_data_prod.bin'
}

# ============== LEITURA DOS DADOS ==============

def read_positions(filename: str, max_positions: Optional[int] = None) -> Tuple[List[Dict], int]:
    """
    Lê posições do arquivo binário
    
    Retorna:
        - Lista de dicionários com cada posição
        - Total de posições no arquivo
    """
    positions = []
    
    with open(filename, 'rb') as f:
        # Lê cabeçalho
        magic, total_count = struct.unpack('4sI', f.read(8))
        
        if magic != b'NNUE':
            raise ValueError(f"Magic number inválido: {magic}")
        
        # Determina quantas posições ler
        num_to_read = min(max_positions or total_count, total_count)
        
        for i in range(num_to_read):
            # Lê features (780 floats = 3120 bytes)
            feat_data = f.read(780 * 4)
            if len(feat_data) < 780 * 4:
                break
            
            features = np.frombuffer(feat_data, dtype=np.float32)
            
            # Lê scores (3 floats = 12 bytes)
            score, result, tactical = struct.unpack('fff', f.read(12))
            
            positions.append({
                'index': i,
                'features': features,
                'score': score,
                'result': result,
                'tactical': tactical
            })
    
    return positions, total_count

# ============== DECODIFICAÇÃO DO TABULEIRO ==============

def decode_board_from_features(features: np.ndarray) -> chess.Board:
    """
    Decodifica as 768 features ONE-HOT para um tabuleiro chess.Board
    """
    board = chess.Board.empty()
    
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                   chess.ROOK, chess.QUEEN, chess.KING]
    
    for square in range(64):
        for piece_idx, piece_type in enumerate(piece_types):
            for color_idx, color in enumerate([chess.WHITE, chess.BLACK]):
                idx = color_idx * (6 * 64) + piece_idx * 64 + square
                if features[idx] > 0.5:
                    board.set_piece_at(square, chess.Piece(piece_type, color))
    
    # Seta o lado a jogar (feature 777)
    if features[777] > 0.5:
        board.turn = chess.BLACK
    else:
        board.turn = chess.WHITE
    
    # Seta direitos de roque (features 768-771)
    if features[768] > 0.5:
        board.castling_rights |= chess.BB_A1  # Brancas, roque rei
    if features[769] > 0.5:
        board.castling_rights |= chess.BB_H1  # Brancas, roque dama
    if features[770] > 0.5:
        board.castling_rights |= chess.BB_A8  # Pretas, roque rei
    if features[771] > 0.5:
        board.castling_rights |= chess.BB_H8  # Pretas, roque dama
    
    return board

def get_position_details(board: chess.Board) -> Dict:
    """
    Extrai detalhes da posição
    """
    details = {
        'phase': '',
        'material_balance': 0,
        'piece_count': 0,
        'attackers': 0,
        'legal_moves': 0,
        'is_check': False,
        'is_checkmate': False,
        'is_stalemate': False,
        'is_endgame': False
    }
    
    # Conta peças
    piece_count = 0
    white_material = 0
    black_material = 0
    
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.25,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
    }
    
    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            piece_count += 1
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
    
    details['piece_count'] = piece_count
    details['material_balance'] = white_material - black_material
    details['legal_moves'] = len(list(board.legal_moves))
    details['is_check'] = board.is_check()
    details['is_checkmate'] = board.is_checkmate()
    details['is_stalemate'] = board.is_stalemate()
    details['is_endgame'] = piece_count <= 10  # Poucas peças = final
    
    # Determina fase do jogo
    if piece_count >= 28:
        details['phase'] = 'Abertura'
    elif piece_count >= 16:
        details['phase'] = 'Meio-jogo'
    else:
        details['phase'] = 'Final'
    
    # Conta atacantes (simplificado)
    attackers = 0
    for square in range(64):
        if board.piece_at(square):
            attackers += len(board.attackers(board.turn, square))
    details['attackers'] = attackers
    
    return details

# ============== ANÁLISE DAS FEATURES ==============

def analyze_features(features: np.ndarray) -> Dict:
    """
    Analisa as 780 features e extrai informações úteis
    """
    analysis = {
        'active_features': 0,
        'extra_features': {},
        'piece_counts': {'white': {}, 'black': {}},
        'castling': {'white': {'kingside': False, 'queenside': False},
                    'black': {'kingside': False, 'queenside': False}},
        'en_passant': None,
        'halfmove_clock': 0,
        'fullmove_number': 0,
        'side_to_move': 'white',
        'material_balance': 0,
        'tactical_threat': 0,
        'king_safety': 0
    }
    
    # Conta features ativas (posições com peças)
    analysis['active_features'] = int(np.sum(features[:768] > 0.5))
    
    # Features extras (768-780)
    extra_names = [
        'castling_white_kingside', 'castling_white_queenside',
        'castling_black_kingside', 'castling_black_queenside',
        'en_passant_file', 'en_passant_rank',
        'halfmove_clock', 'fullmove_number',
        'side_to_move', 'material_balance',
        'tactical_threat', 'king_safety'
    ]
    
    for i, name in enumerate(extra_names):
        analysis['extra_features'][name] = features[768 + i]
    
    # Castling
    analysis['castling']['white']['kingside'] = features[768] > 0.5
    analysis['castling']['white']['queenside'] = features[769] > 0.5
    analysis['castling']['black']['kingside'] = features[770] > 0.5
    analysis['castling']['black']['queenside'] = features[771] > 0.5
    
    # En passant
    if features[772] > 0 or features[773] > 0:
        file_idx = int(features[772] * 7)
        rank_idx = int(features[773] * 7)
        analysis['en_passant'] = chess.square(file_idx, rank_idx)
    
    # Outros
    analysis['halfmove_clock'] = features[774] * 50
    analysis['fullmove_number'] = int(features[775] * 50)
    analysis['side_to_move'] = 'black' if features[777] > 0.5 else 'white'
    analysis['material_balance'] = features[778] * 39
    analysis['tactical_threat'] = features[779]
    analysis['king_safety'] = features[780] if len(features) > 780 else 0
    
    return analysis

# ============== VISUALIZAÇÃO PRINCIPAL ==============

def visualize_position(position: Dict, index: int, show_board: bool = True):
    """
    Visualiza uma posição completa com o formato solicitado
    """
    features = position['features']
    score = position['score']
    result = position['result']
    tactical = position['tactical']
    
    # Decodifica tabuleiro
    board = decode_board_from_features(features)
    details = get_position_details(board)
    analysis = analyze_features(features)
    
    # ===== CABEÇALHO =====
    print(f"\n{'='*80}")
    print(f"📍 POSIÇÃO #{index}")
    print(f"{'='*80}")
    
    # ===== POSIÇÃO =====
    print(f"\n{'='*30} POSIÇÃO {'='*30}")
    print(board)
    print(f"\n📊 FEN: {board.fen()}")
    print(f"🎯 Vez: {'Brancas' if board.turn == chess.WHITE else 'Pretas'}")
    print(f"📋 Movimentos legais: {details['legal_moves']}")
    
    # ===== DETALHES =====
    print(f"\n{'='*30} DETALHES {'='*30}")
    print(f"🔹 Fase do jogo: {details['phase']}")
    print(f"🔹 Peças no tabuleiro: {details['piece_count']}")
    print(f"🔹 Material: Brancas {details['material_balance']:.1f} vs Pretas")
    print(f"🔹 Em xeque: {'Sim' if details['is_check'] else 'Não'}")
    print(f"🔹 Xeque-mate: {'Sim' if details['is_checkmate'] else 'Não'}")
    print(f"🔹 Afogamento: {'Sim' if details['is_stalemate'] else 'Não'}")
    print(f"🔹 Final de jogo: {'Sim' if details['is_endgame'] else 'Não'}")
    print(f"🔹 Atacantes estimados: {details['attackers']}")
    
    # ===== AVALIAÇÕES =====
    print(f"\n{'='*30} AVALIAÇÕES {'='*30}")
    print(f"🎯 Score Stockfish:      {score:+.3f} (centipawns: {score*100:.1f})")
    print(f"🎯 Score Tático:         {tactical:+.3f}")
    print(f"🎯 Score Combinado:      {score*0.7 + tactical*0.3:+.3f}")
    print(f"🎯 Resultado final:      {result:.2f} {'(Vitória Brancas)' if result == 1 else '(Vitória Pretas)' if result == 0 else '(Empate)'}")
    
    # Interpretação do Score
    if abs(score) < 0.5:
        interpretation = "⚖️  Posição equilibrada"
    elif score > 0:
        if score < 1.5:
            interpretation = "↗️  Ligeira vantagem das Brancas"
        elif score < 3:
            interpretation = "✅  Vantagem significativa das Brancas"
        else:
            interpretation = "🔥  Vantagem decisiva das Brancas"
    else:
        if score > -1.5:
            interpretation = "↘️  Ligeira vantagem das Pretas"
        elif score > -3:
            interpretation = "✅  Vantagem significativa das Pretas"
        else:
            interpretation = "🔥  Vantagem decisiva das Pretas"
    
    print(f"💡 Interpretação: {interpretation}")
    
    # ===== FEATURES EXTRAS =====
    print(f"\n{'='*30} FEATURES EXTRAS (12) {'='*30}")
    print(f"📋 Ativas: {analysis['active_features']} posições ocupadas")
    print()
    
    # Formatação em tabela
    print("┌────────────────────────┬──────────┬─────────────────────────────┐")
    print("│ Feature                │ Valor    │ Interpretação               │")
    print("├────────────────────────┼──────────┼─────────────────────────────┤")
    
    feature_descriptions = [
        ('Roque rei (brancas)', analysis['extra_features']['castling_white_kingside'], 
         'Disponível' if analysis['extra_features']['castling_white_kingside'] > 0.5 else 'Indisponível'),
        ('Roque dama (brancas)', analysis['extra_features']['castling_white_queenside'],
         'Disponível' if analysis['extra_features']['castling_white_queenside'] > 0.5 else 'Indisponível'),
        ('Roque rei (pretas)', analysis['extra_features']['castling_black_kingside'],
         'Disponível' if analysis['extra_features']['castling_black_kingside'] > 0.5 else 'Indisponível'),
        ('Roque dama (pretas)', analysis['extra_features']['castling_black_queenside'],
         'Disponível' if analysis['extra_features']['castling_black_queenside'] > 0.5 else 'Indisponível'),
        ('En passant file', analysis['extra_features']['en_passant_file'],
         f"{analysis['en_passant'] if analysis['en_passant'] is not None else 'Nenhum'}"),
        ('En passant rank', analysis['extra_features']['en_passant_rank'],
         f"{analysis['en_passant'] if analysis['en_passant'] is not None else 'Nenhum'}"),
        ('Half-move clock', analysis['extra_features']['halfmove_clock'],
         f"{analysis['halfmove_clock']:.0f} movimentos sem captura/peão"),
        ('Full-move number', analysis['extra_features']['fullmove_number'],
         f"{analysis['fullmove_number']:.0f}"),
        ('Lado a jogar', analysis['extra_features']['side_to_move'],
         f"{analysis['side_to_move'].upper()}"),
        ('Balanço material', analysis['extra_features']['material_balance'],
         f"{analysis['material_balance']:+.2f} (normalizado)"),
        ('Ameaça tática', analysis['extra_features']['tactical_threat'],
         f"{analysis['tactical_threat']:+.3f} (quanto maior, mais tático)"),
        ('Segurança rei', analysis['extra_features']['king_safety'],
         f"{analysis['king_safety']:+.3f} (quanto maior, mais seguro)")
    ]
    
    for name, value, desc in feature_descriptions:
        print(f"│ {name:22} │ {value:8.3f} │ {desc:27} │")
    
    print("└────────────────────────┴──────────┴─────────────────────────────┘")
    
    return board

# ============== FUNÇÕES DE EXPORTAÇÃO ==============

def export_to_csv(filename: str, output: str = "positions.csv", max_positions: int = 100):
    """Exporta posições para CSV"""
    positions, total = read_positions(filename, max_positions)
    
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Cabeçalho com todas as features
        header = ['id', 'score', 'result', 'tactical']
        header += [f'f_{i}' for i in range(780)]
        writer.writerow(header)
        
        for pos in positions:
            row = [pos['index'], pos['score'], pos['result'], pos['tactical']]
            row.extend(pos['features'].tolist())
            writer.writerow(row)
    
    print(f"✅ Exportado para {output}")

def export_to_json(filename: str, output: str = "positions.json", max_positions: int = 20):
    """Exporta posições para JSON com metadados"""
    positions, total = read_positions(filename, max_positions)
    
    data = {
        'total': total,
        'exported': len(positions),
        'positions': []
    }
    
    for pos in positions:
        board = decode_board_from_features(pos['features'])
        analysis = analyze_features(pos['features'])
        
        data['positions'].append({
            'id': pos['index'],
            'fen': board.fen(),
            'score': float(pos['score']),
            'result': float(pos['result']),
            'tactical': float(pos['tactical']),
            'phase': get_position_details(board)['phase'],
            'legal_moves': get_position_details(board)['legal_moves'],
            'extra_features': analysis['extra_features']
        })
    
    with open(output, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Exportado para {output}")

# ============== FUNÇÃO PRINCIPAL ==============

def main():
    """Função principal com menu interativo"""
    
    print("="*80)
    print("🔍 VISUALIZADOR DE DADOS NNUE")
    print("="*80)
    
    # Pede arquivo
    filename = input(f"\n📁 Arquivo de dados [{CONFIG['DATA_FILE']}]: ").strip()
    if not filename:
        filename = CONFIG['DATA_FILE']
    
    try:
        # Carrega dados
        positions, total = read_positions(filename, max_positions=100)
        print(f"\n✅ Carregado {len(positions)} posições de {total}")
        
        while True:
            print("\n" + "="*80)
            print("📋 MENU:")
            print("  1. Ver próxima posição")
            print("  2. Ver posição específica")
            print("  3. Ver posições aleatórias")
            print("  4. Exportar para CSV")
            print("  5. Exportar para JSON")
            print("  6. Estatísticas do dataset")
            print("  7. Sair")
            
            option = input("\nEscolha uma opção: ").strip()
            
            if option == '1':
                # Próxima posição
                if not hasattr(main, 'current_idx'):
                    main.current_idx = 0
                else:
                    main.current_idx += 1
                
                if main.current_idx >= len(positions):
                    main.current_idx = 0
                    print("\n🔄 Voltando ao início")
                
                visualize_position(positions[main.current_idx], main.current_idx)
                input("\nPressione Enter para continuar...")
                
            elif option == '2':
                # Posição específica
                idx = int(input(f"\nÍndice (0-{len(positions)-1}): "))
                if 0 <= idx < len(positions):
                    visualize_position(positions[idx], idx)
                    main.current_idx = idx
                else:
                    print("❌ Índice inválido!")
                input("\nPressione Enter para continuar...")
                
            elif option == '3':
                # Posições aleatórias
                import random
                num = int(input("\nQuantas posições aleatórias? "))
                for _ in range(min(num, 10)):
                    idx = random.randint(0, len(positions)-1)
                    visualize_position(positions[idx], idx)
                    print("\n" + "-"*80)
                input("\nPressione Enter para continuar...")
                
            elif option == '4':
                # Exportar CSV
                max_pos = int(input("\nMáximo de posições para exportar: "))
                export_to_csv(filename, max_positions=max_pos)
                input("\nPressione Enter para continuar...")
                
            elif option == '5':
                # Exportar JSON
                max_pos = int(input("\nMáximo de posições para exportar: "))
                export_to_json(filename, max_positions=max_pos)
                input("\nPressione Enter para continuar...")
                
            elif option == '6':
                # Estatísticas
                print_statistics(positions)
                input("\nPressione Enter para continuar...")
                
            elif option == '7':
                print("\n👋 Saindo...")
                break
            else:
                print("❌ Opção inválida!")
                
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {filename}")
    except Exception as e:
        print(f"❌ Erro: {e}")

def print_statistics(positions: List[Dict]):
    """Imprime estatísticas do dataset"""
    print("\n" + "="*80)
    print("📊 ESTATÍSTICAS DO DATASET")
    print("="*80)
    
    scores = [p['score'] for p in positions]
    results = [p['result'] for p in positions]
    tacticals = [p['tactical'] for p in positions]
    
    print(f"\n📈 Estatísticas dos Scores:")
    print(f"  Média:    {np.mean(scores):+.3f}")
    print(f"  Mediana:  {np.median(scores):+.3f}")
    print(f"  Std Dev:  {np.std(scores):.3f}")
    print(f"  Mínimo:   {np.min(scores):+.3f}")
    print(f"  Máximo:   {np.max(scores):+.3f}")
    
    print(f"\n📈 Estatísticas dos Resultados:")
    wins = sum(1 for r in results if r == 1.0)
    draws = sum(1 for r in results if r == 0.5)
    losses = sum(1 for r in results if r == 0.0)
    
    print(f"  Vitórias Brancas: {wins} ({wins/len(results)*100:.1f}%)")
    print(f"  Empates:          {draws} ({draws/len(results)*100:.1f}%)")
    print(f"  Vitórias Pretas:  {losses} ({losses/len(results)*100:.1f}%)")
    
    print(f"\n📈 Estatísticas Táticas:")
    print(f"  Média:    {np.mean(tacticals):+.3f}")
    print(f"  Std Dev:  {np.std(tacticals):.3f}")
    print(f"  Máximo:   {np.max(tacticals):+.3f}")

if __name__ == "__main__":
    main()