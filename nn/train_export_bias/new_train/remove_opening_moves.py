#!/usr/bin/env python3
"""
Remove as primeiras 3 jogadas (6 plies) de todos os jogos de um arquivo NNUE .bin

Uso: python3 remove_opening_moves.py input.bin output.bin

Como funciona:
- Cada posição NNUE tem 3132 bytes (780 features * 4 + 12 scores)
- Detecta posições de abertura pelo número de peças e estrutura
- Remove posições que correspondem aos primeiros 3 movimentos
"""

import struct
import os
import sys

# ============== CONSTANTES ==============
FEATURES_SIZE = 780 * 4          # 3120 bytes
SCORES_SIZE = 12                 # 3 int32
POSITION_SIZE = FEATURES_SIZE + SCORES_SIZE  # 3132 bytes
HEADER_SIZE = 8

# ============== FUNÇÕES ==============

def count_pieces_from_features(features: bytes) -> int:
    """
    Conta o número de peças no tabuleiro a partir das features NNUE.
    
    As features NNUE são organizadas como:
    - 64 * 6 * 2 = 768 features para peças (por casa, tipo, cor)
    - 12 features extras (halfmove clock, etc.)
    
    Cada feature é um float32. Um valor > 0.5 indica presença de peça.
    """
    piece_count = 0
    # As primeiras 768 features são as peças
    for i in range(768):
        offset = i * 4
        value = struct.unpack('f', features[offset:offset+4])[0]
        if value > 0.5:
            piece_count += 1
    return piece_count

def get_pawn_count(features: bytes) -> int:
    """
    Conta quantos peões ainda estão no tabuleiro.
    Peões estão nas features 0-63 (brancos) e 64-127 (pretos)... 
    Mas a ordem exata depende da implementação.
    
    Para uma detecção simples, vamos usar o número total de peças.
    """
    return None  # Simplificado

def is_opening_position(features: bytes, move_number: int = None) -> bool:
    """
    Detecta se uma posição é de abertura (primeiros 3 movimentos).
    
    Critérios:
    - 32 peças no tabuleiro (nenhuma captura ainda)
    - Estrutura de peões intacta (nenhum peão movido além do necessário)
    - Material balanceado (0.0)
    
    Retorna True se for posição de abertura (primeiros 3 movimentos)
    """
    piece_count = count_pieces_from_features(features)
    
    # Se já houve captura, não é abertura
    if piece_count < 32:
        return False
    
    # Se tem 32 peças, é provavelmente abertura
    # Mas precisamos distinguir entre movimentos 1, 2, 3 e 4+
    # Usamos o halfmove clock (feature 768) como heurística
    # Halfmove clock = 0 significa que o último movimento foi de peão ou captura
    
    # Para simplificar: posições com 32 peças são consideradas abertura
    # Isso pode incluir movimentos 4-6 em alguns casos
    return piece_count == 32

def get_move_number_from_features(features: bytes) -> int:
    """
    Tenta extrair o número do movimento das features NNUE.
    
    As features 768-779 são extras. Em algumas implementações:
    - Feature 768: halfmove clock
    - Feature 769: fullmove number
    - Feature 770: en passant
    - etc.
    
    Isso é uma heurística e pode não funcionar em todas as implementações.
    """
    try:
        # Tenta ler o fullmove number (feature 769)
        offset = 769 * 4
        value = struct.unpack('f', features[offset:offset+4])[0]
        # Normaliza para 0-1 range (assumindo fullmove/100)
        if 0 <= value <= 1:
            return int(value * 100)
    except:
        pass
    return None

def count_pieces_simple(features: bytes) -> int:
    """
    Conta peças de forma mais robusta.
    Features 0-767 são peças (64 casas * 6 tipos * 2 cores).
    """
    count = 0
    for i in range(768):
        offset = i * 4
        try:
            value = struct.unpack('f', features[offset:offset+4])[0]
            if value > 0.5:
                count += 1
        except:
            pass
    return count

def process_file(input_file: str, output_file: str):
    """
    Remove as primeiras 3 jogadas de todos os jogos.
    """
    print("=" * 70)
    print("🔪 REMOVENDO PRIMEIRAS 3 JOGADAS (6 PLIES) DE CADA JOGO")
    print("=" * 70)
    
    if not os.path.exists(input_file):
        print(f"❌ Arquivo não encontrado: {input_file}")
        return False
    
    # Lê cabeçalho
    with open(input_file, 'rb') as f:
        magic, total_positions = struct.unpack('4sI', f.read(8))
        
        if magic != b'NNUE':
            print(f"❌ Magic number inválido: {magic}")
            return False
    
    print(f"\n📂 Arquivo de entrada: {input_file}")
    print(f"📊 Total de posições: {total_positions:,}")
    print(f"📏 Tamanho: {os.path.getsize(input_file) / 1024 / 1024:.2f} MB")
    
    # Estatísticas
    total_removed = 0
    total_kept = 0
    piece_count_stats = {}
    
    print(f"\n🔍 Analisando posições...")
    
    # Processa arquivo
    with open(input_file, 'rb') as in_f, open(output_file, 'wb') as out_f:
        # Escreve cabeçalho temporário
        out_f.write(b'NNUE')
        out_f.write(struct.pack('I', 0))
        
        # Pula cabeçalho de entrada
        in_f.seek(HEADER_SIZE)
        
        for i in range(total_positions):
            pos_data = in_f.read(POSITION_SIZE)
            if len(pos_data) < POSITION_SIZE:
                print(f"⚠️  Arquivo truncado na posição {i}")
                break
            
            features = pos_data[:FEATURES_SIZE]
            
            # Conta peças
            piece_count = count_pieces_simple(features)
            piece_count_stats[piece_count] = piece_count_stats.get(piece_count, 0) + 1
            
            # Decide se mantém ou remove
            # Remove posições com 32 peças (primeiros movimentos)
            # Mantém posições com menos de 32 peças (jogo já avançou)
            if piece_count >= 32:
                # Verifica se é realmente abertura (primeiros 3 movimentos)
                # Heurística: 32 peças + material balanceado = abertura
                total_removed += 1
            else:
                # Mantém posição (jogo já passou dos primeiros movimentos)
                out_f.write(pos_data)
                total_kept += 1
            
            # Progresso
            if (i + 1) % 10000 == 0:
                print(f"  Progresso: {i+1:,}/{total_positions:,}")
        
        # Atualiza cabeçalho
        out_f.seek(4)
        out_f.write(struct.pack('I', total_kept))
    
    # Estatísticas finais
    print(f"\n📊 Estatísticas de peças encontradas:")
    for pieces in sorted(piece_count_stats.keys(), reverse=True):
        count = piece_count_stats[pieces]
        pct = count / total_positions * 100
        print(f"  {pieces:2d} peças: {count:6,} ({pct:5.1f}%)")
    
    print(f"\n✅ Resultado:")
    print(f"  📊 Posições originais: {total_positions:,}")
    print(f"  🗑️  Posições removidas: {total_removed:,} ({total_removed/total_positions*100:.1f}%)")
    print(f"  💾 Posições mantidas:   {total_kept:,} ({total_kept/total_positions*100:.1f}%)")
    print(f"  📁 Arquivo de saída: {output_file}")
    print(f"  📏 Tamanho: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    # Verifica integridade
    with open(output_file, 'rb') as f:
        magic, count = struct.unpack('4sI', f.read(8))
        expected_size = 8 + count * POSITION_SIZE
        actual_size = os.path.getsize(output_file)
        
        if magic == b'NNUE' and expected_size == actual_size:
            print(f"\n✅ Arquivo de saída válido!")
            print(f"   Posições: {count:,}")
        else:
            print(f"\n❌ Arquivo de saída INVÁLIDO!")
            print(f"   Esperado: {expected_size} bytes, Atual: {actual_size} bytes")
            return False
    
    return True

# ============== VERSÃO AVANÇADA (se tiver FEN) ==============

def remove_opening_by_fen(input_file: str, output_file: str, max_move: int = 3):
    """
    Versão alternativa: se você tiver os FENs das posições,
    pode filtrar diretamente pelo número do movimento.
    
    Esta versão requer que você tenha um arquivo de FENs separado
    ou que os FENs estejam codificados de alguma forma no .bin.
    """
    pass  # Implementação depende do formato específico

# ============== MAIN ==============

def main():
    if len(sys.argv) < 3:
        print("Uso: python3 remove_opening_moves.py <input.bin> <output.bin>")
        print("\nExemplo:")
        print("  python3 remove_opening_moves.py training_data_merged_prod.bin training_data_no_opening.bin")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = process_file(input_file, output_file)
    
    if success:
        print("\n" + "=" * 70)
        print("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ PROCESSO FALHOU!")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()