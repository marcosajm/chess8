#!/usr/bin/env python3
"""
Remove primeiras 3 jogadas usando FENs (mais preciso)
Requer que você tenha um arquivo de FENs correspondente
"""

import struct
import os
import sys

FEATURES_SIZE = 780 * 4
POSITION_SIZE = FEATURES_SIZE + 12

def get_move_number_from_fen(fen: str) -> int:
    """
    Extrai o número do movimento de um FEN.
    
    FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
    O último campo é o fullmove number.
    """
    parts = fen.strip().split()
    if len(parts) >= 6:
        try:
            return int(parts[5])
        except ValueError:
            return 0
    return 0

def get_halfmove_from_fen(fen: str) -> int:
    """Extrai o halfmove clock do FEN"""
    parts = fen.strip().split()
    if len(parts) >= 5:
        try:
            return int(parts[4])
        except ValueError:
            return 0
    return 0

def should_remove_position(fen: str, max_move: int = 3) -> bool:
    """
    Decide se a posição deve ser removida.
    
    Remove se:
    - Fullmove number <= max_move
    - OU se o halfmove clock indica que é muito cedo
    """
    move_number = get_move_number_from_fen(fen)
    return move_number <= max_move

def process_with_fens(bin_file: str, fen_file: str, output_file: str, max_move: int = 3):
    """
    Remove posições baseado em FENs correspondentes.
    
    Args:
        bin_file: Arquivo .bin NNUE
        fen_file: Arquivo de texto com um FEN por linha
        output_file: Arquivo .bin de saída
        max_move: Número máximo de movimentos a remover (padrão 3)
    """
    print("=" * 70)
    print(f"🔪 REMOVENDO PRIMEIROS {max_move} MOVIMENTOS (usando FENs)")
    print("=" * 70)
    
    # Lê FENs
    with open(fen_file, 'r') as f:
        fens = [line.strip() for line in f if line.strip()]
    
    print(f"\n📂 FENs carregados: {len(fens):,}")
    
    # Lê cabeçalho do .bin
    with open(bin_file, 'rb') as f:
        magic, total = struct.unpack('4sI', f.read(8))
        if magic != b'NNUE':
            print(f"❌ Magic inválido: {magic}")
            return False
    
    print(f"📊 Posições no .bin: {total:,}")
    
    if len(fens) != total:
        print(f"⚠️  AVISO: Número de FENs ({len(fens)}) != posições no .bin ({total})")
        print(f"   Usando o menor valor: {min(len(fens), total)}")
    
    n = min(len(fens), total)
    
    # Processa
    total_removed = 0
    total_kept = 0
    
    with open(bin_file, 'rb') as in_f, open(output_file, 'wb') as out_f:
        # Cabeçalho temporário
        out_f.write(b'NNUE')
        out_f.write(struct.pack('I', 0))
        
        in_f.seek(8)
        
        for i in range(n):
            pos_data = in_f.read(POSITION_SIZE)
            if len(pos_data) < POSITION_SIZE:
                break
            
            fen = fens[i]
            
            if should_remove_position(fen, max_move):
                total_removed += 1
            else:
                out_f.write(pos_data)
                total_kept += 1
            
            if (i + 1) % 10000 == 0:
                print(f"  Progresso: {i+1:,}/{n:,}")
        
        # Atualiza cabeçalho
        out_f.seek(4)
        out_f.write(struct.pack('I', total_kept))
    
    print(f"\n✅ Resultado:")
    print(f"  🗑️  Removidas: {total_removed:,}")
    print(f"  💾 Mantidas:  {total_kept:,}")
    print(f"  📁 Saída: {output_file}")
    
    return True

def main():
    if len(sys.argv) < 4:
        print("Uso: python3 remove_opening_fen.py <input.bin> <fens.txt> <output.bin> [max_move]")
        print("\nExemplo:")
        print("  python3 remove_opening_fen.py data.bin fens.txt data_no_opening.bin 3")
        sys.exit(1)
    
    bin_file = sys.argv[1]
    fen_file = sys.argv[2]
    output_file = sys.argv[3]
    max_move = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    
    success = process_with_fens(bin_file, fen_file, output_file, max_move)
    
    if success:
        print("\n✅ Concluído!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()