#!/usr/bin/env python3
"""
Ferramenta para Combinar Múltiplos Arquivos NNUE .bin com Remoção de Duplicatas
e Filtragem por Avaliação (Stockfish)
Uso: python3 merge2_nnue.py
"""

import struct
import numpy as np
import os
import glob
from pathlib import Path
from typing import List, Tuple, Set, Optional
import shutil
import time
import hashlib
import math

# ============== CONFIGURAÇÃO ==============
class MergeConfig:
    # Padrão para encontrar arquivos
    INPUT_PATTERN = "training_data*_prod.bin"  # Pode ajustar
    # ou: INPUT_FILES = ["training_data1_prod.bin", "training_data2_prod.bin", ...]
    
    OUTPUT_FILE = "training_data_merged_prod.bin"
    
    # Cria backup antes de mesclar?
    CREATE_BACKUP = True
    
    # Remove arquivos originais após mesclar?
    DELETE_ORIGINALS = False
    
    # Verifica integridade dos arquivos?
    VERIFY_INTEGRITY = True
    
    # Remove posições duplicadas?
    REMOVE_DUPLICATES = True
    
    # Método para detectar duplicatas: 'hash' ou 'exact'
    # 'hash' = mais rápido, menor precisão (usa hash MD5)
    # 'exact' = mais lento, 100% preciso (compara bytes)
    DEDUP_METHOD = 'hash'  # 'hash' ou 'exact'
    
    # Filtrar por avaliação Stockfish?
    ENABLE_EVAL_FILTER = False
    
    # Remover avaliações para qual lado?
    # 'black' = remove melhores posições das Pretas (Stockfish negativo)
    # 'white' = remove melhores posições das Brancas (Stockfish positivo)
    # 'both' = remove ambas
    FILTER_SIDE = 'black'
    
    # Percentual a remover (0-100)
    FILTER_PERCENT = 10  # Remove 10% melhores ou piores
    
    # Remover melhores ou piores?
    # 'best' = remove posições com maior vantagem (melhores para o lado)
    # 'worst' = remove posições com menor vantagem (piores para o lado)
    FILTER_MODE = 'best'
    
    # Score máximo permitido (em centipawns) - alternativa ao percentual
    # Se definido, remove posições com score acima/abaixo deste limite
    MAX_SCORE_LIMIT = None  # Ex: 500 para remover > 5.00

# ============== FUNÇÕES DE LEITURA/ESCRITA ==============

def read_position_score(data: bytes) -> int:
    """
    Lê o score Stockfish de uma posição NNUE
    
    O formato NNUE armazena:
    - 780 floats (features) = 780 * 4 = 3120 bytes
    - 3 scores (int32) = 12 bytes
    - O segundo score (offset 3128) é o Stockfish score em centipawns
    """
    if len(data) >= 3132:  # 780*4 + 12
        # O score está nos bytes 3128 a 3131 (segundo int32)
        score = struct.unpack('i', data[3128:3132])[0]
        return score
    return 0

def get_position_eval(data: bytes) -> int:
    """
    Retorna a avaliação Stockfish da posição em centipawns
    """
    return read_position_score(data)

def parse_stockfish_score(score: int) -> float:
    """
    Converte score em centipawns para valor em pawns
    """
    return score / 100.0

# ============== FUNÇÕES DE FILTRAGEM ==============

def should_remove_position(score: int, 
                          filter_side: str = 'black',
                          filter_mode: str = 'best',
                          filter_percent: float = 10,
                          score_limit: Optional[int] = None,
                          all_scores: Optional[List[int]] = None) -> bool:
    """
    Decide se uma posição deve ser removida baseado em sua avaliação
    
    Args:
        score: Score Stockfish da posição (centipawns)
        filter_side: 'black', 'white', ou 'both'
        filter_mode: 'best' (remover melhores) ou 'worst' (remover piores)
        filter_percent: Percentual a remover (0-100)
        score_limit: Limite absoluto de score (sobrescreve percentual)
        all_scores: Lista de todos os scores para calcular percentis
    
    Returns:
        True se a posição deve ser removida, False caso contrário
    """
    if score is None:
        return False
    
    # Se for usar limite absoluto
    if score_limit is not None:
        if filter_side == 'black':
            # Pretas: score negativo é bom para Pretas
            if filter_mode == 'best':
                # Remove posições muito boas para Pretas (score muito negativo)
                return score <= -abs(score_limit)
            else:
                # Remove posições ruins para Pretas (score muito positivo)
                return score >= abs(score_limit)
        elif filter_side == 'white':
            # Brancas: score positivo é bom para Brancas
            if filter_mode == 'best':
                # Remove posições muito boas para Brancas (score muito positivo)
                return score >= abs(score_limit)
            else:
                # Remove posições ruins para Brancas (score muito negativo)
                return score <= -abs(score_limit)
        else:  # both
            if filter_mode == 'best':
                # Remove posições muito boas para qualquer lado
                return abs(score) >= abs(score_limit)
            else:
                # Remove posições muito ruins para qualquer lado
                return abs(score) <= abs(score_limit)
    
    # Se não tem lista de scores, não pode usar percentual
    if all_scores is None:
        return False
    
    # Método baseado em percentual
    sorted_scores = sorted(all_scores)
    n = len(sorted_scores)
    cutoff_idx = int(n * (filter_percent / 100.0))
    
    if filter_side == 'black':
        # Para Pretas: scores negativos são melhores
        if filter_mode == 'best':
            # Remover os melhores (scores mais negativos)
            threshold = sorted_scores[cutoff_idx] if cutoff_idx < n else sorted_scores[-1]
            return score <= threshold
        else:
            # Remover os piores (scores mais positivos)
            threshold = sorted_scores[n - cutoff_idx - 1] if cutoff_idx < n else sorted_scores[0]
            return score >= threshold
    
    elif filter_side == 'white':
        # Para Brancas: scores positivos são melhores
        if filter_mode == 'best':
            # Remover os melhores (scores mais positivos)
            threshold = sorted_scores[n - cutoff_idx - 1] if cutoff_idx < n else sorted_scores[0]
            return score >= threshold
        else:
            # Remover os piores (scores mais negativos)
            threshold = sorted_scores[cutoff_idx] if cutoff_idx < n else sorted_scores[-1]
            return score <= threshold
    
    else:  # both
        # Remover baseado no valor absoluto
        abs_scores = sorted([abs(s) for s in all_scores])
        threshold = abs_scores[cutoff_idx] if cutoff_idx < n else abs_scores[-1]
        
        if filter_mode == 'best':
            # Remover posições com maiores vantagens (absoluto)
            return abs(score) >= threshold
        else:
            # Remover posições com menores vantagens (absoluto)
            return abs(score) <= threshold

def get_eval_stats(scores: List[int]) -> dict:
    """
    Retorna estatísticas das avaliações
    """
    if not scores:
        return {}
    
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    
    return {
        'min': min(scores),
        'max': max(scores),
        'mean': sum(scores) / n,
        'median': sorted_scores[n // 2],
        'std': math.sqrt(sum((s - sum(scores)/n) ** 2 for s in scores) / n),
        'p10': sorted_scores[int(n * 0.1)],
        'p25': sorted_scores[int(n * 0.25)],
        'p50': sorted_scores[int(n * 0.5)],
        'p75': sorted_scores[int(n * 0.75)],
        'p90': sorted_scores[int(n * 0.9)],
        'count': n
    }

def print_eval_stats(stats: dict):
    """
    Imprime estatísticas das avaliações de forma formatada
    """
    if not stats:
        print("  Sem dados de avaliação")
        return
    
    print("\n  📊 Estatísticas de Avaliação Stockfish:")
    print(f"    Mínimo: {stats['min']/100:.2f} (pawns)")
    print(f"    Máximo: {stats['max']/100:.2f} (pawns)")
    print(f"    Média:  {stats['mean']/100:.2f} (pawns)")
    print(f"    Mediana:{stats['median']/100:.2f} (pawns)")
    print(f"    Desvio: {stats['std']/100:.2f} (pawns)")
    print(f"    P10:    {stats['p10']/100:.2f} (pawns)  (10% piores para Brancas)")
    print(f"    P25:    {stats['p25']/100:.2f} (pawns)  (25% piores para Brancas)")
    print(f"    P50:    {stats['p50']/100:.2f} (pawns)  (50% - equilibrado)")
    print(f"    P75:    {stats['p75']/100:.2f} (pawns)  (25% melhores para Brancas)")
    print(f"    P90:    {stats['p90']/100:.2f} (pawns)  (10% melhores para Brancas)")

# ============== FUNÇÕES PRINCIPAIS ==============

def read_file_info(filename: str) -> Tuple[int, int, bool]:
    """
    Lê informações de um arquivo NNUE sem carregar todos os dados
    
    Retorna:
        - Número de posições no arquivo
        - Tamanho do arquivo em bytes
        - Se é válido
    """
    try:
        with open(filename, 'rb') as f:
            # Lê cabeçalho
            magic, count = struct.unpack('4sI', f.read(8))
            
            if magic != b'NNUE':
                print(f"  ⚠️  Magic number inválido: {filename}")
                return 0, 0, False
            
            # Verifica se o arquivo tem o tamanho correto
            expected_size = 8 + count * (780 * 4 + 12)  # header + cada posição
            actual_size = os.path.getsize(filename)
            
            if actual_size != expected_size:
                print(f"  ⚠️  Tamanho incorreto: {filename}")
                print(f"     Esperado: {expected_size}, Atual: {actual_size}")
                return count, actual_size, False
            
            return count, actual_size, True
            
    except Exception as e:
        print(f"  ❌ Erro ao ler {filename}: {e}")
        return 0, 0, False

def get_position_hash(data: bytes) -> str:
    """
    Calcula hash MD5 de uma posição (features + scores)
    """
    return hashlib.md5(data).hexdigest()

def get_position_key_exact(data: bytes) -> bytes:
    """
    Retorna os dados exatos da posição como chave
    """
    return data

def collect_scores_from_files(files: List[str]) -> List[int]:
    """
    Coleta todos os scores Stockfish de uma lista de arquivos
    """
    all_scores = []
    pos_size = 780 * 4 + 12
    
    print("\n📊 Coletando avaliações dos arquivos...")
    
    for file in files:
        try:
            count, size, valid = read_file_info(file)
            if not valid or count == 0:
                continue
            
            print(f"  Lendo {os.path.basename(file)} ({count:,} posições)")
            
            with open(file, 'rb') as f:
                f.seek(8)  # Pula cabeçalho
                
                for i in range(count):
                    pos_data = f.read(pos_size)
                    if len(pos_data) < pos_size:
                        break
                    
                    score = get_position_eval(pos_data)
                    all_scores.append(score)
                    
                    if (i + 1) % 50000 == 0:
                        print(f"    Progresso: {i + 1:,}/{count:,}")
            
        except Exception as e:
            print(f"  ❌ Erro ao ler {file}: {e}")
    
    return all_scores

def merge_files(files: List[str], output_file: str, 
                verify: bool = True, backup: bool = True,
                remove_duplicates: bool = True,
                dedup_method: str = 'hash',
                enable_eval_filter: bool = False,
                filter_side: str = 'black',
                filter_mode: str = 'best',
                filter_percent: float = 10,
                score_limit: Optional[int] = None) -> bool:
    """
    Mescla múltiplos arquivos NNUE em um único arquivo, removendo duplicatas
    e/ou filtrando por avaliação
    
    Args:
        files: Lista de caminhos dos arquivos
        output_file: Caminho do arquivo de saída
        verify: Verificar integridade dos arquivos
        backup: Criar backup do arquivo de saída se existir
        remove_duplicates: Remover posições duplicadas
        dedup_method: Método de deduplicação ('hash' ou 'exact')
        enable_eval_filter: Habilitar filtro por avaliação
        filter_side: 'black', 'white', ou 'both'
        filter_mode: 'best' ou 'worst'
        filter_percent: Percentual a remover (0-100)
        score_limit: Limite absoluto de score
    """
    
    print("\n" + "="*80)
    print("🔗 MERGE DE ARQUIVOS NNUE COM DEDUPLICAÇÃO E FILTRO DE AVALIAÇÃO")
    print("="*80)
    
    # Filtra apenas arquivos existentes
    files = [f for f in files if os.path.exists(f)]
    
    if not files:
        print("❌ Nenhum arquivo encontrado!")
        return False
    
    print(f"\n📂 Encontrados {len(files)} arquivos:")
    
    # Lê informações de cada arquivo
    file_info = []
    total_positions = 0
    
    for i, file in enumerate(files):
        count, size, valid = read_file_info(file)
        
        if not valid and verify:
            print(f"  ❌ Arquivo inválido: {file}")
            return False
        
        if count == 0:
            print(f"  ⚠️  Arquivo vazio: {file}")
            continue
        
        file_info.append({
            'file': file,
            'count': count,
            'size': size,
            'valid': valid
        })
        
        total_positions += count
        print(f"  {i+1:2d}. {os.path.basename(file):30s} - {count:6,} posições ({size/1024/1024:.2f} MB)")
    
    if not file_info:
        print("❌ Nenhum arquivo válido encontrado!")
        return False
    
    print(f"\n📊 Total de posições (bruto): {total_positions:,}")
    
    # Coleta scores para filtragem se necessário
    all_scores = None
    eval_stats = None
    
    if enable_eval_filter and filter_percent > 0:
        print(f"\n🔍 Coletando avaliações para filtrar {filter_percent}% dos {filter_side}")
        all_scores = collect_scores_from_files(files)
        eval_stats = get_eval_stats(all_scores)
        print_eval_stats(eval_stats)
        
        # Mostra quantas posições serão removidas
        remove_count = int(len(all_scores) * (filter_percent / 100.0))
        print(f"\n  📊 Serão removidas ~{remove_count:,} posições (~{filter_percent}%)")
    
    if remove_duplicates:
        print(f"🔍 Método de deduplicação: {dedup_method.upper()}")
    
    # Verifica se o arquivo de saída existe
    if os.path.exists(output_file):
        if backup:
            backup_file = output_file + ".backup"
            print(f"\n💾 Criando backup: {backup_file}")
            shutil.copy2(output_file, backup_file)
        
        response = input(f"\n⚠️  Arquivo {output_file} já existe. Sobrescrever? (s/N): ")
        if response.lower() != 's':
            print("❌ Operação cancelada!")
            return False
    
    # Mescla os arquivos
    print(f"\n🚀 Mesclando {len(file_info)} arquivos...")
    start_time = time.time()
    
    # Conjunto para armazenar posições únicas
    seen_positions = set()
    duplicate_count = 0
    filtered_count = 0
    pos_size = 780 * 4 + 12  # Tamanho de cada posição em bytes
    
    try:
        with open(output_file, 'wb') as out_f:
            # Escreve cabeçalho inicial (será atualizado depois)
            out_f.write(b'NNUE')
            out_f.write(struct.pack('I', 0))  # Temporário
            
            positions_written = 0
            
            # Para cada arquivo
            for info in file_info:
                print(f"  📖 Processando: {os.path.basename(info['file'])}")
                file_duplicates = 0
                file_filtered = 0
                
                with open(info['file'], 'rb') as in_f:
                    # Pula cabeçalho do arquivo de entrada
                    in_f.seek(8)
                    
                    # Lê e processa cada posição
                    for pos_idx in range(info['count']):
                        # Lê posição completa (features + scores)
                        pos_data = in_f.read(pos_size)
                        
                        if len(pos_data) < pos_size:
                            print(f"    ⚠️  Erro ao ler posição {pos_idx}")
                            break
                        
                        # Verifica se deve ser removida por avaliação
                        should_filter = False
                        if enable_eval_filter and all_scores is not None and filter_percent > 0:
                            score = get_position_eval(pos_data)
                            should_filter = should_remove_position(
                                score, 
                                filter_side, 
                                filter_mode, 
                                filter_percent, 
                                score_limit,
                                all_scores
                            )
                            if should_filter:
                                file_filtered += 1
                                filtered_count += 1
                        
                        # Verifica se é duplicata
                        is_duplicate = False
                        
                        if not should_filter and remove_duplicates:
                            if dedup_method == 'hash':
                                pos_key = get_position_hash(pos_data)
                            else:  # exact
                                pos_key = pos_data
                            
                            if pos_key in seen_positions:
                                is_duplicate = True
                                file_duplicates += 1
                                duplicate_count += 1
                            else:
                                seen_positions.add(pos_key)
                        
                        # Escreve apenas se não for filtrada nem duplicata
                        if not should_filter and not is_duplicate:
                            out_f.write(pos_data)
                            positions_written += 1
                        
                        # Progresso
                        if (pos_idx + 1) % 10000 == 0:
                            print(f"    Progresso: {pos_idx + 1:,}/{info['count']:,} posições")
                    
                    print(f"    ✅ {info['count']:,} posições lidas, {file_duplicates:,} duplicatas, {file_filtered:,} filtradas")
            
            # Volta para atualizar o cabeçalho com o número correto
            out_f.seek(4)
            out_f.write(struct.pack('I', positions_written))
        
        elapsed = time.time() - start_time
        file_size = os.path.getsize(output_file) / 1024 / 1024
        
        print(f"\n✅ Merge concluído com sucesso!")
        print(f"  📁 Arquivo: {output_file}")
        print(f"  📊 Posições únicas: {positions_written:,}")
        
        if remove_duplicates:
            print(f"  🗑️  Duplicatas removidas: {duplicate_count:,}")
        
        if enable_eval_filter and filter_percent > 0:
            print(f"  🎯 Posições filtradas por avaliação: {filtered_count:,}")
            reduction = (1 - positions_written / max(total_positions, 1)) * 100
            print(f"  📊 Redução total: {reduction:.1f}%")
        
        print(f"  💾 Tamanho: {file_size:.2f} MB")
        print(f"  ⏱️  Tempo: {elapsed:.2f} segundos")
        
        # Verifica o arquivo gerado
        if verify:
            print("\n🔍 Verificando integridade do arquivo final...")
            count, size, valid = read_file_info(output_file)
            
            if valid and count == positions_written:
                print(f"  ✅ Arquivo final válido!")
                print(f"  📊 Total de posições: {count:,}")
                print(f"  📏 Tamanho: {size/1024/1024:.2f} MB")
                
                # Coleta estatísticas do arquivo final
                if positions_written > 0:
                    final_scores = []
                    with open(output_file, 'rb') as f:
                        f.seek(8)
                        for i in range(min(positions_written, 10000)):  # Amostra
                            pos_data = f.read(pos_size)
                            if len(pos_data) >= pos_size:
                                final_scores.append(get_position_eval(pos_data))
                    
                    if final_scores:
                        final_stats = get_eval_stats(final_scores)
                        print("\n  📊 Estatísticas (amostra de 10,000 posições):")
                        print(f"    Média: {final_stats['mean']/100:.2f} pawns")
                        print(f"    Mediana: {final_stats['median']/100:.2f} pawns")
                        print(f"    Min/Max: {final_stats['min']/100:.2f} / {final_stats['max']/100:.2f} pawns")
                        
                        # Mostra distribuição
                        extreme_black = sum(1 for s in final_scores if s < -500)
                        extreme_white = sum(1 for s in final_scores if s > 500)
                        balanced = len(final_scores) - extreme_black - extreme_white
                        
                        print(f"    Distribuição:")
                        print(f"      🏆 Brancas ganhando (>5.00): {extreme_white/len(final_scores)*100:.1f}%")
                        print(f"      ⚖️  Equilibradas (±5.00): {balanced/len(final_scores)*100:.1f}%")
                        print(f"      🏆 Pretas ganhando (<-5.00): {extreme_black/len(final_scores)*100:.1f}%")
            else:
                print(f"  ❌ Arquivo final inválido!")
                return False
        
        # Pergunta se deseja deletar os originais
        if MergeConfig.DELETE_ORIGINALS:
            response = input(f"\n🗑️  Deletar arquivos originais? (s/N): ")
            if response.lower() == 's':
                for info in file_info:
                    os.remove(info['file'])
                    print(f"  🗑️  Deletado: {info['file']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erro durante o merge: {e}")
        return False

# ============== FUNÇÕES ADICIONAIS ==============

def analyze_duplicates(filename: str):
    """
    Analisa quantas duplicatas existem em um arquivo
    """
    if not os.path.exists(filename):
        print(f"❌ Arquivo não encontrado: {filename}")
        return
    
    print(f"\n🔍 Analisando duplicatas em: {filename}")
    
    try:
        with open(filename, 'rb') as f:
            # Lê cabeçalho
            magic, total = struct.unpack('4sI', f.read(8))
            
            if magic != b'NNUE':
                print("❌ Arquivo inválido! Magic number incorreto.")
                return
            
            pos_size = 780 * 4 + 12
            seen = set()
            duplicates = 0
            
            print(f"  Total de posições: {total:,}")
            print(f"  Analisando...")
            
            # Lê posições
            for i in range(total):
                pos_data = f.read(pos_size)
                if len(pos_data) < pos_size:
                    print(f"  ⚠️  Arquivo incompleto na posição {i}")
                    break
                    
                pos_hash = hashlib.md5(pos_data).hexdigest()
                
                if pos_hash in seen:
                    duplicates += 1
                else:
                    seen.add(pos_hash)
                
                if (i + 1) % 100000 == 0:
                    print(f"    Progresso: {i + 1:,}/{total:,}")
            
            print(f"\n  📊 Resultado:")
            print(f"    Posições únicas: {total - duplicates:,}")
            print(f"    Duplicatas: {duplicates:,}")
            print(f"    Redução: {(duplicates/total)*100:.1f}%")
            
            # Mostra primeiras 5 duplicatas se houver
            if duplicates > 0:
                print(f"\n  🔍 Primeiras 5 posições duplicadas (hash):")
                # Re-analisa para mostrar exemplos
                f.seek(8)
                seen_hashes = {}
                duplicate_examples = []
                
                for i in range(total):
                    pos_data = f.read(pos_size)
                    if len(pos_data) < pos_size:
                        break
                    pos_hash = hashlib.md5(pos_data).hexdigest()
                    
                    if pos_hash in seen_hashes:
                        if len(duplicate_examples) < 5:
                            duplicate_examples.append((i, pos_hash, seen_hashes[pos_hash]))
                    else:
                        seen_hashes[pos_hash] = i
                
                for i, (idx, hash_val, first_occurrence) in enumerate(duplicate_examples, 1):
                    print(f"    {i}. Posição {idx} duplicada (primeira ocorrência: {first_occurrence})")
                    print(f"       Hash: {hash_val}")
                    
    except Exception as e:
        print(f"  ❌ Erro durante análise: {e}")

def analyze_evaluation(filename: str, sample_size: int = 10000):
    """
    Analisa a distribuição de avaliações em um arquivo
    """
    if not os.path.exists(filename):
        print(f"❌ Arquivo não encontrado: {filename}")
        return
    
    print(f"\n📊 Analisando avaliações em: {filename}")
    
    try:
        with open(filename, 'rb') as f:
            # Lê cabeçalho
            magic, total = struct.unpack('4sI', f.read(8))
            
            if magic != b'NNUE':
                print("❌ Arquivo inválido! Magic number incorreto.")
                return
            
            pos_size = 780 * 4 + 12
            scores = []
            actual_sample = min(sample_size, total)
            
            print(f"  Total de posições: {total:,}")
            print(f"  Amostrando {actual_sample:,} posições...")
            
            # Lê posições
            for i in range(actual_sample):
                pos_data = f.read(pos_size)
                if len(pos_data) < pos_size:
                    break
                score = get_position_eval(pos_data)
                scores.append(score)
            
            if not scores:
                print("  ❌ Nenhum score encontrado!")
                return
            
            stats = get_eval_stats(scores)
            
            print(f"\n  📊 Estatísticas de Avaliação (amostra):")
            print_eval_stats(stats)
            
            # Mostra distribuição por faixas
            print(f"\n  📈 Distribuição:")
            ranges = [
                (-float('inf'), -500, "Pretas ganhando (< -5.00)"),
                (-500, -200, "Pretas vantagem (-5.00 a -2.00)"),
                (-200, -50, "Pretas ligeira (-2.00 a -0.50)"),
                (-50, 50, "Equilibrada (-0.50 a 0.50)"),
                (50, 200, "Brancas ligeira (0.50 a 2.00)"),
                (200, 500, "Brancas vantagem (2.00 a 5.00)"),
                (500, float('inf'), "Brancas ganhando (> 5.00)")
            ]
            
            for low, high, label in ranges:
                count = sum(1 for s in scores if low <= s < high)
                pct = count / len(scores) * 100
                bar = '█' * int(pct / 2)
                print(f"    {label:30s}: {count:6,} ({pct:5.1f}%) {bar}")
            
    except Exception as e:
        print(f"  ❌ Erro durante análise: {e}")

def split_dataset(input_file: str, num_parts: int = 5):
    """
    Divide um arquivo grande em partes menores (útil para processamento paralelo)
    """
    if not os.path.exists(input_file):
        print(f"❌ Arquivo não encontrado: {input_file}")
        return
    
    print(f"\n🔪 Dividindo {input_file} em {num_parts} partes...")
    
    # Lê total de posições
    try:
        with open(input_file, 'rb') as f:
            magic, total = struct.unpack('4sI', f.read(8))
            if magic != b'NNUE':
                print("❌ Arquivo inválido!")
                return
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")
        return
    
    positions_per_part = total // num_parts
    extra = total % num_parts
    
    print(f"  Total: {total:,} posições")
    print(f"  Por parte: ~{positions_per_part:,} posições")
    
    with open(input_file, 'rb') as in_f:
        in_f.seek(8)  # Pula cabeçalho
        
        for part in range(num_parts):
            start = part * positions_per_part
            count = positions_per_part + (extra if part == num_parts - 1 else 0)
            
            output = f"{os.path.splitext(input_file)[0]}_part{part+1:02d}.bin"
            
            with open(output, 'wb') as out_f:
                # Escreve cabeçalho
                out_f.write(b'NNUE')
                out_f.write(struct.pack('I', count))
                
                # Pula para posição correta
                # Cada posição tem 780*4 + 12 bytes
                pos_size = 780 * 4 + 12
                in_f.seek(8 + start * pos_size)
                
                # Lê e escreve as posições
                for _ in range(count):
                    data = in_f.read(pos_size)
                    if len(data) < pos_size:
                        break
                    out_f.write(data)
            
            print(f"  ✅ Criado: {output} ({count:,} posições)")

def compare_files(file1: str, file2: str):
    """
    Compara dois arquivos NNUE para verificar se são consistentes
    """
    if not os.path.exists(file1):
        print(f"❌ Arquivo não encontrado: {file1}")
        return
    if not os.path.exists(file2):
        print(f"❌ Arquivo não encontrado: {file2}")
        return
    
    print(f"\n🔍 Comparando {file1} e {file2}")
    
    try:
        with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
            magic1, count1 = struct.unpack('4sI', f1.read(8))
            magic2, count2 = struct.unpack('4sI', f2.read(8))
            
            if magic1 != b'NNUE' or magic2 != b'NNUE':
                print("❌ Magic number inválido")
                return
            
            if count1 != count2:
                print(f"⚠️  Números de posições diferentes: {count1} vs {count2}")
            
            pos_size = 780 * 4 + 12
            
            # Compara primeira posição
            f1.seek(8)
            f2.seek(8)
            
            data1 = f1.read(pos_size)
            data2 = f2.read(pos_size)
            
            if data1 == data2:
                print("✅ Primeira posição IDÊNTICA")
            else:
                print("⚠️  Primeira posição DIFERENTE")
                
                # Mostra diferenças
                diff_positions = []
                for i, (b1, b2) in enumerate(zip(data1, data2)):
                    if b1 != b2:
                        diff_positions.append(i)
                        if len(diff_positions) > 5:
                            break
                
                print(f"  Diferenças em: {diff_positions}")
                print(f"  Primeira diferença no offset {diff_positions[0] if diff_positions else 'N/A'}")
    except Exception as e:
        print(f"❌ Erro ao comparar arquivos: {e}")

# ============== FUNÇÃO PRINCIPAL ==============

def main():
    """Função principal com menu interativo"""
    
    print("="*80)
    print("🔗 FERRAMENTA DE MERGE NNUE COM DEDUPLICAÇÃO E FILTRO DE AVALIAÇÃO")
    print("="*80)
    
    # Encontra arquivos automaticamente
    pattern = input(f"\n📁 Padrão dos arquivos [{MergeConfig.INPUT_PATTERN}]: ").strip()
    if not pattern:
        pattern = MergeConfig.INPUT_PATTERN
    
    # Filtra arquivos existentes e não inclui o arquivo de saída
    all_files = sorted(glob.glob(pattern))
    # Remove arquivos que já são merged
    all_files = [f for f in all_files if "merged" not in f.lower()]
    
    if not all_files:
        print(f"❌ Nenhum arquivo encontrado com padrão: {pattern}")
        
        # Opção manual
        manual = input("\nDeseja especificar arquivos manualmente? (s/N): ")
        if manual.lower() == 's':
            files_input = input("Arquivos (separados por espaço): ").strip()
            all_files = files_input.split()
    
    if not all_files:
        print("❌ Nenhum arquivo especificado!")
        return
    
    print(f"\n📂 Arquivos encontrados:")
    for i, f in enumerate(all_files):
        try:
            size = os.path.getsize(f) / 1024 / 1024
            print(f"  {i+1:2d}. {f:40s} ({size:.2f} MB)")
        except:
            print(f"  {i+1:2d}. {f:40s} (acesso negado)")
    
    output = input(f"\n📁 Arquivo de saída [{MergeConfig.OUTPUT_FILE}]: ").strip()
    if not output:
        output = MergeConfig.OUTPUT_FILE
    
    # Opções
    print(f"\n⚙️  Opções:")
    print(f"  1. Mesclar todos os arquivos com deduplicação ({len(all_files)} arquivos)")
    print(f"  2. Mesclar sem deduplicação")
    print(f"  3. Filtrar por avaliação (remover melhores/piores posições)")
    print(f"  4. Selecionar arquivos específicos")
    print(f"  5. Analisar duplicatas em um arquivo")
    print(f"  6. Analisar avaliações de um arquivo")
    print(f"  7. Sair")
    
    option = input("\nEscolha uma opção: ").strip()
    
    if option == '2':
        # Mesclar sem deduplicação
        print("\n🔗 Mesclando sem deduplicação...")
        success = merge_files(
            files=all_files,
            output_file=output,
            verify=MergeConfig.VERIFY_INTEGRITY,
            backup=MergeConfig.CREATE_BACKUP,
            remove_duplicates=False
        )
        if success:
            print("\n" + "="*80)
            print("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
            print("="*80)
        return
    
    elif option == '3':
        # Filtrar por avaliação
        print("\n🎯 FILTRO POR AVALIAÇÃO STOCKFISH")
        print("=" * 40)
        
        print("\nQual lado filtrar?")
        print("  1. Pretas (remover posições onde Pretas estão melhores)")
        print("  2. Brancas (remover posições onde Brancas estão melhores)")
        print("  3. Ambos (remover posições extremas)")
        
        side_choice = input("\nEscolha [1]: ").strip()
        
        if side_choice == '2':
            filter_side = 'white'
            side_name = "Brancas"
        elif side_choice == '3':
            filter_side = 'both'
            side_name = "Ambos"
        else:
            filter_side = 'black'
            side_name = "Pretas"
        
        print(f"\nRemover melhores ou piores posições para {side_name}?")
        print("  1. Melhores (remover posições com maior vantagem)")
        print("  2. Piores (remover posições com menor vantagem)")
        
        mode_choice = input("\nEscolha [1]: ").strip()
        filter_mode = 'worst' if mode_choice == '2' else 'best'
        
        print(f"\nMétodo de filtragem:")
        print("  1. Por percentual (ex: remover 10% melhores)")
        print("  2. Por limite absoluto (ex: remover scores > 500 centipawns)")
        
        method_choice = input("\nEscolha [1]: ").strip()
        
        if method_choice == '2':
            limit_str = input("Limite em centipawns (ex: 500 = 5.00 pawns): ")
            try:
                score_limit = int(limit_str)
                filter_percent = 0  # Não usado
                print(f"  ✅ Removendo posições com |score| >= {score_limit} centipawns")
            except:
                print("  ❌ Valor inválido! Usando percentual...")
                score_limit = None
                filter_percent = float(input("Percentual a remover (0-100) [10]: ").strip() or 10)
        else:
            score_limit = None
            filter_percent = float(input("Percentual a remover (0-100) [10]: ").strip() or 10)
        
        print(f"\n📊 Resumo do filtro:")
        print(f"  Lado: {side_name}")
        print(f"  Modo: {filter_mode}")
        if score_limit is not None:
            print(f"  Limite: {score_limit} centipawns")
        else:
            print(f"  Percentual: {filter_percent}%")
        
        confirm = input("\nConfirmar filtro? (s/N): ")
        if confirm.lower() != 's':
            print("❌ Operação cancelada!")
            return
        
        # Executa merge com filtro
        success = merge_files(
            files=all_files,
            output_file=output,
            verify=MergeConfig.VERIFY_INTEGRITY,
            backup=MergeConfig.CREATE_BACKUP,
            remove_duplicates=MergeConfig.REMOVE_DUPLICATES,
            dedup_method=MergeConfig.DEDUP_METHOD,
            enable_eval_filter=True,
            filter_side=filter_side,
            filter_mode=filter_mode,
            filter_percent=filter_percent,
            score_limit=score_limit
        )
        
        if success:
            print("\n" + "="*80)
            print("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
            print("="*80)
        return
    
    elif option == '4':
        print("\nSelecione os arquivos (ex: 1,3,5):")
        indices = input("Índices: ").strip()
        
        selected = []
        for idx_str in indices.split(','):
            try:
                idx = int(idx_str.strip()) - 1
                if 0 <= idx < len(all_files):
                    selected.append(all_files[idx])
            except:
                pass
        
        if selected:
            all_files = selected
            print(f"\n✅ Selecionados {len(all_files)} arquivos")
        else:
            print("❌ Nenhum arquivo selecionado!")
            return
    
    elif option == '5':
        print("\n📂 Arquivos disponíveis para análise de duplicatas:")
        for i, f in enumerate(all_files):
            print(f"  {i+1:2d}. {f}")
        
        file_choice = input("\nNúmero do arquivo ou caminho completo: ").strip()
        
        # Tenta interpretar como número
        try:
            idx = int(file_choice) - 1
            if 0 <= idx < len(all_files):
                file_to_analyze = all_files[idx]
            else:
                file_to_analyze = file_choice
        except:
            file_to_analyze = file_choice
        
        analyze_duplicates(file_to_analyze)
        return
    
    elif option == '6':
        print("\n📂 Arquivos disponíveis para análise de avaliações:")
        for i, f in enumerate(all_files):
            print(f"  {i+1:2d}. {f}")
        
        file_choice = input("\nNúmero do arquivo ou caminho completo: ").strip()
        
        # Tenta interpretar como número
        try:
            idx = int(file_choice) - 1
            if 0 <= idx < len(all_files):
                file_to_analyze = all_files[idx]
            else:
                file_to_analyze = file_choice
        except:
            file_to_analyze = file_choice
        
        sample = input("Tamanho da amostra (padrão 10000): ").strip()
        sample_size = int(sample) if sample else 10000
        analyze_evaluation(file_to_analyze, sample_size)
        return
    
    elif option == '7':
        print("\n👋 Saindo...")
        return
    
    # Opção 1 (padrão): merge normal com deduplicação
    # Método de deduplicação
    print(f"\n🔍 Método de deduplicação:")
    print(f"  [1] Hash (rápido, 99.99% preciso)")
    print(f"  [2] Exact (mais lento, 100% preciso)")
    
    method_choice = input("\nEscolha [1]: ").strip()
    dedup_method = 'hash' if method_choice != '2' else 'exact'
    
    # Confirmação
    print(f"\n📊 Resumo:")
    print(f"  Arquivos: {len(all_files)}")
    total_size = sum(os.path.getsize(f) for f in all_files) / 1024 / 1024
    print(f"  Tamanho total: {total_size:.2f} MB")
    print(f"  Saída: {output}")
    print(f"  Deduplicação: {'Sim' if MergeConfig.REMOVE_DUPLICATES else 'Não'}")
    print(f"  Método: {dedup_method.upper()}")
    
    confirm = input("\nConfirmar merge? (S/n): ").strip()
    if confirm.lower() == 'n':
        print("❌ Operação cancelada!")
        return
    
    # Executa merge
    success = merge_files(
        files=all_files,
        output_file=output,
        verify=MergeConfig.VERIFY_INTEGRITY,
        backup=MergeConfig.CREATE_BACKUP,
        remove_duplicates=MergeConfig.REMOVE_DUPLICATES,
        dedup_method=dedup_method
    )
    
    if success:
        print("\n" + "="*80)
        print("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
        print("="*80)
        
        # Mostra opções adicionais
        print("\n📋 Opções adicionais:")
        print("  1. Ver estatísticas do arquivo")
        print("  2. Dividir em partes menores")
        print("  3. Analisar duplicatas no arquivo final")
        print("  4. Analisar avaliações no arquivo final")
        print("  5. Sair")
        
        option = input("\nEscolha: ").strip()
        
        if option == '1':
            count, size, valid = read_file_info(output)
            print(f"\n📊 Estatísticas:")
            print(f"  Posições: {count:,}")
            print(f"  Tamanho: {size/1024/1024:.2f} MB")
            print(f"  Válido: {'Sim' if valid else 'Não'}")
        
        elif option == '2':
            try:
                parts = int(input("Número de partes: "))
                split_dataset(output, parts)
            except ValueError:
                print("❌ Número inválido!")
        
        elif option == '3':
            analyze_duplicates(output)
        
        elif option == '4':
            sample = input("Tamanho da amostra (padrão 10000): ").strip()
            sample_size = int(sample) if sample else 10000
            analyze_evaluation(output, sample_size)

# ============== EXECUÇÃO RÁPIDA ==============

def quick_merge():
    """
    Função para merge rápido com deduplicação
    """
    pattern = "training_data*_prod.bin"
    output = "training_data_merged_prod.bin"
    
    files = sorted(glob.glob(pattern))
    # Remove arquivos merged existentes
    files = [f for f in files if "merged" not in f.lower()]
    
    if not files:
        print(f"❌ Nenhum arquivo encontrado com padrão: {pattern}")
        return
    
    print(f"🔗 Mesclando {len(files)} arquivos com deduplicação...")
    merge_files(files, output, remove_duplicates=True, dedup_method='hash')

# ============== SCRIPT PRINCIPAL ==============

if __name__ == "__main__":
    # Se quiser executar automaticamente:
    # quick_merge()
    
    # Ou com menu interativo:
    main()