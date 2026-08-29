#!/usr/bin/env python3
"""
Ferramenta para Combinar Múltiplos Arquivos NNUE .bin com Remoção de Duplicatas
Uso: python3 merge_data_nnue_v3.py
"""

import struct
import numpy as np
import os
import glob
from pathlib import Path
from typing import List, Tuple, Set, Dict
import shutil
import time
import hashlib
import random
from collections import Counter

# ============== CONFIGURAÇÃO ==============
class MergeConfig:
    INPUT_PATTERN = "training_data*_prod.bin"
    OUTPUT_FILE = "training_data_merged_prod.bin"
    CREATE_BACKUP = True
    DELETE_ORIGINALS = False
    VERIFY_INTEGRITY = True
    REMOVE_DUPLICATES = True
    DEDUP_METHOD = 'hash'
    
    # Filtros
    FILTER_LOW_SCORES = True
    MIN_SCORE_THRESHOLD = 0.1
    FILTER_HIGH_SCORES = True
    MAX_SCORE_THRESHOLD = 0.99
    FILTER_DRAW_SCORES = True
    DRAW_THRESHOLD = 0.45
    FILTER_OUTLIERS = True
    OUTLIER_STD_MULTIPLIER = 3.0

# ============== FUNÇÕES DE DEDUPLICAÇÃO ==============

def deduplicate_file(input_file: str, output_file: str = None, 
                     method: str = 'hash', backup: bool = True) -> bool:
    """
    Remove duplicatas de um arquivo NNUE existente
    
    Args:
        input_file: Arquivo de entrada
        output_file: Arquivo de saída (se None, sobrescreve o original)
        method: 'hash' ou 'exact'
        backup: Criar backup antes de sobrescrever
    """
    if not os.path.exists(input_file):
        #print(f"❌ Arquivo não encontrado: {input_file}")
        return False
    
    # Se não especificou output, usa o mesmo arquivo (sobrescreve)
    if output_file is None:
        output_file = input_file
    
    #print("\n" + "="*80)
    #print(f"🗑️  REMOVENDO DUPLICATAS DE: {os.path.basename(input_file)}")
    #print("="*80)
    
    # Lê informações do arquivo
    count, size, valid = read_file_info(input_file)
    if not valid:
        #print("❌ Arquivo inválido!")
        return False
    
    #print(f"\n📊 Total de posições: {count:,}")
    #print(f"💾 Tamanho: {size/1024/1024:.2f} MB")
    #print(f"🔍 Método: {method.upper()}")
    
    # Se o arquivo de saída é o mesmo que o de entrada, cria backup
    if input_file == output_file and backup:
        backup_file = input_file + ".backup"
        #print(f"\n💾 Criando backup: {backup_file}")
        shutil.copy2(input_file, backup_file)
    
    # Processa a deduplicação
    #print(f"\n🚀 Processando...")
    start_time = time.time()
    
    pos_size = 780 * 4 + 12
    seen_positions = set()
    duplicate_count = 0
    positions_written = 0
    
    try:
        # Primeiro, lê todas as posições para memória (para arquivos grandes, pode ser pesado)
        # Mas para arquivos de treinamento NNUE, geralmente são pequenos o suficiente
        all_positions = []
        
        with open(input_file, 'rb') as in_f:
            magic, total = struct.unpack('4sI', in_f.read(8))
            
            if magic != b'NNUE':
                #print("❌ Magic number inválido!")
                return False
            
            in_f.seek(8)
            for i in range(total):
                pos_data = in_f.read(pos_size)
                if len(pos_data) < pos_size:
                    #print(f"⚠️  Arquivo incompleto na posição {i}")
                    break
                all_positions.append(pos_data)
                
                if (i + 1) % 10000 == 0:
                    #print(f"  Lendo: {i + 1:,}/{total:,}")
        
        #print(f"  ✅ Lidas {len(all_positions):,} posições")
        
        # Processa as posições únicas
        #print(f"\n🔍 Identificando duplicatas...")
        
        with open(output_file, 'wb') as out_f:
            # Escreve cabeçalho (será atualizado depois)
            out_f.write(b'NNUE')
            out_f.write(struct.pack('I', 0))
            
            for i, pos_data in enumerate(all_positions):
                # Calcula chave
                if method == 'hash':
                    pos_key = hashlib.md5(pos_data).hexdigest()
                else:  # exact
                    pos_key = pos_data
                
                # Verifica se já viu
                if pos_key in seen_positions:
                    duplicate_count += 1
                else:
                    seen_positions.add(pos_key)
                    out_f.write(pos_data)
                    positions_written += 1
                
                # Progresso
                if (i + 1) % 5000 == 0:
                    #print(f"  Processando: {i + 1:,}/{len(all_positions):,}")
            
            # Atualiza cabeçalho
            out_f.seek(4)
            out_f.write(struct.pack('I', positions_written))
        
        elapsed = time.time() - start_time
        new_size = os.path.getsize(output_file) / 1024 / 1024
        
        #print(f"\n✅ Deduplicação concluída!")
        #print(f"  📊 Posições originais: {len(all_positions):,}")
        #print(f"  🗑️  Duplicatas removidas: {duplicate_count:,}")
        #print(f"  📊 Posições únicas: {positions_written:,}")
        #print(f"  📊 Redução: {(duplicate_count/len(all_positions))*100:.1f}%")
        #print(f"  💾 Tamanho original: {size/1024/1024:.2f} MB")
        #print(f"  💾 Tamanho novo: {new_size:.2f} MB")
        #print(f"  💾 Economia: {(size - os.path.getsize(output_file))/1024/1024:.2f} MB")
        #print(f"  ⏱️  Tempo: {elapsed:.2f} segundos")
        
        # Verifica integridade
        count_final, size_final, valid_final = read_file_info(output_file)
        if valid_final and count_final == positions_written:
            #print(f"\n✅ Arquivo final válido!")
            return True
        else:
            #print(f"\n❌ Arquivo final inválido!")
            return False
        
    except Exception as e:
        #print(f"❌ Erro durante deduplicação: {e}")
        return False

def deduplicate_all_files(directory: str = ".", pattern: str = "training_data*_prod.bin"):
    """
    Remove duplicatas de todos os arquivos NNUE em um diretório
    """
    #print("\n" + "="*80)
    #print("🗑️  REMOVENDO DUPLICATAS DE TODOS OS ARQUIVOS")
    #print("="*80)
    
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    files = [f for f in files if "merged" not in f.lower() and "backup" not in f.lower()]
    
    if not files:
        #print("❌ Nenhum arquivo encontrado!")
        return
    
    #print(f"\n📂 Encontrados {len(files)} arquivos:")
    for i, f in enumerate(files, 1):
        count, size, valid = read_file_info(f)
        if valid:
            #print(f"  {i:2d}. {os.path.basename(f):40s} - {count:6,} posições ({size/1024/1024:.2f} MB)")
    
    confirm = input(f"\n⚠️  Remover duplicatas de TODOS os {len(files)} arquivos? (s/N): ")
    if confirm.lower() != 's':
        #print("❌ Operação cancelada!")
        return
    
    total_removed = 0
    total_original = 0
    
    for f in files:
        #print(f"\n{'='*60}")
        #print(f"Processando: {os.path.basename(f)}")
        
        # Cria arquivo temporário
        temp_file = f + ".dedup"
        if deduplicate_file(f, temp_file, backup=False):
            # Substitui o original
            count, size, valid = read_file_info(temp_file)
            if valid:
                # Remove original e renomeia temp
                os.remove(f)
                shutil.move(temp_file, f)
                total_removed += count
                
                # Verifica novo tamanho
                new_count, new_size, _ = read_file_info(f)
                total_original += new_count
                #print(f"  ✅ Arquivo atualizado: {new_count:,} posições")
    
    #print(f"\n{'='*60}")
    #print(f"✅ Processo concluído!")
    #print(f"  Total de posições únicas: {total_original:,}")
    #print(f"  Total removido: {total_removed:,}")

# ============== FUNÇÕES EXISTENTES ==============

def read_file_info(filename: str) -> Tuple[int, int, bool]:
    """Lê informações de um arquivo NNUE"""
    try:
        with open(filename, 'rb') as f:
            magic, count = struct.unpack('4sI', f.read(8))
            
            if magic != b'NNUE':
                return 0, 0, False
            
            expected_size = 8 + count * (780 * 4 + 12)
            actual_size = os.path.getsize(filename)
            
            if actual_size != expected_size:
                return count, actual_size, False
            
            return count, actual_size, True
            
    except Exception as e:
        return 0, 0, False

def get_position_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

def analyze_duplicates(filename: str):
    """Analisa duplicatas em um arquivo"""
    if not os.path.exists(filename):
        #print(f"❌ Arquivo não encontrado: {filename}")
        return
    
    #print(f"\n🔍 Analisando duplicatas em: {filename}")
    
    try:
        with open(filename, 'rb') as f:
            magic, total = struct.unpack('4sI', f.read(8))
            
            if magic != b'NNUE':
                #print("❌ Arquivo inválido!")
                return
            
            pos_size = 780 * 4 + 12
            seen = set()
            duplicates = 0
            duplicate_positions = []
            
            #print(f"  Total de posições: {total:,}")
            #print(f"  Analisando...")
            
            f.seek(8)
            for i in range(total):
                pos_data = f.read(pos_size)
                if len(pos_data) < pos_size:
                    break
                    
                pos_hash = hashlib.md5(pos_data).hexdigest()
                
                if pos_hash in seen:
                    duplicates += 1
                    if len(duplicate_positions) < 10:
                        duplicate_positions.append((i, pos_hash))
                else:
                    seen.add(pos_hash)
                
                if (i + 1) % 100000 == 0:
                    #print(f"    Progresso: {i + 1:,}/{total:,}")
            
            #print(f"\n  📊 Resultado:")
            #print(f"    Posições únicas: {total - duplicates:,}")
            #print(f"    Duplicatas: {duplicates:,}")
            #print(f"    Redução: {(duplicates/total)*100:.1f}%")
            
            if duplicate_positions:
                #print(f"\n  🔍 Primeiras {min(5, len(duplicate_positions))} duplicatas:")
                for idx, (pos, hash_val) in enumerate(duplicate_positions[:5], 1):
                    #print(f"    {idx}. Posição {pos} - Hash: {hash_val}")
            
            return duplicates, total
            
    except Exception as e:
        #print(f"  ❌ Erro durante análise: {e}")
        return None, None

def analyze_position_scores(filename: str):
    """Analisa distribuição de scores"""
    if not os.path.exists(filename):
        #print(f"❌ Arquivo não encontrado: {filename}")
        return
    
    #print(f"\n📊 ANALISANDO DISTRIBUIÇÃO DE SCORES: {filename}")
    
    try:
        with open(filename, 'rb') as f:
            magic, total = struct.unpack('4sI', f.read(8))
            
            if magic != b'NNUE':
                #print("❌ Arquivo inválido!")
                return
            
            scores = []
            pos_size = 780 * 4 + 12
            
            f.seek(8)
            for i in range(min(total, 100000)):
                f.read(780 * 4)
                score_data = f.read(12)
                if len(score_data) < 12:
                    break
                win, draw, loss = struct.unpack('fff', score_data)
                scores.append(win)
            
            if not scores:
                #print("⚠️  Nenhum score válido encontrado!")
                return
            
            scores_array = np.array(scores)
            
            #print(f"\n  📈 Estatísticas dos scores (win probability):")
            #print(f"    Média: {np.mean(scores_array):.4f}")
            #print(f"    Mediana: {np.median(scores_array):.4f}")
            #print(f"    Desvio padrão: {np.std(scores_array):.4f}")
            #print(f"    Mínimo: {np.min(scores_array):.4f}")
            #print(f"    Máximo: {np.max(scores_array):.4f}")
            
            #print(f"\n  📊 Percentis:")
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                #print(f"    {p}%: {np.percentile(scores_array, p):.4f}")
            
            # Distribuição
            bins = np.linspace(0, 1, 11)
            hist, _ = np.histogram(np.clip(scores_array, 0, 1), bins=bins)
            #print(f"\n  📊 Distribuição (0-1):")
            for i in range(len(bins)-1):
                percentage = (hist[i] / len(scores_array)) * 100
                bar = "█" * int(percentage * 2)
                #print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:5d} ({percentage:5.1f}%) {bar}")
            
            return scores_array
            
    except Exception as e:
        #print(f"  ❌ Erro durante análise: {e}")
        return None

def merge_files_advanced(files: List[str], output_file: str, config: MergeConfig) -> bool:
    """Mescla arquivos com filtros avançados"""
    # (mantido o mesmo código do merge anterior)
    # Por brevidade, mantive a função mas você pode copiar a versão completa do código anterior
    #print("\n⚠️  Função merge_files_advanced - implementar com o código anterior")
    return False

# ============== FUNÇÃO PRINCIPAL ==============

def main():
    """Função principal com menu interativo"""
    
    #print("="*80)
    #print("🔗 FERRAMENTA DE MERGE NNUE COM DEDUPLICAÇÃO")
    #print("="*80)
    
    config = MergeConfig()
    
    # Encontra arquivos
    pattern = input(f"\n📁 Padrão dos arquivos [{config.INPUT_PATTERN}]: ").strip()
    if not pattern:
        pattern = config.INPUT_PATTERN
    
    all_files = sorted(glob.glob(pattern))
    all_files = [f for f in all_files if "merged" not in f.lower() and "backup" not in f.lower()]
    
    if not all_files:
        #print(f"❌ Nenhum arquivo encontrado!")
        return
    
    #print(f"\n📂 Arquivos encontrados: {len(all_files)}")
    
    output = input(f"\n📁 Arquivo de saída [{config.OUTPUT_FILE}]: ").strip()
    if not output:
        output = config.OUTPUT_FILE
    
    # Menu principal
    #print(f"\n⚙️  Opções:")
    #print(f"  1. Mesclar todos com filtros avançados")
    #print(f"  2. Mesclar sem filtros (apenas merge)")
    #print(f"  3. Analisar duplicatas em um arquivo")
    #print(f"  4. Analisar distribuição de scores")
    #print(f"  5. Configurar filtros")
    #print(f"  6. 🗑️  REMOVER DUPLICATAS DE UM ARQUIVO EXISTENTE")
    #print(f"  7. 🗑️  REMOVER DUPLICATAS DE TODOS OS ARQUIVOS")
    #print(f"  8. Sair")
    
    option = input("\nEscolha: ").strip()
    
    if option == '6':
        # Opção 6: Remover duplicatas de um arquivo existente
        #print("\n📂 Arquivos disponíveis:")
        files_for_dedup = sorted(glob.glob("training_data*_prod.bin"))
        files_for_dedup = [f for f in files_for_dedup if "backup" not in f.lower()]
        
        for i, f in enumerate(files_for_dedup, 1):
            try:
                count, size, valid = read_file_info(f)
                if valid:
                    #print(f"  {i:2d}. {f:45s} ({count:6,} posições, {size/1024/1024:.2f} MB)")
                else:
                    #print(f"  {i:2d}. {f:45s} (⚠️  inválido)")
            except:
                #print(f"  {i:2d}. {f:45s}")
        
        choice = input("\nNúmero do arquivo ou caminho completo: ").strip()
        
        # Tenta interpretar como número
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files_for_dedup):
                file_to_dedup = files_for_dedup[idx]
            else:
                file_to_dedup = choice
        except:
            file_to_dedup = choice
        
        # Pergunta se quer sobrescrever ou criar novo
        overwrite = input(f"\nSobrescrever {os.path.basename(file_to_dedup)}? (s/N): ").lower() == 's'
        if not overwrite:
            new_file = input("Nome do novo arquivo: ").strip()
            if not new_file:
                new_file = file_to_dedup + ".dedup"
        else:
            new_file = file_to_dedup
        
        # Método de deduplicação
        method = input("Método (hash/exact) [hash]: ").strip() or 'hash'
        
        # Executa deduplicação
        deduplicate_file(file_to_dedup, new_file, method=method, backup=True)
        return
    
    elif option == '7':
        # Opção 7: Remover duplicatas de todos os arquivos
        confirm = input("\n⚠️  Isso irá MODIFICAR TODOS os arquivos. Continuar? (s/N): ")
        if confirm.lower() == 's':
            deduplicate_all_files()
        else:
            #print("❌ Operação cancelada!")
        return
    
    elif option == '8':
        #print("\n👋 Saindo...")
        return
    
    # ... resto do código (opções 1-5 mantidas)

# ============== EXECUÇÃO ==============

if __name__ == "__main__":
    main()