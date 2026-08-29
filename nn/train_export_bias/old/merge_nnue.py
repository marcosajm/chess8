#!/usr/bin/env python3
"""
Ferramenta para Combinar Múltiplos Arquivos NNUE .bin
Uso: python3 merge_nnue_data.py
"""

import struct
import numpy as np
import os
import glob
from pathlib import Path
from typing import List, Tuple
import shutil
import time

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
                #print(f"  ⚠️  Magic number inválido: {filename}")
                return 0, 0, False
            
            # Verifica se o arquivo tem o tamanho correto
            expected_size = 8 + count * (780 * 4 + 12)  # header + cada posição
            actual_size = os.path.getsize(filename)
            
            if actual_size != expected_size:
                #print(f"  ⚠️  Tamanho incorreto: {filename}")
                #print(f"     Esperado: {expected_size}, Atual: {actual_size}")
                return count, actual_size, False
            
            return count, actual_size, True
            
    except Exception as e:
        #print(f"  ❌ Erro ao ler {filename}: {e}")
        return 0, 0, False

def merge_files(files: List[str], output_file: str, 
                verify: bool = True, backup: bool = True) -> bool:
    """
    Mescla múltiplos arquivos NNUE em um único arquivo
    
    Args:
        files: Lista de caminhos dos arquivos
        output_file: Caminho do arquivo de saída
        verify: Verificar integridade dos arquivos
        backup: Criar backup do arquivo de saída se existir
    """
    
    #print("\n" + "="*80)
    #print("🔗 MERGE DE ARQUIVOS NNUE")
    #print("="*80)
    
    # Filtra apenas arquivos existentes
    files = [f for f in files if os.path.exists(f)]
    
    if not files:
        #print("❌ Nenhum arquivo encontrado!")
        return False
    
    #print(f"\n📂 Encontrados {len(files)} arquivos:")
    
    # Lê informações de cada arquivo
    file_info = []
    total_positions = 0
    
    for i, file in enumerate(files):
        count, size, valid = read_file_info(file)
        
        if not valid and verify:
            #print(f"  ❌ Arquivo inválido: {file}")
            return False
        
        if count == 0:
            #print(f"  ⚠️  Arquivo vazio: {file}")
            continue
        
        file_info.append({
            'file': file,
            'count': count,
            'size': size,
            'valid': valid
        })
        
        total_positions += count
        #print(f"  {i+1:2d}. {os.path.basename(file):30s} - {count:6,} posições ({size/1024/1024:.2f} MB)")
    
    if not file_info:
        #print("❌ Nenhum arquivo válido encontrado!")
        return False
    
    #print(f"\n📊 Total de posições: {total_positions:,}")
    #print(f"📊 Tamanho estimado: {total_positions * (780*4 + 12) / 1024 / 1024:.2f} MB")
    
    # Verifica se o arquivo de saída existe
    if os.path.exists(output_file):
        if backup:
            backup_file = output_file + ".backup"
            #print(f"\n💾 Criando backup: {backup_file}")
            shutil.copy2(output_file, backup_file)
        
        response = input(f"\n⚠️  Arquivo {output_file} já existe. Sobrescrever? (s/N): ")
        if response.lower() != 's':
            #print("❌ Operação cancelada!")
            return False
    
    # Mescla os arquivos
    #print(f"\n🚀 Mesclando {len(file_info)} arquivos...")
    start_time = time.time()
    
    try:
        with open(output_file, 'wb') as out_f:
            # Escreve cabeçalho inicial (será atualizado depois)
            out_f.write(b'NNUE')
            out_f.write(struct.pack('I', total_positions))
            
            positions_written = 0
            
            # Para cada arquivo
            for info in file_info:
                #print(f"  📖 Processando: {os.path.basename(info['file'])}")
                
                with open(info['file'], 'rb') as in_f:
                    # Pula cabeçalho do arquivo de entrada
                    in_f.seek(8)
                    
                    # Lê e escreve cada posição
                    for pos_idx in range(info['count']):
                        # Lê features (780 floats = 3120 bytes)
                        feat_data = in_f.read(780 * 4)
                        if len(feat_data) < 780 * 4:
                            #print(f"    ⚠️  Erro ao ler posição {pos_idx}")
                            break
                        
                        # Lê scores (3 floats = 12 bytes)
                        score_data = in_f.read(12)
                        if len(score_data) < 12:
                            #print(f"    ⚠️  Erro ao ler scores {pos_idx}")
                            break
                        
                        # Escreve no arquivo de saída
                        out_f.write(feat_data)
                        out_f.write(score_data)
                        
                        positions_written += 1
                        
                        # Progresso
                        if (pos_idx + 1) % 10000 == 0:
                            #print(f"    Progresso: {pos_idx + 1:,}/{info['count']:,} posições")
                    
                    #print(f"    ✅ {info['count']:,} posições escritas")
            
            # Volta para atualizar o cabeçalho com o número correto
            out_f.seek(4)
            out_f.write(struct.pack('I', positions_written))
        
        elapsed = time.time() - start_time
        file_size = os.path.getsize(output_file) / 1024 / 1024
        
        #print(f"\n✅ Merge concluído com sucesso!")
        #print(f"  📁 Arquivo: {output_file}")
        #print(f"  📊 Posições: {positions_written:,}")
        #print(f"  💾 Tamanho: {file_size:.2f} MB")
        #print(f"  ⏱️  Tempo: {elapsed:.2f} segundos")
        
        # Verifica o arquivo gerado
        if verify:
            #print("\n🔍 Verificando integridade do arquivo final...")
            count, size, valid = read_file_info(output_file)
            
            if valid and count == positions_written:
                #print(f"  ✅ Arquivo final válido!")
                #print(f"  📊 Total de posições: {count:,}")
                #print(f"  📏 Tamanho: {size/1024/1024:.2f} MB")
            else:
                #print(f"  ❌ Arquivo final inválido!")
                return False
        
        # Pergunta se deseja deletar os originais
        if MergeConfig.DELETE_ORIGINALS:
            response = input(f"\n🗑️  Deletar arquivos originais? (s/N): ")
            if response.lower() == 's':
                for info in file_info:
                    os.remove(info['file'])
                    #print(f"  🗑️  Deletado: {info['file']}")
        
        return True
        
    except Exception as e:
        #print(f"\n❌ Erro durante o merge: {e}")
        return False

# ============== FUNÇÕES ADICIONAIS ==============

def split_dataset(input_file: str, num_parts: int = 5):
    """
    Divide um arquivo grande em partes menores (útil para processamento paralelo)
    """
    #print(f"\n🔪 Dividindo {input_file} em {num_parts} partes...")
    
    # Lê total de posições
    with open(input_file, 'rb') as f:
        magic, total = struct.unpack('4sI', f.read(8))
        if magic != b'NNUE':
            #print("❌ Arquivo inválido!")
            return
    
    positions_per_part = total // num_parts
    extra = total % num_parts
    
    #print(f"  Total: {total:,} posições")
    #print(f"  Por parte: ~{positions_per_part:,} posições")
    
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
            
            #print(f"  ✅ Criado: {output} ({count:,} posições)")

def compare_files(file1: str, file2: str):
    """
    Compara dois arquivos NNUE para verificar se são consistentes
    """
    #print(f"\n🔍 Comparando {file1} e {file2}")
    
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        magic1, count1 = struct.unpack('4sI', f1.read(8))
        magic2, count2 = struct.unpack('4sI', f2.read(8))
        
        if magic1 != b'NNUE' or magic2 != b'NNUE':
            #print("❌ Magic number inválido")
            return
        
        if count1 != count2:
            #print(f"⚠️  Números de posições diferentes: {count1} vs {count2}")
        
        pos_size = 780 * 4 + 12
        
        # Compara primeira posição
        f1.seek(8)
        f2.seek(8)
        
        data1 = f1.read(pos_size)
        data2 = f2.read(pos_size)
        
        if data1 == data2:
            #print("✅ Primeira posição IDÊNTICA")
        else:
            #print("⚠️  Primeira posição DIFERENTE")
            
            # Mostra diferenças
            diff_positions = []
            for i, (b1, b2) in enumerate(zip(data1, data2)):
                if b1 != b2:
                    diff_positions.append(i)
                    if len(diff_positions) > 5:
                        break
            
            #print(f"  Diferenças em: {diff_positions}")
            #print(f"  Primeira diferença no offset {diff_positions[0] if diff_positions else 'N/A'}")

# ============== FUNÇÃO PRINCIPAL ==============

def main():
    """Função principal com menu interativo"""
    
    #print("="*80)
    #print("🔗 FERRAMENTA DE MERGE NNUE")
    #print("="*80)
    
    # Encontra arquivos automaticamente
    pattern = input(f"\n📁 Padrão dos arquivos [{MergeConfig.INPUT_PATTERN}]: ").strip()
    if not pattern:
        pattern = MergeConfig.INPUT_PATTERN
    
    files = sorted(glob.glob(pattern))
    
    if not files:
        #print(f"❌ Nenhum arquivo encontrado com padrão: {pattern}")
        
        # Opção manual
        manual = input("\nDeseja especificar arquivos manualmente? (s/N): ")
        if manual.lower() == 's':
            files_input = input("Arquivos (separados por espaço): ").strip()
            files = files_input.split()
    
    if not files:
        #print("❌ Nenhum arquivo especificado!")
        return
    
    #print(f"\n📂 Arquivos encontrados:")
    for i, f in enumerate(files):
        size = os.path.getsize(f) / 1024 / 1024
        #print(f"  {i+1:2d}. {f:40s} ({size:.2f} MB)")
    
    output = input(f"\n📁 Arquivo de saída [{MergeConfig.OUTPUT_FILE}]: ").strip()
    if not output:
        output = MergeConfig.OUTPUT_FILE
    
    # Opções
    #print(f"\n⚙️  Opções:")
    #print(f"  1. Mesclar todos os arquivos ({len(files)} arquivos)")
    #print(f"  2. Selecionar arquivos específicos")
    #print(f"  3. Sair")
    
    option = input("\nEscolha uma opção: ").strip()
    
    if option == '2':
        #print("\nSelecione os arquivos (ex: 1,3,5):")
        indices = input("Índices: ").strip()
        
        selected = []
        for idx_str in indices.split(','):
            try:
                idx = int(idx_str.strip()) - 1
                if 0 <= idx < len(files):
                    selected.append(files[idx])
            except:
                pass
        
        if selected:
            files = selected
            #print(f"\n✅ Selecionados {len(files)} arquivos")
        else:
            #print("❌ Nenhum arquivo selecionado!")
            return
    
    elif option == '3':
        #print("\n👋 Saindo...")
        return
    
    # Confirmação
    #print(f"\n📊 Resumo:")
    #print(f"  Arquivos: {len(files)}")
    total_size = sum(os.path.getsize(f) for f in files) / 1024 / 1024
    #print(f"  Tamanho total: {total_size:.2f} MB")
    #print(f"  Saída: {output}")
    
    confirm = input("\nConfirmar merge? (S/n): ").strip()
    if confirm.lower() == 'n':
        #print("❌ Operação cancelada!")
        return
    
    # Executa merge
    success = merge_files(
        files=files,
        output_file=output,
        verify=MergeConfig.VERIFY_INTEGRITY,
        backup=MergeConfig.CREATE_BACKUP
    )
    
    if success:
        #print("\n" + "="*80)
        #print("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
        #print("="*80)
        
        # Mostra opções adicionais
        #print("\n📋 Opções adicionais:")
        #print("  1. Ver estatísticas do arquivo")
        #print("  2. Dividir em partes menores")
        #print("  3. Sair")
        
        option = input("\nEscolha: ").strip()
        
        if option == '1':
            count, size, valid = read_file_info(output)
            #print(f"\n📊 Estatísticas:")
            #print(f"  Posições: {count:,}")
            #print(f"  Tamanho: {size/1024/1024:.2f} MB")
            #print(f"  Válido: {'Sim' if valid else 'Não'}")
        
        elif option == '2':
            parts = int(input("Número de partes: "))
            split_dataset(output, parts)

# ============== EXECUÇÃO RÁPIDA ==============

def quick_merge():
    """
    Função para merge rápido - apenas mescla todos os arquivos do padrão
    """
    pattern = "training_data*_prod.bin"
    output = "training_data_merged_prod.bin"
    
    files = sorted(glob.glob(pattern))
    
    if not files:
        #print(f"❌ Nenhum arquivo encontrado com padrão: {pattern}")
        return
    
    #print(f"🔗 Mesclando {len(files)} arquivos...")
    merge_files(files, output)

# ============== SCRIPT PRINCIPAL ==============

if __name__ == "__main__":
    # Se quiser executar automaticamente:
    # quick_merge()
    
    # Ou com menu interativo:
    main()
