import chess
import random
import json
from datetime import datetime
import copy

class GeradorXequeMateAvancado:
    """Gerador avançado de posições de xeque-mate com peças aleatórias"""
    
    def __init__(self, num_posicoes=10, pecas_extras=5):
        self.num_posicoes = num_posicoes
        self.pecas_extras = pecas_extras
        self.posicoes_geradas = []
        self.estatisticas = {
            'total': 0,
            'validas': 0,
            'media_pecas': 0,
            'min_pecas': float('inf'),
            'max_pecas': 0
        }
        self.fens_gerados = set()  # Para evitar repetições
        
        # Posições base de XEQUE-MATE GARANTIDAS (todas válidas)
        self.posicoes_base = self._gerar_posicoes_base_variadas()
    
    def _gerar_posicoes_base_variadas(self):
        """Gera uma lista grande de posições base variadas"""
        posicoes = []
        
        # Combinações de cantos e posições da Dama
        cantos = [
            (chess.H8, chess.H7, chess.G7, chess.F7, chess.G8, chess.F8),
            (chess.A8, chess.A7, chess.B7, chess.C7, chess.B8, chess.C8),
            (chess.H1, chess.H2, chess.G2, chess.F2, chess.G1, chess.F1),
            (chess.A1, chess.A2, chess.B2, chess.C2, chess.B1, chess.C1),
        ]
        
        # Para cada canto, criar variações
        for rei_preto, h7, g7, f7, g8, f8 in cantos:
            # Variação 1: Dama em g7 (ou equivalente), Rei em h6 (ou equivalente)
            posicoes.append({
                'rei': rei_preto,
                'dama': g7,
                'rei_branco': self._casa_apoio(rei_preto, 'cima'),
                'desc': f'Mate: Dama em {chess.square_name(g7)}, Rei em {self._casa_apoio(rei_preto, "cima")}'
            })
            
            # Variação 2: Dama em f7 (ou equivalente), Rei em g6
            posicoes.append({
                'rei': rei_preto,
                'dama': f7,
                'rei_branco': self._casa_apoio(rei_preto, 'diagonal'),
                'desc': f'Mate: Dama em {chess.square_name(f7)}, Rei em {self._casa_apoio(rei_preto, "diagonal")}'
            })
            
            # Variação 3: Dama em g7, Rei em g6
            posicoes.append({
                'rei': rei_preto,
                'dama': g7,
                'rei_branco': self._casa_apoio(rei_preto, 'lado'),
                'desc': f'Mate: Dama em {chess.square_name(g7)}, Rei em {self._casa_apoio(rei_preto, "lado")}'
            })
        
        return posicoes
    
    def _casa_apoio(self, rei_preto, tipo):
        """Retorna casa adequada para o Rei branco apoiar o mate"""
        # Converte para coordenadas
        file = chess.square_file(rei_preto)
        rank = chess.square_rank(rei_preto)
        
        if tipo == 'cima':
            # Uma casa acima (ex: h8 -> h7)
            new_rank = rank - 1 if rank > 0 else rank + 1
            return chess.square(file, new_rank)
        elif tipo == 'diagonal':
            # Diagonal para dentro (ex: h8 -> g7)
            new_file = file - 1 if file > 0 else file + 1
            new_rank = rank - 1 if rank > 0 else rank + 1
            return chess.square(new_file, new_rank)
        else:  # lado
            # Lado (ex: h8 -> g8)
            new_file = file - 1 if file > 0 else file + 1
            return chess.square(new_file, rank)
    
    def gerar_posicao_base(self):
        """Gera uma posição base de xeque-mate válida"""
        # Embaralha as posições para variar
        random.shuffle(self.posicoes_base)
        
        for pos in self.posicoes_base:
            board = chess.Board()
            board.clear()
            
            # Coloca as peças
            board.set_piece_at(pos['rei'], chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(pos['dama'], chess.Piece(chess.QUEEN, chess.WHITE))
            board.set_piece_at(pos['rei_branco'], chess.Piece(chess.KING, chess.WHITE))
            
            # Verifica se é xeque-mate
            if board.is_checkmate():
                return board, pos['desc']
        
        # Fallback: posição garantida (mais segura)
        return self._posicao_garantida()
    
    def _posicao_garantida(self):
        """Posição de xeque-mate garantida"""
        board = chess.Board()
        board.clear()
        # Rei preto em h8, Dama em f7, Rei branco em h6
        board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.F7, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(chess.H6, chess.Piece(chess.KING, chess.WHITE))
        return board, "Mate garantido: Dama f7, Rei h6"
    
    def adicionar_pecas_inteligente(self, board, num_pecas):
        """
        Adiciona peças aleatórias de forma inteligente,
        tentando usar as novas peças para complementar o xeque
        """
        if num_pecas <= 0:
            return board, 0
        
        tipos = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        cores = [chess.WHITE, chess.BLACK]
        
        # Obtém casas vazias
        casas_vazias = [sq for sq in chess.SQUARES if board.piece_at(sq) is None]
        random.shuffle(casas_vazias)
        
        adicionadas = 0
        tentativas = 0
        max_tentativas = num_pecas * 50  # Mais tentativas
        
        # Identifica o Rei preto
        rei_preto = board.king(chess.BLACK)
        if rei_preto is None:
            return board, 0
        
        # Casas ao redor do Rei preto (onde podem ser colocadas peças para bloquear fuga)
        casas_vizinhas = []
        for dr in [-1, 0, 1]:
            for df in [-1, 0, 1]:
                if dr == 0 and df == 0:
                    continue
                file = chess.square_file(rei_preto) + df
                rank = chess.square_rank(rei_preto) + dr
                if 0 <= file < 8 and 0 <= rank < 8:
                    sq = chess.square(file, rank)
                    if board.piece_at(sq) is None:
                        casas_vizinhas.append(sq)
        
        # Embaralha as casas vizinhas para variar
        random.shuffle(casas_vizinhas)
        
        # Primeiro, tenta colocar peças perto do Rei preto para bloquear fugas
        for casa in casas_vizinhas:
            if adicionadas >= num_pecas:
                break
            
            # Tenta diferentes peças
            for _ in range(10):
                tipo = random.choice(tipos)
                cor = random.choice(cores)
                peca = chess.Piece(tipo, cor)
                
                backup = board.copy()
                board.set_piece_at(casa, peca)
                
                if board.is_checkmate() and not board.is_check():
                    adicionadas += 1
                    break
                else:
                    board = backup
        
        # Depois, tenta em outras casas
        for casa in casas_vazias:
            if adicionadas >= num_pecas:
                break
            
            tentativas += 1
            if tentativas > max_tentativas:
                break
            
            # Tenta diferentes combinações
            for _ in range(10):
                tipo = random.choice(tipos)
                cor = random.choice(cores)
                peca = chess.Piece(tipo, cor)
                
                backup = board.copy()
                board.set_piece_at(casa, peca)
                
                if board.is_checkmate() and not board.is_check():
                    adicionadas += 1
                    break
                else:
                    board = backup
        
        return board, adicionadas
    
    def gerar_todas_posicoes(self):
        """Gera todas as posições de xeque-mate"""
        #print(f"\n{'='*60}")
        #print(f"GERADOR DE XEQUE-MATE (VERSÃO AVANÇADA)")
        #print(f"{'='*60}")
        #print(f"Posições: {self.num_posicoes}")
        #print(f"Peças extras: {self.pecas_extras}")
        #print(f"{'='*60}\n")
        
        for i in range(self.num_posicoes):
            #print(f"\n--- Posição {i+1}/{self.num_posicoes} ---")
            
            # Gera posição base
            board, desc = self.gerar_posicao_base()
            #print(f"Base: {desc}")
            
            # Adiciona peças extras de forma inteligente
            board_final, adicionadas = self.adicionar_pecas_inteligente(board, self.pecas_extras)
            
            total_pecas = len(board_final.piece_map())
            e_mate = board_final.is_checkmate()
            
            # Verifica se é uma posição nova
            fen_atual = board_final.fen()
            if fen_atual in self.fens_gerados:
                #print("⚠ Posição repetida! Recriando...")
                # Tenta novamente
                board_final, adicionadas = self.adicionar_pecas_inteligente(board, self.pecas_extras)
                fen_atual = board_final.fen()
                e_mate = board_final.is_checkmate()
            
            self.fens_gerados.add(fen_atual)
            
            # Estatísticas
            self.estatisticas['total'] += 1
            if e_mate:
                self.estatisticas['validas'] += 1
                self.estatisticas['media_pecas'] += total_pecas
                self.estatisticas['min_pecas'] = min(self.estatisticas['min_pecas'], total_pecas)
                self.estatisticas['max_pecas'] = max(self.estatisticas['max_pecas'], total_pecas)
            
            # Salva posição
            posicao = {
                'id': i + 1,
                'fen': fen_atual,
                'descricao': f"{desc} (+{adicionadas} peças)",
                'num_pecas': total_pecas,
                'is_checkmate': e_mate,
                'timestamp': datetime.now().isoformat()
            }
            self.posicoes_geradas.append(posicao)
            
            # Mostra resultado
            status = "✓" if e_mate else "✗"
            #print(f"{status} Xeque-mate: {e_mate} | Peças: {total_pecas}")
            self._mostrar_tabuleiro(board_final)
            #print(f"FEN: {fen_atual}")
            
            # Se não for mate, mostra debug
            if not e_mate:
                rei_preto = board_final.king(chess.BLACK)
                if rei_preto:
                    #print(f"Rei preto em: {chess.square_name(rei_preto)}")
                    # Mostra casas de fuga
                    casas_fuga = []
                    for move in board_final.legal_moves:
                        if board_final.piece_at(move.to_square) and board_final.piece_at(move.to_square).piece_type == chess.KING:
                            casas_fuga.append(chess.square_name(move.to_square))
                    if casas_fuga:
                        #print(f"⚠ Casas de fuga: {', '.join(casas_fuga)}")
                    else:
                        #print("✓ Rei sem casas de fuga!")
        
        # Calcula médias
        if self.estatisticas['validas'] > 0:
            self.estatisticas['media_pecas'] /= self.estatisticas['validas']
        else:
            self.estatisticas['min_pecas'] = 0
        
        return self.posicoes_geradas, self.estatisticas
    
    def _mostrar_tabuleiro(self, board):
        """Mostra o tabuleiro formatado"""
        #print("  +---+---+---+---+---+---+---+---+")
        for rank in range(7, -1, -1):
            #print(f"{rank+1} |", end="")
            for file in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece:
                    symbol = piece.symbol()
                    if piece.color == chess.WHITE:
                        #print(f" {symbol} |", end="")
                    else:
                        #print(f" {symbol.lower()} |", end="")
                else:
                    #print("   |", end="")
            #print("\n  +---+---+---+---+---+---+---+---+")
        #print("    a   b   c   d   e   f   g   h")
    
    def salvar_stockfish(self, nome_arquivo=None):
        """Salva posições no formato Stockfish"""
        if nome_arquivo is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"xeques_mate_{timestamp}.txt"
        
        with open(nome_arquivo, 'w') as f:
            f.write("# POSIÇÕES DE XEQUE-MATE\n")
            f.write(f"# Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Formato: ID | FEN | Peças | Status\n")
            f.write("# " + "="*70 + "\n\n")
            
            for pos in self.posicoes_geradas:
                status = "MATE" if pos['is_checkmate'] else "INVALIDO"
                f.write(f"{pos['id']:3d} | {pos['fen']} | {pos['num_pecas']:2d} | {status}\n")
            
            # Comandos Stockfish
            f.write("\n" + "#"*70 + "\n")
            f.write("# COMANDOS STOCKFISH:\n")
            for pos in self.posicoes_geradas[:3]:
                if pos['is_checkmate']:
                    f.write(f"# position fen {pos['fen']}\n")
                    f.write(f"# go depth 20\n")
                    f.write("# ---\n")
        
        #print(f"\n✓ Salvo: {nome_arquivo}")
        return nome_arquivo
    
    def salvar_json(self, nome_arquivo=None):
        """Salva posições em JSON"""
        if nome_arquivo is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"xeques_mate_{timestamp}.json"
        
        dados = {
            'metadata': {
                'criado_em': datetime.now().isoformat(),
                'total': self.estatisticas['total'],
                'validas': self.estatisticas['validas'],
                'media_pecas': self.estatisticas['media_pecas'],
                'min_pecas': self.estatisticas['min_pecas'],
                'max_pecas': self.estatisticas['max_pecas']
            },
            'posicoes': self.posicoes_geradas
        }
        
        with open(nome_arquivo, 'w') as f:
            json.dump(dados, f, indent=2, ensure_ascii=False)
        
        #print(f"✓ Salvo: {nome_arquivo}")
        return nome_arquivo
    
    def salvar_apenas_validas(self, nome_arquivo=None):
        """Salva APENAS as posições válidas em formato compacto"""
        if nome_arquivo is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nome_arquivo = f"xeques_mate_validas_{timestamp}.txt"
        
        validas = [p for p in self.posicoes_geradas if p['is_checkmate']]
        
        with open(nome_arquivo, 'w') as f:
            f.write("# SOMENTE POSIÇÕES VÁLIDAS DE XEQUE-MATE\n")
            f.write(f"# Total: {len(validas)} posições\n")
            f.write("# " + "="*60 + "\n\n")
            
            for pos in validas:
                f.write(f"{pos['fen']}\n")
        
        #print(f"✓ Salvo (somente válidas): {nome_arquivo}")
        return nome_arquivo
    
    def resumo(self):
        """Mostra resumo final"""
        #print("\n" + "="*60)
        #print("RESUMO FINAL")
        #print("="*60)
        #print(f"Total: {self.estatisticas['total']}")
        #print(f"Válidas: {self.estatisticas['validas']}")
        if self.estatisticas['total'] > 0:
            #print(f"Taxa: {self.estatisticas['validas']/self.estatisticas['total']*100:.1f}%")
        #print(f"Média de peças: {self.estatisticas['media_pecas']:.1f}")
        #print(f"Mínimo: {self.estatisticas['min_pecas']}")
        #print(f"Máximo: {self.estatisticas['max_pecas']}")
        #print("="*60)


# ============================================
# EXECUÇÃO PRINCIPAL
# ============================================

if __name__ == "__main__":
    # CONFIGURAÇÕES
    NUM_POSICOES = 10
    PECAS_EXTRAS = 5
    
    # Cria gerador
    gerador = GeradorXequeMateAvancado(NUM_POSICOES, PECAS_EXTRAS)
    
    # Gera posições
    posicoes, stats = gerador.gerar_todas_posicoes()
    
    # Salva em diferentes formatos
    #print("\n" + "="*60)
    #print("SALVANDO ARQUIVOS...")
    #print("="*60)
    
    arquivo_txt = gerador.salvar_stockfish()
    arquivo_json = gerador.salvar_json()
    arquivo_validas = gerador.salvar_apenas_validas()
    
    # Mostra resumo
    gerador.resumo()
    
    # Mostra exemplo de uso
    #print("\n" + "="*60)
    #print("COMO USAR COM STOCKFISH")
    #print("="*60)
    if posicoes and any(p['is_checkmate'] for p in posicoes):
        primeira_valida = next(p for p in posicoes if p['is_checkmate'])
        #print(f"stockfish")
        #print(f"position fen {primeira_valida['fen']}")
        #print(f"go depth 20")
    
    #print("\n✓ CONCLUÍDO!")