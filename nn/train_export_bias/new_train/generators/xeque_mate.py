import chess
import random
import json
from datetime import datetime

class GeradorXequeMateAvancado:
    """Gerador avançado de posições de xeque-mate com peças aleatórias"""
    
    def __init__(self, num_posicoes=10, pecas_extras=5):
        self.num_posicoes = num_posicoes
        self.pecas_extras = pecas_extras
        self.posicoes_geradas = []
        self.estatisticas = {
            'total': 0,
            'validas': 0,
            'media_pecas': 0.0,
            'min_pecas': float('inf'),
            'max_pecas': 0
        }
        self.fens_gerados = set()  # Para evitar repetições
        
        # Posições base de XEQUE-MATE GARANTIDAS (todas válidas)
        self.posicoes_base = self._gerar_posicoes_base_variadas()
    
    def _gerar_posicoes_base_variadas(self):
        """Gera uma lista grande de posições base variadas"""
        posicoes = []
        cantos = [
            (chess.H8, chess.H7, chess.G7, chess.F7, chess.G8, chess.F8),
            (chess.A8, chess.A7, chess.B7, chess.C7, chess.B8, chess.C8),
            (chess.H1, chess.H2, chess.G2, chess.F2, chess.G1, chess.F1),
            (chess.A1, chess.A2, chess.B2, chess.C2, chess.B1, chess.C1),
        ]
        
        for rei_preto, h7, g7, f7, g8, f8 in cantos:
            # Variação 1: Dama em g7, Rei em cima
            posicoes.append({
                'rei': rei_preto,
                'dama': g7,
                'rei_branco': self._casa_apoio(rei_preto, 'cima'),
                'desc': f'Mate: Dama em {chess.square_name(g7)}, Rei em {self._casa_apoio(rei_preto, "cima")}'
            })
            
            # Variação 2: Dama em f7, Rei em diagonal
            posicoes.append({
                'rei': rei_preto,
                'dama': f7,
                'rei_branco': self._casa_apoio(rei_preto, 'diagonal'),
                'desc': f'Mate: Dama em {chess.square_name(f7)}, Rei em {self._casa_apoio(rei_preto, "diagonal")}'
            })
            
            # Variação 3: Dama em g7, Rei ao lado
            posicoes.append({
                'rei': rei_preto,
                'dama': g7,
                'rei_branco': self._casa_apoio(rei_preto, 'lado'),
                'desc': f'Mate: Dama em {chess.square_name(g7)}, Rei em {self._casa_apoio(rei_preto, "lado")}'
            })
        
        return posicoes
    
    def _casa_apoio(self, rei_preto, tipo):
        """Retorna casa adequada para o Rei branco apoiar o mate"""
        file = chess.square_file(rei_preto)
        rank = chess.square_rank(rei_preto)
        
        if tipo == 'cima':
            new_rank = rank - 1 if rank > 0 else rank + 1
            return chess.square(file, new_rank)
        elif tipo == 'diagonal':
            new_file = file - 1 if file > 0 else file + 1
            new_rank = rank - 1 if rank > 0 else rank + 1
            return chess.square(new_file, new_rank)
        else:  # lado
            new_file = file - 1 if file > 0 else file + 1
            return chess.square(new_file, rank)
    
    def gerar_posicao_base(self):
        """Gera uma posição base de xeque-mate válida"""
        random.shuffle(self.posicoes_base)
        
        for pos in self.posicoes_base:
            board = chess.Board()
            board.clear()
            
            board.set_piece_at(pos['rei'], chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(pos['dama'], chess.Piece(chess.QUEEN, chess.WHITE))
            board.set_piece_at(pos['rei_branco'], chess.Piece(chess.KING, chess.WHITE))
            
            if board.is_checkmate():
                return board, pos['desc']
        
        return self._posicao_garantida()
    
    def _posicao_garantida(self):
        """Posição de xeque-mate garantida (fallback)"""
        board = chess.Board()
        board.clear()
        board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(chess.G7, chess.Piece(chess.QUEEN, chess.WHITE))  # Dama em g7 (controla h8)
        board.set_piece_at(chess.H6, chess.Piece(chess.KING, chess.WHITE))
        return board, "Mate garantido: Dama g7, Rei h6"
    
    def adicionar_pecas_inteligente(self, board, num_pecas):
        """Adiciona peças aleatórias de forma inteligente, tentando manter mate"""
        if num_pecas <= 0:
            return board, 0
        
        tipos = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        cores = [chess.WHITE, chess.BLACK]
        
        casas_vazias = [sq for sq in chess.SQUARES if board.piece_at(sq) is None]
        random.shuffle(casas_vazias)
        
        adicionadas = 0
        tentativas = 0
        max_tentativas = num_pecas * 50
        
        rei_preto = board.king(chess.BLACK)
        if rei_preto is None:
            return board, 0
        
        # Casas ao redor do Rei preto (prioridade para bloquear fugas)
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
        
        random.shuffle(casas_vizinhas)
        
        # Primeiro, tenta colocar peças perto do Rei preto
        for casa in casas_vizinhas:
            if adicionadas >= num_pecas:
                break
            
            for _ in range(10):
                tipo = random.choice(tipos)
                cor = random.choice(cores)
                peca = chess.Piece(tipo, cor)
                
                backup = board.copy()
                board.set_piece_at(casa, peca)
                
                # Verifica se mantém o xeque-mate e o Rei está em xeque
                if board.is_checkmate() and board.is_check():
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
            
            for _ in range(10):
                tipo = random.choice(tipos)
                cor = random.choice(cores)
                peca = chess.Piece(tipo, cor)
                
                backup = board.copy()
                board.set_piece_at(casa, peca)
                
                if board.is_checkmate() and board.is_check():
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
            
            # Adiciona peças extras
            board_final, adicionadas = self.adicionar_pecas_inteligente(board, self.pecas_extras)
            
            total_pecas = len(board_final.piece_map())
            e_mate = board_final.is_checkmate()
            em_xeque = board_final.is_check()
            
            fen_atual = board_final.fen()
            
            # Verifica se é repetida
            if fen_atual in self.fens_gerados:
                #print("⚠ Posição repetida! Regenerando...")
                # Tenta novamente com uma base diferente
                board, desc = self.gerar_posicao_base()
                board_final, adicionadas = self.adicionar_pecas_inteligente(board, self.pecas_extras)
                fen_atual = board_final.fen()
                e_mate = board_final.is_checkmate()
                em_xeque = board_final.is_check()
                total_pecas = len(board_final.piece_map())
            
            self.fens_gerados.add(fen_atual)
            
            # Atualiza estatísticas APENAS se for mate válido
            if e_mate and em_xeque:
                self.estatisticas['total'] += 1
                self.estatisticas['validas'] += 1
                self.estatisticas['media_pecas'] += total_pecas
                self.estatisticas['min_pecas'] = min(self.estatisticas['min_pecas'], total_pecas)
                self.estatisticas['max_pecas'] = max(self.estatisticas['max_pecas'], total_pecas)
                
                posicao = {
                    'id': i + 1,
                    'fen': fen_atual,
                    'descricao': f"{desc} (+{adicionadas} peças)",
                    'num_pecas': total_pecas,
                    'is_checkmate': True,
                    'is_check': True,
                    'timestamp': datetime.now().isoformat()
                }
                self.posicoes_geradas.append(posicao)
                
                # Mostra resultado
                #print(f"✅ XEQUE-MATE VÁLIDO | Peças: {total_pecas}")
            else:
                #print(f"❌ Posição inválida (não é mate) - ignorando")
                # Mostra diagnóstico
                rei_preto = board_final.king(chess.BLACK)
                if rei_preto:
                    #print(f"Rei preto em: {chess.square_name(rei_preto)}")
                    #print(f"Em xeque: {em_xeque}")
                    if not em_xeque:
                        #print("❌ Rei NÃO está em xeque!")
                    else:
                        #print("✅ Rei está em xeque, mas não é mate")
            
            # Mostra o tabuleiro sempre
            self._mostrar_tabuleiro(board_final)
            #print(f"FEN: {fen_atual}")
            
            # Mostra análise detalhada
            rei_preto = board_final.king(chess.BLACK)
            if rei_preto:
                atacantes = board_final.attackers(chess.WHITE, rei_preto)
                if atacantes:
                    #print(f"Atacado por: {', '.join(chess.square_name(sq) for sq in atacantes)}")
                
                # Mostra casas de fuga
                casas_fuga = []
                for move in board_final.legal_moves:
                    if board_final.piece_at(move.to_square) and board_final.piece_at(move.to_square).piece_type == chess.KING:
                        casas_fuga.append(chess.square_name(move.to_square))
                
                if casas_fuga:
                    #print(f"⚠ Casas de fuga: {', '.join(casas_fuga)}")
                else:
                    #print("✅ Sem casas de fuga!")
        
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
            f.write("# POSIÇÕES DE XEQUE-MATE VÁLIDAS\n")
            f.write(f"# Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# Formato: ID | FEN | Peças | Status\n")
            f.write("# " + "="*70 + "\n\n")
            
            for pos in self.posicoes_geradas:
                status = "MATE ✓" if pos['is_checkmate'] else "INVALIDO ✗"
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
        #print(f"Total de posições válidas geradas: {self.estatisticas['validas']}")
        #print(f"Média de peças: {self.estatisticas['media_pecas']:.1f}")
        #print(f"Mínimo de peças: {self.estatisticas['min_pecas'] if self.estatisticas['min_pecas'] != float('inf') else 0}")
        #print(f"Máximo de peças: {self.estatisticas['max_pecas']}")
        #print("="*60)


# ============================================
# EXECUÇÃO PRINCIPAL
# ============================================

if __name__ == "__main__":
    # CONFIGURAÇÕES
    NUM_POSICOES = 10
    PECAS_EXTRAS = 5
    
    #print("="*60)
    #print("GERADOR DE XEQUE-MATE")
    #print("="*60)
    #print(f"Posições: {NUM_POSICOES}")
    #print(f"Peças extras: {PECAS_EXTRAS}")
    #print("="*60)
    
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
    
    #print("\n✓ CONCLUÍDO!")