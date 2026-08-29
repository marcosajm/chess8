#!/usr/bin/env python3
"""
Chess Game Visualizer - View and navigate generated games
Supports both forward and reverse game navigation
"""

import chess
import chess.svg
import numpy as np
import struct
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cairosvg
import io
import os
import json
from pathlib import Path
from typing import List, Optional, Tuple
import webbrowser

# ============== Game Loader ==============
class GameLoader:
    """Load and parse chess games from binary training data"""
    
    @staticmethod
    def load_games(filename: str, max_games: Optional[int] = None) -> List[List[dict]]:
        """Load games from binary file"""
        #print(f"Loading games from {filename}...")
        
        games = []
        current_game = []
        last_fen = None
        game_count = 0
        
        try:
            with open(filename, 'rb') as f:
                magic, count = struct.unpack('4sI', f.read(8))
                if magic != b'NNUE':
                    #print(f"Invalid file format: {magic}")
                    return []
                
                #print(f"Found {count} positions in file")
                
                # Read all positions
                for i in range(count):
                    if i % 50000 == 0 and i > 0:
                        #print(f"  Processed {i} positions...")
                    
                    # Read features
                    feat_data = f.read(780 * 4)
                    if len(feat_data) < 780 * 4:
                        break
                    
                    features = np.frombuffer(feat_data, dtype=np.float32)
                    
                    # Read metadata
                    score, result, tactical = struct.unpack('fff', f.read(12))
                    
                    # Reconstruct FEN from features (simplified)
                    fen = GameLoader.features_to_fen(features)
                    
                    # Check if this is a new game (based on position change)
                    if last_fen is not None and fen != last_fen:
                        # If current_game has content, save it as a game
                        if len(current_game) > 2:
                            games.append(current_game)
                            game_count += 1
                            
                            # Limit games
                            if max_games and game_count >= max_games:
                                break
                            
                            # Reset for next game
                            current_game = []
                    
                    # Add position to current game
                    current_game.append({
                        'fen': fen,
                        'score': score,
                        'result': result,
                        'tactical': tactical,
                        'features': features,
                        'move_number': len(current_game) + 1
                    })
                    
                    last_fen = fen
                
                # Add the last game
                if len(current_game) > 2:
                    games.append(current_game)
                    game_count += 1
                
        except Exception as e:
            #print(f"Error loading file: {e}")
            return []
        
        #print(f"Loaded {len(games)} games from {filename}")
        return games
    
    @staticmethod
    def features_to_fen(features: np.ndarray) -> str:
        """Convert features back to FEN (simplified reconstruction)"""
        # This is a simplified reconstruction - for exact FEN you'd need
        # to store the FEN string in the data file
        
        # For now, generate a pseudo-FEN from piece positions
        piece_symbols = {0: 'P', 1: 'N', 2: 'B', 3: 'R', 4: 'Q', 5: 'K'}
        
        # Create an 8x8 board
        board_2d = [[''] * 8 for _ in range(8)]
        
        # Extract pieces from features (simplified)
        for square in range(64):
            for color in [0, 1]:  # 0=white, 1=black
                for piece_type in range(6):
                    idx = color * (6 * 64) + piece_type * 64 + square
                    if idx < len(features) and features[idx] > 0.5:
                        piece = piece_symbols[piece_type]
                        if color == 1:  # Black
                            piece = piece.lower()
                        row = 7 - (square // 8)
                        col = square % 8
                        board_2d[row][col] = piece
        
        # Convert to FEN
        fen_parts = []
        for row in board_2d:
            empty = 0
            row_str = ''
            for cell in row:
                if cell:
                    if empty > 0:
                        row_str += str(empty)
                        empty = 0
                    row_str += cell
                else:
                    empty += 1
            if empty > 0:
                row_str += str(empty)
            fen_parts.append(row_str)
        
        fen = '/'.join(fen_parts)
        fen += " w - - 0 1"  # Default to white to move
        
        return fen

# ============== Game Visualizer ==============
class ChessGameVisualizer:
    def __init__(self, root, games: List[List[dict]], title: str = "Chess Game Visualizer"):
        self.root = root
        self.games = games
        self.current_game_index = 0
        self.current_move_index = 0
        self.is_forward = True  # True = forward, False = reverse
        self.is_playing = False
        self.play_delay = 1000  # ms between moves
        
        self.root.title(title)
        self.root.geometry("1000x750")
        self.root.configure(bg='#2b2b2b')
        
        # Setup UI
        self.setup_ui()
        
        # Display first game
        if games:
            self.display_game(0)
    
    def setup_ui(self):
        """Setup the user interface"""
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', padding=6, font=('Arial', 10))
        style.configure('TLabel', font=('Arial', 10))
        
        # Canvas for chess board
        self.canvas = tk.Canvas(main_frame, width=600, height=600, bg='#f0d9b5', highlightthickness=2, highlightbackground='#555')
        self.canvas.pack(side=tk.TOP, pady=10)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        # Navigation buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.LEFT, padx=10)
        
        self.btn_first = ttk.Button(btn_frame, text="⏮ First", command=self.first_move, width=8)
        self.btn_first.pack(side=tk.LEFT, padx=2)
        
        self.btn_prev = ttk.Button(btn_frame, text="◄ Prev", command=self.prev_move, width=8)
        self.btn_prev.pack(side=tk.LEFT, padx=2)
        
        self.btn_play = ttk.Button(btn_frame, text="▶ Play", command=self.toggle_play, width=8)
        self.btn_play.pack(side=tk.LEFT, padx=2)
        
        self.btn_next = ttk.Button(btn_frame, text="Next ►", command=self.next_move, width=8)
        self.btn_next.pack(side=tk.LEFT, padx=2)
        
        self.btn_last = ttk.Button(btn_frame, text="⏭ Last", command=self.last_move, width=8)
        self.btn_last.pack(side=tk.LEFT, padx=2)
        
        # Direction toggle
        dir_frame = ttk.Frame(control_frame)
        dir_frame.pack(side=tk.LEFT, padx=20)
        
        self.dir_label = ttk.Label(dir_frame, text="Mode:", font=('Arial', 10, 'bold'))
        self.dir_label.pack(side=tk.LEFT, padx=5)
        
        self.btn_forward = ttk.Button(dir_frame, text="→ Forward", command=self.set_forward_mode, width=10)
        self.btn_forward.pack(side=tk.LEFT, padx=2)
        
        self.btn_reverse = ttk.Button(dir_frame, text="← Reverse", command=self.set_reverse_mode, width=10)
        self.btn_reverse.pack(side=tk.LEFT, padx=2)
        
        # Speed control
        speed_frame = ttk.Frame(control_frame)
        speed_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT, padx=5)
        
        self.speed_var = tk.IntVar(value=1000)
        self.speed_scale = ttk.Scale(speed_frame, from_=200, to=3000, orient=tk.HORIZONTAL,
                                     variable=self.speed_var, length=150,
                                     command=self.update_speed)
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        
        self.speed_label = ttk.Label(speed_frame, text="1.0s", width=5)
        self.speed_label.pack(side=tk.LEFT, padx=5)
        
        # Info panel
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        # Game info
        self.game_label = ttk.Label(info_frame, text="Game 0/0", font=('Arial', 12, 'bold'))
        self.game_label.pack(side=tk.LEFT, padx=10)
        
        self.move_label = ttk.Label(info_frame, text="Move 0/0", font=('Arial', 12))
        self.move_label.pack(side=tk.LEFT, padx=20)
        
        self.score_label = ttk.Label(info_frame, text="Score: 0.00", font=('Arial', 12))
        self.score_label.pack(side=tk.LEFT, padx=20)
        
        # Game list
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Label(list_frame, text="Game:").pack(side=tk.LEFT, padx=5)
        
        self.game_combo = ttk.Combobox(list_frame, state='readonly', width=20)
        self.game_combo.pack(side=tk.LEFT, padx=5)
        self.game_combo.bind('<<ComboboxSelected>>', self.on_game_select)
        
        # Update game list
        if self.games:
            game_list = [f"Game {i+1} ({len(game)} moves)" for i, game in enumerate(self.games)]
            self.game_combo['values'] = game_list
            if game_list:
                self.game_combo.current(0)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def get_current_game(self) -> Optional[List[dict]]:
        """Get the current game"""
        if 0 <= self.current_game_index < len(self.games):
            return self.games[self.current_game_index]
        return None
    
    def get_current_position(self) -> Optional[dict]:
        """Get the current position in the current game"""
        game = self.get_current_game()
        if not game:
            return None
        
        if self.is_forward:
            if 0 <= self.current_move_index < len(game):
                return game[self.current_move_index]
        else:
            if 0 <= self.current_move_index < len(game):
                return game[len(game) - 1 - self.current_move_index]
        
        return None
    
    def display_game(self, game_index: int):
        """Display a specific game"""
        if not self.games:
            return
        
        self.current_game_index = game_index % len(self.games)
        self.current_move_index = 0
        self.update_display()
    
    def update_display(self):
        """Update the display with current position"""
        game = self.get_current_game()
        if not game:
            return
        
        position = self.get_current_position()
        if not position:
            return
        
        # Update info
        total_moves = len(game)
        move_number = self.current_move_index + 1
        
        self.game_label.config(text=f"Game {self.current_game_index + 1}/{len(self.games)}")
        
        if self.is_forward:
            self.move_label.config(text=f"Move {move_number}/{total_moves}")
        else:
            self.move_label.config(text=f"Move {total_moves - self.current_move_index}/{total_moves} (reverse)")
        
        self.score_label.config(text=f"Score: {position.get('score', 0):.3f}")
        
        # Update status
        status_text = f"Game {self.current_game_index + 1} - "
        status_text += f"Position {move_number}/{total_moves} - "
        status_text += f"Score: {position.get('score', 0):.3f}"
        if 'result' in position and position['result'] > 0:
            status_text += f" - Result: {position['result']:.1f}"
        self.status_var.set(status_text)
        
        # Update buttons
        if self.is_forward:
            self.btn_prev.config(state='normal' if move_number > 1 else 'disabled')
            self.btn_next.config(state='normal' if move_number < total_moves else 'disabled')
            self.btn_first.config(state='normal' if move_number > 1 else 'disabled')
            self.btn_last.config(state='normal' if move_number < total_moves else 'disabled')
        else:
            self.btn_prev.config(state='normal' if move_number < total_moves else 'disabled')
            self.btn_next.config(state='normal' if move_number > 1 else 'disabled')
            self.btn_first.config(state='normal' if move_number < total_moves else 'disabled')
            self.btn_last.config(state='normal' if move_number > 1 else 'disabled')
        
        # Draw the board
        self.draw_board(position.get('fen', ''))
    
    def draw_board(self, fen: str):
        """Draw the chess board"""
        try:
            board = chess.Board(fen) if fen else chess.Board()
            
            # Generate SVG
            svg = chess.svg.board(board=board, size=600, lastmove=board.peek() if board.move_stack else None)
            
            # Convert SVG to PNG using cairosvg
            png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
            
            # Convert to ImageTk
            image = Image.open(io.BytesIO(png_data))
            photo = ImageTk.PhotoImage(image)
            
            # Display on canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor='nw', image=photo)
            self.canvas.image = photo  # Keep reference
            
        except Exception as e:
            #print(f"Error drawing board: {e}")
            self.canvas.delete("all")
            self.canvas.create_text(300, 300, text=f"Error loading position\n{fen}", 
                                   font=('Arial', 20), fill='red')
    
    # Navigation methods
    def first_move(self):
        self.current_move_index = 0
        self.update_display()
    
    def last_move(self):
        game = self.get_current_game()
        if game:
            self.current_move_index = len(game) - 1
        self.update_display()
    
    def prev_move(self):
        if self.is_forward:
            if self.current_move_index > 0:
                self.current_move_index -= 1
        else:
            game = self.get_current_game()
            if game and self.current_move_index < len(game) - 1:
                self.current_move_index += 1
        self.update_display()
    
    def next_move(self):
        if self.is_forward:
            game = self.get_current_game()
            if game and self.current_move_index < len(game) - 1:
                self.current_move_index += 1
        else:
            if self.current_move_index > 0:
                self.current_move_index -= 1
        self.update_display()
    
    def set_forward_mode(self):
        self.is_forward = True
        self.current_move_index = 0
        self.btn_forward.config(relief=tk.SUNKEN)
        self.btn_reverse.config(relief=tk.RAISED)
        self.update_display()
    
    def set_reverse_mode(self):
        self.is_forward = False
        self.current_move_index = 0
        self.btn_reverse.config(relief=tk.SUNKEN)
        self.btn_forward.config(relief=tk.RAISED)
        self.update_display()
    
    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.config(text="⏸ Pause")
            self.play_animation()
        else:
            self.btn_play.config(text="▶ Play")
    
    def play_animation(self):
        if not self.is_playing:
            return
        
        # Check if we reached the end
        game = self.get_current_game()
        if not game:
            self.toggle_play()
            return
        
        if self.is_forward:
            if self.current_move_index >= len(game) - 1:
                self.toggle_play()
                return
            self.current_move_index += 1
        else:
            if self.current_move_index >= len(game) - 1:
                self.toggle_play()
                return
            self.current_move_index += 1
        
        self.update_display()
        
        # Schedule next move
        self.root.after(self.play_delay, self.play_animation)
    
    def update_speed(self, value):
        """Update playback speed"""
        speed = self.speed_var.get()
        self.play_delay = speed
        self.speed_label.config(text=f"{speed/1000:.1f}s")
    
    def on_game_select(self, event):
        """Handle game selection from combobox"""
        index = self.game_combo.current()
        if index >= 0:
            self.display_game(index)

# ============== Main Application ==============
class ChessGameViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Game Viewer")
        self.root.geometry("800x600")
        
        # Menu
        menubar = tk.Menu(root)
        root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Data File...", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Export Game as PGN", command=self.export_pgn)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)
        
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Game Info", command=self.show_game_info)
        view_menu.add_command(label="Position Statistics", command=self.show_position_stats)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
        # Main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Label
        ttk.Label(main_frame, text="Load a training data file to view games", 
                 font=('Arial', 14)).pack(expand=True)
        
        ttk.Button(main_frame, text="Open Data File", command=self.open_file,
                  width=20).pack(pady=20)
        
        self.visualizer = None
        self.current_file = None
    
    def open_file(self):
        """Open a data file"""
        filename = filedialog.askopenfilename(
            title="Select Training Data File",
            filetypes=[("Binary files", "*.bin"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            # Load games
            games = GameLoader.load_games(filename, max_games=500)
            
            if not games:
                messagebox.showerror("Error", f"No games found in {filename}")
                return
            
            # Create visualizer
            self.current_file = filename
            
            # Clear main frame
            for widget in self.root.winfo_children():
                if widget != self.root.menu:  # Keep menu
                    widget.destroy()
            
            # Create visualizer
            self.visualizer = ChessGameVisualizer(self.root, games, 
                                                  f"Chess Game Viewer - {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def export_pgn(self):
        """Export current game as PGN"""
        if not self.visualizer or not self.visualizer.games:
            messagebox.showinfo("Info", "No game loaded")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save PGN",
            defaultextension=".pgn",
            filetypes=[("PGN files", "*.pgn"), ("All files", "*.*")]
        )
        
        if not filename:
            return
        
        try:
            game = self.visualizer.get_current_game()
            if not game:
                return
            
            # Reconstruct game moves from FENs
            moves = []
            last_board = None
            
            for pos in game:
                fen = pos.get('fen', '')
                if fen:
                    board = chess.Board(fen)
                    if last_board:
                        # Try to find the move
                        for move in last_board.legal_moves:
                            test_board = last_board.copy()
                            test_board.push(move)
                            if test_board.fen() == board.fen():
                                moves.append(move.uci())
                                break
                    last_board = board
            
            # Write PGN
            with open(filename, 'w') as f:
                f.write('[Event "Generated Game"]\n')
                f.write(f'[Site "NNUE Training"]\n')
                f.write(f'[Date "{time.strftime("%Y.%m.%d")}"]\n')
                f.write(f'[Round "{self.visualizer.current_game_index + 1}"]\n')
                f.write('[White "NNUE"]\n')
                f.write('[Black "NNUE"]\n')
                f.write(f'[Result "*"]\n')
                f.write(f'[PlyCount "{len(moves)}"]\n\n')
                
                # Write moves in PGN format
                for i, move in enumerate(moves):
                    if i % 2 == 0:
                        f.write(f"{i//2 + 1}. ")
                    f.write(f"{move} ")
                f.write("*\n")
            
            messagebox.showinfo("Success", f"PGN saved to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export PGN:\n{str(e)}")
    
    def show_game_info(self):
        """Show information about the current game"""
        if not self.visualizer or not self.visualizer.games:
            return
        
        game = self.visualizer.get_current_game()
        if not game:
            return
        
        info = f"Game {self.visualizer.current_game_index + 1} of {len(self.visualizer.games)}\n"
        info += f"Total moves: {len(game)}\n"
        
        if game:
            scores = [pos.get('score', 0) for pos in game]
            info += f"Score range: {min(scores):.3f} to {max(scores):.3f}\n"
            info += f"Average score: {sum(scores)/len(scores):.3f}\n"
            
            if 'result' in game[-1]:
                info += f"Game result: {game[-1]['result']:.1f}\n"
            
            # Count tactical positions
            tactical_count = sum(1 for pos in game if pos.get('tactical', 0) != 0)
            info += f"Tactical positions: {tactical_count}/{len(game)}"
        
        messagebox.showinfo("Game Information", info)
    
    def show_position_stats(self):
        """Show statistics about positions"""
        if not self.visualizer or not self.visualizer.games:
            return
        
        total_games = len(self.visualizer.games)
        total_positions = sum(len(game) for game in self.visualizer.games)
        
        stats = f"Total games: {total_games}\n"
        stats += f"Total positions: {total_positions}\n"
        stats += f"Average positions per game: {total_positions/total_games:.1f}\n\n"
        
        stats += "Score distribution:\n"
        all_scores = [pos.get('score', 0) for game in self.visualizer.games for pos in game]
        if all_scores:
            stats += f"  Min: {min(all_scores):.3f}\n"
            stats += f"  Max: {max(all_scores):.3f}\n"
            stats += f"  Mean: {sum(all_scores)/len(all_scores):.3f}\n"
            
            # Score buckets
            buckets = [-1.0, -0.5, 0.0, 0.5, 1.0]
            stats += "  Distribution:\n"
            for i in range(len(buckets)-1):
                count = sum(1 for s in all_scores if buckets[i] <= s < buckets[i+1])
                pct = count/len(all_scores)*100
                stats += f"    [{buckets[i]:.1f}, {buckets[i+1]:.1f}]: {count:4d} ({pct:.1f}%)\n"
        
        messagebox.showinfo("Position Statistics", stats)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """Chess Game Visualizer

View games generated from NNUE training data.

Features:
- Navigate forward and reverse
- Playback animation
- Export to PGN
- Game statistics

Created for NNUE Training Pipeline
"""
        messagebox.showinfo("About", about_text)

# ============== Main ==============
def main():
    root = tk.Tk()
    app = ChessGameViewerApp(root)
    root.mainloop()

if __name__ == "__main__":
    # Check dependencies
    try:
        import cairosvg
        import PIL
    except ImportError:
        #print("Installing required dependencies...")
        #print("Run: pip install cairosvg pillow")
        #print("On Ubuntu: sudo apt-get install libcairo2-dev")
    
    main()