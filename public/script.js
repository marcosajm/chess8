// script.js
(async () => {
  const Module = await ChessModule();

  // ---- WASM wrappers ----
  const init_board_wasm  = Module.cwrap('init_board_wasm', null, []);
  const get_board_ptr    = Module.cwrap('get_board_ptr_wasm', 'number', []);
  const make_move_wasm   = Module.cwrap('make_move_wasm', 'number', ['number', 'number']);
  const get_turn_wasm    = Module.cwrap('get_current_turn_wasm', 'number', []);
  const ai_move_wasm     = Module.cwrap('find_ai_move_wasm_depth', 'number', ['number', 'number']); // packed from|(to<<8)
  const promote_wasm     = Module.cwrap('promote_pawn_wasm', null, ['number','number']);
  const pending_wasm     = Module.cwrap('get_pawn_promotion_pending_index_wasm', 'number', []);
  const game_state_wasm  = Module.cwrap('get_game_state_wasm', 'number', []);
  const evaluate_wasm    = Module.cwrap('evaluate_board', 'number', []);
  const set_difficulty   = typeof Module._set_difficulty_wasm === 'function'
    ? Module.cwrap('set_difficulty_wasm', 'number', ['number'])
    : null;

  // ---- UI ----
  const boardEl = document.getElementById('board');
  const statusEl = document.getElementById('status');
  const sideSel = document.getElementById('side');
  const aiSel = document.getElementById('ai');
  const depthSel = document.getElementById('depth');
  const difficultySel = document.getElementById('difficulty');
  const promoMask = document.getElementById('promotionModalOverlay');
  const undoBtn = document.getElementById('undo');
  const hintBtn = document.getElementById('hint');
  const flipBtn = document.getElementById('flip');
  const newBtn = document.getElementById('newGame');
  const gameOverModal = document.getElementById('gameOverModalOverlay');
  const gameOverTitle        = document.getElementById('gameOverTitle');
  const gameOverMessage      = document.getElementById('gameOverMessage');
  const gameOverResetButton  = document.getElementById('gameOverResetButton');

  const promoteQueenBtn  = document.getElementById('promoteQueen');
  const promoteRookBtn   = document.getElementById('promoteRook');
  const promoteBishopBtn = document.getElementById('promoteBishop');
  const promoteKnightBtn = document.getElementById('promoteKnight');

  const moveStack = [];
  let selected = null;
  let flipped = false;
  let openingBookMovesPlayed = 0;
  let openingRandomState = 0;
  let selectedOpening = null;
  let previousOpeningIndex = -1;

  const openingBook = [
    { moves: ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5'], weight: 30 }, // Ruy Lopez
    { moves: ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4'], weight: 25 }, // Italian Game
    { moves: ['e2e4', 'e7e5', 'g1f3', 'd7d5', 'e4d5'], weight: 15 }, // Scotch Game
    { moves: ['e2e4', 'e7e5', 'g1f3', 'g8f6', 'f3e5'], weight: 10 }, // Petrov Defense
    { moves: ['e2e4', 'e7e5', 'g1f3', 'd7d6', 'd2d4'], weight: 10 }, // Philidor Defense
    { moves: ['d2d4', 'd7d5', 'c2c4', 'e7e6', 'b1c3'], weight: 15 }, // Queen's Gambit
    { moves: ['e2e4', 'c7c5', 'g1f3', 'd7d6', 'd2d4'], weight: 15 }  // Sicilian Defense
  ];

  function reseedOpeningRandom() {
    const seedSource = new Uint32Array(1);
    if (globalThis.crypto && globalThis.crypto.getRandomValues) {
      globalThis.crypto.getRandomValues(seedSource);
      openingRandomState = seedSource[0];
    } else {
      openingRandomState = (Date.now() ^ (performance.now() * 1000)) >>> 0;
    }
    if (openingRandomState === 0) openingRandomState = 0x6d2b79f5;
  }

  function openingRandom() {
    // xorshift32: small, fast PRNG suitable for book move selection.
    openingRandomState ^= openingRandomState << 13;
    openingRandomState ^= openingRandomState >>> 17;
    openingRandomState ^= openingRandomState << 5;
    return (openingRandomState >>> 0) / 0x100000000;
  }

  function chooseOpeningLine(played) {
    const candidates = openingBook
      .map((line, index) => ({ line, index }))
      .filter(({ line, index }) =>
        played.length < line.moves.length &&
          played.every((move, moveIndex) => move === line.moves[moveIndex]));
      if (candidates.length > 1 && previousOpeningIndex >= 0) {
        const alternatives = candidates.filter(({ index }) => index !== previousOpeningIndex);
        if (alternatives.length) candidates.splice(0, candidates.length, ...alternatives);
      }
    if (!candidates.length) return null;
    const totalWeight = candidates.reduce((sum, item) => sum + item.line.weight, 0);
    let threshold = openingRandom() * totalWeight;

    for (const candidate of candidates) {
      threshold -= candidate.line.weight;
      if (threshold < 0) {
        previousOpeningIndex = candidate.index;
        console.info(`[BOOK] Selected line ${candidate.index + 1}: ${candidate.line.moves.join(' ')}`);
        return candidate.line.moves;
      }
    }

    const fallback = candidates[candidates.length - 1];
    previousOpeningIndex = fallback.index;
    console.info(`[BOOK] Selected line ${fallback.index + 1}: ${fallback.line.moves.join(' ')}`);
    return fallback.line.moves;
  }

  function moveToBookNotation(from, to) {
    return `${squareLabel(from)}${squareLabel(to)}`;
  }

  function getOpeningBookMove() {
    if (openingBookMovesPlayed >= 2) return null;

    const played = moveStack.map(move => moveToBookNotation(move.from, move.to));
    if (!selectedOpening) {
      selectedOpening = chooseOpeningLine(played);
    }
    if (!selectedOpening) return null;

    if (played.length >= selectedOpening.length ||
        !played.every((move, index) => move === selectedOpening[index])) {
      return null;
    }

    return selectedOpening[played.length];
  }

  function parseBookMove(notation) {
    return {
      from: frToIdx(notation.charCodeAt(0) - 97, Number(notation[1]) - 1),
      to: frToIdx(notation.charCodeAt(2) - 97, Number(notation[3]) - 1)
    };
  }

  // White piece codes: (p+1) -> (0+1=1, 1+1=2, 2+1=3, 3+1=4, 4+1=5, 5+1=6)
  // Order: P=1, N=2, B=3, R=4, Q=5, K=6

  // Black piece codes: (p+7) -> (0+7=7, 1+7=8, 2+7=9, 3+7=10, 4+7=11, 5+7=12)
  // Order: P=7, N=8, B=9, R=10, Q=11, K=12

  const WHITE_PAWN = 1, WHITE_QUEEN = 5;
  const BLACK_PAWN = 7, BLACK_QUEEN = 11;

  // P-N-B-R-Q-K order: 1=P, 2=N, 3=B, 4=R, 5=Q, 6=K
  const PIECE_UNI = {
    0: '', 
    1: '♟',  // Pawn
    2: '♞',  // Knight (was Rook)
    3: '♝',  // Bishop (was Knight)
    4: '♜',  // Rook (was Bishop)
    5: '♛',  // Queen
    6: '♚',  // King

    7: '♟',  // Pawn (black)
    8: '♞',  // Knight (black)
    9: '♝',  // Bishop (black)
    10: '♜', // Rook (black)
    11: '♛', // Queen (black)
    12: '♚'  // King (black)
  };

  const idxToFR = (sq) => ({ f: sq & 7, r: sq >> 3 });
  const frToIdx = (f, r) => (r * 8 + f);
  const userColor = () => sideSel.value;
  const aiColor = () => aiSel.value;
  const sideLabel = (s) => s === 0 ? 'White' : 'Black';

  function getBoardArray() {
    const ptr = get_board_ptr();
    const view = new Int8Array(Module.HEAP8.buffer, ptr, 64);
    return Array.from(view);
  }

  function getPieceAt(sq){
    return getBoardArray()[sq] || 0;
  }

  function squareLabel(sq){
    const {f,r}=idxToFR(sq);
    return String.fromCharCode(97+f)+(1+r);
  }

  function checkGameOver() {
    const state = game_state_wasm();
    if (state === 0) return false;
    let title = "Game Over", message = "";
    switch (state) {
      case 1:
        title = "Checkmate!";
        message = `${sideLabel(get_turn_wasm())} is checkmated.`;
        break;
      case 2: message = "Draw by stalemate."; break;
      case 4: message = "Draw by 50-move rule."; break;
      case 5: message = "Draw by insufficient material."; break;
      default: message = "Game ended.";
    }
    gameOverTitle.textContent = title;
    gameOverMessage.textContent = message;
    gameOverModal.classList.remove('hidden');
    return true;
  }

  function drawBoard(lastMove) {
    boardEl.innerHTML = '';
    const arr = getBoardArray();
    for (let rr = 7; rr >= 0; rr--) {
      for (let ff = 0; ff < 8; ff++) {
        const r = flipped ? 7 - rr : rr;
        const f = flipped ? 7 - ff : ff;
        const sq = frToIdx(f, r);
        const v = arr[sq];
        const div = document.createElement('div');
        div.className = `sq ${((r + f) & 1) === 0 ? 'dark' : 'light'}`;
        div.dataset.sq = sq;
        div.title = squareLabel(sq);
        if (lastMove && (sq === lastMove.from || sq === lastMove.to)) div.classList.add('lastmove');
        if (selected === sq) div.classList.add('highlight');
        div.textContent = v ? PIECE_UNI[v] || '' : '';
        div.style.color = v ? ((v >= 1 && v <= 6) ? '#fff' : '#000') : '';
        div.addEventListener('click', onSquareClick);
        boardEl.appendChild(div);
      }
    }

    const evalScore = evaluate_wasm();
    const turn = get_turn_wasm();
    const state = game_state_wasm();
    let msg = `Turn: ${sideLabel(turn)} | Eval: ${evalScore}`;
    if (state === 1) msg += ' • <span class="bad">Checkmate</span>';
    else if (state === 2) msg += ' • Draw (stalemate)';
    else if (state === 4) msg += ' • Draw (50-move)';
    else if (state === 5) msg += ' • Draw (insufficient material)';
    statusEl.innerHTML = msg;
  }

  function promotionPendingIndex(){
    try { return pending_wasm(); } catch(e){ return -1; }
  }

  // Robust promotion detection:
  // - figure out if the mover was a pawn and moved to final rank,
  // - then after move check board + pending + moveRes to decide if we must prompt.
  function isPawnPromotionTargetBeforeMove(pieceCode, toSq){
    if (!pieceCode) return false;
    const rank = toSq >> 3;
    if (pieceCode === WHITE_PAWN && rank === 7) return true;
    if (pieceCode === BLACK_PAWN && rank === 0) return true;
    return false;
  }

  // Show promotion chooser; returns piece code to pass to promote_wasm()
  async function showPromotionChooserAndPromote(toSq, moverSide){
    // return a promise that resolves once promote_wasm called
    return new Promise(resolve => {
      // show overlay
      promoMask.classList.remove('hidden');

      // handler helpers
      function cleanup() {
        promoMask.classList.add('hidden');
        promoteQueenBtn.removeEventListener('click', qh);
        promoteRookBtn.removeEventListener('click', rh);
        promoteBishopBtn.removeEventListener('click', bh);
        promoteKnightBtn.removeEventListener('click', nh);
        promoMask.removeEventListener('click', overlayClick);
      }

      function doPromote(pieceType) {
        cleanup();
        promote_wasm(toSq, pieceType);
        drawBoard({from:-1,to:toSq});
        resolve(true);
      }

      // --- FIX: Use correct piece codes based on P-N-B-R-Q-K order ---
      const codeMapWhite = { 
        q: WHITE_QUEEN, // 5
        r: 4,           // Rook piece code
        b: 3,           // Bishop piece code
        n: 2            // Knight piece code
      };

      const codeMapBlack = { 
        q: BLACK_QUEEN, // 11
        r: 10,          // Rook piece code
        b: 9,           // Bishop piece code
        n: 8           // Knight piece code
      };

      const qh = () => doPromote(moverSide===0 ? codeMapWhite.q : codeMapBlack.q);
      const rh = () => doPromote(moverSide===0 ? codeMapWhite.r : codeMapBlack.r);
      const bh = () => doPromote(moverSide===0 ? codeMapWhite.b : codeMapBlack.b);
      const nh = () => doPromote(moverSide===0 ? codeMapWhite.n : codeMapBlack.n);

      // overlay click (outside buttons) => auto-queen fallback
      function overlayClick(e){
        if (e.target === promoMask) {
          // choose queen as default
          doPromote(moverSide===0 ? codeMapWhite.q : codeMapBlack.q);
        }
      }

      promoteQueenBtn.addEventListener('click', qh);
      promoteRookBtn.addEventListener('click', rh);
      promoteBishopBtn.addEventListener('click', bh);
      promoteKnightBtn.addEventListener('click', nh);
      promoMask.addEventListener('click', overlayClick);
    });
  }

  function getMoverPieceBefore(fromSq){
    return getPieceAt(fromSq);
  }

  // Main click handler
  async function onSquareClick(e){
    const sq = Number(e.currentTarget.dataset.sq);
    if (selected === null) {
      selected = sq; drawBoard(); return;
    }
    if (sq === selected) {
      selected = null; drawBoard(); return;
    }

    const from = selected, to = sq;
    selected = null;

    // Save mover piece BEFORE the move (critical for captures)
    const moverPieceBefore = getMoverPieceBefore(from);
    const willBePawnPromotion = isPawnPromotionTargetBeforeMove(moverPieceBefore, to);

    const moveRes = make_move_wasm(from, to); // 0 invalid, 1 ok, 2 promo possibly
    console.debug('[MOVE] from->to', from, '->', to, 'res=', moveRes, 'pending=', promotionPendingIndex());

    if (moveRes === 0) {
      drawBoard(); return;
    }

    moveStack.push({ from, to, isBook: false });

     if (willBePawnPromotion) {
      const moverSide = (moverPieceBefore >=1 && moverPieceBefore <=6) ? 0 : 1;
      await showPromotionChooserAndPromote(to, moverSide);
    }

/* 
    if (willBePawnPromotion && (moveRes === 2 || pending >= 0 || pieceAfter === moverPieceBefore || pieceAfter === 0)) {
      // don't prompt if engine already promoted to a non-pawn piece
      if (!(pieceAfter !== moverPieceBefore && pieceAfter !== 0)) {
        const moverSide = (moverPieceBefore >=1 && moverPieceBefore <=6) ? 0 : 1;
        await showPromotionChooserAndPromote(to, moverSide);
      }
    } */

    drawBoard({ from, to });
    if (checkGameOver()) return;

    await maybeEngineMove();
  }

  // AI move flow
  async function maybeEngineMove(){
    if (aiColor() === 'none') return;
    const turn = get_turn_wasm();
    const aiSide = (aiColor()==='white'?0:1);
    if (turn !== aiSide) return;

    const bookNotation = getOpeningBookMove();
    if (bookNotation) {
      const bookMove = parseBookMove(bookNotation);
      const bookResult = make_move_wasm(bookMove.from, bookMove.to);
      if (bookResult !== 0) {
        openingBookMovesPlayed++;
        moveStack.push({ ...bookMove, isBook: true });
        console.info(`[BOOK] Move ${openingBookMovesPlayed}/3: ${bookNotation}`);
        drawBoard(bookMove);
        checkGameOver();
        return;
      }
    }

    const requestedDepth = Number(depthSel.value) || 1;
    const depth = Math.max(1, Math.trunc(requestedDepth));
    const packed = ai_move_wasm(aiSide, depth);
    if (packed < 0) return;

    const from = packed & 0xff;
    const to = (packed >> 8) & 0xff;
    if (from < 0 || from > 63 || to < 0 || to > 63) {
      console.warn('[AI] Invalid packed move received:', packed);
      return;
    }

    const moverPieceBefore = getMoverPieceBefore(from);
    const willBePawnPromotion = isPawnPromotionTargetBeforeMove(moverPieceBefore, to);

    const res = make_move_wasm(from, to);
    const pending = promotionPendingIndex();
    const pieceAfter = getPieceAt(to);

    if (willBePawnPromotion && (res === 2 || pending >= 0 || pieceAfter === moverPieceBefore || pieceAfter === 0)) {
      // auto-queen for AI by default
      const queenCode = (moverPieceBefore === WHITE_PAWN) ? WHITE_QUEEN : BLACK_QUEEN;
      // if engine expects manual promotion (pending) or left pawn, call promote
      if (pieceAfter === moverPieceBefore || pieceAfter === 0 || pending >= 0 || res === 2) {
        promote_wasm(to, queenCode);
      }
    }

    moveStack.push({ from, to, isBook: false });
    drawBoard({ from, to });
    checkGameOver();
  }

  // Buttons
  newBtn.addEventListener('click', async () => {
    init_board_wasm();
    moveStack.length = 0;
    openingBookMovesPlayed = 0;
    selectedOpening = null;
    reseedOpeningRandom();
    selected = null;
    drawBoard();
    await maybeEngineMove();
  });

  flipBtn.addEventListener('click', () => { flipped = !flipped; drawBoard(); });

  undoBtn.addEventListener('click', () => {
    if (!moveStack.length) return;
    const history = moveStack.slice(0, -1);
    init_board_wasm();
    for (const mv of history) {
      const moverPieceBefore = getMoverPieceBefore(mv.from);
      const r = make_move_wasm(mv.from, mv.to);
      if (isPawnPromotionTargetBeforeMove(moverPieceBefore, mv.to) && (r === 2 || promotionPendingIndex() >= 0 || getPieceAt(mv.to) === moverPieceBefore)) {
        // replay auto-queen for simplicity
        promote_wasm(mv.to, moverPieceBefore === WHITE_PAWN ? WHITE_QUEEN : BLACK_QUEEN);
      }
    }
    moveStack.length = history.length;
    openingBookMovesPlayed = history.filter(move => move.isBook).length;
    selectedOpening = null;
    drawBoard(history[history.length-1] || null);
  });

  hintBtn.addEventListener('click', () => {
    const turn = get_turn_wasm();
    const depth = Math.max(1, Number(depthSel.value) || 1);
    const packed = ai_move_wasm(turn, depth);
    if (packed < 0) { statusEl.innerHTML = 'No legal moves.'; return; }
    const from = packed & 0xff;
    const to = (packed >> 8) & 0xff;
    if (from < 0 || from > 63 || to < 0 || to > 63) {
      statusEl.innerHTML = 'No legal moves.';
      return;
    }
    statusEl.innerHTML = `Hint: ${squareLabel(from)} → ${squareLabel(to)}`;
  });

  difficultySel.addEventListener('change', (e) => {
    const level = Number(e.target.value) || 2;
    if (!set_difficulty) {
      statusEl.innerHTML = 'Difficulty is unavailable in this engine build';
      return;
    }
    const success = set_difficulty(level);
    if (success) {
      statusEl.innerHTML = `Difficulty set to Level ${level}`;
    } else {
      statusEl.innerHTML = `Failed to load Level ${level} model`;
    }
  });

  undoBtn.disabled = false;

  gameOverResetButton.onclick = async () => {
    gameOverModal.classList.add('hidden');
    init_board_wasm();
    moveStack.length = 0;
    openingBookMovesPlayed = 0;
    selectedOpening = null;
    reseedOpeningRandom();
    selected = null;
    drawBoard();
    await maybeEngineMove();
  };

  // Start
  init_board_wasm();
  openingBookMovesPlayed = 0;
  selectedOpening = null;
  reseedOpeningRandom();
  drawBoard();
  await maybeEngineMove();

})();