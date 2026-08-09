/* ===== Connect Four ===== */

const C4 = {
  ROWS: 6,
  COLS: 7,
  board: [],   // [row][col] = null | 1 | 2
  current: 1,  // 1 = p1, 2 = p2
  over: false,
  resultRecorded: false,
};

function c4EmptyBoard() {
  return Array.from({ length: C4.ROWS }, () => Array(C4.COLS).fill(null));
}

function c4Init() {
  C4.board = c4EmptyBoard();
  C4.current = 1;
  C4.over = false;
  C4.resultRecorded = false;
  c4Render();
  c4UpdateStatus();
  c4UpdatePlayerInfo();
  if (currentPlayers.p1.isAI) setTimeout(c4AiMove, 500);
}

function c4Reset() { c4Init(); }

function c4Render() {
  const boardEl = document.getElementById('c4-board');
  const btnsEl = document.getElementById('c4-col-btns');

  // Column drop buttons
  btnsEl.innerHTML = Array.from({ length: C4.COLS }, (_, col) => `
    <button class="c4-col-btn" id="c4-btn-${col}" onclick="c4Drop(${col})" ${C4.over ? 'disabled' : ''}>▼</button>
  `).join('');

  // Board cells (top row first)
  const winCells = c4CheckWin(C4.board);
  const winSet = new Set(winCells ? winCells.map(([r, c]) => `${r},${c}`) : []);

  boardEl.innerHTML = '';
  for (let row = 0; row < C4.ROWS; row++) {
    for (let col = 0; col < C4.COLS; col++) {
      const val = C4.board[row][col];
      const isWin = winSet.has(`${row},${col}`);
      const div = document.createElement('div');
      div.className = `c4-cell${val === 1 ? ' p1' : val === 2 ? ' p2' : ''}${isWin ? ' win-cell' : ''}`;
      boardEl.appendChild(div);
    }
  }
}

function c4Drop(col) {
  if (C4.over) return;
  const activePlayer = C4.current === 1 ? currentPlayers.p1 : currentPlayers.p2;
  if (activePlayer.isAI) return;
  c4PlacePiece(col);
}

function c4PlacePiece(col) {
  const row = c4LowestEmpty(C4.board, col);
  if (row === -1) return; // column full
  C4.board[row][col] = C4.current;
  c4Render();
  c4Advance();
}

function c4LowestEmpty(board, col) {
  for (let r = C4.ROWS - 1; r >= 0; r--) {
    if (board[r][col] === null) return r;
  }
  return -1;
}

function c4Advance() {
  const winCells = c4CheckWin(C4.board);
  if (winCells) {
    C4.over = true;
    const winner = C4.current === 1 ? currentPlayers.p1 : currentPlayers.p2;
    c4UpdateStatus(`🎉 ${winner.name} wins!`, 'won');
    c4Render();
    if (!C4.resultRecorded) {
      C4.resultRecorded = true;
      recordGameResult('connect4', currentPlayers.p1, currentPlayers.p2, winner.id);
    }
    return;
  }
  if (C4.board[0].every(c => c !== null)) {
    C4.over = true;
    c4UpdateStatus("🤝 It's a draw!", 'draw');
    if (!C4.resultRecorded) {
      C4.resultRecorded = true;
      recordGameResult('connect4', currentPlayers.p1, currentPlayers.p2, null);
    }
    return;
  }
  C4.current = C4.current === 1 ? 2 : 1;
  c4UpdateStatus();
  const next = C4.current === 1 ? currentPlayers.p1 : currentPlayers.p2;
  if (next.isAI) setTimeout(c4AiMove, 450);
}

function c4UpdateStatus(msg, cls) {
  const el = document.getElementById('c4-status');
  if (msg) {
    el.textContent = msg;
    el.className = `game-status ${cls || 'playing'}`;
    return;
  }
  const isP1 = C4.current === 1;
  const active = isP1 ? currentPlayers.p1 : currentPlayers.p2;
  const disc = isP1 ? '🔴' : '🟡';
  el.textContent = `${active.name}'s turn (${disc})${active.isAI ? ' 🤖' : ''}`;
  el.className = 'game-status playing';
}

function c4UpdatePlayerInfo() {
  const el = document.getElementById('c4-player-info');
  const p1 = currentPlayers.p1;
  const p2 = currentPlayers.p2;
  el.innerHTML = `
    <div style="display:flex;gap:2rem;justify-content:center;flex-wrap:wrap">
      <div style="text-align:center">
        <div style="font-size:2rem">${p1.avatar === 'default' ? '😀' : p1.avatar}</div>
        <div><strong>${esc(p1.name)}</strong> ${p1.isAI ? '<span class="ai-badge">AI</span>' : ''}</div>
        <div style="color:var(--text-dim);font-size:0.85rem">🔴 · ELO ${p1.elo}</div>
      </div>
      <div style="font-size:2rem;display:flex;align-items:center">⚔️</div>
      <div style="text-align:center">
        <div style="font-size:2rem">${p2.avatar === 'default' ? '😀' : p2.avatar}</div>
        <div><strong>${esc(p2.name)}</strong> ${p2.isAI ? '<span class="ai-badge">AI</span>' : ''}</div>
        <div style="color:var(--text-dim);font-size:0.85rem">🟡 · ELO ${p2.elo}</div>
      </div>
    </div>
  `;
}

/* ---- C4 Win check ---- */
function c4CheckWin(board) {
  const directions = [[0,1],[1,0],[1,1],[1,-1]];
  for (let r = 0; r < C4.ROWS; r++) {
    for (let c = 0; c < C4.COLS; c++) {
      const v = board[r][c];
      if (!v) continue;
      for (const [dr, dc] of directions) {
        const cells = [[r, c]];
        for (let k = 1; k < 4; k++) {
          const nr = r + dr * k, nc = c + dc * k;
          if (nr < 0 || nr >= C4.ROWS || nc < 0 || nc >= C4.COLS || board[nr][nc] !== v) break;
          cells.push([nr, nc]);
        }
        if (cells.length === 4) return cells;
      }
    }
  }
  return null;
}

/* ---- C4 AI (heuristic + minimax depth 5) ---- */
function c4AiMove() {
  if (C4.over) return;
  const aiPiece = C4.current;
  const humanPiece = aiPiece === 1 ? 2 : 1;
  const col = c4BestMove(C4.board, aiPiece, humanPiece);
  if (col !== -1) c4PlacePiece(col);
}

function c4BestMove(board, aiPiece, humanPiece) {
  const DEPTH = 5;
  let bestScore = -Infinity;
  let bestCol = -1;
  const cols = c4OrderedCols();
  for (const col of cols) {
    const row = c4LowestEmpty(board, col);
    if (row === -1) continue;
    const nb = c4CopyBoard(board);
    nb[row][col] = aiPiece;
    const score = c4Minimax(nb, DEPTH - 1, -Infinity, Infinity, false, aiPiece, humanPiece);
    if (score > bestScore) { bestScore = score; bestCol = col; }
  }
  return bestCol;
}

function c4OrderedCols() {
  // Prefer center columns
  return [3, 2, 4, 1, 5, 0, 6];
}

function c4Minimax(board, depth, alpha, beta, isMaximizing, aiPiece, humanPiece) {
  const win = c4CheckWin(board);
  if (win) {
    const winner = board[win[0][0]][win[0][1]];
    return winner === aiPiece ? 1000 + depth : -(1000 + depth);
  }
  if (board[0].every(c => c !== null) || depth === 0) return c4ScoreBoard(board, aiPiece);

  const piece = isMaximizing ? aiPiece : humanPiece;
  let best = isMaximizing ? -Infinity : Infinity;

  for (const col of c4OrderedCols()) {
    const row = c4LowestEmpty(board, col);
    if (row === -1) continue;
    const nb = c4CopyBoard(board);
    nb[row][col] = piece;
    const score = c4Minimax(nb, depth - 1, alpha, beta, !isMaximizing, aiPiece, humanPiece);
    if (isMaximizing) {
      best = Math.max(best, score);
      alpha = Math.max(alpha, score);
    } else {
      best = Math.min(best, score);
      beta = Math.min(beta, score);
    }
    if (beta <= alpha) break;
  }
  return best;
}

function c4ScoreBoard(board, aiPiece) {
  let score = 0;
  // Center column preference
  const centerCol = board.map(row => row[3]);
  score += centerCol.filter(c => c === aiPiece).length * 3;

  const directions = [[0,1],[1,0],[1,1],[1,-1]];
  for (let r = 0; r < C4.ROWS; r++) {
    for (let c = 0; c < C4.COLS; c++) {
      for (const [dr, dc] of directions) {
        const window = [];
        for (let k = 0; k < 4; k++) {
          const nr = r + dr * k, nc = c + dc * k;
          if (nr < 0 || nr >= C4.ROWS || nc < 0 || nc >= C4.COLS) break;
          window.push(board[nr][nc]);
        }
        if (window.length === 4) score += c4ScoreWindow(window, aiPiece);
      }
    }
  }
  return score;
}

function c4ScoreWindow(w, aiPiece) {
  const opp = aiPiece === 1 ? 2 : 1;
  const aiCount = w.filter(c => c === aiPiece).length;
  const oppCount = w.filter(c => c === opp).length;
  const empty = w.filter(c => c === null).length;
  if (aiCount === 4) return 100;
  if (aiCount === 3 && empty === 1) return 5;
  if (aiCount === 2 && empty === 2) return 2;
  if (oppCount === 3 && empty === 1) return -4;
  return 0;
}

function c4CopyBoard(board) {
  return board.map(row => row.slice());
}
