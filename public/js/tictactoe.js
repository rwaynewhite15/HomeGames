/* ===== Tic-Tac-Toe ===== */

const TTT = {
  board: Array(9).fill(null),
  current: 'X',  // 'X' = p1, 'O' = p2
  over: false,
  resultRecorded: false,
};

const TTT_WINS = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];

function tttInit() {
  TTT.board = Array(9).fill(null);
  TTT.current = 'X';
  TTT.over = false;
  TTT.resultRecorded = false;
  tttRender();
  tttUpdateStatus();
  tttUpdatePlayerInfo();
  // If AI is p1, take turn
  if (currentPlayers.p1.isAI) setTimeout(tttAiMove, 500);
}

function tttReset() {
  tttInit();
}

function tttRender() {
  const board = document.getElementById('ttt-board');
  board.innerHTML = TTT.board.map((cell, i) => {
    const taken = cell !== null;
    const winLine = tttCheckWin(TTT.board);
    const isWin = winLine && winLine.includes(i);
    return `<div class="ttt-cell ${taken ? 'taken' : ''} ${isWin ? 'win-cell' : ''}"
      onclick="tttClick(${i})">${cell ? (cell === 'X' ? '✕' : '◯') : ''}</div>`;
  }).join('');
}

function tttClick(i) {
  if (TTT.over || TTT.board[i] !== null) return;
  const isP1Turn = TTT.current === 'X';
  const activePlayer = isP1Turn ? currentPlayers.p1 : currentPlayers.p2;
  if (activePlayer.isAI) return; // AI's turn, ignore click

  TTT.board[i] = TTT.current;
  tttAdvance();
}

function tttAdvance() {
  tttRender();
  const winLine = tttCheckWin(TTT.board);
  if (winLine) {
    TTT.over = true;
    const winner = TTT.current === 'X' ? currentPlayers.p1 : currentPlayers.p2;
    tttUpdateStatus(`🎉 ${winner.name} wins!`, 'won');
    if (!TTT.resultRecorded) {
      TTT.resultRecorded = true;
      recordGameResult('tictactoe', currentPlayers.p1, currentPlayers.p2, winner.id);
    }
    return;
  }
  if (TTT.board.every(c => c !== null)) {
    TTT.over = true;
    tttUpdateStatus("🤝 It's a draw!", 'draw');
    if (!TTT.resultRecorded) {
      TTT.resultRecorded = true;
      recordGameResult('tictactoe', currentPlayers.p1, currentPlayers.p2, null);
    }
    return;
  }
  TTT.current = TTT.current === 'X' ? 'O' : 'X';
  tttUpdateStatus();
  const nextPlayer = TTT.current === 'X' ? currentPlayers.p1 : currentPlayers.p2;
  if (nextPlayer.isAI) setTimeout(tttAiMove, 400);
}

function tttUpdateStatus(msg, cls) {
  const el = document.getElementById('ttt-status');
  if (msg) {
    el.textContent = msg;
    el.className = `game-status ${cls || 'playing'}`;
    return;
  }
  const isP1Turn = TTT.current === 'X';
  const active = isP1Turn ? currentPlayers.p1 : currentPlayers.p2;
  const symbol = isP1Turn ? '✕' : '◯';
  el.textContent = `${active.name}'s turn (${symbol})${active.isAI ? ' 🤖' : ''}`;
  el.className = 'game-status playing';
}

function tttUpdatePlayerInfo() {
  const el = document.getElementById('ttt-player-info');
  const p1 = currentPlayers.p1;
  const p2 = currentPlayers.p2;
  el.innerHTML = `
    <div style="display:flex;gap:2rem;justify-content:center;flex-wrap:wrap">
      <div style="text-align:center">
        <div style="font-size:2rem">${p1.avatar === 'default' ? '😀' : p1.avatar}</div>
        <div><strong>${esc(p1.name)}</strong> ${p1.isAI ? '<span class="ai-badge">AI</span>' : ''}</div>
        <div style="color:var(--text-dim);font-size:0.85rem">✕ · ELO ${p1.elo}</div>
      </div>
      <div style="font-size:2rem;display:flex;align-items:center">⚔️</div>
      <div style="text-align:center">
        <div style="font-size:2rem">${p2.avatar === 'default' ? '😀' : p2.avatar}</div>
        <div><strong>${esc(p2.name)}</strong> ${p2.isAI ? '<span class="ai-badge">AI</span>' : ''}</div>
        <div style="color:var(--text-dim);font-size:0.85rem">◯ · ELO ${p2.elo}</div>
      </div>
    </div>
  `;
}

/* ---- TTT Win check ---- */
function tttCheckWin(board) {
  for (const line of TTT_WINS) {
    const [a, b, c] = line;
    if (board[a] && board[a] === board[b] && board[a] === board[c]) return line;
  }
  return null;
}

/* ---- TTT Minimax AI ---- */
function tttAiMove() {
  if (TTT.over) return;
  const aiSymbol = currentPlayers.p1.isAI ? 'X' : 'O';
  const humanSymbol = aiSymbol === 'X' ? 'O' : 'X';
  const best = tttMinimax(TTT.board.slice(), aiSymbol, aiSymbol, humanSymbol);
  if (best.index !== -1) {
    TTT.board[best.index] = TTT.current;
    tttAdvance();
  }
}

function tttMinimax(board, aiSym, humanSym, currentSym) {
  const win = tttCheckWin(board);
  if (win) {
    const winner = board[win[0]];
    return { score: winner === aiSym ? 10 : -10 };
  }
  const empty = board.map((v, i) => v === null ? i : null).filter(v => v !== null);
  if (empty.length === 0) return { score: 0 };

  const moves = [];
  for (const i of empty) {
    board[i] = currentSym;
    const next = currentSym === aiSym ? humanSym : aiSym;
    const result = tttMinimax(board, aiSym, humanSym, next);
    moves.push({ index: i, score: result.score });
    board[i] = null;
  }

  if (currentSym === aiSym) {
    return moves.reduce((best, m) => m.score > best.score ? m : best, { score: -Infinity, index: -1 });
  } else {
    return moves.reduce((best, m) => m.score < best.score ? m : best, { score: Infinity, index: -1 });
  }
}
