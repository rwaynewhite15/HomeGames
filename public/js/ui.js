/* ===== UI helpers & shared state ===== */

const AVATARS = ['😀','😎','🤖','👾','🦊','🐼','🐸','🦁','🐯','🦄','👻','🎃','🍕','🎸','🚀','⚡','🔥','🌊'];
const AI_PLAYER_ID = 'ai-player';

let players = [];        // cached player list
let currentGameType = null;
let currentPlayers = { p1: null, p2: null }; // { id, name, avatar, elo, isAI }

/* ---- View routing ---- */
function showView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  const view = document.getElementById('view-' + name);
  if (view) view.classList.add('active');
  const navBtn = document.getElementById('nav-' + name);
  if (navBtn) navBtn.classList.add('active');

  if (name === 'lobby') loadLobbyLeaderboard();
  if (name === 'players') loadPlayersView();
  if (name === 'leaderboard') loadLeaderboardView();
  if (name === 'history') loadHistoryView();
}

/* ---- Toast ---- */
function showToast(msg, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

/* ---- Modal helpers ---- */
function openModal(id) { document.getElementById(id).classList.add('open'); }
function closeModal(id) { document.getElementById(id).classList.remove('open'); }

/* ---- ELO badge ---- */
function eloBadge(elo, rank) {
  let cls = 'elo-normal';
  if (rank === 1) cls = 'elo-gold';
  else if (rank === 2) cls = 'elo-silver';
  else if (rank === 3) cls = 'elo-bronze';
  return `<span class="elo-badge ${cls}">${elo}</span>`;
}

function rankMedal(rank) {
  if (rank === 1) return '🥇';
  if (rank === 2) return '🥈';
  if (rank === 3) return '🥉';
  return `#${rank}`;
}

function winPct(p) {
  const total = p.wins + p.losses + p.draws;
  if (total === 0) return '—';
  return Math.round((p.wins / total) * 100) + '%';
}

/* ---- Player loading ---- */
async function refreshPlayers() {
  players = await API.getPlayers();
  return players;
}

function getAiPlayer() {
  return { id: AI_PLAYER_ID, name: 'AI', avatar: '🤖', elo: 1200, isAI: true };
}

/* ---- Lobby leaderboard ---- */
async function loadLobbyLeaderboard() {
  try {
    await refreshPlayers();
    const tbody = document.getElementById('lobby-lb-body');
    const humanPlayers = players.filter(p => p.id !== AI_PLAYER_ID);
    if (humanPlayers.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" style="color:var(--text-dim);padding:1.5rem;text-align:center">No players yet. <a href="#" onclick="showView('players')" style="color:var(--accent)">Add some!</a></td></tr>`;
      return;
    }
    tbody.innerHTML = humanPlayers.slice(0, 5).map((p, i) => `
      <tr>
        <td>${rankMedal(i + 1)}</td>
        <td>${p.avatar === 'default' ? '😀' : p.avatar} ${esc(p.name)}</td>
        <td>${eloBadge(p.elo, i + 1)}</td>
        <td>${p.wins}/${p.losses}/${p.draws}</td>
      </tr>
    `).join('');
  } catch (e) {
    console.error(e);
  }
}

/* ---- Players view ---- */
async function loadPlayersView() {
  try {
    await refreshPlayers();
    renderPlayersTable();
  } catch (e) {
    showToast('Failed to load players', 'error');
  }
}

function renderPlayersTable() {
  const tbody = document.getElementById('players-body');
  const humanPlayers = players.filter(p => p.id !== AI_PLAYER_ID);
  if (humanPlayers.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="color:var(--text-dim);padding:1.5rem;text-align:center">No players yet.</td></tr>`;
    return;
  }
  tbody.innerHTML = humanPlayers.map(p => `
    <tr>
      <td style="font-size:1.5rem">${p.avatar === 'default' ? '😀' : p.avatar}</td>
      <td>${esc(p.name)}</td>
      <td>${eloBadge(p.elo, 0)}</td>
      <td>${p.wins}/${p.losses}/${p.draws}</td>
      <td>
        <button class="btn btn-secondary btn-sm" onclick="openEditPlayer('${p.id}')">✏️ Edit</button>
        <button class="btn btn-danger btn-sm" onclick="deletePlayer('${p.id}')">🗑️</button>
      </td>
    </tr>
  `).join('');
}

/* ---- Leaderboard view ---- */
async function loadLeaderboardView() {
  try {
    await refreshPlayers();
    const tbody = document.getElementById('leaderboard-body');
    const humanPlayers = players.filter(p => p.id !== AI_PLAYER_ID);
    if (humanPlayers.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="color:var(--text-dim);padding:1.5rem;text-align:center">No players yet.</td></tr>`;
      return;
    }
    tbody.innerHTML = humanPlayers.map((p, i) => `
      <tr>
        <td>${rankMedal(i + 1)}</td>
        <td>${p.avatar === 'default' ? '😀' : p.avatar} ${esc(p.name)}</td>
        <td>${eloBadge(p.elo, i + 1)}</td>
        <td>${p.wins}</td>
        <td>${p.losses}</td>
        <td>${p.draws}</td>
        <td>${winPct(p)}</td>
      </tr>
    `).join('');
  } catch (e) { showToast('Failed to load leaderboard', 'error'); }
}

/* ---- History view ---- */
async function loadHistoryView() {
  try {
    await refreshPlayers();
    const history = await API.getHistory();
    const tbody = document.getElementById('history-body');
    if (history.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" style="color:var(--text-dim);padding:1.5rem;text-align:center">No games played yet.</td></tr>`;
      return;
    }
    const playerMap = {};
    players.forEach(p => { playerMap[p.id] = p; });
    playerMap[AI_PLAYER_ID] = { name: 'AI', avatar: '🤖' };

    tbody.innerHTML = history.map(h => {
      const ids = JSON.parse(h.player_ids);
      const names = ids.map(id => {
        const p = playerMap[id];
        return p ? `${p.avatar === 'default' ? '😀' : p.avatar} ${esc(p.name)}` : id;
      }).join(' vs ');
      const winner = h.winner_id ? (playerMap[h.winner_id] ? esc(playerMap[h.winner_id].name) : h.winner_id) : '🤝 Draw';
      const date = new Date(h.completed_at + (h.completed_at.endsWith('Z') ? '' : 'Z'));
      return `<tr>
        <td>${date.toLocaleDateString()} ${date.toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}</td>
        <td>${gameLabel(h.game_type)}</td>
        <td>${names}</td>
        <td>${winner}</td>
      </tr>`;
    }).join('');
  } catch (e) { showToast('Failed to load history', 'error'); }
}

function gameLabel(t) {
  const labels = { tictactoe: '⭕ Tic-Tac-Toe', connect4: '🔴 Connect Four' };
  return labels[t] || t;
}

/* ---- Add / Edit player ---- */
function buildAvatarGrid(selected) {
  const grid = document.getElementById('avatar-grid');
  grid.innerHTML = AVATARS.map(a => `
    <div class="avatar-opt ${a === selected ? 'selected' : ''}" onclick="selectAvatar(this,'${a}')">${a}</div>
  `).join('');
}

function selectAvatar(el, avatar) {
  document.querySelectorAll('.avatar-opt').forEach(o => o.classList.remove('selected'));
  el.classList.add('selected');
}

function getSelectedAvatar() {
  const sel = document.querySelector('.avatar-opt.selected');
  return sel ? sel.textContent : AVATARS[0];
}

function openAddPlayer() {
  document.getElementById('modal-player-title').textContent = 'Add Player';
  document.getElementById('player-name-input').value = '';
  document.getElementById('edit-player-id').value = '';
  buildAvatarGrid(AVATARS[0]);
  openModal('modal-player');
  setTimeout(() => document.getElementById('player-name-input').focus(), 100);
}

function openEditPlayer(id) {
  const p = players.find(x => x.id === id);
  if (!p) return;
  document.getElementById('modal-player-title').textContent = 'Edit Player';
  document.getElementById('player-name-input').value = p.name;
  document.getElementById('edit-player-id').value = id;
  buildAvatarGrid(p.avatar === 'default' ? AVATARS[0] : p.avatar);
  openModal('modal-player');
}

async function savePlayer() {
  const name = document.getElementById('player-name-input').value.trim();
  const avatar = getSelectedAvatar();
  const editId = document.getElementById('edit-player-id').value;
  if (!name) { showToast('Please enter a name', 'error'); return; }
  try {
    if (editId) {
      await API.updatePlayer(editId, name, avatar);
      showToast('Player updated!', 'success');
    } else {
      await API.createPlayer(name, avatar);
      showToast(`Welcome, ${name}!`, 'success');
    }
    closeModal('modal-player');
    await loadPlayersView();
  } catch (e) {
    showToast(e.message, 'error');
  }
}

async function deletePlayer(id) {
  const p = players.find(x => x.id === id);
  if (!p) return;
  if (!confirm(`Delete player "${p.name}"? This cannot be undone.`)) return;
  try {
    await API.deletePlayer(id);
    showToast('Player deleted', 'info');
    await loadPlayersView();
  } catch (e) {
    showToast(e.message, 'error');
  }
}

/* ---- Game setup modal ---- */
function openGameSetup(gameType) {
  currentGameType = gameType;
  const titles = { tictactoe: '⭕ Tic-Tac-Toe Setup', connect4: '🔴 Connect Four Setup' };
  document.getElementById('modal-game-title').textContent = titles[gameType] || 'Game Setup';

  const humanPlayers = players.filter(p => p.id !== AI_PLAYER_ID);
  const opts = humanPlayers.map(p =>
    `<option value="${p.id}">${p.avatar === 'default' ? '😀' : p.avatar} ${esc(p.name)} (ELO: ${p.elo})</option>`
  ).join('');

  const aiOpt = `<option value="${AI_PLAYER_ID}">🤖 AI Player</option>`;

  const p1 = document.getElementById('setup-p1');
  const p2 = document.getElementById('setup-p2');

  if (humanPlayers.length === 0) {
    p1.innerHTML = `<option value="">— No players —</option>`;
    p2.innerHTML = aiOpt;
    showToast('Add players first!', 'error');
    showView('players');
    return;
  }

  p1.innerHTML = opts;
  p2.innerHTML = opts + aiOpt;

  // Default: p2 = second human or AI
  if (humanPlayers.length >= 2) {
    p2.value = humanPlayers[1].id;
  } else {
    p2.value = AI_PLAYER_ID;
  }

  openModal('modal-game-setup');
}

function startGame() {
  const p1Id = document.getElementById('setup-p1').value;
  const p2Id = document.getElementById('setup-p2').value;
  if (p1Id === p2Id) { showToast('Player 1 and Player 2 must be different!', 'error'); return; }

  const humanPlayers = players.filter(p => p.id !== AI_PLAYER_ID);
  const findPlayer = id => id === AI_PLAYER_ID ? getAiPlayer() : { ...humanPlayers.find(p => p.id === id), isAI: false };

  currentPlayers.p1 = findPlayer(p1Id);
  currentPlayers.p2 = findPlayer(p2Id);

  closeModal('modal-game-setup');

  if (currentGameType === 'tictactoe') {
    showView('tictactoe');
    tttInit();
  } else if (currentGameType === 'connect4') {
    showView('connect4');
    c4Init();
  }
}

/* ---- XSS escape ---- */
function esc(str) {
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ---- Record game result and update UI ---- */
async function recordGameResult(gameType, p1, p2, winnerId) {
  try {
    const result = await API.recordResult(gameType, [p1.id, p2.id], winnerId);
    // Show ELO changes
    const changeA = result.playerA.change >= 0 ? `+${result.playerA.change}` : result.playerA.change;
    const changeB = result.playerB.change >= 0 ? `+${result.playerB.change}` : result.playerB.change;
    if (!p1.isAI) showToast(`${p1.name}: ELO ${changeA} → ${result.playerA.newElo}`, changeA >= 0 ? 'success' : 'info');
    if (!p2.isAI) showToast(`${p2.name}: ELO ${changeB} → ${result.playerB.newElo}`, changeB >= 0 ? 'success' : 'info');
    await refreshPlayers();
  } catch (e) {
    console.error('Failed to record result:', e);
  }
}
