const express = require('express');
const router = express.Router();
const { v4: uuidv4 } = require('uuid');
const db = require('../src/database');
const { calculateElo } = require('../src/elo');

const AI_PLAYER_ID = 'ai-player';

function ensureAiPlayer() {
  const existing = db.prepare('SELECT id FROM players WHERE id = ?').get(AI_PLAYER_ID);
  if (!existing) {
    db.prepare(
      "INSERT INTO players (id, name, avatar, elo) VALUES (?, 'AI', 'robot', 1200)"
    ).run(AI_PLAYER_ID);
  }
}

// GET game history (most recent first)
router.get('/history', (req, res) => {
  const limit = Math.min(parseInt(req.query.limit) || 50, 200);
  const rows = db.prepare(`
    SELECT gh.*, p.name AS winner_name
    FROM game_history gh
    LEFT JOIN players p ON gh.winner_id = p.id
    ORDER BY gh.completed_at DESC
    LIMIT ?
  `).all(limit);
  res.json(rows);
});

// POST record a completed game result
// Body: { game_type, player_ids: [id, id], winner_id: id | null (draw) }
router.post('/result', (req, res) => {
  const { game_type, player_ids, winner_id } = req.body;
  if (!game_type || !Array.isArray(player_ids) || player_ids.length !== 2) {
    return res.status(400).json({ error: 'game_type and exactly 2 player_ids required' });
  }

  ensureAiPlayer();

  const [idA, idB] = player_ids;
  const playerA = db.prepare('SELECT * FROM players WHERE id = ?').get(idA);
  const playerB = db.prepare('SELECT * FROM players WHERE id = ?').get(idB);
  if (!playerA || !playerB) {
    return res.status(404).json({ error: 'One or more players not found' });
  }

  let result;
  if (!winner_id) {
    result = 'draw';
  } else if (winner_id === idA) {
    result = 'A';
  } else if (winner_id === idB) {
    result = 'B';
  } else {
    return res.status(400).json({ error: 'winner_id must be one of the player_ids or null' });
  }

  const { newA, newB, changeA, changeB } = calculateElo(playerA.elo, playerB.elo, result);

  const sessionId = uuidv4();

  const updateStats = db.transaction(() => {
    db.prepare("INSERT INTO game_sessions (id, game_type, status, completed_at) VALUES (?, ?, 'completed', datetime('now'))")
      .run(sessionId, game_type);

    const winnerName = winner_id === idA ? playerA.name : winner_id === idB ? playerB.name : null;

    db.prepare(`
      INSERT INTO game_history (session_id, game_type, winner_id, player_ids)
      VALUES (?, ?, ?, ?)
    `).run(sessionId, game_type, winner_id || null, JSON.stringify(player_ids));

    // Update player A
    db.prepare(`
      UPDATE players SET elo = ?, wins = wins + ?, losses = losses + ?, draws = draws + ?
      WHERE id = ?
    `).run(
      newA,
      result === 'A' ? 1 : 0,
      result === 'B' ? 1 : 0,
      result === 'draw' ? 1 : 0,
      idA
    );

    // Update player B
    db.prepare(`
      UPDATE players SET elo = ?, wins = wins + ?, losses = losses + ?, draws = draws + ?
      WHERE id = ?
    `).run(
      newB,
      result === 'B' ? 1 : 0,
      result === 'A' ? 1 : 0,
      result === 'draw' ? 1 : 0,
      idB
    );
  });

  updateStats();

  res.json({
    session_id: sessionId,
    playerA: { id: idA, newElo: newA, change: changeA },
    playerB: { id: idB, newElo: newB, change: changeB },
  });
});

module.exports = router;
