const express = require('express');
const router = express.Router();
const { v4: uuidv4 } = require('uuid');
const db = require('../src/database');

// GET all players (sorted by ELO desc)
router.get('/', (req, res) => {
  const players = db.prepare(
    'SELECT * FROM players ORDER BY elo DESC'
  ).all();
  res.json(players);
});

// GET single player
router.get('/:id', (req, res) => {
  const player = db.prepare('SELECT * FROM players WHERE id = ?').get(req.params.id);
  if (!player) return res.status(404).json({ error: 'Player not found' });
  res.json(player);
});

// POST create player
router.post('/', (req, res) => {
  const { name, avatar } = req.body;
  if (!name || typeof name !== 'string' || name.trim().length === 0) {
    return res.status(400).json({ error: 'Name is required' });
  }
  const trimmed = name.trim().slice(0, 32);
  const id = uuidv4();
  try {
    db.prepare(
      'INSERT INTO players (id, name, avatar) VALUES (?, ?, ?)'
    ).run(id, trimmed, avatar || 'default');
    const player = db.prepare('SELECT * FROM players WHERE id = ?').get(id);
    res.status(201).json(player);
  } catch (err) {
    if (err.message.includes('UNIQUE')) {
      return res.status(409).json({ error: 'Player name already exists' });
    }
    throw err;
  }
});

// PUT update player name/avatar
router.put('/:id', (req, res) => {
  const { name, avatar } = req.body;
  const player = db.prepare('SELECT * FROM players WHERE id = ?').get(req.params.id);
  if (!player) return res.status(404).json({ error: 'Player not found' });
  const newName = (name || player.name).trim().slice(0, 32);
  const newAvatar = avatar || player.avatar;
  try {
    db.prepare(
      'UPDATE players SET name = ?, avatar = ? WHERE id = ?'
    ).run(newName, newAvatar, req.params.id);
    res.json(db.prepare('SELECT * FROM players WHERE id = ?').get(req.params.id));
  } catch (err) {
    if (err.message.includes('UNIQUE')) {
      return res.status(409).json({ error: 'Player name already exists' });
    }
    throw err;
  }
});

// DELETE player
router.delete('/:id', (req, res) => {
  const player = db.prepare('SELECT * FROM players WHERE id = ?').get(req.params.id);
  if (!player) return res.status(404).json({ error: 'Player not found' });
  db.prepare('DELETE FROM players WHERE id = ?').run(req.params.id);
  res.json({ message: 'Player deleted' });
});

module.exports = router;
