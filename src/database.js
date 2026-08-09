const Database = require('better-sqlite3');
const path = require('path');
const fs = require('fs');

let db;

if (process.env.NODE_ENV === 'test') {
  db = new Database(':memory:');
} else {
  const DATA_DIR = path.join(__dirname, '..', 'data');
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
  const DB_PATH = path.join(DATA_DIR, 'gamnight.db');
  db = new Database(DB_PATH);
}

db.exec(`
  PRAGMA journal_mode=WAL;

  CREATE TABLE IF NOT EXISTS players (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    avatar TEXT DEFAULT 'default',
    elo INTEGER DEFAULT 1200,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    draws INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
  );

  CREATE TABLE IF NOT EXISTS game_sessions (
    id TEXT PRIMARY KEY,
    game_type TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    created_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT
  );

  CREATE TABLE IF NOT EXISTS game_participants (
    session_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    role TEXT,
    result TEXT,
    elo_change INTEGER DEFAULT 0,
    PRIMARY KEY (session_id, player_id),
    FOREIGN KEY (session_id) REFERENCES game_sessions(id),
    FOREIGN KEY (player_id) REFERENCES players(id)
  );

  CREATE TABLE IF NOT EXISTS game_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    game_type TEXT NOT NULL,
    winner_id TEXT,
    player_ids TEXT NOT NULL,
    completed_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES game_sessions(id)
  );
`);

module.exports = db;
