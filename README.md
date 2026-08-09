# HomeGames 🎮

A locally-hosted GameNight browser webapp for playing games with friends and AI players on your home network. Run this server (from a Raspberry Pi for low power) to host a local network game room with player profiles including ELO Ratings. Games are being added periodically.

## Features

- **Player Profiles** — Create and manage players with custom avatars
- **ELO Rating System** — Automatic ELO calculation after every game
- **Leaderboard** — Ranked player standings with win/loss/draw stats
- **Game History** — Complete log of all games played
- **AI Players** — Play against a built-in AI for every game
- **Games included:**
  - ⭕ **Tic-Tac-Toe** — Classic 3×3 with perfect Minimax AI
  - 🔴 **Connect Four** — 7×6 grid with Alpha-Beta pruning AI

## Quick Start

### Prerequisites
- [Node.js](https://nodejs.org/) v18 or higher
- npm (comes with Node.js)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd HomeGames

# Install dependencies
npm install

# Start the server
npm start
```

Then open your browser to **http://localhost:3000**.

### Network Play (LAN)

To play with others on your local network, find your machine's local IP address:

```bash
# macOS / Linux
ip addr show   # or: ifconfig

# Windows
ipconfig
```

Then share `http://<your-local-ip>:3000` with friends on the same network.

### Raspberry Pi (Low Power Server)

```bash
# Run in background with nohup
nohup npm start &

# Or use PM2 for automatic restarts
npm install -g pm2
pm2 start server.js --name homegames
pm2 startup   # enable autostart on boot
pm2 save
```

## Project Structure

```
HomeGames/
├── server.js          # Express server entry point
├── src/
│   ├── database.js    # SQLite database setup
│   └── elo.js         # ELO rating calculations
├── routes/
│   ├── players.js     # Player CRUD API
│   └── games.js       # Game results & history API
├── public/
│   ├── index.html     # Single-page app
│   ├── css/style.css  # Styles
│   └── js/
│       ├── api.js       # Fetch API helpers
│       ├── ui.js        # UI rendering & player management
│       ├── tictactoe.js # Tic-Tac-Toe game + Minimax AI
│       ├── connect4.js  # Connect Four game + AI
│       └── app.js       # App entry point
├── tests/
│   └── basic.test.js  # API integration tests
└── data/              # SQLite database (auto-created, gitignored)
```

## API Reference

### Players

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/players` | List all players (sorted by ELO) |
| GET | `/api/players/:id` | Get player by ID |
| POST | `/api/players` | Create a new player `{ name, avatar? }` |
| PUT | `/api/players/:id` | Update player `{ name?, avatar? }` |
| DELETE | `/api/players/:id` | Delete a player |

### Games

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/games/result` | Record game result `{ game_type, player_ids, winner_id }` |
| GET | `/api/games/history` | Get game history (query: `?limit=50`) |

## Running Tests

```bash
npm test
```

## Configuration

The server port can be changed with the `PORT` environment variable:

```bash
PORT=8080 npm start
```

