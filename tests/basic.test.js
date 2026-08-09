/* Basic integration tests for HomeGames API */
const assert = require('assert');
const http = require('http');

// Override DB path for tests
process.env.NODE_ENV = 'test';

// Use in-memory DB for testing by patching database module
const Database = require('better-sqlite3');
const path = require('path');

// Patch: use in-memory DB for tests
const originalDb = require('../src/database');

const app = require('../server');

let server;
let base;

function request(method, path, body) {
  return new Promise((resolve, reject) => {
    const data = body ? JSON.stringify(body) : null;
    const opts = {
      method,
      path,
      hostname: 'localhost',
      port: server.address().port,
      headers: { 'Content-Type': 'application/json' },
    };
    const req = http.request(opts, res => {
      let buf = '';
      res.on('data', d => buf += d);
      res.on('end', () => {
        try { resolve({ status: res.statusCode, body: JSON.parse(buf) }); }
        catch (e) { resolve({ status: res.statusCode, body: buf }); }
      });
    });
    req.on('error', reject);
    if (data) req.write(data);
    req.end();
  });
}

async function run() {
  server = app.listen(0);
  base = `http://localhost:${server.address().port}`;
  let passed = 0, failed = 0;

  async function test(name, fn) {
    try {
      await fn();
      console.log(`  ✓ ${name}`);
      passed++;
    } catch (e) {
      console.error(`  ✗ ${name}: ${e.message}`);
      failed++;
    }
  }

  console.log('\n=== HomeGames API Tests ===\n');

  // Players API
  let createdPlayerId;
  await test('POST /api/players - create player', async () => {
    const r = await request('POST', '/api/players', { name: 'TestPlayer', avatar: '😀' });
    assert.strictEqual(r.status, 201);
    assert.strictEqual(r.body.name, 'TestPlayer');
    assert.strictEqual(r.body.elo, 1200);
    createdPlayerId = r.body.id;
  });

  await test('POST /api/players - duplicate name returns 409', async () => {
    const r = await request('POST', '/api/players', { name: 'TestPlayer' });
    assert.strictEqual(r.status, 409);
  });

  await test('POST /api/players - missing name returns 400', async () => {
    const r = await request('POST', '/api/players', { name: '' });
    assert.strictEqual(r.status, 400);
  });

  await test('GET /api/players - returns array', async () => {
    const r = await request('GET', '/api/players');
    assert.strictEqual(r.status, 200);
    assert(Array.isArray(r.body));
    assert(r.body.length >= 1);
  });

  await test('GET /api/players/:id - returns player', async () => {
    const r = await request('GET', `/api/players/${createdPlayerId}`);
    assert.strictEqual(r.status, 200);
    assert.strictEqual(r.body.name, 'TestPlayer');
  });

  await test('PUT /api/players/:id - update name', async () => {
    const r = await request('PUT', `/api/players/${createdPlayerId}`, { name: 'UpdatedPlayer' });
    assert.strictEqual(r.status, 200);
    assert.strictEqual(r.body.name, 'UpdatedPlayer');
  });

  // Games API
  let player2Id;
  await test('POST /api/players - create second player', async () => {
    const r = await request('POST', '/api/players', { name: 'Player2' });
    assert.strictEqual(r.status, 201);
    player2Id = r.body.id;
  });

  await test('POST /api/games/result - record win', async () => {
    const r = await request('POST', '/api/games/result', {
      game_type: 'tictactoe',
      player_ids: [createdPlayerId, player2Id],
      winner_id: createdPlayerId,
    });
    assert.strictEqual(r.status, 200);
    assert(r.body.playerA.change > 0, 'Winner ELO should increase');
    assert(r.body.playerB.change < 0, 'Loser ELO should decrease');
  });

  await test('POST /api/games/result - record draw', async () => {
    const r = await request('POST', '/api/games/result', {
      game_type: 'connect4',
      player_ids: [createdPlayerId, player2Id],
      winner_id: null,
    });
    assert.strictEqual(r.status, 200);
  });

  await test('GET /api/games/history - returns array', async () => {
    const r = await request('GET', '/api/games/history');
    assert.strictEqual(r.status, 200);
    assert(Array.isArray(r.body));
    assert(r.body.length >= 2);
  });

  await test('POST /api/games/result - invalid player returns 404', async () => {
    const r = await request('POST', '/api/games/result', {
      game_type: 'tictactoe',
      player_ids: ['nonexistent-id', player2Id],
      winner_id: null,
    });
    assert.strictEqual(r.status, 404);
  });

  await test('DELETE /api/players/:id - delete player', async () => {
    const r = await request('DELETE', `/api/players/${player2Id}`);
    assert.strictEqual(r.status, 200);
  });

  // ELO module
  await test('ELO: winner gains rating', () => {
    const { calculateElo } = require('../src/elo');
    const result = calculateElo(1200, 1200, 'A');
    assert(result.changeA > 0);
    assert(result.changeB < 0);
    assert.strictEqual(result.newA + result.newB, 2400 + result.changeA + result.changeB);
  });

  await test('ELO: draw near-equal ratings minimal change', () => {
    const { calculateElo } = require('../src/elo');
    const result = calculateElo(1200, 1200, 'draw');
    assert.strictEqual(result.changeA, 0);
    assert.strictEqual(result.changeB, 0);
  });

  console.log(`\n${passed} passed, ${failed} failed\n`);
  server.close();
  process.exit(failed > 0 ? 1 : 0);
}

run().catch(e => { console.error(e); process.exit(1); });
